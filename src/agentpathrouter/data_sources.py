"""Loaders for AEE trace corpora.

Each loader returns ``list[dict]`` where every dict has at least:

    {
        "id":        str,            # unique trace id
        "tools":     list[str],      # ordered tool-call sequence (already extracted)
        "args":      dict,           # shared args (used for PathCache state hash)
        "raw":       Any,            # raw row for downstream debugging
    }

Loaders that depend on HuggingFace ``datasets`` raise ``DatasetUnavailable``
when the package is missing or the network call fails — the driver catches
that and falls back to the synthetic corpus.

Sources wired here (PRD §6 and follow-ups):
    - yunjue                  YunjueTech/Yunjue-Agent-Traces      (PRD §6.1)
    - nemotron_agentic        nvidia/Nemotron-Agentic-v1          (recommended add)
    - hermes_reasoning        lambda/hermes-agent-reasoning-traces (recommended add)
    - hermes_filtered         DJLougen/hermes-agent-traces-filtered (recommended add)
    - tau_bench               local directory of tau2-bench simulation JSON files
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from .entropy import extract_tool_sequence, extract_tool_sequence_from_messages


class DatasetUnavailable(RuntimeError):
    """Raised when a remote dataset can't be loaded (network, auth, missing dep)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_load(repo: str, subset: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise DatasetUnavailable(f"datasets package not installed: {e}") from e
    try:
        if subset:
            return load_dataset(repo, subset, split=split)
        return load_dataset(repo, split=split)
    except Exception as e:  # noqa: BLE001 — datasets raises a broad set
        raise DatasetUnavailable(f"could not load {repo}:{subset or ''}:{split}: {e}") from e


def _walk_for_messages(row: Any) -> list[dict] | None:
    """Find the first list-of-dicts that looks like a chat ``messages`` list."""
    candidates = ("messages", "conversations", "conversation", "chat", "turns", "trace")
    if isinstance(row, dict):
        for k in candidates:
            v = row.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        # one level deeper, e.g. row["data"]["messages"]
        for v in row.values():
            if isinstance(v, dict):
                inner = _walk_for_messages(v)
                if inner:
                    return inner
    return None


def _extract_tools(row: dict) -> list[str]:
    """Try structured-message extraction first, then regex on a serialised dump."""
    msgs = _walk_for_messages(row)
    if msgs:
        seq = extract_tool_sequence_from_messages(msgs)
        if seq:
            return seq
    # Fall back: serialise and regex
    try:
        blob = json.dumps(row, default=str)
    except (TypeError, ValueError):
        blob = str(row)
    return extract_tool_sequence(blob)


def _normalise(rows: Iterable[dict], id_field: str = "id") -> list[dict]:
    """Apply ``_extract_tools`` to every row, dropping ones with no tool calls."""
    out: list[dict] = []
    for i, r in enumerate(rows):
        tools = _extract_tools(r)
        if not tools:
            continue
        rid = r.get(id_field) or r.get("trace_id") or f"row-{i:06d}"
        # ``args`` for cache keying: use whatever input fields look stable.
        args = {
            "q": (r.get("question") or r.get("query") or r.get("task") or r.get("prompt") or "")[:64],
        }
        out.append({"id": str(rid), "tools": tools, "args": args, "raw": r})
    return out


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------


def load_yunjue(subset: str = "finsearchcomp", split: str = "train") -> list[dict]:
    """Yunjue Agent Traces — PRD §6.1 primary source.

    Schema: ``id`` / ``dataset`` / ``question`` (b64) / ``log`` (b64).
    """
    ds = _hf_load("YunjueTech/Yunjue-Agent-Traces", subset, split)
    rows = []
    for r in ds:
        try:
            q = base64.b64decode(r["question"]).decode("utf-8", errors="replace")
            log = base64.b64decode(r["log"]).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        rows.append({"id": r.get("id"), "question": q, "log": log})
    # Yunjue logs are flat strings → regex path
    out = []
    for i, r in enumerate(rows):
        tools = extract_tool_sequence(r["log"])
        if not tools:
            continue
        out.append({
            "id": str(r.get("id") or f"yunjue-{i:06d}"),
            "tools": tools,
            "args": {"q": r["question"][:64]},
            "raw": r,
        })
    return out


def load_nemotron_agentic(split: str = "train") -> list[dict]:
    """nvidia/Nemotron-Agentic-v1 — synthetic multi-turn tool-use trajectories."""
    ds = _hf_load("nvidia/Nemotron-Agentic-v1", subset=None, split=split)
    return _normalise(ds)


def load_hermes_reasoning(split: str = "train") -> list[dict]:
    """lambda/hermes-agent-reasoning-traces — multi-turn tool-call + reasoning."""
    ds = _hf_load("lambda/hermes-agent-reasoning-traces", subset=None, split=split)
    return _normalise(ds)


def load_hermes_filtered(split: str = "train") -> list[dict]:
    """DJLougen/hermes-agent-traces-filtered — quality-pruned Hermes traces."""
    ds = _hf_load("DJLougen/hermes-agent-traces-filtered", subset=None, split=split)
    return _normalise(ds)


# ---------------------------------------------------------------------------
# τ-bench: local directory of JSON simulation files
# ---------------------------------------------------------------------------
#
# τ-bench (sierra-research/tau2-bench) writes simulation traces to
# ``data/simulations/`` as JSON. The exact schema varies by release, so we
# scan any JSON files in the given directory and reuse the structured-message
# / regex pipeline. To produce traces:
#
#     git clone https://github.com/sierra-research/tau2-bench
#     cd tau2-bench && uv sync && uv run python -m tau2 run-and-eval \
#         --domain retail --agent-llm gpt-4o-mini --num-trials 200
#     # then point our loader at tau2-bench/data/simulations/
#


def load_tau_bench(dir_path: str | Path) -> list[dict]:
    """Load τ-bench simulation traces from a local directory of JSON files.

    Handles three layouts the tau2-bench repo uses interchangeably:
        * a flat list of simulation dicts
        * a single simulation dict
        * a results envelope ``{timestamp, info, tasks, simulations: [...]}``
          (produced by `tau2 run-and-eval`; this is the format shipped in
          ``tau2-bench/data/tau2/results/final/``).
    """
    p = Path(dir_path)
    if not p.exists():
        raise DatasetUnavailable(f"tau-bench dir not found: {p}")
    rows: list[dict] = []
    for jf in sorted(p.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Unwrap results envelope first.
        if isinstance(data, dict) and isinstance(data.get("simulations"), list):
            sims = data["simulations"]
        else:
            sims = data if isinstance(data, list) else [data]
        for i, sim in enumerate(sims):
            if isinstance(sim, dict):
                sim.setdefault("id", f"{jf.stem}#{i}")
                rows.append(sim)
    if not rows:
        raise DatasetUnavailable(f"no JSON traces found under {p}")
    return _normalise(rows)


# ---------------------------------------------------------------------------
# TRAIL benchmark: span-tree (OpenTelemetry-style) traces
# ---------------------------------------------------------------------------
#
# TRAIL (github.com/patronus-ai/trail-benchmark) ships 117 GAIA + 31
# SWE-Bench traces as nested OTel span trees. Each span has a ``span_name``
# (function/tool name) and optionally ``child_spans``. For AEE entropy
# analysis we extract the in-tree-order sequence of "agent-decision" spans:
#
#   * GAIA convention: tool calls are spans whose name ends with "Tool".
#   * SWE-Bench convention: agents emit code instead of named tool spans,
#     so we additionally keep "Step N" iteration markers and the terminal
#     "FinalAnswerTool" to recover a meaningful step sequence.
#
# Layout:
#   trail-benchmark/benchmarking/data/GAIA/<hash>.json
#   trail-benchmark/benchmarking/data/SWE Bench/<hash>.json
#

import re as _re

_TRAIL_TOOL_PAT = _re.compile(r"(.*Tool|Step\s+\d+)$")


def _trail_walk_tools(span: dict) -> list[str]:
    """Pre-order DFS of a span tree, keeping spans whose names look like
    agent decisions (tool calls or step markers)."""
    out: list[str] = []
    name = span.get("span_name")
    if isinstance(name, str) and _TRAIL_TOOL_PAT.match(name):
        out.append(name)
    for child in span.get("child_spans") or []:
        out.extend(_trail_walk_tools(child))
    return out


def load_trail(dir_path: str | Path, subset: str = "all") -> list[dict]:
    """Load TRAIL traces.

    ``subset``:
        * ``"gaia"``       — 117 GAIA traces (multi-step web/research agents)
        * ``"swe_bench"``  — 31 SWE-Bench traces (coding agents)
        * ``"all"``        — both (default)

    ``dir_path`` should point at the repo root (containing
    ``benchmarking/data/``) or at ``benchmarking/data/`` directly.
    """
    root = Path(dir_path)
    if not root.exists():
        raise DatasetUnavailable(f"trail dir not found: {root}")

    # Be flexible about where the user pointed us.
    candidates = [root, root / "benchmarking" / "data", root / "data"]
    data_dir = next((c for c in candidates if (c / "GAIA").exists()), None)
    if data_dir is None:
        raise DatasetUnavailable(
            f"could not find TRAIL data dir under {root} "
            "(expected benchmarking/data/GAIA)"
        )

    subdirs = []
    if subset in ("gaia", "all"):
        subdirs.append(data_dir / "GAIA")
    if subset in ("swe_bench", "all"):
        subdirs.append(data_dir / "SWE Bench")
    if not subdirs:
        raise ValueError(f"unknown TRAIL subset: {subset!r}")

    out: list[dict] = []
    for sd in subdirs:
        if not sd.exists():
            continue
        for jf in sorted(sd.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            spans = data.get("spans") or []
            tools: list[str] = []
            for s in spans:
                tools.extend(_trail_walk_tools(s))
            if not tools:
                continue
            out.append({
                "id": data.get("trace_id") or jf.stem,
                "tools": tools,
                "args": {"subset": sd.name, "task": jf.stem[:32]},
                "raw": {"file": str(jf)},
            })
    if not out:
        raise DatasetUnavailable(f"no usable TRAIL traces under {data_dir}")
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


SOURCES = {
    "yunjue": load_yunjue,
    "nemotron_agentic": load_nemotron_agentic,
    "hermes_reasoning": load_hermes_reasoning,
    "hermes_filtered": load_hermes_filtered,
    "tau_bench": load_tau_bench,
    "trail": load_trail,
}


def load(source: str, **kwargs) -> list[dict]:
    """Load a named source. Raises ``DatasetUnavailable`` on failure."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(SOURCES)}")
    fn = SOURCES[source]
    try:
        return fn(**kwargs)
    except DatasetUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 — wrap unexpected loader errors
        print(f"[{source}] unexpected loader error: {e}", file=sys.stderr)
        raise DatasetUnavailable(str(e)) from e
