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
    """Load τ-bench simulation traces from a local directory of JSON files."""
    p = Path(dir_path)
    if not p.exists():
        raise DatasetUnavailable(f"tau-bench dir not found: {p}")
    rows: list[dict] = []
    for jf in sorted(p.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # A file may hold a single sim or a list of sims
        sims = data if isinstance(data, list) else [data]
        for i, sim in enumerate(sims):
            if isinstance(sim, dict):
                sim.setdefault("id", f"{jf.stem}#{i}")
                rows.append(sim)
    if not rows:
        raise DatasetUnavailable(f"no JSON traces found under {p}")
    return _normalise(rows)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


SOURCES = {
    "yunjue": load_yunjue,
    "nemotron_agentic": load_nemotron_agentic,
    "hermes_reasoning": load_hermes_reasoning,
    "hermes_filtered": load_hermes_filtered,
    "tau_bench": load_tau_bench,
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
