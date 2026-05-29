"""Shannon entropy + coverage-curve utilities over agent tool-call sequences.

This is the Phase 1 deliverable from the PRD: given a corpus of execution
traces, extract the tool sequence per trace, compute the Shannon entropy of
the *path* distribution (a path = full ordered tuple of tools), and report
how much of the corpus is covered by the top-N most frequent paths.

The starter code in PRD §7 is folded in verbatim where reasonable; the regex
in ``extract_tool_sequence`` is the only thing tuned to a specific log format
and will need adjustment per dataset.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Tool-sequence extraction
# ---------------------------------------------------------------------------

# Common shapes seen in agent traces. Ordered most-specific → most-generic.
_TOOL_PATTERNS = [
    re.compile(r"tool_call:\s*([A-Za-z_][\w\-]*)"),
    re.compile(r'"tool_name"\s*:\s*"([A-Za-z_][\w\-]*)"'),
    re.compile(r'"tool"\s*:\s*"([A-Za-z_][\w\-]*)"'),
    re.compile(r'"function"\s*:\s*\{\s*"name"\s*:\s*"([A-Za-z_][\w\-]*)"'),
    # Hermes-style: <tool_call>{"name": "X", "arguments": ...}</tool_call>
    re.compile(r'<tool_call>\s*\{\s*"name"\s*:\s*"([A-Za-z_][\w\-]*)"'),
    re.compile(r"<tool>\s*([A-Za-z_][\w\-]*)\s*</tool>"),
    re.compile(r"Action:\s*([A-Za-z_][\w\-]*)"),
]


def extract_tool_sequence(log: str) -> list[str]:
    """Return ordered list of tool names invoked in a single trace log.

    Tries several known formats; uses the first pattern that produces matches.
    Returns ``[]`` if no tool calls are detectable.
    """
    for pat in _TOOL_PATTERNS:
        matches = pat.findall(log)
        if matches:
            return matches
    return []


def extract_tool_calls_with_args_from_messages(
    messages: list[dict],
    requestor_filter: str | None = None,
) -> list[tuple[str, dict]]:
    """Like ``extract_tool_sequence_from_messages`` but also returns args.

    Each entry is ``(tool_name, arguments_dict)`` in invocation order. Used
    by the state-hash cache so per-call args (e.g. ``{"order_id": "X"}``)
    properly distinguish cache keys, instead of relying on a coarse
    trace-level arg proxy.

    Args may be a dict (already parsed) or a JSON string (some providers
    serialise the call); both are normalised to a dict.

    ``requestor_filter`` (optional): tau-bench tool_calls carry a
    ``requestor`` field ("assistant" or "user" — the user simulator also
    calls tools through the same schema). Pass ``"assistant"`` to keep
    only agent-initiated calls; default None keeps all.
    """
    import json as _json

    def _parse_args(raw) -> dict:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = _json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"_raw": raw}
            except _json.JSONDecodeError:
                return {"_raw": raw}
        return {}

    from_assistant: list[tuple[str, dict]] = []
    from_tool_role: list[tuple[str, dict]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            if requestor_filter is not None:
                req = tc.get("requestor")
                if req is not None and req != requestor_filter:
                    continue
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name")
            args_raw = fn.get("arguments") if "arguments" in fn else tc.get("arguments")
            if name:
                from_assistant.append((name, _parse_args(args_raw)))
        content = msg.get("content")
        if isinstance(content, str) and "<tool_call>" in content:
            # Hermes-style. Pull the whole JSON block to keep the args.
            for m in re.finditer(
                r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, flags=re.DOTALL
            ):
                try:
                    obj = _json.loads(m.group(1))
                except _json.JSONDecodeError:
                    continue
                name = obj.get("name")
                if name:
                    from_assistant.append((name, _parse_args(obj.get("arguments") or {})))
        if msg.get("role") == "tool":
            name = msg.get("name") or msg.get("tool_name")
            if name:
                from_tool_role.append((name, {}))  # tool-role turns rarely carry args
    return from_assistant or from_tool_role


def extract_tool_sequence_from_messages(messages: list[dict]) -> list[str]:
    """Walk a structured chat-completions ``messages`` list and pull tool names.

    Handles three common shapes:
        - OpenAI-style: ``msg["tool_calls"] = [{"function": {"name": ...}}, ...]``
        - Hermes / Nemotron-style: assistant ``content`` contains
          ``<tool_call>{"name": "X", "arguments": {...}}</tool_call>`` blocks
        - Generic ``msg["name"]`` on role=="tool" turns (in invocation order)

    Preference order: if the corpus emits assistant-side calls (tool_calls
    arrays or Hermes tags), use those. The ``role=="tool"`` response turns
    are used only when there are no assistant-side calls anywhere — they're
    redundant otherwise and would double-count.

    Returns ``[]`` if no tool calls can be found structurally.
    """
    from_assistant: list[str] = []
    from_tool_role: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name") or tc.get("name")
                if name:
                    from_assistant.append(name)
        content = msg.get("content")
        if isinstance(content, str) and "<tool_call>" in content:
            for m in re.finditer(
                r'<tool_call>\s*\{\s*"name"\s*:\s*"([A-Za-z_][\w\-]*)"', content
            ):
                from_assistant.append(m.group(1))
        if msg.get("role") == "tool":
            name = msg.get("name") or msg.get("tool_name")
            if name:
                from_tool_role.append(name)
    return from_assistant or from_tool_role


# ---------------------------------------------------------------------------
# Entropy + coverage
# ---------------------------------------------------------------------------


def path_entropy(sequences: Iterable[Sequence[str]]) -> float:
    """Shannon entropy (bits) of the distribution over full execution paths.

    Each path is the ordered tuple of tool names. A corpus where every trace
    follows the same path has entropy 0; a corpus where every path is unique
    has entropy ``log2(N)``.
    """
    paths = [tuple(s) for s in sequences]
    counts = Counter(paths)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


@dataclass
class CoverageStats:
    unique_paths: int
    total_traces: int
    entropy_bits: float
    top_n: int
    top_n_coverage: float  # fraction of traces explained by top-N paths
    most_common: list[tuple[tuple[str, ...], int]]


def coverage_curve(
    sequences: Iterable[Sequence[str]], top_n: int = 10
) -> CoverageStats:
    """Return entropy + top-N coverage stats for a corpus of tool sequences."""
    paths = [tuple(s) for s in sequences]
    total = len(paths)
    counts = Counter(paths)
    top = counts.most_common(top_n)
    top_cov = sum(v for _, v in top) / total if total else 0.0
    return CoverageStats(
        unique_paths=len(counts),
        total_traces=total,
        entropy_bits=path_entropy(paths),
        top_n=top_n,
        top_n_coverage=top_cov,
        most_common=top,
    )


def coverage_at_k(sequences: Iterable[Sequence[str]], ks: Sequence[int]) -> dict[int, float]:
    """Return {k: fraction-covered-by-top-k-paths} for each k in ``ks``."""
    paths = [tuple(s) for s in sequences]
    total = len(paths)
    counts = Counter(paths)
    ordered = [v for _, v in counts.most_common()]
    out: dict[int, float] = {}
    for k in ks:
        out[k] = (sum(ordered[:k]) / total) if total else 0.0
    return out
