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
    re.compile(r'"tool"\s*:\s*"([A-Za-z_][\w\-]*)"'),
    re.compile(r'"function"\s*:\s*\{\s*"name"\s*:\s*"([A-Za-z_][\w\-]*)"'),
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
