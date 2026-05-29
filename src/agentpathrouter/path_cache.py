"""PathCache — (execution_state_hash → cached tool output).

A step in an agent run is uniquely determined (for caching purposes) by:

    (tool_name, ordered-history-of-prior-tools, tool-input-args)

so the state hash is computed over that triple. Hits return the previously
observed output; misses fall through to the real tool call and populate the
cache.

A cache hit is only *sound* if the tool is deterministic — the same triple
must always yield the same output. P3.1 (audit_cache_determinism.py) found
that stateful / observation tools (e.g. ``toggle_airplane_mode``,
``check_network_status``) violate this: ~56% of cache hits on tau-bench
telecom returned stale outputs. PathCache therefore supports a per-tool
*denylist* of non-idempotent tools that are never cached. The denylist can
be supplied directly or learned from a labelled corpus with
``DeterminismFilter.from_observations``.

Backed by a plain dict for now — swap in Redis / sqlite once the system
moves out of single-process evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable


def state_hash(tool: str, history: tuple[str, ...], args: dict[str, Any]) -> str:
    """Deterministic hash for a (tool, prior-tool-history, args) triple."""
    payload = json.dumps(
        {"tool": tool, "history": list(history), "args": args},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    denied: int = 0  # lookups skipped because the tool is on the denylist

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


def _output_signature(value: Any) -> str:
    """Stable signature of a tool output, for determinism checking."""
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class DeterminismFilter:
    """Decides whether a tool's outputs are safe to cache.

    A tool is *non-deterministic* (unsafe to cache) if the same
    ``(tool, history, args)`` state has been observed producing more than
    one distinct output. ``denylist`` holds tools judged unsafe.
    """

    denylist: set[str] = field(default_factory=set)

    def is_cacheable(self, tool: str) -> bool:
        return tool not in self.denylist

    @classmethod
    def from_observations(
        cls,
        observations: Iterable[tuple[str, tuple[str, ...], dict, Any]],
        max_nondet_share: float = 0.05,
        min_nondet_states: int = 2,
    ) -> "DeterminismFilter":
        """Learn a denylist from labelled ``(tool, history, args, output)`` rows.

        For each tool, group observations by state hash, tracking the
        observation count and the set of distinct output signatures per
        state. A *repeated* state was seen ≥2 times. A *non-deterministic*
        state has >1 distinct output (necessarily repeated). A tool whose
        non-deterministic share of repeated states exceeds
        ``max_nondet_share`` (and has at least ``min_nondet_states`` such
        states) is denylisted.

        This mirrors ``scripts/audit_cache_determinism.py`` exactly, turned
        into a runtime filter.
        """
        # tool -> state_hash -> [observation_count, {output signatures}]
        seen: dict[str, dict[str, list]] = defaultdict(dict)
        for tool, history, args, output in observations:
            h = state_hash(tool, history, args)
            slot = seen[tool].get(h)
            if slot is None:
                seen[tool][h] = [1, {_output_signature(output)}]
            else:
                slot[0] += 1
                slot[1].add(_output_signature(output))

        denylist: set[str] = set()
        for tool, states in seen.items():
            repeated = [s for s in states.values() if s[0] >= 2]
            if not repeated:
                continue
            nondet = sum(1 for s in repeated if len(s[1]) > 1)
            share = nondet / len(repeated)
            if share > max_nondet_share and nondet >= min_nondet_states:
                denylist.add(tool)
        return cls(denylist=denylist)


@dataclass
class PathCache:
    store: dict[str, Any] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)
    determinism: DeterminismFilter = field(default_factory=DeterminismFilter)

    def get(
        self, tool: str, history: tuple[str, ...], args: dict[str, Any]
    ) -> tuple[bool, Any]:
        """Return ``(hit, value)``. ``value`` is None on miss.

        Tools on the determinism denylist are never served from cache —
        the lookup is counted as ``denied`` and returns a miss so the real
        tool runs.
        """
        if not self.determinism.is_cacheable(tool):
            self.stats.denied += 1
            return False, None
        key = state_hash(tool, history, args)
        if key in self.store:
            self.stats.hits += 1
            return True, self.store[key]
        self.stats.misses += 1
        return False, None

    def put(
        self,
        tool: str,
        history: tuple[str, ...],
        args: dict[str, Any],
        value: Any,
    ) -> None:
        # Don't populate the cache for denylisted tools either.
        if not self.determinism.is_cacheable(tool):
            return
        self.store[state_hash(tool, history, args)] = value

    def __len__(self) -> int:
        return len(self.store)
