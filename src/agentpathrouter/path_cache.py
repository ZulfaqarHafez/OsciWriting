"""PathCache — (execution_state_hash → cached tool output).

A step in an agent run is uniquely determined (for caching purposes) by:

    (tool_name, ordered-history-of-prior-tools, tool-input-args)

so the state hash is computed over that triple. Hits return the previously
observed output; misses fall through to the real tool call and populate the
cache.

Backed by a plain dict for now — swap in Redis / sqlite once the system
moves out of single-process evaluation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


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

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return self.hits / n if n else 0.0


@dataclass
class PathCache:
    store: dict[str, Any] = field(default_factory=dict)
    stats: CacheStats = field(default_factory=CacheStats)

    def get(
        self, tool: str, history: tuple[str, ...], args: dict[str, Any]
    ) -> tuple[bool, Any]:
        """Return ``(hit, value)``. ``value`` is None on miss."""
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
        self.store[state_hash(tool, history, args)] = value

    def __len__(self) -> int:
        return len(self.store)
