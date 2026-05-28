"""Speculative prefetcher.

When the entropy estimator's confidence in its top-1 next-tool prediction
exceeds threshold ``T``, fire that tool *in parallel* with the LLM's
reasoning step. If the LLM agrees, the result is already in hand; otherwise
the speculative result is discarded (cost of one wasted tool call).

The PRD requires this to be evaluated as part of the ablation
(cache-only vs cache+speculation vs cache+speculation+small-model-routing),
so the prefetcher exposes counters for ``fires``, ``hits``, and ``misses``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .entropy_estimator import NgramEntropyEstimator


@dataclass
class PrefetchStats:
    fires: int = 0  # speculative tool actually launched
    hits: int = 0   # speculation matched the LLM's next-tool choice
    misses: int = 0  # speculation was wrong; result discarded

    @property
    def precision(self) -> float:
        return self.hits / self.fires if self.fires else 0.0


@dataclass
class SpeculativePrefetcher:
    estimator: NgramEntropyEstimator
    tools: dict[str, Callable[[dict], Any]]
    threshold: float = 0.7  # T from the PRD; tune empirically
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=4))
    stats: PrefetchStats = field(default_factory=PrefetchStats)

    def maybe_speculate(
        self, history: tuple[str, ...], args: dict
    ) -> Optional[tuple[str, "Future[Any]"]]:
        """Launch a speculative tool call if estimator confidence ≥ threshold.

        Returns ``(predicted_tool, future)`` or ``None`` if confidence is low.
        Caller is expected to either ``future.result()`` (if the LLM agrees)
        or drop it (if the LLM picks a different tool).
        """
        tool, p = self.estimator.top1(list(history))
        if p < self.threshold or tool not in self.tools:
            return None
        fn = self.tools[tool]
        self.stats.fires += 1
        future = self._executor.submit(fn, args)
        return tool, future

    def record(self, predicted: str, actual: str) -> None:
        if predicted == actual:
            self.stats.hits += 1
        else:
            self.stats.misses += 1

    def close(self) -> None:
        self._executor.shutdown(wait=False)
