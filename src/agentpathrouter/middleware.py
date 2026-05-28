"""AgentPathRouter middleware — orchestrates cache + estimator + prefetcher.

Sits between the agent orchestrator and the LLM/tool layer. For each step
in an agent run:

    1. Ask the estimator for next-tool distribution and confidence.
    2. If confidence ≥ T, speculatively pre-fire the predicted tool.
    3. When the LLM emits its chosen tool call:
         a. If state in PathCache → return cached output (no LLM round-trip needed).
         b. Elif speculative prefetch matched → consume that result.
         c. Else → execute the tool normally and populate the cache.

For evaluation, the middleware exposes aggregate counters so the PRD §5.3
metrics (token / latency / cost reduction, cache hit rate, speculation
precision) can be derived per-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .entropy_estimator import NgramEntropyEstimator
from .path_cache import PathCache
from .speculative import SpeculativePrefetcher


@dataclass
class RunMetrics:
    steps: int = 0
    cache_hits: int = 0
    spec_hits: int = 0
    full_calls: int = 0  # neither cache nor speculation helped

    def as_dict(self) -> dict[str, float]:
        n = self.steps or 1
        return {
            "steps": self.steps,
            "cache_hit_rate": self.cache_hits / n,
            "speculation_hit_rate": self.spec_hits / n,
            "full_call_rate": self.full_calls / n,
        }


@dataclass
class AgentPathRouter:
    tools: dict[str, Callable[[dict], Any]]
    cache: PathCache = field(default_factory=PathCache)
    estimator: Optional[NgramEntropyEstimator] = None
    prefetcher: Optional[SpeculativePrefetcher] = None
    confidence_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.estimator is None:
            self.estimator = NgramEntropyEstimator()
        if self.prefetcher is None:
            self.prefetcher = SpeculativePrefetcher(
                estimator=self.estimator,
                tools=self.tools,
                threshold=self.confidence_threshold,
            )

    # ------------------------------------------------------------------
    # Simulated run: feed it the agent's *actual* tool sequence and shared
    # args. Returns per-run metrics + the produced outputs. Used by the
    # evaluation harness — a real deployment would call ``step`` from
    # inside the agent loop instead.
    # ------------------------------------------------------------------

    def step(
        self,
        history: tuple[str, ...],
        actual_tool: str,
        args: dict,
        metrics: RunMetrics,
    ) -> Any:
        metrics.steps += 1

        # 1. Try speculative prefetch — launch BEFORE we "know" actual_tool.
        spec = self.prefetcher.maybe_speculate(history, args) if self.prefetcher else None

        # 2. Cache lookup is keyed on the *actual* tool.
        hit, value = self.cache.get(actual_tool, history, args)
        if hit:
            metrics.cache_hits += 1
            # Drain any speculative future to avoid leaks
            if spec is not None:
                predicted, fut = spec
                self.prefetcher.record(predicted, actual_tool)
                fut.cancel()
            return value

        # 3. Did speculation match?
        if spec is not None:
            predicted, fut = spec
            self.prefetcher.record(predicted, actual_tool)
            if predicted == actual_tool:
                value = fut.result()
                metrics.spec_hits += 1
                self.cache.put(actual_tool, history, args, value)
                return value
            else:
                fut.cancel()

        # 4. Fall through: actually call the tool.
        metrics.full_calls += 1
        value = self.tools[actual_tool](args)
        self.cache.put(actual_tool, history, args, value)
        return value

    def run_trace(self, tools: list[str], args: dict) -> tuple[list[Any], RunMetrics]:
        """Execute one full trace step-by-step and return outputs + metrics."""
        metrics = RunMetrics()
        outputs: list[Any] = []
        for i, tool in enumerate(tools):
            history = tuple(tools[:i])
            outputs.append(self.step(history, tool, args, metrics))
        return outputs, metrics
