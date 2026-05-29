"""AgentPathRouter middleware — orchestrates cache + estimator + prefetcher + routing.

Sits between the agent orchestrator and the LLM/tool layer. For each step:

    1. PathCache lookup on the (tool, history, args) state hash.
       Hit ⇒ no LLM call, return cached output.
    2. If the entropy estimator's confidence on its top-1 next-tool
       prediction is ≥ ``small_model_threshold`` AND small-model routing
       is enabled, route the decision to the small model. The small model
       commits to the predicted tool; quality regressions are tracked
       (predicted ≠ actual on a routed step).
    3. Otherwise the frontier model runs. While it's reasoning, the
       speculative prefetcher may pre-fire the predicted tool if
       confidence ≥ ``confidence_threshold``. If the LLM agrees, we save
       latency (but not LLM tokens).
    4. Execute the tool. Populate the cache.

For evaluation, the middleware exposes aggregate counters so the PRD §5.3
metrics (token / latency / cost reduction, cache hit rate, speculation
precision, quality preservation) can be derived per-run.
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
    spec_misses: int = 0         # speculation fired but the LLM picked a different tool
    full_calls: int = 0          # frontier model handled the step
    small_model_calls: int = 0   # small-model routing handled the step
    small_model_errors: int = 0  # routed but predicted ≠ actual (quality regression)

    def as_dict(self) -> dict[str, float]:
        n = self.steps or 1
        return {
            "steps": self.steps,
            "cache_hit_rate": self.cache_hits / n,
            "speculation_hit_rate": self.spec_hits / n,
            "full_call_rate": self.full_calls / n,
            "small_model_route_rate": self.small_model_calls / n,
            "small_model_error_rate": (
                self.small_model_errors / self.small_model_calls
                if self.small_model_calls else 0.0
            ),
            # Step-level quality regression: routed-and-wrong / total steps.
            # PRD §5.3 caps this at 2%.
            "quality_regression_rate": self.small_model_errors / n,
        }

    def __iadd__(self, other: "RunMetrics") -> "RunMetrics":
        self.steps += other.steps
        self.cache_hits += other.cache_hits
        self.spec_hits += other.spec_hits
        self.spec_misses += other.spec_misses
        self.full_calls += other.full_calls
        self.small_model_calls += other.small_model_calls
        self.small_model_errors += other.small_model_errors
        return self


@dataclass
class AgentPathRouter:
    tools: dict[str, Callable[[dict], Any]]
    cache: PathCache = field(default_factory=PathCache)
    estimator: Optional[NgramEntropyEstimator] = None
    prefetcher: Optional[SpeculativePrefetcher] = None
    confidence_threshold: float = 0.7
    use_small_model_routing: bool = False
    # Routing decisions need a higher confidence bar than speculation,
    # because a wrong route is a quality regression while wrong speculation
    # only wastes one tool call.
    small_model_threshold: float = 0.85
    use_speculation: bool = True

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

    def step(
        self,
        history: tuple[str, ...],
        actual_tool: str,
        args: dict,
        metrics: RunMetrics,
    ) -> Any:
        metrics.steps += 1

        # 1. Cache lookup (free, no LLM needed at all).
        hit, value = self.cache.get(actual_tool, history, args)
        if hit:
            metrics.cache_hits += 1
            return value

        # 2. Small-model routing arm (PRD §5.2 third component).
        if self.use_small_model_routing:
            predicted, conf = self.estimator.top1(list(history))
            if conf >= self.small_model_threshold and predicted in self.tools:
                metrics.small_model_calls += 1
                if predicted != actual_tool:
                    # Routed wrong — the small model would have called the
                    # wrong tool. Count as a quality regression but still
                    # produce the correct output downstream so the trace
                    # continues (mirrors a "human/judge corrects it" flow).
                    metrics.small_model_errors += 1
                value = self.tools[actual_tool](args)
                self.cache.put(actual_tool, history, args, value)
                return value

        # 3. Frontier path, with optional speculation for latency.
        spec = None
        if self.use_speculation and self.prefetcher:
            spec = self.prefetcher.maybe_speculate(history, args)

        if spec is not None:
            predicted, fut = spec
            self.prefetcher.record(predicted, actual_tool)
            if predicted == actual_tool:
                value = fut.result()
                metrics.spec_hits += 1
                self.cache.put(actual_tool, history, args, value)
                return value
            # Speculation fired but the LLM chose a different tool — the
            # tool we pre-fired ran to completion (or was cancelled mid-
            # flight) and its work is discarded. That's a wasted tool
            # execution: counted against speculation in the cost model
            # when tool_execution_usd > 0.
            metrics.spec_misses += 1
            fut.cancel()

        metrics.full_calls += 1
        value = self.tools[actual_tool](args)
        self.cache.put(actual_tool, history, args, value)
        return value

    def run_trace(
        self,
        tools: list[str],
        args: dict,
        per_step_args: list[dict] | None = None,
    ) -> tuple[list[Any], RunMetrics]:
        """Execute one full trace step-by-step and return outputs + metrics.

        ``per_step_args`` (optional) supplies a separate args dict per tool
        call. When provided, the cache key for step ``i`` uses
        ``per_step_args[i]`` instead of the shared ``args``. This matters
        for real corpora where individual tool calls carry their own
        arguments (e.g. ``find_user_id_by_email(email="...")``); using the
        per-call args is what makes cache hits a measure of real
        cacheability rather than corpus-level replay structure.
        """
        metrics = RunMetrics()
        outputs: list[Any] = []
        for i, tool in enumerate(tools):
            history = tuple(tools[:i])
            step_args = per_step_args[i] if per_step_args is not None else args
            outputs.append(self.step(history, tool, step_args, metrics))
        return outputs, metrics
