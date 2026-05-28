"""Cost model for AEE evaluation.

Translates per-step counters (cache hits, speculation hits, small-model
routes, full frontier calls) into USD per run and per 1000 runs — the PRD
§5.3 headline cost metric.

Defaults reflect Anthropic public pricing as of May 2026:

    Frontier:  claude-opus-4-7    — $15  in / $75  out  per MTok
    Small:     claude-haiku-4-5   — $1   in / $5   out  per MTok

(See `src/redundancy/cost_model.py` for the existing, more elaborate
pricing module used by the redundancy study. This one is intentionally
scoped to what AEE needs.)

Tokens-per-step is a rough average for one tool-calling decision: the LLM
sees the running context + tool definitions and emits a function-call
response. 800 tokens (70% input / 30% output) is a reasonable midpoint and
is the same order of magnitude across most public benchmarks; the model
exposes ``tokens_per_step`` so it can be tuned per dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float


# Anthropic pricing, May 2026.
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "frontier": ModelPrice(input_per_mtok=15.0, output_per_mtok=75.0),  # Opus 4.7
    "small":    ModelPrice(input_per_mtok=1.0,  output_per_mtok=5.0),    # Haiku 4.5
}


@dataclass
class CostModel:
    tokens_per_step: int = 800
    input_frac: float = 0.7
    prices: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICES))

    def step_cost(self, model: str) -> float:
        """USD cost of one LLM-driven tool-selection step on the given model."""
        if model == "cache":
            return 0.0
        p = self.prices[model]
        inp = self.tokens_per_step * self.input_frac
        out = self.tokens_per_step * (1 - self.input_frac)
        return (inp * p.input_per_mtok + out * p.output_per_mtok) / 1_000_000

    # Headline numbers ------------------------------------------------------

    def cost_breakdown(self, metrics) -> dict[str, float]:
        """Compute USD costs from a ``RunMetrics``-shaped object.

        Speculation hits DO NOT reduce token cost (the LLM still ran to
        decide the next tool; the speculative tool just happened in
        parallel). They only cut latency.
        """
        frontier_step = self.step_cost("frontier")
        small_step = self.step_cost("small")

        # Steps where the frontier ran end-to-end:
        #   full_calls + spec_hits (spec doesn't reduce LLM cost)
        frontier_calls = getattr(metrics, "full_calls", 0) + getattr(metrics, "spec_hits", 0)
        small_calls = getattr(metrics, "small_model_calls", 0)
        cache_hits = getattr(metrics, "cache_hits", 0)

        total = (
            frontier_calls * frontier_step
            + small_calls * small_step
            # cache hits cost nothing on the LLM side
        )
        baseline = (frontier_calls + small_calls + cache_hits) * frontier_step
        return {
            "usd_total": round(total, 6),
            "usd_baseline_full_frontier": round(baseline, 6),
            "usd_saved": round(baseline - total, 6),
            "pct_saved": round(1 - (total / baseline), 4) if baseline else 0.0,
            "frontier_step_usd": round(frontier_step, 6),
            "small_step_usd": round(small_step, 6),
        }

    def per_1000_runs(self, metrics_list) -> dict[str, float]:
        """Aggregate ``cost_breakdown`` across many traces, normalised to 1k runs."""
        n = len(metrics_list) or 1
        totals = {"usd_total": 0.0, "usd_baseline_full_frontier": 0.0,
                  "usd_saved": 0.0}
        for m in metrics_list:
            b = self.cost_breakdown(m)
            for k in totals:
                totals[k] += b[k]
        scale = 1000.0 / n
        return {
            "n_runs": n,
            "usd_per_1000_runs": round(totals["usd_total"] * scale, 4),
            "usd_per_1000_runs_baseline": round(totals["usd_baseline_full_frontier"] * scale, 4),
            "usd_per_1000_runs_saved": round(totals["usd_saved"] * scale, 4),
            "pct_saved": (
                round(1 - (totals["usd_total"] / totals["usd_baseline_full_frontier"]), 4)
                if totals["usd_baseline_full_frontier"] else 0.0
            ),
        }
