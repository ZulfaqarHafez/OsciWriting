from agentpathrouter import CostModel, ModelPrice, RunMetrics


def test_step_cost_frontier_vs_small():
    cm = CostModel()
    frontier = cm.step_cost("frontier")
    small = cm.step_cost("small")
    # Haiku 4.5 is ~15x cheaper than Opus 4.7 (1+5 vs 15+75 weighted)
    assert frontier > small > 0
    assert frontier / small > 5  # at least 5x cheaper


def test_cache_step_cost_is_zero():
    assert CostModel().step_cost("cache") == 0.0


def test_cost_breakdown_zero_when_all_cache():
    m = RunMetrics(steps=10, cache_hits=10)
    b = CostModel().cost_breakdown(m)
    assert b["usd_total"] == 0.0
    assert b["pct_saved"] == 1.0


def test_cost_breakdown_baseline_when_all_full():
    m = RunMetrics(steps=10, full_calls=10)
    b = CostModel().cost_breakdown(m)
    assert b["usd_total"] == b["usd_baseline_full_frontier"]
    assert b["pct_saved"] == 0.0


def test_speculation_does_not_reduce_cost():
    """Spec hits save latency but the LLM still ran — cost stays at frontier."""
    spec_only = RunMetrics(steps=10, spec_hits=10)
    full_only = RunMetrics(steps=10, full_calls=10)
    cm = CostModel()
    assert cm.cost_breakdown(spec_only)["usd_total"] == cm.cost_breakdown(full_only)["usd_total"]


def test_small_model_routing_reduces_cost():
    routed = RunMetrics(steps=10, small_model_calls=10)
    full = RunMetrics(steps=10, full_calls=10)
    cm = CostModel()
    assert cm.cost_breakdown(routed)["usd_total"] < cm.cost_breakdown(full)["usd_total"]


def test_per_1000_runs_scales_correctly():
    cm = CostModel()
    metrics = [RunMetrics(steps=5, full_calls=5) for _ in range(100)]
    out = cm.per_1000_runs(metrics)
    assert out["n_runs"] == 100
    # 100 runs × 5 frontier steps = 500 steps. Per 1000 runs that's 5000 steps.
    expected = 5_000 * cm.step_cost("frontier")
    assert abs(out["usd_per_1000_runs"] - expected) < 0.01


def test_custom_prices_override():
    cm = CostModel(prices={
        "frontier": ModelPrice(100.0, 100.0),
        "small": ModelPrice(1.0, 1.0),
    })
    assert cm.step_cost("frontier") == cm.step_cost("small") * 100
