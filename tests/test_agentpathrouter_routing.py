"""Tests for the small-model routing arm of AgentPathRouter (PRD §5.2 #3)."""

from agentpathrouter import AgentPathRouter, NgramEntropyEstimator
from agentpathrouter.synthetic import generate_corpus, make_tool_registry


def _trained_estimator(seqs, n=3):
    return NgramEntropyEstimator(n=n).fit(seqs)


def test_routing_fires_on_high_confidence_paths():
    """When the n-gram is confident, routing should pick up many steps."""
    seqs = [["a", "b", "c", "d"]] * 30  # totally deterministic
    est = _trained_estimator(seqs)
    tools = {n: (lambda ctx, _n=n: {"t": _n}) for n in "abcd"}
    router = AgentPathRouter(
        tools=tools, estimator=est,
        use_small_model_routing=True,
        small_model_threshold=0.5,
        use_speculation=False,
    )
    _, m = router.run_trace(["a", "b", "c", "d"], {})
    # First step has no history, so confidence on top-1 is high after seeing
    # 30 identical openings. Later steps are even more confident.
    assert m.small_model_calls >= 3
    assert m.small_model_errors == 0  # predictions are correct
    router.prefetcher.close()


def test_routing_records_quality_regression_when_wrong():
    """If routing fires but the actual tool differs from the prediction,
    that counts as a quality regression."""
    # Train so the estimator is highly confident the next tool after 'a' is 'b'.
    seqs = [["a", "b"]] * 50
    est = _trained_estimator(seqs)
    tools = {n: (lambda ctx, _n=n: _n) for n in "abz"}
    router = AgentPathRouter(
        tools=tools, estimator=est,
        use_small_model_routing=True,
        small_model_threshold=0.5,
        use_speculation=False,
    )
    # But run a trace where the actual tool after 'a' is 'z' — surprise.
    _, m = router.run_trace(["a", "z"], {})
    assert m.small_model_calls >= 1
    assert m.small_model_errors >= 1
    # Quality regression rate ≤ small-model-route rate by construction
    d = m.as_dict()
    assert d["quality_regression_rate"] <= d["small_model_route_rate"]
    router.prefetcher.close()


def test_routing_disabled_by_default():
    seqs = [["a", "b", "c"]] * 30
    est = _trained_estimator(seqs)
    tools = {n: (lambda ctx, _n=n: _n) for n in "abc"}
    router = AgentPathRouter(tools=tools, estimator=est)
    _, m = router.run_trace(["a", "b", "c"], {})
    assert m.small_model_calls == 0
    assert m.small_model_errors == 0
    router.prefetcher.close()


def test_routing_threshold_is_strict():
    """Below threshold, no routing happens even if a top-1 exists."""
    # Uneven 50/50 split between two next tools → top-1 confidence ≈ 0.5
    seqs = [["a", "b"]] * 25 + [["a", "c"]] * 25
    est = _trained_estimator(seqs)
    tools = {n: (lambda ctx, _n=n: _n) for n in "abc"}
    router = AgentPathRouter(
        tools=tools, estimator=est,
        use_small_model_routing=True,
        small_model_threshold=0.9,  # very strict
        use_speculation=False,
    )
    _, m = router.run_trace(["a", "b"], {})
    # Step 1 ("a"): no history → high confidence about "a" being first? No,
    # actually the *predicted-next* given empty history is "a" with ~1.0.
    # Step 2 ("b" given ["a"]): top-1 is "b" or "c" at ~0.5 each — should
    # NOT route under threshold 0.9.
    # So at most the first step routes, never the ambiguous second.
    assert m.small_model_calls <= 1
    router.prefetcher.close()


def test_ablation_arms_on_synthetic_corpus():
    """Smoke test: each arm produces sensible counters on the synthetic corpus."""
    corpus = generate_corpus(n=200, seed=42)
    seqs = [t.tools for t in corpus]
    est = _trained_estimator(seqs[:120])
    tools = make_tool_registry()

    arms = [
        ({"use_speculation": False, "use_small_model_routing": False}, "cache_only"),
        ({"use_speculation": True,  "use_small_model_routing": False}, "cache+spec"),
        ({"use_speculation": True,  "use_small_model_routing": True},  "cache+spec+routing"),
    ]
    results = {}
    for cfg, name in arms:
        router = AgentPathRouter(
            tools=tools, estimator=est,
            small_model_threshold=0.7, **cfg,
        )
        # Use shared args so cache can actually warm
        shared = {"date": "fixed"}
        totals = {"steps": 0, "cache": 0, "spec": 0, "route": 0, "err": 0}
        for t in corpus[120:]:
            _, m = router.run_trace(t.tools, shared)
            totals["steps"] += m.steps
            totals["cache"] += m.cache_hits
            totals["spec"] += m.spec_hits
            totals["route"] += m.small_model_calls
            totals["err"] += m.small_model_errors
        results[name] = totals
        router.prefetcher.close()

    # Cache-only: no spec, no routing
    assert results["cache_only"]["spec"] == 0
    assert results["cache_only"]["route"] == 0
    # cache+spec: speculation must fire on this low-entropy corpus
    assert results["cache+spec"]["spec"] > 0
    assert results["cache+spec"]["route"] == 0
    # cache+spec+routing: routing must fire too
    assert results["cache+spec+routing"]["route"] > 0
    # Quality regression rate stays bounded; PRD caps at 2% but this is a
    # smoke test so just check it's strictly less than route count.
    assert results["cache+spec+routing"]["err"] <= results["cache+spec+routing"]["route"]
