from agentpathrouter import AgentPathRouter, NgramEntropyEstimator
from agentpathrouter.path_cache import PathCache, state_hash
from agentpathrouter.synthetic import generate_corpus, make_tool_registry


def test_path_cache_hit_miss_roundtrip():
    cache = PathCache()
    hit, _ = cache.get("a", (), {"x": 1})
    assert not hit
    cache.put("a", (), {"x": 1}, 42)
    hit, v = cache.get("a", (), {"x": 1})
    assert hit and v == 42
    assert cache.stats.hit_rate == 0.5  # one miss, one hit


def test_state_hash_distinguishes_history_and_args():
    h1 = state_hash("t", ("a", "b"), {"x": 1})
    h2 = state_hash("t", ("b", "a"), {"x": 1})  # different order
    h3 = state_hash("t", ("a", "b"), {"x": 2})  # different args
    assert h1 != h2 != h3 and h1 != h3


def test_ngram_estimator_high_confidence_on_repeated_path():
    seqs = [["fetch", "compute", "render"]] * 20
    est = NgramEntropyEstimator(n=3).fit(seqs)
    nxt, p = est.top1(["fetch"])
    assert nxt == "compute"
    assert p > 0.8


def test_router_high_cache_hit_on_synthetic_corpus():
    corpus = generate_corpus(n=100, seed=1)
    seqs = [t.tools for t in corpus]
    est = NgramEntropyEstimator(n=3).fit(seqs[:60])
    router = AgentPathRouter(
        tools=make_tool_registry(), estimator=est, confidence_threshold=0.7
    )
    total_steps = cache_hits = 0
    # Use the SAME args across runs so the cache is actually warm
    shared_args = {"date": "2026-05-28", "portfolio": "A"}
    for t in corpus[60:]:
        _, m = router.run_trace(t.tools, shared_args)
        total_steps += m.steps
        cache_hits += m.cache_hits
    assert total_steps > 0
    # With a fixed-shape workflow and identical args, repeated paths must hit
    # the cache the second time onward.
    assert cache_hits / total_steps > 0.5
    router.prefetcher.close()
