"""Tests for the per-tool determinism filter (P3.1 follow-up)."""

from agentpathrouter import DeterminismFilter, PathCache


def test_denylisted_tool_never_hits():
    cache = PathCache(determinism=DeterminismFilter(denylist={"check_status"}))
    cache.put("check_status", (), {"x": 1}, "OK")
    hit, val = cache.get("check_status", (), {"x": 1})
    assert hit is False  # denylisted → forced miss
    assert val is None
    assert cache.stats.denied == 1
    assert len(cache) == 0  # put was also suppressed


def test_non_denylisted_tool_caches_normally():
    cache = PathCache(determinism=DeterminismFilter(denylist={"check_status"}))
    cache.put("get_order", (), {"id": "W1"}, {"total": 5})
    hit, val = cache.get("get_order", (), {"id": "W1"})
    assert hit is True
    assert val == {"total": 5}


def test_default_filter_allows_everything():
    cache = PathCache()
    cache.put("anything", (), {}, 42)
    hit, val = cache.get("anything", (), {})
    assert hit is True and val == 42


def test_from_observations_flags_nondeterministic_tool():
    # check_status: several distinct states, each observed twice with
    # DIFFERENT outputs → many non-deterministic states (like real telecom).
    obs = []
    for device in ["A", "B", "C", "D"]:
        obs.append(("check_status", (), {"device": device}, "on"))
        obs.append(("check_status", (), {"device": device}, "off"))  # nondet
    # get_order: deterministic — same args → same output, observed repeatedly.
    for order in ["W1", "W2", "W3"]:
        obs.append(("get_order", (), {"id": order}, {"total": 5}))
        obs.append(("get_order", (), {"id": order}, {"total": 5}))
    filt = DeterminismFilter.from_observations(obs)
    assert "check_status" in filt.denylist
    assert "get_order" not in filt.denylist


def test_from_observations_ignores_single_observation_states():
    # A tool seen once per distinct state can't be judged non-deterministic.
    obs = [(f"tool_{i}", (), {"k": i}, f"out_{i}") for i in range(20)]
    filt = DeterminismFilter.from_observations(obs)
    assert filt.denylist == set()


def test_from_observations_respects_min_nondet_states():
    # Only ONE non-deterministic state → below default min_nondet_states=2,
    # so the tool is NOT denylisted even though share would be high.
    obs = [
        ("t", (), {"a": 1}, "x"),
        ("t", (), {"a": 1}, "y"),  # 1 nondet state
        ("t", (), {"a": 2}, "z"),
        ("t", (), {"a": 2}, "z"),  # deterministic repeated state
    ]
    filt = DeterminismFilter.from_observations(obs)
    assert "t" not in filt.denylist


def test_denied_lookups_counted_separately_from_misses():
    cache = PathCache(determinism=DeterminismFilter(denylist={"d"}))
    cache.get("d", (), {})       # denied
    cache.get("ok", (), {})      # miss (not in store, not denied)
    assert cache.stats.denied == 1
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0
