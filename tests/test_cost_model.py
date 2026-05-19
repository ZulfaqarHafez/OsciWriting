"""PRD §4a arithmetic + threshold solver."""

from redundancy.cost_model import (
    blended_cost,
    derive_thresholds,
    min_p_cache,
    savings,
    sensitivity_table,
)
from redundancy.config import CostConstants


def test_no_cache_is_pure_miss_path():
    # p_cache=0 -> every request pays lookup + miss path
    c = blended_cost(0.0, 0.5, 0.002, 0.05)
    assert abs(c - (0.002 + (0.5 * 0.05 + 0.5 * 1.0))) < 1e-12


def test_full_cache_is_just_lookup():
    assert abs(blended_cost(1.0, 0.5, 0.002, 0.05) - 0.002) < 1e-12


def test_savings_complement():
    assert abs(savings(0.3, 0.5, 0.002, 0.05) - (1 - blended_cost(0.3, 0.5, 0.002, 0.05))) < 1e-12


def test_min_p_cache_roundtrip():
    # plugging the solved min p_cache back yields exactly s_target
    r = min_p_cache(0.50, 0.50, 0.002, 0.05)
    assert r.feasible
    assert abs(savings(r.min_p_cache, 0.50, 0.002, 0.05) - 0.50) < 1e-9


def test_min_p_cache_infeasible_when_lookup_too_dear():
    r = min_p_cache(0.999, 0.5, 0.05, 0.05)  # c_cache 0.05 > 1 - 0.999
    assert not r.feasible


def test_min_p_cache_boundary_is_one_minus_c_cache():
    # max achievable savings is exactly 1 - c_cache (all cached, still pays lookup)
    c_cache = 0.002
    feasible = min_p_cache(1.0 - c_cache - 1e-6, 0.5, c_cache, 0.05)
    infeasible = min_p_cache(1.0 - c_cache + 1e-6, 0.5, c_cache, 0.05)
    assert feasible.feasible
    assert not infeasible.feasible


def test_derive_thresholds_ties_t1_t3():
    d = derive_thresholds(CostConstants())
    assert d.T1 == d.T3
    assert d.provisional is True


def test_sensitivity_table_shape():
    rows = sensitivity_table(CostConstants())
    assert len(rows) == 4 * 3
    assert {"c_small", "p_small", "min_p_cache", "feasible"} <= set(rows[0])
