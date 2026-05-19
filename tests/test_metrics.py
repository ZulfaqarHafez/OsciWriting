"""Pure metric math (PRD §8.5). No sklearn/scipy needed."""

import numpy as np

from redundancy.metrics import (
    fraction_above,
    h4_correlation,
    pair_cosines,
    pearson,
    spearman,
    _rank,
)


def test_fraction_above():
    v = np.array([0.1, 0.5, 0.9, 0.95, 1.0])
    assert fraction_above(v, 0.9) == 3 / 5
    assert fraction_above(np.array([]), 0.5) == 0.0


def test_pearson_perfect():
    x = np.arange(10.0)
    assert abs(pearson(x, 2 * x + 1) - 1.0) < 1e-12
    assert abs(pearson(x, -x) + 1.0) < 1e-12


def test_pearson_constant_is_nan():
    assert np.isnan(pearson(np.ones(5), np.arange(5.0)))


def test_rank_handles_ties():
    # tied values share the average rank (1.5, 1.5) then 3
    np.testing.assert_allclose(_rank(np.array([5.0, 5.0, 9.0])), [1.5, 1.5, 3.0])


def test_spearman_monotonic_nonlinear():
    x = np.arange(1, 11.0)
    y = x ** 3  # monotonic but very nonlinear -> Spearman == 1, Pearson < 1
    assert abs(spearman(x, y) - 1.0) < 1e-12
    assert pearson(x, y) < 0.99


def test_pair_cosines_on_normalized_vectors():
    e = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = pair_cosines(e, [(0, 1), (0, 2)])
    np.testing.assert_allclose(c, [1.0, 0.0], atol=1e-6)


def test_h4_correlation_keys():
    e = np.random.RandomState(0).randn(20, 4).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    out = h4_correlation(e, e, [(0, 1), (2, 3), (4, 5), (6, 7)])
    assert {"spearman", "pearson", "n_pairs"} <= set(out)
    # response==prompt embeddings -> identical cosines -> perfect rank corr
    assert abs(out["spearman"] - 1.0) < 1e-9
