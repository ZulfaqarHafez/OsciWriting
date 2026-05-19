"""H1 degeneracy guard (PRD §8.4, added v2.1)."""

from redundancy.cluster import is_degenerate


def test_degenerate_when_coverage_pinned_and_no_noise():
    # the pilot's false-pass shape: cov 1.0 everywhere, 0 noise, no variation
    assert is_degenerate(cov_min=1.0, cov_max=1.0, noise_median=0.0)


def test_not_degenerate_with_some_noise():
    assert not is_degenerate(cov_min=0.99, cov_max=1.0, noise_median=0.05)


def test_not_degenerate_when_sweep_varies():
    # coverage moves across the grid -> real structure, not a single mega-blob
    assert not is_degenerate(cov_min=0.55, cov_max=0.95, noise_median=0.0)


def test_not_degenerate_low_coverage():
    assert not is_degenerate(cov_min=0.30, cov_max=0.40, noise_median=0.10)
