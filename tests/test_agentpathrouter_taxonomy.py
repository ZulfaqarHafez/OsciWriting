"""Tests for the workflow-regime taxonomy (the new headline contribution)."""

from agentpathrouter import Regime, classify, classify_with_clusters
from agentpathrouter.synthetic import generate_corpus


def test_collapsed_corpus_is_deterministic():
    seqs = [("a", "b", "c")] * 100
    r = classify(seqs)
    assert r.regime is Regime.DETERMINISTIC
    assert r.path_entropy_bits == 0.0
    assert r.top3_coverage == 1.0
    assert "deterministic pipeline" in r.recommendation.lower()


def test_all_unique_corpus_is_full_agent():
    # 100 distinct single-step paths → entropy ratio = 1.0, top-10 = 10%
    seqs = [(f"tool_{i}",) for i in range(100)]
    r = classify(seqs)
    assert r.regime is Regime.FULL_AGENT
    assert r.entropy_ratio > 0.99
    assert r.top10_coverage <= 0.10
    assert "frontier" in r.recommendation.lower()


def test_synthetic_financial_workflow_is_deterministic_regime():
    """The PRD's canonical 'daily financial report' workflow should land in
    the DETERMINISTIC regime — 80% on one path, top-5 covers everything."""
    corpus = generate_corpus(n=500, seed=0)
    seqs = [t.tools for t in corpus]
    r = classify(seqs)
    assert r.regime is Regime.DETERMINISTIC
    assert r.top3_coverage >= 0.9


def test_hybrid_regime_when_moderately_spread():
    # Two roughly equal common paths + a long tail
    seqs = (
        [("a", "b")] * 40
        + [("a", "c")] * 40
        + [(f"x{i}",) for i in range(40)]
    )
    r = classify(seqs)
    assert r.regime is Regime.HYBRID


def test_regime_report_serialises():
    r = classify([("a", "b")] * 10)
    d = r.as_dict()
    assert d["regime"] == "deterministic"
    assert "path_entropy_bits" in d
    assert "top3_coverage" in d
    assert "recommendation" in d
    assert "rationale" in d


def test_single_path_corpus_does_not_divide_by_zero():
    r = classify([("a",)] * 50)
    assert r.regime is Regime.DETERMINISTIC
    assert r.entropy_ratio == 0.0


# ---- Within-task classifier ---------------------------------------------


def test_within_task_collapsed_is_deterministic():
    clusters = {f"task_{i}": [("a", "b", "c")] * 4 for i in range(10)}
    r = classify_with_clusters(clusters)
    assert r.regime is Regime.DETERMINISTIC
    assert r.path_entropy_bits == 0.0


def test_within_task_high_entropy_is_full_agent():
    # 10 tasks × 8 unique trial paths each → within-task H = log2(8) = 3
    # bits. Should land in HYBRID, not FULL.
    clusters = {
        f"task_{i}": [(f"t{i}_{j}",) for j in range(8)]
        for i in range(10)
    }
    r = classify_with_clusters(clusters)
    # 3 bits is in the hybrid band
    assert r.regime is Regime.HYBRID

    # Now push to 6 bits: 64 unique trials per task
    big = {
        f"task_{i}": [(f"t{i}_{j}",) for j in range(64)]
        for i in range(5)
    }
    r2 = classify_with_clusters(big)
    assert r2.regime is Regime.FULL_AGENT


def test_within_task_falls_back_to_corpus_when_no_replays():
    # Each cluster has exactly one trial → no within-task signal.
    clusters = {f"task_{i}": [("a", "b")] for i in range(20)}
    r = classify_with_clusters(clusters)
    # Should defer to corpus-level classify; single path → DETERMINISTIC
    assert r.regime is Regime.DETERMINISTIC


def test_within_task_ignores_single_trial_clusters():
    """Multi-trial clusters should drive the verdict; single-trial ones
    must not push it lower by contributing zero-entropy datapoints."""
    clusters = {
        # 8 distinct uniform paths → 3 bits, squarely in the HYBRID band
        # (DET ≤ 2.0, FULL > 4.0).
        "multi": [(f"p{j}",) for j in range(8)],
        "single_1": [("x",)],
        "single_2": [("y",)],
    }
    r = classify_with_clusters(clusters)
    # The signal is dominated by the single multi-trial cluster (3 bits H);
    # the single-trial clusters are skipped. Result lands in HYBRID, not DET.
    assert r.regime is Regime.HYBRID
