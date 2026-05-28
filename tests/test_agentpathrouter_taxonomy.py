"""Tests for the workflow-regime taxonomy (the new headline contribution)."""

from agentpathrouter import Regime, classify
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
