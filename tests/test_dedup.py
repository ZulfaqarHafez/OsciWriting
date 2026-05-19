"""Dedup stage (PRD §8.6, added v2.1)."""

import numpy as np

from redundancy.data import Record
from redundancy.dedup import DedupStats, exact_dedup, near_dedup_indices, normalize


def _rec(p: str) -> Record:
    return Record(hash=str(hash(p)), prompt=p, response="r")


def test_normalize_collapses_case_and_whitespace():
    assert normalize("  Write   an\nEMAIL ") == "write an email"


def test_exact_dedup_collapses_normalized_duplicates():
    recs = [
        _rec("Write an email"),
        _rec("write   an    email"),  # same after normalize
        _rec("WRITE AN EMAIL\n"),  # same after normalize
        _rec("Write a report"),
    ]
    kept, stats = exact_dedup(recs)
    assert [r.prompt for r in kept] == ["Write an email", "Write a report"]
    assert stats.before == 4 and stats.after == 2
    assert stats.rate == 0.5


def test_exact_dedup_keeps_first_occurrence_stable():
    recs = [_rec("a"), _rec("b"), _rec("a")]
    kept, _ = exact_dedup(recs)
    assert [r.prompt for r in kept] == ["a", "b"]


def test_dedup_stats_rate_edge():
    assert DedupStats(0, 0).rate == 0.0
    assert DedupStats(10, 10).rate == 0.0


def test_near_dedup_collapses_blob_to_one_representative():
    # three near-identical unit vectors + one orthogonal -> keep 2
    base = np.array([1.0, 0.0], dtype=np.float32)
    near = np.array([np.cos(0.01), np.sin(0.01)], dtype=np.float32)  # cos ~0.99995
    near2 = np.array([np.cos(0.02), np.sin(0.02)], dtype=np.float32)
    orth = np.array([0.0, 1.0], dtype=np.float32)
    emb = np.vstack([base, near, near2, orth])
    keep = near_dedup_indices(emb, threshold=0.98)
    assert 0 in keep  # first of the blob is the representative
    assert 3 in keep  # orthogonal point survives
    assert len(keep) == 2


def test_near_dedup_keeps_all_when_distinct():
    emb = np.eye(4, dtype=np.float32)
    keep = near_dedup_indices(emb, threshold=0.98)
    assert len(keep) == 4
