"""Deduplication (added PRD v2.1). PRD §8.6.

The N=5000 pilot showed WildChat's writing subset is dominated by mass copy-paste
viral prompts (one "Midjourney prompt generator" jailbreak repeated verbatim
hundreds of times). Without collapsing those, H1 degenerates (a few duplicate blobs
swallow everything), and H3/H4 measure copy-paste volume, not the semantic
redundancy-across-distinct-phrasings the thesis is about. Caching identical strings
is a hashmap; it is not the project.

Two passes:
- exact: collapse records whose whitespace/case-normalized prompt is identical.
  Cheap, no embedding, run on the record pool before filtering.
- near: greedy collapse of points with cosine >= 0.98 to one representative. Run
  per arm after embedding, before any metric.

Both keep the first occurrence (stable) so seeded reproducibility holds. The dedup
rate is itself a reported finding (PRD §8.6, §10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .data import Record

_WS = re.compile(r"\s+")

NEAR_THRESHOLD = 0.98


def normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


@dataclass
class DedupStats:
    before: int
    after: int

    @property
    def rate(self) -> float:
        return 0.0 if self.before == 0 else 1.0 - self.after / self.before

    def as_dict(self) -> dict:
        return {"before": self.before, "after": self.after, "rate": self.rate}


def exact_dedup(records: list[Record]) -> tuple[list[Record], DedupStats]:
    seen: set[str] = set()
    kept: list[Record] = []
    for r in records:
        key = normalize(r.prompt)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept, DedupStats(len(records), len(kept))


def near_dedup_indices(
    emb: np.ndarray, threshold: float = NEAR_THRESHOLD
) -> np.ndarray:
    """Indices to KEEP. Greedy: walk in order; keep a point, then drop every
    later point within cosine >= threshold of it (one representative per blob)."""
    n = len(emb)
    if n == 0:
        return np.empty(0, dtype=int)
    from sklearn.neighbors import NearestNeighbors  # heavy; local

    nn = NearestNeighbors(metric="cosine").fit(emb)
    # cosine distance = 1 - cosine similarity
    neighbors = nn.radius_neighbors(
        emb, radius=1.0 - threshold, return_distance=False
    )
    removed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for i in range(n):
        if removed[i]:
            continue
        keep.append(i)
        for j in neighbors[i]:
            if j > i:
                removed[j] = True
    return np.array(keep, dtype=int)


def near_dedup(
    emb: np.ndarray, threshold: float = NEAR_THRESHOLD
) -> tuple[np.ndarray, DedupStats]:
    keep = near_dedup_indices(emb, threshold)
    return keep, DedupStats(len(emb), len(keep))
