"""H1/H3/H4 metric math. PRD §8.5.

Correlation (H4) is computed over pairs sampled across the FULL similarity range,
not nearest-neighbor pairs only — pairing every anchor with its top-k neighbors
range-restricts prompt cosine and attenuates the coefficient (the v1 bug). Spearman
is primary; the scatter plot must be inspected before the number is trusted.

numpy-only helpers (pearson/spearman/fraction_above/cosines) so the unit tests run
without sklearn; NN search imports sklearn locally.
"""

from __future__ import annotations

import random

import numpy as np


def fraction_above(values: np.ndarray, threshold: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.asarray(values) >= threshold))


def pair_cosines(emb: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Cosine for each (i, j). Embeddings are L2-normalized so this is a dot."""
    a = emb[[i for i, _ in pairs]]
    b = emb[[j for _, j in pairs]]
    return np.einsum("ij,ij->i", a, b).astype(np.float64)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), matching scipy's tie handling."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rank(x), _rank(y))


def nn_similarity(emb: np.ndarray) -> np.ndarray:
    """For each row, cosine to its nearest *other* row (H3, PRD §8.5)."""
    from sklearn.neighbors import NearestNeighbors  # heavy; local

    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(emb)
    dist, _ = nn.kneighbors(emb)
    return (1.0 - dist[:, 1]).astype(np.float64)


def stratified_pairs(
    emb: np.ndarray,
    n_pairs: int,
    n_bins: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pairs spread across cosine bins in [0, 1] (PRD §8.5, H4).

    Random pairs cover the low/mid range; a nearest-neighbor pass seeds the rare
    high-similarity bins so every stratum is populated.
    """
    rng = random.Random(seed)
    n = len(emb)
    bins: list[list[tuple[int, int]]] = [[] for _ in range(n_bins)]
    per_bin = max(1, n_pairs // n_bins)

    def add(i: int, j: int, c: float) -> None:
        if i == j:
            return
        b = min(n_bins - 1, max(0, int(c * n_bins)))
        if len(bins[b]) < per_bin * 4:  # keep a buffer to subsample from
            bins[b].append((i, j))

    # random pairs
    for _ in range(n_pairs * 6):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        add(i, j, float(emb[i] @ emb[j]))

    # NN pass to fill high bins
    from sklearn.neighbors import NearestNeighbors  # heavy; local

    k = min(15, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb)
    dist, idx = nn.kneighbors(emb)
    for i in range(n):
        for col in range(1, k):
            add(i, int(idx[i, col]), 1.0 - float(dist[i, col]))

    out: list[tuple[int, int]] = []
    for b in bins:
        rng.shuffle(b)
        out.extend(b[:per_bin])
    return out


def h4_correlation(
    prompt_emb: np.ndarray,
    response_emb: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict:
    pc = pair_cosines(prompt_emb, pairs)
    rc = pair_cosines(response_emb, pairs)
    return {
        "n_pairs": len(pairs),
        "spearman": spearman(pc, rc),  # primary (PRD §8.5)
        "pearson": pearson(pc, rc),
        "prompt_cos": pc.tolist(),
        "response_cos": rc.tolist(),
    }
