"""HDBSCAN clustering + the H1 sweep. PRD §3 (H1), §8.4, §8.5.

H1 is reported as the envelope over a hyperparameter grid, not a single number. A
high noise fraction across the sweep is recorded as "not separable at this
granularity" — distinct from "no structure exists" (PRD §8.4).
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .config import CONFIG
from .reduce import reduce


def cluster(
    reduced: np.ndarray, min_cluster_size: int = 20, min_samples: int = 5
) -> np.ndarray:
    import hdbscan  # heavy; local

    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return model.fit_predict(reduced)


def top_n_coverage(labels: np.ndarray, n: int = CONFIG.top_n_clusters) -> float:
    """Sum of the N largest cluster sizes / total points (noise included)."""
    total = len(labels)
    if total == 0:
        return 0.0
    sizes = Counter(int(c) for c in labels if int(c) != -1)
    top = sorted(sizes.values(), reverse=True)[:n]
    return sum(top) / total


def noise_fraction(labels: np.ndarray) -> float:
    total = len(labels)
    if total == 0:
        return 1.0
    return float(np.sum(labels == -1)) / total


@dataclass
class SweepResult:
    cells: list[dict]
    coverage_median: float
    coverage_min: float
    coverage_max: float
    noise_median: float
    separable: bool  # False if noise > 0.60 across most of the grid


def sweep(emb: np.ndarray, n: int = CONFIG.top_n_clusters) -> SweepResult:
    """Grid over n_components x min_cluster_size x min_samples (PRD §8.4)."""
    g = CONFIG.sweep
    cells: list[dict] = []
    reduced_cache: dict[int, np.ndarray] = {}
    for nc in g.n_components:
        if nc not in reduced_cache:
            reduced_cache[nc] = reduce(emb, n_components=nc)
        red = reduced_cache[nc]
        for mcs in g.min_cluster_size:
            for ms in g.min_samples:
                labels = cluster(red, min_cluster_size=mcs, min_samples=ms)
                cells.append(
                    {
                        "n_components": nc,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                        "coverage": top_n_coverage(labels, n),
                        "noise_fraction": noise_fraction(labels),
                        "n_clusters": len(set(int(c) for c in labels) - {-1}),
                    }
                )
    cov = [c["coverage"] for c in cells]
    noise = [c["noise_fraction"] for c in cells]
    high_noise = sum(1 for x in noise if x > 0.60)
    return SweepResult(
        cells=cells,
        coverage_median=statistics.median(cov),
        coverage_min=min(cov),
        coverage_max=max(cov),
        noise_median=statistics.median(noise),
        separable=high_noise <= len(noise) // 2,
    )


def cluster_examples(
    labels: np.ndarray, prompts: list[str], top_k: int = CONFIG.top_k_inspect
) -> list[dict]:
    """Top-k clusters by size with a few example prompts each (PRD §10)."""
    sizes = Counter(int(c) for c in labels if int(c) != -1)
    out: list[dict] = []
    for cid, size in sizes.most_common(top_k):
        idx = [i for i, c in enumerate(labels) if int(c) == cid][:5]
        out.append(
            {
                "cluster": cid,
                "size": size,
                "examples": [prompts[i][:300] for i in idx],
            }
        )
    return out


def response_indices_by_cluster(
    labels: np.ndarray, top_k: int = CONFIG.top_k_inspect, per_cluster: int = 8
) -> dict[int, list[int]]:
    """Row indices per top cluster for the H2 judge sample (PRD §8.5)."""
    sizes = Counter(int(c) for c in labels if int(c) != -1)
    result: dict[int, list[int]] = {}
    for cid, _ in sizes.most_common(top_k):
        result[cid] = [i for i, c in enumerate(labels) if int(c) == cid][:per_cluster]
    return result
