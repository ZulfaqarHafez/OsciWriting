"""UMAP reduction before clustering. PRD §8.4.

HDBSCAN density estimation degrades on raw 384-dim vectors, so reduce first
(BERTopic-style). Seeded for reproducibility (PRD §5 conventions).
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG


def reduce(
    emb: np.ndarray,
    n_components: int = CONFIG.umap.n_components,
    seed: int = CONFIG.seed,
) -> np.ndarray:
    from umap import UMAP  # heavy; local

    reducer = UMAP(
        n_neighbors=CONFIG.umap.n_neighbors,
        n_components=n_components,
        metric=CONFIG.umap.metric,
        random_state=seed,
    )
    return reducer.fit_transform(emb).astype(np.float32, copy=False)
