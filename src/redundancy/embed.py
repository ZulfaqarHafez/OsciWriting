"""Sentence-embedding wrapper. PRD §8.3.

all-MiniLM-L6-v2, L2-normalized so cosine == dot product. The model is loaded once
and cached at module level. The embedding confound is acknowledged in the PRD: every
absolute cosine here is embedding-relative and descriptive only; the decision rests
on the judge (H5) and the control gaps, not on these numbers in isolation.
"""

from __future__ import annotations

import numpy as np

from .config import CONFIG

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy; local

        _MODEL = SentenceTransformer(CONFIG.embed_model)
    return _MODEL


def embed(
    texts: list[str], batch_size: int = CONFIG.embed_batch_size
) -> np.ndarray:
    """Return float32 (n, 384) L2-normalized embeddings."""
    vecs = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32, copy=False)
