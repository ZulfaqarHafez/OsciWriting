"""Tunables for the redundancy study. PRD §5 conventions, §3 hypotheses, §4a costs.

Nothing here is decided by feel that the PRD says should be derived: the THRESHOLDS
below are PROVISIONAL placeholders. The real T1/T3/S3/T5 are produced by
``cost_model.derive_thresholds`` once real prices are plugged into COST. A run
executed on these placeholders is a pilot, not the decision run (PRD §4a).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

# datasets[<name>] -> (hf_id, split)
DATASETS = {
    "wildchat": ("allenai/WildChat-1M", "train"),
    "lmsys": ("lmsys/lmsys-chat-1m", "train"),
}
LANGUAGE = "English"


@dataclass(frozen=True)
class CostConstants:
    """PRD §4a, all normalized to c_frontier = 1.0. PROVISIONAL until real prices.

    Replace these from the actual frontier / small-model / cache pricing before
    the decision run, then re-run ``cost_model --write-doc``.
    """

    c_cache: float = 0.002  # one cache lookup (embed + vector search)
    c_small: float = 0.05  # small-model cost as a fraction of frontier
    p_small: float = 0.50  # fraction of cache-misses routed to the small model
    s_target: float = 0.50  # "meaningful" = >= 2x cost reduction
    provisional: bool = True


@dataclass(frozen=True)
class Thresholds:
    """PRD §3. PROVISIONAL placeholders; see module docstring."""

    T1: float = 0.40  # H1 top-50 cluster coverage
    S3: float = 0.90  # H3 nearest-neighbor cosine band (calibrated from H5)
    T3: float = 0.10  # H3 fraction of prompts at >= S3
    T5: float = 0.70  # H5 judge acceptability rate (gating)
    H4_rho: float = 0.50  # H4 Spearman
    control_gap_h1: float = 0.10  # subject must beat control by this (fraction)
    control_gap_h3: float = 0.05
    control_gap_h5: float = 0.25
    provisional: bool = True


@dataclass(frozen=True)
class SweepGrid:
    """PRD §8.4 — H1 is an envelope over this grid, not one number."""

    min_cluster_size: tuple[int, ...] = (15, 20, 30, 50)
    min_samples: tuple[int, ...] = (3, 5, 10)
    n_components: tuple[int, ...] = (5, 10, 20)


@dataclass(frozen=True)
class UMAPParams:
    n_neighbors: int = 15
    n_components: int = 10
    metric: str = "cosine"


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    embed_model: str = EMBED_MODEL
    judge_model: str = JUDGE_MODEL
    embed_batch_size: int = 256
    top_n_clusters: int = 50  # H1
    top_k_inspect: int = 10  # H2 / cluster examples
    h4_pairs: int = 10_000
    h4_bins: int = 10
    h5_pairs_per_band: int = 300
    h5_bands: tuple[tuple[float, float], ...] = (
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 0.95),
        (0.95, 1.00),
    )
    cost: CostConstants = field(default_factory=CostConstants)
    thresholds: Thresholds = field(default_factory=Thresholds)
    sweep: SweepGrid = field(default_factory=SweepGrid)
    umap: UMAPParams = field(default_factory=UMAPParams)


CONFIG = Config()


def run_timestamp() -> str:
    """Filesystem-safe UTC stamp (no colons — PRD §5, Windows)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def hf_home() -> str | None:
    return os.environ.get("HF_HOME") or None


def judge_api_key() -> str | None:
    return os.environ.get("JUDGE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
