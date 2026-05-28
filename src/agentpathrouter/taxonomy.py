"""Workflow-regime taxonomy — the headline contribution.

Given a corpus of agent execution traces, classify the workload into one
of three regimes and recommend an architecture:

    DETERMINISTIC  — path entropy is near zero. The "agent" is a dressed-up
                     pipeline. Recommendation: replace agentic reasoning
                     with a deterministic pipeline; reserve LLM calls for
                     genuinely open-ended steps (e.g. final synthesis).

    HYBRID         — moderate path entropy. Most runs follow a small set of
                     paths; a long tail of edge cases still needs LLM
                     reasoning. Recommendation: cache + speculative
                     prefetch + small-model routing for the predictable
                     majority, frontier model for the tail.

    FULL_AGENT     — high path entropy. Each input drives a genuinely
                     different execution. Recommendation: keep the
                     frontier-model agent; AEE interventions don't help.

The regime thresholds are PRELIMINARY — calibrated on the synthetic
corporate-workflow corpus only. Final values should come from a sweep
across Yunjue / Nemotron-Agentic / Hermes once the HF egress is unblocked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .entropy import coverage_at_k, coverage_curve


class Regime(str, Enum):
    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"
    FULL_AGENT = "full_agent"


@dataclass(frozen=True)
class RegimeReport:
    regime: Regime
    path_entropy_bits: float
    entropy_ratio: float        # entropy / log2(N_unique_paths) — 0 = collapsed, 1 = uniform
    top3_coverage: float
    top10_coverage: float
    recommendation: str
    rationale: str

    def as_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "path_entropy_bits": round(self.path_entropy_bits, 4),
            "entropy_ratio": round(self.entropy_ratio, 4),
            "top3_coverage": round(self.top3_coverage, 4),
            "top10_coverage": round(self.top10_coverage, 4),
            "recommendation": self.recommendation,
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# Thresholds (preliminary; calibrate on real corpora)
# ---------------------------------------------------------------------------
#
# Two signals are combined because either alone can mislead:
#
#   * entropy_ratio = H(paths) / log2(|unique paths|)
#       Low ratio = mass concentrated on a few paths even if the path
#       VOCABULARY is large. This is the "is it actually random?" signal.
#       Range [0, 1].
#
#   * top-K coverage
#       Fraction of traces covered by the K most common paths. Direct
#       answer to "how many paths do I need to handle to cover most runs?"
#
# A corpus is DETERMINISTIC if either signal is extreme, and FULL_AGENT
# only if BOTH are diffuse. HYBRID is the middle.

T_DETERMINISTIC_ENTROPY_RATIO = 0.30
T_DETERMINISTIC_TOP3_COVERAGE = 0.90

T_FULL_AGENT_ENTROPY_RATIO = 0.75
T_FULL_AGENT_TOP10_COVERAGE = 0.50  # less than half the traces hit top-10


def classify(sequences) -> RegimeReport:
    """Classify a corpus and produce the recommendation."""
    cov = coverage_curve(sequences, top_n=10)
    cov_k = coverage_at_k(sequences, [3, 10])
    H = cov.entropy_bits
    n_unique = cov.unique_paths

    # entropy_ratio normalises against the maximum possible entropy for the
    # observed unique-path count. With 1 path the ratio is 0 by convention.
    if n_unique <= 1:
        ratio = 0.0
    else:
        ratio = H / math.log2(n_unique)

    top3 = cov_k[3]
    top10 = cov_k[10]

    # DETERMINISTIC if the corpus is collapsed in either dimension.
    if ratio < T_DETERMINISTIC_ENTROPY_RATIO or top3 >= T_DETERMINISTIC_TOP3_COVERAGE:
        regime = Regime.DETERMINISTIC
        recommendation = (
            "Replace agentic reasoning with a deterministic pipeline. "
            "Hard-code the top-K paths; reserve LLM calls only for steps "
            "that produce genuinely open-ended output (e.g. final "
            "summarisation). Expect 80-90% cost reduction at <1% quality drop."
        )
        triggers = []
        if ratio < T_DETERMINISTIC_ENTROPY_RATIO:
            triggers.append(
                f"entropy ratio {ratio:.2f} < {T_DETERMINISTIC_ENTROPY_RATIO}"
            )
        if top3 >= T_DETERMINISTIC_TOP3_COVERAGE:
            triggers.append(
                f"top-3 coverage {top3*100:.1f}% ≥ "
                f"{T_DETERMINISTIC_TOP3_COVERAGE*100:.0f}%"
            )
        rationale = (
            "Triggered by: " + "; ".join(triggers) + ". "
            "The LLM is being asked to make decisions that are functionally "
            "predetermined."
        )
    elif ratio > T_FULL_AGENT_ENTROPY_RATIO and top10 < T_FULL_AGENT_TOP10_COVERAGE:
        regime = Regime.FULL_AGENT
        recommendation = (
            "Keep the frontier-model agent. Path-level caching and routing "
            "will not yield meaningful savings on this workload; reasoning "
            "is genuinely required per-input."
        )
        rationale = (
            f"Entropy ratio {ratio:.2f} (>{T_FULL_AGENT_ENTROPY_RATIO}) and "
            f"top-10 coverage only {top10*100:.1f}% — execution paths are "
            "near-uniformly distributed across the observed vocabulary."
        )
    else:
        regime = Regime.HYBRID
        recommendation = (
            "Apply AgentPathRouter: PathCache + speculative prefetch + "
            "small-model routing for the predictable majority, frontier "
            "model on the long tail. Expect 50-80% cost reduction at <2% "
            "quality drop. Tune small-model confidence threshold to land "
            "under the quality cap."
        )
        rationale = (
            f"Entropy ratio {ratio:.2f}, top-3 coverage {top3*100:.1f}%, "
            f"top-10 coverage {top10*100:.1f}% — concentrated enough to "
            "exploit, diffuse enough that a deterministic pipeline would "
            "miss meaningful edge cases."
        )

    return RegimeReport(
        regime=regime,
        path_entropy_bits=H,
        entropy_ratio=ratio,
        top3_coverage=top3,
        top10_coverage=top10,
        recommendation=recommendation,
        rationale=rationale,
    )
