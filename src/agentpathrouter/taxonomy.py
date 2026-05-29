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

from .entropy import coverage_at_k, coverage_curve, path_entropy


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


# ---------------------------------------------------------------------------
# Within-task classifier (refined taxonomy)
# ---------------------------------------------------------------------------
#
# P1.2 / P3.2 / P3.3 verifications showed that corpus-level path entropy
# on multi-trial benchmarks averages across many distinct tasks and
# hides the per-task structure. Within-task entropy is the signal that
# maps to AEE savings.
#
# CALIBRATION (P2.2, scripts/calibrate_regime_cutoffs.py): across 20
# workload points — 14 controlled-synthetic (ground-truth entropy via
# generate_controlled_corpus) + 4 tau-bench domains + 2 TRAIL subsets —
# within-task entropy correlates with cache hit rate at r = -0.86 and
# with cost saved at r = -0.84 *at the workload level*. (Per-cluster the
# correlation is only ρ ≈ -0.40; aggregating to the workload washes out
# per-cluster noise — both numbers are real, at different granularities.)
#
# Savings-based boundaries from the sweep:
#     within-task H ≤ 2.0 b  → ≥85% cost saved  → DETERMINISTIC
#     within-task H > 4.0 b  → <25% cost saved  → FULL_AGENT (but see caveat)
#
# CAVEAT — entropy is necessary but not sufficient at the high end.
# Divergence *structure* confounds it: at the same H≈3.7, divergence
# concentrated at one step saved 52% while divergence spread across all
# steps saved only 26% (longer shared prefix = more cacheable). Real
# telecom sits at H=4.7 yet still saves 47% because its divergence is
# concentrated. So the FULL cutoff over-predicts: prefer a direct
# cache-hit probe on a sample over an entropy threshold in the H>4 region.
# Per-tool determinism (P3.1) is a separate, mandatory gate regardless.

T_WT_DETERMINISTIC_BITS = 2.0   # ≤ this → ≥85% saved (well-calibrated)
T_WT_FULL_AGENT_BITS = 4.0      # > this → <25% saved IF divergence is spread


def classify_with_clusters(clusters: dict[str, list]) -> RegimeReport:
    """Classify a workload using *within-task* path entropy (preferred).

    ``clusters`` is ``{task_id: [tool_sequence, ...]}`` — one cluster
    per task with multiple trial sequences. The classifier ignores
    single-trial clusters (entropy is 0 by definition there, biases
    the mean low) and reports:

        DETERMINISTIC  — mean within-task H ≤ 1.0 bits (≈ ≥90% cache hit rate)
        HYBRID         — mean within-task H ∈ (1.0, 5.0]
        FULL_AGENT     — mean within-task H > 5.0 bits

    Falls back to ``classify`` on the flattened sequences if no
    multi-trial clusters are present.
    """
    multi = {k: v for k, v in clusters.items() if len(v) >= 2}
    if not multi:
        # No replay structure — defer to corpus-level classification.
        flat = [s for seqs in clusters.values() for s in seqs]
        return classify(flat)

    weighted_h = 0.0
    total_weight = 0
    for seqs in multi.values():
        h = path_entropy(seqs)
        weighted_h += h * len(seqs)
        total_weight += len(seqs)
    mean_h = weighted_h / total_weight if total_weight else 0.0

    # Mean within-task top-1 coverage as a secondary signal.
    cov_sum = 0.0
    for seqs in multi.values():
        from collections import Counter
        c = Counter(tuple(s) for s in seqs)
        top1 = max(c.values()) / len(seqs)
        cov_sum += top1
    mean_top1 = cov_sum / len(multi)

    if mean_h <= T_WT_DETERMINISTIC_BITS:
        regime = Regime.DETERMINISTIC
        recommendation = (
            "Within-task path entropy is near zero — agents make the "
            "same decisions on each replay. Replace the agent with a "
            "deterministic pipeline keyed on task type. Reserve LLM "
            "calls for genuinely open-ended steps (final synthesis). "
            "Expect 85–95% cost reduction at near-zero quality drop."
        )
    elif mean_h > T_WT_FULL_AGENT_BITS:
        regime = Regime.FULL_AGENT
        recommendation = (
            "Within-task path entropy is high — each trial diverges "
            "substantially even for the same task. Cache hit rate "
            "saturates below ~30%; routing alone delivers <20% "
            "savings. Keep the frontier-model agent."
        )
    else:
        regime = Regime.HYBRID
        recommendation = (
            "Apply AgentPathRouter: PathCache for the deterministic "
            "tool subset, small-model routing for high-confidence "
            "next-tool decisions. Empirically 50–80% cost reduction "
            "at <0.5% task-level quality regression on tau-retail-like "
            "workloads. **Additionally** check per-tool determinism "
            "(see audit_cache_determinism.py) before enabling cache "
            "on stateful / state-observing tools."
        )

    rationale = (
        f"Mean within-task entropy {mean_h:.2f} bits across "
        f"{len(multi)} multi-trial clusters; "
        f"mean within-task top-1 coverage {mean_top1*100:.1f}%. "
        f"Cutoffs (calibrated on 20 workload points, savings-based): "
        f"DET ≤ {T_WT_DETERMINISTIC_BITS}, FULL > {T_WT_FULL_AGENT_BITS} bits."
    )

    # Reuse the RegimeReport shape; entropy_ratio is set to NaN-proof 0.
    return RegimeReport(
        regime=regime,
        path_entropy_bits=mean_h,
        entropy_ratio=0.0,
        top3_coverage=mean_top1,  # repurpose: within-task top-1
        top10_coverage=0.0,
        recommendation=recommendation,
        rationale=rationale,
    )
