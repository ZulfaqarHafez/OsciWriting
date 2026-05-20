"""PRD §4a cost model. Thresholds are OUTPUTS of this, not inputs.

Every request first hits the cache; on miss it is routed to either the small model
or the frontier model. All costs are normalized to ``c_frontier = 1.0``.

    C = c_cache + (1 - p_cache) * [ p_small * c_small + (1 - p_small) * 1.0 ]
    S = 1 - C

Given a target savings ``s_target`` and a routing split ``p_small``, the minimum
viable cache-hit rate ``p_cache`` is closed-form; that minimum becomes the H1 / H3
threshold the clustering and nearest-neighbor analysis must clear (PRD §3, §4a).

Pure-Python on purpose: this must import and test without the ML stack.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from .config import CONFIG, DOCS, CostConstants


def miss_path_cost(p_small: float, c_small: float) -> float:
    """Expected cost of a cache-miss request before adding the lookup cost."""
    return p_small * c_small + (1.0 - p_small) * 1.0


def blended_cost(p_cache: float, p_small: float, c_cache: float, c_small: float) -> float:
    return c_cache + (1.0 - p_cache) * miss_path_cost(p_small, c_small)


def savings(p_cache: float, p_small: float, c_cache: float, c_small: float) -> float:
    return 1.0 - blended_cost(p_cache, p_small, c_cache, c_small)


@dataclass(frozen=True)
class MinCacheResult:
    min_p_cache: float
    feasible: bool
    reason: str


def min_p_cache(
    s_target: float, p_small: float, c_cache: float, c_small: float
) -> MinCacheResult:
    """Smallest p_cache that achieves S >= s_target.

    Max achievable savings is ``1 - c_cache`` (everything cached still pays the
    lookup), so the target is infeasible exactly when ``s_target > 1 - c_cache``
    i.e. ``headroom <= 0``. Below that, ``needed`` is always in (-inf, 1).
    """
    headroom = 1.0 - s_target - c_cache
    if headroom <= 0.0:
        return MinCacheResult(
            1.0, False, "max savings is 1 - c_cache; target unreachable, lower s_target or c_cache"
        )
    m = miss_path_cost(p_small, c_small)
    needed = 1.0 - headroom / m
    if needed < 0.0:
        return MinCacheResult(
            0.0, True, "target met with no cache; redundancy not required for the economics"
        )
    return MinCacheResult(needed, True, "ok")


@dataclass(frozen=True)
class DerivedThresholds:
    """PRD §4a: T1 and T3 are derived; S3/T5 are calibrated at run time from H5."""

    T1: float
    T3: float
    S3: float
    T5: float
    feasible: bool
    note: str
    provisional: bool


def derive_thresholds(cost: CostConstants, base=CONFIG.thresholds) -> DerivedThresholds:
    res = min_p_cache(cost.s_target, cost.p_small, cost.c_cache, cost.c_small)
    derived = round(res.min_p_cache, 4)
    note = res.reason if not res.feasible else (
        f"min viable cache-hit rate = {derived:.4f} at p_small={cost.p_small}"
    )
    return DerivedThresholds(
        T1=derived,
        T3=derived,
        S3=base.S3,  # calibrated from H5 at run time, not from prices
        T5=base.T5,  # judge acceptability target; default from PRD §3
        feasible=res.feasible,
        note=note,
        provisional=cost.provisional,
    )


def sensitivity_table(cost: CostConstants) -> list[dict]:
    """Min p_cache across plausible c_small and p_small. PRD §12 requires this."""
    rows: list[dict] = []
    for c_small in (0.02, 0.05, 0.10, 0.20):
        for p_small in (0.25, 0.50, 0.75):
            r = min_p_cache(cost.s_target, p_small, cost.c_cache, c_small)
            rows.append(
                {
                    "c_small": c_small,
                    "p_small": p_small,
                    "min_p_cache": round(r.min_p_cache, 4),
                    "feasible": r.feasible,
                }
            )
    return rows


def render_doc(cost: CostConstants) -> str:
    d = derive_thresholds(cost)
    sens = sensitivity_table(cost)
    flag = "PROVISIONAL — replace with real prices before the decision run" if cost.provisional else "FROZEN"
    lines = [
        "# Cost Model and Derived Thresholds (PRD §4a)",
        "",
        f"**Status:** {flag}",
        "",
    ]
    if not cost.provisional:
        lines += [
            "**Source:** Anthropic API pricing, May 2026: Opus 4.7 = $5/$25 per "
            "MTok (frontier), Haiku 4.5 = $1/$5 per MTok (small). 1:5 ratio "
            "across both input and output → `c_small = 0.20` with no token-mix "
            "assumption.",
            "",
        ]
    lines += [
        "## Constants (normalized to c_frontier = 1.0)",
        "",
        "| Constant | Value | Meaning |",
        "| --- | --- | --- |",
        f"| c_cache | {cost.c_cache} | one cache lookup (embed + vector search) |",
        f"| c_small | {cost.c_small} | small-model cost as fraction of frontier |",
        f"| p_small | {cost.p_small} | fraction of cache-misses routed to small model |",
        f"| s_target | {cost.s_target} | required savings to call the project worthwhile |",
        "",
        "## Derived thresholds",
        "",
        f"- **T1 (H1 top-50 coverage)** = `{d.T1}`",
        f"- **T3 (H3 fraction at >= S3)** = `{d.T3}`",
        f"- **S3** = `{d.S3}` (calibrated from H5 at run time; this is the PRD default)",
        f"- **T5 (H5 acceptability, gating)** = `{d.T5}` (PRD §3 default)",
        f"- feasible: `{d.feasible}` — {d.note}",
        "",
        "## Sensitivity (min viable cache-hit rate)",
        "",
        "| c_small | p_small | min p_cache | feasible |",
        "| --- | --- | --- | --- |",
    ]
    for r in sens:
        lines.append(
            f"| {r['c_small']} | {r['p_small']} | {r['min_p_cache']} | {r['feasible']} |"
        )
    lines += [
        "",
        "## How this binds the rubric",
        "",
        "T1 and T3 above are not chosen by feel; they are the minimum cache-hit rate "
        "that makes blended cost clear `s_target`. If the writing subset cannot reach "
        "T1/T3 (above control, PRD §4), the economics do not work and the rubric says "
        "so. Re-run this module after replacing the constants with real pricing and "
        "commit the frozen version before the decision run.",
        "",
    ]
    return "\n".join(lines)


def write_doc(cost: CostConstants = CONFIG.cost) -> str:
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "cost_model.md"
    path.write_text(render_doc(cost), encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PRD §4a cost model / threshold solver")
    ap.add_argument("--write-doc", action="store_true", help="write docs/cost_model.md")
    args = ap.parse_args(argv)
    cost = CONFIG.cost
    d = derive_thresholds(cost)
    print(json.dumps(asdict(d), indent=2))
    if args.write_doc:
        print(f"wrote {write_doc(cost)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
