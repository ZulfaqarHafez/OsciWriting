"""Prefetch ROI model (PRD v3 §4a).

Per-turn ROI of speculative prefetch:

    roi_per_turn = effective_hit × latency_value_per_hit − K × c_speculate

where `effective_hit = hit_rate(K, T_pred) × quality_at(T_serve)`, all costs
normalized to `c_frontier = 1.0`. HP4 passes iff some (K, T_pred, T_serve,
c_speculate) tuple yields roi > 0.

Provisional defaults match the PRD §3 placeholder structure; freeze
`docs/cost_model_v3.md` before any decision run.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class V3Constants:
    """All costs normalized to c_frontier = 1.0. PROVISIONAL until set."""

    latency_value_per_hit: float = 0.30  # value (units of frontier cost) of one cache hit's latency win
    c_speculate_options: tuple[float, ...] = (0.20, 0.50, 1.00)  # Haiku / Sonnet / Opus speculator
    K_grid: tuple[int, ...] = (1, 3, 5, 10, 25)
    t_pred_grid: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95)
    t_serve_grid: tuple[float, ...] = (0.90, 0.95, 0.98)
    provisional: bool = True


def roi_per_turn(
    hit_rate: float,
    quality: float,
    K: int,
    c_speculate: float,
    latency_value: float,
) -> float:
    return hit_rate * quality * latency_value - K * c_speculate


@dataclass(frozen=True)
class ParetoCell:
    K: int
    t_pred: float
    t_serve: float
    c_speculate: float
    hit_rate: float
    quality: float
    effective: float
    roi: float
    feasible: bool


def pareto_sweep(
    hit_table: dict[tuple[int, float], float],
    quality_table: dict[float, float],
    constants: V3Constants,
) -> list[ParetoCell]:
    """Cross every (K, t_pred, t_serve, c_speculate); compute ROI; flag feasible.

    `hit_table[(K, t_pred)]` = HP1 hit rate.
    `quality_table[t_serve]` = HP2 judge acceptability at that serve threshold.
    """
    cells: list[ParetoCell] = []
    for K, t_pred, t_serve, c_s in itertools.product(
        constants.K_grid,
        constants.t_pred_grid,
        constants.t_serve_grid,
        constants.c_speculate_options,
    ):
        hit = hit_table.get((K, t_pred), 0.0)
        quality = quality_table.get(t_serve, 0.0)
        roi = roi_per_turn(hit, quality, K, c_s, constants.latency_value_per_hit)
        eff = hit * quality
        cells.append(
            ParetoCell(
                K=K,
                t_pred=t_pred,
                t_serve=t_serve,
                c_speculate=c_s,
                hit_rate=hit,
                quality=quality,
                effective=eff,
                roi=roi,
                feasible=roi > 0,
            )
        )
    return cells


def best_cell(cells: list[ParetoCell]) -> ParetoCell | None:
    feasible = [c for c in cells if c.feasible]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c.roi)


def render_doc(constants: V3Constants) -> str:
    flag = "PROVISIONAL — set real values before the decision run" if constants.provisional else "FROZEN"
    lines = [
        "# Prefetch ROI Model (PRD v3 §4a)",
        "",
        f"**Status:** {flag}",
        "",
        "Per-turn ROI of speculative prefetch, all costs normalized to "
        "`c_frontier = 1.0`:",
        "",
        "```",
        "roi_per_turn = hit_rate × quality_at_T_serve × latency_value_per_hit",
        "             − K × c_speculate",
        "```",
        "",
        "## Constants",
        "",
        "| Constant | Value | Meaning |",
        "| --- | --- | --- |",
        f"| latency_value_per_hit | {constants.latency_value_per_hit} | value (fraction of frontier cost) of one prefetch hit's latency win |",
        f"| c_speculate options | {list(constants.c_speculate_options)} | Haiku ~0.20, Sonnet ~0.50, Opus ~1.0 |",
        f"| K grid | {list(constants.K_grid)} | candidates pre-generated per turn |",
        f"| T_pred grid | {list(constants.t_pred_grid)} | hit threshold (predicted vs actual cosine) |",
        f"| T_serve grid | {list(constants.t_serve_grid)} | quality threshold (only serve cached if this close) |",
        "",
        "## How HP4 reads this",
        "",
        "HP4 passes iff there exists a (K, T_pred, T_serve, c_speculate) tuple "
        "with `roi_per_turn > 0`. The Pareto sweep (PRD §8.6) reports every cell; "
        "`prefetch.best_cell()` returns the max-ROI feasible cell (or None → Abandon "
        "on economic grounds even if HP1/HP2 passed).",
        "",
    ]
    return "\n".join(lines)


def write_doc(constants: V3Constants = V3Constants()) -> str:
    from .config import DOCS

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "cost_model_v3.md"
    path.write_text(render_doc(constants), encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PRD v3 §4a ROI model")
    ap.add_argument("--write-doc", action="store_true")
    args = ap.parse_args(argv)
    c = V3Constants()
    print(json.dumps(asdict(c), indent=2))
    if args.write_doc:
        print(f"wrote {write_doc(c)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
