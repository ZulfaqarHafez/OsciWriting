"""Regime-cutoff calibration across many workload slices.

P2.2 left the regime cutoffs "preliminary — 4 corpora is not enough."
This harness widens the workload variety without needing HuggingFace:

    1. tau-bench sliced by domain          (4 points)
    2. tau-bench sliced by (domain, model) (up to ~16 points)
    3. TRAIL sliced by subset              (2 points, corpus-level only)
    4. a CONTROLLED synthetic entropy sweep with known ground truth
       (generate_controlled_corpus at varying concentration)

For each slice it measures:
    - within-task path entropy (mean over multi-trial clusters)
    - actual cache hit rate
    - actual cost saved at a fixed routing threshold
    - upper-bound task-level regression (where reward is available)

Then it derives data-driven regime cutoffs by finding the within-task
entropy boundaries where (a) cache hit rate crosses ~90% (DET/HYBRID)
and (b) cache hit rate falls below ~30% (HYBRID/FULL), and compares
them to the current taxonomy constants.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import (  # noqa: E402
    AgentPathRouter, CostModel, NgramEntropyEstimator, RunMetrics,
)
from agentpathrouter.data_sources import load  # noqa: E402
from agentpathrouter.entropy import path_entropy  # noqa: E402
from agentpathrouter.synthetic import generate_controlled_corpus  # noqa: E402
from agentpathrouter.taxonomy import (  # noqa: E402
    T_WT_DETERMINISTIC_BITS, T_WT_FULL_AGENT_BITS,
)


def _stub(name: str):
    def fn(ctx):
        return {"t": name}
    fn.__name__ = name
    return fn


def within_task_entropy(rows: list[dict]) -> tuple[float, int]:
    clusters: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for r in rows:
        raw = r.get("raw") if isinstance(r.get("raw"), dict) else {}
        tid = raw.get("task_id") or r.get("task_id")
        if tid:
            clusters[str(tid)].append(tuple(r["tools"]))
    multi = {k: v for k, v in clusters.items() if len(v) >= 2}
    if not multi:
        return float("nan"), 0
    wsum, w = 0.0, 0
    for seqs in multi.values():
        wsum += path_entropy(seqs) * len(seqs)
        w += len(seqs)
    return (wsum / w if w else 0.0), len(multi)


def evaluate(rows: list[dict], route_threshold: float = 0.95) -> dict:
    """Run the full ablation on a workload slice and collect metrics."""
    seqs = [tuple(r["tools"]) for r in rows]
    n_train = max(1, int(len(rows) * 0.6))
    train = seqs[:n_train]
    test = rows[n_train:] or rows

    est = NgramEntropyEstimator(n=3).fit(train)
    tools = {t: _stub(t) for t in {x for s in seqs for x in s}}
    cost = CostModel()

    # cache+spec+routing arm
    router = AgentPathRouter(
        tools=tools, estimator=est,
        small_model_threshold=route_threshold,
        use_speculation=True, use_small_model_routing=True,
    )
    agg = RunMetrics()
    per_run = []
    for r in test:
        psa = r.get("tool_args")
        if not (isinstance(psa, list) and len(psa) == len(r["tools"])):
            psa = None
        _, m = router.run_trace(r["tools"], r.get("args") or {}, per_step_args=psa)
        agg += m
        per_run.append(m)
    router.prefetcher.close()

    wt_h, n_clusters = within_task_entropy(rows)
    c = cost.per_1000_runs(per_run)

    # baseline cost for pct-saved
    base_router = AgentPathRouter(
        tools=tools, estimator=est, use_speculation=False,
        use_small_model_routing=False,
    )
    from agentpathrouter.path_cache import PathCache

    class _Null(PathCache):
        def get(self, *a, **k):
            self.stats.misses += 1
            return False, None
        def put(self, *a, **k):
            return None
    base_router.cache = _Null()
    base_runs = []
    for r in test:
        psa = r.get("tool_args")
        if not (isinstance(psa, list) and len(psa) == len(r["tools"])):
            psa = None
        _, m = base_router.run_trace(r["tools"], r.get("args") or {}, per_step_args=psa)
        base_runs.append(m)
    base_router.prefetcher.close()
    base_cost = cost.per_1000_runs(base_runs)["usd_per_1000_runs"]
    saved = 1 - (c["usd_per_1000_runs"] / base_cost) if base_cost else 0.0

    return {
        "n_traces": len(rows),
        "within_task_entropy": round(wt_h, 4) if wt_h == wt_h else None,
        "n_multi_trial_clusters": n_clusters,
        "cache_hit_rate": round(agg.cache_hits / agg.steps, 4) if agg.steps else 0.0,
        "cost_saved": round(saved, 4),
        "step_qreg": round(agg.small_model_errors / agg.steps, 4) if agg.steps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-root", default="/tmp/aee_corpora/tau2-bench/data/tau2/results/final")
    ap.add_argument("--tau-domains", default="/tmp/aee_corpora")  # has tau_retail etc.
    ap.add_argument("--trail-dir", default="/tmp/aee_corpora/trail-benchmark")
    ap.add_argument("--route-threshold", type=float, default=0.95)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "regime_calibration.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    points = []

    # ---- 1. Controlled synthetic sweep (ground truth) ----
    # Two variant counts so the sweep spans low entropy (8 variants, max
    # 3 bits) AND the high-entropy region (64 variants, max 6 bits) where
    # the HYBRID/FULL transition should appear.
    print("=== Controlled synthetic entropy sweep ===")
    print(f"  {'variants':>8} {'conc':>6}  {'within-H':>9}  {'cache%':>7}  {'saved%':>7}")
    # (n_variants, concentration, divergence_breadth)
    sweep = [
        (8, c, 1) for c in [6.0, 4.0, 2.5, 1.5, 1.0, 0.5]
    ] + [
        (64, c, 1) for c in [2.0, 1.2, 0.8, 0.4]
    ] + [
        # High divergence breadth: variants share little structure, so
        # cacheability can collapse → probes the HYBRID/FULL boundary.
        (64, c, 7) for c in [1.5, 0.8, 0.4, 0.1]
    ]
    for n_var, conc, breadth in sweep:
        rows = generate_controlled_corpus(
            n_tasks=40, trials_per_task=20, n_variants=n_var,
            concentration=conc, divergence_breadth=breadth, seed=1,
        )
        ev = evaluate(rows, args.route_threshold)
        ev.update({"slice": f"synthetic_v{n_var}_c{conc}_b{breadth}", "kind": "synthetic"})
        points.append(ev)
        print(f"  {n_var:>5}/{breadth} {conc:>6}  {ev['within_task_entropy']:>9}  "
              f"{ev['cache_hit_rate']*100:>6.1f}%  {ev['cost_saved']*100:>6.1f}%")

    # ---- 2. tau-bench by domain ----
    print("\n=== tau-bench by domain ===")
    print(f"  {'domain':>18}  {'within-H':>9}  {'cache%':>7}  {'saved%':>7}")
    domain_dirs = {
        "retail": f"{args.tau_domains}/tau_retail",
        "airline": f"{args.tau_domains}/tau_airline",
        "telecom": f"{args.tau_domains}/tau_telecom",
        "telecom-workflow": f"{args.tau_domains}/tau_telecom-workflow",
    }
    for dom, d in domain_dirs.items():
        if not Path(d).exists():
            continue
        try:
            rows = load("tau_bench", dir_path=d)
        except Exception as e:  # noqa: BLE001
            print(f"  {dom}: load failed ({e})")
            continue
        ev = evaluate(rows, args.route_threshold)
        ev.update({"slice": f"tau:{dom}", "kind": "tau_domain"})
        points.append(ev)
        wh = ev["within_task_entropy"]
        print(f"  {dom:>18}  {wh if wh is not None else 'n/a':>9}  "
              f"{ev['cache_hit_rate']*100:>6.1f}%  {ev['cost_saved']*100:>6.1f}%")

    # ---- 3. TRAIL subsets (corpus-level only; no replays) ----
    print("\n=== TRAIL subsets (no within-task replays) ===")
    for sub in ["gaia", "swe_bench"]:
        try:
            rows = load("trail", dir_path=args.trail_dir, subset=sub)
        except Exception as e:  # noqa: BLE001
            print(f"  {sub}: {e}")
            continue
        ev = evaluate(rows, args.route_threshold)
        ev.update({"slice": f"trail:{sub}", "kind": "trail"})
        points.append(ev)
        print(f"  {sub:>18}  corpus-H n/a  cache {ev['cache_hit_rate']*100:.1f}%  "
              f"saved {ev['cost_saved']*100:.1f}%")

    # ---- 4. Derive data-driven cutoffs from points with within-task H ----
    wt_points = [p for p in points if p["within_task_entropy"] is not None]
    wt_points.sort(key=lambda p: p["within_task_entropy"])

    def pearson(xs, ys):
        if len(xs) < 2:
            return 0.0
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        import math
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx * dy else 0.0

    xs = [p["within_task_entropy"] for p in wt_points]
    cache_ys = [p["cache_hit_rate"] for p in wt_points]
    saved_ys = [p["cost_saved"] for p in wt_points]
    r_cache = pearson(xs, cache_ys)
    r_saved = pearson(xs, saved_ys)

    # Cutoffs by ACTIONABILITY (savings-based):
    #   DET  = highest within-task H still saving >= 85% (just pipeline it)
    #   FULL = lowest within-task H where savings drop below 25% (skip AEE)
    det_boundary = None
    for p in wt_points:
        if p["cost_saved"] >= 0.85:
            det_boundary = p["within_task_entropy"]
    full_boundary = None
    for p in wt_points:
        if p["cost_saved"] < 0.25:
            full_boundary = p["within_task_entropy"]
            break

    calibration = {
        "route_threshold": args.route_threshold,
        "n_workload_points": len(points),
        "n_points_with_within_task_entropy": len(wt_points),
        "correlation_within_task_H_vs_cache_hit": round(r_cache, 4),
        "correlation_within_task_H_vs_cost_saved": round(r_saved, 4),
        "current_cutoffs": {
            "deterministic_max_bits": T_WT_DETERMINISTIC_BITS,
            "full_agent_min_bits": T_WT_FULL_AGENT_BITS,
        },
        "data_driven_cutoffs": {
            "deterministic_max_bits": det_boundary,
            "full_agent_min_bits": full_boundary,
            "method": "DET = highest within-task H still saving >=85%; "
                      "FULL = lowest within-task H with <25% saved",
            "note": ("full_boundary == None means no tested replay workload "
                     "is genuinely FULL_AGENT — even the highest-entropy real "
                     "domain still benefits from AEE"),
        },
        "points": points,
    }
    args.out.write_text(json.dumps(calibration, indent=2))

    print("\n=== Calibration ===")
    print(f"workload points: {len(points)}  (with within-task H: {len(wt_points)})")
    print(f"correlation within-task H vs cache hit:  r = {r_cache:.3f}")
    print(f"correlation within-task H vs cost saved: r = {r_saved:.3f}")
    print(f"current cutoffs:     DET<= {T_WT_DETERMINISTIC_BITS}  FULL> {T_WT_FULL_AGENT_BITS}")
    print(f"data-driven (savings-based): DET<= {det_boundary}  FULL> {full_boundary}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
