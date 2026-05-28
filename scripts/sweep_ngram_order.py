"""Sweep n-gram order — was n=3 a good default?

The EntropyEstimator default is a 3-gram (2-tool history). That choice
was a guess. This script sweeps n ∈ {1, 2, 3, 4, 5, 6} and reports
the four metrics that depend on the estimator's quality:

    - speculation hit rate     (steps where spec correctly predicted)
    - speculation precision    (hits / total fires)
    - small-model route rate   (steps where confidence ≥ T_route)
    - task-level UB regression (worst-case quality drop)

Better n should push the first two up and the fourth down.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import (  # noqa: E402
    AgentPathRouter, CostModel, NgramEntropyEstimator, RunMetrics,
)
from agentpathrouter.data_sources import load  # noqa: E402
from agentpathrouter.path_cache import PathCache  # noqa: E402


def _stub_tool(name: str):
    def fn(ctx):
        return {"tool": name}
    fn.__name__ = name
    return fn


def run_one(rows, train_seqs, n_order, threshold, route_threshold):
    est = NgramEntropyEstimator(n=n_order).fit(train_seqs)
    tools = {t: _stub_tool(t) for t in {x for s in train_seqs for x in s}}
    # Ensure all test tools are registered too
    for r in rows:
        for t in r["tools"]:
            tools.setdefault(t, _stub_tool(t))

    router = AgentPathRouter(
        tools=tools, estimator=est,
        confidence_threshold=threshold,
        small_model_threshold=route_threshold,
        use_speculation=True,
        use_small_model_routing=True,
    )
    agg = RunMetrics()
    for r in rows:
        psa = r.get("tool_args")
        if not (isinstance(psa, list) and len(psa) == len(r["tools"])):
            psa = None
        _, m = router.run_trace(r["tools"], r.get("args") or {}, per_step_args=psa)
        agg += m
    spec_stats = router.prefetcher.stats
    router.prefetcher.close()
    return {
        "n_order": n_order,
        "spec_hit_rate": round(agg.spec_hits / agg.steps, 4) if agg.steps else 0.0,
        "spec_fires": spec_stats.fires,
        "spec_precision": round(spec_stats.precision, 4),
        "small_model_route_rate": round(agg.small_model_calls / agg.steps, 4) if agg.steps else 0.0,
        "step_level_regression": round(agg.small_model_errors / agg.steps, 4) if agg.steps else 0.0,
        "cache_hit_rate": round(agg.cache_hits / agg.steps, 4) if agg.steps else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tau_bench")
    ap.add_argument("--tau-bench-dir", type=str,
                    default="/tmp/aee_corpora/tau_retail")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Speculation confidence threshold.")
    ap.add_argument("--route-threshold", type=float, default=0.95,
                    help="Small-model routing threshold.")
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "ngram_order_sweep.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = load(args.source, dir_path=args.tau_bench_dir)
    seqs = [tuple(r["tools"]) for r in rows]
    n_train = max(1, int(len(rows) * args.train_frac))
    train_seqs = seqs[:n_train]
    test_rows = rows[n_train:]

    results = []
    print(f"corpus: {args.source}  train_seqs={len(train_seqs)}  test_rows={len(test_rows)}")
    print(f"speculation T={args.threshold}  routing T={args.route_threshold}")
    print()
    print(f"  {'n':>3}  {'cache%':>7}  {'spec%':>7}  {'fires':>7}  {'precision':>10}  "
          f"{'route%':>7}  {'step-qreg%':>11}")
    for n in args.orders:
        r = run_one(test_rows, train_seqs, n, args.threshold, args.route_threshold)
        results.append(r)
        print(f"  {n:>3}  {r['cache_hit_rate']*100:>6.1f}%  "
              f"{r['spec_hit_rate']*100:>6.1f}%  {r['spec_fires']:>7}  "
              f"{r['spec_precision']*100:>9.1f}%  "
              f"{r['small_model_route_rate']*100:>6.1f}%  "
              f"{r['step_level_regression']*100:>10.2f}%")

    args.out.write_text(json.dumps({
        "source": args.source,
        "speculation_threshold": args.threshold,
        "route_threshold": args.route_threshold,
        "results": results,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
