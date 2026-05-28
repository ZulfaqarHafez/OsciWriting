"""Test the refined-taxonomy claim: within-task entropy predicts cacheability.

After the v2 measurement, findings.md asserted that *within-task path
entropy* is the signal that actually predicts cache hit rate on multi-
trial corpora. That's the claim that lets us reframe the taxonomy as
"cluster traces by task, then classify on per-cluster entropy."

If the claim is true, per-cluster within-task entropy and per-cluster
cache hit rate should correlate strongly. This script measures the
correlation across all (task_id) clusters in a corpus.

Usage:
    python scripts/test_entropy_cacheability_correlation.py \\
        --source tau_bench \\
        --tau-bench-dir /tmp/aee_corpora/tau2-bench/data/tau2/results/final
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import AgentPathRouter, RunMetrics  # noqa: E402
from agentpathrouter.data_sources import load  # noqa: E402
from agentpathrouter.entropy import path_entropy  # noqa: E402
from agentpathrouter.path_cache import PathCache  # noqa: E402


def _stub_tool(name: str):
    def fn(ctx):
        return {"tool": name, "ctx_hash": hash(json.dumps(ctx, sort_keys=True, default=str))}
    fn.__name__ = name
    return fn


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx * dy else 0.0


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    def rank(vs):
        # Average ranks for ties
        s = sorted((v, i) for i, v in enumerate(vs))
        r = [0.0] * len(vs)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1][0] == s[i][0]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[s[k][1]] = avg_rank
            i = j + 1
        return r
    return pearson_r(rank(xs), rank(ys))


def cluster_cache_hit_rate(traces: list[dict]) -> tuple[float, int]:
    """Run a fresh PathCache over a cluster's traces and report hit rate."""
    tools_used = {t for trace in traces for t in trace["tools"]}
    registry = {n: _stub_tool(n) for n in tools_used}
    cache = PathCache()
    router = AgentPathRouter(
        tools=registry,
        cache=cache,
        use_speculation=False,
        use_small_model_routing=False,
    )
    agg = RunMetrics()
    for tr in traces:
        psa = tr.get("tool_args")
        if not (isinstance(psa, list) and len(psa) == len(tr["tools"])):
            psa = None
        _, m = router.run_trace(tr["tools"], tr.get("args") or {}, per_step_args=psa)
        agg += m
    router.prefetcher.close()
    return (agg.cache_hits / agg.steps if agg.steps else 0.0), agg.steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["tau_bench", "trail"])
    ap.add_argument("--tau-bench-dir", type=str, default=None)
    ap.add_argument("--trail-dir", type=str, default=None)
    ap.add_argument("--min-trials-per-cluster", type=int, default=2)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "entropy_cacheability.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "tau_bench":
        rows = load("tau_bench", dir_path=args.tau_bench_dir)
    else:
        rows = load("trail", dir_path=args.trail_dir, subset="all")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tid = (r.get("raw") or {}).get("task_id") if isinstance(r.get("raw"), dict) else None
        tid = tid or r.get("task_id") or r["id"]
        by_task[str(tid)].append(r)

    points = []  # (cluster_id, n_trials, within_entropy, cache_hit_rate, n_steps)
    for tid, traces in by_task.items():
        if len(traces) < args.min_trials_per_cluster:
            continue
        seqs = [tuple(t["tools"]) for t in traces]
        h = path_entropy(seqs)
        hr, n_steps = cluster_cache_hit_rate(traces)
        points.append({
            "task_id": tid, "n_trials": len(traces),
            "within_entropy_bits": round(h, 4),
            "cache_hit_rate": round(hr, 4),
            "n_steps": n_steps,
        })

    if not points:
        print("No multi-trial clusters; cannot measure correlation.")
        return

    xs = [p["within_entropy_bits"] for p in points]
    ys = [p["cache_hit_rate"] for p in points]
    r = pearson_r(xs, ys)
    rho = spearman_rho(xs, ys)

    # Quartile breakdown — averages of cache hit rate by entropy quartile
    sorted_pairs = sorted(zip(xs, ys))
    q = len(sorted_pairs) // 4
    quartiles = []
    for i in range(4):
        lo = i * q
        hi = len(sorted_pairs) if i == 3 else (i + 1) * q
        chunk = sorted_pairs[lo:hi]
        if not chunk:
            continue
        quartiles.append({
            "quartile": i + 1,
            "entropy_range": (round(chunk[0][0], 3), round(chunk[-1][0], 3)),
            "n_clusters": len(chunk),
            "mean_cache_hit_rate": round(statistics.mean(c[1] for c in chunk), 4),
        })

    result = {
        "source": args.source,
        "n_clusters_total": len(points),
        "min_trials_per_cluster": args.min_trials_per_cluster,
        "mean_entropy": round(statistics.mean(xs), 4),
        "mean_cache_hit_rate": round(statistics.mean(ys), 4),
        "pearson_r": round(r, 4),
        "spearman_rho": round(rho, 4),
        "expected_sign": "negative (higher entropy → lower cacheability)",
        "claim_verdict": (
            "STRONG support" if rho < -0.5 else
            "MODERATE support" if rho < -0.3 else
            "WEAK support" if rho < -0.1 else
            "NO support — claim falsified"
        ),
        "entropy_quartiles": quartiles,
    }
    args.out.write_text(json.dumps({**result, "points": points}, indent=2))

    print(f"clusters with ≥{args.min_trials_per_cluster} trials: {len(points)}")
    print(f"mean within-task entropy: {result['mean_entropy']} bits")
    print(f"mean cache hit rate:      {result['mean_cache_hit_rate']*100:.1f}%")
    print(f"Pearson r:                {r:.4f}")
    print(f"Spearman ρ:               {rho:.4f}")
    print(f"verdict:                  {result['claim_verdict']}")
    print()
    print("By entropy quartile (low to high):")
    print(f"  {'Q':>2}  {'entropy range':>16}  {'n':>4}  {'mean cache hit':>14}")
    for q in quartiles:
        print(f"  Q{q['quartile']}  "
              f"[{q['entropy_range'][0]:>5.2f}, {q['entropy_range'][1]:>5.2f}]  "
              f"{q['n_clusters']:>4}  {q['mean_cache_hit_rate']*100:>13.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
