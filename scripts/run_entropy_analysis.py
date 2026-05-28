"""Phase 1 + Phase 4 driver: entropy analysis and AgentPathRouter evaluation.

Loads a named trace corpus (Yunjue, Nemotron-Agentic, two Hermes sets,
τ-bench, or a synthetic fallback), then:

    1. Computes Shannon entropy + top-N coverage on the path distribution
       (PRD §5.1 empirical claim).
    2. Trains the n-gram entropy estimator on a train split.
    3. Runs the AgentPathRouter middleware on a held-out test split and
       reports cache hit rate, speculation precision, and full-call rate
       (PRD §5.3 primary metrics — proxies for token/cost reduction).

Writes machine-readable results to ``results/agentic_execution_entropy/``.

Sources:
    auto             try yunjue, fall back to synthetic
    yunjue           HF: YunjueTech/Yunjue-Agent-Traces (finsearchcomp split)
    nemotron_agentic HF: nvidia/Nemotron-Agentic-v1
    hermes_reasoning HF: lambda/hermes-agent-reasoning-traces
    hermes_filtered  HF: DJLougen/hermes-agent-traces-filtered
    tau_bench        local dir of tau2-bench simulation JSON files (--tau-bench-dir)
    synthetic        in-process synthetic financial-report corpus (PRD §6.3 stand-in)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import (  # noqa: E402
    AgentPathRouter,
    CostModel,
    NgramEntropyEstimator,
    RunMetrics,
    classify,
    coverage_curve,
)
from agentpathrouter.data_sources import DatasetUnavailable, SOURCES, load  # noqa: E402
from agentpathrouter.entropy import coverage_at_k  # noqa: E402
from agentpathrouter.synthetic import generate_corpus, make_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _synthetic_rows(n: int, seed: int) -> list[dict]:
    """Synthetic corpus shaped like the data_sources loaders' output."""
    return [
        {"id": t.trace_id, "tools": t.tools, "args": t.inputs, "raw": t}
        for t in generate_corpus(n=n, seed=seed)
    ]


def load_corpus(source: str, args: argparse.Namespace) -> tuple[str, list[dict]]:
    """Return ``(source_label_used, rows)``. Falls back to synthetic if requested."""
    if source == "synthetic":
        return "synthetic", _synthetic_rows(args.n_synthetic, args.seed)

    if source == "tau_bench":
        if not args.tau_bench_dir:
            raise SystemExit("--source tau_bench requires --tau-bench-dir <path>")
        rows = load("tau_bench", dir_path=args.tau_bench_dir)
        return f"tau_bench:{args.tau_bench_dir}", rows

    if source == "trail":
        if not args.trail_dir:
            raise SystemExit("--source trail requires --trail-dir <path>")
        rows = load("trail", dir_path=args.trail_dir, subset=args.trail_subset)
        return f"trail:{args.trail_subset}", rows

    if source == "auto":
        # PRD primary source first; fall back silently to synthetic.
        try:
            rows = load("yunjue")
            return "yunjue:finsearchcomp", rows
        except DatasetUnavailable as e:
            print(f"[auto] yunjue unavailable ({e}); falling back to synthetic",
                  file=sys.stderr)
            return "synthetic", _synthetic_rows(args.n_synthetic, args.seed)

    # Named HF source
    rows = load(source)
    return source, rows


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase1_entropy(sequences: list[tuple[str, ...]], outdir: Path) -> dict:
    """Compute + persist entropy / coverage stats."""
    stats = coverage_curve(sequences, top_n=10)
    ks = [1, 3, 5, 10, 20, 50, 100]
    cov_k = coverage_at_k(sequences, ks)

    summary = {
        "unique_paths": stats.unique_paths,
        "total_traces": stats.total_traces,
        "path_entropy_bits": round(stats.entropy_bits, 4),
        "top10_coverage": round(stats.top_n_coverage, 4),
        "coverage_at_k": {str(k): round(v, 4) for k, v in cov_k.items()},
        "top10_paths": [
            {"path": list(path), "count": count, "share": round(count / stats.total_traces, 4)}
            for path, count in stats.most_common
        ],
    }
    (outdir / "phase1_entropy.json").write_text(json.dumps(summary, indent=2))
    return summary


ARMS = {
    # PRD §9 Phase 4 ablation: three configurations of the router.
    "baseline":               {"use_speculation": False, "use_small_model_routing": False, "no_cache": True},
    "cache_only":             {"use_speculation": False, "use_small_model_routing": False},
    "cache+spec":             {"use_speculation": True,  "use_small_model_routing": False},
    "cache+spec+routing":     {"use_speculation": True,  "use_small_model_routing": True},
}


def _run_arm(
    arm_name: str,
    cfg: dict,
    est: NgramEntropyEstimator,
    test_traces: list[tuple[tuple[str, ...], dict]],
    tools: dict,
    threshold: float,
    small_model_threshold: float,
    cost_model: CostModel,
) -> dict:
    """Run one ablation arm and return its aggregate metrics + cost block."""
    # The baseline arm models "full frontier inference, no AEE": disable
    # the cache by passing a one-shot cache that never returns hits.
    from agentpathrouter.path_cache import PathCache as _Cache

    class _NullCache(_Cache):
        def get(self, *args, **kwargs):  # type: ignore[override]
            self.stats.misses += 1
            return False, None
        def put(self, *args, **kwargs):  # type: ignore[override]
            return None

    cache = _NullCache() if cfg.get("no_cache") else _Cache()
    router = AgentPathRouter(
        tools=tools,
        cache=cache,
        estimator=est,
        confidence_threshold=threshold,
        small_model_threshold=small_model_threshold,
        use_speculation=cfg.get("use_speculation", True),
        use_small_model_routing=cfg.get("use_small_model_routing", False),
    )

    agg = RunMetrics()
    per_run = []
    t0 = time.perf_counter()
    for seq, args in test_traces:
        if any(t not in tools for t in seq):
            continue
        _, m = router.run_trace(list(seq), args)
        agg += m
        per_run.append(m)
    elapsed = time.perf_counter() - t0

    spec_stats = router.prefetcher.stats if router.prefetcher else None
    summary = {
        "arm": arm_name,
        "elapsed_sec": round(elapsed, 4),
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.as_dict().items()},
        "speculation_fires": spec_stats.fires if spec_stats else 0,
        "speculation_precision": round(spec_stats.precision, 4) if spec_stats else 0,
        "cache_size": len(router.cache),
        "cost": cost_model.per_1000_runs(per_run),
    }
    if router.prefetcher:
        router.prefetcher.close()
    return summary


def phase4_router_eval(
    train_seqs: list[tuple[str, ...]],
    test_traces: list[tuple[tuple[str, ...], dict]],
    outdir: Path,
    threshold: float,
    small_model_threshold: float = 0.85,
    tools: dict | None = None,
) -> dict:
    """Train n-gram once, run the §9 Phase-4 ablation across all arms.

    Returns a dict keyed by arm name, plus a ``cost_reduction_vs_baseline``
    block summarising the headline PRD §5.3 numbers.
    """
    est = NgramEntropyEstimator(n=3).fit(train_seqs)
    if tools is None:
        tools = _make_universal_registry(train_seqs, test_traces)
    cost_model = CostModel()

    results: dict[str, dict] = {}
    for arm_name, cfg in ARMS.items():
        results[arm_name] = _run_arm(
            arm_name, cfg, est, test_traces, tools,
            threshold, small_model_threshold, cost_model,
        )

    # Cost reduction vs the baseline arm.
    base = results["baseline"]["cost"]["usd_per_1000_runs"]
    for arm_name, summary in results.items():
        usd = summary["cost"]["usd_per_1000_runs"]
        summary["cost"]["pct_saved_vs_baseline"] = (
            round(1 - usd / base, 4) if base else 0.0
        )

    (outdir / "phase4_router_eval.json").write_text(json.dumps(results, indent=2))
    return results


def _make_universal_registry(train_seqs, test_traces) -> dict:
    """Build a tool registry covering every tool name observed in the corpus.

    For real corpora (Nemotron / Hermes / τ-bench) we don't have callable
    implementations of every tool, so stub each one as a deterministic
    function of its inputs — that's enough to measure cache/spec rates.
    """
    from agentpathrouter.synthetic import make_tool_registry as _base
    reg = _base()
    seen = set()
    for s in train_seqs:
        seen.update(s)
    for seq, _ in test_traces:
        seen.update(seq)

    def _stub(name: str):
        def fn(ctx: dict) -> dict:
            seed = hash((name, json.dumps(ctx, sort_keys=True, default=str))) & 0xFFFF_FFFF
            return {"tool": name, "value": seed}
        fn.__name__ = name
        return fn

    for name in seen:
        reg.setdefault(name, _stub(name))
    return reg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AEE entropy + router eval.")
    ap.add_argument(
        "--source",
        choices=["auto", "synthetic", *sorted(SOURCES)],
        default="auto",
    )
    ap.add_argument("--tau-bench-dir", type=str, default=None,
                    help="Directory of tau2-bench simulation JSON files (for --source tau_bench).")
    ap.add_argument("--trail-dir", type=str, default=None,
                    help="Path to a cloned trail-benchmark repo (for --source trail).")
    ap.add_argument("--trail-subset", type=str, default="all",
                    choices=["all", "gaia", "swe_bench"])
    ap.add_argument("--n-synthetic", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Confidence threshold for speculative prefetch.")
    ap.add_argument("--small-model-threshold", type=float, default=0.85,
                    help="Higher confidence bar for small-model routing (a wrong "
                         "route is a quality regression, so the bar is stricter).")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "results" / "agentic_execution_entropy",
    )
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    try:
        source_used, rows = load_corpus(args.source, args)
    except DatasetUnavailable as e:
        print(f"[error] could not load --source {args.source}: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- normalise to (sequences, args_list) ----
    sequences: list[tuple[str, ...]] = []
    args_list: list[dict] = []
    for r in rows:
        tools = tuple(r["tools"])
        if not tools:
            continue
        sequences.append(tools)
        args_list.append(r.get("args") or {})

    if not sequences:
        print("No tool sequences extracted; check the loader for this source.",
              file=sys.stderr)
        sys.exit(2)

    # ---- phase 1 ----
    print(f"[source] {source_used}  traces={len(sequences)}")
    p1 = phase1_entropy(sequences, args.outdir)
    print(f"[phase1] unique_paths={p1['unique_paths']}  entropy={p1['path_entropy_bits']} bits  "
          f"top10_coverage={p1['top10_coverage']*100:.1f}%")

    # ---- regime classification (the headline contribution) ----
    regime = classify(sequences)
    print(f"[regime] {regime.regime.value.upper()}  "
          f"(entropy_ratio={regime.entropy_ratio:.2f}, top3={regime.top3_coverage*100:.1f}%)")
    print(f"  rationale: {regime.rationale}")
    print(f"  recommend: {regime.recommendation}")
    (args.outdir / "regime.json").write_text(json.dumps(regime.as_dict(), indent=2))

    # ---- train/test split ----
    n_train = max(1, int(len(sequences) * args.train_frac))
    train = sequences[:n_train]
    test = list(zip(sequences[n_train:], args_list[n_train:]))
    if not test:
        test = list(zip(sequences, args_list))

    # ---- phase 4 (full §9 ablation) ----
    p4 = phase4_router_eval(
        train, test, args.outdir,
        threshold=args.threshold,
        small_model_threshold=args.small_model_threshold,
    )
    print("[phase4 ablation]")
    print(f"  {'arm':<22} {'cache':>6} {'spec':>6} {'route':>6} {'qreg':>6} "
          f"{'USD/1k':>10} {'saved':>7}")
    for name in ("baseline", "cache_only", "cache+spec", "cache+spec+routing"):
        a = p4[name]
        c = a["cost"]
        print(
            f"  {name:<22} "
            f"{a['cache_hit_rate']*100:>5.1f}% "
            f"{a['speculation_hit_rate']*100:>5.1f}% "
            f"{a['small_model_route_rate']*100:>5.1f}% "
            f"{a['quality_regression_rate']*100:>5.2f}% "
            f"{c['usd_per_1000_runs']:>10.4f} "
            f"{c['pct_saved_vs_baseline']*100:>6.1f}%"
        )

    # ---- combined ----
    combined = {
        "source": source_used,
        "regime": regime.as_dict(),
        "phase1": p1,
        "phase4": p4,
    }
    out_name = f"summary_{source_used.replace(':', '_').replace('/', '_')}.json"
    (args.outdir / out_name).write_text(json.dumps(combined, indent=2))
    (args.outdir / "summary.json").write_text(json.dumps(combined, indent=2))
    print(f"[ok] wrote results to {args.outdir / out_name}")


if __name__ == "__main__":
    main()
