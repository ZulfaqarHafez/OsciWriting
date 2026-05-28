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
    NgramEntropyEstimator,
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


def phase4_router_eval(
    train_seqs: list[tuple[str, ...]],
    test_traces: list[tuple[tuple[str, ...], dict]],
    outdir: Path,
    threshold: float,
    tools: dict | None = None,
) -> dict:
    """Train n-gram, run router on test set, return aggregate metrics."""
    est = NgramEntropyEstimator(n=3).fit(train_seqs)
    if tools is None:
        tools = _make_universal_registry(train_seqs, test_traces)
    router = AgentPathRouter(
        tools=tools, estimator=est, confidence_threshold=threshold
    )

    t0 = time.perf_counter()
    agg = {"steps": 0, "cache_hits": 0, "spec_hits": 0, "full_calls": 0}
    for seq, args in test_traces:
        # If the test sequence uses tools the registry doesn't know about
        # (real corpora often have a long-tail vocabulary), skip cleanly.
        if any(t not in tools for t in seq):
            continue
        _, m = router.run_trace(list(seq), args)
        agg["steps"] += m.steps
        agg["cache_hits"] += m.cache_hits
        agg["spec_hits"] += m.spec_hits
        agg["full_calls"] += m.full_calls
    elapsed = time.perf_counter() - t0

    spec_stats = router.prefetcher.stats if router.prefetcher else None
    summary = {
        "n_test_traces": len(test_traces),
        "elapsed_sec": round(elapsed, 4),
        "confidence_threshold": threshold,
        "steps": agg["steps"],
        "cache_hit_rate": round(agg["cache_hits"] / agg["steps"], 4) if agg["steps"] else 0,
        "speculation_hit_rate": round(agg["spec_hits"] / agg["steps"], 4) if agg["steps"] else 0,
        "full_call_rate": round(agg["full_calls"] / agg["steps"], 4) if agg["steps"] else 0,
        "speculation_fires": spec_stats.fires if spec_stats else 0,
        "speculation_precision": round(spec_stats.precision, 4) if spec_stats else 0,
        "cache_size": len(router.cache),
    }
    if router.prefetcher:
        router.prefetcher.close()
    (outdir / "phase4_router_eval.json").write_text(json.dumps(summary, indent=2))
    return summary


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
    ap.add_argument("--n-synthetic", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--threshold", type=float, default=0.7)
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

    # ---- train/test split ----
    n_train = max(1, int(len(sequences) * args.train_frac))
    train = sequences[:n_train]
    test = list(zip(sequences[n_train:], args_list[n_train:]))
    if not test:
        test = list(zip(sequences, args_list))

    # ---- phase 4 ----
    p4 = phase4_router_eval(train, test, args.outdir, args.threshold)
    print(f"[phase4] cache_hit={p4['cache_hit_rate']*100:.1f}%  "
          f"spec_hit={p4['speculation_hit_rate']*100:.1f}%  "
          f"full_call={p4['full_call_rate']*100:.1f}%  "
          f"spec_precision={p4['speculation_precision']*100:.1f}%")

    # ---- combined ----
    combined = {"source": source_used, "phase1": p1, "phase4": p4}
    out_name = f"summary_{source_used.replace(':', '_').replace('/', '_')}.json"
    (args.outdir / out_name).write_text(json.dumps(combined, indent=2))
    (args.outdir / "summary.json").write_text(json.dumps(combined, indent=2))
    print(f"[ok] wrote results to {args.outdir / out_name}")


if __name__ == "__main__":
    main()
