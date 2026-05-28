"""Phase 1 + Phase 4 driver: entropy analysis and AgentPathRouter evaluation.

Loads either the Yunjue finsearchcomp split (if HuggingFace is reachable) or
a synthetic corporate-workflow corpus, then:

    1. Computes Shannon entropy + top-N coverage on the path distribution
       (PRD §5.1 empirical claim).
    2. Trains the n-gram entropy estimator on a train split.
    3. Runs the AgentPathRouter middleware on a held-out test split and
       reports cache hit rate, speculation precision, and full-call rate
       (PRD §5.3 primary metrics — proxies for token/cost reduction).

Writes machine-readable results to ``results/agentic_execution_entropy/``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import (  # noqa: E402
    AgentPathRouter,
    NgramEntropyEstimator,
    coverage_curve,
    extract_tool_sequence,
)
from agentpathrouter.entropy import coverage_at_k  # noqa: E402
from agentpathrouter.synthetic import generate_corpus, make_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_yunjue(subset: str = "finsearchcomp", split: str = "train") -> list[dict]:
    """Load + base64-decode Yunjue traces. Returns ``[]`` if HF unreachable."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[yunjue] datasets package not installed", file=sys.stderr)
        return []

    try:
        ds = load_dataset("YunjueTech/Yunjue-Agent-Traces", subset, split=split)
    except Exception as e:  # noqa: BLE001
        print(f"[yunjue] load failed ({e!r}) — falling back to synthetic", file=sys.stderr)
        return []

    out = []
    for row in ds:
        try:
            q = base64.b64decode(row["question"]).decode("utf-8", errors="replace")
            log = base64.b64decode(row["log"]).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        out.append({"id": row.get("id"), "question": q, "log": log})
    return out


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
) -> dict:
    """Train n-gram, run router on test set, return aggregate metrics.

    ``test_traces`` is a list of ``(tool_sequence, shared_args)`` pairs.
    """
    est = NgramEntropyEstimator(n=3).fit(train_seqs)
    tools = make_tool_registry()
    router = AgentPathRouter(
        tools=tools, estimator=est, confidence_threshold=threshold
    )

    t0 = time.perf_counter()
    agg = {"steps": 0, "cache_hits": 0, "spec_hits": 0, "full_calls": 0}
    for seq, args in test_traces:
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Run AEE entropy + router eval.")
    ap.add_argument(
        "--source",
        choices=["auto", "yunjue", "synthetic"],
        default="auto",
        help="auto = try yunjue, fall back to synthetic",
    )
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
    rows: list[dict] = []
    source_used = "synthetic"
    if args.source in ("auto", "yunjue"):
        rows = load_yunjue()
        if rows:
            source_used = "yunjue:finsearchcomp"

    if not rows:
        if args.source == "yunjue":
            print("Yunjue requested but unavailable.", file=sys.stderr)
            sys.exit(1)
        synth = generate_corpus(n=args.n_synthetic, seed=args.seed)
        rows = [
            {
                "id": t.trace_id,
                "question": json.dumps(t.inputs),
                "log": t.to_log(),
                "_args": t.inputs,
            }
            for t in synth
        ]

    # ---- extract tool sequences ----
    sequences: list[tuple[str, ...]] = []
    test_traces: list[tuple[tuple[str, ...], dict]] = []
    args_list: list[dict] = []
    for r in rows:
        seq = tuple(extract_tool_sequence(r["log"]))
        if not seq:
            continue
        sequences.append(seq)
        args_list.append(r.get("_args", {"q": r.get("question", "")[:32]}))

    if not sequences:
        print("No tool sequences extracted; check regex in entropy.extract_tool_sequence",
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
        # very small corpus — fall back to in-sample eval (synthetic only)
        test = list(zip(sequences, args_list))

    # ---- phase 4 ----
    p4 = phase4_router_eval(train, test, args.outdir, args.threshold)
    print(f"[phase4] cache_hit={p4['cache_hit_rate']*100:.1f}%  "
          f"spec_hit={p4['speculation_hit_rate']*100:.1f}%  "
          f"full_call={p4['full_call_rate']*100:.1f}%  "
          f"spec_precision={p4['speculation_precision']*100:.1f}%")

    # ---- combined ----
    combined = {"source": source_used, "phase1": p1, "phase4": p4}
    (args.outdir / "summary.json").write_text(json.dumps(combined, indent=2))
    print(f"[ok] wrote results to {args.outdir}")


if __name__ == "__main__":
    main()
