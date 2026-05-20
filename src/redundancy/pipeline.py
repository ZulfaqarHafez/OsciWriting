"""End-to-end orchestration. PRD §8.1, §9, §14.

Three arms (writing subject, unfiltered-random control, scrambled null) run through
identical embed -> UMAP -> HDBSCAN -> metrics -> judge code. The decision uses
subject-minus-control, with H5 gating (PRD §4). `--no-judge` runs the pilot without
API cost; a pilot makes no decision (the H5 gate is undecidable without the judge).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import date

import numpy as np

from . import baseline, filters
from .config import CONFIG, RESULTS, DOCS, run_timestamp
from .cost_model import derive_thresholds


def _build_h5_items(prompt_emb, prompts, responses, bands, per_band, seed):
    """Banded substitutability items: response written for A, judged against B."""
    rng = random.Random(seed)
    n = len(prompt_emb)
    from sklearn.neighbors import NearestNeighbors

    k = min(20, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(prompt_emb)
    dist, idx = nn.kneighbors(prompt_emb)
    buckets: dict[str, list[tuple[int, int]]] = {f"{lo:.2f}-{hi:.2f}": [] for lo, hi in bands}
    for i in range(n):
        for col in range(1, k):
            c = 1.0 - float(dist[i, col])
            j = int(idx[i, col])
            for lo, hi in bands:
                if lo <= c < hi:
                    key = f"{lo:.2f}-{hi:.2f}"
                    if len(buckets[key]) < per_band * 3:
                        buckets[key].append((i, j))
                    break
    items: list[dict] = []
    for key, pairs in buckets.items():
        rng.shuffle(pairs)
        for i, j in pairs[:per_band]:
            items.append(
                {"prompt_b": prompts[j], "response_a": responses[i], "band": key}
            )
    return items


def run(
    n: int, dataset: str, use_judge: bool, seed: int, filter_name: str = "strict"
) -> dict:
    from .data import load_records, prompts as get_prompts, responses as get_responses
    from .dedup import exact_dedup, near_dedup
    from .embed import embed
    from .reduce import reduce as umap_reduce
    from .cluster import (
        cluster,
        cluster_examples,
        sweep,
        top_n_coverage,
        response_indices_by_cluster,
    )
    from . import metrics as M
    from . import report as R

    th = derive_thresholds(CONFIG.cost)
    ct = CONFIG.thresholds  # static thresholds: H4_rho + control gaps (not derived)
    th_full = {
        **asdict(th),
        "H4_rho": ct.H4_rho,
        "control_gap_h1": ct.control_gap_h1,
        "control_gap_h3": ct.control_gap_h3,
        "control_gap_h5": ct.control_gap_h5,
    }
    ts = run_timestamp()
    out_dir = RESULTS / f"run_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(n, dataset=dataset, seed=seed)
    records, exact_stats = exact_dedup(records)  # PRD §8.6 (added v2.1)

    p_all = get_prompts(records)
    strict_idx = filters.apply(p_all, "strict")
    recall_idx = filters.apply(p_all, "recall")

    subject_idx = recall_idx if filter_name == "recall" else strict_idx
    subject = [records[i] for i in subject_idx]
    if len(subject) < 50:
        raise RuntimeError(
            f"only {len(subject)} writing prompts (filter={filter_name}) from "
            f"{len(records)} deduped records; increase --n (PRD §8.2, §8.6)"
        )
    unfiltered = baseline.unfiltered_random(records, len(subject), seed)
    scrambled = baseline.scrambled(subject, seed)

    s_prompts = get_prompts(subject)
    s_resp_all = get_responses(subject)
    u_prompts = get_prompts(unfiltered)
    z_prompts = get_prompts(scrambled)
    z_resp_all = get_responses(scrambled)

    # Near-dup collapse per arm (PRD §8.6): one representative per cosine>=0.98
    # blob, before any metric, so H1/H3/H4 are not driven by copy-paste volume.
    s_pe_full = embed(s_prompts)
    s_keep, s_near = near_dedup(s_pe_full)
    s_prompts = [s_prompts[i] for i in s_keep]
    s_resp = [s_resp_all[i] for i in s_keep]
    s_pe = s_pe_full[s_keep]
    s_re = embed(s_resp)

    u_pe_full = embed(u_prompts)
    u_keep, u_near = near_dedup(u_pe_full)
    u_pe = u_pe_full[u_keep]

    z_pe_full = embed(z_prompts)
    z_keep, z_near = near_dedup(z_pe_full)
    z_prompts = [z_prompts[i] for i in z_keep]
    z_resp = [z_resp_all[i] for i in z_keep]
    z_pe = z_pe_full[z_keep]

    if len(s_pe) < 50:
        raise RuntimeError(
            f"only {len(s_pe)} subject prompts survive dedup (exact+near); "
            "increase --n — the writing subset is mostly verbatim spam (PRD §8.6)"
        )

    # H1 — coverage envelope vs unfiltered control
    s_sweep = sweep(s_pe)
    u_sweep = sweep(u_pe)
    h1_gap = s_sweep.coverage_median - u_sweep.coverage_median
    h1_pass = (
        s_sweep.separable
        and not s_sweep.degenerate
        and s_sweep.coverage_median >= th.T1
        and h1_gap >= CONFIG.thresholds.control_gap_h1
    )

    canon = cluster(umap_reduce(s_pe))
    examples = cluster_examples(canon, s_prompts)

    # H3 — nearest-neighbor similarity vs scrambled null
    s_nn = M.nn_similarity(s_pe)
    z_nn = M.nn_similarity(z_pe)
    s3 = th.S3
    h3_frac = M.fraction_above(s_nn, s3)
    h3_ctrl = M.fraction_above(z_nn, s3)
    h3_gap = h3_frac - h3_ctrl
    h3_pass = h3_frac >= th.T3 and h3_gap >= CONFIG.thresholds.control_gap_h3

    # H4 — full-range stratified correlation
    pairs = M.stratified_pairs(s_pe, CONFIG.h4_pairs, CONFIG.h4_bins, seed)
    h4 = M.h4_correlation(s_pe, s_re, pairs)
    h4_pass = not np.isnan(h4["spearman"]) and h4["spearman"] >= ct.H4_rho

    headline = {
        "run": ts,
        "n": n,
        "dataset": dataset,
        "seed": seed,
        "thresholds": th_full,
        "filters": {
            "records_after_exact_dedup": len(records),
            "strict_n": len(strict_idx),
            "recall_n": len(recall_idx),
            "active": filter_name,
            "note": "rubric uses conservative (recall) numbers if they diverge — PRD §8.2",
        },
        "dedup": {
            "exact": exact_stats.as_dict(),
            "near_subject": s_near.as_dict(),
            "near_unfiltered": u_near.as_dict(),
            "near_scrambled": z_near.as_dict(),
            "note": "WildChat writing subset is heavily verbatim-duplicated — PRD §8.6",
        },
        "H1": {
            "coverage_median": s_sweep.coverage_median,
            "coverage_min": s_sweep.coverage_min,
            "coverage_max": s_sweep.coverage_max,
            "noise_median": s_sweep.noise_median,
            "separable": s_sweep.separable,
            "degenerate": s_sweep.degenerate,
            "control_unfiltered": u_sweep.coverage_median,
            "gap": h1_gap,
            "pass": bool(h1_pass),
        },
        "H3": {
            "S3": s3,
            "fraction_at_S3": h3_frac,
            "control_scrambled": h3_ctrl,
            "gap": h3_gap,
            "percentiles": {
                str(t): M.fraction_above(s_nn, t) for t in (0.7, 0.8, 0.9, 0.95)
            },
            "pass": bool(h3_pass),
        },
        "H4": {
            "spearman": h4["spearman"],
            "pearson": h4["pearson"],
            "n_pairs": h4["n_pairs"],
            "pass": bool(h4_pass),
        },
    }

    if use_judge:
        from .judge import Judge

        judge = Judge(out_dir / "judge_transcripts.jsonl")
        cl_idx = response_indices_by_cluster(canon)
        clusters_resp = {
            cid: [s_resp[i] for i in idxs] for cid, idxs in cl_idx.items()
        }
        h2 = judge.templatedness(clusters_resp)

        s_items = _build_h5_items(
            s_pe, s_prompts, s_resp, CONFIG.h5_bands, CONFIG.h5_pairs_per_band, seed
        )
        for it in s_items:
            it["arm"] = "subject"
        z_items = _build_h5_items(
            z_pe, z_prompts, z_resp, CONFIG.h5_bands, CONFIG.h5_pairs_per_band, seed
        )
        for it in z_items:
            it["arm"] = "scrambled"
        h5_subj = judge.substitutability(s_items)
        h5_ctrl = judge.substitutability(z_items)

        cal = R.calibrate_s3(h5_subj["rate_by_band"], CONFIG.h5_bands, th.T5)
        best_rate = max(h5_subj["rate_by_band"].values(), default=0.0)
        ctrl_floor = max(h5_ctrl["rate_by_band"].values(), default=0.0)
        h5_gap = best_rate - ctrl_floor
        h5_pass = (
            cal is not None
            and best_rate >= th.T5
            and h5_gap >= CONFIG.thresholds.control_gap_h5
        )
        headline["H2"] = {**h2, "pass": bool(h2["pass"])}
        headline["H5"] = {
            "rate_by_band": h5_subj["rate_by_band"],
            "control_floor_by_band": h5_ctrl["rate_by_band"],
            "calibrated_S3": cal,
            "best_rate": best_rate,
            "gap": h5_gap,
            "gating": True,
            "pass": bool(h5_pass),
        }
        headline["decision"] = R.decide(headline)
    else:
        headline["H2"] = {"pass": False, "note": "skipped (pilot, --no-judge)"}
        headline["H5"] = {
            "pass": False,
            "gap": 0.0,
            "best_rate": None,
            "note": "skipped (pilot, --no-judge)",
        }
        headline["decision"] = {
            "outcome": "Pilot — no decision",
            "action": "Run without --no-judge for the H5-gated decision (PRD §9).",
            "h5_gate": "NOT EVALUATED",
        }

    R.write_headline(out_dir, headline)
    R.write_cluster_examples(out_dir, examples)
    R.write_summary(out_dir, headline)
    R.write_figures(
        out_dir,
        {
            "cluster_sizes": [
                int((canon == c).sum()) for c in sorted(set(canon)) if c != -1
            ],
            "nn_subject": s_nn.tolist(),
            "nn_scrambled": z_nn.tolist(),
            "pr_prompt": h4["prompt_cos"],
            "pr_response": h4["response_cos"],
            "h1_subject": s_sweep.coverage_median,
            "h1_control": u_sweep.coverage_median,
            "h3_subject": h3_frac,
            "h3_control": h3_ctrl,
            "h5_subject": headline["H5"].get("best_rate") or 0.0,
            "h5_control": max(
                (headline["H5"].get("control_floor_by_band") or {}).values(),
                default=0.0,
            ),
        },
    )

    (out_dir / "config.json").write_text(
        json.dumps(
            {"config": asdict(CONFIG), "argv": {"n": n, "dataset": dataset, "seed": seed, "judge": use_judge}},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (RESULTS / "latest.txt").write_text(f"run_{ts}", encoding="utf-8")
    _log_decision(headline)
    return headline


def _log_decision(headline: dict) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    log = DOCS / "decision_log.md"
    entry = (
        f"\n## {date.today().isoformat()} — run {headline['run']}\n\n"
        f"- N={headline['n']} dataset={headline['dataset']} "
        f"thresholds_provisional={headline['thresholds'].get('provisional')}\n"
        f"- Outcome: **{headline['decision']['outcome']}** "
        f"(H5 gate: {headline['decision']['h5_gate']})\n"
        f"- Action: {headline['decision']['action']}\n"
    )
    prior = log.read_text(encoding="utf-8") if log.exists() else "# Decision log\n"
    log.write_text(prior + entry, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LLM redundancy study pipeline (PRD §8.1)")
    ap.add_argument("--n", type=int, default=5000, help="raw English records to sample")
    ap.add_argument(
        "--dataset",
        choices=["lmsys", "wildchat"],
        default="lmsys",
        help="primary is LMSYS (PRD §7, v2.1); wildchat retained as contaminated fallback",
    )
    ap.add_argument("--no-judge", action="store_true", help="pilot: skip H2/H5 (no API cost)")
    ap.add_argument(
        "--filter",
        choices=["strict", "recall"],
        default="strict",
        help="writing filter (PRD §8.2). 'recall' is the conservative arm — rubric "
        "uses it when strict/recall yields diverge materially.",
    )
    ap.add_argument("--seed", type=int, default=CONFIG.seed)
    args = ap.parse_args(argv)

    if CONFIG.cost.provisional:
        print(
            "WARNING: thresholds are PROVISIONAL. This is a pilot, not the decision "
            "run. Freeze docs/cost_model.md with real prices first (PRD §4a).\n"
        )
    headline = run(
        args.n,
        args.dataset,
        use_judge=not args.no_judge,
        seed=args.seed,
        filter_name=args.filter,
    )
    print(json.dumps(headline["decision"], indent=2))
    print(f"\nresults/{(RESULTS / 'latest.txt').read_text().strip()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
