"""Prefetch feasibility pipeline (PRD v3 §8.1).

Reuses load_conversations + the v2.2 §15 multi-turn pair extraction. Adds:
- per-pair serve-fire detection (find the best cross-conversation neighbor for
  each pair under the MiniLM retriever),
- HP2 judge calls (acceptability of cached response from neighbor for the
  actual next prompt),
- Pareto sweep over (K, T_pred, T_serve, c_speculate) via cost_model_v3,
- §4 rubric → outcome.

Embedding is recomputed from the cached multi-turn parquet — no dataset
re-stream needed once `load_conversations` has run once.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import RESULTS
from .cost_model_v3 import V3Constants, pareto_sweep, best_cell


HP1_LIFT_REQ = 0.10  # PRD v3 HP1 minimum lift over random
HP2_QUALITY_REQ = 0.60  # PRD v3 HP2 minimum acceptability at the chosen T_serve


@dataclass(frozen=True)
class PrefetchPair:
    conv_hash: str
    turn_idx: int
    anchor: str
    target: str
    target_response: str  # asst reply to `target` — the cached response if this turn is the predicted next


def _extract_pairs(convs) -> list[PrefetchPair]:
    out: list[PrefetchPair] = []
    for c in convs:
        for i in range(c.n_pairs - 1):
            out.append(
                PrefetchPair(
                    conv_hash=c.hash,
                    turn_idx=i,
                    anchor=c.user_turns[i],
                    target=c.user_turns[i + 1],
                    target_response=c.asst_turns[i + 1],
                )
            )
    return out


def _best_cross_conv_neighbor(
    neighbor_idx: np.ndarray,
    target_emb: np.ndarray,
    conv_hashes: list[str],
    k_max: int = 25,
) -> list[tuple[int, float]]:
    """For each pair i, the (best_j, cosine) over the top-k_max cross-conv neighbors."""
    n = len(conv_hashes)
    out: list[tuple[int, float]] = []
    for i in range(n):
        this_conv = conv_hashes[i]
        keep: list[int] = []
        for j in neighbor_idx[i]:
            j = int(j)
            if j == i or conv_hashes[j] == this_conv:
                continue
            keep.append(j)
            if len(keep) >= k_max:
                break
        if not keep:
            out.append((-1, 0.0))
            continue
        sims = target_emb[keep] @ target_emb[i]
        bi = int(np.argmax(sims))
        out.append((keep[bi], float(sims[bi])))
    return out


def _stratified_serve_sample(
    best_neighbors: list[tuple[int, float]],
    bands: tuple[tuple[float, float], ...] = (
        (0.90, 0.95),
        (0.95, 0.98),
        (0.98, 1.01),
    ),
    per_band: int = 100,
    seed: int = 42,
) -> list[int]:
    """Pair indices stratified across serve-cosine bands for HP2 judging."""
    rng = random.Random(seed)
    by_band: dict[tuple[float, float], list[int]] = {b: [] for b in bands}
    for i, (j, c) in enumerate(best_neighbors):
        if j < 0:
            continue
        for lo, hi in bands:
            if lo <= c < hi:
                by_band[(lo, hi)].append(i)
                break
    picks: list[int] = []
    for b, idxs in by_band.items():
        rng.shuffle(idxs)
        picks.extend(idxs[:per_band])
    return picks


def _quality_table(
    judgments: list[dict],
    t_serve_grid: tuple[float, ...],
) -> dict[float, float]:
    """For each T_serve in grid, acceptability rate over pairs whose serve cosine ≥ T_serve."""
    table: dict[float, float] = {}
    for t in t_serve_grid:
        eligible = [j for j in judgments if j["serve_cosine"] >= t]
        if not eligible:
            table[t] = 0.0
            continue
        s = sum(
            {"ACCEPTABLE": 1.0, "BORDERLINE": 0.5}.get(j["verdict"], 0.0)
            for j in eligible
        )
        table[t] = s / len(eligible)
    return table


def _hp1_lift(hit_minilm: dict[str, float], hit_random: dict[str, float]) -> dict:
    """Best (K, T_pred) lift cell across the existing multi_turn hit table format."""
    best = {"key": None, "minilm": 0.0, "random": 0.0, "lift": -1.0}
    for k, v in hit_minilm.items():
        r = hit_random.get(k, 0.0)
        lift = v - r
        if lift > best["lift"]:
            best = {"key": k, "minilm": v, "random": r, "lift": lift}
    return best


def _parse_hit_key(key: str) -> tuple[int, float]:
    # "K=5@T=0.8" -> (5, 0.8)
    K_part, T_part = key.split("@")
    return int(K_part.split("=")[1]), float(T_part.split("=")[1])


def decide(hp1_pass: bool, hp2_pass: bool, hp3_pass: bool, hp4_pass: bool) -> dict:
    """PRD v3 §4 matrix."""
    if not hp1_pass:
        return {
            "outcome": "Abandon",
            "action": "Cross-user prediction is fake; no re-scope.",
        }
    if not hp2_pass:
        return {
            "outcome": "Abandon",
            "action": "No T_serve clears the safety bar; prefetch would serve bad answers.",
        }
    if not hp4_pass:
        if hp3_pass:
            return {
                "outcome": "Re-scope to within-session only",
                "action": "Cross-user prefetch doesn't pay back; within-session predictability is real (HP3). Build single-user prefetcher.",
            }
        return {
            "outcome": "Re-scope to narrower domain",
            "action": "HP1+HP2 hold but no Pareto cell is positive; try constrained-domain dataset.",
        }
    return {
        "outcome": "Commit to prefetch project",
        "action": "Design architecture; pick best Pareto cell.",
    }


def run(
    n: int,
    dataset: str = "lmsys",
    seed: int = 42,
    judge_enabled: bool = True,
    constants: V3Constants = V3Constants(),
) -> dict:
    from .data import load_conversations
    from .embed import embed
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .multi_turn import _nn_indices, _random_indices, hit_table, K_LIST, T_LIST

    convs = load_conversations(n, dataset=dataset, seed=seed)
    pairs = _extract_pairs(convs)
    if not pairs:
        raise RuntimeError("no multi-turn pairs after extraction")

    anchors = [p.anchor for p in pairs]
    targets = [p.target for p in pairs]
    conv_hashes = [p.conv_hash for p in pairs]
    n_pairs = len(pairs)

    print(f"convs={len(convs)} pairs={n_pairs}; embedding anchors+targets...")
    anchor_emb = embed(anchors)
    target_emb = embed(targets)

    k_buf = max(K_LIST) + 20
    print("MiniLM kNN...")
    minilm_idx = _nn_indices(anchor_emb, k_buf)
    print("TF-IDF kNN...")
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    anchor_tfidf = vec.fit_transform(anchors)
    tfidf_idx = _nn_indices(anchor_tfidf, k_buf)
    random_idx = _random_indices(n_pairs, k_buf, seed)

    print("HP1 hit tables...")
    hit_minilm = hit_table(minilm_idx, conv_hashes, target_emb)
    hit_tfidf = hit_table(tfidf_idx, conv_hashes, target_emb)
    hit_random = hit_table(random_idx, conv_hashes, target_emb)

    # HP1 verdict: best lift across cells must clear 0.10
    best_lift_minilm = _hp1_lift(hit_minilm, hit_random)
    best_lift_tfidf = _hp1_lift(hit_tfidf, hit_random)
    hp1_pass = max(best_lift_minilm["lift"], best_lift_tfidf["lift"]) >= HP1_LIFT_REQ

    # HP3: same-conv consecutive cosine
    same_cos = (anchor_emb * target_emb).sum(axis=1)
    hp3_p50 = float(np.median(same_cos))
    hp3_frac_ge_07 = float((same_cos >= 0.7).mean())
    hp3_pass = hp3_p50 >= 0.60 and hp3_frac_ge_07 >= 0.30

    # HP2 sample: for MiniLM (the primary retriever for serve-decisions),
    # find best cross-conv neighbor per pair → stratified sample at serve cosine.
    print("HP2 best-neighbor scan...")
    best_neighbors = _best_cross_conv_neighbor(minilm_idx, target_emb, conv_hashes)
    judgments: list[dict] = []
    if judge_enabled:
        from .judge import Judge

        ts0 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = RESULTS / f"prefetch_{ts0}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sample_idx = _stratified_serve_sample(best_neighbors, seed=seed)
        print(f"HP2 judging {len(sample_idx)} pairs across serve-cosine bands...")
        judge = Judge(run_dir / "judge_transcripts.jsonl")
        from .judge import _SUBST_PROMPT  # type: ignore
        for i in sample_idx:
            j, c = best_neighbors[i]
            prompt_b = pairs[i].target
            response_a = pairs[j].target_response
            txt = judge._ask(
                "hp2",
                _SUBST_PROMPT.format(
                    prompt_b=prompt_b[:1500], response_a=response_a[:1500]
                ),
                serve_cosine=c,
                pair_i=i,
                neighbor_j=j,
            )
            judgments.append(
                {
                    "pair_i": i,
                    "neighbor_j": j,
                    "serve_cosine": c,
                    "verdict": Judge._verdict(txt),
                }
            )
        quality_table = _quality_table(judgments, constants.t_serve_grid)
        # HP2 passes if SOME t_serve achieves ≥ 60%
        hp2_pass = any(v >= HP2_QUALITY_REQ for v in quality_table.values())
    else:
        ts0 = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = RESULTS / f"prefetch_{ts0}"
        run_dir.mkdir(parents=True, exist_ok=True)
        quality_table = {t: 0.0 for t in constants.t_serve_grid}
        hp2_pass = False  # not evaluable without judge

    # HP4 Pareto
    hit_for_sweep: dict[tuple[int, float], float] = {}
    for k_str, v in hit_minilm.items():
        K, T = _parse_hit_key(k_str)
        if K in constants.K_grid and T in constants.t_pred_grid:
            hit_for_sweep[(K, T)] = v
    cells = pareto_sweep(hit_for_sweep, quality_table, constants)
    best = best_cell(cells)
    hp4_pass = best is not None

    verdict = decide(hp1_pass, hp2_pass, hp3_pass, hp4_pass)

    headline = {
        "kind": "prefetch_feasibility_v3",
        "run": run_dir.name,
        "dataset": dataset,
        "n_conversations": len(convs),
        "n_pairs": n_pairs,
        "judge_enabled": judge_enabled,
        "HP1": {
            "best_minilm": best_lift_minilm,
            "best_tfidf": best_lift_tfidf,
            "lift_required": HP1_LIFT_REQ,
            "pass": hp1_pass,
        },
        "HP2": {
            "quality_at_t_serve": quality_table,
            "required": HP2_QUALITY_REQ,
            "n_judged": len(judgments),
            "pass": hp2_pass,
        },
        "HP3": {
            "same_conv_p50": hp3_p50,
            "same_conv_frac_ge_0.7": hp3_frac_ge_07,
            "pass": hp3_pass,
        },
        "HP4": {
            "best_cell": asdict(best) if best else None,
            "n_feasible_cells": sum(1 for c in cells if c.feasible),
            "n_total_cells": len(cells),
            "pass": hp4_pass,
        },
        "decision": verdict,
        "constants": asdict(constants),
    }

    (run_dir / "headline_numbers.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    # Pareto CSV
    csv_lines = ["K,t_pred,t_serve,c_speculate,hit_rate,quality,effective,roi,feasible"]
    for c in cells:
        csv_lines.append(
            f"{c.K},{c.t_pred},{c.t_serve},{c.c_speculate},"
            f"{c.hit_rate:.4f},{c.quality:.4f},{c.effective:.4f},{c.roi:.4f},{c.feasible}"
        )
    (run_dir / "pareto.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    _write_summary(run_dir, headline)
    (RESULTS / "latest.txt").write_text(run_dir.name, encoding="utf-8")
    return headline


def _write_summary(run_dir: Path, h: dict) -> None:
    d = h["decision"]
    lines = [
        "# Prefetch feasibility — auto-summary (PRD v3)",
        "",
        f"**Decision: {d['outcome']}**",
        "",
        f"- Action: {d['action']}",
        "",
        "## Hypotheses",
        "",
        f"- **HP1 (predictability above chance):** {h['HP1']['pass']} | "
        f"best MiniLM lift = {h['HP1']['best_minilm']['lift']:.4f} at "
        f"{h['HP1']['best_minilm']['key']}; best TF-IDF lift = "
        f"{h['HP1']['best_tfidf']['lift']:.4f} at {h['HP1']['best_tfidf']['key']}",
        f"- **HP2 (serve-quality safety):** {h['HP2']['pass']} | "
        f"max quality across T_serve = {max(h['HP2']['quality_at_t_serve'].values(), default=0):.3f}",
        f"- **HP3 (within-session predictability):** {h['HP3']['pass']} | "
        f"p50 = {h['HP3']['same_conv_p50']:.3f}, frac ≥0.7 = {h['HP3']['same_conv_frac_ge_0.7']:.3f}",
        f"- **HP4 (budget ROI):** {h['HP4']['pass']} | "
        f"{h['HP4']['n_feasible_cells']} / {h['HP4']['n_total_cells']} cells positive-ROI",
        "",
    ]
    if h["HP4"]["best_cell"]:
        bc = h["HP4"]["best_cell"]
        lines += [
            "## Best Pareto cell",
            "",
            f"- K = {bc['K']}, T_pred = {bc['t_pred']}, T_serve = {bc['t_serve']}, "
            f"c_speculate = {bc['c_speculate']}",
            f"- hit = {bc['hit_rate']:.4f} × quality = {bc['quality']:.4f} = "
            f"effective {bc['effective']:.4f}; roi = {bc['roi']:.4f}",
            "",
        ]
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prefetch feasibility pipeline (PRD v3)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dataset", choices=["lmsys", "wildchat"], default="lmsys")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-judge", action="store_true", help="skip HP2 (pilot)")
    args = ap.parse_args(argv)
    h = run(args.n, args.dataset, args.seed, judge_enabled=not args.no_judge)
    print(json.dumps(h["decision"], indent=2))
    print(f"\nresults/{(RESULTS / 'latest.txt').read_text().strip()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
