"""Personal Prompt Prefetch — HP1/HP2/HP3/HP4 evaluation (PRD v4 §8).

Operates on ONE user's chronological multi-session history. For each
conversation, looks at its first prompt and asks: does any PRIOR
conversation's first prompt sit at cosine ≥ T in MiniLM embedding space?
That gives HP1 (recurring fraction). HP3 restricts to the user's first 5
conversations. HP2 (quality) and HP4 (Pareto ROI) optionally fire after
HP1.

Loader is Claude.ai export; design generalizes to any chronological
Conversation list.
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

HP1_RECUR_T = 0.85
HP1_REQ = 0.30
HP2_REQ = 0.60
HP3_REQ = 0.15
COLD_START_N = 5


@dataclass(frozen=True)
class MatchRecord:
    i: int  # current conversation index (chronological)
    j: int  # prior conversation index with max cosine (or -1 if none)
    cosine: float


def find_prior_matches(prompt_emb: np.ndarray) -> list[MatchRecord]:
    """For each i ≥ 1, max cosine to any prior j ∈ [0..i-1]. i=0 → no prior."""
    out: list[MatchRecord] = [MatchRecord(0, -1, 0.0)]
    for i in range(1, len(prompt_emb)):
        sims = prompt_emb[:i] @ prompt_emb[i]
        j = int(np.argmax(sims))
        out.append(MatchRecord(i, j, float(sims[j])))
    return out


def hp1_recurring_fraction(matches: list[MatchRecord], threshold: float = HP1_RECUR_T) -> dict:
    eligible = [m for m in matches if m.j != -1]
    n_eligible = len(eligible)
    if n_eligible == 0:
        return {"frac": 0.0, "n_eligible": 0, "n_recurring": 0, "threshold": threshold, "pass": False}
    n_recurring = sum(1 for m in eligible if m.cosine >= threshold)
    frac = n_recurring / n_eligible
    return {
        "frac": frac,
        "n_eligible": n_eligible,
        "n_recurring": n_recurring,
        "threshold": threshold,
        "pass": frac >= HP1_REQ,
    }


def hp3_cold_start(matches: list[MatchRecord], threshold: float = HP1_RECUR_T) -> dict:
    """Hit@1 at threshold over the user's first COLD_START_N conversations."""
    cold = [m for m in matches if 1 <= m.i < COLD_START_N + 1]
    if not cold:
        return {"frac": 0.0, "n": 0, "pass": False}
    hits = sum(1 for m in cold if m.cosine >= threshold)
    return {"frac": hits / len(cold), "n": len(cold), "pass": (hits / len(cold)) >= HP3_REQ}


def cosine_distribution(matches: list[MatchRecord]) -> dict:
    eligible = [m.cosine for m in matches if m.j != -1]
    if not eligible:
        return {"n": 0}
    arr = np.array(eligible)
    return {
        "n": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "frac_ge_0.7": float((arr >= 0.7).mean()),
        "frac_ge_0.85": float((arr >= 0.85).mean()),
        "frac_ge_0.95": float((arr >= 0.95).mean()),
    }


def _hp2_sample(
    matches: list[MatchRecord],
    bands: tuple[tuple[float, float], ...] = ((0.85, 0.90), (0.90, 0.95), (0.95, 1.01)),
    per_band: int = 40,
    seed: int = 42,
) -> list[int]:
    """Indices of `matches` to judge for HP2, stratified across serve bands."""
    rng = random.Random(seed)
    buckets: dict[tuple[float, float], list[int]] = {b: [] for b in bands}
    for m in matches:
        if m.j == -1:
            continue
        for lo, hi in bands:
            if lo <= m.cosine < hi:
                buckets[(lo, hi)].append(m.i)
                break
    picks: list[int] = []
    for b, idxs in buckets.items():
        rng.shuffle(idxs)
        picks.extend(idxs[:per_band])
    return picks


def _quality_table(judgments: list[dict], t_serve_grid: tuple[float, ...]) -> dict:
    out: dict[float, float] = {}
    for t in t_serve_grid:
        eligible = [j for j in judgments if j["serve_cosine"] >= t]
        if not eligible:
            out[t] = 0.0
            continue
        score = sum(
            {"ACCEPTABLE": 1.0, "BORDERLINE": 0.5}.get(j["verdict"], 0.0)
            for j in eligible
        )
        out[t] = score / len(eligible)
    return out


def decide(hp1_pass: bool, hp2_pass: bool, hp3_pass: bool, hp4_pass: bool) -> dict:
    if not hp1_pass:
        return {"outcome": "Abandon", "action": "Recurring-task fraction too low; this user's behavior is too exploratory for ambient prefetch."}
    if not hp2_pass:
        return {"outcome": "Abandon", "action": "Even within-user, response substitution fails; prefetch would serve unacceptable answers."}
    if not hp4_pass:
        if hp3_pass:
            return {"outcome": "Commit power-user-only", "action": "HP1+HP2+HP3 hold but ROI infeasible — restrict scope to high-frequency users where K can be small."}
        return {"outcome": "Re-scope to KV-warm-only", "action": "Predict + serve quality both hold; only ROI fails. Use prompt caching / KV prewarming instead of full response prefetch."}
    if not hp3_pass:
        return {"outcome": "Commit, power-user-only", "action": "v4 product ships with cold-start disabled (no recurring history → no prefetch)."}
    return {"outcome": "Commit to project", "action": "Personal prefetch is feasible end-to-end. Design architecture."}


def run(
    zip_path: str | Path,
    user_tag: str = "personal",
    judge_enabled: bool = True,
    constants: V3Constants = V3Constants(),
) -> dict:
    from .user_history import load_claude_export, first_prompts, first_responses
    from .embed import embed

    convs = load_claude_export(zip_path)
    if len(convs) < 30:
        raise RuntimeError(
            f"only {len(convs)} non-empty conversations; HP1 measurement requires ≥30 "
            "(PRD v4 §7). Run as descriptive only."
        )
    prompts = first_prompts(convs)
    responses = first_responses(convs)
    print(f"convs={len(convs)}; embedding conversation-start prompts...")
    prompt_emb = embed(prompts)

    matches = find_prior_matches(prompt_emb)
    hp1 = hp1_recurring_fraction(matches)
    hp3 = hp3_cold_start(matches)
    dist = cosine_distribution(matches)

    judgments: list[dict] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS / f"personal_{user_tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if judge_enabled and hp1["pass"]:
        from .judge import Judge, _SUBST_PROMPT  # type: ignore

        sample_idx = _hp2_sample(matches)
        print(f"HP2 judging {len(sample_idx)} pairs (HP1 passed, gate fired)...")
        judge = Judge(run_dir / "judge_transcripts.jsonl")
        for i in sample_idx:
            m = matches[i]
            prompt_b = prompts[m.i]
            response_a = responses[m.j]
            txt = judge._ask(
                "hp2_personal",
                _SUBST_PROMPT.format(
                    prompt_b=prompt_b[:1500], response_a=response_a[:1500]
                ),
                serve_cosine=m.cosine,
                i=m.i,
                j=m.j,
            )
            judgments.append(
                {
                    "i": m.i,
                    "j": m.j,
                    "serve_cosine": m.cosine,
                    "verdict": Judge._verdict(txt),
                }
            )
        quality_table = _quality_table(judgments, constants.t_serve_grid)
        hp2_pass = any(v >= HP2_REQ for v in quality_table.values())
    else:
        quality_table = {t: 0.0 for t in constants.t_serve_grid}
        hp2_pass = False

    # HP4 — build a per-user hit table at (K=1, T_pred) from the
    # find_prior_matches structure: top-1 match per conversation. For
    # K>1 we'd need top-K which we don't compute here; treat K=1 as the
    # honest measurement and report.
    hit_at_t1 = {(1, t): sum(1 for m in matches if m.j != -1 and m.cosine >= t) / max(1, len([m for m in matches if m.j != -1]))
                 for t in constants.t_pred_grid}
    # Inject just K=1 cells; the rest of the grid would need more retrieval work
    cells = pareto_sweep(hit_at_t1, quality_table, constants)
    best = best_cell(cells)
    hp4_pass = best is not None

    verdict = decide(hp1["pass"], hp2_pass, hp3["pass"], hp4_pass)

    headline = {
        "kind": "personal_prefetch_v4",
        "run": run_dir.name,
        "user_tag": user_tag,
        "n_conversations": len(convs),
        "judge_enabled": judge_enabled,
        "HP1": {
            **hp1,
            "required": HP1_REQ,
        },
        "HP2": {
            "quality_at_t_serve": quality_table,
            "required": HP2_REQ,
            "n_judged": len(judgments),
            "pass": hp2_pass,
        },
        "HP3": {
            **hp3,
            "required": HP3_REQ,
        },
        "HP4": {
            "best_cell": asdict(best) if best else None,
            "n_feasible": sum(1 for c in cells if c.feasible),
            "n_total": len(cells),
            "pass": hp4_pass,
            "note": "Only K=1 hit rates are populated; larger K would need top-K retrieval (not implemented yet).",
        },
        "cosine_distribution": dist,
        "decision": verdict,
        "constants": asdict(constants),
    }

    (run_dir / "headline_numbers.json").write_text(
        json.dumps(headline, indent=2), encoding="utf-8"
    )
    _write_examples(run_dir, matches, convs)
    _write_summary(run_dir, headline)
    (RESULTS / "latest.txt").write_text(run_dir.name, encoding="utf-8")
    return headline


def _write_examples(run_dir: Path, matches: list[MatchRecord], convs) -> None:
    """Top recurring pairs (highest cosines), for sanity inspection."""
    recurring = sorted(
        [m for m in matches if m.j != -1 and m.cosine >= HP1_RECUR_T],
        key=lambda m: -m.cosine,
    )[:20]
    lines = ["# Top recurring prompt pairs (PRD v4)", ""]
    for m in recurring:
        lines += [
            f"## Match: cos = {m.cosine:.3f}",
            "",
            f"**New (i={m.i}):** {convs[m.i].user_turns[0][:300].strip()}",
            "",
            f"**Prior (j={m.j}):** {convs[m.j].user_turns[0][:300].strip()}",
            "",
            "---",
            "",
        ]
    (run_dir / "recurring_clusters.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary(run_dir: Path, h: dict) -> None:
    d = h["decision"]
    lines = [
        "# Personal Prompt Prefetch — auto-summary (PRD v4)",
        "",
        f"**User tag:** {h['user_tag']} | conversations: {h['n_conversations']}",
        "",
        f"**Decision: {d['outcome']}**",
        "",
        f"- Action: {d['action']}",
        "",
        "## Hypotheses",
        "",
        f"- **HP1 (recurring fraction):** {h['HP1']['pass']} | "
        f"{h['HP1']['frac']:.3f} of {h['HP1']['n_eligible']} eligible conversations "
        f"have a prior at cosine ≥ {h['HP1']['threshold']} (required ≥ {h['HP1']['required']})",
        f"- **HP2 (serve-quality):** {h['HP2']['pass']} | "
        f"max quality across T_serve = {max(h['HP2']['quality_at_t_serve'].values(), default=0):.3f} (req ≥ {h['HP2']['required']})",
        f"- **HP3 (cold-start):** {h['HP3']['pass']} | "
        f"{h['HP3']['frac']:.3f} hit rate over first {COLD_START_N} convs (req ≥ {h['HP3']['required']})",
        f"- **HP4 (ROI):** {h['HP4']['pass']} | "
        f"{h['HP4']['n_feasible']} / {h['HP4']['n_total']} cells feasible",
        "",
        "## Cosine distribution (current prompt ↔ best prior prompt)",
        "",
    ]
    for k, v in h["cosine_distribution"].items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.3f}")
        else:
            lines.append(f"- {k}: {v}")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Personal Prompt Prefetch (PRD v4)")
    ap.add_argument("--zip", required=True, help="path to Claude export zip")
    ap.add_argument("--tag", default="personal")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args(argv)
    h = run(args.zip, user_tag=args.tag, judge_enabled=not args.no_judge)
    print(json.dumps(h["decision"], indent=2))
    print(f"\nresults/{(RESULTS / 'latest.txt').read_text().strip()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
