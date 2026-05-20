"""Judge audit (PRD §8.5).

A 100-pair human spot-check validates the judge before its numbers are trusted.
This module samples H5 records from a run's ``judge_transcripts.jsonl``, writes a
markdown labeling sheet with the judge's verdict shown but the human verdict
blank, and scores agreement once filled in. PRD threshold: >=80% agreement to
trust the judge; below, the rubric falls back to fully-manual n=200.

Older transcripts (pre v2.1) did not record band/arm — the sampler labels them
"unknown" and still produces a usable audit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from .config import CONFIG, RESULTS, judge_api_key

_SUBST_PROMPT = (
    "A user sent PROMPT_B. Below is a RESPONSE that was originally written for a "
    "different but similar prompt. Judge ONLY whether this RESPONSE would be an "
    "acceptable answer to PROMPT_B as-is. Answer strictly as: "
    "VERDICT: ACCEPTABLE|BORDERLINE|UNACCEPTABLE followed by one short reason.\n\n"
    "PROMPT_B:\n{prompt_b}\n\nRESPONSE:\n{response_a}"
)
INTER_LLM_MODEL = "claude-opus-4-7"  # different from CONFIG.judge_model (Haiku 4.5)

_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Z]+)", re.IGNORECASE)
_PROMPT_RE = re.compile(
    r"PROMPT_B:\s*\n(?P<prompt_b>.*?)\n\s*\nRESPONSE:\s*\n(?P<response_a>.*)$",
    re.DOTALL,
)
_VALID = {"ACCEPTABLE", "BORDERLINE", "UNACCEPTABLE"}


def _verdict(text: str) -> str:
    m = _VERDICT_RE.search(text or "")
    return m.group(1).upper() if m else "UNKNOWN"


def _parse_subst(prompt: str) -> tuple[str, str]:
    m = _PROMPT_RE.search(prompt or "")
    if not m:
        return ("", "")
    return m.group("prompt_b").strip(), m.group("response_a").strip()


def _h5_records(transcripts: Path) -> list[dict]:
    out: list[dict] = []
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") != "h5":
            continue
        out.append(r)
    return out


def sample(run_dir: Path, n: int = 100, seed: int = 42) -> tuple[Path, Path]:
    """Stratify across bands when present, otherwise random. Writes the MD sheet
    plus a JSONL companion (machine-readable original judge calls)."""
    transcripts = run_dir / "judge_transcripts.jsonl"
    records = _h5_records(transcripts)
    if not records:
        raise RuntimeError(f"no h5 records in {transcripts}")

    rng = random.Random(seed)
    # subject-arm only when arm is recorded; otherwise treat all as subject (older runs)
    pool = [r for r in records if r.get("arm", "subject") == "subject"]
    if len(pool) < n:
        picked = list(pool)
    elif any("band" in r for r in pool):
        from collections import defaultdict

        by_band: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_band[r.get("band", "unknown")].append(r)
        per = max(1, n // len(by_band))
        picked = []
        for band, rs in by_band.items():
            rng.shuffle(rs)
            picked.extend(rs[:per])
        rng.shuffle(picked)
        picked = picked[:n]
    else:
        rng.shuffle(pool)
        picked = pool[:n]

    md_lines: list[str] = [
        "# Judge audit sample (PRD §8.5)",
        "",
        f"Sample of {len(picked)} H5 subject-arm pairs from `judge_transcripts.jsonl`. "
        "For each pair, replace `_` after **Your verdict:** with one of "
        "`ACCEPTABLE` / `BORDERLINE` / `UNACCEPTABLE`. Then run "
        "`python -m redundancy.audit score --run <run_dir>` to get the agreement rate. "
        "PRD §8.5: ≥80% agreement validates the judge; below that, fall back to "
        "fully-manual n=200.",
        "",
    ]
    jsonl_rows: list[dict] = []
    for i, r in enumerate(picked, 1):
        prompt_b, response_a = _parse_subst(r.get("prompt", ""))
        judge_v = _verdict(r.get("reply", ""))
        band = r.get("band", "unknown")
        md_lines += [
            f"## Pair {i}",
            f"- **Band:** {band}",
            f"- **Judge verdict:** {judge_v}",
            "",
            "**PROMPT_B:**",
            "",
            "> " + prompt_b.replace("\n", "\n> ")[:2000],
            "",
            "**RESPONSE (originally for prompt A):**",
            "",
            "> " + response_a.replace("\n", "\n> ")[:2000],
            "",
            "**Your verdict:** _",
            "",
            "---",
            "",
        ]
        jsonl_rows.append(
            {
                "id": i,
                "band": band,
                "judge_verdict": judge_v,
                "prompt_b": prompt_b,
                "response_a": response_a,
            }
        )

    md_path = run_dir / "audit_sample.md"
    jsonl_path = run_dir / "audit_sample.jsonl"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in jsonl_rows), encoding="utf-8"
    )
    return md_path, jsonl_path


def _parse_filled(md_path: Path) -> list[tuple[int, str, str]]:
    """Return [(pair_id, judge_verdict, human_verdict)] from a filled-in sheet."""
    text = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n## Pair (\d+)\n", text)
    rows: list[tuple[int, str, str]] = []
    for i in range(1, len(blocks), 2):
        pid = int(blocks[i])
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        jv = re.search(r"\*\*Judge verdict:\*\*\s*([A-Z]+)", body)
        hv = re.search(r"\*\*Your verdict:\*\*\s*([A-Za-z_]+)", body)
        rows.append(
            (
                pid,
                (jv.group(1).upper() if jv else "UNKNOWN"),
                (hv.group(1).upper() if hv else "_"),
            )
        )
    return rows


def score(run_dir: Path) -> dict:
    md_path = run_dir / "audit_sample.md"
    rows = _parse_filled(md_path)
    total = len(rows)
    labeled = [r for r in rows if r[2] in _VALID]
    unlabeled = total - len(labeled)
    auditable = [r for r in labeled if r[1] in _VALID]
    skipped_judge_unknown = len(labeled) - len(auditable)
    if not auditable:
        return {
            "total": total,
            "unlabeled": unlabeled,
            "skipped_judge_unknown": skipped_judge_unknown,
            "agreement": None,
            "verdict": "no auditable pairs — fill in **Your verdict:** for more rows",
        }
    agree = sum(1 for _, jv, hv in auditable if jv == hv)
    rate = agree / len(auditable)
    verdict = (
        "judge validated (≥80% — H5 numbers trusted per PRD §8.5)"
        if rate >= 0.80
        else "judge NOT validated (<80% — PRD §8.5 falls back to fully-manual n=200)"
    )
    return {
        "total": total,
        "labeled": len(labeled),
        "unlabeled": unlabeled,
        "skipped_judge_unknown": skipped_judge_unknown,
        "auditable": len(auditable),
        "agreements": agree,
        "agreement": rate,
        "verdict": verdict,
    }


def inter_llm_label(
    run_dir: Path, model: str = INTER_LLM_MODEL
) -> dict:
    """Re-label the audit sample with a DIFFERENT Claude (Opus vs the Haiku judge).

    This is a sensitivity check, NOT the PRD §8.5 spot-check. Two Claude models
    share lineage; high agreement here only shows Claude is self-consistent on
    the task, not that the judge tracks human acceptability. Low agreement IS a
    real signal — it means the judge prompt is model-sensitive and fragile.
    Findings must label this result accordingly.
    """
    if not judge_api_key():
        raise RuntimeError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY (or JUDGE_API_KEY)."
        )
    from anthropic import Anthropic  # heavy; local

    jsonl_path = run_dir / "audit_sample.jsonl"
    rows = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    client = Anthropic(api_key=judge_api_key())

    out_path = run_dir / "audit_inter_llm.jsonl"
    out_path.write_text("", encoding="utf-8")  # truncate / start fresh
    n_match = 0
    n_judge_unknown = 0
    n_opus_unknown = 0
    confusion: dict[tuple[str, str], int] = {}

    for row in rows:
        prompt = _SUBST_PROMPT.format(
            prompt_b=(row.get("prompt_b") or "")[:1500],
            response_a=(row.get("response_a") or "")[:1500],
        )
        # Opus 4.7 deprecated `temperature`; rely on the model's default.
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = msg.content[0].text if msg.content else ""
        opus_v = _verdict(reply)
        judge_v = row.get("judge_verdict", "UNKNOWN")
        if judge_v == "UNKNOWN":
            n_judge_unknown += 1
        if opus_v == "UNKNOWN":
            n_opus_unknown += 1
        if judge_v in _VALID and opus_v in _VALID:
            confusion[(judge_v, opus_v)] = confusion.get((judge_v, opus_v), 0) + 1
            if judge_v == opus_v:
                n_match += 1
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "band": row.get("band"),
                        "judge_verdict": judge_v,
                        "opus_verdict": opus_v,
                        "opus_reply": reply,
                    }
                )
                + "\n"
            )

    auditable = sum(v for v in confusion.values())
    rate = (n_match / auditable) if auditable else None
    return {
        "model": model,
        "judge_model": CONFIG.judge_model,
        "n_total": len(rows),
        "n_auditable": auditable,
        "n_judge_unknown": n_judge_unknown,
        "n_opus_unknown": n_opus_unknown,
        "agreements": n_match,
        "agreement_rate": rate,
        "confusion_judge_vs_opus": {f"{j}->{o}": c for (j, o), c in confusion.items()},
        "interpretation": (
            "INTER-LLM SENSITIVITY CHECK — NOT PRD §8.5 human audit. Two Claude "
            "models share lineage; this measures self-consistency, not judge "
            "validity. Low agreement = judge is prompt/model-sensitive (real "
            "signal); high agreement is NOT validation."
        ),
    }


_NSFW_RE = re.compile(
    r"\b("
    r"porn(?:o|ographic|ography)?|nsfw|erotic(?:a|ally)?|fetish|kink|"
    r"(?:hand|blow|rim)\s*job|cum(?:ming|shot)?|orgasm|"
    r"masturbat\w+|intercourse|coitus|"
    r"(?:big|tight|wet|hairy|small)?\s*(?:dick|cock|penis|pussy|vagina|clitoris|clit|boobs?|tits?|breasts?|nipples?|ass(?:hole)?|butt)|"
    r"nude|naked|nudity|undress(?:ed|ing)?|"
    r"fuck(?:ing|ed|er)?|shit|bitch|whore|slut|"
    r"sex(?:ual|ually|y)?|"
    r"horny|aroused|moan(?:ing|s)?|"
    r"anal|oral\s*sex|deepthroat|gangbang"
    r")\b",
    re.IGNORECASE,
)


def _is_nsfw(text: str) -> bool:
    return bool(_NSFW_RE.search(text or ""))


def binary_content_rule_label(run_dir: Path) -> dict:
    """Apply a user-supplied binary CONTENT-TYPE rule and compare to judge.

    Rule (user, 2026-05-20): NSFW content -> UNACCEPTABLE; everything else ->
    ACCEPTABLE. Explicitly NOT a substitutability audit (which asks whether the
    response answers PROMPT_B). Documented as such in findings so the agreement
    rate is not misread as PRD §8.5 validation.
    """
    jsonl_path = run_dir / "audit_sample.jsonl"
    rows = [
        json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    out_path = run_dir / "audit_binary_content_rule.jsonl"
    out_path.write_text("", encoding="utf-8")

    n_match = 0
    n_nsfw = 0
    auditable: list[tuple[str, str]] = []  # (judge, user)
    for row in rows:
        prompt_b = row.get("prompt_b") or ""
        response_a = row.get("response_a") or ""
        nsfw = _is_nsfw(prompt_b) or _is_nsfw(response_a)
        user_v = "UNACCEPTABLE" if nsfw else "ACCEPTABLE"
        judge_v = row.get("judge_verdict", "UNKNOWN")
        if nsfw:
            n_nsfw += 1
        if judge_v in _VALID:
            auditable.append((judge_v, user_v))
            if judge_v == user_v:
                n_match += 1
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "judge_verdict": judge_v,
                        "user_verdict_by_rule": user_v,
                        "nsfw_match": nsfw,
                    }
                )
                + "\n"
            )

    confusion: dict[str, int] = {}
    for j, u in auditable:
        confusion[f"{j}->{u}"] = confusion.get(f"{j}->{u}", 0) + 1
    rate = (n_match / len(auditable)) if auditable else None
    return {
        "rule": "NSFW match -> UNACCEPTABLE; else -> ACCEPTABLE (user-supplied 2026-05-20)",
        "warning": (
            "This is NOT the PRD §8.5 substitutability audit. The rule classifies "
            "by content type, not by whether the response answers PROMPT_B. The "
            "agreement rate cannot be interpreted as validating or invalidating "
            "the judge."
        ),
        "n_total": len(rows),
        "n_nsfw_match": n_nsfw,
        "n_auditable": len(auditable),
        "agreements": n_match,
        "agreement_rate": rate,
        "confusion_judge_vs_user": confusion,
        "interpretation_for_findings": (
            ">=80% agreement here would NOT validate the judge per PRD §8.5; "
            "<80% would NOT invalidate it. The rule is orthogonal to the audit's "
            "question."
        ),
    }


def _resolve(run: str | None) -> Path:
    if run:
        candidate = RESULTS / run
        if not candidate.exists():
            candidate = Path(run)
        return candidate
    return RESULTS / (RESULTS / "latest.txt").read_text().strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PRD §8.5 judge audit (sample / score)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="write audit_sample.md for hand-labeling")
    s.add_argument("--run", help="run dir name under results/ (default: latest)")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score", help="compute agreement on a filled-in sheet")
    sc.add_argument("--run", help="run dir name under results/ (default: latest)")
    il = sub.add_parser(
        "inter-llm",
        help="sensitivity check: Opus re-labels the sample; compare to Haiku judge. NOT the PRD §8.5 audit.",
    )
    il.add_argument("--run", help="run dir name under results/ (default: latest)")
    il.add_argument("--model", default=INTER_LLM_MODEL)
    br = sub.add_parser(
        "binary-content-rule",
        help="apply user-supplied content-type rule (NSFW=UNACCEPTABLE/else=ACCEPTABLE) and compare to judge. NOT the PRD §8.5 audit.",
    )
    br.add_argument("--run", help="run dir name under results/ (default: latest)")
    args = ap.parse_args(argv)

    run_dir = _resolve(args.run)
    if args.cmd == "sample":
        md, jl = sample(run_dir, n=args.n, seed=args.seed)
        print(f"wrote {md}")
        print(f"wrote {jl}")
        print("Fill in 'Your verdict:' lines, then: redundancy.audit score --run", run_dir.name)
        return 0
    if args.cmd == "inter-llm":
        out = inter_llm_label(run_dir, model=args.model)
        print(json.dumps(out, indent=2))
        return 0
    if args.cmd == "binary-content-rule":
        out = binary_content_rule_label(run_dir)
        print(json.dumps(out, indent=2))
        return 0
    print(json.dumps(score(run_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
