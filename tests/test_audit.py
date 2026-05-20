"""Audit module (PRD §8.5)."""

import json

from redundancy.audit import _parse_subst, _parse_filled, _verdict, sample, score


def test_verdict_parse():
    assert _verdict("VERDICT: ACCEPTABLE\nreason...") == "ACCEPTABLE"
    assert _verdict("verdict: borderline") == "BORDERLINE"
    assert _verdict("no verdict here") == "UNKNOWN"


def test_subst_prompt_parse():
    p = (
        "preamble...\n\nPROMPT_B:\nwrite an email about leave\n\nRESPONSE:\n"
        "Dear team, I will be out next week.\n"
    )
    pb, ra = _parse_subst(p)
    assert pb == "write an email about leave"
    assert ra.startswith("Dear team")


def _make_run(tmp_path, kind_rows):
    run = tmp_path / "run_test"
    run.mkdir()
    (run / "judge_transcripts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in kind_rows), encoding="utf-8"
    )
    return run


def test_sample_writes_files_and_skips_non_h5(tmp_path):
    rows = [
        {"kind": "h2", "prompt": "...", "reply": "VERDICT: YES"},
        *[
            {
                "kind": "h5",
                "prompt": f"PROMPT_B:\nprompt {i}\n\nRESPONSE:\nresp {i}",
                "reply": "VERDICT: ACCEPTABLE",
                "band": "0.90-0.95",
                "arm": "subject",
            }
            for i in range(5)
        ],
    ]
    run = _make_run(tmp_path, rows)
    md, jl = sample(run, n=3, seed=1)
    assert md.exists() and jl.exists()
    lines = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert all(d["judge_verdict"] == "ACCEPTABLE" for d in lines)


def test_score_perfect_agreement(tmp_path):
    rows = [
        {
            "kind": "h5",
            "prompt": f"PROMPT_B:\np{i}\n\nRESPONSE:\nr{i}",
            "reply": "VERDICT: ACCEPTABLE",
            "band": "0.90-0.95",
            "arm": "subject",
        }
        for i in range(4)
    ]
    run = _make_run(tmp_path, rows)
    sample(run, n=4, seed=1)
    md = (run / "audit_sample.md").read_text(encoding="utf-8")
    md = md.replace("**Your verdict:** _", "**Your verdict:** ACCEPTABLE")
    (run / "audit_sample.md").write_text(md, encoding="utf-8")
    out = score(run)
    assert out["auditable"] == 4
    assert out["agreement"] == 1.0
    assert "validated" in out["verdict"]


def test_score_below_threshold(tmp_path):
    rows = [
        {
            "kind": "h5",
            "prompt": f"PROMPT_B:\np{i}\n\nRESPONSE:\nr{i}",
            "reply": "VERDICT: ACCEPTABLE",
            "band": "0.90-0.95",
            "arm": "subject",
        }
        for i in range(5)
    ]
    run = _make_run(tmp_path, rows)
    sample(run, n=5, seed=1)
    md = (run / "audit_sample.md").read_text(encoding="utf-8")
    # 2/5 agree -> 40%
    md = md.replace("**Your verdict:** _", "**Your verdict:** ACCEPTABLE", 2)
    md = md.replace("**Your verdict:** _", "**Your verdict:** UNACCEPTABLE")
    (run / "audit_sample.md").write_text(md, encoding="utf-8")
    out = score(run)
    assert out["auditable"] == 5
    assert out["agreement"] == 0.4
    assert "NOT validated" in out["verdict"]
