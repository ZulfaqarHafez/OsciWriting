"""Extractor robustness audit.

For each tau-bench sim, count the ground-truth ``tool_calls`` directly
in the raw messages, then count what
``extract_tool_calls_with_args_from_messages`` returns, and compare.

Three things this catches:
    1. Missed tool calls (extracted < ground truth)  → extractor bug.
    2. Phantom tool calls (extracted > ground truth) → extractor bug.
    3. Sims with zero tool calls → expected for refusal/policy-stop
       sims; should not be a large fraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter.entropy import extract_tool_calls_with_args_from_messages  # noqa: E402


def ground_truth_count(sim: dict, requestor: str = "assistant") -> int:
    """Count agent-initiated tool_calls (``requestor == "assistant"``).

    tau-bench's user simulator also emits tool_calls with
    ``requestor == "user"``; the loader filters those out, so the
    ground truth for "agent decisions" must match.
    """
    n = 0
    for m in sim.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            req = tc.get("requestor")
            if req is not None and req != requestor:
                continue
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name:
                n += 1
    return n


def audit(results_dir: Path) -> dict:
    total_sims = 0
    zero_truth = 0
    zero_extracted_with_truth = 0
    mismatches: list[dict] = []
    diff_counter: Counter = Counter()
    sum_truth = 0
    sum_extracted = 0

    for jf in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for sim in d.get("simulations") or []:
            total_sims += 1
            truth = ground_truth_count(sim)
            # Match what the tau-bench loader does: only keep assistant-
            # initiated tool calls (user-simulator calls are scripted /
            # not agent decisions).
            extracted = extract_tool_calls_with_args_from_messages(
                sim.get("messages") or [], requestor_filter="assistant"
            )
            ex_count = len(extracted)
            sum_truth += truth
            sum_extracted += ex_count
            if truth == 0:
                zero_truth += 1
            elif ex_count == 0:
                zero_extracted_with_truth += 1
            if truth != ex_count:
                diff_counter[ex_count - truth] += 1
                if len(mismatches) < 10:
                    mismatches.append({
                        "file": jf.name,
                        "sim_id": sim.get("id"),
                        "ground_truth": truth,
                        "extracted": ex_count,
                        "delta": ex_count - truth,
                    })

    matches = total_sims - sum(diff_counter.values())
    return {
        "total_sims": total_sims,
        "sims_with_zero_truth": zero_truth,
        "sims_truth_nonzero_extracted_zero": zero_extracted_with_truth,
        "sims_exact_match": matches,
        "sims_mismatched": sum(diff_counter.values()),
        "match_rate": round(matches / total_sims, 4) if total_sims else 0.0,
        "sum_ground_truth_calls": sum_truth,
        "sum_extracted_calls": sum_extracted,
        "total_call_delta": sum_extracted - sum_truth,
        "delta_histogram": dict(sorted(diff_counter.items())),
        "sample_mismatches": mismatches,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "extractor_audit.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    r = audit(args.results_dir)
    args.out.write_text(json.dumps(r, indent=2))

    print(f"total sims:                    {r['total_sims']:,}")
    print(f"sims with zero ground-truth:   {r['sims_with_zero_truth']:,}")
    print(f"sims truth>0 but extracted=0:  {r['sims_truth_nonzero_extracted_zero']:,}")
    print(f"sims exact match:              {r['sims_exact_match']:,} "
          f"({r['match_rate']*100:.2f}%)")
    print(f"sims mismatched:               {r['sims_mismatched']:,}")
    print()
    print(f"ground-truth total tool_calls: {r['sum_ground_truth_calls']:,}")
    print(f"extracted total tool_calls:    {r['sum_extracted_calls']:,}")
    print(f"net delta (extracted - truth): {r['total_call_delta']:+,}")
    if r["delta_histogram"]:
        print(f"delta histogram (extracted - truth): {r['delta_histogram']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
