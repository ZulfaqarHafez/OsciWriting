"""Cache-determinism audit: does PathCache's (tool, history, args) → output
assumption hold on real data?

PathCache hits return a previously-observed output for a state hash.
That's only sound if the same (tool, prior-tool-history, args) triple
always produces the same output. On tau-bench we can check this
directly: pair each ``tool_call`` with its corresponding ``role==tool``
response message (matched by ``tool_call_id``), then group by state
hash and look for triples with multiple distinct outputs.

Non-deterministic triples are silently corrupting any cache hit that
lands on them — quality regressions that the step-level and even the
task-level metrics may not catch, because we never re-execute the
tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def state_hash(tool: str, history: tuple[str, ...], args: dict) -> str:
    payload = json.dumps(
        {"tool": tool, "history": list(history), "args": args},
        sort_keys=True, default=str,
    )
    return sha256(payload.encode()).hexdigest()


def output_hash(content) -> str:
    """Normalised hash for a tool output (string-or-JSON)."""
    if content is None:
        return "_null"
    if isinstance(content, str):
        # Some tools return JSON-as-string; canonicalise when possible.
        try:
            obj = json.loads(content)
            return sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
        except (json.JSONDecodeError, ValueError):
            return sha256(content.encode()).hexdigest()
    return sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def walk_sim(sim: dict) -> list[tuple[str, tuple[str, ...], dict, str]]:
    """Return ``[(tool, history_prefix, args, output_hash), ...]`` for one sim."""
    pairs: list[tuple[str, tuple[str, ...], dict, str]] = []
    # First pass: collect tool_calls in order with their ids and args.
    call_seq: list[tuple[str, str, dict]] = []  # (id, name, args)
    outputs: dict[str, str] = {}  # tool_call_id → output_hash
    for m in sim.get("messages") or []:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                tid = tc.get("id") or ""
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args = {"_raw": args}
                if name and tid:
                    call_seq.append((tid, name, args or {}))
        elif m.get("role") == "tool":
            tid = m.get("id") or m.get("tool_call_id") or ""
            if tid:
                outputs[tid] = output_hash(m.get("content"))

    history: list[str] = []
    for tid, name, args in call_seq:
        out = outputs.get(tid)
        if out is None:
            history.append(name)
            continue
        pairs.append((name, tuple(history), args, out))
        history.append(name)
    return pairs


def audit(results_dir: Path) -> dict:
    by_state: dict[str, set[str]] = defaultdict(set)
    by_state_counts: dict[str, int] = defaultdict(int)
    by_state_tool: dict[str, str] = {}
    total_observations = 0

    for jf in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for sim in d.get("simulations") or []:
            for tool, hist, args, out in walk_sim(sim):
                h = state_hash(tool, hist, args)
                by_state[h].add(out)
                by_state_counts[h] += 1
                by_state_tool[h] = tool
                total_observations += 1

    # How many distinct state hashes? How many were observed more than once?
    n_states = len(by_state)
    repeated_states = {h: c for h, c in by_state_counts.items() if c >= 2}
    n_repeated = len(repeated_states)

    # Non-deterministic = state hash with >1 distinct output across observations
    nondet_states = {h for h, outs in by_state.items() if len(outs) > 1}
    n_nondet = len(nondet_states)
    n_nondet_repeated = sum(1 for h in repeated_states if h in nondet_states)

    # Quality impact: of the cache HITS that would occur (n_observations - n_distinct_states),
    # how many would land on a non-deterministic state and might return a stale output?
    hits_total = sum(c - 1 for c in by_state_counts.values() if c >= 2)
    nondet_hits = sum(
        by_state_counts[h] - 1 for h in nondet_states if by_state_counts[h] >= 2
    )

    # Per-tool breakdown of non-determinism (only meaningfully repeated ones)
    per_tool_nondet: dict[str, dict] = defaultdict(lambda: {"states": 0, "nondet": 0})
    for h, tool in by_state_tool.items():
        if by_state_counts[h] >= 2:
            per_tool_nondet[tool]["states"] += 1
            if h in nondet_states:
                per_tool_nondet[tool]["nondet"] += 1
    worst_tools = sorted(
        [(t, v["nondet"], v["states"], v["nondet"] / v["states"])
         for t, v in per_tool_nondet.items() if v["states"] >= 5],
        key=lambda r: -r[3],
    )[:10]

    return {
        "total_observations": total_observations,
        "distinct_state_hashes": n_states,
        "repeated_state_hashes": n_repeated,
        "nondeterministic_state_hashes": n_nondet,
        "nondeterministic_share_of_repeated": round(n_nondet_repeated / n_repeated, 4) if n_repeated else 0.0,
        "cache_hits_possible": hits_total,
        "cache_hits_on_nondet_states": nondet_hits,
        "stale_hit_rate": round(nondet_hits / hits_total, 4) if hits_total else 0.0,
        "worst_tools_by_nondet_share": [
            {"tool": t, "nondet": nd, "total_repeated_states": ts, "share": round(s, 4)}
            for t, nd, ts, s in worst_tools
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "cache_determinism.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    r = audit(args.results_dir)
    args.out.write_text(json.dumps(r, indent=2))

    print(f"total (tool_call, output) observations: {r['total_observations']:,}")
    print(f"distinct (tool, history, args) state hashes: {r['distinct_state_hashes']:,}")
    print(f"repeated state hashes (≥2 observations): {r['repeated_state_hashes']:,}")
    print(f"non-deterministic state hashes (multiple outputs): {r['nondeterministic_state_hashes']:,}")
    print(f"  share of repeated hashes that are non-deterministic: "
          f"{r['nondeterministic_share_of_repeated']*100:.2f}%")
    print()
    print(f"Total possible cache hits (cumulative re-observations): {r['cache_hits_possible']:,}")
    print(f"  of those, hits on non-deterministic states: {r['cache_hits_on_nondet_states']:,}")
    print(f"  STALE HIT RATE: {r['stale_hit_rate']*100:.2f}%")
    print()
    if r["worst_tools_by_nondet_share"]:
        print("Worst offenders (tools with most non-determinism, ≥5 repeated states):")
        print(f"  {'tool':<35} {'nondet':>8} {'states':>8} {'share':>8}")
        for row in r["worst_tools_by_nondet_share"]:
            print(f"  {row['tool']:<35} {row['nondet']:>8} {row['total_repeated_states']:>8} "
                  f"{row['share']*100:>7.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
