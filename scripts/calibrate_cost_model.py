"""Calibrate the AEE CostModel against tau-bench's actual ``agent_cost``.

tau-bench records the real USD spent per simulation in each row's
``agent_cost`` field, plus the LLM identifier in
``info.agent_info.llm``. We extract per-sim cost and per-sim step
count (one step ≈ one assistant turn that emits a decision), pool
across all sims for each model, and compare against what the AEE
``CostModel`` predicts at its default tokens-per-step.

This answers the P1 question: *is 800 tokens/step a reasonable
assumption?* If the implied tokens/step from the data is wildly
different, the headline cost numbers in findings.md need rescaling.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agentpathrouter import CostModel, ModelPrice  # noqa: E402


# Published May 2026 list prices, USD per MTok. Approximate; sources:
# Anthropic, OpenAI, public price pages. Used to convert "USD per step
# under model X" into "implied tokens per step", which we can then
# compare to the AEE default of 800.
PROVIDER_PRICES: dict[str, ModelPrice] = {
    # OpenAI
    "gpt-4.1-2025-04-14":            ModelPrice(2.00,  8.00),
    "gpt-4.1-mini-2025-04-14":       ModelPrice(0.40,  1.60),
    "gpt-4o-2024-08-06":             ModelPrice(2.50, 10.00),
    "gpt-4o-mini-2024-07-18":        ModelPrice(0.15,  0.60),
    "o4-mini-2025-04-16":            ModelPrice(1.10,  4.40),
    # Anthropic
    "claude-3-7-sonnet-20250219":    ModelPrice(3.00, 15.00),
    "claude-opus-4-7":               ModelPrice(15.0, 75.00),
    "claude-haiku-4-5-20251001":     ModelPrice(1.00,  5.00),
    # Fallback
    "_default":                      ModelPrice(2.00,  8.00),
}


def step_count(sim: dict) -> int:
    """Count assistant turns that look like a model decision.

    Either an assistant message with tool_calls, or an assistant message
    with non-empty content. Plain user/system/tool turns don't count.
    """
    n = 0
    for m in sim.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if m.get("tool_calls"):
            n += 1
        elif (m.get("content") or "").strip():
            n += 1
    return n


def tool_call_count(sim: dict) -> int:
    n = 0
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            if tc:
                n += 1
    return n


def calibrate(results_dir: Path) -> dict:
    by_model_steps: dict[str, list[float]] = defaultdict(list)   # USD/step samples
    by_model_calls: dict[str, list[float]] = defaultdict(list)   # USD/tool-call samples
    by_model_n: dict[str, int] = defaultdict(int)
    total_sims = 0
    total_cost = 0.0

    for jf in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(jf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        model = ((d.get("info") or {}).get("agent_info") or {}).get("llm")
        if not model:
            continue
        for sim in d.get("simulations") or []:
            cost = sim.get("agent_cost")
            if cost is None or cost <= 0:
                continue
            s = step_count(sim)
            c = tool_call_count(sim)
            total_sims += 1
            total_cost += cost
            by_model_n[model] += 1
            if s > 0:
                by_model_steps[model].append(cost / s)
            if c > 0:
                by_model_calls[model].append(cost / c)

    # Convert empirical USD/step under each model into implied tokens/step,
    # using the model's published price + the AEE 70/30 input/output split.
    INPUT_FRAC = 0.7
    rows = []
    for model, samples in sorted(by_model_steps.items(), key=lambda kv: -len(kv[1])):
        if not samples:
            continue
        usd_per_step = statistics.median(samples)
        price = PROVIDER_PRICES.get(model, PROVIDER_PRICES["_default"])
        blended_per_mtok = INPUT_FRAC * price.input_per_mtok + (1 - INPUT_FRAC) * price.output_per_mtok
        implied_tokens = (usd_per_step / blended_per_mtok) * 1_000_000 if blended_per_mtok else 0
        rows.append({
            "model": model,
            "n_sims": by_model_n[model],
            "median_usd_per_step": round(usd_per_step, 6),
            "median_usd_per_tool_call": round(
                statistics.median(by_model_calls[model]) if by_model_calls[model] else 0, 6
            ),
            "blended_price_per_mtok": round(blended_per_mtok, 4),
            "implied_tokens_per_step": round(implied_tokens, 1),
        })

    # AEE default for comparison
    aee = CostModel()  # uses Opus-4.7 frontier price, 800 tokens/step
    return {
        "total_sims": total_sims,
        "total_usd_observed": round(total_cost, 4),
        "aee_default_tokens_per_step": aee.tokens_per_step,
        "aee_default_input_frac": aee.input_frac,
        "aee_predicted_usd_per_frontier_step": round(aee.step_cost("frontier"), 6),
        "per_model": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "results" / "agentic_execution_entropy" / "cost_calibration.json")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = calibrate(args.results_dir)
    args.out.write_text(json.dumps(result, indent=2))

    # Pretty print
    print(f"sims with non-zero cost: {result['total_sims']}")
    print(f"total observed USD:      {result['total_usd_observed']}")
    print(f"AEE default tokens/step: {result['aee_default_tokens_per_step']}")
    print()
    print(f"{'model':<35} {'n':>5} {'USD/step':>10} {'USD/call':>10} {'$/MTok':>8} {'impl.tok/step':>14}")
    for r in result["per_model"]:
        print(f"{r['model']:<35} {r['n_sims']:>5} {r['median_usd_per_step']:>10.6f} "
              f"{r['median_usd_per_tool_call']:>10.6f} {r['blended_price_per_mtok']:>8.2f} "
              f"{r['implied_tokens_per_step']:>14.0f}")
    impl = [r["implied_tokens_per_step"] for r in result["per_model"]]
    if impl:
        print(f"\nimplied tokens/step across models: median={statistics.median(impl):.0f} "
              f"min={min(impl):.0f} max={max(impl):.0f}")
        print(f"AEE assumption (800):  "
              f"{'OK' if 400 <= statistics.median(impl) <= 1600 else 'OUT OF RANGE'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
