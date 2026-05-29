"""Synthetic corporate-workflow trace generator (Phase 2 of the PRD).

Models a canonical "daily financial report" agent that calls a fixed set of
tools with mostly-deterministic branching. Used to:

    1. Stand in for Yunjue traces when HuggingFace is unreachable.
    2. Give a controlled corpus where the *true* path distribution is known
       so the entropy estimator can be evaluated against ground truth.

Each run yields a Trace: a tuple of tool names plus the input context that
drove the branching. No external dependencies — pure stdlib so it runs
anywhere.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Trace:
    trace_id: str
    tools: list[str]
    inputs: dict
    outputs: dict = field(default_factory=dict)

    def to_log(self) -> str:
        """Render in a ``tool_call:`` form so ``extract_tool_sequence`` finds it."""
        lines = [f"# trace {self.trace_id}", f"inputs: {json.dumps(self.inputs)}"]
        for t in self.tools:
            lines.append(f"tool_call: {t}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workflow definition
# ---------------------------------------------------------------------------
#
# "Daily Financial Report" agent — canonical low-entropy corporate workflow.
# Mostly the same path every morning; small minority of inputs branch off
# into reconciliation or escalation sub-paths.

_BASE_PATH = [
    "fetch_market_open",
    "fetch_portfolio_positions",
    "compute_pnl",
    "fetch_news_headlines",
    "summarize_news",
    "render_report",
    "email_report",
]


def _build_trace(rng: random.Random, i: int) -> Trace:
    """Generate one synthetic trace.

    Branching distribution (matches PRD assumption of low-entropy corporate
    workflows: ~80% on a single canonical path):

        80%  base path
        10%  base + reconciliation sub-path (data quality flag)
         5%  base + escalation (risk threshold breach)
         3%  base + both reconciliation and escalation
         2%  rare edge-case path (full re-run, manual review)
    """
    r = rng.random()
    tools = list(_BASE_PATH)
    inputs: dict = {"date_offset": i, "portfolio": rng.choice(["A", "B", "C"])}

    if r < 0.80:
        scenario = "base"
    elif r < 0.90:
        scenario = "reconcile"
        # insert reconciliation sub-path after pnl compute
        idx = tools.index("compute_pnl") + 1
        tools[idx:idx] = ["flag_data_quality", "reconcile_positions"]
    elif r < 0.95:
        scenario = "escalate"
        tools.append("escalate_to_risk")
    elif r < 0.98:
        scenario = "reconcile+escalate"
        idx = tools.index("compute_pnl") + 1
        tools[idx:idx] = ["flag_data_quality", "reconcile_positions"]
        tools.append("escalate_to_risk")
    else:
        scenario = "rare_edgecase"
        # full re-run plus manual review — long, weird path
        tools = [
            "fetch_market_open",
            "fetch_portfolio_positions",
            "compute_pnl",
            "flag_data_quality",
            "fetch_alt_data_source",
            "reconcile_positions",
            "compute_pnl",
            "manual_review",
            "fetch_news_headlines",
            "summarize_news",
            "render_report",
            "request_human_approval",
            "email_report",
        ]

    inputs["scenario"] = scenario
    return Trace(trace_id=f"synth-{i:06d}", tools=tools, inputs=inputs)


def generate_corpus(n: int = 500, seed: int = 0) -> list[Trace]:
    """Generate ``n`` synthetic traces with the fixed branching distribution."""
    rng = random.Random(seed)
    return [_build_trace(rng, i) for i in range(n)]


# ---------------------------------------------------------------------------
# Controlled-entropy generator (for regime-cutoff calibration)
# ---------------------------------------------------------------------------
#
# generate_corpus above has a *fixed* branching distribution. To calibrate
# the regime cutoffs we need to dial within-task entropy across the full
# range and observe how cache hit rate / cost savings respond. This
# generator produces ``n_tasks`` tasks, each with ``trials_per_task``
# replays drawn from ``n_variants`` distinct execution paths under a
# Zipfian distribution whose skew is set by ``concentration``:
#
#     concentration -> inf : all trials take variant 0   (within-task H = 0)
#     concentration = 1    : Zipf(1) — moderate skew
#     concentration -> 0   : near-uniform over variants   (H -> log2(n_variants))
#
# Per-call args are made deterministic per (task, step) so PathCache hits
# whenever the same path recurs within a task — i.e. cacheability tracks
# within-task entropy by construction, which is exactly the relationship
# we want to *measure* the strength of, not assume.


def _variant_path(base_len: int, variant: int, divergence_breadth: int = 1) -> list[str]:
    """A deterministic execution path for a given variant index.

    ``divergence_breadth`` controls how many of the ``base_len`` steps a
    non-zero variant replaces with variant-specific tools. breadth=1
    diverges at one step (most steps stay shared → high cacheability
    floor); breadth=base_len makes every step variant-specific (no shared
    structure → cacheability can collapse to ~0, enabling a true
    FULL_AGENT regime in the calibration sweep).
    """
    tools = [f"step_{j}" for j in range(base_len)]
    if variant > 0:
        breadth = max(1, min(divergence_breadth, base_len))
        # Spread the diverging positions evenly through the path.
        for k in range(breadth):
            pos = (variant + k * (base_len // breadth + 1)) % base_len
            tools[pos] = f"branch_{variant}_{pos}"
    return tools


def generate_controlled_corpus(
    n_tasks: int = 50,
    trials_per_task: int = 16,
    n_variants: int = 8,
    concentration: float = 1.0,
    base_len: int = 7,
    divergence_breadth: int = 1,
    seed: int = 0,
) -> list[dict]:
    """Generate a corpus with tunable within-task entropy.

    Returns rows in the same shape the data-source loaders emit
    (``id``, ``tools``, ``tool_args``, ``args``, ``task_id``, ``reward``,
    ``raw``) so the calibration harness can feed them straight through
    the router.

    ``concentration`` controls the Zipf skew over variant paths: large =>
    low within-task entropy, small => high. Each task gets an independent
    variant ordering so tasks differ from one another.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    for t in range(n_tasks):
        # Per-task weights over variants: Zipfian raised to ``concentration``.
        order = list(range(n_variants))
        rng.shuffle(order)
        weights = [1.0 / ((rank + 1) ** concentration) for rank in range(n_variants)]
        for trial in range(trials_per_task):
            variant = rng.choices(order, weights=weights, k=1)[0]
            tools = _variant_path(base_len, variant, divergence_breadth)
            # Deterministic per-call args keyed on (task, step position):
            # identical whenever the same path recurs in the same task.
            tool_args = [{"task": t, "pos": j} for j in range(len(tools))]
            rows.append({
                "id": f"ctrl-{t:04d}-{trial:03d}",
                "tools": tools,
                "tool_args": tool_args,
                "args": {"task": t},
                "task_id": f"ctrl-task-{t:04d}",
                "reward": 1.0,
                "raw": {"task_id": f"ctrl-task-{t:04d}", "variant": variant},
            })
    return rows


# ---------------------------------------------------------------------------
# Tool stubs (used by the middleware when actually "executing" a synthetic run)
# ---------------------------------------------------------------------------


def make_tool_registry() -> dict[str, Callable[[dict], dict]]:
    """Return ``{tool_name: callable}`` for the synthetic financial agent.

    Each tool is deterministic given its inputs — that's what makes the
    PathCache hit rate meaningful: identical inputs ⇒ identical outputs.
    """

    def _det(name: str) -> Callable[[dict], dict]:
        def fn(ctx: dict) -> dict:
            # Output is a deterministic function of the input context — so the
            # cache key (state hash) is meaningful and cache hits are correct.
            seed = hash((name, json.dumps(ctx, sort_keys=True))) & 0xFFFF_FFFF
            return {"tool": name, "value": seed}

        fn.__name__ = name
        return fn

    names = set(_BASE_PATH) | {
        "flag_data_quality",
        "reconcile_positions",
        "escalate_to_risk",
        "fetch_alt_data_source",
        "manual_review",
        "request_human_approval",
    }
    return {n: _det(n) for n in names}
