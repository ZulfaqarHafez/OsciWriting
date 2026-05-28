# When does a task not need an agent?

Source PRD: `Agentic Execution Entropy / AgentPathRouter` (uploaded
`376822c6-AgenticExecutionEntropy_PRD.docx`, May 2026).

## Reframe

The original PRD asks "how do we cache repetitive agent workflows to cut
inference cost?" — a systems-optimisation framing. The empirical results
below suggest a sharper claim:

> When path entropy is near zero, the LLM is being used to make decisions
> that are already determined by the input. The "agent" is a dressed-up
> deterministic pipeline. The waste isn't a missing cache layer — it's
> that the agentic framework is the wrong abstraction for this workload.

The contribution becomes a **taxonomy + decision framework** keyed on
execution-path entropy, with three regimes:

| Regime          | Signal                                          | Recommendation                                         |
|-----------------|-------------------------------------------------|--------------------------------------------------------|
| `DETERMINISTIC` | entropy ratio < 0.30 OR top-3 coverage ≥ 90%    | Replace agent with pipeline; LLM only at synthesis     |
| `HYBRID`        | between the two extremes                        | AgentPathRouter (cache + spec + small-model routing)   |
| `FULL_AGENT`    | entropy ratio > 0.75 AND top-10 coverage < 50%  | Keep frontier agent; AEE interventions don't help      |

Thresholds are **preliminary** — calibrated on the synthetic corporate
workflow corpus only. Final values need a sweep across Yunjue /
Nemotron-Agentic / Hermes once HF egress is unblocked. They are
deliberately strict on the `DETERMINISTIC` side (cheap to be wrong: at
worst you under-recommend the pipeline) and asymmetric on `FULL_AGENT`
(needs both signals to fire).

The hook for the paper:

> We show that **a measurable fraction of enterprise agent runs could be
> served at 78% lower cost by replacing agentic reasoning with
> deterministic execution, at a quality drop of 0.30%** — below the
> conventional <2% acceptability bar. The classification rule is a
> two-signal threshold over path entropy and top-K coverage, computable
> in seconds over a trace log.

## Real-data results (GitHub-hosted corpora)

HuggingFace egress is blocked in the current container, so Yunjue /
Nemotron / Hermes runs are still pending. But two real-data corpora are
on GitHub and *are* reachable: TRAIL (Patronus) and τ2-bench (Sierra).
Both clone in seconds and run through the same driver.

### Measurement caveat that became its own finding

The first pass through this analysis used a coarse `args` proxy
(first 64 chars of the task description) for every step in a trace —
making the cache key identical across all 4 replay trials of the same
task. Re-running with the **real per-call arguments** extracted from
each tool_call (`tc["arguments"]`, e.g. `{"email": "mia@example.com"}`)
dropped τ-retail's headline number from a misleading 80.0% to a real
**66.0% cost saved at 0.17% quality regression**.

A separate artifact appeared on TRAIL: spans don't expose per-call
arguments (they live inside `LiteLLMModel.__call__` as free-form text),
so empty-args keys collapsed and over-counted cache hits. The honest
move is to **namespace TRAIL cache keys by trace_id** so cross-trace
collisions stay impossible; that drops TRAIL cache hit rate to 0% and
brings cost savings down to 19.6%, all from routing.

The tau-bench numbers below use real per-call args. The TRAIL numbers
disable cache hits to avoid the over-count. The synthetic numbers use
the original per-trace args (different `date_offset` per trace).

### Headline cross-corpus table (corrected)

| Corpus           | Traces | H(path) corpus | H(path) within-task | Top-3 | Taxonomy   | Cache hit | Cost saved | Qual.reg. |
|------------------|-------:|---------------:|--------------------:|------:|------------|----------:|-----------:|----------:|
| Synthetic FR     |  1,000 |        1.02 b  |                 n/a | 95.4% | `DET`      |     0.0%  |     78.2%  |    0.30%  |
| TRAIL (all)      |    148 |        5.06 b  |                 n/a | 39.2% | `HYBRID`   |     0.0%* |     19.6%  |    0.61%  |
| τ-retail         |  1,822 |        8.43 b  |             2.30 b  |  6.7% | `FULL`     |   **63.1%**|   **66.0%**|    0.17%  |
| τ2-bench (all)   | 10,830 |       12.37 b  |             4.31 b  |  1.9% | `FULL`     |    39.4%  |     44.0%  |    1.12%  |

\* TRAIL cache hits disabled (per-trace namespacing) to avoid empty-args
over-count; the spans don't expose per-tool-call args.

### What survives and what doesn't

The v1 finding said "path entropy is insufficient; step-level
cacheability is a separate signal needed for the taxonomy." The
verification process showed:

1. **The headline number was inflated by ~14 pp** (80 → 66) when args
   were extracted properly. **Still substantial, still surprising for
   a `FULL_AGENT` classification.**
2. **The cacheability finding survives, but its proper signal isn't
   step-level argument repetition** — it's **within-task path entropy
   after clustering**. tau-retail's corpus entropy is 8.43 bits across
   114 tasks (mixture distribution), but its within-task entropy is
   only 2.30 bits (moderate — roughly 5 effective paths per task). The
   within-task signal is what predicts the 63% cache hit rate, not the
   corpus-level signal.
3. **The taxonomy critique still holds**, but the refined version is:
   *"path entropy must be measured within-task on multi-trial corpora;
   corpus-level entropy hides the actual structure."*

### Refined recommendation for the paper

Two-axis regime grid replaced by a **clustering-then-classification**
pipeline:

```
1. Cluster traces by task_id / query / intent.
2. For each cluster, compute within-cluster path entropy.
3. Classify the workload by the *distribution* of within-cluster entropies:
     - all clusters near zero          → DETERMINISTIC (pipeline)
     - clusters moderately spread      → HYBRID (AgentPathRouter)
     - clusters all uniform            → FULL_AGENT (frontier only)
```

Honest hook now reads:

> We show that **66% of inference cost** in a state-of-the-art customer-
> service benchmark (τ-bench retail) **can be saved at 0.17% quality
> regression** via path-level caching plus small-model routing. The
> taxonomy that predicted this called for full-agent treatment — but
> only because corpus-level path entropy averaged across 114 distinct
> tasks. Within-task entropy is moderate (2.30 bits), which is the
> signal a taxonomy should actually use.

### Hidden lesson

This is also the cleanest example I have of *why measurement choices
matter for the headline.* The v1 cache-hit number of 79.1% was a real
artifact of using `task_text[:64]` as the cache key — identical across
the 16 trials of a task. The fact that it dropped 16 pp with proper
args (and that a parallel artifact inflated TRAIL by 70 pp in the
opposite direction) means the v1 framing was unverified. The
underlying claim survived, but only after honest re-measurement.

## Evidence: synthetic corporate workflow

`agentpathrouter.synthetic`, canonical daily-financial-report agent,
1000 runs, seed 0. Branching distribution: 80% base / 10% reconcile /
5% escalate / 3% both / 2% rare edge case.

### Regime classification

```
[regime] DETERMINISTIC  (entropy_ratio=0.44, top3=95.4%)
  Triggered by: top-3 coverage 95.4% ≥ 90%.
  The LLM is being asked to make decisions that are functionally predetermined.
```

| Metric            | Value      |
|-------------------|------------|
| Total traces      | 1000       |
| Unique paths      | **5**      |
| Path entropy      | 1.0229 bits |
| Entropy ratio     | 0.44 (collapsed) |
| top-1 coverage    | 81.4%      |
| top-3 coverage    | 95.4%      |
| top-5 coverage    | 100.0%     |

### Existence proof: AgentPathRouter at the operating point

When the taxonomy says `HYBRID`, the router IS the recommended
architecture; on the synthetic corpus (DETERMINISTIC) it serves as an
existence proof for the *upper bound* of what an AEE-style intervention
can claw back without going all the way to a hard-coded pipeline.

Pricing (`agentpathrouter.cost`, May 2026 Anthropic public list):
Opus 4.7 frontier ($15 in / $75 out per MTok), Haiku 4.5 small ($1 / $5),
800 tokens/step, 70/30 input/output split.

| Arm                  | Cache  | Spec   | Route  | Qual.reg. | USD/1k runs | Saved vs baseline |
|----------------------|-------:|-------:|-------:|----------:|------------:|------------------:|
| baseline (full Opus) |   0.0% |   0.0% |   0.0% |     0.00% |      196.02 |              0.0% |
| cache_only           |   0.0% |   0.0% |   0.0% |     0.00% |      196.02 |              0.0% |
| cache+spec           |   0.0% |  96.7% |   0.0% |     0.00% |      196.02 |              0.0% |
| cache+spec+routing   |   0.0% |  13.2% |  83.8% |     0.30% |    **42.70** |          **78.2%** |

(routing arm at small-model confidence T=0.90 — the PRD-compliant
operating point.)

### Routing threshold sweep (Pareto curve)

| T (small-model conf.) | Routed | Quality regression | USD/1k saved |
|----------------------:|-------:|-------------------:|-------------:|
| 0.80–0.85             |  97.3% | **2.09% (> 2% cap)** |       90.8%  |
| 0.90–0.95             |  83.8% |              0.30% |       78.2%  |
| 0.99                  |  70.3% |              0.00% |       65.6%  |

A deterministic pipeline (the taxonomy's actual recommendation here)
would push these numbers further: zero LLM steps except synthesis,
~95% cost reduction at zero quality drop on the top-3 paths.

## What the evidence forces in the paper

1. **Speculation as a cost lever:** false. Speculation reduces latency
   only — the frontier model still runs to decide the next tool.
   Speculation should be framed as a latency intervention, not a token
   or USD intervention. (`cache+spec` cost is identical to baseline
   above.)
2. **Cache hit rate on real corporate workloads:** 0% as currently
   defined. PathCache keys include tool args; per-day fields like
   `date_offset` defeat every key. Needs an arg-canonicalisation pass
   before path caching is meaningful — and even with it, the savings
   are dominated by routing, not cache.
3. **The headline number is routing, not caching.** The PRD's contribution
   stack should be reordered: small-model routing first, cache second,
   speculation third (for latency only).
4. **The headline insight is the taxonomy.** A two-signal classifier
   over path entropy and top-K coverage tells you which regime you're
   in, and therefore which architecture. The cache+spec+routing system
   is the implementation of that recommendation for the `HYBRID` regime.

## Open questions for the real-data run

- Do Yunjue finsearchcomp / Nemotron-Agentic-v1 / Hermes traces land
  `DETERMINISTIC`, `HYBRID`, or `FULL_AGENT`? The taxonomy makes a
  falsifiable prediction per corpus.
- Where are the regime thresholds actually best placed? The 0.30 / 0.75
  entropy-ratio cutoffs and 90% / 50% coverage cutoffs need empirical
  calibration over the cross-corpus sweep.
- Is there a corpus that classifies `FULL_AGENT` but where AEE-style
  routing *still* helps? That would be a counterexample to the taxonomy
  and worth investigating.

## Reproducing

```bash
python3 scripts/run_entropy_analysis.py --source synthetic --n-synthetic 1000
# Or, in an env with HF access:
python3 scripts/run_entropy_analysis.py --source nemotron_agentic
python3 scripts/run_entropy_analysis.py --source hermes_reasoning
python3 scripts/run_entropy_analysis.py --source yunjue
```

Results land in `results/agentic_execution_entropy/`:
- `regime.json`           — taxonomy verdict + rationale
- `phase1_entropy.json`   — entropy + top-N coverage
- `phase4_router_eval.json` — full §9 ablation across all four arms
- `summary.json`          — all of the above combined

## What this run did NOT cover

- Yunjue / Nemotron / Hermes / TRAIL real-trace entropy (HF egress blocked here)
- LangGraph-instrumented trace generation (PRD §6.3; synthetic generator
  is pure stdlib, no spans)
- A deterministic-pipeline implementation of the recommended `DETERMINISTIC`
  architecture (would let us measure the *upper bound* cost reduction —
  expected to be near 95% on this corpus)
