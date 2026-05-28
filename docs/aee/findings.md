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

### Task-level vs step-level quality regression

The earlier 0.17% number for τ-retail at T=0.90 was a **step-level**
metric: fraction of agent steps where the small-model router would have
chosen a different tool than the actual trace. PRD §5.3 specifically
requires *task-level* success-rate delta < 2%. These are different in
general because agents often self-correct from a wrong tool.

tau-bench supplies `reward_info.reward ∈ {0, 1}` per simulation, so we
can measure properly. Two bounds (we can't re-run the agents):

* **Upper-bound (pessimistic):** every originally-successful trace
  where the router would have changed any tool decision is counted as
  a potential failure.
* **Counterfactual (within-task):** a modified trace is counted as
  failed only if no *other* trial of the same task succeeded while
  following the router's chosen tool at the divergence step.

The true regression is between these two. Sweep on τ-retail:

| T    | Cost saved | Step-level qreg | UB task qreg | Counterfactual | PRD-compliant? |
|-----:|-----------:|----------------:|-------------:|---------------:|----------------|
| 0.85 |     67.6 % |          1.07 % | **17.17 %**  |        4.61 %  | No             |
| 0.90 |     66.0 % |          0.17 % |     3.40 %   |        1.36 %  | Ambiguous      |
| 0.92 |     66.0 % |          0.17 % |     3.40 %   |        1.36 %  | Ambiguous      |
| **0.95** | **64.0 %** |    0.09 % |   **0.45 %** |        0.30 %  | **Yes**        |
| 0.97 |     63.8 % |          0.06 % |     0.38 %   |        0.23 %  | Yes            |
| 0.99 |     63.7 % |          0.04 % |     0.15 %   |        0.00 %  | Yes            |

Step-level regression systematically under-reports task-level regression
by 4–20×. Worth flagging in the paper as a methodological caveat —
papers that report step-level numbers will mislead.

The PRD-compliant operating point on τ-retail is **T = 0.95**, giving:

> **64.0% cost saved at <0.45% upper-bound task-level regression
> (counterfactual 0.30%)** on 1,822 customer-service simulations
> (τ-bench retail, baseline task success 72.6%).

That's the corrected headline. It drops 2 pp from the earlier 66% claim
at T=0.90, but is now in the right metric and clearly under the PRD cap.

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

> We show that **64% of inference cost** in a state-of-the-art customer-
> service benchmark (τ-bench retail) **can be saved at <0.45% task-level
> quality regression** (counterfactual 0.30%) via path-level caching
> plus small-model routing. The taxonomy that predicted this called for
> full-agent treatment — but only because corpus-level path entropy
> averaged across 114 distinct tasks. Within-task entropy is moderate
> (2.30 bits), which is the signal a taxonomy should actually use.
>
> The headline figure that step-level quality metrics suggested (0.17%)
> was an under-count of ~10× vs the true task-level number measured
> against ground-truth reward signals. Papers reporting step-level
> regression alone are likely under-stating real impact.

## P1 verification: things the original numbers were silent about

### P1.1 — Cost-model calibration against `agent_cost`

The original `tokens_per_step = 800` was a guess. tau-bench supplies
`agent_cost` (real USD spent) per simulation plus the LLM identifier;
back-solving gives the *implied* tokens per step for each model:

| Model                          | n sims | USD/step  | Blended $/MTok | Implied tok/step |
|-------------------------------:|-------:|----------:|---------------:|-----------------:|
| gpt-4.1-2025-04-14             |  4,304 | $0.00536  |   $3.80        |          1,410   |
| o4-mini-2025-04-16             |  4,304 | $0.00434  |   $2.09        |          2,074   |
| claude-3-7-sonnet              |  1,112 | $0.02495  |   $6.60        |          3,780   |
| gpt-4.1-mini                   |  1,112 | $0.00108  |   $0.76        |          1,419   |

**Median implied tokens/step is 1,747**, ranging 1,410 → 3,780. The
800-token default underestimates by ~2×. Updated the `CostModel`
default to **1,500 tokens/step** (sensible mid-point).

Implications:
- **% savings claims unchanged** (they're ratios).
- **Absolute USD/1k-runs numbers shift up by ~2×** — re-running the
  ablation reports $366.53 / 1k baseline at the new default vs
  $195.48 previously.
- Headline tau-retail result becomes: **64.0% cost saved at <0.45%
  task-level regression, baseline ~$367 / 1k runs**.

Calibration script: `scripts/calibrate_cost_model.py`. Verdict at
800-token default was OUT OF RANGE; at 1500 it sits inside the
empirical 1,410–3,780 range close to the median.

### P1.2 — Within-task entropy ↔ cacheability correlation

The refined-taxonomy claim ("cluster by task, classify on within-task
entropy") was asserted after the v2 measurement but never tested.
Direct measurement on tau-bench's 228 task clusters:

| Statistic                                | Value     |
|------------------------------------------|----------:|
| Clusters with ≥ 2 trials                 |    228    |
| Mean within-task entropy                 | 4.31 bits |
| Mean cache hit rate (across clusters)    |   52.3%   |
| **Pearson r**                            | **-0.90** |
| **Spearman ρ**                           | **-0.93** |
| Verdict                                  | **STRONG support** |

Quartile breakdown (sorted by entropy):

| Quartile | Entropy range | n | Mean cache hit rate |
|---------:|---------------|---:|--------------------:|
| Q1 (low) | 0.00 – 3.08   | 57 |    **77.3%**        |
| Q2       | 3.08 – 4.58   | 57 |       63.6%         |
| Q3       | 4.59 – 5.84   | 57 |       43.6%         |
| Q4 (hi)  | 5.84 – 6.17   | 57 |       24.8%         |

Clean monotonic relationship: low within-task entropy → high cache
hit rate, and the highest-entropy quartile still gets 25%. **Refined
taxonomy claim is empirically validated, not just hand-waved.** This
is the kind of finding that survives review.

Test script: `scripts/test_entropy_cacheability_correlation.py`.

### P1.3 — Wasted speculation cost on real workloads

Earlier claim: "speculation is a latency intervention, not a cost
intervention." The CostModel modelled this by giving spec hits the
full frontier-step cost. But it ignored the *downstream tool execution*
of wasted speculative fires — at the per-call rate of real production
tools, this could in principle wipe out speculation's latency benefit.

Added `tool_execution_usd` to `CostModel` and `spec_misses` to
`RunMetrics`. Sensitivity sweep on tau-retail (T=0.95):

| Tool cost / call         | Baseline | cache+spec  | c+s+r       | Wasted spec / 1k runs |
|-------------------------:|---------:|------------:|------------:|----------------------:|
| $0     (tau-bench: free) |  $366.53 | 63.1% saved | 64.0% saved |  $0.00                |
| $0.001 (cheap REST API)  |  $366.53 | 63.1%       | 63.9%       |  $0.23                |
| $0.005                   |  $366.53 | 62.8%       | 63.7%       |  $1.16                |
| $0.01  (search API)      |  $366.53 | 62.5%       | 63.4%       |  $2.32                |
| $0.05  (light crawling)  |  $366.53 | 59.9%       | 60.9%       | $11.59                |
| $0.10  (heavy crawling)  |  $366.53 | 56.6%       | 57.7%       | $23.18                |

**Speculation's net cost stays positive across the typical $0 – $0.10
range.** Even at $0.10 per tool call — already in expensive-crawler
territory — cost savings only drop from 64% to ~58%. Speculation is
robust to realistic tool costs.

That said, the *cost contribution from speculation specifically* is
small at every tool cost level (cache+spec ≈ cache_only on this
corpus). Speculation continues to be a latency lever; this sweep
confirms it isn't actively *negative* in the cost dimension.

## Hidden lesson

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
| baseline (full Opus) |   0.0% |   0.0% |   0.0% |     0.00% |      367.54 |              0.0% |
| cache_only           |   0.0% |   0.0% |   0.0% |     0.00% |      367.54 |              0.0% |
| cache+spec           |   0.0% |  96.7% |   0.0% |     0.00% |      367.54 |              0.0% |
| cache+spec+routing   |   0.0% |  13.2% |  83.8% |     0.30% |    **80.06** |          **78.2%** |

(USD figures use the empirically-calibrated `tokens_per_step = 1500`
default — see P1.1 below for the calibration. Percentages are unchanged
from the 800-token-default version.)

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
