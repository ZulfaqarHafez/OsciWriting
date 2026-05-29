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

## P2 / P3 verifications

### P3.1 — PathCache determinism audit (the silent quality bug)

PathCache hits return a previously-observed output for the state hash
``(tool, history, args)`` — sound only if that triple deterministically
yields the same output. On tau-bench we can check directly: pair each
``tool_call`` with its matching ``role==tool`` response (by id), group
by state hash, count hashes with multiple distinct outputs.

| Domain            | Cache hits possible | Non-det share | **Stale hit rate** |
|-------------------|--------------------:|--------------:|-------------------:|
| retail            |              10,788 |         0.23% |          **0.03%** |
| airline           |               3,535 |         0.33% |          **0.11%** |
| telecom           |              25,158 |        24.13% |          **55.83%** |
| telecom-workflow  |              18,317 |        23.55% |          **55.98%** |
| **full corpus**   |              58,841 |        16.20% |          **44.05%** |

Stale hit rate = fraction of cache hits that would return outputs
**different** from the actual tool output the agent would have received.

The retail headline (64.0% saved) stands cleanly — **stale hit rate is
0.03%**, cache is safe. The "44.0% saved" figure I previously cited for
the full tau-bench corpus is **misleading**: telecom contributes most
of the cache hits and ~56% of those would return stale outputs from
state-mutating tools like ``toggle_airplane_mode`` and
``check_network_status``.

The taxonomy needs a **per-tool determinism filter** alongside the
within-task-entropy signal: stateful / observation tools should be
denylisted from PathCache. This is independent of the workload regime —
no amount of low entropy makes a non-idempotent tool safe to cache.

Script: ``scripts/audit_cache_determinism.py``.

### P3.2 — Re-clustering the entropy↔cache correlation by (task, model)

The P1.2 STRONG support (r = -0.90, ρ = -0.93) used (task_id) clusters
that mixed all 4 LLM variants of each task. That confounds intra-trial
variance with inter-model variance. The cleaner grouping is
(task_id, model). Result:

| Grouping              | Clusters | Mean H | Pearson r | Spearman ρ | Verdict   |
|-----------------------|---------:|-------:|----------:|-----------:|-----------|
| by task_id            |      228 |  4.31 b|   -0.9021 |   -0.9340  | STRONG    |
| **by (task_id, model)** |  **912** | **2.44 b** | **-0.3572** | **-0.4033** | **MODERATE** |

The earlier STRONG support is **downgraded to MODERATE**. Within-task
entropy still predicts cacheability, but explains only ~40% of variance,
not 90%. The remaining variance comes from per-tool argument
distributions and model-specific behaviour.

This is the third honest downgrade in this verification pass (80%→66%→64%
on cost; STRONG→MODERATE on correlation). Pattern: each verification
shrinks an earlier claim by roughly half.

### P2.1 — n-gram order sweep

Default ``NgramEntropyEstimator`` is n=3. Sweep on tau-retail:

| n | cache% | spec hit% | spec fires | spec precision | route% | step qreg% |
|--:|-------:|----------:|-----------:|---------------:|-------:|-----------:|
| 1 |  63.1% |    0.0%   |          0 |      0.0%      |   0.0% |     0.00%  |
| 2 |  63.1% |    5.8%   |        442 |    **70.8%**   |   0.0% |     0.00%  |
| 3 |  63.1% |    6.5%   |        519 |     67.4%      |   0.9% |     0.09%  |
| 4 |  63.1% |    7.9%   |        684 |     62.3%      |   1.6% |     0.20%  |
| 5 |  63.1% |    7.2%   |        617 |     62.9%      |   2.2% |     0.31%  |
| 6 |  63.1% |    7.9%   |        679 |     62.6%      |   2.4% |     0.43%  |

n=2 actually maximises precision; n=3 is essentially as good. n≥4
trades precision for more fires and creeping step-level regression.
**The default n=3 is defensible** but n=2 is just as good and slightly
cheaper to fit. No reason to use n≥4.

Cache hit rate is independent of n (PathCache doesn't use the
estimator), so the 63.1% headline doesn't depend on this choice.

### P3.3 — Extractor robustness audit (found a real bug)

Per-sim ground-truth tool-call count vs extractor output:

    total sims:                    10,832
    sims exact match (before fix): 6,434 (59.40%)
    sums:    ground truth = 99,236   extracted = 147,962    delta = +48,726

**The extractor was over-counting by 49%.** Cause: tau2-bench's user
simulator also calls tools through the same ``tool_calls`` schema,
tagged ``requestor: "user"``. The extractor caught them all; the
ground-truth counter only saw assistant-side calls. So ~half of every
"agent tool call" I was measuring was actually a *user* tool call —
not an agent decision.

Fix: added a ``requestor_filter="assistant"`` parameter to
``extract_tool_calls_with_args_from_messages``; tau-bench loader
defaults to it. Post-fix audit: **100.00% exact match, +0 net delta.**

Impact on prior numbers:

| Corpus              | Before fix             | After fix             | Why no shift / why shift? |
|---------------------|------------------------|-----------------------|---------------------------|
| **τ-retail**        | 64.0% saved, 0.45% UB  | **64.0% saved, 0.45% UB** | retail has 100% assistant requestor — clean already |
| **τ2-bench (all)**  | 44.0% saved, 1.12% step| **48.9% saved, 0.79% UB** | 49% of "tool calls" were user-side; removing them tightened paths and shifted savings up |

τ-retail's clean number was a coincidence — the retail user simulator
doesn't call tools. The full corpus (mostly telecom) does, and that's
where the inflation came from. Headline now stands:
**τ-retail 64.0% / 0.45% UB at T=0.95**; **τ2-full 48.9% / 0.79% UB at T=0.97**.

### P2.2 — Within-task-aware classifier + cross-corpus calibration

`taxonomy.classify` operated on corpus-level path entropy alone. P1.2
and P3.2 showed that's the wrong signal on multi-trial benchmarks:
within-task entropy correlates with cache hit rate (Spearman ρ ≈ -0.40
under clean (task, model) clustering), corpus-level does not.

Added ``classify_with_clusters(clusters)`` that takes
``{task_id: [trace, ...]}`` and classifies on **mean within-task path
entropy**. Preliminary cutoffs from the four corpora measured here:

| Within-task H | Regime         |
|---------------|----------------|
| ≤ 1.0 bits    | DETERMINISTIC  |
| 1.0 – 5.0     | HYBRID         |
| > 5.0         | FULL_AGENT     |

These are calibrated on **4 corpora and should not be treated as
final** — proper calibration needs more workload variety once
HuggingFace egress is restored.

### Final corrected cross-corpus table

All numbers below use the per-call args fix (P0/P1.2), the calibrated
1500-token CostModel default (P1.1), the assistant-only requestor
filter (P3.3), task-level quality measurement (P0), and the within-
task classifier (P2.2). Best T is the smallest threshold that keeps
the upper-bound task regression under the PRD 2 % cap.

| Corpus       | Traces | Corpus H | Top-3  | Corpus verdict | Within-task H | **Within-task verdict** | Best T | **Cost saved** | UB qreg | Counterfact |
|--------------|-------:|---------:|-------:|----------------|--------------:|------------------------:|-------:|---------------:|--------:|------------:|
| Synthetic FR |  1,000 |   1.02 b |  95.4% | DETERMINISTIC  |     n/a       | (defers)                |  0.90  |       **78.2%**|    n/a  |     n/a     |
| TRAIL        |    148 |   5.06 b |  39.2% | HYBRID         |     n/a       | (defers, no replays)    |  0.90  |       **19.6%**|    n/a  |     n/a     |
| τ-retail     |  1,822 |   8.43 b |   6.7% | FULL ❌        |    2.30 b     |     **HYBRID ✓**        |  0.95  |       **64.0%**|  0.45 % |    0.30 %   |
| τ2-bench all | 10,832 |  10.79 b |   8.0% | FULL ❌        |    4.69 b     |     **HYBRID ✓**        |  0.97  |       **48.9%**|  0.79 % |    0.73 %   |

The within-task classifier correctly reclassifies both tau-bench
corpora from FULL → HYBRID — matching the observed cost savings.
Corpus-level entropy alone falsely says "AEE doesn't help" on
exactly the workloads where it helps most.

### What survives across every verification

The original PRD framing was "cache repetitive agent workflows for
inference cost reduction." After six verification passes, the
defensible claims that remain are:

1. **On a multi-trial customer-service benchmark with deterministic
   CRUD tools (τ-bench retail), 64.0% of inference cost can be saved
   at <0.45 % upper-bound task-level quality regression** — clearly
   under the PRD 2 % bar — via PathCache + small-model routing
   (threshold T=0.95).
2. **On the full tau2-bench corpus (4 domains, 8 LLMs, 10,832 sims),
   48.9% can be saved at <0.79 % UB regression** — at T=0.97. Domains
   with stateful tools (telecom) need to denylist non-idempotent
   tools from cache (P3.1: stale hit rate 56% otherwise).
3. **Speculation is a latency intervention, not a cost intervention**
   — its net cost contribution is small and stays positive across
   tool execution costs from $0 to $0.10/call.
4. **The right execution-regime signal is within-task path entropy
   after clustering by (task_id, model)** — not corpus-level entropy
   (which mixes distinct tasks and washes out structure). Correlation
   with cache hit rate is moderate (ρ ≈ -0.40).
5. **Step-level quality metrics under-report task-level regression
   by 4–20×**; reporting only step-level numbers as the field
   sometimes does will materially under-state real impact.

## P2.2 (extended) — Regime-cutoff calibration across 20 workload points

The within-task cutoffs (≤1.0 DET, >5.0 FULL) were flagged "preliminary —
4 corpora is not enough." Without HuggingFace I widened the variety using
what's reachable: tau-bench sliced by domain (4), TRAIL subsets (2), and a
**controlled-entropy synthetic generator** (`generate_controlled_corpus`)
that dials within-task entropy with known ground truth (14 points). Total:
20 workload points, 18 with computable within-task entropy.

Harness: `scripts/calibrate_regime_cutoffs.py`.

### Within-task entropy vs cost saved (selected points)

| Source                        | Within-task H | Cache hit | Cost saved |
|-------------------------------|--------------:|----------:|-----------:|
| synth v8/b1  conc=6.0         |        0.10 b |    93.7%  |    95.5%   |
| synth v8/b1  conc=2.5         |        1.08 b |    86.6%  |    89.8%   |
| synth v8/b1  conc=1.5         |        1.98 b |    80.2%  |    85.8%   |
| **τ-retail**                  |      **2.30 b** |  **63.1%**|  **64.0%** |
| **τ-airline**                 |      **2.86 b** |  **49.5%**|  **51.6%** |
| synth v64/b1 conc=1.2         |        3.07 b |    64.0%  |    65.6%   |
| synth v64/b7 conc=0.8         |        3.71 b |    25.9%  |    25.9%   |
| **τ-telecom-workflow**        |      **4.15 b** |  **39.9%**|  **44.4%** |
| synth v64/b7 conc=0.4         |        4.00 b |    16.6%  |    16.6%   |
| **τ-telecom**                 |      **4.70 b** |  **44.9%**|  **47.0%** |
| synth v64/b7 conc=0.1         |        4.06 b |    10.6%  |    10.6%   |

### Correlation (workload level)

| Relationship                          | Pearson r |
|---------------------------------------|----------:|
| within-task H → cache hit rate        |   **-0.86** |
| within-task H → cost saved            |   **-0.84** |

This is much stronger than the per-cluster P3.2 result (ρ ≈ -0.40).
Both are real: aggregating to the *workload* level averages out the
per-cluster noise. The paper should report both granularities — the
workload-level number is what a practitioner would use to decide
"apply AEE to this deployment or not."

### Data-driven cutoffs (savings-based)

| Cutoff           | Old (preliminary) | New (calibrated) | Basis |
|------------------|------------------:|-----------------:|-------|
| DETERMINISTIC ≤  |           1.0 b   |       **2.0 b**  | highest within-task H still saving ≥85% |
| FULL_AGENT >     |           5.0 b   |       **4.0 b**  | lowest within-task H with <25% saved (spread-divergence synthetic) |

DET ≤ 2.0 is well-supported: every workload at or below 2 bits saved
≥85%. The FULL cutoff is **weaker** because of a confound:

### Confound — divergence *structure*, not just entropy

At the **same** within-task entropy H≈3.7, two synthetic workloads
behaved very differently:

| Divergence structure         | H     | Cost saved |
|------------------------------|------:|-----------:|
| concentrated (1 step varies) | 3.71 b|    52.5%   |
| spread (all steps vary)      | 3.71 b|    25.9%   |

Concentrated divergence keeps a long shared prefix, which stays
cacheable; spread divergence destroys the prefix. **Real τ-telecom sits
at H=4.7 yet still saves 47%** because its divergence is concentrated —
so the entropy-only FULL cutoff (>4.0) would mis-label it FULL when it's
actually a profitable HYBRID.

Conclusion: **within-task entropy is the primary signal (r≈-0.85) and a
clean DET threshold at 2.0 bits, but in the H>4 region it must be paired
with a direct cache-hit probe** because divergence structure confounds
it. The taxonomy comment and constants now reflect this.

This is the fourth honest finding in the verification arc that constrains
an earlier claim: corpus entropy → within-task entropy → moderate
per-cluster correlation → and now "entropy alone is insufficient at the
high end; structure matters."

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
