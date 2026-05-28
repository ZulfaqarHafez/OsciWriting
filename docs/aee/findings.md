# Agentic Execution Entropy — Phase 1 findings (synthetic corpus)

Source PRD: `Agentic Execution Entropy / AgentPathRouter`
(uploaded `376822c6-AgenticExecutionEntropy_PRD.docx`, May 2026).

This is the in-environment execution of the PRD. HuggingFace was blocked in
this container (`host_not_allowed` on `huggingface.co`), so the Yunjue
`finsearchcomp` split could not be downloaded here. The driver
(`scripts/run_entropy_analysis.py`) already supports it — pass
`--source yunjue` from an environment with network access.

The synthetic corporate-workflow corpus (`agentpathrouter.synthetic`,
canonical daily financial report agent, 1000 runs, seed 0) is the substitute
for now. The branching distribution is documented in source:

| Scenario              | Probability |
|-----------------------|-------------|
| base                  | 80%         |
| reconcile             | 10%         |
| escalate              | 5%          |
| reconcile + escalate  | 3%          |
| rare edge case        | 2%          |

## Phase 1 — entropy + coverage (PRD §5.1)

| Metric              | Value      |
|---------------------|------------|
| Total traces        | 1000       |
| Unique paths        | **5**      |
| Path entropy        | **1.0229 bits** |
| top-1 coverage      | 81.4%      |
| top-3 coverage      | 95.4%      |
| top-5 coverage      | 100.0%     |

PRD §5.1 predicted "top 10 paths cover 70–90% of runs in structured
workflows." The synthetic corpus comes in at the high end of that band —
top-3 covers 95.4%, which sets the expectation we'd want to beat
(or at least match) on Yunjue finsearchcomp once that's reachable.

## Phase 4 — AgentPathRouter eval (PRD §5.3)

n-gram estimator (n=3), confidence threshold T = 0.7, 600/400 train/test
split.

| Metric                 | Value   |
|------------------------|---------|
| Test traces            | 400     |
| Total agent steps      | 2970    |
| Cache hit rate         | 0.0%    |
| Speculation hit rate   | **96.7%** |
| Full-call rate         | 3.3%    |
| Speculation precision  | 97.6%   |
| Speculation fires      | 2942    |

### The cache-vs-speculation split is the headline finding

Cache hit rate is 0% because every synthetic trace carries a distinct
`date_offset` (modelling "different day, same workflow"), so the state hash
(tool + history + args) never repeats across runs. That is realistic —
PathCache as defined only fires on identical re-invocations, which in a
production corporate setting requires either (a) tool arg canonicalisation
that strips per-day fields, or (b) tool-level caching that's aware of which
args are deterministic vs incidental.

Speculation does not have that problem: it exploits low entropy in the
*sequence shape*, independent of args. 96.7% of steps are correctly
predicted in advance, which directly translates to latency savings (the
predicted tool result is already in hand when the LLM emits its choice).

### Implications for the ablation in PRD §9 Phase 4

The PRD calls for a three-way ablation:
1. cache only
2. cache + speculation
3. cache + speculation + small-model routing

The synthetic-corpus result suggests **cache-only will look very weak**
unless tool-arg canonicalisation is added as a preprocessing step. Without
it, the headline cost reduction has to come from speculation + routing,
not from path-level caching of *outputs*. Worth flagging before running
the real eval.

## Reproducing

```bash
python3 scripts/run_entropy_analysis.py --source synthetic --n-synthetic 1000
# Or, in an env with HF access:
python3 scripts/run_entropy_analysis.py --source yunjue
```

Results land in `results/agentic_execution_entropy/{summary,phase1_entropy,phase4_router_eval}.json`.

## What this run did NOT cover

- Yunjue / TRAIL real-trace entropy (blocked: no HuggingFace egress)
- LangGraph-instrumented trace generation (PRD §6.3 uses LangGraph +
  AgentTrace SDK; the synthetic generator here is pure stdlib and skips
  span instrumentation)
- Small-model routing arm (PRD §5.2 third component)
- Token / USD cost numbers — only step-level proxy rates (hit/miss/full)
  are reported. Wiring real model-cost numbers needs the cost-model from
  the existing `redundancy` package, or the PRD's preferred pricing source.
