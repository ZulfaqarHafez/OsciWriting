# Agentic Execution Entropy (AEE)

Implementation of the PRD
*"Exploiting Repetitive Corporate Agent Workflows for Inference Cost Reduction"*
(uploaded `376822c6-AgenticExecutionEntropy_PRD.docx`), **reframed** around
a workflow-regime taxonomy — see `docs/aee/findings.md`.

> **Headline claim.** Path execution entropy is a measurable signal that
> classifies an agent workload into one of three regimes
> (`DETERMINISTIC` / `HYBRID` / `FULL_AGENT`). When the regime is
> `DETERMINISTIC`, the LLM is being used to make decisions that are
> already determined by the input — the right intervention is to
> replace the agent with a deterministic pipeline, not to cache it.
> `AgentPathRouter` is the recommended architecture for the `HYBRID`
> middle case.

## Layout

```
src/agentpathrouter/
    taxonomy.py                  REGIME CLASSIFIER — the headline contribution
    entropy.py            §5.1   path entropy, top-N coverage, tool-seq extraction
    data_sources.py              loaders: yunjue / nemotron / hermes / tau-bench
    synthetic.py          §6.3   synthetic "daily financial report" trace generator
    path_cache.py         §5.2.1 PathCache (state-hash → cached output)
    entropy_estimator.py  §5.2.2 n-gram next-tool predictor
    speculative.py        §5.2.3 SpeculativePrefetcher (parallel pre-fire)
    middleware.py                AgentPathRouter — cache + spec + small-model routing
    cost.py                      CostModel with May-2026 Anthropic pricing

scripts/run_entropy_analysis.py  driver: load → entropy → regime → §9 ablation
tests/test_agentpathrouter_*.py  unit tests (44, all passing)
docs/aee/findings.md             Reframed findings + Phase 4 ablation numbers
results/agentic_execution_entropy/  machine-readable JSON results
```

## PRD phase status

| Phase | Description                              | Status                                     |
|-------|------------------------------------------|--------------------------------------------|
| 1     | Data + entropy analysis                  | Done on synthetic; Yunjue/Nemotron/Hermes wired, blocked by HF egress |
| 1b    | **Regime taxonomy** (new headline)       | Done — `agentpathrouter.taxonomy.classify`; thresholds preliminary, need cross-corpus calibration |
| 2     | Synthetic corpus                         | Done (pure-stdlib generator, no LangGraph dep) |
| 3     | System build (cache + estimator + spec + small-model routing) | Done; cost model wired with May 2026 Anthropic pricing |
| 4     | Evaluation + writing                     | Full §9 ablation runs locally; threshold sweep documented in `docs/aee/findings.md`. Paper-writing still ahead. |

## Data sources wired into `scripts/run_entropy_analysis.py`

Pick with `--source <name>`. All HF sources fall through to a clean
`DatasetUnavailable` error when network is blocked.

| `--source` | Repo / location | Why |
|---|---|---|
| `yunjue` | `YunjueTech/Yunjue-Agent-Traces` (finsearchcomp) | PRD §6.1 primary |
| `nemotron_agentic` | `nvidia/Nemotron-Agentic-v1` | Large synthetic multi-turn trajectories |
| `hermes_reasoning` | `lambda/hermes-agent-reasoning-traces` | Multi-turn tool calls + reasoning blocks |
| `hermes_filtered` | `DJLougen/hermes-agent-traces-filtered` | Quality-pruned Hermes subset |
| `tau_bench` | local dir (`--tau-bench-dir`) | τ-bench (sierra-research/tau2-bench) — generate locally, then point loader at `data/simulations/` |
| `synthetic` | in-process | PRD §6.3 stand-in (always works) |
| `auto` | yunjue → synthetic | Default; tries yunjue, silent fallback |

### Using τ2-bench's shipped simulation results

The repo ships ~10,830 real simulations in
`data/tau2/results/final/` across retail / airline / telecom / telecom-
workflow domains. No LLM credits needed — clone and point the loader:

```bash
git clone https://github.com/sierra-research/tau2-bench /tmp/tau2-bench
python3 scripts/run_entropy_analysis.py \
    --source tau_bench --tau-bench-dir /tmp/tau2-bench/data/tau2/results/final
```

To generate fresh traces, see the upstream README (needs LLM API keys).

### TRAIL benchmark

```bash
git clone https://github.com/patronus-ai/trail-benchmark /tmp/trail-benchmark
python3 scripts/run_entropy_analysis.py \
    --source trail --trail-dir /tmp/trail-benchmark --trail-subset all
```

Subsets: `gaia` (117 traces), `swe_bench` (31 traces), `all` (default).

## Not yet done (explicit follow-ups)

- **Second taxonomy axis: step-level cacheability.** Real-data falsification
  on τ-retail (80% cost saved at 0.11% quality regression despite
  `FULL_AGENT` classification) shows path entropy alone is insufficient.
  The 2D regime grid is proposed in `docs/aee/findings.md`. Implementation:
  add a cacheability signal (fraction of `(tool, args)` triples that
  repeat in-corpus) and switch `taxonomy.classify` to a 2D rule.
- **Calibrate regime thresholds** on Yunjue / Nemotron / Hermes — current cutoffs are synthetic-only.
- **Deterministic-pipeline implementation** of a top-K-path workflow as the comparison point for the `DETERMINISTIC` regime (expected to push cost savings from ~78% to ~95%).
- Run on real Yunjue / Nemotron / Hermes — drivers ready, blocked on HF egress in this container.
- LangGraph + AgentTrace SDK span instrumentation (PRD §6.3).
- TRAIL benchmark error-pattern analysis (PRD §6.2 — secondary, lower priority).
- Paper itself, now with the reframe as the spine.
