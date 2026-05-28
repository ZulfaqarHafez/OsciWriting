# Agentic Execution Entropy (AEE)

Implementation of the PRD
*"Exploiting Repetitive Corporate Agent Workflows for Inference Cost Reduction"*
(uploaded `376822c6-AgenticExecutionEntropy_PRD.docx`).

## Layout

```
src/agentpathrouter/
    entropy.py            §5.1   path entropy, top-N coverage, tool-seq extraction
    synthetic.py          §6.3   synthetic "daily financial report" trace generator
    path_cache.py         §5.2.1 PathCache (state-hash → cached output)
    entropy_estimator.py  §5.2.2 n-gram next-tool predictor
    speculative.py        §5.2.3 SpeculativePrefetcher (parallel pre-fire)
    middleware.py                AgentPathRouter — orchestrates the three above

scripts/run_entropy_analysis.py  driver: Yunjue OR synthetic → entropy + router eval
tests/test_agentpathrouter_*.py  unit tests (11, all passing)
docs/aee/findings.md             Phase 1 + Phase 4 results on synthetic corpus
results/agentic_execution_entropy/  machine-readable JSON results
```

## PRD phase status

| Phase | Description                              | Status                                     |
|-------|------------------------------------------|--------------------------------------------|
| 1     | Data + entropy analysis                  | Done on synthetic; Yunjue path wired, blocked by HF egress |
| 2     | Synthetic corpus                         | Done (pure-stdlib generator, no LangGraph dep) |
| 3     | System build (cache + estimator + spec)  | Done (in-memory; ready for Redis/sqlite swap) |
| 4     | Evaluation + writing                     | Step-level rates reported; token/USD cost & small-model routing arm not yet wired |

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

### Generating τ-bench traces locally

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench && uv sync
uv run python -m tau2 run-and-eval --domain retail \
    --agent-llm gpt-4o-mini --num-trials 200
# back in this repo:
python3 scripts/run_entropy_analysis.py \
    --source tau_bench --tau-bench-dir /path/to/tau2-bench/data/simulations
```

## Not yet done (explicit follow-ups)

- Run Phase 1 on real Yunjue / Nemotron / Hermes — drivers ready, blocked on HF egress in this container.
- LangGraph + AgentTrace SDK span instrumentation (PRD §6.3).
- Small-model routing arm of the ablation (PRD §5.2 third component).
- Wire token/USD cost numbers — either through `src/redundancy/cost_model.py`
  or fresh pricing per PRD §5.3.
- TRAIL benchmark error-pattern analysis (PRD §6.2 — secondary, lower priority).
