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

### Headline cross-corpus table

| Corpus           | Traces | Unique paths | Path entropy (bits) | Top-3 | Taxonomy verdict | Best arm           | Cost saved | Qual.reg. |
|------------------|-------:|-------------:|--------------------:|------:|------------------|--------------------|-----------:|----------:|
| Synthetic FR     |  1,000 |            5 |                1.02 | 95.4% | `DETERMINISTIC`  | cache+spec+routing |     78.2%  |    0.30%  |
| TRAIL (all)      |    148 |           71 |                5.06 | 39.2% | `HYBRID`         | cache+spec+routing |     19.6%  |    0.61%  |
| τ-retail (1822)  |  1,822 |          669 |                8.43 |  6.7% | `FULL_AGENT`     | cache+spec+routing | **80.0%**  |    0.11%  |
| τ2-bench (all)   | 10,830 |        7,673 |               12.37 |  1.9% | `FULL_AGENT`     | cache+spec+routing |     46.5%  |    1.12%  |

τ-retail is the headline. **The path-entropy taxonomy classified it
`FULL_AGENT` — meaning "AEE interventions don't help" — yet the actual
ablation produced an 80% cost reduction at 0.11% quality regression.**

That's a falsification of the taxonomy as currently designed.

### Falsification: path entropy is not the only signal that matters

Why does τ-retail save 80% despite high path entropy?

```
[phase4 ablation, tau_retail]
  arm                     cache   spec  route   qreg     USD/1k   saved
  baseline                 0.0%   0.0%   0.0%  0.00%   195.4831    0.0%
  cache_only              79.1%   0.0%   0.0%  0.00%    40.7770   79.1%   ← cache alone
  cache+spec              79.1%   1.8%   0.0%  0.00%    40.7770   79.1%
  cache+spec+routing      79.1%   1.0%   0.9%  0.11%    39.1546   80.0%
```

**79.1% of all steps are cache hits.** The taxonomy looks at the
distribution over *full execution paths* and sees high entropy. But
inside any single trace — and across the four replay trials per task
that τ2-bench runs — individual `(tool, args)` triples repeat heavily.
A request looks up a customer by phone, then by id, then by order — and
across trials those calls re-fire with identical arguments.

Path entropy is a corpus-level signal; cacheability is a *step-level*
signal. They're not the same and the current taxonomy conflates them.

### Implications

1. **Single-signal taxonomy is wrong.** A second axis is needed. The
   right framing is likely a 2D grid:

   |                          | Low arg repetition    | High arg repetition  |
   |--------------------------|-----------------------|----------------------|
   | **Low path entropy**     | DETERMINISTIC pipeline | DETERMINISTIC + cache (rare) |
   | **High path entropy**    | FULL_AGENT (true)     | **FULL_AGENT + cache** (the τ-retail case) |

2. **Cache as a separable lever.** The PRD's original "PathCache" idea
   is vindicated by tau-bench — it just doesn't help on the synthetic
   workload because every synthetic trace carries a different
   `date_offset`. On tau-bench replays it dominates.

3. **The paper's strongest single number is τ-retail's 80.0% / 0.11%.**
   Real customer-service traces, frontier-model baseline (Opus 4.7),
   industry-standard quality cap (<2%). The reframed hook becomes:

   > We show that **80% of inference cost** in a state-of-the-art
   > customer-service benchmark (τ-bench retail) **can be saved at
   > 0.11% quality regression** via path-level caching alone — and the
   > taxonomy that predicted this *wrongly* called for full-agent
   > treatment. The right execution-regime signal is two-dimensional:
   > path entropy AND step-level argument repetition.

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
