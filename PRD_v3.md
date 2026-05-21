# LLM Prefetch Latency Study (PRD v3)

**Status:** Pre-experiment, design locked (v3)
**Author:** Zul Fhagez
**Last updated:** 2026-05-21
**Decision deadline:** within 1 week of data collection completion

---

## 0. Changelog (v2.2 → v3)

v1–v2.2 answered "can we cache LLM responses to *reduce inference cost at
acceptable quality*?" — and reached Abandon via the H5 substitutability gate on
LMSYS-Chat-1M. That study is complete and documented in `PRD.md` + `findings.md`.

v3 answers a different question that's been the real motivation all along:
**"can we speculatively prefetch likely next prompts' responses to reduce
perceived latency, like Gmail prefetching emails on login?"** This is a
latency-optimization problem, not a cost-reduction one. Specifically:

1. **Reframed primary metric.** Hit rate, not savings. Cost is bounded by budget,
   not minimized; quality is preserved by serve-threshold gating; latency is
   the outcome being purchased.
2. **H5 substitutability stops being the gate.** Wrong predictions in prefetch
   are silently discarded — the user's actual prompt then runs on-demand
   exactly like it would have anyway. There is no quality degradation from a
   miss. Quality is guaranteed by the *serve* threshold (only commit a
   pre-generated response when the actual prompt is close enough that it
   serves), not by averaging acceptability across cosine bands.
3. **Multi-turn becomes the domain.** v2 PRD §13 explicitly excluded multi-turn
   ("first-turn only"). v3 makes multi-turn the central data: the question is
   *what does the user ask next given what they just asked*.
4. **Decision rubric becomes a Pareto check, not a pass/fail.** Prefetch is a
   tradeoff (compute budget for latency benefit). The rubric defines a
   feasibility region (K, T_pred, T_serve, compute_budget) and asks whether
   any point in the region clears latency-value vs compute-cost.
5. **What carries over.** Dataset (LMSYS-Chat-1M; can also use WildChat as a
   cautionary contrast); deduplication (§8.6 of v2.2); MiniLM + TF-IDF
   retrievers (v2.2 §15 multi-turn investigation prefigured this); the H1
   degeneracy guard logic; Windows-safe paths; cost-model arithmetic skeleton
   (different inputs, same structure).

---

## 1. Purpose

Determine, with quantitative evidence, whether speculative prefetch of LLM
responses based on user prompt history is feasible — i.e., whether a system that
pre-generates K likely next-prompt responses per turn can reliably serve the
user's actual next prompt from local cache, at a compute budget that's
proportional to the latency value gained.

This study is the precondition for committing to or abandoning a prefetch-based
assistant product. As before, the study is not the product; the product (if
warranted) is downstream of the numbers.

The motivation is the web-app prefetch pattern: when you log into Gmail it
prefetches your inbox so scrolling feels instant. Translated to LLM use: when
you send a prompt, the system speculates 1–K likely follow-ups in the
background; if your next prompt matches a speculation, your perceived latency
to first token is essentially zero.

---

## 2. Background

### Why this is a different study than v2.2

The v2.2 cost-cache study showed substitutability (H5) fails on LMSYS English
writing prompts (best ~13–15% at cosine ≥0.95 versus 70% required). That kills
cost-reduction caching: you can't replace frontier calls with cached responses
without quality loss. But it does *not* kill prefetch, because prefetch never
substitutes — it only serves on near-exact hit, and falls back to on-demand
otherwise. **The substitutability ceiling caps the *serve threshold*, not the
*hit threshold*.**

### Adjacent prior art

- **Speculative decoding** (token-level, draft+target model): well-deployed for
  inference latency. Different mechanism — operates *during* generation, not
  *between* generations. Doesn't address whole-prompt anticipation.
- **Anthropic prompt caching** (KV-cache reuse on repeated system prompts):
  deployed, useful, doesn't predict — only reuses identical prefixes.
- **Web prefetch (rel="prefetch", service workers)**: the operational template
  for what we're proposing — speculate, store locally, serve on click, throw
  away on tab close.
- **FrugalGPT / cascade routing**: different problem (route by complexity).

No widely-deployed open-domain LLM whole-prompt prefetch exists. This study
asks whether one would work.

---

## 3. Hypotheses

For LMSYS-Chat-1M conversations with ≥2 English user turns:

**HP1 (predictability above chance).** For some retriever R ∈ {MiniLM, TF-IDF}
and some (K, T_pred) ∈ ({1, 3, 5, 10, 25} × {0.7, 0.8, 0.9, 0.95}), the
empirical Hit@K@T_pred for retriever R exceeds the random-baseline rate by at
least 10 percentage points. Hit means: among the top-K nearest turn_N in
*other* conversations under R's similarity, at least one of their turn_{N+1}
lies within MiniLM cosine ≥ T_pred of the actual turn_{N+1}. If no
(R, K, T_pred) cell clears this, cross-user prediction is fake and the project
is **Abandon**.

**HP2 (serve-quality safety).** At a serve threshold T_serve ≥ 0.95 (very
near-duplicate), the cached response's acceptability for the actual prompt
must be ≥ 60% — judged the same way as v2.2 H5 (Claude Haiku with the audit
sampling discipline). Below 60%, prefetch serves bad answers and the user
sees them — defeats the whole point. T_serve is *calibrated* from this data,
not asserted.

**HP3 (within-session predictability is at least as strong as cross-user).**
The same-conversation consecutive cosine distribution has p50 ≥ 0.60 and
≥ 30% of pairs at cosine ≥ 0.70. (If true, within-session prefetch is the
practical first line — predict-by-rephrasing-the-last-turn — and cross-user is
secondary.)

**HP4 (budget feasibility).** There exists a (K, T_pred, T_serve) tuple
satisfying HP1 and HP2 such that the speculative cost per turn — defined as
`B × c_speculate` where `B = K` candidates and `c_speculate` is the cost of
generating one candidate via the chosen speculator model — divided by the
expected latency value per hit, yields a positive ROI under stated assumptions
(§4a). If no such tuple exists, prefetch is uneconomical even where it works.

---

## 4. Decision rubric

| HP1 | HP2 | HP4 | Outcome | Action |
| --- | --- | --- | --- | --- |
| Fail | any | any | **Abandon** | Document; cross-user prediction is fake on this corpus. No re-scope. |
| Pass | Fail | any | **Abandon** | Prefetch would serve bad answers; the cosine signal is not safety-sufficient. |
| Pass | Pass | Fail | **Re-scope to narrower domain** | Cross-user prefetch works on the safe band but doesn't pay back; try a constrained-domain dataset (PRD v3 §13 listed re-scope candidates). |
| Pass | Pass | Pass | **Commit to project** | Design architecture; pick (K, T_pred, T_serve). Begin §6 timeline. |
| Pass on HP3 only | — | — | **Re-scope to within-session only** | Cross-user is uneconomical or unsafe but in-session predictability is real. Build a within-session prefetcher (single-user mode), skip cross-corpus warm-boot. |

HP1 is necessary across HP2/HP4 — without prediction, nothing works. HP2 is a
quality gate but unlike v2.2 H5, it's a *calibrated* threshold (we tune
T_serve up until safety is met); failure here means *no* T_serve produces
60%+ acceptability, not "this specific T_serve doesn't." HP3 is the
within-session fallback path.

### 4a. Cost / value model

All costs normalized to `c_frontier = 1.0` (Opus 4.7 at $5 in / $25 out per
MTok, frozen as of v2.2). New constants:

- `c_speculate` — cost of generating one speculative candidate. Options:
  - Frontier itself (Opus 4.7): `c_speculate = 1.0`.
  - Small-model speculation (Haiku 4.5): `c_speculate = 0.2`.
  - Cheap-model speculation (Haiku Lite if available): lower.
- `K` — number of candidates pre-generated per turn (the speculation budget).
- `latency_value_per_hit` — the value of saving one frontier-latency
  (typically 2–5 seconds) per turn. **Author must set this** based on the
  product context. Defaults: `0.5` (50% of frontier cost) for high-latency-
  sensitive products, `0.1` for cost-conscious ones.

Per-turn ROI:

```
roi_per_turn = hit_rate(K, T_pred, T_serve) × latency_value_per_hit
             − K × c_speculate
```

HP4 passes iff there exists (K, T_pred, T_serve) where `roi_per_turn > 0`.

Author must freeze `latency_value_per_hit` and `c_speculate` in
`docs/cost_model_v3.md` **before any decision run**, then re-run
`redundancy.cost_model_v3 --write-doc`. Until that's frozen the analysis is a
pilot, not the decision run — same discipline as v2.2.

---

## 5. Repository structure

Builds on the v2.2 layout (`src/redundancy/`). New / modified files:

```
src/redundancy/
  multi_turn.py         # v2.2 added; v3 makes it the primary metric module
  prefetch.py           # NEW: HP1–HP4 evaluation, Pareto sweep, decision
  cost_model_v3.py      # NEW: per-turn ROI arithmetic + threshold solver
  data.py               # load_conversations (v2.2); used as-is

results/
  prefetch_<TS>/        # NEW prefix (distinct from pipeline & multiturn)
    config.json
    headline_numbers.json
    pareto.csv          # K × T_pred × T_serve × hit × quality × ROI
    figures/
      hit_vs_K.png
      pareto.png
      same_conv_cosine.png
    summary.md

docs/
  cost_model_v3.md      # frozen latency_value + c_speculate
  decision_log.md       # extended with v3 entries
  findings_v3.md        # post-run write-up
  architecture_v3.md    # conditional on Pass
```

Conventions: seeded everywhere (SEED=42); Windows-safe timestamps (no colons);
`results/latest.txt` pointer; no notebook output committed.

---

## 6. Environment setup

Same Python 3.11+ and dependency set as v2.2 (`pyproject.toml`). The judge
(Claude Haiku 4.5) is needed for HP2 only; HP1/HP3 are judge-free. New
optional: a small-model speculator endpoint if you want to evaluate
speculative-decoding-style cost (otherwise `c_speculate` = analytical).

HF token + LMSYS license must be in place (already cleared in v2.2).
`ANTHROPIC_API_KEY` required for HP2.

---

## 7. Data sources

**Primary: LMSYS-Chat-1M** with full conversations (≥2 English user turns).
Reservoir-sample via `data.load_conversations`. v2.2 §13's first-turn
restriction is *amended* here — multi-turn is the entire point.

**Cautionary: WildChat-1M.** Still demoted to fallback for the same template-
contamination reasons; usable only with §8.6 dedup + the template-family
caveat from v2.2 findings.

### Sample size

- Pilot: 5,000 conversations (yields ~8–15k turn-pairs depending on multi-turn
  rate). Per-conversation kept only if `n_pairs ≥ 1`.
- Decision run: 50,000 conversations target. N-stability acceptance test
  (5k vs 50k headline-numbers within 15% relative) inherited from v2.2 §7
  discipline.

---

## 8. Methodology

### 8.1 Pipeline stages

```
[raw LMSYS]
  -> language + multi-turn filter         (data.load_conversations)
  -> EXACT dedup on full conversations    (dedup.exact, extended)
  -> pair extraction (turn_N -> turn_{N+1}) (multi_turn.extract_pairs)
  -> embed anchors + targets (MiniLM)     (embed.embed)
  -> TF-IDF anchors                       (multi_turn._nn_indices)
  -> kNN over both retrievers + random baseline
  -> hit@K@T_pred table per retriever × controls   (HP1)
  -> same-conv consecutive cosine                  (HP3)
  -> for each high-cosine cell (cos ≥ 0.95):
       judge acceptability of cached-vs-actual    (HP2)
  -> Pareto sweep over (K, T_pred, T_serve, c_speculate, latency_value)
       compute ROI per cell                       (HP4)
  -> decide per §4 rubric                         (prefetch.decide)
  -> write headline + figures + Pareto CSV        (prefetch.report)
```

Three retrievers run identically:
- **MiniLM** (consistent with v2.2 study).
- **TF-IDF** (lexical-overlap, "graph similarity" arm).
- **Random** baseline (no signal → null floor).

### 8.2 Dedup

Same as v2.2 §8.6, applied to conversations rather than first-turn records:
exact normalized-text on each user turn; near-dup at cos ≥ 0.98 on anchor
embeddings before metric computation. Within-conversation duplicates (user
repeats themselves) are NOT collapsed — they're informative for HP3.

### 8.3 Embedding choice and confound

Unchanged from v2.2 §8.3: `all-MiniLM-L6-v2`, L2-normalized. The confound is
the same — MiniLM cosine doesn't track substitutability, but for prefetch the
relevant property is *prediction accuracy* (does our retriever surface the
right neighbor), and we test two retrievers explicitly to see if a different
similarity helps.

### 8.4 Hit@K@T_pred metric

For each pair (turn_N, turn_{N+1}) and retriever R:

1. Find top-K nearest turn_N* in OTHER conversations under R's similarity.
2. Pull their turn_{N+1}* (the responses you would have pre-generated).
3. Compute MiniLM cosine between actual turn_{N+1} and each candidate.
4. Hit = max cosine ≥ T_pred.

Reported as a (R × K × T_pred) cube. Lift = retriever's rate − random's rate.
HP1 passes iff *some* (R, K, T_pred) cell has lift ≥ 10pp.

### 8.5 Quality at serve (HP2)

For pairs where hit fires at T_serve ≥ 0.95 (the candidate landed
near-duplicate to actual): for a stratified sample of ≥300 such pairs, ask
the judge whether the cached response (the one written for the near-duplicate
prompt) would be acceptable for the actual prompt. Acceptability rate at
T_serve = (ACCEPTABLE + 0.5·BORDERLINE) / total. HP2 passes iff this rate
≥ 60% at the best feasible T_serve.

This is similar to v2.2 H5 but with two critical differences:
- Computed *only* on hits, not across the full cosine distribution.
- The T_serve is chosen by the system designer; we report the
  acceptability-vs-T_serve curve and HP2 passes if any feasible T_serve point
  has rate ≥ 60%.

100-pair human spot-check still owed (PRD §8.5 of v2.2 discipline) before
the result is publication-grade.

### 8.6 Pareto sweep (HP4)

Grid: K ∈ {1, 3, 5, 10, 25} × T_pred ∈ {0.7, 0.8, 0.9, 0.95} × T_serve ∈
{0.90, 0.95, 0.98} × c_speculate ∈ {0.20, 0.50, 1.00}. For each cell:

```
hit_rate     = hit@K@T_pred (from §8.4)
quality      = acceptability at T_serve (from §8.5)
effective    = hit_rate × quality              # net useful prefetch
roi_per_turn = effective × latency_value_per_hit − K × c_speculate
```

Output: a CSV / DataFrame of every cell + `roi_per_turn`. HP4 = "exists cell
with `roi_per_turn > 0`."

---

## 9. Decision procedure

1. Compute HP1 → fail = Abandon (Document; no re-scope).
2. Compute HP2 → fail = Abandon (Document; serve threshold inherently unsafe).
3. Compute HP3 in parallel; record.
4. Compute HP4 Pareto sweep.
5. Apply §4 matrix.
6. Record full decision in `docs/decision_log.md`.
7. If Pass, draft `docs/architecture_v3.md`.

A failed HP1 or HP2 cannot be re-scoped via threshold tuning — they're
*calibrated*, not asserted. A failed HP4 with passed HP1+HP2 reduces to the
within-session-only branch via HP3.

---

## 10. Deliverables

| Artifact | Format | Location |
| --- | --- | --- |
| Frozen cost-model | Markdown | `docs/cost_model_v3.md` |
| Headline numbers | JSON | `results/prefetch_latest/headline_numbers.json` |
| Pareto sweep | CSV | `results/prefetch_latest/pareto.csv` |
| Figures | PNG | `results/prefetch_latest/figures/` |
| Judge transcripts (HP2) | JSONL | `results/prefetch_latest/judge_transcripts.jsonl` |
| Auto-summary | Markdown | `results/prefetch_latest/summary.md` |
| Findings | Markdown | `docs/findings_v3.md` |
| Architecture (conditional) | Markdown | `docs/architecture_v3.md` |

---

## 11. Timeline (1 week, same discipline as v2.2)

| Day | Task |
| --- | --- |
| 1 | Freeze cost-model-v3 (`latency_value_per_hit`, `c_speculate`); set up prefetch module |
| 2 | Pilot on N=5,000 conversations; debug pipeline; N-stability check |
| 3 | Full N=50,000 multi-turn run (no judge); compute HP1, HP3, hit table |
| 4 | Judge run for HP2 on T_serve sample (~300 high-cos pairs); 100-pair human spot-check |
| 5 | Pareto sweep (HP4); apply §4; decision log entry |
| 6–7 | Architecture draft if Pass; findings_v3.md if Abandon / re-scope |

---

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Cross-user prediction is fake (HP1 fails) | Medium | High | Multi-turn data answers it cleanly; same-conv fallback (HP3) |
| Quality at T_serve too low (HP2 fails) | Medium | High | Calibrated, not asserted; if no T_serve clears 60% the verdict is honest |
| Compute budget infeasible (HP4 fails) | Medium | Medium | Pareto exposes feasibility region; re-scope path is built into §4 |
| Multi-turn data is sparse / LMSYS conversations short | Medium | Medium | Pilot reveals; fall back to WildChat multi-turn or constrained-domain dataset |
| Author re-pivots before measuring | High | Critical | Same risk as v2.2; pre-committed rubric, same discipline |
| Latency value chosen post-hoc to fit results | Medium | High | Must freeze in `cost_model_v3.md` BEFORE Day 3, same as v2.2 §4a |
| Within-session-only result is over-claimed as general | Medium | Medium | Decision log explicitly distinguishes; HP3-only branch is narrower scope |

---

## 13. Out of scope

- **Cost-reduction caching.** Settled by v2.2; not re-litigated.
- **Cross-language prefetch.** English only.
- **Cross-provider model comparison.** One model per role at a time.
- **Online learning / continual fine-tuning of the predictor.** Out of scope;
  predictor is a fixed retriever (MiniLM or TF-IDF) for now.
- **Token-level speculative decoding.** Different problem; already solved by
  Anthropic/OpenAI/etc. internally.
- **System implementation.** This study is the precondition.

---

## 14. Next steps after this PRD

1. Freeze `latency_value_per_hit` and `c_speculate` in `docs/cost_model_v3.md`.
   This is yours to set; defaults are placeholders.
2. Implement `src/redundancy/prefetch.py` and `cost_model_v3.py`. Reuse the
   v2.2 `multi_turn.py` module wholesale.
3. Run the pilot at N=5,000. The multi-turn run *already in progress* (v2.2
   §15 investigation) is essentially this pilot — re-purpose its output.
4. Run HP2 judge on the high-cosine hits.
5. Pareto sweep + decision.
6. Either `docs/architecture_v3.md` (positive) or `docs/findings_v3.md`
   (negative / re-scope).

Do not skip steps. Do not start the architecture before the Pareto is computed.
Do not freeze thresholds after seeing results. Do not declare HP2 calibrated
without the human spot-check from §8.5 of v2.2 discipline.

---

## 15. What carries over from v2.2 (and what doesn't)

| v2.2 artifact | Status in v3 |
| --- | --- |
| H1 / H3 / H4 first-turn results | Useful as descriptive background; not decision-bearing |
| H5 substitutability result (12.8% / 15.3%) | **Provides T_serve calibration** — at cos ≥ 0.95, judged acceptability is 12.8–15.3%. To get HP2 ≥ 60%, T_serve must be ≥ ~0.97 *or* the prefetch must add a quality guard beyond MiniLM cosine |
| Dedup (§8.6) | Carried over as-is |
| H1 degeneracy guard | Available; may not fire on multi-turn data, but kept |
| Cost model arithmetic (§4a) | Reused structurally; new constants |
| Inter-LLM sensitivity check | Pattern carries over; can be invoked on HP2 transcripts |
| Audit module (PRD §8.5) | Re-used for HP2 human spot-check |
| LMSYS dataset + access | Same |
| `--filter strict\|recall` | Available; multi-turn extraction may need its own filter calibration |
| `Abandon` decision on cost-cache | **Standing.** v3 doesn't relitigate it. |

The two studies are **independent answers to independent questions**, both
using the same dataset and pipeline infrastructure. v3 is not a retry of
v2.2; it's the study v1 should have been if the underlying goal had been
named correctly at the start.
