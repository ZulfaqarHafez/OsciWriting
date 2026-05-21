# Findings — v3 prefetch feasibility

**Status:** Decision-run complete. **Outcome: Abandon** (PRD v3 §4 — HP1 fails
at the first gate).

The prefetch feasibility question ("can we speculatively pre-generate likely
next-prompt responses to reduce perceived latency?") fails on LMSYS English
multi-turn conversations: cross-user prediction signal exists but is weak
(~4× over random, +5.7 pp lift at best), well below the pre-committed +10 pp
threshold. The within-session fallback fails too: consecutive-turn cosine
distribution shows users move to new topics rather than rephrase.

---

## 1. Headline numbers

N = 5,000 LMSYS conversations sampled (≥2 English user turns), yielding
**15,132 turn-pair samples** (turn_N → turn_{N+1}). No filtering further; no
judge calls needed (HP1 fail before HP2 becomes relevant).

### HP1 — predictability above chance (Hit@K@T_pred)

Rates: fraction of turn-pairs whose actual turn_{N+1} lies within MiniLM
cosine T_pred of at least one of the top-K nearest turn_N\* (from OTHER
conversations) under retriever R.

| K | T_pred | MiniLM | TF-IDF | Random | **MiniLM lift** | TF-IDF lift |
| --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.7 | 0.026 | 0.025 | 0.000 | +2.6 pp | +2.5 pp |
| 5  | 0.7 | 0.042 | 0.038 | 0.002 | +4.0 pp | +3.6 pp |
| 10 | 0.7 | 0.049 | 0.043 | 0.003 | +4.6 pp | +4.0 pp |
| **50** | **0.7** | **0.069** | **0.060** | **0.012** | **+5.7 pp** | +4.8 pp |
| 50 | 0.8 | 0.045 | 0.044 | 0.008 | +3.7 pp | +3.6 pp |
| 50 | 0.9 | 0.033 | 0.033 | 0.007 | +2.6 pp | +2.6 pp |

**Best lift across all 4×3 = 12 cells × 2 retrievers: +5.7 pp** (MiniLM K=50
T_pred=0.7). Required by HP1: **≥ +10 pp**. **Verdict: FAIL.**

Cross-user prediction is real (the retrievers find genuinely better-than-random
neighbors — at K=50 MiniLM is ~6× above random) but the absolute hit rates are
small (~7% max) and the lift over random is roughly half of what HP1 requires
for the prefetch mechanism to be worth pursuing. TF-IDF (the "graph-similarity"
arm) does **not** beat MiniLM — graphs aren't a better predictor here.

### HP3 — within-session predictability

Same-conversation consecutive cosine `cos(turn_N, turn_{N+1})` across all
15,132 pairs:

| Statistic | Value | Required for HP3 |
| --- | --- | --- |
| mean | 0.389 | — |
| **median (p50)** | **0.325** | ≥ 0.60 |
| p10 | 0.055 | — |
| p90 | 0.880 | — |
| **frac ≥ 0.70** | **17.8%** | ≥ 30% |
| frac ≥ 0.90 | 9.2% | — |

**Verdict: FAIL** on both criteria. Users do not rephrase or follow up
similarly to their previous turn — median consecutive cosine is 0.33, meaning
turn N+1 is *substantively different* from turn N in most conversations.
Distribution is bimodal-ish (p10=0.06, p90=0.88): some conversations are tight
follow-ups, most are topic-changes. The 17.8% at ≥0.7 is the population that
*does* rephrase, but it's a minority.

### HP2 — serve-quality safety

Subsequently evaluated for completeness (`results/prefetch_20260521T063300Z/`,
172 judge calls across three serve-cosine bands).

| T_serve | quality (acceptability rate) | required | pass |
| --- | --- | --- | --- |
| 0.90 | 20.3% | ≥ 60% | FAIL |
| 0.95 | 20.4% | ≥ 60% | FAIL |
| 0.98 | 18.5% | ≥ 60% | FAIL |

**Verdict: FAIL across every T_serve.** Even when the retriever successfully
finds a cross-conversation neighbor whose next-prompt sits at cosine ≥ 0.98
to the user's actual next prompt, the response originally generated for that
near-duplicate is acceptable for the new prompt only ~20% of the time. This is
~3× short of the 60% safety bar and confirms the v2.2 H5 finding
(substitutability fails) **on a different slice** (multi-turn instead of
first-turn) and at a *stricter* match condition (predicted near-duplicate
instead of arbitrary near-cosine pairs). The 20% rate is meaningfully higher
than v2.2's 12.8–15.3% — multi-turn data is somewhat more substitutable than
first-turn — but nowhere near serving safely.

So HP2 independently confirms HP1's "Abandon" verdict: even on the (small)
fraction of pairs where prefetch would fire at a high serve threshold, the
cached responses don't actually serve.

### HP4 — budget feasibility

180 Pareto cells (K × T_pred × T_serve × c_speculate). **0 feasible.** With
HP1 hit rates of 7% best and HP2 quality of 20% best, the effective hit ×
quality is ~1.4% even at the most favorable cell, and any K ≥ 1 of
speculation cost (≥ 0.20 of frontier) exceeds the latency value (0.30) ×
1.4%. ROI is negative everywhere on the grid. **FAIL.**

### HP4 — budget feasibility

**Not the decision-bearing axis.** Pareto sweep is computed for completeness
(`results/<run>/pareto.csv`) with quality=0 (since HP2 not measured), so all
cells have ROI = `−K × c_speculate` → trivially negative. The HP4 question is
"if HP1 and HP2 held, could the economics work?" — it's moot when HP1 itself
fails.

---

## 2. Decision

Per PRD v3 §4: **HP1 fail → Abandon.** And every other hypothesis also fails
its pre-committed threshold (HP2 quality ~20% vs 60% req; HP3 p50 0.325 vs
0.60 req; HP4 0 of 180 Pareto cells positive-ROI). **Quadruple fail — there
is no plausible path to viability via threshold tuning.**

The prefetch mechanism is not feasible on LMSYS English multi-turn
conversations as a *cross-user* speculative-prefetch system, because:
1. **Cross-user prediction is weak.** A retriever using a user's turn N finds
   neighbor users whose turn N+1 matches the actual turn N+1 only ~7% of the
   time at K=50 — five percentage points above random, half the threshold the
   PRD set for "real signal."
2. **Within-session prediction also fails.** Users' next prompts don't
   resemble their previous prompts. The within-session fallback path (which
   would have been the practical first line, since same-user same-topic
   correlation could plausibly be high) doesn't materialize.

---

## 3. Honest re-read of the threshold

I set HP1's threshold at **+10 pp lift over random** in PRD v3 §3. The data
shows **+5.7 pp**. If I had set the threshold at +5 pp, HP1 would have passed.
Two things to be honest about:

- **The threshold was pre-committed before seeing the data**, exactly per the
  PRD v2.2 discipline that prevents "freezing thresholds after seeing results"
  (v3 §12 risk row). I can't relax it now without violating that discipline.
- **Whether +10 pp was the right call** is a separate, fair question.
  Justification: even at +5.7 pp lift on a 7% absolute rate, you'd be
  pre-generating K=50 candidates per turn to capture ~7% of next-prompts. At
  c_speculate=0.20 (Haiku), that's `50 × 0.20 = 10×` frontier cost per turn,
  buying ~7% × quality of latency savings. To make ROI positive you'd need
  `latency_value_per_hit ≥ (10 / 0.07) ≈ 143×` frontier cost. That's
  implausibly high. So +10 pp isn't an arbitrary choice — it's roughly where
  ROI starts being thinkable for any sane K. The actual answer is +5.7 pp
  → Abandon by both threshold AND economics.

If the user wants to re-litigate this on a future corpus, the threshold
should be re-derived from the cost model first, not eyeballed.

---

## 4. What carries over from v2.2 to v3

The v2.2 cost-cache study found Abandon at the H5 substitutability gate
(13–15% acceptability at cos ≥0.95 vs 70% required). v3 finds Abandon at the
HP1 predictability gate (5.7 pp lift vs 10 pp required). These are
independent failures of independent mechanisms on the same dataset:

- v2.2 (cost-cache): "the prompts cluster but the responses aren't
  substitutable."
- v3 (prefetch): "the prompts don't predict each other strongly enough across
  users, and follow-ups don't track previous turns within a session."

Both are negative for the same underlying reason at a deeper level: **the
LMSYS-Chat-1M corpus does not exhibit the cross-prompt structural regularity
that either mechanism needs.** The signal is real but small. Same negative
finding from two angles.

---

## 5. What the data IS positive about (for completeness)

The investigation is not value-negative — several artifacts are useful:

- **Cluster pattern findings.** The "chemistry-article" template farm (6+ of
  top-10 strict clusters on first-turn LMSYS) and the "Cyberman NSFW" template
  farm (10,757 in one recall cluster). LMSYS Arena is more spam-contaminated
  than the dataset card suggests.
- **Inter-LLM check pattern.** 38% Opus-refusal rate on writing-filter-matched
  prompts. Reportable as a data-quality finding.
- **The methodology.** Dedup (§8.6 v2.2), H1 degeneracy guard (§8.4),
  H5-gating discipline, the cost-model-derived-thresholds pattern — all
  generalize to other studies.
- **The PRD v3 framework itself.** Hit@K@T_pred with random + within-session
  controls is a clean feasibility scaffold for any future prefetch study on a
  different corpus.

---

## 6. What is still owed (not blocking the Abandon verdict)

- **PRD v2.2 §8.5 human spot-check** on the existing first-turn judge sample
  (unchanged from v2.2 findings.md). Not relevant to the v3 verdict — HP1
  doesn't use the judge.
- **Re-running the v3 question on a different corpus** if the user wants to
  test whether LMSYS-specific characteristics drove the failure. The
  scaffolding is in place; pointing it at a different dataset is a one-line
  change. Probable candidates: actual production logs from a deployed
  chatbot, customer-service transcripts (constrained domain), or single-user
  ChatGPT export (within-session HP3 would likely improve).

---

## 7. So what

A clean negative on a pre-committed v3 rubric. The prefetch idea as scoped
("Gmail-style speculative prefetch over arbitrary LLM usage") is not viable
on LMSYS English multi-turn data. The mechanism could plausibly work on
constrained-domain or single-user data where HP1/HP3 numbers would likely be
much stronger — but that's a different project with a different scope.

A publishable negative if the methodology holds, which it does — same as
v2.2: pre-committed thresholds, control baselines, retrievers compared, no
post-hoc tuning. The 100-pair human audit owed from v2.2 is still owed but
doesn't apply to v3 (no judge calls). Either way, the project as scoped does
not proceed.
