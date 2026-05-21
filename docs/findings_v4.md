# Findings — v4 personal prefetch feasibility

**Status:** Decision-run complete. **Outcome: Abandon** (PRD v4 §4 — HP1 fails
at the first gate).

The personal/per-user version of the prefetch idea fails on the author's
actual Claude.ai chat history: recurring-task fraction is **7.4%** of
conversations vs the pre-committed ≥ 30% required. The median cosine between
any conversation-start prompt and its closest prior in the same history is
**0.40** — that's "loosely related topics," not "same task again." Three
prefetch/cache studies have now each Abandoned at their first gate (v2.2 H5,
v3 HP1 cross-user, v4 HP1 within-user); the cross-prompt structural
regularity that any version of this idea needs is not present in this data.

---

## 1. Headline numbers

Input: author's Claude.ai export (`data-747...zip`), 159 raw conversations →
**148 non-empty → 136 with at least one paired user/assistant turn**.
Chronological order preserved (oldest first).

| Hypothesis | Value | Threshold | Pass |
| --- | --- | --- | --- |
| HP1 recurring fraction (cos ≥ 0.85) | **7.4%** (10 / 135 eligible) | ≥ 30% | **FAIL** |
| HP1 at relaxed cos ≥ 0.70 | 14.8% | — (descriptive) | — |
| HP3 cold-start hit-rate (first 5 conv) | 20.0% | ≥ 15% | PASS (n=5; noise) |
| HP4 Pareto cells positive ROI | 0 of 12 (K=1 only) | ≥ 1 | FAIL |
| HP2 quality | NOT EVALUATED | — | gate fired at HP1 |

Per PRD v4 §9 step 2: failed HP1 → Abandon, no re-scope, no further judge
calls. HP3 passing is on n=5 and is too noisy to claim.

### Cosine distribution (current prompt → best prior)

```
n          = 135
mean       = 0.46
p10        = 0.23
median p50 = 0.40   (PRD v4 expectation: ≥ 0.50)
p90        = 0.78
frac ≥ 0.70 = 14.8%
frac ≥ 0.85 =  7.4%
frac ≥ 0.95 =  5.9%
```

The distribution centers around 0.40 — typical of *new topics with vocabulary
overlap*, not recurring tasks. p90 = 0.78 means even the top tenth of
conversations only weakly resembles its closest historical neighbor.

---

## 2. What the data tells us about this user's actual usage

The 10 conversations that *did* recur cluster into ~3 workflows visible from
their top-cosine matches:

- **Lecture-notes generation** (multiple matches at cos ≈ 0.99): "help me
  generate notes for this lecture / 2 lectures / with charts and visuals."
- **PDF-input processing** (multiple cos 1.000 matches): "understand each pdf
  input and then remove the example and case study to generate …" — same
  exact prompt reused across new conversations.
- **Project / PRD drafting**: "do a project plan / PRD based on all the
  information I have provided."

Several recurring pairs are byte-identical (cos = 1.000) — the same prompt
was sent in new conversations rather than continuing the previous chat.
That's interesting product-usage information: even when this user repeats a
task, they start a fresh conversation each time.

**The shape of this user's Claude usage is ~93% exploration, ~7% workflow.**
For a hypothetical workflow-focused user (e.g., somebody who uses Claude only
for daily lecture-notes, every weekday), HP1 on the same threshold would
plausibly pass. This user's usage doesn't fit that profile.

The descriptive finding is honest and concrete: **personal prefetch could
work for users with workflow-shaped usage; this user has exploration-shaped
usage; HP1 measures the gap quantitatively.**

---

## 3. The three-study consolidation

| Study | Mechanism | Failed at | Magnitude of fail |
| --- | --- | --- | --- |
| v2.2 | Cross-user response caching | H5 substitutability | 12.8–15.3% vs req 70% (~5×) |
| v3 | Cross-user response prefetch | HP1 prediction lift | +5.7 pp vs req 10 pp (~½) |
| **v4** | **Per-user response prefetch** | **HP1 recurring fraction** | **7.4% vs req 30% (~4×)** |

Three independent gates failed by **multiples**, not by tenths. The thresholds
were pre-committed each time (v2.2/v3/v4 all use the v2.2-established
discipline: thresholds frozen before data, rubric mechanical). No mechanism in
the prefetch / response-cache family clears any of its first gates on this
data.

The common cause: real-world LLM usage in the corpora we've examined (LMSYS
open-domain English + this user's Claude history) does not exhibit the
cross-prompt structural regularity any of these mechanisms requires. Real
conversations explore; cached responses don't transfer; users don't repeat
themselves often enough.

---

## 4. What could change the picture

Future studies that *could* produce a non-Abandon outcome, none of which the
current PRD framework already covers:

1. **Workflow-focused user data.** A user whose Claude/ChatGPT usage is
   predominantly recurring tasks (research assistant for a specific
   long-running project, customer service operator, code-reviewer). HP1 on
   their data would plausibly exceed 30%.
2. **Constrained-domain corpus.** A dataset filtered to one task family
   (code-completion logs, IT-support tickets, translation requests) where
   prompt-type concentration is structurally high.
3. **Direction C from the v3→v4 framing**: drop response prefetch entirely,
   measure KV-cache reuse / prompt-caching savings only. Different metric,
   different bar. Anthropic prompt caching already does this as a product
   feature; not novel research.

None of these is a refinement of the current PRD; they're separate studies.

---

## 5. So what

A clean negative on a pre-committed v4 rubric, applied to the author's own
data. The personal-prefetch idea is not feasible *for this user*. The
underlying mechanism could plausibly work for a workflow-focused user — the
study quantifies the gap (7.4% actual vs 30% required, ~4× short) and
identifies what kind of usage profile would make it work.

Combined with v2.2 and v3, the prefetch/cache idea is decisively Abandon
across all three reasonable mechanisms on the data available. The next move
is not another variant of the same study; it's either a different problem
(KV-prewarming as a productized engineering improvement, not a research
question) or a different dataset (workflow-focused user logs).

The PRD discipline held throughout: thresholds set before data, rubric
mechanical, no post-hoc re-litigation. Three Abandons on three pre-committed
rubrics is itself a methodologically defensible outcome.
