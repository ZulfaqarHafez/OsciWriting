# Personal Prompt Prefetch Study (PRD v4)

**Status:** Pre-experiment, design locked (v4)
**Author:** Zul Fhagez
**Last updated:** 2026-05-21
**Decision deadline:** within 1 week of data collection completion

---

## 0. Changelog (v3 → v4)

v2.2 (cost-cache) and v3 (cross-user prefetch) both reached Abandon on LMSYS
open-domain prompts for the same underlying reason: **embedding-similar prompts
across users don't carry substitutable responses (12.8–20% acceptability vs the
60–70% required) and next-prompt prediction across users is weak (+5.7 pp lift
vs the +10 pp required).** Two independent mechanisms, one root cause.

v4 bends the root cause. Single change: **the predictor and the cache are
per-user, drawn from one user's own multi-session history**. The Gmail
analogy made the prior framing literal: "when you log into Gmail it prefetches
*YOUR* inbox, not random inboxes." v2.2/v3 were prefetching random inboxes.

Specific changes from v3:

1. **Cross-user → per-user.** All matching is within one user's conversation
   history. Cross-user generalization is not measured.
2. **Multi-turn → multi-session.** v3 measured turn N → turn N+1 within a
   single conversation. v4 measures conversation N → conversation N+1 (or any
   future conversation) for the same user.
3. **Generalist subject → recurring-task subject.** v4 cares about the *fraction
   of a user's prompts that are recurring tasks* (HP1), not the cross-user
   coverage of the prompt space.
4. **HP4 ROI math.** Same shape as v3 §4a, but `c_speculate` for a single user's
   own predictor is much smaller (lookup over their own embeddings, no
   cross-user inference). The bar moves.
5. **Quality threshold (HP2) UNCHANGED at ≥60%.** This is the substitutability
   bar regardless of mechanism — quality is quality. v4 just predicts the same
   user will accept their previously-generated response, which v3 strongly
   suggests is more likely than cross-user.
6. **New cold-start hypothesis (HP3).** v3 didn't have one because cross-user
   is its own cold-start. Per-user has a clear cold-start regime (user with
   <5 conversations) that needs measuring.

---

## 1. Purpose

Determine whether per-user, multi-session prefetch is feasible: when a user
sends a new prompt, the system retrieves from THIS USER'S history and serves
a previously-generated response if the new prompt matches a recurring pattern
in their behavior.

The mechanism is the literal Gmail prefetch translation:

```
User opens chat / starts typing
  → system looks up THIS USER'S recurring prompts in their history
  → predicts which is most likely now (time-of-day, recency, pattern)
  → pre-warms KV cache and/or pre-generates top-K candidates
On actual prompt arrival:
  → if matches a candidate at T_serve → instant from cache
  → else → on-demand frontier call (no degradation)
```

v4 is the precondition study for a personal-assistant prefetch product. The
study is not the product.

---

## 2. Background

### Why per-user changes the math

Three properties that fail cross-user plausibly hold within-user:

| Property | Cross-user (v3 measured) | Per-user (v4 hypothesis) | Why |
| --- | --- | --- | --- |
| Predictability | +5.7 pp over random | +20–50 pp expected | Same person → recurring tasks (standup, summaries, debugging) at high frequency |
| Substitutability | 13–20% | 50–80% expected | Same preferences, same style, same context → previously-acceptable answers transfer |
| Within-session continuation | p50 = 0.325 | p50 = 0.5+ expected | Within one user's career of chats, *some* are clearly continuations of prior tasks |

These are predictions, not measurements yet. v4 tests them.

### Adjacent prior art

- **Anthropic prompt caching** (KV reuse): cheap, deployed, doesn't predict.
- **GitHub Copilot personalization**: per-developer style adaptation; closest
  analog to v4 in spirit, different mechanism (model fine-tuning vs cache).
- **Recommendation systems** with per-user models: well-established that
  per-user signal dominates cross-user for behavior prediction.
- v3 finding: cross-user prediction has *some* signal (+5.7 pp ≠ 0) but below
  threshold. This isn't "zero signal"; per-user should be a multiple of it.

---

## 3. Hypotheses

For one or more individual users with sufficient history, on their own
multi-session prompt log:

**HP1 (recurring-task fraction).** ≥ **30%** of a user's conversation-start
prompts have at least one prior conversation-start prompt with MiniLM cosine
≥ 0.85 in their own history. This is "how much of what you ask is recognizably
something you've asked before." If under 30%, prefetch can't be ambient — too
rare to feel like Gmail.

**HP2 (per-user substitutability).** For a stratified sample of pairs where
the user's new prompt matches a prior prompt at cosine ≥ 0.85, judge-rated
acceptability of the prior conversation's response for the new prompt is ≥
**60%** at some serve threshold T_serve ∈ {0.85, 0.90, 0.95}. Calibrated, not
asserted. Carries over the v3 HP2 quality bar — quality requirements don't
change because the mechanism is personal.

**HP3 (cold-start coverage).** For users with ≤ 5 prior conversations, hit@K=5
at T=0.85 is ≥ **15%**. Determines whether v4 only helps power users or also
helps newcomers — if cold-start hit rate is near zero, the product needs a
warm-up phase before it earns its keep.

**HP4 (per-user ROI).** Given v3-frozen `latency_value_per_hit` and a much-
reduced per-user `c_speculate` (own-history lookup is cheap; the speculation
cost is just the LLM gen for the predicted candidate, plus an embedding
lookup ≈ 0.0001), there exists (K, T_pred, T_serve) with `roi_per_turn > 0`.

---

## 4. Decision rubric

| HP1 | HP2 | HP3 | HP4 | Outcome | Action |
| --- | --- | --- | --- | --- | --- |
| Fail | any | any | any | **Abandon** | Recurring-task fraction too low; user behavior is too exploratory for prefetch to feel ambient. No re-scope. |
| Pass | Fail | any | any | **Abandon** | Even within-user, response substitution fails; serving from cache would degrade quality. This kills the entire prefetch idea, since v3 already killed cross-user. |
| Pass | Pass | Fail | Pass | **Commit, power-user-only** | Build v4 product for users with ≥30 conversations. Cold-start path is separate problem (caching disabled for new users). |
| Pass | Pass | Pass | Fail | **Re-scope to KV-warm-only** | Predictability and quality hold but full-response prefetch isn't economic. Fall back to prompt-caching / KV-prewarming (Direction C from the v3 → v4 framing). |
| Pass | Pass | Pass | Pass | **Commit to project** | Build personal prefetch for all users, with cold-start handled via in-session learning. |
| Pass | Pass | Fail | Fail | **Marginal: try constrained domain** | Per-user signal real but neither cold-start nor general ROI works. Re-scope to a constrained user-domain combination. |

HP1 is the entry gate. HP2 is the quality gate. HP3 and HP4 inform scope, not
viability per se.

### 4a. Cost / value model

Reuses v3 `cost_model_v3.py` structure with per-user adjustments:

```
c_speculate_personal = c_embed_lookup + c_gen
                     = 0.0001 + c_speculate_v3   (essentially same as v3)

# But effective K is smaller for personal: we don't speculate the universe of
# possible follow-ups, only the top-K from this user's history.
# K typically 1-3 (your last few common recurring prompts).
```

The big change is the EXPECTED hit rate (now 30%+ instead of 7%) which makes
the ROI inequality much more comfortable. Concretely: with v3's
`latency_value_per_hit = 0.30`, hit_rate = 0.30, quality = 0.60, K = 1,
c_speculate = 0.20:

```
roi = 0.30 × 0.60 × 0.30 − 1 × 0.20 = 0.054 − 0.20 = −0.146
```

Still negative. So `latency_value_per_hit` needs to be ≥ ~1.1 for ROI to clear
zero — i.e., the saved latency must be worth more than the cost of one Haiku
call. That's a defensible product claim for a high-touch personal assistant
("instant on common tasks is worth ~$0.005 of perceived speed-up"). Author
must set `latency_value_per_hit` realistically in `docs/cost_model_v4.md` and
freeze before Day 3 — same v2.2/v3 discipline.

---

## 5. Repository structure

Builds on v3. New / modified:

```
src/redundancy/
  user_history.py       # NEW: load + structure a single user's multi-session export
  personal_prefetch.py  # NEW: per-user predictor + HP1–HP4 evaluation
  cost_model_v4.py      # NEW: per-user ROI math (mostly inherits v3)

results/
  personal_<USER_TAG>_<TS>/
    config.json
    headline_numbers.json
    pareto.csv
    recurring_clusters.md   # which user prompts recur, with examples
    figures/
      cosine_distribution.png
      hit_curve.png
    summary.md

docs/
  cost_model_v4.md      # frozen latency_value + per-user constants
  findings_v4.md        # post-run write-up
  architecture_v4.md    # conditional on Pass
```

`<USER_TAG>` is a non-identifying label (e.g., "personal", "user1"). Multi-
user studies are out of scope for v4 — one user, one run, honest scope.

---

## 6. Environment setup

Same Python 3.11+, same v2.2/v3 dependencies. The judge (Haiku 4.5) is needed
for HP2 only. New: a user's chat export.

**Data input formats supported:**
- ChatGPT export (`conversations.json` from chat.openai.com export).
- Claude.ai export (similar structure).
- Generic JSONL with one conversation per line (documented schema).

`user_history.py` provides loaders for each, all normalized to the same
`Conversation` schema used by v3.

---

## 7. Data sources

**Primary: user's own ChatGPT or Claude export.** Sample size = the user's
total conversations (typically 50–500 for an engaged power user). No external
gating, no NSFW concentration, no template-batch contamination. Sample is
biased (n=1 user) and the result is "feasibility for this user" — but that's
the honest claim.

**Fallback (not recommended): LMSYS-IP-clustered.** Group LMSYS conversations
by source IP + day to approximate "same user, multiple sessions." Probably
yields ~10–30 conversations per cluster, possibly. Distribution is noisy,
IPs are NATed, but it's available if the user export isn't.

**N requirement**: ≥ 30 conversations for HP1 measurement; ≥ 5 for HP3
cold-start measurement. Below 30, fall back to descriptive analysis only.

---

## 8. Methodology

### 8.1 Pipeline stages

```
[user's chat export]
  -> parse into Conversation list           (user_history.load_export)
  -> chronological sort by timestamp        (preserve causality)
  -> exact dedup on normalized prompts      (carry over §8.6 of v2.2)
  -> embed all conversation-start prompts   (embed.embed)
  -> for each conversation i, look at conversations [0..i-1]:
       find max cosine to prior prompt      (the within-user kNN)
  -> distribution of max-cosine: HP1        (frac at cosine ≥ 0.85)
  -> for high-cosine pairs:
       judge response acceptability         (HP2; ~50–100 calls)
  -> cold-start subset: convs 0-4 of user   (HP3)
  -> Pareto sweep over (K, T_pred, T_serve) (HP4)
  -> apply §4 rubric → outcome             (personal_prefetch.decide)
  -> write report                          (personal_prefetch.report)
```

### 8.2 Recurring-pattern definition

A prompt p_i "recurs" if there exists j < i with cosine(p_i, p_j) ≥ 0.85 in
MiniLM embedding space. The 0.85 threshold is *lower* than v2.2/v3's 0.95
because:
- Within-user prompts vary less in surface form than cross-user.
- The substitutability hypothesis (HP2) gates serve-quality regardless of
  T_pred, so we can be permissive at the retrieval step.
- Empirically, the recurring-tasks people care about (summaries, translations,
  debugging) sit at 0.7–0.95 within a single user's phrasing variance.

### 8.3 Quality at serve (HP2)

For pairs (p_i, p_j) with cosine ≥ 0.85, ask the judge: would the response
to p_j (the user's actual prior response that they presumably accepted, since
they didn't re-ask) be acceptable for the new prompt p_i? Same prompt
template as v3 HP2, same scoring scheme. Sample ≥ 100 pairs across cosine
bands 0.85/0.90/0.95 (smaller than v2.2/v3 because the universe is smaller —
maybe 100–300 high-cosine pairs total exist for a typical user).

### 8.4 Cold-start (HP3)

Subset to the user's first 5 conversations. For each, check whether *anything*
in the prior 0–4 conversations has cosine ≥ 0.85. The expected rate is much
lower (no history → no matches). Report the curve.

### 8.5 Pareto sweep (HP4)

Same shape as v3 §8.6 with `c_speculate_personal` ≈ v3's c_speculate (Haiku
cost) + tiny lookup cost. K grid: {1, 3, 5}. T_pred grid: {0.85, 0.90, 0.95}.
T_serve grid: {0.85, 0.90, 0.95}. Output Pareto CSV; best feasible cell.

---

## 9. Decision procedure

1. Load + dedup user history.
2. Compute HP1 (recurring fraction). Fail → **Abandon, no re-scope**.
3. Compute HP3 (cold-start) in parallel.
4. Run HP2 judge. Fail → **Abandon, no re-scope**.
5. Compute HP4 Pareto sweep.
6. Apply §4 matrix.
7. Record in `docs/decision_log.md`.
8. Write `findings_v4.md`.

Same v2.2/v3 discipline: thresholds frozen pre-data; rubric mechanical; no
post-hoc re-litigation.

---

## 10. Deliverables

| Artifact | Format | Location |
| --- | --- | --- |
| Frozen cost-model | Markdown | `docs/cost_model_v4.md` |
| Headline numbers | JSON | `results/personal_<tag>_latest/headline_numbers.json` |
| Recurring-pattern examples | Markdown | `results/personal_<tag>_latest/recurring_clusters.md` |
| Pareto sweep | CSV | `results/personal_<tag>_latest/pareto.csv` |
| Figures | PNG | `results/personal_<tag>_latest/figures/` |
| Judge transcripts | JSONL | `results/personal_<tag>_latest/judge_transcripts.jsonl` |
| Findings | Markdown | `docs/findings_v4.md` |
| Architecture (conditional) | Markdown | `docs/architecture_v4.md` |

---

## 11. Timeline (1 week — same discipline)

| Day | Task |
| --- | --- |
| 1 | Freeze cost-model-v4; collect user's chat export; implement `user_history.py` |
| 2 | Pilot run: HP1 + HP3 (no judge); inspect recurring-pattern examples |
| 3 | HP2 judge run (~100–300 calls); full Pareto sweep |
| 4 | Decision per §4; write findings_v4.md |
| 5–7 | If Pass, draft architecture; if Abandon, document |

Much faster than v2/v3 because there's no big dataset download; user data is
local and small.

---

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| n=1 user, weak generalization | Certain | Medium | Honest scope: "feasibility for this user, suggestive for similar users" |
| User's history too short | Medium | High | Fall back to descriptive-only; no rubric verdict |
| User's history is exploratory (no recurring) → HP1 fails | Medium | High | An honest Abandon. The product is for power users with workflows, not explorers |
| HP2 still fails within-user | Low | High | Means substitutability is fundamental, not cross-user-specific. Strong negative across all three studies |
| HP3 fails → cold-start gap | Medium | Medium | Product can ship with cold-start disabled; addressed by in-session learning later |
| Author re-pivots before measuring | High | Critical | Same risk every study. Pre-committed thresholds |
| User's export doesn't load / format unsupported | Medium | Low | Document supported formats; provide a generic JSONL converter |

---

## 13. Out of scope

- **Cross-user generalization claims.** v4 is per-user; multi-user is a future
  study.
- **Online learning / continual model fine-tuning.** Predictor is a fixed
  retriever (MiniLM) over the user's static history snapshot.
- **Privacy infrastructure.** This is a research feasibility study; not a
  product deployment with proper data handling.
- **Cold-start prediction *mechanisms*.** HP3 *measures* the cold-start gap but
  doesn't solve it. Solutions (e.g., bootstrap from cross-user) are future
  work.
- **Multi-modal prompts.** Text only.

---

## 14. Next steps after this PRD

1. **You decide / provide the user export.** Without data this study is
   notional. Options in §7. If you have a ChatGPT/Claude export, drop it
   under `data/user_<tag>/`. Or pick a fallback strategy.
2. Freeze `latency_value_per_hit` (yours to set; needs to be defensibly ≥ 1.1
   per §4a or HP4 mathematically can't pass).
3. Implement `user_history.py` + `personal_prefetch.py` + `cost_model_v4.py`.
4. Run the pipeline. Apply rubric.
5. Document.

Do not skip steps. Do not freeze thresholds after seeing the data. Do not
re-scope on a failed HP1 or HP2 — these are quality/viability gates, not
parameters.

---

## 15. What carries over from v2.2 + v3 (and what doesn't)

| v2.2/v3 artifact | Status in v4 |
| --- | --- |
| Embedding (MiniLM) + dedup | Reused as-is |
| Judge (Haiku 4.5) + audit module | Reused for HP2 |
| Cost model arithmetic structure | Inherited; new constants |
| H1 degeneracy guard (v2.2 §8.4) | Available, probably won't fire — per-user has small N |
| TF-IDF "graph similarity" arm | **Dropped.** v3 showed it doesn't beat MiniLM. Stick with one retriever for v4 |
| LMSYS dataset | **Replaced** by user's export. LMSYS-IP-clustering is the (worse) fallback |
| Findings (v2.2, v3) | **Standing**, independent. v4 is a third study on a different question and different data |
| Audit discipline (PRD §8.5) | Carries over; still owed for v2.2 first-turn data |

v4 is the most likely study to produce a positive result *because the prior
two negative results gave us a clear understanding of where the bottleneck
lives* (cross-user open-domain). Honest scoping > overclaiming.

If v4 also Abandons — i.e., HP1 or HP2 fails even within-user — then the
prefetch idea is dead across all three reasonable mechanisms, and the next
move is Direction C (KV-warm-only, no response prefetch) or to abandon the
project entirely.
