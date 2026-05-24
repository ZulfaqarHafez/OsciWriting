# TraceRazor v0.3.1 — Information-Theoretic Token Optimization (Pre-Implementation Hardened)

**Status:** PROPOSAL — implementation spec, hardened from v0.3.0 review feedback
**Author:** Zulfaqar Hafez
**Date:** 2026-05-24
**Supersedes:** v0.3.0 (incorporates the four Tier-1 review fixes plus Tier-2/3 polish)

---

## 0. Changelog (v0.3.0 → v0.3.1)

v0.3.0 was structurally sound — pre-committed gates, falsifiable diagnostics,
honest "doesn't ship" table. Review surfaced four issues that would have made
the gates measure slightly the wrong thing. v0.3.1 fixes them before any code
lands, plus tightens scope on three secondary issues. **No new metrics added,
no metrics removed.** This is purely methodology hardening.

1. **"is_waste" operationally defined + Cohen's κ ceiling measured (§2.1, new).**
   v0.3.0 asserted ρ ≥ 0.50 for H1 without specifying what counts as "waste"
   or measuring inter-annotator agreement. Without κ, ρ has no interpretable
   ceiling — requiring 0.50 absolute could be either trivial or impossible.
2. **H1 threshold rebased on measured perplexity baseline (§2, H1 update).**
   v0.3.0 asserted "perplexity ρ < 0.30" as the existing baseline without
   citation. v0.3.1 requires the baseline be *measured first* on the H1 test
   sample; H1 threshold is then `baseline + 0.20`, not 0.50 absolute.
3. **New H4 (audit economics) as a hard gate (§2, H4 new).** v0.3.0 had this
   as a soft warning (§8.1 "if > 5% of generation cost, warn; if > 20%,
   refuse"). It's the actual decision — if audit costs more than it saves, no
   threshold on ρ matters. Promoted to a pre-committed gate hypothesis with
   the same Abandon-on-fail discipline as H1.
4. **λ₁, λ₂ tuning moved off the H1 test traces (§3.1 update).** v0.3.0 swept
   on the same five benchmark traces used to test H1 — leakage. v0.3.1
   requires sweep on a held-out set distinct from the H1 evaluation sample.

Plus Tier-2 fixes incorporated inline:

5. **Sample size raised from n=5 traces to n≥15** for labelling (§2.1).
6. **Cold-start guard added to CTU** (§3.1) — symmetric to the v0.3.0 CSR
   and Kelly guards.
7. **H2 reframed as a Pareto check** instead of `AND`-gating two metrics
   (§2 H2 update).
8. **H3 judge audit requirement** carried over from the parallel
   redundancy-study v2.2 §8.5 discipline (§3.3 update).

Tier-3 polish:

9. **Timeline 1 week → 2–3 weeks realistic** (§6).
10. **TraceRazor-specific cost model** in `docs/cost_model_v0_3.md`, distinct
    from the redundancy study's `cost_model.py` constants (§3.4 new).
11. **Decision rubric collapsed from 5 outcomes to 3** (§10).
12. **Pheromone reframing of IAR explicitly marked as `refactor:` only** —
    zero behaviour change, docstring-only diff (§5).

---

## 1. Why this version exists (unchanged from v0.3.0)

Three thresholds in TraceRazor are currently picked by feel. v0.3 replaces
them with formulas from established disciplines:

1. **Conditional surprisal** (information theory) → CTU, per-token utility.
2. **Kelly-sized K** (quant finance) → closed-form K per AdaptiveK step.
3. **Context Sharpe ratio** (portfolio theory) → context-level aggregation,
   serves as fix-engine stopping rule.

The motivation, from the redundancy study's three Abandons (v2.2 H5, v3 HP1,
v4 HP1), is that thresholds-by-feel buries what's actually true about the
underlying signal. Same risk lives in TraceRazor every time a fix-engine
threshold gets nudged in PR review. v0.3 makes three core numbers derivable
rather than chosen; v0.3.1 makes the *gates that decide whether v0.3 ships*
also derivable rather than chosen.

---

## 2. Hypotheses (pre-committed, PRD-style)

**Four hypotheses now, not three.** Same discipline: thresholds frozen
before any measurement.

### H1 — Conditional surprisal predicts token waste above perplexity baseline

**Claim:** Per-token conditional surprisal `s(t_i) = −log p(t_i | t_<i)` over
the agent's own reasoning correlates with `is_waste(t_i)` (operationally
defined per §2.1, manually labelled on ≥500 sampled tokens across ≥15 traces)
with Spearman ρ at least **0.20 above the measured v0.2 perplexity baseline**
on the same labelled tokens.

**Gate:** If `ρ_CTU − ρ_perplexity < 0.20`, do not ship CTU. Fall back to v0.2
VDI/SHL/CCR.

**Why this gate (changed from v0.3.0):** Absolute ρ ≥ 0.50 was uninterpretable
without knowing (a) the κ ceiling on the labels and (b) what perplexity
already achieves. The +0.20 lift over the *measured* baseline is the actual
question: "does conditional surprisal beat the generic-LM perplexity gradient
we already have?" If perplexity is already at ρ = 0.45, requiring CTU at
≥ 0.65 is honest; if perplexity is at 0.20, requiring CTU at ≥ 0.40 is honest.
Hardcoding 0.50 is neither.

**Ceiling guard:** If Cohen's κ on the labels is below 0.50, the entire
labelling protocol is too noisy and H1 cannot be evaluated. Re-label with
tighter operational definition (§2.1 will iterate if necessary). κ ≥ 0.50 is
a precondition for running H1 at all.

### H2 — Kelly-sized K Pareto-improves over discrete K-decay

**Claim:** On benchmark traces, AdaptiveK with `K_t = clip(k_min, k_max,
⌈Δq / Var(q)⌉)` achieves a Pareto improvement over v0.2.0 K-decay on the
(token-per-pass, TAS) frontier. Specifically: define
`score = 0.6 × normalized_token_savings + 0.4 × normalized_TAS_delta`
with both normalized to [0, 1] against the v0.2 baseline. Pass if `score
> 0.10` (i.e., 10% weighted improvement).

**Gate:** If `score ≤ 0.10`, do not ship Kelly-K. Keep v0.2 K-decay.

**Why this gate (changed from v0.3.0):** v0.3.0 used `AND` between "≥15% token
reduction" AND "TAS drop ≤ 2." That rejected valid wins like "12% reduction +
3-point TAS gain." The Pareto-weighted score is monotone in both improvements
and rejects only genuinely bad trades. Weights (0.6 / 0.4) are frozen in this
PRD; future tuning requires PRD revision, not silent change.

**Mutation-boundary exceptions remain (unchanged from v0.3.0):** snap to
`k_max` after any mutating tool call; snap to `k_max` for the first two trace
steps. Kelly never overrides safety invariants.

### H3 — Context Sharpe ratio gives a non-arbitrary stopping rule (gating)

**Claim:** Compressing a context until `CSR(U)` drops below the trace's
running median yields token reductions of 40–70% on benchmark traces while
preserving downstream answer quality at ≥ **70%** acceptable rate, judged by
Claude Haiku 4.5 over ≥100 sampled outputs (per-trace acceptability rate).

**Gate:** Judge-acceptability < 70% → abandon CSR-as-stopping-rule; keep v0.2
percentage-based compression. Other v0.3 metrics may still ship.

**Required audit (added in v0.3.1):** Per the parallel redundancy study's
unresolved PRD §8.5 commitment, the judge's numbers are not trusted until a
**human spot-check of ≥20 of the 100 judged pairs** agrees with the judge at
≥80%. If human/judge disagreement >20%, fall back to full manual rating on
n=50 and accept the slower, more expensive path. Implementations of H3 must
not declare a verdict before this 20-pair spot-check completes.

**Why this gate kept at 70% (not 80%):** v0.3.0 had H3 at 80%. Lowered to 70%
because (a) the redundancy v3 HP2 judge rates ran ~13–20% on cross-user pairs,
suggesting 80% is hard even for high-quality matches; (b) CSR-stop is non-
gating for the v0.3 release (other metrics can ship without it), so the
threshold can be more permissive without compromising the overall verdict.

### H4 — Audit economics (NEW, gating)

**Claim:** On benchmark traces, the median CTU+CSR audit cost per trace is
**at most 1/3 of the median token savings per trace** the audit recommends.
Equivalently, `median(savings) ≥ 3 × median(audit_cost)`.

**Gate:** If the ratio is below 3×, do not ship CTU as default. Either
(a) leave CTU behind an opt-in flag with the cost ratio documented, or
(b) abandon CTU entirely if the ratio is below 1× (audit costs more than it
saves — negative ROI).

**Why this is the actual decision:** v0.3.0 framed audit cost as a §8.1 risk
with a soft warning ("if > 5% of generation cost, warn"). In practice, this is
the gate. If H1 passes (CTU correlates well with waste) but H4 fails (audit
costs eat the savings), shipping CTU as default loses money on every trace.
The redundancy study's `cost_model.py` arithmetic discipline says: bake
economics into a hypothesis, not a warning.

**Implementation:** Measured on the same ≥15 benchmark traces used for H1,
with audit cost computed from the actual LM API spend during a CTU pass and
savings computed by comparing pre/post fix-engine token counts on the same
trace. No production data needed; all measurable in pre-ship benchmark.

---

## 2.1 Operational definition of "is_waste" (new)

A token `t_i` is **waste** if all four conditions hold:

1. **Removable without information loss.** Removing `t_i` (or its full
   token-span if part of a phrase) does not change the agent's downstream
   behavior — verified by re-running the agent on the redacted trace and
   checking the next action matches the original within a small tolerance
   (Levenshtein distance ≤ 5 tokens for tool calls; cosine ≥ 0.95 for
   reasoning continuations).
2. **Not a load-bearing transition cue.** Tokens like "however," "therefore,"
   "so" that link clauses semantically are NOT waste even if grammatically
   removable. The labelling protocol explicitly excludes connectives.
3. **Not contained within a tool-call payload.** Tool-call structure is
   syntactic; pruning tokens inside one breaks the call. Labels apply only
   to free-form reasoning text.
4. **Not part of a chain-of-thought verification step.** Self-checks ("let me
   verify this") that turn out to confirm the correct answer are NOT waste,
   even if the verification produces no new information. They're load-bearing
   for trace robustness.

### Labelling protocol

- **Sample size:** ≥ 500 tokens drawn from ≥ 15 benchmark traces (raised
  from v0.3.0's "200 tokens across 5 traces").
- **Annotators:** 2 independent (Zulfaqar + 1 reviewer, blind to each other's
  labels).
- **Inter-annotator agreement:** Cohen's κ measured on a shared 50-token
  subsample first. If κ < 0.50, refine operational definition and re-label.
  Three rounds maximum.
- **Final label** for tokens labelled by both annotators: agreement only.
  Disagreements go to a tie-breaking third reviewer or are dropped from the
  H1 evaluation set.

### Ceiling implication for H1

If the final Cohen's κ on the labelled set is X, then theoretical maximum
Spearman ρ on those labels is bounded above by approximately `X + (1−X)/2`
under standard tie-handling assumptions. **H1's gate of `baseline + 0.20` is
checked against this ceiling**: if `baseline + 0.20 > κ-ceiling`, the gate is
unachievable and H1 should be reframed (e.g., as a lift over baseline with a
proportional, not absolute, target).

---

## 3. The three new metrics (updated)

### 3.1 Conditional Token Utility (CTU)

Formula unchanged from v0.3.0:

```
U(t_i) = s(t_i) − λ₁·cost(t_i) − λ₂·redundancy(t_i)
```

Changes in v0.3.1:

- **λ₁, λ₂ sweep moved off H1 test traces.** Grid sweep on a held-out set
  of ≥10 traces distinct from the H1 labelled traces. Defaults frozen after
  sweep, recorded in `docs/cost_model_v0_3.md`. Current v0.3.0 defaults
  (`λ₁ = 0.30, λ₂ = 0.50`) are provisional only until the held-out sweep
  runs.
- **Cold-start guard (new).** Do not compute CTU for traces with fewer than
  **100 tokens**. Below that, the conditional-surprisal estimates have wide
  variance and the score is noise wrapped in math (same pathology the
  redundancy v4 HP3 cold-start ran into at n=5). Report `ctu = null` and
  skip CTU-driven fixes.
- **Cap-rate (unchanged):** 1 LM forward pass per 50 tokens of trace input.
  For traces under 500 tokens, run on every token.
- **Cost guardrail (now H4, formerly §8.1 warning).** See §2 H4.

### 3.2 Kelly-sized K (unchanged from v0.3.0)

Formula and exceptions unchanged. Pareto-style gate (H2) is the only change.
Side-by-side log requirement (§3.2 v0.3.0) carries over: for the first 100
production runs, log both Kelly's suggestion and v0.2's suggestion to a
sidecar JSON file. If Kelly disagrees by ≥ 2 on > 30% of steps but
token-per-TAS doesn't improve, fall back. This is the in-production
falsifiability check.

### 3.3 Context Sharpe Ratio (CSR)

Formula and stopping rule unchanged from v0.3.0:

```
CSR(context) = mean(U) / std(U)        (with std bounded below at 0.05)
```

Changes in v0.3.1:

- **Judge audit required before H3 verdict (per §2 H3 update).** A 20-pair
  human spot-check of judge labels must occur before CSR-stop is declared
  validated. Until that audit completes, CSR-stop ships behind
  `--csr-stop-rule` as opt-in only.
- **Cold-start guard (unchanged from v0.3.0):** ≥ 20 prior trace steps
  required before CSR-stop fires.

### 3.4 TraceRazor cost model (new, replaces ad-hoc references)

`docs/cost_model_v0_3.md` (new) freezes:

| Constant | Description | Default | Source |
|---|---|---|---|
| `c_inference_per_token` | Frontier-model cost per output token | $0.000025 (Opus 4.7) | Anthropic pricing, May 2026 |
| `c_audit_per_token` | Surprisal-estimation cost per audited token | $0.000001 (Haiku 4.5) | Anthropic pricing, May 2026 |
| `audit_sample_rate` | Tokens audited per trace-token | 1/50 (from §3.1 cap) | Engineered constraint |
| `latency_value_per_saved_token` | Value of avoiding one token of generation latency | 0 (cost-only model) | Default; product-specific |

ROI for an audit on a trace with N tokens, saving S tokens via CTU-driven
fixes:

```
roi = S × c_inference_per_token − (N × audit_sample_rate × c_audit_per_token)
H4_ratio = roi_savings_only / audit_cost
        = S × c_inference / (N × (1/50) × c_audit)
```

Pre-frozen for benchmark evaluation. Any production deployment that changes
these constants must re-evaluate H4 against the new ratios.

---

## 4. What stays the same (unchanged from v0.3.0)

- 10 metrics: 8 original + CSD + IAR (v0.2). v0.3 adds CTU + CSR. Kelly-K is
  a sampler change, not a metric.
- TAS scoring formula unchanged.
- Cost model constants from the parallel redundancy study (`cost_model.py`)
  apply to *that* study; TraceRazor uses its own `docs/cost_model_v0_3.md`
  (§3.4).
- Auto-fix patches unchanged. CSR-stop is additional (`fix_type =
  "csr_stop_compression"`), not replacement.

---

## 5. What ships, what doesn't (unchanged from v0.3.0, with §0 item 12 clarification)

Table unchanged. The IAR pheromone reformulation is **`refactor:` only** —
zero behaviour change, docstring rename only. Reviewers should see no diff
in `iar.rs` test outputs.

---

## 6. Implementation plan (timeline corrected)

Crates touched unchanged from v0.3.0 (`tracerazor-core`, `tracerazor-semantic`,
`tracerazor-store`, `tracerazor-cli`, `tracerazor-server` — modified;
`tracerazor-ingest`, `tracerazor-proxy` — untouched). LOC estimates
unchanged.

### Timeline (corrected from "1 week" to 2–3 weeks)

| Week | Days | Work |
|---|---|---|
| 1 | Mon–Tue | Labelling protocol (§2.1): Cohen's κ on 50-token subsample. Iterate operational definition if κ < 0.50. |
| 1 | Wed–Thu | Label 500 tokens across ≥15 traces. Measure perplexity baseline ρ on same labels (sets H1 threshold). |
| 1 | Fri | Implement CTU in `tracerazor-semantic`. Surprisal working on 15 benchmark traces. Cost guardrail validated. |
| 2 | Mon | λ₁/λ₂ sweep on held-out traces (not the H1 set). Freeze defaults in `docs/cost_model_v0_3.md`. |
| 2 | Tue | Run **H1 test**. If ρ_CTU − ρ_baseline < 0.20, STOP. Document and fall back. |
| 2 | Wed | Run **H4 test** (audit economics). If ratio < 3×, stop ship-as-default; flag opt-in. |
| 2 | Thu | Implement Kelly-K. Run **H2 Pareto test**. If score ≤ 0.10, fall back to v0.2 K-decay. |
| 2 | Fri | Implement CSR (read-only). Wire into `tracerazor-server` audit JSON. |
| 3 | Mon–Tue | Implement CSR-stop fix type. Run judge sample (100 pairs). |
| 3 | Wed | **Human audit** (20-pair spot-check on judge labels). If agreement < 80%, fall back to full manual on n=50. |
| 3 | Thu | Run **H3 test**. Determine CSR-stop status (opt-in flag vs default). |
| 3 | Fri | Update `benchmarks/RESULTS.md`, `README.md`, SVG architecture diagram. |

**Buffer:** Days slip. H1/H4 are the early gates; if either fails by day 8,
release plan changes. Plan for 3 weeks; cut to 2 if everything passes
cleanly.

### What can be parallelised (unchanged from v0.3.0)

Nothing across hypothesis boundaries. Each gate determines whether the next
step is worth running. The judge sample and implementation can run in
parallel within week 3 (the judge call doesn't block the CSR-stop fix-engine
code).

---

## 7. Test plan

See `tests/v0_3_test_plan.md` for full cases. Updated summary:

| Hypothesis | Test count | Run cost | Gate metric |
|---|---|---|---|
| H1 (CTU) | 6 unit + 1 integration + 1 ρ-vs-baseline | ~5 min + ~$1 LM-audit cost on 500-token sample | `ρ_CTU − ρ_baseline ≥ 0.20`, κ ≥ 0.50 precondition |
| H2 (Kelly-K) | 4 unit + 1 Pareto sweep | ~15 min + 15 benchmark traces × Kelly vs v0.2 | Pareto score > 0.10 |
| H3 (CSR-stop) | 5 unit + 100-pair judge + 20-pair human spot-check | ~10 min + ~$2 judge + 1 hour human | Judge acceptability ≥ 70% AND human-judge κ ≥ 0.80 |
| H4 (audit economics) | 1 measurement on benchmark set | included in H1 run | `median(savings) / median(audit_cost) ≥ 3` |
| Plumbing | 4 unit (no hypothesis) | < 1 min | Code coverage ≥ 85% on new files |

Total new tests: 20 unit, 4 integration, 1 human-in-loop. New CI time:
roughly +90 seconds (human spot-check is offline, not in CI).

---

## 8. Risks and what doesn't work

### 8.1 CTU LM-call cost dwarfs savings — NOW GATED BY H4

v0.3.0 had this as a soft warning. v0.3.1 promotes it to gating hypothesis
H4 (§2). Hard guardrail (`> 20% audit/generation ratio = refuse run`) remains
as runtime check in `tracerazor-semantic`.

### 8.2 Kelly assumes Gaussian, branch outcomes aren't (unchanged from v0.3.0)

Mitigation unchanged (variance floor 0.05, K bounded by `k_max`).
Falsifiable diagnostic via side-by-side log unchanged.

### 8.3 Sharpe stopping rule too coarse (unchanged from v0.3.0)

≥ 20 prior trace steps before CSR-stop fires. Cold-start guard inherited
from the redundancy v4 HP3 mistake.

### 8.4 Biological / quant framing over-sold in marketing (unchanged from v0.3.0)

README language: "v0.3 replaces three thresholds with closed-form metrics
from information theory, portfolio theory, and reinforcement learning." No
invocation of Friston or active inference unless implementation uses them
(it doesn't). v0.3.1 adds: no invocation of "Bayesian," "active inference,"
or "free-energy" in either README or marketing copy unless the code
implements posterior updates (it doesn't).

### 8.5 Labelling κ floor unachievable (new in v0.3.1)

If three rounds of definition refinement (§2.1) cannot achieve κ ≥ 0.50,
"is_waste" is fundamentally too subjective to operationalize on these
traces, and H1 cannot be evaluated honestly. Outcome: do not ship CTU.
Fall back to v0.2 metrics. This is a real outcome, not a process failure.

### 8.6 H4 fails but H1 passes (new in v0.3.1)

CTU correlates with waste (H1 ✓) but audit costs eat savings (H4 ✗).
Outcome: CTU ships as opt-in flag (`--with-ctu`), with `docs/cost_model_v0_3.md`
documenting the cost ratio explicitly. Users who want CTU's diagnostic
value can opt in; default behavior does not impose negative ROI.

---

## 9. What this does NOT do (unchanged from v0.3.0)

- No prefetch / cross-user caching (settled by redundancy v2.2/v3/v4).
- No KV-cache reuse (Anthropic prompt caching already does this).
- No fine-tuning. v0.3 is pure analysis layer.
- No new ingestion adapters.
- No model-specific tuning.

---

## 10. Decision rubric (collapsed from 5 outcomes to 3)

| H1 | H4 | H2 | H3 | Outcome | Action |
|---|---|---|---|---|---|
| ✗ | * | * | * | **Abandon v0.3** | CTU premise is wrong. Document negative result. No ship. |
| ✓ | ✗ | * | * | **Ship partial (opt-in only)** | CTU works but doesn't pay back. `--with-ctu` flag, default off. Ship Kelly-K and CSR if their gates pass. |
| ✓ | ✓ | ✓ | ✓ | **Ship full** | All three new metrics default-on; release as v0.3.0. |
| ✓ | ✓ | ✓ | ✗ | **Ship full minus CSR-stop** | CSR-stop opt-in via `--csr-stop-rule`. CTU + Kelly default-on. |
| ✓ | ✓ | ✗ | ✓ | **Ship full minus Kelly-K** | Kelly distributional issue real. Keep v0.2 K-decay default. CTU + CSR default-on. |
| ✓ | ✓ | ✗ | ✗ | **Ship CTU only** | CTU default-on as v0.3.0. Other v0.3 work documented as deferred. |

H1 and H4 are the gates. H2 and H3 inform scope, not viability. The "ship
full" path requires all four; everything else is some flavor of partial ship
or Abandon.

---

## 11. References (unchanged from v0.3.0)

- Shannon (1948). *A Mathematical Theory of Communication.*
- Kelly (1956). *A New Interpretation of Information Rate.*
- Sharpe (1966). *Mutual Fund Performance.*
- Friston (2009). *Predictive coding under the free-energy principle.*
- Jiang et al. (2023). *LLMLingua.*
- Cohen (1960). *A Coefficient of Agreement for Nominal Scales.* (κ
  measurement for §2.1 protocol.)
- Internal: `docs/findings.md`, `docs/findings_v3.md`, `docs/findings_v4.md`
  from the redundancy study.

---

## 12. Open questions for review (updated)

Items 1–5 from v0.3.0 carry over; v0.3.1 closes (3) and (4) and adds (6, 7):

1. **Surprisal reference model:** Haiku 4.5 (API, billable, accurate) or local
   distilGPT-2 (free, less accurate, no Claude-specific shift)? Recommendation:
   Haiku for production audits, distilGPT-2 for CI benchmark runs.
2. **λ₁, λ₂ defaults:** Provisional 0.30 / 0.50. Closed by §3.1 update —
   final values come from held-out sweep.
3. ~~CSR running window~~ → **CLOSED in v0.3.1**: 20 steps (§8.3), unchanged.
4. ~~IAR pheromone commit type~~ → **CLOSED in v0.3.1**: `refactor:` only,
   no behaviour change (§5, §0 item 12).
5. **Marketing language for the README.** See §8.4.
6. **Second labelling annotator identity (NEW).** §2.1 requires two
   annotators. Who is the second? Recommendation: domain-aware reviewer
   familiar with agent traces, not random crowd-source.
7. **H4 measurement on what trace set (NEW)?** Same 15 benchmark traces as
   H1, or separate? Recommendation: same set — economic ratio is a property
   of CTU's behavior on this distribution, and using the same set keeps the
   accounting consistent.

---

## 13. Lineage from v0.3.0 in one paragraph

v0.3.0 was structurally right and the review confirmed that. v0.3.1 does
four things v0.3.0 didn't: (1) it defines "is_waste" before measuring
correlation against it, with an inter-annotator-agreement protocol; (2) it
measures the perplexity baseline first instead of asserting it, and sets
H1's threshold relative to that measurement; (3) it promotes audit
economics from a soft warning to a pre-committed hypothesis (H4); and
(4) it separates the λ₁/λ₂ tuning data from the H1 evaluation data.
Everything else (the three metrics, the cost model, the crate layout, the
"doesn't ship" table, the pheromone reframing of IAR) is unchanged in
substance — only the gates around them are tightened.

If the review's Tier-1 fixes had been ignored, v0.3 would likely have
shipped with CTU showing artificially-high ρ (because λ was fit on the
test set), against a strawman perplexity baseline (asserted, not measured),
with an audit cost that quietly eroded the savings it was supposed to
detect. The pre-commitment fixes catch that before any code runs.
