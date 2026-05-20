# Findings

**Status:** Decision-run complete. Outcome: **Abandon** (PRD §9 — failed H5 gate).
**Last updated:** 2026-05-20

The substitutability premise the project rests on is false on LMSYS English-writing
first-turn prompts. Embedding-space similarity (H3) and prompt↔response correlation
(H4) are real and at expected magnitudes; templated outputs (H2) are not; cross-user
substitutability of cached responses (H5) is far below the level the economics
require and is not above the scrambled-null floor by the rubric's margin. The
methodology was hardened twice during execution (dedup added v2.1, dataset switched
WildChat → LMSYS, H1 degeneracy guard added) and the negative result survived every
hardening.

---

## 1. Headline numbers

Both arms used N = 50,000 raw LMSYS-Chat-1M records, English first-turn extraction,
exact dedup → 43,979, then per-arm near dedup (cosine ≥ 0.98). Judge = Claude Haiku
4.5 (`claude-haiku-4-5-20251001`).

**Cost-model thresholds frozen 2026-05-20** from public Anthropic API pricing:
Opus 4.7 = $5 / $25 per MTok (frontier), Haiku 4.5 = $1 / $5 per MTok (small).
The 1:5 ratio holds across both input and output → `c_small = 0.20`. Plugged into
PRD §4a, this yields **T1 = T3 = 0.17** — the min cache-hit / coverage rate the
economics need to clear `S_target = 0.50` (2× cost reduction). Headline `pass`
booleans below were computed at run time with the v2.1 placeholders (T1=T3=0.05),
so they over-state H1/H3 passes against the *frozen* bar. A re-read at the frozen
thresholds is in §5; the H5 gate is **unchanged** (T5 = 0.70 is from PRD §3, not
cost-derived).

| Metric | Strict filter (subject n=6,481) | Recall filter (subject n=12,880) | Pass |
| --- | --- | --- | --- |
| H1 coverage median (sweep range) | 0.966 (0.57→1.00) | 0.982 (0.38→1.00) | TRUE both |
| H1 control gap | +0.362 (vs unfiltered 0.605) | +0.508 (vs unfiltered 0.474) | TRUE both |
| H1 degeneracy guard | not triggered | not triggered | — |
| H2 templated (judge YES of 10) | 2 / 10 | 4 / 10 | false both |
| H3 fraction @ cos ≥ 0.9 | 21.4% (control 9.4%, gap +12.0 pp) | 15.2% (control 7.0%, gap +8.2 pp) | TRUE both |
| H4 Spearman ρ | 0.790 (Pearson 0.778, n=10,000 pairs) | 0.775 (Pearson 0.767) | TRUE both |
| **H5 best subject rate (band)** | **12.8% @ [0.95, 1.00)** | **15.3% @ [0.95, 1.00)** | **FALSE both** |
| H5 calibrated S3 (lowest band ≥ T5) | none (T5=0.70 not met by any band) | none | — |
| H5 control gap | **−15.7 pp** | **+1.2 pp** | required ≥ +25 pp |
| Rubric decision | **Abandon** (H5 gate FAILED) | **Abandon** (H5 gate FAILED) | — |

Run directories: `results/run_20260520T115124Z` (strict, 2124 judge calls),
`results/run_20260520T134220Z` (recall, 2156 judge calls).

### Dedup rates (PRD §8.6 first-class finding)

| Pass | Strict run | Recall run |
| --- | --- | --- |
| Exact (normalized-text hash, full record pool) | 50,000 → 43,979 (12.0%) | identical (same pool) |
| Near (cosine ≥ 0.98), subject | 6,481 → 6,073 (6.3%) | 12,880 → 12,273 (4.7%) |
| Near, unfiltered random control | 6,481 → 6,438 (0.7%) | 12,880 → 12,738 (1.1%) |
| Near, scrambled null control | 0% (correct) | 0% (correct) |

LMSYS is far less spam-contaminated than WildChat. The N=5,000 WildChat pilot
showed cosine-0.98 dedup catching only the 0.98+ tail of one viral
Midjourney-prompt-generator template — the *real* WildChat contaminant was a
template family living at cosine 0.90–0.97, which dedup left intact and which then
produced a degenerate 2-blob clustering scored as a false H1 pass. That finding
motivated (a) the dataset switch and (b) the symmetric H1 degeneracy guard
(`cluster.is_degenerate`). On LMSYS the guard never had to fire — the clustering
shows actual sweep variation and noise.

## 2. The decisive metric: H5 substitutability

H5 measures whether a response written for prompt A is an acceptable answer to
prompt B, judged by Claude Haiku 4.5 on banded cosine pairs (subject vs scrambled
null), per-band acceptability rate = (ACCEPTABLE + 0.5 × BORDERLINE) / total.

| Cosine band | Strict subject | Strict control | Recall subject | Recall control |
| --- | --- | --- | --- | --- |
| 0.70–0.80 | 4.3% | 0.8% | 6.8% | 1.8% |
| 0.80–0.90 | 1.8% | 1.3% | 2.2% | **5.5%** (inverted) |
| 0.90–0.95 | 5.2% | **9.3%** (inverted) | 10.0% | 8.8% |
| 0.95–1.00 | 12.8% | **28.6%** (n≈7) | 15.3% | 14.1% |

Acceptability tops out at 12.8% (strict) / 15.3% (recall) in the highest cosine
band, versus the 70% required for the §4a cost model to clear `S_target=0.50`. No
band reaches T5 → S3 cannot be calibrated → the gate fails. Per PRD §9 a failed H5
is **Abandon**, no re-scope.

### Control-gap anomalies (recorded; not used to retreat from the verdict)

In strict-arm bands [0.90, 0.95) and [0.95, 1.00) the scrambled-control acceptability
exceeded subject. The [0.95, 1.00) result is from ~7 control pairs (token-shuffled
prompts rarely sit at cos ≥ 0.95) and is small-N noise; recall fixed this by
populating the band. The [0.80, 0.90) recall inversion (subject 2.2% < control 5.5%)
is from a fuller sample and is harder to dismiss — most plausibly the judge prompt
treats some scrambled gibberish + generic-sounding responses as "borderline"
disproportionately. Either way, on the headline magnitude (15% << 70%) the
inversion does not change the verdict.

## 3. Sensitivity checks (NOT PRD §8.5 audit)

### Inter-LLM (Claude Opus 4.7 vs Claude Haiku 4.5, 100 strict-run H5 pairs)

- 62 of 100 pairs auditable (Opus completed a verdict); 38 of 100 Opus refused under
  safety policy.
- Of the 62 auditable: **96.8% agreement** (60 / 62), almost entirely
  UNACCEPTABLE → UNACCEPTABLE (56). The two disagreements both have Opus rating
  *milder* than Haiku (UNACCEPTABLE → BORDERLINE / ACCEPTABLE); if there is any
  bias direction, the judge is the stricter of the two.

This is a sensitivity check, **not** the PRD §8.5 human spot-check. Two Claude
models share lineage; high agreement here is consistent with the Abandon but does
not validate the judge against human acceptability. **The PRD §8.5 audit remains
outstanding.** Without it the result is robust internally and across configurations
but is not yet PRD-complete for publication.

### Opus 38% refusal rate is itself a data-quality finding

A substantial fraction of LMSYS-Arena prompts matched by the "writing" filter are
content Opus 4.7 declines to process (jailbreak / explicit roleplay / NSFW). The
writing-task framing is therefore an over-statement: a non-trivial slice of what
the filter selects is not, in plain reading, a writing task. This does not change
the H5 verdict (Haiku does process those prompts and rates the substitutions
overwhelmingly UNACCEPTABLE) but it qualifies the "real-world writing usage" claim
on LMSYS-Arena specifically.

## 4. What the methodology hardening produced (PRD v2.1)

- **Dedup (v2.1 §8.6).** Added in response to the WildChat pilot, where one viral
  template produced 738 + 147 = 885 near-identical prompts in the top 2 clusters.
  Dedup made the metric arithmetic meaningful; it did *not* change the qualitative
  conclusion on LMSYS, but it would have produced a false positive on WildChat.
- **H1 symmetric degeneracy guard (v2.1 §8.4).** Caught the post-dedup WildChat
  case where `coverage_max == coverage_min == 1.0` across the entire sweep — a
  catastrophically uninformative clustering that was scoring as H1 pass. Never had
  to fire on LMSYS.
- **Dataset switch WildChat → LMSYS (v2.1 §7).** Withdrawal of the v1 "directly
  comparable to OpenAI's WildChat paper" framing. LMSYS carries its own validity
  caveats (Arena = model-stress traffic, not organic single-model usage; ~38% NSFW
  on the writing slice per the refusal analysis), now documented.

## 5. What does NOT change the verdict

- **Frozen cost thresholds (T1 = T3 = 0.17)** at real Anthropic prices. H1 still
  passes both arms by a wide margin (coverage 0.97–0.98 vs 0.17). H3 re-reads:
  strict 21.4% > 0.17 (still pass); **recall 15.2% < 0.17 (now fail)**. Either
  way, H5 remains the gate and **H5 fails both arms** at 12.8% / 15.3% vs T5 =
  0.70 — and T5 is not cost-derived, so no plausible re-pricing rescues it.
- **Filter choice (strict vs recall).** Both arms Abandon. The recall arm gives
  the cleaner picture (no high-band control sparsity); H5 still tops out at 15.3%.
- **Stronger judge (Opus on the safer subset).** Headline survives.

## 6. What is still owed before this is publishable

1. **PRD §8.5 human spot-check on the 100-pair audit sample**
   (`results/run_20260520T115124Z/audit_sample.md`). Until done, the result is
   "internally robust" not "judge-validated." Inter-LLM agreement (96.8%) is
   consistent with the verdict but does not replace the human audit. If
   judge/human agreement < 80%, PRD §8.5 falls back to fully-manual n=200.
2. ~~**Frozen cost-model constants**~~ — done 2026-05-20. Opus 4.7 / Haiku 4.5
   pricing in `docs/cost_model.md`, T1 = T3 = 0.17 derived. §5 captures the
   re-read against the frozen bar.
3. **A note on the [0.80, 0.90) recall control inversion** in the published
   version — minor methodological caveat, does not change the gate.

## 7. So what

A clean negative on a pre-committed rubric. The project's value proposition —
"cache + small-model router meaningfully reduces inference cost at acceptable
quality" — is contradicted by the data on this corpus + slice: similarity is
abundant, *substitutability* is not. The PRD's risk row "user pivots before
running the experiment" is the failure mode that was avoided; the rubric did
decide, and the decision is Abandon.

A publishable negative if the audit clears; a strongly-suggestive negative
otherwise. Either way: the project as scoped does not proceed without redefining
"acceptable quality" downward (which would invalidate the framing in PRD §1) or
changing slice/dataset in a way that defeats the original motivation.
