# Decision log

One entry per major decision, dated (PRD §5).

## 2026-05-19 — PRD v1 → v2

- v1 metrics measured similarity; the decision requires substitutability + economics.
- Added gating hypothesis **H5 (substitutability)**; a failed H5 → Abandon with no
  re-scope. Thresholds T1/T3 derived from the §4a cost model, not chosen. Added
  unfiltered + scrambled controls. Fixed H2 (trivially-passable bar), H4 (range
  restriction), filter-bias framing, embedding-confound disclosure, Windows-safe
  paths, internal contradiction on sample size.

## 2026-05-19 — Scaffold implemented (PRD §14 steps 1, 3)

- Repo structure, `src/redundancy/` modules, tests, notebook stubs created from PRD
  v2. Cost-model constants remain PROVISIONAL — no decision run until real prices
  are frozen in `docs/cost_model.md` (PRD §4a, §14 step 2).
- Pending (require user inputs / compute): real prices + S_target; HF token +
  WildChat license; judge API key; pilot N-stability acceptance test (PRD §7);
  filter calibration (`notebooks/02_filter_calibration.ipynb`).

## 2026-05-19 — run 20260519T150139Z

- N=5000 dataset=wildchat thresholds_provisional=True
- Outcome: **Pilot — no decision** (H5 gate: NOT EVALUATED)
- Action: Run without --no-judge for the H5-gated decision (PRD §9).

## 2026-05-19 — Pilot finding: WildChat writing subset is verbatim-spam-dominated → PRD v2.1

- First pilot exposed that the top 4 clusters absorbed all 1474 strict-filtered
  prompts; one "Midjourney prompt generator" jailbreak appeared verbatim 586 + 276
  + 26 times. H1 degenerated (coverage 1.0, noise 0.0, control gap 0.0 → correctly
  failed by the rubric); H3 hit 61% @ cos≥0.9 and H4 Spearman 0.82, both inflated
  by literal duplicates rather than semantic redundancy.
- Decision: PRD v1/v2 had no dedup stage — a real methodology hole. Amended to
  **v2.1**: added `dedup.py` (exact normalized-text + near cos≥0.98), wired before
  metrics in all three arms, dedup rate reported as a finding (PRD §8.6, §12).
- Also fixed an unrelated bug the pilot surfaced: `DerivedThresholds` lacked
  `H4_rho`/control-gap fields used by pipeline/report (now merged into `th_full`).
- Next: re-pilot at N=5000 with dedup; if subject survivors are too few, raise N
  (the writing subset is mostly spam, so post-dedup yield is low).

## 2026-05-19 — run 20260519T151621Z

- N=5000 dataset=wildchat thresholds_provisional=True
- Outcome: **Pilot — no decision** (H5 gate: NOT EVALUATED)
- Action: Run without --no-judge for the H5-gated decision (PRD §9).

## 2026-05-20 — Dedup@0.98 insufficient → switch to LMSYS + H1 degeneracy guard

- Re-pilot with dedup: exact only 2.6%, near@0.98 subject 10.6%. WildChat
  contaminant is a viral *template* (Midjourney prompt generator): huge fixed
  preamble, tiny variable concept → distinct requests sit at cosine 0.90–0.97,
  *below* the 0.98 cut. Post-dedup clustering was still 2 mega-blobs (738 template
  + 571 grab-bag), 0 noise, coverage 1.0 across all 36 sweep cells. Rubric scored
  this as **H1 pass** (false pass) because the control coverage merely dropped to
  0.73 so the control-gap check passed.
- Decision (user): (a) switch primary dataset WildChat → **LMSYS-Chat-1M**;
  WildChat demoted to cautionary fallback (PRD §7). v1 "comparable to OpenAI's
  WildChat paper" claim withdrawn. (b) Add a **symmetric degeneracy guard**
  (`cluster.is_degenerate`): flat-1.0 / ~0-noise / no-sweep-variation → H1 Fail
  regardless of T1/control gap (PRD §8.4).
- Not chosen: template-family collapse / filter-tightening (revisit only if LMSYS
  also shows template contamination).
- Tests: 30 pass (added test_cluster.py, test_dedup.py). Next: re-pilot on LMSYS.

## 2026-05-20 — run 20260519T161458Z

- N=5000 dataset=lmsys thresholds_provisional=True
- Outcome: **Pilot — no decision** (H5 gate: NOT EVALUATED)
- Action: Run without --no-judge for the H5-gated decision (PRD §9).

## 2026-05-20 — run 20260520T052203Z

- N=5000 dataset=lmsys thresholds_provisional=True
- Outcome: **Abandon** (H5 gate: FAILED)
- Action: Document negative result in docs/findings.md. No re-scope on a failed H5.

## 2026-05-20 — run 20260520T115124Z

- N=50000 dataset=lmsys thresholds_provisional=True
- Outcome: **Abandon** (H5 gate: FAILED)
- Action: Document negative result in docs/findings.md. No re-scope on a failed H5.

## 2026-05-20 — run 20260520T134220Z

- N=50000 dataset=lmsys thresholds_provisional=True
- Outcome: **Abandon** (H5 gate: FAILED)
- Action: Document negative result in docs/findings.md. No re-scope on a failed H5.

## 2026-05-20 — Consolidated decision: Abandon survives strict + recall + inter-LLM

- run 20260520T115124Z (strict, n=6,481): H5 best 12.8% @ [0.95,1.00), gap −15.7 pp.
- run 20260520T134220Z (recall, n=12,880): H5 best 15.3% @ [0.95,1.00), gap +1.2 pp.
- Inter-LLM sensitivity check (Opus 4.7 vs Haiku 4.5, 100 strict-run pairs):
  96.8% agreement on the 62 auditable pairs; 38 declined by Opus safety layer
  (separate finding about LMSYS-Arena NSFW contamination on the writing slice).
  Explicitly NOT the PRD §8.5 human audit.
- Code added: pipeline `--filter strict|recall`; judge transcripts now record
  `band` and `arm`; `audit.py` (sample / score / inter-llm CLI) + tests.
- `docs/findings.md` rewritten as the consolidated negative-result write-up.
- Outstanding for PRD-completeness: human spot-check on the 100-pair audit sample
  (currently only inter-LLM-validated); freeze cost-model constants.

## 2026-05-20 — Cost-model constants frozen (PRD §4a)

- Source: live Anthropic API pricing lookup (WebFetch + WebSearch, May 2026).
  Opus 4.7 = $5/$25 per MTok (frontier), Haiku 4.5 = $1/$5 per MTok (small).
  Clean 1:5 ratio across input and output → `c_small = 0.20` (placeholder was
  0.05, which assumed a 20× cheaper small model — only 5× in reality).
- Derived thresholds: **T1 = T3 = 0.17** (was 0.05). T5 = 0.70 unchanged (PRD §3,
  not cost-derived). `provisional=False`; doc regenerated.
- Re-read of decision runs at frozen bar: H1 still passes both arms; H3 strict
  still passes (21.4% > 0.17), **H3 recall now fails** (15.2% < 0.17); **H5 gate
  still fails both arms** (12.8% / 15.3% vs 0.70). Verdict unchanged: Abandon.
- Outstanding for PRD-completeness: human PRD §8.5 audit (user time only).

## 2026-05-20 — Binary content-rule label applied (NOT the §8.5 audit)

- User-supplied rule: NSFW content → UNACCEPTABLE; else → ACCEPTABLE.
  Mismatch with §8.5 (which asks about substitutability, not content) was
  explained beforehand; user chose to apply anyway with explicit documentation.
- Result: 11/100 agreement; 86/100 are `UNACCEPTABLE → ACCEPTABLE` (judge said
  the substitution didn't substantively answer the prompt; the rule said the
  content wasn't NSFW so it was OK). 12/100 NSFW matches.
- Recorded in `audit_binary_content_rule.jsonl` and findings §3. The 11% does
  **not** invalidate the judge or trigger the §8.5 fall-back, because the
  content-rule question and the audit question are orthogonal.
- PRD §8.5 substitutability audit still owed.

## 2026-05-21 — PRD v3 (prefetch feasibility) drafted + decision-run complete: Abandon

- v2.2 cost-cache study reached Abandon. User clarified actual interest:
  Gmail-style speculative prefetch (latency optimization), not cost reduction.
  Wrote `PRD_v3.md` and a separate v3 pipeline (`prefetch.py`,
  `cost_model_v3.py`, frozen `docs/cost_model_v3.md`). v2.2 study unchanged.
- v3 §15 multi-turn pilot (15,132 turn-pairs from N=5k LMSYS conversations):
  - **HP1 (predictability lift)** best +5.7 pp (MiniLM K=50 T_pred=0.7);
    threshold +10 pp; **FAIL**. TF-IDF arm matched MiniLM (+4.8 pp) —
    graph-similarity didn't beat the embedding retriever.
  - **HP3 (within-session)** p50 = 0.325 (req ≥0.60); frac ≥0.7 = 17.8%
    (req ≥30%); **FAIL**. Users don't rephrase next prompts.
  - **HP2/HP4** not evaluated — per §9 step 1, failed HP1 → Abandon
    regardless of downstream hypotheses. Skipped the judge API spend.
- **Decision: Abandon.** `docs/findings_v3.md` written; honest
  re-read of the +10 pp threshold included (eyeballed but defensible — even at
  +5.7 pp lift the per-turn ROI is implausible at any latency-value).
- Two independent gates on the same dataset now Abandon:
  - v2.2: H5 substitutability (responses don't transfer at high cosine).
  - v3: HP1 predictability (next prompts don't predict each other above margin).
  Same dataset; the corpus simply does not exhibit the cross-prompt structural
  regularity that *either* mechanism needs.

## 2026-05-21 — v3 full pipeline (with judge) confirms quadruple FAIL

- After "try again" the full prefetch.py was re-run WITH the HP2 judge
  (run prefetch_20260521T063300Z, 172 judge calls). Added retry-on-5xx logic
  in judge._ask (transient Anthropic 500 mid-run on first attempt).
- All four hypotheses fail their pre-committed thresholds:
  - HP1 lift +5.67 pp (req +10 pp).
  - **HP2** quality at T_serve=0.95: 20.4% (req 60%) — confirms v2.2 H5 finding
    on multi-turn slice at stricter match condition. ~20% vs 12.8–15.3% on
    first-turn = multi-turn is somewhat more substitutable, but nowhere near
    safe to serve.
  - HP3 p50=0.325 (req 0.60).
  - HP4 0 of 180 Pareto cells positive-ROI.
- findings_v3.md updated with full data including HP2/HP4. Verdict robust.
