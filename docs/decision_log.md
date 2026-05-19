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
