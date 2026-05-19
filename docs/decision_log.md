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
