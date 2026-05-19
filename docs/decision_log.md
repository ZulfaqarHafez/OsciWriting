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
