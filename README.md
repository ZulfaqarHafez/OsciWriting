# LLM Workload Redundancy Study

Measures whether real-world LLM usage (WildChat-1M, writing subset) is repetitive
**and substitutable** enough to justify a cache + small-model routing layer. Full
design, hypotheses, and the pre-committed decision rubric live in [PRD.md](PRD.md).

This repo is the *measurement*, not the project. The project is downstream of the
numbers this produces.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

Copy-Item .env.example .env   # then fill HF_TOKEN, HF_HOME, JUDGE_API_KEY
huggingface-cli login --token $env:HF_TOKEN

# 1. Freeze thresholds from real prices BEFORE any run (PRD §4a, §14 step 2)
python -m redundancy.cost_model --write-doc

# 2. Pilot (no judge, no API cost) — PRD §11 Day 2
python -m redundancy.pipeline --n 5000 --no-judge

# 3. Full decision run with judge — PRD §11 Day 3-4
python -m redundancy.pipeline --n 50000

pytest        # cost-model + filter + metric unit tests
```

## What runs

`redundancy.pipeline` executes the three arms from PRD §8.1 (writing subject,
unfiltered-random control, scrambled null control) through identical
embed → UMAP → HDBSCAN → metrics → judge → report code, then writes a timestamped
`results/run_<UTC>/` directory and points `results/latest.txt` at it.

## Order of operations (do not skip — PRD §14)

1. `cost_model` → freeze T1/T3/S3/T5 in `docs/cost_model.md`.
2. Pilot at N=5000 + N-stability acceptance test (PRD §7).
3. Filter calibration (`notebooks/02_filter_calibration.ipynb`).
4. Full run, then judge (H2/H5) with the 100-pair spot-check.
5. Apply the §9 decision procedure — **H5 gates everything**.

A run on placeholder thresholds is a pilot, not the decision run.
