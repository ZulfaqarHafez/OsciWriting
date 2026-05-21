# LLM Workload Redundancy Study

**Status:** Pre-experiment, design locked (v2.2)
**Author:** Zul Fhagez
**Last updated:** 2026-05-19
**Decision deadline:** within 1 week of data collection completion

---

## 0. Changelog (v1 → v2)

This revision exists because v1's metrics measured *similarity* while the decision
the study must produce depends on *substitutability* and *economics*. Changes:

1. **New gating hypothesis H5 (substitutability).** Cosine proximity is not response
   interchangeability. H5 directly measures, via judge rating, whether a cached
   response is acceptable for a near-neighbor prompt. H5 gates the entire decision:
   if H5 fails, the outcome is "abandon" regardless of H1–H4.
2. **Thresholds are now derived from a cost model**, not chosen by feel. Section 4a
   defines the economics; H1/H3 thresholds are outputs of that arithmetic. The v1
   numbers (0.40, 0.9@10%) are retained only as provisional placeholders until real
   pricing is plugged in.
3. **Baseline/control added.** The pipeline now runs identically on an unfiltered
   random sample and a scrambled control. Decisions use the *gap* between the writing
   subset and the control, not absolute thresholds alone.
4. **H2 redefined.** The v1 0.70 MiniLM-cosine bar was trivially passable by any two
   topical English texts. H2 is now a templatedness / judge-substitutability measure.
5. **H4 sampling fixed.** v1 paired anchors only with top-5 nearest neighbors
   (range restriction → attenuated, biased Pearson). H4 now samples pairs across the
   full similarity range, reports Spearman primary, and requires scatter inspection.
6. **Filter bias reclassified.** v1 claimed false negatives "only shrink the sample."
   They are directional (drop implicitly-phrased requests, keep formulaic ones), so
   they bias H1/H3 toward passing. Documented as a threat, with a recall-oriented
   second pass for comparison.
7. **Embedding confound made explicit.** All MiniLM cosine thresholds are
   embedding-relative and treated as descriptive, not decisive. The decision rests on
   H5 (judge) and the control gap, which are embedding-independent.
8. **UMAP → HDBSCAN.** Density clustering on raw 384-dim vectors is unreliable; a
   UMAP reduction precedes HDBSCAN. A hyperparameter sweep replaces the single-number
   H1 reading.
9. **Windows-safe paths.** No symlinks (win32 needs admin/Developer Mode). `results/latest`
   is a `latest.txt` pointer file. Timestamps are filesystem-safe (`run_20260519T214100Z`).
10. **Internal contradiction fixed.** v1 §7 cited an N-stability result from a pilot
    v1 §14 said was not yet run. Sample-size rationale is now a pilot acceptance test,
    not a cited result.
11. **External-validity caveat added.** WildChat is consumer free-tier access; the
    motivating findings are enterprise/paid. The gap is now named in §12.

### v2 → v2.1 (post-pilot)

12. **Deduplication stage added (§8.6).** The first N=5000 pilot showed WildChat's
    writing subset is dominated by mass copy-paste viral prompts — one "Midjourney
    prompt generator" jailbreak repeated *verbatim* hundreds of times (top 4
    clusters absorbed all 1474 strict-filtered prompts; H1 coverage degenerated to
    1.0 with 0.0 noise; H3 hit 61% @ cos≥0.9). Without collapsing duplicates the
    metrics measure copy-paste volume, not semantic redundancy across distinct
    phrasings — and caching identical strings is a hashmap, not the project. v2.1
    adds an exact (normalized-text) + near (cosine ≥ 0.98) dedup stage before any
    metric, and reports the dedup rate as a first-class finding.
13. **Primary dataset switched to LMSYS-Chat-1M; H1 degeneracy guard added.** The
    re-pilot proved cosine-0.98 dedup does *not* remove the WildChat template
    contaminant (it lives at 0.90–0.97: huge shared preamble, tiny variable
    payload) and the post-dedup clustering was still 2 mega-blobs / 0 noise, which
    the rubric falsely scored as H1 pass. Two fixes: (a) WildChat → cautionary
    fallback, LMSYS-Chat-1M promoted to primary (§7), and (b) a symmetric
    degeneracy guard so a flat-1.0 / zero-noise coverage envelope is an H1 Fail
    (§8.4). The v1 "directly comparable to OpenAI's WildChat paper" claim is
    withdrawn.

### v2.1 → v2.2 (post-decision investigation)

14. **Multi-turn predictability investigation added (§15).** The first-turn
    semantic-cache study reached Abandon; user asked whether *predictive
    pre-generation* (speculate likely next prompts, pre-compute responses) is
    feasible. That's a different mechanism. §15 measures it as an investigation
    (not a new go/no-go) on conversations with ≥2 user turns, with both a MiniLM
    retriever and a TF-IDF "graph-similarity" retriever, and the H5-gated
    rubric of §3-§4 remains unchanged for the first-turn question.

---

## 1. Purpose

Determine, with quantitative evidence, whether real-world LLM usage is repetitive
*and substitutable* enough that an adaptive cache + small-model routing layer would
meaningfully reduce inference cost **at acceptable quality**. This study is the
precondition for committing to or abandoning the proposed thesis project.

Two words in that sentence carry the weight: *substitutable* (a cached answer must
actually serve a different-but-similar prompt, not merely embed near it) and
*acceptable quality* (the substituted answer must be good enough). v1 measured
neither. v2 makes both first-class.

The study is not the project. The project (whatever final shape it takes) is
downstream of this measurement.

---

## 2. Background

Two recent published findings motivate this study:

1. **OpenAI, NBER w34255 (Sep 2025), "How People Use ChatGPT":** three topic
   categories (Practical Guidance, Seeking Information, Writing) account for ~80% of
   all ChatGPT conversations. Writing dominates work-related tasks.
2. **Anthropic Economic Index (Jan 2026):** the top 10 work tasks account for 24% of
   all Claude.ai conversations out of 3,000+ unique tasks. Long-tailed, large head.

If usage is concentrated at the *task* level, prompts within a task should cluster,
responses within a cluster should share structure, and — critically — one response
should be reusable across many prompts in the cluster. The first two are necessary
but not sufficient; the third is what makes caching work, and it is what H5 tests.

**External-validity caveat (new in v2):** the motivating findings describe paid /
enterprise usage. The dataset (WildChat) is consumer free-tier access via a public
HuggingFace Space, skewed toward roleplay, jailbreak attempts, and users seeking free
access. Concentration observed (or not observed) here does not transfer cleanly to
the enterprise setting the motivation describes. This is a named threat in §12, not a
silent assumption.

---

## 3. Hypotheses

For the "writing" subset of WildChat-1M first-turn user prompts, with all
embedding-space numbers computed on `all-MiniLM-L6-v2`:

**H1 (clustering):** The top 50 prompt clusters cover at least **T1** of all writing
prompts, where **T1 is derived from the cost model in §4a** (provisional placeholder:
40%). Reported with a hyperparameter sweep, not a single number.

**H2 (intra-cluster templatedness):** Within the top 10 clusters, responses are
template-like: a judge rates "could a single parameterized template with slot-filling
produce all of these?" as yes for at least 60% of sampled clusters. (v1's 0.70 MiniLM
bar is retained only as a descriptive statistic, not a gate.)

**H3 (cacheability):** At least **T3** of prompts have a nearest-neighbor cosine ≥
**S3** to another prompt, where **T3 and S3 are derived from the cost model in §4a**
(provisional placeholders: 10% at cosine 0.9). The cosine threshold S3 is calibrated
against H5, not asserted.

**H4 (correlation):** Spearman correlation between prompt-pair similarity and
response-pair similarity is at least 0.5, computed over pairs sampled across the
**full** similarity range (not nearest-neighbor pairs only).

**H5 (substitutability — GATING):** For prompt pairs at or above similarity band S3,
the judge-rated acceptability rate of serving prompt A's response to prompt B is at
least **T5** (provisional placeholder: 70%), and this rate is at least 25 percentage
points above the same rate measured on the scrambled control. **If H5 fails, the
study outcome is "abandon" regardless of H1–H4.** H1–H4 without H5 only show prompts
look alike; H5 is the only hypothesis that tests whether one answer serves many.

---

## 4. Decision rubric

H5 is evaluated first and gates everything.

| H5 | H1 | H2 | H3 | H4 | Outcome | Action |
| --- | --- | --- | --- | --- | --- | --- |
| **Fail** | any | any | any | any | **Abandon** | Document negative result in `docs/findings.md`. No further analysis. |
| Pass | Pass | Pass | Pass | Pass | Strong positive | Commit to project, design architecture |
| Pass | Pass | Pass | Fail | Pass | Weak positive | Re-scope to in-conversation caching, not cross-user |
| Pass | Fail | Pass | Mixed | Pass | Domain mismatch | Re-run on a narrower slice (emails only, reports only) |
| Pass | Fail | Fail | Fail | Fail | Marginal | Re-scope to narrowest viable slice; one attempt only |
| Pass | any other combination | | | | Anomalous | Investigate, then re-decide |

Every "Pass" above is also conditional on the **control gap**: the writing-subset
metric must exceed the unfiltered-random and scrambled-control metric by the margin
stated per hypothesis. A threshold cleared in absolute terms but not above control
counts as a Fail for that hypothesis.

Decisions are based on the headline numbers and the control gap, not on intuition or
partial-result hope.

### 4a. Cost model (thresholds are outputs of this, not inputs)

"Meaningful cost reduction" is computable. Every request first hits the cache; on miss
it is routed to either the small model or the frontier model.

Let, all costs normalized to `c_frontier = 1.0`:

- `c_cache` — cost of one cache lookup (embedding + vector search). Estimate ≈ 0.002.
- `c_small` — small-model cost as a fraction of frontier. Estimate ≈ 0.05.
- `p_cache` — fraction of requests served from cache (this is what H3 estimates).
- `p_small` — fraction of cache-misses routed to the small model.

Blended cost per request:

```
C = c_cache + (1 - p_cache) * [ p_small * c_small + (1 - p_small) * 1.0 ]
```

Savings `S = 1 - C`. Define "meaningful" as `S ≥ S_target`. **The author must set
`S_target` and plug in real prices before the full run** (`S_target = 0.50`, i.e. a
2× cost reduction, is the provisional default). Given `S_target`, solve for the
`(p_cache, p_small)` frontier; the minimum viable `p_cache` becomes **T3/T1** and the
required cache precision becomes the calibration target for **S3** and **T5**. A
worked example with the provisional constants lives in
`notebooks/00_cost_model.ipynb` and is committed as `docs/cost_model.md` before Day 3.

Until those numbers are filled in, the placeholders in §3 stand, but a run executed
on placeholders is a pilot, not the decision run.

---

## 5. Repository structure

Tree is relative to the repository root (this repo *is* the project root; there is no
extra `llm-redundancy-study/` wrapper directory).

```
.
├── README.md                      # short version of this PRD, with quickstart
├── PRD.md                         # this file
├── pyproject.toml                 # project metadata and pinned dependencies
├── .env.example                   # template for HF_TOKEN, JUDGE_API_KEY
├── .gitignore                     # excludes data/, results/, .env, __pycache__
│
├── src/
│   └── redundancy/
│       ├── __init__.py
│       ├── config.py              # tunables: sample size, model names, thresholds
│       ├── data.py                # dataset loading, first-turn extraction, lang filter
│       ├── filters.py             # task filter (writing) + recall-oriented variant
│       ├── baseline.py            # unfiltered-random and scrambled control samplers
│       ├── embed.py               # sentence-transformer wrapper, batching
│       ├── reduce.py              # UMAP dimensionality reduction (pre-clustering)
│       ├── cluster.py             # HDBSCAN clustering, param sweep, cluster stats
│       ├── judge.py               # LLM-judge for H2 templatedness and H5 substitutability
│       ├── metrics.py             # coverage, NN sim, full-range correlation
│       ├── report.py              # generates results/summary.md from numbers
│       └── pipeline.py            # orchestrates the full run end-to-end
│
├── notebooks/
│   ├── 00_cost_model.ipynb        # derive T1/T3/S3/T5 from real prices
│   ├── 01_data_exploration.ipynb  # sanity-check the dataset before pipelining
│   ├── 02_filter_calibration.ipynb # tune writing-task filter, precision & recall
│   ├── 03_pilot_run.ipynb         # small-sample dry run (N=5000) for debugging
│   └── 04_results_inspection.ipynb # interactive exploration of final results
│
├── data/
│   ├── raw/                       # cached HF dataset shards (gitignored)
│   └── processed/                 # filtered DataFrames as parquet (gitignored)
│
├── results/
│   ├── run_<TS>/                  # TS = filesystem-safe UTC, e.g. 20260519T214100Z
│   │   ├── config.json            # the exact config used for this run
│   │   ├── headline_numbers.json  # H1–H5 numbers + control gaps + supporting stats
│   │   ├── cluster_examples.md    # human-readable top-cluster prompts
│   │   ├── judge_transcripts.jsonl # raw H2/H5 judge calls for audit
│   │   ├── figures/
│   │   │   ├── cluster_size_distribution.png
│   │   │   ├── nn_similarity_distribution.png
│   │   │   ├── prompt_vs_response_similarity.png
│   │   │   └── subset_vs_control.png
│   │   └── summary.md             # auto-generated interpretation against rubric
│   └── latest.txt                 # plain text: name of the most recent run dir
│
├── tests/
│   ├── test_filters.py            # unit tests for keyword filter precision/recall
│   ├── test_metrics.py            # unit tests for metric calculations
│   └── test_cost_model.py         # unit tests for §4a arithmetic + threshold solver
│
└── docs/
    ├── decision_log.md            # one entry per major decision, dated
    ├── cost_model.md              # frozen §4a constants and derived thresholds
    ├── prior_art.md               # related work, organized
    └── findings.md                # final write-up, populated post-experiment
```

**Conventions:**

- All randomness seeded with `SEED = 42` in `config.py`. UMAP is also seeded;
  note that sentence-transformers batch order can introduce sub-1e-6 float
  nondeterminism on GPU — acceptable, documented, not chased.
- Data caching uses HuggingFace's default cache directory. **No symlink** (win32);
  the cache path is read from `HF_HOME` in `.env` and recorded in `config.json`.
- All results live in `results/run_<TS>/` with `TS` a filesystem-safe UTC stamp
  (`YYYYMMDDThhmmssZ`, no colons). `results/latest.txt` contains the current run's
  directory name as plain text — read it, don't follow a symlink.
- Every run writes its full config to `config.json` for reproducibility.
- No notebook output is committed. The pipeline is the source of truth.

---

## 6. Environment setup

**Python:** 3.11 or 3.12.

**Dependencies** (pin in `pyproject.toml`):

```
datasets >= 2.16
sentence-transformers >= 2.5
umap-learn >= 0.5.5
hdbscan >= 0.8.33
scikit-learn >= 1.4
pandas >= 2.2
numpy >= 1.26, < 2.0
matplotlib >= 3.8
tqdm >= 4.66
python-dotenv >= 1.0
pyarrow >= 14.0
huggingface-hub >= 0.20
anthropic >= 0.40            # LLM judge for H2/H5 (or substitute provider)
```

Note: `numpy < 2.0` with recent `hdbscan` / `scikit-learn` / `umap-learn` can have
binary-compat friction; if wheels conflict, pin `numpy == 1.26.4` explicitly.

**Hardware:** CPU-only workable but slow (~15 min to embed 50k prompts vs ~2 min on
a T4). Colab free tier (T4) or a GPU laptop recommended. 16GB RAM comfortable.

**HuggingFace access:**

1. Create a HuggingFace account.
2. Accept the WildChat-1M license: https://huggingface.co/datasets/allenai/WildChat-1M
3. Create a read token at https://huggingface.co/settings/tokens
4. Copy `.env.example` to `.env`; set `HF_TOKEN`, `HF_HOME`, and `JUDGE_API_KEY`.
5. `huggingface-cli login --token $env:HF_TOKEN` once per machine (PowerShell syntax).

If WildChat approval is slow, fall back to LMSYS-Chat-1M (`lmsys/lmsys-chat-1m`).

**Judge access:** H2 and H5 require an LLM judge. Default: Claude Haiku 4.5
(`claude-haiku-4-5-20251001`) for cost. Budget ≈ 5k judge calls for the full run
(2k H5 pairs + H2 cluster checks + control). Estimate cost before Day 3.

---

## 7. Data sources

### Primary: LMSYS-Chat-1M (promoted in v2.1)
1,000,000 conversations across 25 LLMs (`lmsys/lmsys-chat-1m`). First-turn English
extraction is identical to WildChat (`conversation` list, `language`,
`conversation_id`). Promoted to primary because the N=5000 WildChat pilot showed
its writing subset is dominated by one viral "Midjourney prompt generator"
template (see §8.6 and the decision log).

**Known skew:** LMSYS is Chatbot-Arena traffic (users comparing models), so prompts
skew toward model-stress-testing and may be more polished or unusual than organic
single-model usage. This is a *different* validity threat than WildChat's spam, and
it weakens the v1 "directly comparable to OpenAI's WildChat paper" framing — that
comparability claim is **withdrawn** in v2.1. Report this skew in findings; do not
over-claim transfer to enterprise usage (§12).

### Fallback: WildChat-1M (contaminated — cautionary)
`allenai/WildChat-1M`, 1,000,000 ChatGPT conversations. Retained only as a
cautionary comparison; the pilot proved its writing subset is template-spam
dominated. Do not use for the decision run without the §8.6 dedup *and* a
template-family handling step that the pilot showed cosine-0.98 dedup does not
provide.

### Sample size — pilot acceptance test (not a cited result)
N = 50,000 first-turn English writing prompts is the target for the decision run.
Whether N = 5,000 is already stable is **not yet known** and is decided by the pilot:

> **Pilot acceptance test (Day 2):** run the full pipeline at N=5,000 and again at
> N=50,000 on the same seed. If any headline number (H1, H3, H4, H5) differs by more
> than 15% relative between the two, N=50,000 is insufficient — escalate to N=150,000
> and re-test. If they agree within 15%, N=50,000 is the decision-run size.

No sample-size claim is asserted in advance; it is an output of Day 2.

---

## 8. Methodology

### 8.1 Pipeline stages

```
[raw dataset]
  -> language + first-turn filter            (data.py)
  -> EXACT dedup (normalized-text hash)      (dedup.py)   -- §8.6, before filtering
  -> three parallel arms:
       (a) writing filter           (filters.py)   -- the subject
       (b) unfiltered random sample (baseline.py)  -- control 1
       (c) scrambled-prompt control (baseline.py)  -- control 2 (null)
  -> prompt embedding                          (embed.py)
  -> NEAR dedup per arm (cosine >= 0.98)       (dedup.py)   -- §8.6, before metrics
  -> UMAP reduction                            (reduce.py)
  -> HDBSCAN clustering + param sweep          (cluster.py)
  -> metrics                                   (metrics.py)
       - H1 top-N coverage (+ sweep envelope)
       - H3 NN-similarity distribution
       - H4 full-range prompt/response Spearman
  -> judge                                     (judge.py)
       - H2 cluster templatedness
       - H5 substitutability (subject vs control)
  -> report                                    (report.py)
       - headline_numbers.json (incl. control gaps)
       - figures/*.png
       - summary.md auto-populated against §4 rubric
```

All three arms (a/b/c) run through identical embedding/reduction/clustering/metrics
code. The decision uses (a) minus (b)/(c), not (a) alone.

### 8.2 Filter specification and its bias

The writing-task filter is a regex keyword matcher over the first 500 characters of
the user prompt: task verbs (write, draft, compose, generate, create, rewrite,
proofread, summarize) × output types (email, letter, report, essay, message, …).

**Bias statement (corrected from v1):** false negatives are **not** random. A
precision-tuned regex keeps explicitly-phrased requests and drops implicitly-phrased
ones ("I need to tell my landlord the heater's broken"). Explicit phrasing is exactly
what clusters, so this filter shifts subset composition *toward* H1/H3 passing. v1's
claim that "false negatives only shrink the sample" is withdrawn.

Mitigation: `filters.py` also implements a recall-oriented variant (intent
classification over the full prompt, looser threshold). Calibration notebook reports
H1/H3 under both filters; if they diverge materially, the conservative (recall-
oriented) numbers are the ones the rubric uses.

Calibration (`notebooks/02_filter_calibration.ipynb`): sample 200 prompts (half
matched, half unmatched), manually label, report precision (target 0.85+) and recall
(target 0.50+) for the strict filter and recall for the loose one.

### 8.3 Embedding choice and the embedding confound

`sentence-transformers/all-MiniLM-L6-v2`: 384-dim, ~120MB, runs anywhere, L2-
normalized so cosine = dot product, matches the embedding used in TraceRazor.

**Confound statement (new in v2):** MiniLM cosine reflects topic/surface form, not
response interchangeability. Therefore every absolute cosine threshold (S3, the H2
descriptive stat) is **embedding-relative and descriptive only**. The decision-
bearing signals are H5 (judge acceptability) and the control gaps, both embedding-
independent. S3 is not asserted; it is *calibrated* — chosen as the cosine band at
which the H5 acceptability rate first exceeds T5 (see 8.5).

Do not change the embedding mid-study; only swap on a clean re-run. Alternatives if a
re-run is needed: `BAAI/bge-small-en-v1.5`, `mxbai-embed-large-v1`.

### 8.4 Reduction + clustering

UMAP → HDBSCAN (BERTopic-style), because HDBSCAN density estimation degrades on raw
384-dim vectors.

- UMAP: `n_neighbors=15`, `n_components=10`, `metric="cosine"`, `random_state=SEED`.
- HDBSCAN: `min_cluster_size=20`, `min_samples=5`, `metric="euclidean"` on the UMAP
  output, `cluster_selection_method="eom"`.

**Sweep, not a single number:** H1 is reported as an envelope over
`min_cluster_size ∈ {15, 20, 30, 50}` × `min_samples ∈ {3, 5, 10}` and over
`n_components ∈ {5, 10, 20}`. The headline H1 is the *median* coverage across the
sweep with min/max reported. Noise fraction is reported for every cell — if noise
exceeds 60% across most of the sweep, H1 is recorded as Fail with the note "structure
not separable at this granularity," distinct from "no structure exists."

**Degeneracy guard (added v2.1).** The high-noise guard above has a symmetric blind
spot the pilot walked straight through: coverage pinned at ~1.0 with ~0 noise and
*no variation across the 36-cell sweep* means HDBSCAN swallowed everything into a
couple of mega-blobs — equally meaningless, but it was scoring as an H1 **pass**.
H1 is now also recorded as Fail when `coverage_min ≥ 0.99` AND `noise_median ≤ 0.01`
AND `coverage_max − coverage_min < 0.01` (`cluster.is_degenerate`). A real
concentration signal leaves a tail and moves as sweep params change; a flat-1.0
envelope does not. This guard fires *regardless* of T1 and the control gap.

### 8.5 Metric definitions

**H1 — top-N coverage:** Σ(sizes of N largest clusters) / total prompts (incl.
noise). Reported as the sweep envelope (8.4). Subject value must exceed control
value by ≥ 10 percentage points to count as Pass.

**H2 — intra-cluster templatedness:** for each of the top 10 clusters, sample 8
responses, send to the judge: "Could one parameterized template with slot-filling
plausibly produce all of these? Yes/No + one-line reason." Pass = yes for ≥ 6 of 10
clusters. MiniLM mean-pairwise-cosine is also recorded per cluster as a descriptive
companion, explicitly **not** a gate.

**H3 — nearest-neighbor similarity:** for each prompt, cosine to its nearest other
prompt. Report the full distribution and the fraction crossing 0.7/0.8/0.9/0.95.
S3 is *calibrated* from H5 (below), not fixed at 0.9. Pass = fraction at calibrated
S3 ≥ T3 (from §4a) AND ≥ control fraction + 5 points.

**H4 — prompt/response correlation:** sample 10,000 prompt pairs **stratified across
similarity bins** (uniform over cosine ∈ [0,1] in 0.1-wide strata, ~1000/bin, drawn
from the full subject set — random pairs cover the low end, NN search seeds the high
end). For each pair compute prompt cosine and response cosine. Report **Spearman ρ**
(primary) and Pearson r (secondary). The scatter plot
(`prompt_vs_response_similarity.png`) must be inspected before the coefficient is
trusted; non-monotonic shape overrides the number. Pass = Spearman ρ ≥ 0.5.

**H5 — substitutability (gating):** build prompt pairs in similarity bands
[0.70,0.80), [0.80,0.90), [0.90,0.95), [0.95,1.0). For ≥ 300 pairs per band, ask the
judge: "Here is prompt B and a response that was written for prompt A. Is this
response an acceptable answer to prompt B? Acceptable / Borderline / Unacceptable +
reason." Acceptability rate per band = (Acceptable + 0.5·Borderline) / total.
- **S3 calibration:** S3 := the lowest band whose acceptability rate ≥ T5.
- Run the identical H5 protocol on the scrambled control to get a floor.
- Pass = a calibrated S3 exists with rate ≥ T5 AND ≥ control floor + 25 points.
- Judge prompts, raw responses, and reasons are written to
  `judge_transcripts.jsonl` for audit. A 100-pair manual spot-check validates the
  judge before its numbers are trusted; if judge/human disagree > 20%, H5 reverts to
  fully manual rating on a reduced sample (n=200) and the timeline absorbs it.

### 8.6 Deduplication (added v2.1, post-pilot)

The N=5000 pilot showed WildChat's writing subset is dominated by mass copy-paste
viral prompts (one Midjourney-prompt-generator jailbreak repeated verbatim hundreds
of times). Verbatim spam makes H1 degenerate (a few duplicate blobs swallow
everything; coverage → 1.0, noise → 0.0, control gap → 0.0) and inflates H3/H4
toward "we found copies of the same string," which a hashmap already solves. The
thesis is about *semantic* redundancy across *distinct* user phrasings, so
duplicates must be collapsed first.

- **Exact pass** (`dedup.exact_dedup`): collapse records whose case- and
  whitespace-normalized prompt is identical. Runs on the record pool *before* the
  writing filter and arm split. No embedding needed.
- **Near pass** (`dedup.near_dedup`): per arm, after embedding, greedily keep one
  representative per blob of prompts within cosine ≥ 0.98 of each other. Runs
  *before* any metric so H1/H3/H4 see deduplicated input. The cosine cutoff is
  deliberately high (0.98) so it removes only near-verbatim copies, not the
  semantic near-neighbors that H3/H5 are *supposed* to measure.
- Both passes keep the first occurrence (stable) so seeding holds.
- **The dedup rate is a reported finding**, not just preprocessing: "what fraction
  of WildChat 'writing' is verbatim/near-verbatim spam" is independently
  interesting and a stated deliverable (`headline_numbers.json["dedup"]`, §10).
- All three arms are deduplicated identically; the scrambled null collapses far
  less (token-shuffled prompts are not duplicates), which is itself the contrast.

---

## 9. Decision procedure

1. Compute H1–H4 and control arms.
2. Compute H5 (judge, audited).
3. **If H5 fails → Abandon.** Stop. Write `docs/findings.md`. Do not re-scope on a
   failed H5; a failed H5 means the substitutability premise is false for this
   domain, and narrowing the slice cannot manufacture substitutability.
4. If H5 passes, apply the §4 matrix using control-gap-adjusted Pass/Fail.
5. Record the decision and every threshold's provenance in `docs/decision_log.md`.

A negative result with clean methodology and a calibrated judge is a publishable
contribution — no one has measured substitutability (not just similarity) on WildChat
with this rigor. Document it as such in `docs/findings.md`.

---

## 10. Deliverables

| Artifact | Format | Location | Required |
| --- | --- | --- | --- |
| Cost model + derived thresholds | Markdown | `docs/cost_model.md` | Yes, before Day 3 |
| Headline numbers (incl. control gaps) | JSON | `results/latest/headline_numbers.json` | Yes |
| Judge transcripts | JSONL | `results/latest/judge_transcripts.jsonl` | Yes |
| Cluster examples write-up | Markdown | `results/latest/cluster_examples.md` | Yes |
| Figures | PNG | `results/latest/figures/` | Yes |
| Auto-generated summary | Markdown | `results/latest/summary.md` | Yes |
| Decision log entry | Markdown | `docs/decision_log.md` | Yes |
| Findings write-up | Markdown | `docs/findings.md` | Yes, post-decision |
| Architecture doc (if positive) | Markdown | `docs/architecture.md` | Conditional |

(`results/latest` above means the directory named in `results/latest.txt`.)

---

## 11. Timeline

Still a ~1-week study. H5 and the control add work; they replace, not supplement, the
v1 false confidence, so the budget holds. If it stretches past 2 weeks, scope has
expanded incorrectly.

| Day | Task |
| --- | --- |
| 1 | Repo setup, deps, HF auth, dataset cache, **cost model → docs/cost_model.md, thresholds frozen** |
| 2 | Pilot at N=5000, filter calibration (strict + recall), **pilot acceptance test for N** |
| 3 | Full run at N=50,000, all three arms, H1–H4 + figures |
| 4 | Judge run for H2 + H5, 100-pair judge spot-check |
| 5 | Manual cluster inspection, control-gap analysis, apply §4 rubric, decision_log |
| 6–7 | If positive, draft architecture doc. If H5 failed / negative, write findings and stop. |

---

## 12. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| LMSYS license delay | Medium | Low | WildChat is the (contaminated) fallback; pipeline supports both |
| Strict filter biases subset toward H1/H3 | High | High | Recall-oriented second filter; rubric uses conservative numbers (8.2) |
| Verbatim copy-paste spam dominates metrics | Confirmed on WildChat | High | Exact + near dedup (8.6); WildChat demoted, LMSYS primary (7) |
| cos≥0.98 dedup misses viral templates (0.90–0.97) | Confirmed (re-pilot) | High | Dataset switch to LMSYS (7); revisit template-family collapse if LMSYS also shows it |
| Degenerate clustering scored as H1 pass | Confirmed (re-pilot) | High | Symmetric degeneracy guard: flat-1.0/0-noise envelope → H1 Fail (8.4) |
| LMSYS Arena skew (model-stress prompts, not organic) | Medium | Medium | Named in §7/findings; comparability-to-OpenAI claim withdrawn |
| MiniLM cosine ≠ substitutability | High | High | S3 calibrated from H5, not asserted; H5 gates the decision (8.3, 8.5) |
| HDBSCAN finds no separable structure | Medium | High | UMAP pre-reduction; param sweep; "not separable" ≠ "no structure" (8.4) |
| Judge unreliable | Medium | High | 100-pair human spot-check; fallback to manual n=200 (8.5) |
| WildChat (free consumer) ≠ enterprise motivation | High | Medium | Named as external-validity limit in findings; do not over-claim transfer |
| Embeddings too generic | Low | Medium | Decision rests on H5/judge, which is embedding-independent |
| Ambiguous H1–H4 with H5 pass | Medium | Low | §4 matrix covers it; default re-scope |
| Cost-model constants wrong | Medium | High | Constants frozen in `docs/cost_model.md` Day 1; sensitivity table required |
| **User pivots before running the experiment** | High | Critical | Pre-committed rubric; H5 gate; the rubric, not vibes, decides |

The last risk remains the most likely failure mode. If the experiment is deferred or
skipped, the study has failed regardless of any other outcome.

---

## 13. Out of scope

- No fine-tuning. No multi-turn analysis (first-turn only). No cross-lingual analysis
  (English only). No cross-provider comparison (one dataset at a time). No live API
  testing beyond the judge calls required for H2/H5. No system implementation — this
  study informs the architecture; it is not the architecture.

Anything here can be a follow-up after the core decision.

---

## 14. Next steps after this PRD

1. Initialize the repository with the structure in §5.
2. Write the cost model (`notebooks/00_cost_model.ipynb` → `docs/cost_model.md`),
   plug in real prices, freeze T1/T3/S3/T5. **Do this before any pipeline run.**
3. Implement `src/redundancy/` from scratch (no prior `wildchat_redundancy_experiment.py`
   exists) per the methodology in §8.
4. Run the pilot at N=5000; run the pilot acceptance test for N (§7).
5. Calibrate both filters in `02_filter_calibration.ipynb`.
6. Execute the full run at N=50,000 across all three arms.
7. Run the judge for H2 and H5; do the 100-pair spot-check.
8. Apply the §9 decision procedure (H5 gate first).
9. Update `docs/decision_log.md` and `docs/findings.md`.
10. Either write `docs/architecture.md` (positive) or document and stop (negative).

Do not skip steps. Do not start the architecture before the numbers are in. Do not
freeze thresholds after seeing results. Do not re-scope on a failed H5. Do not start
a new project idea before this one is resolved.

---

## 15. Multi-turn predictability (added v2.2)

**Status:** Investigation, not a re-decision. The first-turn semantic-cache
study reached Abandon per §9; that verdict stands. §15 measures a *different
mechanism* — speculative pre-generation — to scope a possible follow-on
project. No rubric, no Abandon framing.

**Question.** Given a user's turn N, how often does the actual turn N+1 lie
within cosine T of the *responses you would have pre-generated* by looking up
the K nearest turn N\* in OTHER conversations? Predictability above the random
baseline is necessary (not sufficient) for speculative pre-generation to pay
off; substitutability (the H5 result, ~13–15% in the highest band) still
applies on top.

**Metric.** Hit@K@T per (K ∈ {1, 5, 10, 50}, T ∈ {0.7, 0.8, 0.9}). Computed for
both a MiniLM retriever (consistent with the rest of the study) and a TF-IDF
retriever (the "graph-similarity" arm — lexical-overlap similarity over the
bipartite prompt-token graph, contrasting MiniLM's smoothed-out semantics).
Reported with two controls: random K\*-indices (does retrieval beat guessing?)
and within-conversation `cosine(turn_N, turn_{N+1})` (would within-session
caching be the easier mechanism?). Quality at hit is NOT judged in this
investigation; §15 measures only "did the retrieval predict close to the
right next prompt." Combine with the §3 H5 result for the full feasibility
picture.

**Data.** Re-loaded LMSYS-Chat-1M keeping full conversations (§13's first-turn
restriction is amended here). Seeded reservoir sample of N conversations with
≥2 English user turns. New parquet cache `lmsys_multiturn_n*.parquet`.

**Implementation.** `src/redundancy/multi_turn.py` (loader wrapper, pair
extract, MiniLM + TF-IDF kNN, hit@K@T, controls, CLI). Same-conv neighbors are
filtered before counting. CLI: `python -m redundancy.multi_turn --n 5000`.

**Reading the result.** Lift over random = the retrieval signal. If MiniLM and
TF-IDF lifts converge at zero, prediction is not feasible on this corpus. If
TF-IDF beats MiniLM, the "graph-similarity is a better predictor than
embedding-cosine" hypothesis gets traction. Same-conv consecutive cosine high
would suggest the within-session caching variant is the practical line.
