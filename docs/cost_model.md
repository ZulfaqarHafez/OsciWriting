# Cost Model and Derived Thresholds (PRD §4a)

**Status:** FROZEN

**Source:** Anthropic API pricing, May 2026: Opus 4.7 = $5/$25 per MTok (frontier), Haiku 4.5 = $1/$5 per MTok (small). 1:5 ratio across both input and output → `c_small = 0.20` with no token-mix assumption.

## Constants (normalized to c_frontier = 1.0)

| Constant | Value | Meaning |
| --- | --- | --- |
| c_cache | 0.002 | one cache lookup (embed + vector search) |
| c_small | 0.2 | small-model cost as fraction of frontier |
| p_small | 0.5 | fraction of cache-misses routed to small model |
| s_target | 0.5 | required savings to call the project worthwhile |

## Derived thresholds

- **T1 (H1 top-50 coverage)** = `0.17`
- **T3 (H3 fraction at >= S3)** = `0.17`
- **S3** = `0.9` (calibrated from H5 at run time; this is the PRD default)
- **T5 (H5 acceptability, gating)** = `0.7` (PRD §3 default)
- feasible: `True` — min viable cache-hit rate = 0.1700 at p_small=0.5

## Sensitivity (min viable cache-hit rate)

| c_small | p_small | min p_cache | feasible |
| --- | --- | --- | --- |
| 0.02 | 0.25 | 0.3404 | True |
| 0.02 | 0.5 | 0.0235 | True |
| 0.02 | 0.75 | 0.0 | True |
| 0.05 | 0.25 | 0.3469 | True |
| 0.05 | 0.5 | 0.0514 | True |
| 0.05 | 0.75 | 0.0 | True |
| 0.1 | 0.25 | 0.3574 | True |
| 0.1 | 0.5 | 0.0945 | True |
| 0.1 | 0.75 | 0.0 | True |
| 0.2 | 0.25 | 0.3775 | True |
| 0.2 | 0.5 | 0.17 | True |
| 0.2 | 0.75 | 0.0 | True |

## How this binds the rubric

T1 and T3 above are not chosen by feel; they are the minimum cache-hit rate that makes blended cost clear `s_target`. If the writing subset cannot reach T1/T3 (above control, PRD §4), the economics do not work and the rubric says so. Re-run this module after replacing the constants with real pricing and commit the frozen version before the decision run.
