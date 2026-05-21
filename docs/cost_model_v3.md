# Prefetch ROI Model (PRD v3 §4a)

**Status:** PROVISIONAL — set real values before the decision run

Per-turn ROI of speculative prefetch, all costs normalized to `c_frontier = 1.0`:

```
roi_per_turn = hit_rate × quality_at_T_serve × latency_value_per_hit
             − K × c_speculate
```

## Constants

| Constant | Value | Meaning |
| --- | --- | --- |
| latency_value_per_hit | 0.3 | value (fraction of frontier cost) of one prefetch hit's latency win |
| c_speculate options | [0.2, 0.5, 1.0] | Haiku ~0.20, Sonnet ~0.50, Opus ~1.0 |
| K grid | [1, 3, 5, 10, 25] | candidates pre-generated per turn |
| T_pred grid | [0.7, 0.8, 0.9, 0.95] | hit threshold (predicted vs actual cosine) |
| T_serve grid | [0.9, 0.95, 0.98] | quality threshold (only serve cached if this close) |

## How HP4 reads this

HP4 passes iff there exists a (K, T_pred, T_serve, c_speculate) tuple with `roi_per_turn > 0`. The Pareto sweep (PRD §8.6) reports every cell; `prefetch.best_cell()` returns the max-ROI feasible cell (or None → Abandon on economic grounds even if HP1/HP2 passed).
