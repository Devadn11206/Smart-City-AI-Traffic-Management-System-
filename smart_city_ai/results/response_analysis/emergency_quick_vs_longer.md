# Emergency Run Comparison: quick_test vs longer_test

Run names: `quick_test` (20 steps) vs `longer_test` (25 steps)

## Summary Metrics

| Metric | quick_test (final) | longer_test (final) |
|---|---:|---:|
| Emergency vehicles detected | 3 | 2 |
| Active corridors (TLS) | 3 | 3 |
| Blocked edges | 0 | 0 |
| Avg speed (m/s) | 13.65 | 13.89 |
| Queue | 0.05 | 0.00 |
| Wait (s) | 0.00 | 0.00 |
| Throughput | 0.00 | 0.00 |
| Corridor efficiency | 0.94 | 0.60 |
| ETA estimate (s) | 160.78 | 160.45 |
| Delay probability | 0.16 | 1.00 |
| Route risk | 0.08 | 0.50 |

## Observations
- Both runs detected active emergency vehicles and created synchronized corridors.
- Average speeds are similar; `quick_test` shows slightly lower queue but higher corridor efficiency.
- `longer_test` produced a higher delay probability and route risk in its single checkpoint (likely due to different active vehicles and routing choices during that run).

## Files
- `results/emergency_metrics/quick_test_metrics.csv`
- `results/emergency_metrics/quick_test_latest.json`
- `results/emergency_metrics/longer_test_metrics.csv`
- `results/emergency_metrics/longer_test_latest.json`

## Next Actions
- (Optional) Generate comparative plots (`queue`, `avg_speed`, `corridor_efficiency`, `eta`) using the two CSVs.
- Run a full evaluation (e.g., 1800s) to produce statistically meaningful comparisons.
