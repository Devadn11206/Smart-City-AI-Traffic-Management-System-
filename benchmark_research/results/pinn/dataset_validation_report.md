# Dataset Validation Report
**Total Records:** 142560

## Missing Values
No missing values found.

## Descriptive Statistics
|                  |   count |          mean |           std |            min |           25% |           50% |            75% |           max |   unique_values |
|:-----------------|--------:|--------------:|--------------:|---------------:|--------------:|--------------:|---------------:|--------------:|----------------:|
| traffic_density  |  142560 |     0.0495832 |     0.0219934 |    0.000916422 |     0.0336022 |     0.0479789 |      0.0647605 |      0.112238 |             989 |
| queue_length     |  142560 |   139.679     |    77.0393    |    0           |    77         |   132         |    195         |    365        |             366 |
| throughput       |  142560 |     3.47666   |     3.70734   |    0           |     0         |     3         |      6         |     29        |              27 |
| avg_speed        |  142560 |     4.8792    |     2.35819   |    0           |     3.12412   |     4.77052   |      6.5358    |     13.661    |           47162 |
| fuel_consumption |  142560 | 84397.4       | 35018.1       | 2369.16        | 59713         | 82375.1       | 108133         | 187058        |           47265 |
| current_phase    |  142560 |     2.42525   |     2.05111   |    0           |     0         |     2         |      4         |      6        |               4 |
| phase_duration   |  142560 |    46.3298    |   192.724     |   10           |    10         |    10         |     20         |   2510        |             251 |

## Variance Check
All features possess sufficient variance (no constant columns).

## Outlier Report (Z-score > 3)
|    | Feature          |   Outlier Count |
|---:|:-----------------|----------------:|
|  0 | traffic_density  |               0 |
|  1 | queue_length     |               0 |
|  2 | throughput       |            1761 |
|  3 | avg_speed        |              81 |
|  4 | fuel_consumption |               0 |
|  5 | current_phase    |               0 |
|  6 | phase_duration   |            2589 |

## Feature Correlation Matrix
![Correlation Matrix](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\correlation_matrix.png)

## Feature Distributions
![traffic_density](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_traffic_density.png)
![queue_length](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_queue_length.png)
![throughput](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_throughput.png)
![avg_speed](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_avg_speed.png)
![fuel_consumption](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_fuel_consumption.png)
![current_phase](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_current_phase.png)
![phase_duration](C:\Users\ddnan\intenrship_IIIT\benchmark_research\results\pinn\diagnostics\hist_phase_duration.png)