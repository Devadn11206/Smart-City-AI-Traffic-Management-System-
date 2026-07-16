# 4x4 Benchmark Final Audited Paper Summary

This table presents the normalized weighted scoring out of 7.0, along with the corrected 'Best Episode' computation using a 10-episode rolling average to filter initialization artifacts.

| Rank | Algorithm | Weighted Score (/7.0) | Best Episode | Reward Variance | Wait (s) | Queue | Throughput | Travel Time (s) | Fuel (ml) | Speed (m/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **DQN** | 5.48 | 6 | 3738.6 | 15266.8 | 126.7 | 16385.9 | 871.8 | 4.23e+09 | 2.79 |
| 2 | **DDQN** | 5.00 | 22 | 3131.5 | 15267.6 | 127.5 | 16446.4 | 875.0 | 4.27e+09 | 2.76 |
| 3 | **IDQN** | 4.82 | 1 | 3433.1 | 15501.2 | 128.4 | 16751.5 | 862.2 | 4.29e+09 | 2.76 |
| 4 | **PPO** | 4.72 | 1 | 2456.1 | 12175.8 | 127.8 | 16458.7 | 992.3 | 4.21e+09 | 2.59 |
| 5 | **PRESSLIGHT** | 4.25 | 136 | 153122214.5 | 13739.7 | 125.0 | 17926.2 | 896.5 | 4.23e+09 | 2.44 |
| 6 | **COLIGHT** | 1.87 | 2 | 43942027.3 | 17035.1 | 134.0 | 14235.3 | 913.6 | 4.32e+09 | 2.63 |
