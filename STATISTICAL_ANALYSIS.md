# Statistical Analysis Summary

## 1. Multi-Scale RL Benchmark
Mean Waiting Times across algorithms in the final 10% of episodes:
```text
Network  Algorithm 
2x2      COLIGHT         301.544735
         DDQN            131.225492
         DQN             134.737424
         IDQN            137.016342
         PRESSLIGHT       66.084545
4x4      COLIGHT       17171.779896
         DDQN          15983.138636
         DQN           15685.424631
         IDQN          14770.933037
         PPO           12190.686989
         PRESSLIGHT    14587.779100
6x6      COLIGHT       14215.949495
         DDQN          13277.329221
         DQN           13237.361296
         IDQN          13756.000501
         PPO           11613.262056
         PRESSLIGHT    15447.219641
```

## 2. PINN vs RL Baselines
Percentage Improvements in Waiting Time for PINN-integrated models:
- **DQN 6x6**: 51.56% improvement in waiting time.
- **PPO 6x6**: -134.31% improvement in waiting time.
