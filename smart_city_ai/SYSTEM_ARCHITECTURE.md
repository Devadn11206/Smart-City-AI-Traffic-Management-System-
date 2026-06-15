# System Architecture

```mermaid
flowchart TD
    A[SUMO Traffic Simulation] --> B[Traffic State Collection]
    B --> C[PINN Future Congestion Prediction]
    C --> D[ACO Intelligent Routing]
    D --> E[RL Traffic Signal Optimization]
    E --> F[Emergency Green Corridor Coordination]
    F --> G[Dashboard Visualization]
    G --> H[Real-Time Smart City Analytics]
```

The system combines a SUMO digital twin, RL signal control, PINN congestion forecasting, ACO route optimization, and emergency corridor coordination. The dashboard reads the latest metrics and benchmark artifacts for presentation and evaluation.