# Installation Guide

1. Install SUMO and set `SUMO_HOME`.
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure the trained PINN checkpoint exists at `pinn_models/trained_pinn.pth`.
4. Run a smoke simulation:

```bash
python integration/run_emergency_system.py --duration 20 --pinn pinn_models/trained_pinn.pth
```

5. Launch the dashboard:

```bash
streamlit run dashboards/app.py
```