Demo: Launching the Streamlit Dashboard

Quick start

1. Install dependencies (recommended in a venv):

```
python -m pip install -r requirements.txt
```

2. Ensure SUMO is installed and `SUMO_HOME` is set in your environment.

3. Launch the demo (choose one):

- Cross-platform Python launcher:

```
python scripts/start_demo.py
```

- Windows batch wrapper:

```
scripts\start_demo.bat
```

- Direct Streamlit invocation:

```
python -m streamlit run dashboards/app.py
```

Notes

- The launcher uses the current Python executable to run Streamlit so it inherits
  the active virtual environment.
- For long-running SUMO scenarios, the dashboard can launch background jobs but
  running heavy simulations locally may take significant time.
