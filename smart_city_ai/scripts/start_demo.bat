@echo off
REM Start the Streamlit dashboard (Windows wrapper)
set SCRIPT_DIR=%~dp0
python -m streamlit run "%SCRIPT_DIR%..\dashboards\app.py"
