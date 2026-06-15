#!/usr/bin/env python3
"""Start the project demo: launches the Streamlit dashboard.

Usage:
  python scripts/start_demo.py

This script is cross-platform and will invoke the Streamlit runner using the
current Python executable.
"""

import os
import sys
import subprocess


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(repo_root, "dashboards", "app.py")

    if not os.path.exists(app_path):
        print(f"ERROR: Dashboard app not found at: {app_path}")
        sys.exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    print("Launching demo with:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print("Streamlit exited with code:", e.returncode)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
