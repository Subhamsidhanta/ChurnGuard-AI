"""Canonical Streamlit dashboard entry point."""
# Streamlit executes this file as a standalone script, so importing
# ``app.dashboard`` would collide with this file's ``app`` module name.
from pathlib import Path
import runpy
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

runpy.run_path(str(Path(__file__).with_name("dashboard.py")), run_name="__main__")
