"""Regenerate every paper figure into figures/ in the tone of Fig. 2.

    python figures/src/make_all.py

Inputs are read from results/ (and harness/results/x25, x26 for Fig. S4's selected epochs); nothing there is
written. The per-experiment scripts under results/exp*/ are left as they were.
"""
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
for script in ["make_fig1.py", "make_fig2.py", "make_fig3.py", "make_fig4.py", "make_fig5.py"]:
    print(f"==== {script}")
    runpy.run_path(str(SRC / script), run_name="__main__")
