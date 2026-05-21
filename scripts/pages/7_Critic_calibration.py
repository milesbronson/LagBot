"""Critic calibration — V(s_0) vs G_0 scatter + residual drift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from scripts._dashboard_lib import select_run, setup_sidebar  # noqa: E402
from src.training.registry_report import (  # noqa: E402
    load_value_calibration, summary_table,
)


st.set_page_config(page_title="Critic calibration", layout="wide")
registry, metrics_dir = setup_sidebar()
rows = summary_table(registry, metrics_dir)
selected_run = select_run(rows)

st.title(f"Critic calibration — {selected_run}")
calib = load_value_calibration(selected_run, metrics_dir) if selected_run else None
if not calib or not calib.get("pairs"):
    st.warning("No value_calibration.json yet for this run.")
    st.stop()

pairs = calib["pairs"]
v = np.array([p["value"] for p in pairs])
g = np.array([p["actual_return"] for p in pairs])
bias = float((v - g).mean())
mae = float(np.abs(v - g).mean())

st.caption(
    f"gamma = {calib.get('gamma', '?')}   "
    f"n = {len(pairs)}   "
    f"mean bias (V − G_0) = {bias:+.4f}   "
    f"MAE = {mae:.4f}"
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("V(s_0) vs G_0")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(v, g, s=10, alpha=0.4)
    lo, hi = float(min(v.min(), g.min())), float(max(v.max(), g.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="V = G_0 (ideal)")
    ax.set_xlabel("predicted V(s_0)")
    ax.set_ylabel("actual discounted return G_0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.subheader("Residual (V − G_0) over training time")
    ts = [p["timestep"] for p in pairs]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(ts, v - g, s=8, alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("V(s_0) − G_0")
    ax.set_title("Above 0 = critic over-estimates; below = under-estimates")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
