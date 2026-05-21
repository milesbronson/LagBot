"""Per-street action breakdown — fold/call/raise/all-in stacked over
training time, one panel per street."""

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
    load_street_breakdown, summary_table,
)


st.set_page_config(page_title="Per-street actions", layout="wide")
registry, metrics_dir = setup_sidebar()
rows = summary_table(registry, metrics_dir)
selected_run = select_run(rows)

st.title(f"Per-street action breakdown — {selected_run}")
sb = load_street_breakdown(selected_run, metrics_dir) if selected_run else None
if not sb or not sb.get("timesteps"):
    st.warning("No street_breakdown.json yet for this run.")
    st.stop()

ts = sb["timesteps"]
streets = ("preflop", "flop", "turn", "river")
actions = ("fold", "call", "raise", "all_in")
action_colors = {
    "fold": "#d62728",
    "call": "#7f7f7f",
    "raise": "#1f77b4",
    "all_in": "#ff7f0e",
}

col1, col2 = st.columns(2)
for i, street in enumerate(streets):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.subheader(street.capitalize())
        fig, ax = plt.subplots(figsize=(6, 3.5))
        stack = np.array([
            [snap[street][a] for snap in sb["distributions"]]
            for a in actions
        ])
        ax.stackplot(
            ts, stack,
            labels=actions,
            colors=[action_colors[a] for a in actions],
            alpha=0.85,
        )
        ax.set_xlabel("training step")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
