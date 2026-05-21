"""Card deep-dive — pick a single card and inspect every signal we have:
metadata, behaviour radar, matchup history, profit-vs-opponent, per-run
training curves (if its training run produced metrics)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scripts._dashboard_lib import (  # noqa: E402
    card_vector, radar_plot, setup_sidebar, stat_distance,
)
from src.training.registry_report import load_run_metrics  # noqa: E402


st.set_page_config(page_title="Card deep dive", layout="wide")
registry, metrics_dir = setup_sidebar()

st.title("Card deep dive")
st.caption("Everything we know about one card in the registry.")

all_cards = registry.all()
if not all_cards:
    st.warning("Registry is empty.")
    st.stop()

# Sort PPO cards by generation desc, then alphabetical for everything else.
ppo_cards = sorted(
    [c for c in all_cards if c.kind == "ppo"],
    key=lambda c: (-c.generation, c.created_at),
)
other_cards = sorted(
    [c for c in all_cards if c.kind != "ppo"], key=lambda c: c.id,
)
options = [c.id for c in ppo_cards + other_cards]
default = st.session_state.get("deep_dive_card", options[0])
selected_id = st.sidebar.selectbox(
    "Card", options=options,
    index=options.index(default) if default in options else 0,
    key="deep_dive_card",
)
card = registry.get(selected_id)


# --- Header / metadata -----------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("kind", card.kind)
c2.metric("generation", card.generation)
c3.metric("hands observed", int((card.behavior_stats or {}).get("hands_observed", 0)))
c4.metric("matchups recorded", len(card.matchup_history or {}))

meta_rows = [
    ("id", str(card.id)),
    ("name", str(card.name)),
    ("parent", str(card.parent_id or "—")),
    ("path", str(card.path or "—")),
    ("created_at", str(card.created_at)),
    ("total timesteps", str(card.total_timesteps) if card.total_timesteps else "—"),
    ("trained against", ", ".join(card.trained_against_ids) or "—"),
]
st.table(pd.DataFrame(meta_rows, columns=["field", "value"]))


# --- Radar -----------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Behaviour-stat radar")
    bs = card.behavior_stats or {}
    if int(bs.get("hands_observed", 0)) == 0:
        st.info("No behaviour stats recorded yet.")
    else:
        anchors = [c for c in all_cards if c.kind == "anchor"]
        if anchors:
            d = {a.id: stat_distance(card, a) for a in anchors}
            nearest = min(d, key=d.get)
            nearest_card = registry.get(nearest)
            st.caption(f"Nearest anchor: **{nearest}** (d = {d[nearest]:.3f})")
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            radar_plot(
                ax,
                vectors=[card_vector(card), card_vector(nearest_card)],
                labels=[card.id, nearest],
                colors=["#1f77b4", "#d62728"],
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Seed anchors to enable nearest-anchor overlay.")

with right:
    st.subheader("Profit vs each observer")
    mh = card.matchup_history or {}
    if not mh:
        st.info("No matchups recorded against this card yet.")
    else:
        df_mh = pd.DataFrame([
            {
                "observer": obs_id,
                "hands": int(entry.get("hands_played", 0)),
                "avg_profit/hand": round(float(entry.get("avg_profit", 0.0)), 3),
                "total_profit": round(float(entry.get("total_profit", 0.0)), 1),
                "last_seen_ts": int(entry.get("last_seen_timestep", 0)),
            }
            for obs_id, entry in mh.items()
        ]).sort_values("avg_profit/hand", ascending=False)
        st.dataframe(df_mh, width="stretch", hide_index=True)

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df_mh["avg_profit/hand"]]
        ax.barh(df_mh["observer"], df_mh["avg_profit/hand"], color=colors)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("avg chip profit per hand (positive = observer wins)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close(fig)


# --- Training-run curves (if available) ------------------------------
st.subheader("Training curves for this card's run")
run_name = card.id
metrics = load_run_metrics(run_name, metrics_dir)
if not metrics or not metrics.get("timesteps"):
    st.info(f"No metrics.json found for run {run_name!r}.")
else:
    ts = metrics["timesteps"]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if metrics.get("rewards"):
            ax.plot(ts, metrics["rewards"], alpha=0.4, label="raw")
        if metrics.get("avg_reward_100"):
            ax.plot(ts, metrics["avg_reward_100"], linewidth=2, label="avg(100)")
        ax.set_title("Reward")
        ax.set_xlabel("step")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if metrics.get("win_rate"):
            ax.plot(ts, metrics["win_rate"], color="#2ca02c", linewidth=2)
            ax.set_ylim(0, 1)
        ax.set_title("Win rate")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
