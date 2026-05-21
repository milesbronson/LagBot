"""Play-styles page — radar charts, stratum membership, PCA scatter.

The behaviour-stat math comes from ``OpponentSampler`` so this view
agrees exactly with the production stratifier.
"""

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
    _AF_SCALE, _STAT_KEYS,
    card_vector, pca_2d, radar_plot, setup_sidebar, stat_distance,
)


st.set_page_config(page_title="Play styles", layout="wide")
registry, _ = setup_sidebar()

st.title("Play styles")
st.caption(
    "Behaviour-stat view of every card in the registry. Stat vector "
    "is what the stratifier uses to bucket cards by nearest anchor."
)

all_cards = registry.all()
anchors = [c for c in all_cards if c.kind == "anchor"]
observed = [
    c for c in all_cards
    if int((c.behavior_stats or {}).get("hands_observed", 0)) > 0
]

if not anchors:
    st.warning(
        "No anchor cards registered. Run `python scripts/seed_anchors.py` first."
    )
    st.stop()
if not observed:
    st.warning("No cards with recorded behaviour stats yet.")
    st.stop()


# --- Stratum membership ---------------------------------------------
st.subheader("Stratum membership")
rows = []
for c in observed:
    d = {a.id: stat_distance(c, a) for a in anchors}
    nearest = min(d, key=d.get)
    rows.append({
        "card": c.id,
        "kind": c.kind,
        "gen": c.generation,
        "nearest anchor": nearest,
        "distance": round(d[nearest], 3),
        "hands_observed": int(c.behavior_stats.get("hands_observed", 0)),
    })
df_strat = pd.DataFrame(rows).sort_values(["nearest anchor", "gen"])
col_t, col_h = st.columns([2, 1])
with col_t:
    st.dataframe(df_strat, width="stretch", hide_index=True)
with col_h:
    counts = df_strat["nearest anchor"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(counts.index, counts.values, color="#1f77b4")
    ax.set_xlabel("cards in stratum")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    st.pyplot(fig)
    plt.close(fig)


# --- 2D PCA ----------------------------------------------------------
st.subheader("2D PCA of behaviour-stat space")
st.caption("Anchors marked with stars; observed cards colored by stratum.")
with st.expander("What are PC1 and PC2?"):
    st.markdown(
        "Every card lives in a **9-dimensional behaviour-stat space** — "
        "the same vector the stratifier uses to bucket cards by nearest "
        "anchor:\n\n"
        "`vpip, pfr, three_bet_percent, cbet_percent, fold_to_cbet_percent, "
        "went_to_showdown_percent, win_at_showdown_percent, wwsf_percent, "
        "af_scaled` (`af / 10`, capped at 1).\n\n"
        "PCA finds the two orthogonal directions in that 9-D cloud along "
        "which the cards in *your registry* vary the most, then projects "
        "every card onto them.\n\n"
        "- **PC1** = direction of greatest spread across the cards you "
        "have. For poker stats this usually tracks **looseness + "
        "aggression** (high vpip / pfr / 3-bet / af → one end).\n"
        "- **PC2** = next-largest direction, orthogonal to PC1. Usually "
        "tracks **showdown style** — high WTSD / W$SD / WWSF (plays big "
        "pots) vs. low (avoids showdown), independent of pure "
        "looseness.\n\n"
        "The axes are unitless. What matters is **relative distance** "
        "between points: close together = plays similarly across all 9 "
        "stats; far apart = different style. The loadings depend on the "
        "specific registry — if everything collapses near one cluster, "
        "your training has converged toward one archetype."
    )

anchor_vecs = np.array([card_vector(a) for a in anchors])
obs_vecs = np.array([card_vector(c) for c in observed])
proj = pca_2d(np.vstack([anchor_vecs, obs_vecs]))
proj_a, proj_o = proj[: len(anchors)], proj[len(anchors):]

anchor_index = {a.id: i for i, a in enumerate(anchors)}
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(8, 5))
for i, c in enumerate(observed):
    d = {a.id: stat_distance(c, a) for a in anchors}
    nearest = min(d, key=d.get)
    ax.scatter(
        proj_o[i, 0], proj_o[i, 1],
        color=cmap(anchor_index[nearest] % 10),
        s=30, alpha=0.7,
    )
for i, a in enumerate(anchors):
    ax.scatter(
        proj_a[i, 0], proj_a[i, 1],
        color=cmap(i % 10), marker="*", s=240,
        edgecolor="black", linewidth=0.8,
        label=a.id.replace("anchor_", "").replace("_v0", ""),
    )
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc="best", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)


# --- Per-card radar --------------------------------------------------
st.subheader("Radar chart per card")
st.caption(
    "All axes normalised to [0, 1]; aggression factor divided by "
    f"{_AF_SCALE:.0f} and capped at 1."
)
col_pick, col_overlay = st.columns(2)
with col_pick:
    selected_card = st.selectbox(
        "Card", options=[c.id for c in observed],
        index=len(observed) - 1,
    )
with col_overlay:
    overlay_anchor = st.selectbox(
        "Overlay anchor",
        options=["(nearest)"] + [a.id for a in anchors],
    )
sel = registry.get(selected_card)
sel_vec = card_vector(sel)
if overlay_anchor == "(nearest)":
    d = {a.id: stat_distance(sel, a) for a in anchors}
    overlay_id = min(d, key=d.get)
else:
    overlay_id = overlay_anchor
anc = registry.get(overlay_id)
anc_vec = card_vector(anc)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
radar_plot(
    ax,
    vectors=[sel_vec, anc_vec],
    labels=[selected_card, overlay_id],
    colors=["#1f77b4", "#d62728"],
    title=(
        f"{selected_card}  vs  {overlay_id}  "
        f"(d = {float(np.linalg.norm(sel_vec - anc_vec)):.3f})"
    ),
)
st.pyplot(fig)
plt.close(fig)

with st.expander("All anchors on one radar"):
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = [cmap(i % 10) for i in range(len(anchors))]
    radar_plot(
        ax,
        vectors=[card_vector(a) for a in anchors],
        labels=[a.id.replace("anchor_", "").replace("_v0", "") for a in anchors],
        colors=colors,
        title="Anchor archetypes",
    )
    st.pyplot(fig)
    plt.close(fig)

with st.expander("Raw behaviour stats"):
    keys = ["hands_observed"] + list(_STAT_KEYS) + ["af"]
    rows_stats = []
    for c in observed + anchors:
        row = {"card": c.id, "kind": c.kind, "gen": c.generation}
        bs = c.behavior_stats or {}
        for k in keys:
            row[k] = (int(bs.get(k, 0)) if k == "hands_observed"
                      else round(float(bs.get(k, 0.0)), 3))
        rows_stats.append(row)
    st.dataframe(pd.DataFrame(rows_stats), width="stretch", hide_index=True)
