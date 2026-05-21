"""Policy compare — pick two PPO cards, run them through the same
batch of hands, and compare action distributions side-by-side.

This is the headless A/B harness for the solver-style workflow: you
get to see whether two policies actually play differently, broken down
by street and by hole-card class.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scripts._dashboard_lib import (  # noqa: E402
    action_distribution, action_labels, load_ppo_model, setup_sidebar,
)
from src.poker_env.opponent_tracker import Street  # noqa: E402
from src.poker_env.texas_holdem_env import TexasHoldemEnv  # noqa: E402


st.set_page_config(page_title="Policy compare", layout="wide")
registry, _ = setup_sidebar()

st.title("Policy compare")
st.caption(
    "Run two PPO policies through the same sequence of hands. We log "
    "every decision (not just the sampled action) so the comparison "
    "reflects the underlying policy, not RNG noise."
)

ppo_cards = [c for c in registry.all() if c.kind == "ppo" and c.path]
if len(ppo_cards) < 2:
    st.warning("Need at least two PPO cards with saved paths in the registry.")
    st.stop()
ppo_ids = [c.id for c in ppo_cards]


# --- Inputs ----------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    a_id = st.selectbox("Policy A", ppo_ids, index=len(ppo_ids) - 1, key="cmp_a")
with c2:
    default_b = max(0, len(ppo_ids) - 2)
    b_id = st.selectbox("Policy B", ppo_ids, index=default_b, key="cmp_b")

c1, c2, c3 = st.columns(3)
with c1:
    num_hands = st.number_input("Hands to sample", min_value=10, max_value=2000, value=200, step=10)
with c2:
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42)
with c3:
    raise_bins_str = st.text_input("Raise bins (csv)", value="0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0")

raise_bins = [float(x) for x in raise_bins_str.split(",") if x.strip()]

if a_id == b_id:
    st.warning("Pick two different policies to compare.")
    st.stop()

run = st.button("Run comparison", type="primary")
if not run:
    st.info("Set the inputs above and hit **Run comparison**.")
    st.stop()


# --- Inference -------------------------------------------------------
a_path = registry.get(a_id).path
b_path = registry.get(b_id).path
with st.spinner("Loading models..."):
    model_a = load_ppo_model(a_path)
    model_b = load_ppo_model(b_path)


def _hole_card_bucket(env: TexasHoldemEnv, player_id: int) -> str:
    """Crude preflop hole-card class: 'premium', 'medium', 'trash'."""
    player = env.game_state.players[player_id]
    if len(player.hand) < 2:
        return "unknown"
    r1 = (player.hand[0] >> 8) & 0xF
    r2 = (player.hand[1] >> 8) & 0xF
    hi, lo = max(r1, r2), min(r1, r2)
    pair = r1 == r2
    if pair and hi >= 8:                   # TT+
        return "premium"
    if hi == 12 and lo >= 8:               # AT+
        return "premium"
    if pair:
        return "medium"
    if hi >= 10 and lo >= 8:               # KQ, KJ, KT, QJ, QT, JT
        return "medium"
    return "trash"


def _street_name(street_value: int) -> str:
    return {0: "preflop", 1: "flop", 2: "turn", 3: "river"}.get(street_value, "?")


def _run_batch():
    env = TexasHoldemEnv(
        num_players=2, starting_stack=1000,
        small_blind=5, big_blind=10,
        track_opponents=True, learning_agent_id=0,
        raise_bins=raise_bins,
    )
    n_actions = env.action_space.n
    if model_a.action_space.n != n_actions or model_b.action_space.n != n_actions:
        st.error(
            f"Action-space mismatch: env has {n_actions}, "
            f"A has {model_a.action_space.n}, B has {model_b.action_space.n}. "
            "Pick raise bins that match both models."
        )
        return None

    # Aggregations: probability per action, plus per-street and per-bucket.
    overall = {"A": np.zeros(n_actions), "B": np.zeros(n_actions)}
    per_street = {
        "A": {s: np.zeros(n_actions) for s in ("preflop", "flop", "turn", "river")},
        "B": {s: np.zeros(n_actions) for s in ("preflop", "flop", "turn", "river")},
    }
    per_bucket = {
        "A": defaultdict(lambda: np.zeros(n_actions)),
        "B": defaultdict(lambda: np.zeros(n_actions)),
    }
    counts = {"A": 0, "B": 0}
    per_street_count = {"A": defaultdict(int), "B": defaultdict(int)}
    per_bucket_count = {"A": defaultdict(int), "B": defaultdict(int)}
    profit = {"A": 0.0, "B": 0.0}

    progress = st.progress(0.0, text="Running hands...")
    for hand_idx in range(int(num_hands)):
        obs, _ = env.reset(seed=int(seed) + hand_idx)
        stack_before = env.game_state.players[0].starting_stack_this_hand
        done = False
        while not done:
            current = env.game_state.get_current_player()
            # Query BOTH policies on every decision the current seat
            # would actually make — but only ONE drives the env so the
            # game-tree we sample is consistent. Walk A; record B's
            # would-be choice probabilities on the SAME observation.
            pa = action_distribution(model_a, obs)
            pb = action_distribution(model_b, obs)
            tag = "A" if current.player_id == 0 else "B"

            overall["A"] += pa
            overall["B"] += pb
            counts["A"] += 1
            counts["B"] += 1

            street = _street_name(env.game_state.betting_round.value)
            per_street["A"][street] += pa
            per_street["B"][street] += pb
            per_street_count["A"][street] += 1
            per_street_count["B"][street] += 1

            if street == "preflop":
                bucket = _hole_card_bucket(env, current.player_id)
                per_bucket["A"][bucket] += pa
                per_bucket["B"][bucket] += pb
                per_bucket_count["A"][bucket] += 1
                per_bucket_count["B"][bucket] += 1

            # Drive the env using A as both seats so the hand sequence is
            # deterministic w.r.t. A. (Side-by-side play with seat-rotation
            # would require running 2 independent games and aligning.)
            action, _ = model_a.predict(obs, deterministic=False)
            obs, _, terminated, _, _ = env.step(int(action))
            done = terminated

        delta = env.game_state.players[0].stack - stack_before
        profit["A"] += delta
        profit["B"] -= delta
        progress.progress((hand_idx + 1) / int(num_hands), text=f"hand {hand_idx + 1}/{num_hands}")

    progress.empty()

    def _avg(d, n):
        return d / max(n, 1)

    return {
        "labels": action_labels(raise_bins),
        "overall": {k: _avg(overall[k], counts[k]) for k in ("A", "B")},
        "per_street": {
            k: {s: _avg(per_street[k][s], per_street_count[k][s])
                for s in per_street[k]}
            for k in ("A", "B")
        },
        "per_bucket": {
            k: {b: _avg(per_bucket[k][b], per_bucket_count[k][b])
                for b in per_bucket[k]}
            for k in ("A", "B")
        },
        "counts": counts,
        "profit": profit,
    }


result = _run_batch()
if result is None:
    st.stop()


# --- Render ----------------------------------------------------------
labels = result["labels"]


def _grouped_bar(ax, pa, pb, labels, title):
    x = np.arange(len(labels))
    w = 0.4
    ax.bar(x - w / 2, pa, w, label="A", color="#1f77b4")
    ax.bar(x + w / 2, pb, w, label="B", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("prob")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=8)


st.subheader("Overall action distribution")
fig, ax = plt.subplots(figsize=(10, 4))
_grouped_bar(ax, result["overall"]["A"], result["overall"]["B"], labels,
             f"Averaged over {result['counts']['A']} decisions")
st.pyplot(fig)
plt.close(fig)


st.subheader("Per-street action distribution")
cols = st.columns(2)
streets = ("preflop", "flop", "turn", "river")
for i, street in enumerate(streets):
    with cols[i % 2]:
        pa = result["per_street"]["A"][street]
        pb = result["per_street"]["B"][street]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        _grouped_bar(ax, pa, pb, labels, street.capitalize())
        st.pyplot(fig)
        plt.close(fig)


st.subheader("Preflop action by hole-card class")
buckets = ("premium", "medium", "trash")
cols = st.columns(3)
for i, b in enumerate(buckets):
    with cols[i]:
        pa = result["per_bucket"]["A"].get(b)
        pb = result["per_bucket"]["B"].get(b)
        if pa is None or pb is None or not np.any(pa) or not np.any(pb):
            st.info(f"No {b} hands sampled.")
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _grouped_bar(ax, pa, pb, labels, b)
        st.pyplot(fig)
        plt.close(fig)


# --- Summary stats ---------------------------------------------------
st.subheader("Headline numbers")
diff = result["overall"]["A"] - result["overall"]["B"]
df_diff = pd.DataFrame({
    "action": labels,
    "A prob": np.round(result["overall"]["A"], 3),
    "B prob": np.round(result["overall"]["B"], 3),
    "A − B": np.round(diff, 3),
}).sort_values("A − B", key=lambda s: s.abs(), ascending=False)
st.dataframe(df_diff, width="stretch", hide_index=True)
st.caption(
    f"Driver policy: A. Game tree was sampled by A's actions; B's "
    f"probabilities were recorded on the same observations. "
    f"A net chip delta over {int(num_hands)} hands: "
    f"{result['profit']['A']:+.0f}."
)
