"""Scenario lab — manually compose a decision point, feed it to one
or more policies, and visualise their full action distribution.

Inputs you can dial in:
- Hole cards for hero
- Board (0, 3, 4, or 5 cards)
- Stacks for hero and villain (as fraction of starting stack)
- Position (button or BB)
- Override the opponent tracker stats the policy sees (VPIP / PFR /
  AF / 3-bet% / etc.) — this drives the 12-feature opponent block in
  the 161-dim observation.

The dashboard builds a fresh TexasHoldemEnv, sets the state, then
calls each selected policy's get_distribution() — so what you see is
the policy's *full softmax* over the action space, not just the
sampled action.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from treys import Card  # noqa: E402

from scripts._dashboard_lib import (  # noqa: E402
    action_distribution, action_labels, get_observation, load_ppo_model,
    setup_sidebar,
)
from src.poker_env.texas_holdem_env import TexasHoldemEnv  # noqa: E402


st.set_page_config(page_title="Scenario lab", layout="wide")
registry, _ = setup_sidebar()

st.title("Scenario lab")
st.caption(
    "Build a specific decision point, then ask one or more policies "
    "what they would do. Useful for sanity-checking a policy's view "
    "of premium hands, marginal spots, or how opponent profiles "
    "(VPIP / PFR / AF) change its decisions."
)

ppo_cards = [c for c in registry.all() if c.kind == "ppo" and c.path]
if not ppo_cards:
    st.warning("No PPO cards with paths in the registry.")
    st.stop()


# --- Policy picker ---------------------------------------------------
picked_ids = st.multiselect(
    "Policies to query",
    options=[c.id for c in ppo_cards],
    default=[ppo_cards[-1].id],
)
if not picked_ids:
    st.info("Pick at least one policy.")
    st.stop()


# --- Cards -----------------------------------------------------------
RANKS = list("23456789TJQKA")
SUITS = ["s", "h", "d", "c"]
SUIT_GLYPH = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_COLOR = {"s": "#222", "c": "#222", "h": "#d62728", "d": "#d62728"}
CARD_STRS = [r + s for r in RANKS for s in SUITS]


def _card_picker(label: str, default: str, key: str, used: set) -> str:
    """Rank-row × suit-row grid picker. Replaces the 52-deep selectbox.

    Persists the selection in ``st.session_state[key]``. Disables any
    rank/suit combo already used elsewhere on the page so it's impossible
    to pick a duplicate.
    """
    if key not in st.session_state:
        st.session_state[key] = default
    current = st.session_state[key]
    cur_rank, cur_suit = current[0], current[1]

    st.markdown(
        f"**{label}** &nbsp; "
        f"<span style='font-size:1.2em;color:{SUIT_COLOR[cur_suit]}'>"
        f"{cur_rank}{SUIT_GLYPH[cur_suit]}</span>",
        unsafe_allow_html=True,
    )
    # Rank row.
    rcols = st.columns(13)
    for i, r in enumerate(RANKS):
        candidate = r + cur_suit
        disabled = candidate != current and candidate in used
        if rcols[i].button(
            r, key=f"{key}_r_{r}",
            type=("primary" if r == cur_rank else "secondary"),
            disabled=disabled,
            width="stretch",
        ):
            st.session_state[key] = r + cur_suit
            st.rerun()
    # Suit row.
    scols = st.columns(4)
    for i, s in enumerate(SUITS):
        candidate = cur_rank + s
        disabled = candidate != current and candidate in used
        if scols[i].button(
            SUIT_GLYPH[s], key=f"{key}_s_{s}",
            type=("primary" if s == cur_suit else "secondary"),
            disabled=disabled,
            width="stretch",
        ):
            st.session_state[key] = cur_rank + s
            st.rerun()
    return st.session_state[key]


st.subheader("Cards")
board_n = st.radio(
    "Board cards", options=[0, 3, 4, 5], index=0, horizontal=True, key="board_n",
)

slot_keys = ["h1", "h2"] + [f"b{i}" for i in range(board_n)]
slot_labels = ["Hole 1", "Hole 2"] + [f"Board {i+1}" for i in range(board_n)]
slot_defaults = ["Ah", "Kh", "Qs", "Jd", "2c", "7h", "Td"][: len(slot_keys)]

# Pre-compute which cards are already taken by *other* slots so the picker
# can disable them. We need each slot's current value first.
current_vals = {k: st.session_state.get(k, d) for k, d in zip(slot_keys, slot_defaults)}

ncols = 2 if board_n == 0 else (3 if board_n == 3 else 4)
rows_of_cols = []
for start in range(0, len(slot_keys), ncols):
    rows_of_cols.append(st.columns(ncols))

picks = {}
for i, (k, lbl, d) in enumerate(zip(slot_keys, slot_labels, slot_defaults)):
    row = rows_of_cols[i // ncols]
    col = row[i % ncols]
    used = {v for k2, v in current_vals.items() if k2 != k}
    with col:
        picks[k] = _card_picker(lbl, d, k, used)

hole1 = picks["h1"]
hole2 = picks["h2"]
board_cards = [picks[f"b{i}"] for i in range(board_n)]
all_chosen = [hole1, hole2] + board_cards

# Defence-in-depth: picker should prevent dupes, but if state got out of
# sync (e.g. user shrank the board after picking) just dedupe silently.
if len(set(all_chosen)) != len(all_chosen):
    st.error("Duplicate cards selected — pick distinct cards.")
    st.stop()


# --- Stack / position / blinds --------------------------------------
st.subheader("Stack & position")
c1, c2, c3, c4 = st.columns(4)
with c1:
    starting_stack = st.number_input("Starting stack", min_value=100, max_value=10_000, value=1000, step=100)
with c2:
    hero_stack = st.number_input("Hero stack", min_value=0, max_value=starting_stack * 2, value=int(starting_stack), step=10)
with c3:
    villain_stack = st.number_input("Villain stack", min_value=0, max_value=starting_stack * 2, value=int(starting_stack), step=10)
with c4:
    big_blind = st.number_input("Big blind", min_value=1, max_value=1000, value=10, step=1)
small_blind = big_blind // 2

c1, c2 = st.columns(2)
with c1:
    hero_position = st.selectbox("Hero position", options=["button (SB)", "big blind"], index=0)
with c2:
    raise_bins_str = st.text_input("Raise bins (csv)", value="0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0")
raise_bins = [float(x) for x in raise_bins_str.split(",") if x.strip()]


# --- Villain profile override ---------------------------------------
st.subheader("Villain profile (drives the opponent-tracker block of the obs)")
st.caption(
    "These feed the same 12-feature block the policy saw during "
    "training. Anchor presets snap the sliders to a known archetype."
)
anchors = [c for c in registry.all() if c.kind == "anchor"]
preset = st.selectbox(
    "Preset",
    options=["(custom)"] + [a.id for a in anchors],
    index=0,
)
if preset != "(custom)" and registry.get(preset):
    base = registry.get(preset).behavior_stats
else:
    base = {
        "vpip": 0.5, "pfr": 0.3, "af": 2.0, "three_bet_percent": 0.1,
        "cbet_percent": 0.5, "fold_to_cbet_percent": 0.5,
        "went_to_showdown_percent": 0.3, "win_at_showdown_percent": 0.5,
        "wwsf_percent": 0.4, "fold_to_3bet_after_raise_percent": 0.5,
        "squeeze_percent": 0.05, "hands_observed": 200,
    }

c1, c2, c3 = st.columns(3)
with c1:
    vpip = st.slider("VPIP", 0.0, 1.0, float(base.get("vpip", 0.5)), 0.01)
    pfr = st.slider("PFR", 0.0, 1.0, float(base.get("pfr", 0.3)), 0.01)
    three_bet = st.slider("3-bet %", 0.0, 1.0, float(base.get("three_bet_percent", 0.1)), 0.01)
    cbet = st.slider("C-bet %", 0.0, 1.0, float(base.get("cbet_percent", 0.5)), 0.01)
with c2:
    fold_cbet = st.slider("Fold to C-bet %", 0.0, 1.0, float(base.get("fold_to_cbet_percent", 0.5)), 0.01)
    wtsd = st.slider("Went to showdown %", 0.0, 1.0, float(base.get("went_to_showdown_percent", 0.3)), 0.01)
    wsd = st.slider("Win at showdown %", 0.0, 1.0, float(base.get("win_at_showdown_percent", 0.5)), 0.01)
    wwsf = st.slider("WWSF %", 0.0, 1.0, float(base.get("wwsf_percent", 0.4)), 0.01)
with c3:
    af = st.slider("AF (aggression factor)", 0.0, 50.0, float(base.get("af", 2.0)), 0.1)
    fold_3bet = st.slider("Fold to 3-bet after raise %", 0.0, 1.0, float(base.get("fold_to_3bet_after_raise_percent", 0.5)), 0.01)
    squeeze = st.slider("Squeeze %", 0.0, 1.0, float(base.get("squeeze_percent", 0.05)), 0.01)
    hands_observed = st.number_input("Villain hands observed", min_value=0, max_value=10_000, value=int(base.get("hands_observed", 200)))

villain_priors = {
    "hands_observed": int(hands_observed),
    "vpip": vpip, "pfr": pfr, "af": af,
    "three_bet_percent": three_bet, "cbet_percent": cbet,
    "fold_to_cbet_percent": fold_cbet,
    "went_to_showdown_percent": wtsd, "win_at_showdown_percent": wsd,
    "wwsf_percent": wwsf,
    "fold_to_3bet_after_raise_percent": fold_3bet,
    "squeeze_percent": squeeze,
}


# --- Build env state ------------------------------------------------
def _build_env():
    env = TexasHoldemEnv(
        num_players=2, starting_stack=int(starting_stack),
        small_blind=int(small_blind), big_blind=int(big_blind),
        track_opponents=True, learning_agent_id=0,
        raise_bins=raise_bins,
    )
    env.reset(seed=0)

    # Override stacks and cards. Hero = player 0 with chosen position.
    hero_pid = 0
    villain_pid = 1
    env.game_state.button_position = hero_pid if "button" in hero_position else villain_pid

    env.game_state.players[hero_pid].hand = [Card.new(hole1), Card.new(hole2)]
    # Pick a benign placeholder for villain — equity calc doesn't see it
    # but we need 2 cards so internal invariants hold.
    used = set(all_chosen)
    placeholder = []
    for c in CARD_STRS:
        if c not in used:
            placeholder.append(c)
            used.add(c)
            if len(placeholder) == 2:
                break
    env.game_state.players[villain_pid].hand = [Card.new(p) for p in placeholder]

    env.game_state.players[hero_pid].stack = int(hero_stack)
    env.game_state.players[villain_pid].stack = int(villain_stack)

    env.game_state.community_cards = [Card.new(c) for c in board_cards]

    # Tell the env it's hero's turn.
    env.game_state.current_player_idx = hero_pid

    # Seed villain priors so the opponent block of the obs reflects
    # the slider values.
    env.opponent_tracker.seed_priors({villain_pid: villain_priors})
    return env


env = _build_env()


# --- Sanity card grid -----------------------------------------------
st.markdown("**Scenario summary**")
st.write({
    "hero hole": [hole1, hole2],
    "board": board_cards,
    "hero stack": int(hero_stack),
    "villain stack": int(villain_stack),
    "blinds": f"{int(small_blind)}/{int(big_blind)}",
    "hero position": hero_position,
})


# --- Inference -------------------------------------------------------
st.subheader("Policy action distributions")
labels = action_labels(raise_bins)
obs = get_observation(env)

cols = st.columns(min(len(picked_ids), 3))
for i, pid in enumerate(picked_ids):
    col = cols[i % len(cols)]
    with col:
        card = registry.get(pid)
        try:
            model = load_ppo_model(card.path)
        except Exception as exc:
            st.error(f"failed to load {pid}: {exc}")
            continue
        if model.action_space.n != env.action_space.n:
            st.error(
                f"{pid}: action space mismatch "
                f"(model={model.action_space.n}, env={env.action_space.n})"
            )
            continue
        probs = action_distribution(model, obs)
        st.markdown(f"**{pid}**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(labels, probs, color="#1f77b4")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("prob")
        ax.grid(True, alpha=0.3, axis="x")
        for j, p in enumerate(probs):
            ax.text(p + 0.01, j, f"{p:.2f}", va="center", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
