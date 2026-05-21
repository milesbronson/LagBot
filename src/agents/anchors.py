"""
Hand-coded anchor agents — the backbone of the opponent pool.

Each anchor represents a distinct, classical poker archetype (tight/loose
× passive/aggressive plus a few extreme caricatures). Anchors are *not*
retrained per run; they are class instances seeded once into the registry
with canonical behaviour_stats. The stratified sampler bins every learned
checkpoint to its nearest anchor by behaviour-stat distance, so the
learner is always exposed to opponents spanning the full play-style
manifold instead of self-similar mirrors of itself.

Design contract
---------------
- Each anchor reads live game state via ``self._env`` (set by
  ``BaseAgent.bind_env``). Anchors that do not have an env bound fall
  back to safe defaults (check / call).
- All decisions key off ``env._calculate_hand_strength`` (cached per
  street) plus ``env.get_valid_actions`` (always honoured). An anchor
  never returns an illegal action.
- Each class exports:
    - ``ANCHOR_ID``: registry id, also used by the seeding script.
    - ``ARCHETYPE``: short string label for telemetry and stratum names.
    - ``CANONICAL_STATS``: the behaviour stats the seeding script writes
      to the AgentCard. Used by the stratified sampler as the cluster
      centroid for cards that quantise to this archetype.
"""

from typing import Optional, List
import random

import numpy as np

from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def _amount_to_call(env) -> int:
    p = env.game_state.get_current_player()
    return env.game_state.pot_manager.current_bet - p.current_bet


def _equity(env) -> float:
    """Hand strength for the current actor (Monte Carlo, cached per street)."""
    p = env.game_state.get_current_player()
    return env._calculate_hand_strength(p.hand, env.game_state.community_cards)


def _is_preflop(env) -> bool:
    return env.game_state.betting_round.value == 0


def _pick_raise_bin(env, target_frac: float, valid: List[int]) -> Optional[int]:
    """Closest legal raise-bin action to a desired pot-fraction.

    Returns the action index or ``None`` if no raise size is legal at
    this decision point (e.g. stack too short)."""
    bins = env.raise_bins
    best_action = None
    best_diff = float("inf")
    for i, frac in enumerate(bins):
        action = 2 + i
        if action in valid:
            diff = abs(frac - target_frac)
            if diff < best_diff:
                best_diff = diff
                best_action = action
    return best_action


def _all_in(env, valid: List[int]) -> Optional[int]:
    idx = 2 + len(env.raise_bins)
    return idx if idx in valid else None


def _fold_or_check(env, valid: List[int]) -> int:
    """Give-up decision: fold if facing a bet, check if free.

    Folding when checking is free is always strictly dominated, so the
    anchors never do it — that would inflate VPIP without changing the
    archetype's intent."""
    if _amount_to_call(env) > 0 and 0 in valid:
        return 0
    return 1


def _raise_then_call_then_check(
    env, target_frac: float, valid: List[int]
) -> int:
    """Try a raise at ``target_frac``; if no raise is legal, call/check."""
    a = _pick_raise_bin(env, target_frac, valid)
    if a is not None:
        return a
    return 1


# ---------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------

class AnchorAgent(BaseAgent):
    """Common scaffolding for hand-coded archetypes."""

    ANCHOR_ID: str = ""
    ARCHETYPE: str = ""
    CANONICAL_STATS: dict = {}

    def __init__(self, name: Optional[str] = None, rng: Optional[random.Random] = None):
        super().__init__(name or self.ARCHETYPE)
        self.rng = rng if rng is not None else random.Random(0)

    def select_action(
        self, observation: np.ndarray, valid_actions: Optional[list] = None
    ) -> int:
        env = self._env
        if env is None:
            # Unbound anchor — return check/call so we never crash a duel.
            return 1
        valid = list(valid_actions) if valid_actions is not None else env.get_valid_actions()
        if not valid:
            return 1
        return self._decide(env, valid)

    def _decide(self, env, valid: List[int]) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------

class TightPassive(AnchorAgent):
    """Rock — only puts money in with strong hands, never raises.

    Folds to any bet without a premium holding; calls down with strong
    made hands; checks freely otherwise."""

    ANCHOR_ID = "anchor_tight_passive_v0"
    ARCHETYPE = "tight_passive"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.15,
        "pfr": 0.04,
        "af": 0.33,
        "three_bet_percent": 0.09,
        "cbet_percent": 0.00,
        "fold_to_cbet_percent": 1.00,
        "went_to_showdown_percent": 0.04,
        "win_at_showdown_percent": 0.33,
        "wwsf_percent": 0.08,
        "fold_to_3bet_after_raise_percent": 0.80,
        "squeeze_percent": 0.01,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        threshold = 0.70 if _is_preflop(env) else 0.62
        if eq >= threshold:
            return 1
        return _fold_or_check(env, valid)


class TightAggressive(AnchorAgent):
    """TAG — raises a tight range, plays straightforward postflop."""

    ANCHOR_ID = "anchor_tight_aggressive_v0"
    ARCHETYPE = "tight_aggressive"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.67,
        "pfr": 0.51,
        "af": 3.33,
        "three_bet_percent": 0.51,
        "cbet_percent": 0.65,
        "fold_to_cbet_percent": 1.00,
        "went_to_showdown_percent": 0.26,
        "win_at_showdown_percent": 0.48,
        "wwsf_percent": 0.25,
        "fold_to_3bet_after_raise_percent": 0.55,
        "squeeze_percent": 0.05,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        if _is_preflop(env):
            if eq >= 0.62:
                return _raise_then_call_then_check(env, 1.0, valid)
            if eq >= 0.55:
                return 1
            return _fold_or_check(env, valid)
        if eq >= 0.65:
            return _raise_then_call_then_check(env, 0.75, valid)
        if eq >= 0.55:
            return 1
        return _fold_or_check(env, valid)


class LooseAggressive(AnchorAgent):
    """LAG — wide opening range, frequent aggression, bluffs."""

    ANCHOR_ID = "anchor_loose_aggressive_v0"
    ARCHETYPE = "loose_aggressive"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.96,
        "pfr": 0.89,
        "af": 5.71,
        "three_bet_percent": 0.84,
        "cbet_percent": 0.74,
        "fold_to_cbet_percent": 0.00,
        "went_to_showdown_percent": 0.59,
        "win_at_showdown_percent": 0.48,
        "wwsf_percent": 0.60,
        "fold_to_3bet_after_raise_percent": 0.40,
        "squeeze_percent": 0.09,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        # Add a small random bluff impulse so postflop aggression
        # frequency lands in LAG territory even on bricked boards.
        bluff = self.rng.random() < 0.25
        if _is_preflop(env):
            if eq >= 0.48 or bluff:
                return _raise_then_call_then_check(env, 1.0, valid)
            if eq >= 0.40:
                return 1
            return _fold_or_check(env, valid)
        if eq >= 0.50 or bluff:
            return _raise_then_call_then_check(env, 0.75, valid)
        if eq >= 0.40:
            return 1
        return _fold_or_check(env, valid)


class LoosePassive(AnchorAgent):
    """Calling station — pays off with marginal hands, rarely raises."""

    ANCHOR_ID = "anchor_loose_passive_v0"
    ARCHETYPE = "loose_passive"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.90,
        "pfr": 0.07,
        "af": 0.00,
        "three_bet_percent": 0.09,
        "cbet_percent": 0.05,
        "fold_to_cbet_percent": 0.00,
        "went_to_showdown_percent": 0.65,
        "win_at_showdown_percent": 0.53,
        "wwsf_percent": 0.63,
        "fold_to_3bet_after_raise_percent": 0.30,
        "squeeze_percent": 0.01,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        if eq >= 0.80:
            # Only raises with monsters — and small.
            return _raise_then_call_then_check(env, 0.5, valid)
        if eq >= 0.35:
            return 1
        return _fold_or_check(env, valid)


class Shover(AnchorAgent):
    """Maniac — frequent all-ins, especially preflop. Caricature class."""

    ANCHOR_ID = "anchor_shover_v0"
    ARCHETYPE = "shover"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.99,
        "pfr": 0.95,
        "af": 49.00,
        "three_bet_percent": 0.90,
        "cbet_percent": 0.30,
        "fold_to_cbet_percent": 0.00,
        "went_to_showdown_percent": 0.65,
        "win_at_showdown_percent": 0.47,
        "wwsf_percent": 0.80,
        "fold_to_3bet_after_raise_percent": 0.20,
        "squeeze_percent": 0.18,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        all_in = _all_in(env, valid)

        if _is_preflop(env):
            # Shove with any equity > 0.45 or ~50% bluff frequency.
            if all_in is not None and (eq >= 0.45 or self.rng.random() < 0.5):
                return all_in
            return 1

        if all_in is not None and eq >= 0.50:
            return all_in
        if eq >= 0.40:
            return 1
        return _fold_or_check(env, valid)


class MinRaiser(AnchorAgent):
    """Always opens for the smallest legal raise. Slightly loose preflop."""

    ANCHOR_ID = "anchor_min_raiser_v0"
    ARCHETYPE = "min_raiser"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.90,
        "pfr": 0.81,
        "af": 7.00,
        "three_bet_percent": 0.70,
        "cbet_percent": 0.55,
        "fold_to_cbet_percent": 0.00,
        "went_to_showdown_percent": 0.54,
        "win_at_showdown_percent": 0.54,
        "wwsf_percent": 0.63,
        "fold_to_3bet_after_raise_percent": 0.50,
        "squeeze_percent": 0.06,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        smallest_bin_frac = env.raise_bins[0]
        if _is_preflop(env):
            if eq >= 0.50:
                return _raise_then_call_then_check(env, smallest_bin_frac, valid)
            if eq >= 0.42:
                return 1
            return _fold_or_check(env, valid)
        if eq >= 0.55:
            return _raise_then_call_then_check(env, smallest_bin_frac, valid)
        if eq >= 0.45:
            return 1
        return _fold_or_check(env, valid)


class OverBettor(AnchorAgent):
    """Polarised over-bettor — when it bets, it goes large."""

    ANCHOR_ID = "anchor_overbettor_v0"
    ARCHETYPE = "overbettor"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.62,
        "pfr": 0.62,
        "af": 1.00,
        "three_bet_percent": 0.63,
        "cbet_percent": 0.50,
        "fold_to_cbet_percent": 1.00,
        "went_to_showdown_percent": 0.28,
        "win_at_showdown_percent": 0.51,
        "wwsf_percent": 0.29,
        "fold_to_3bet_after_raise_percent": 0.45,
        "squeeze_percent": 0.08,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        largest_bin_frac = env.raise_bins[-1]
        if _is_preflop(env):
            if eq >= 0.55:
                return _raise_then_call_then_check(env, largest_bin_frac, valid)
            return _fold_or_check(env, valid)
        if eq >= 0.60:
            all_in = _all_in(env, valid)
            if all_in is not None and eq >= 0.75:
                return all_in
            return _raise_then_call_then_check(env, largest_bin_frac, valid)
        return _fold_or_check(env, valid)


class Counterpuncher(AnchorAgent):
    """Bluff-catching TAG — tight preflop, calls down postflop with any
    showdown equity instead of folding to every c-bet. Designed to
    punish hyper-aggressive policies that win by perpetual aggression
    against folders."""

    ANCHOR_ID = "anchor_counterpuncher_v0"
    ARCHETYPE = "counterpuncher"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.60,
        "pfr": 0.18,
        "af": 1.20,
        "three_bet_percent": 0.10,
        "cbet_percent": 0.40,
        "fold_to_cbet_percent": 0.15,
        "went_to_showdown_percent": 0.55,
        "win_at_showdown_percent": 0.56,
        "wwsf_percent": 0.48,
        "fold_to_3bet_after_raise_percent": 0.30,
        "squeeze_percent": 0.04,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        if _is_preflop(env):
            if eq >= 0.68:
                return _raise_then_call_then_check(env, 0.75, valid)
            # Wide preflop call range — must defend blinds vs raise-everything
            # bots, otherwise they win by attrition without ever facing us
            # postflop.
            if eq >= 0.30:
                return 1
            return _fold_or_check(env, valid)
        # Postflop: rarely raises. Calls down with any showdown equity to
        # punish multi-street bluff barrels.
        if eq >= 0.75:
            return _raise_then_call_then_check(env, 0.6, valid)
        if eq >= 0.28:
            return 1
        return _fold_or_check(env, valid)


class Trapper(AnchorAgent):
    """Slowplay specialist — never raises postflop, just check-calls with
    any equity. Punishes hyper-aggressive barrelers who fire multiple
    streets at uncontested pots."""

    ANCHOR_ID = "anchor_trapper_v0"
    ARCHETYPE = "trapper"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.70,
        "pfr": 0.00,
        "af": 0.20,
        "three_bet_percent": 0.00,
        "cbet_percent": 0.00,
        "fold_to_cbet_percent": 0.05,
        "went_to_showdown_percent": 0.65,
        "win_at_showdown_percent": 0.58,
        "wwsf_percent": 0.45,
        "fold_to_3bet_after_raise_percent": 0.10,
        "squeeze_percent": 0.00,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        if _is_preflop(env):
            # Wide preflop call range. Never open-raises (pure slowplay).
            if eq >= 0.35:
                return 1
            return _fold_or_check(env, valid)
        # Postflop: never raises. Calls down with very wide equity to
        # snap off the bot's multi-street bluffs.
        if eq >= 0.25:
            return 1
        return _fold_or_check(env, valid)


class ThreeBetBully(AnchorAgent):
    """Light 3-bettor — re-raises wide preflop to punish wide opens.
    Forces the original raiser to either fold or play a bloated pot
    out of position with marginal hands."""

    ANCHOR_ID = "anchor_three_bet_bully_v0"
    ARCHETYPE = "three_bet_bully"
    CANONICAL_STATS = {
        "hands_observed": 1000,
        "vpip": 0.40,
        "pfr": 0.32,
        "af": 4.00,
        "three_bet_percent": 0.45,
        "cbet_percent": 0.70,
        "fold_to_cbet_percent": 0.40,
        "went_to_showdown_percent": 0.22,
        "win_at_showdown_percent": 0.46,
        "wwsf_percent": 0.55,
        "fold_to_3bet_after_raise_percent": 0.55,
        "squeeze_percent": 0.20,
    }

    def _decide(self, env, valid: List[int]) -> int:
        eq = _equity(env)
        if _is_preflop(env):
            facing_raise = _amount_to_call(env) > env.big_blind
            if facing_raise:
                # 3-bet wide and BIG to make raise-everything bots pay.
                # Pot-sized 3-bet against opens — they have to fold or
                # play a bloated pot OOP with marginal hands.
                if eq >= 0.42:
                    return _raise_then_call_then_check(env, 2.5, valid)
                # Call the rest with showdown value rather than folding
                # — denies the bot easy steal frequency.
                if eq >= 0.30:
                    return 1
                return _fold_or_check(env, valid)
            # No raise to face — open semi-wide too.
            if eq >= 0.50:
                return _raise_then_call_then_check(env, 1.0, valid)
            return _fold_or_check(env, valid)
        # Postflop: call down wider against aggression; only raise with
        # the goods so we don't get re-raised off equity.
        if eq >= 0.65:
            return _raise_then_call_then_check(env, 0.75, valid)
        if eq >= 0.32:
            return 1
        return _fold_or_check(env, valid)


# ---------------------------------------------------------------------
# Registry of all anchors (consumed by seeding script + stratifier).
# ---------------------------------------------------------------------

ALL_ANCHORS = (
    TightPassive,
    TightAggressive,
    LooseAggressive,
    LoosePassive,
    Shover,
    MinRaiser,
    OverBettor,
    Counterpuncher,
    Trapper,
    ThreeBetBully,
)


def build_all_anchors(rng: Optional[random.Random] = None) -> List[AnchorAgent]:
    """Instantiate one of each archetype. Shared rng keeps deterministic
    test schedules across the whole pool."""
    return [cls(rng=rng) for cls in ALL_ANCHORS]
