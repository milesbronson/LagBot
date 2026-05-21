"""
Tests for the hand-coded anchor agents.

We do not try to assert the *exact* canonical stats — those are noisy
over hundreds of hands. Instead we assert each archetype lands in its
expected behaviour band, and that the archetypes are mutually
distinguishable on at least one axis. That mirrors how the stratified
sampler will quantise live cards.
"""

import random

import pytest

from src.agents.anchors import (
    ALL_ANCHORS,
    AnchorAgent,
    LooseAggressive,
    LoosePassive,
    MinRaiser,
    OverBettor,
    Shover,
    TightAggressive,
    TightPassive,
    build_all_anchors,
)
from src.agents.random_agent import RandomAgent
from src.poker_env.texas_holdem_env import TexasHoldemEnv


def _play(anchor: AnchorAgent, hands: int = 300, seed: int = 11):
    """Play ``hands`` heads-up between the anchor (seat 0) and a
    RandomAgent. Returns the tracker profile observed on the anchor."""
    env = TexasHoldemEnv(
        num_players=2,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        track_opponents=True,
        learning_agent_id=0,
        raise_bins=[0.5, 1.0, 2.0],
    )
    opponent = RandomAgent()
    anchor.seat(env.game_state.players[0])
    anchor.bind_env(env)
    opponent.seat(env.game_state.players[1])
    opponent.bind_env(env)

    obs, _ = env.reset(seed=seed)
    played = 0
    while played < hands:
        cur = env.game_state.get_current_player()
        agent = anchor if cur.player_id == 0 else opponent
        valid = env.get_valid_actions()
        action = agent.select_action(obs, valid)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            played += 1
            if played < hands:
                obs, _ = env.reset(seed=seed + played)
    return env.opponent_tracker.opponents.get(0)


class TestAnchorContract:
    def test_all_anchors_have_ids_and_archetypes(self):
        seen_ids = set()
        seen_archetypes = set()
        for cls in ALL_ANCHORS:
            assert cls.ANCHOR_ID, f"{cls.__name__} missing ANCHOR_ID"
            assert cls.ARCHETYPE, f"{cls.__name__} missing ARCHETYPE"
            assert cls.ANCHOR_ID not in seen_ids, f"duplicate ANCHOR_ID {cls.ANCHOR_ID}"
            assert cls.ARCHETYPE not in seen_archetypes, f"duplicate ARCHETYPE {cls.ARCHETYPE}"
            seen_ids.add(cls.ANCHOR_ID)
            seen_archetypes.add(cls.ARCHETYPE)

    def test_canonical_stats_are_fractions(self):
        for cls in ALL_ANCHORS:
            s = cls.CANONICAL_STATS
            for key in (
                "vpip", "pfr", "three_bet_percent", "cbet_percent",
                "fold_to_cbet_percent", "went_to_showdown_percent",
                "win_at_showdown_percent", "wwsf_percent",
                "fold_to_3bet_after_raise_percent", "squeeze_percent",
            ):
                assert 0.0 <= s[key] <= 1.0, (
                    f"{cls.__name__}.CANONICAL_STATS[{key!r}]={s[key]} "
                    "must be a fraction in [0, 1] (tracker emits fractions)"
                )
            assert s["af"] >= 0.0

    def test_build_all_returns_one_of_each(self):
        anchors = build_all_anchors()
        assert len(anchors) == len(ALL_ANCHORS)
        assert {type(a) for a in anchors} == set(ALL_ANCHORS)

    def test_unbound_anchor_returns_check_call(self):
        """An anchor with no env bound must never crash a duel — return
        the universally-legal call/check action."""
        a = TightPassive()
        action = a.select_action(observation=None, valid_actions=[0, 1])
        assert action == 1

    def test_anchor_respects_valid_actions(self):
        """Anchors must never return an illegal action."""
        rng = random.Random(0)
        for cls in ALL_ANCHORS:
            anchor = cls(rng=rng)
            env = TexasHoldemEnv(
                num_players=2, starting_stack=1000,
                small_blind=5, big_blind=10,
                track_opponents=True, learning_agent_id=0,
                raise_bins=[0.5, 1.0, 2.0],
            )
            anchor.seat(env.game_state.players[0])
            anchor.bind_env(env)
            opponent = RandomAgent()
            opponent.seat(env.game_state.players[1])
            opponent.bind_env(env)

            obs, _ = env.reset(seed=7)
            for _ in range(2000):
                cur = env.game_state.get_current_player()
                agent = anchor if cur.player_id == 0 else opponent
                valid = env.get_valid_actions()
                action = agent.select_action(obs, valid)
                assert action in valid, (
                    f"{cls.__name__} returned illegal action {action} "
                    f"(valid: {valid})"
                )
                obs, _, terminated, _, _ = env.step(action)
                if terminated:
                    obs, _ = env.reset()


class TestArchetypeBands:
    """Each archetype must land in its empirical behaviour band over
    hundreds of hands. Bands are generous so flaky-seed jitter doesn't
    break CI, but tight enough that swapping archetype logic would fail."""

    def test_tight_passive_is_tight_and_passive(self):
        profile = _play(TightPassive(rng=random.Random(0)))
        assert profile is not None
        assert profile.vpip < 0.30, f"TP vpip too high: {profile.vpip}"
        assert profile.pfr < 0.15, f"TP pfr too high: {profile.pfr}"
        assert profile.af < 1.0, f"TP af too high: {profile.af}"

    def test_tight_aggressive_raises_a_bunch(self):
        profile = _play(TightAggressive(rng=random.Random(0)))
        assert profile is not None
        assert profile.pfr > 0.30, f"TAG pfr too low: {profile.pfr}"
        assert profile.af > 1.5, f"TAG af too low: {profile.af}"

    def test_loose_aggressive_plays_wide_and_attacks(self):
        profile = _play(LooseAggressive(rng=random.Random(0)))
        assert profile is not None
        assert profile.vpip > 0.85, f"LAG vpip too low: {profile.vpip}"
        assert profile.pfr > 0.70, f"LAG pfr too low: {profile.pfr}"
        assert profile.af > 2.0, f"LAG af too low: {profile.af}"

    def test_loose_passive_is_a_calling_station(self):
        profile = _play(LoosePassive(rng=random.Random(0)))
        assert profile is not None
        assert profile.vpip > 0.70, f"LP vpip too low: {profile.vpip}"
        assert profile.pfr < 0.20, f"LP pfr too high: {profile.pfr}"
        assert profile.af < 0.5, f"LP af too high: {profile.af}"

    def test_shover_shoves(self):
        profile = _play(Shover(rng=random.Random(0)))
        assert profile is not None
        assert profile.vpip > 0.90, f"Shover vpip too low: {profile.vpip}"
        assert profile.pfr > 0.80, f"Shover pfr too low: {profile.pfr}"
        # AF blows up because nearly every action is a raise; sanity ≥ 5.
        assert profile.af > 5.0, f"Shover af too low: {profile.af}"

    def test_min_raiser_raises_a_lot(self):
        profile = _play(MinRaiser(rng=random.Random(0)))
        assert profile is not None
        assert profile.pfr > 0.50, f"MR pfr too low: {profile.pfr}"
        assert profile.af > 1.0, f"MR af too low: {profile.af}"

    def test_overbettor_is_polarised(self):
        profile = _play(OverBettor(rng=random.Random(0)))
        assert profile is not None
        # Polarised: when it plays, it raises (PFR ≈ VPIP).
        assert profile.pfr > 0.45, f"OB pfr too low: {profile.pfr}"
        # Tight-ish — fewer hands than the loose archetypes.
        assert profile.vpip < 0.80, f"OB vpip too high: {profile.vpip}"


class TestArchetypeDistinguishability:
    """The stratifier needs archetypes spread out in stat-space.
    Verify each pair differs on at least one of (vpip, pfr, af)."""

    def test_archetypes_are_mutually_distinguishable(self):
        anchors = [cls.CANONICAL_STATS for cls in ALL_ANCHORS]
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                a, b = anchors[i], anchors[j]
                diffs = (
                    abs(a["vpip"] - b["vpip"]),
                    abs(a["pfr"] - b["pfr"]),
                    # Cap AF distance — Shover's AF=49 would otherwise
                    # always dominate. Same idea the stratifier will use.
                    min(abs(a["af"] - b["af"]) / 10.0, 1.0),
                )
                assert max(diffs) > 0.10, (
                    f"{ALL_ANCHORS[i].__name__} and {ALL_ANCHORS[j].__name__} "
                    f"too close in (vpip, pfr, af): {a}, {b}"
                )
