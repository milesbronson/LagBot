"""Tests for OpponentTracker prior-seeding, obs blending, and snapshot
write-back — the persistent-opponent-profile pipeline."""

import pytest

from src.poker_env.opponent_tracker import (
    OpponentTracker,
    PRIOR_BLEND_HANDS,
    UNKNOWN_OPPONENT_DEFAULTS,
)


class TestSeedPriors:
    def test_seed_replaces_priors_dict(self):
        t = OpponentTracker()
        t.seed_priors({1: {"hands_observed": 50, "vpip": 0.4}})
        assert 1 in t.priors
        assert t.priors[1]["vpip"] == 0.4

        t.seed_priors({2: {"hands_observed": 30, "vpip": 0.6}})
        assert 1 not in t.priors
        assert t.priors[2]["vpip"] == 0.6

    def test_seed_is_a_copy_not_reference(self):
        t = OpponentTracker()
        src = {1: {"hands_observed": 10, "vpip": 0.3}}
        t.seed_priors(src)
        src[1]["vpip"] = 0.9
        assert t.priors[1]["vpip"] == 0.3


class TestObsBlending:
    def _features(self, tracker):
        feats = tracker.get_observation_features(
            hero_id=0, opponent_ids=[1], max_opponents=9, features_per_opponent=12,
        )
        return feats[:12]

    def test_no_prior_no_live_returns_unknown_defaults(self):
        t = OpponentTracker()
        assert self._features(t) == UNKNOWN_OPPONENT_DEFAULTS

    def test_prior_only_replaces_defaults(self):
        t = OpponentTracker()
        t.seed_priors({1: {
            "hands_observed": 80,
            "vpip": 0.7, "pfr": 0.5, "af": 1.5,
            "three_bet_percent": 0.2, "cbet_percent": 0.6,
            "fold_to_cbet_percent": 0.4, "went_to_showdown_percent": 0.3,
            "win_at_showdown_percent": 0.55, "wwsf_percent": 0.45,
            "fold_to_3bet_after_raise_percent": 0.7, "squeeze_percent": 0.08,
        }})
        slot = self._features(t)
        assert slot[0] == pytest.approx(0.7)  # vpip
        assert slot[1] == pytest.approx(0.5)  # pfr
        assert slot[2] == pytest.approx(0.5)  # af 1.5 / 3.0
        assert slot[11] == pytest.approx(0.8)  # confidence = 80/100

    def test_prior_fades_as_live_hands_accumulate(self):
        t = OpponentTracker()
        t.seed_priors({1: {"hands_observed": 100, "vpip": 1.0, "pfr": 1.0}})

        # Spawn live profile with hands_played = half the blend threshold.
        t.start_hand(1, [{"id": 1, "name": "p"}], dealer_position=0,
                     small_blind=1, big_blind=2)
        opp = t.opponents[1]
        opp.vpip = 0.0
        opp.pfr = 0.0
        opp.hands_played = int(PRIOR_BLEND_HANDS / 2)  # halfway through fade

        vpip = self._features(t)[0]
        # Halfway: live (0.0) * 0.5 + prior (1.0) * 0.5 = 0.5
        assert vpip == pytest.approx(0.5, abs=1e-3)

    def test_live_takes_over_after_blend_threshold(self):
        t = OpponentTracker()
        t.seed_priors({1: {"hands_observed": 1000, "vpip": 1.0}})

        t.start_hand(1, [{"id": 1, "name": "p"}], dealer_position=0,
                     small_blind=1, big_blind=2)
        opp = t.opponents[1]
        opp.vpip = 0.0
        opp.hands_played = int(PRIOR_BLEND_HANDS) + 10

        vpip = self._features(t)[0]
        assert vpip == pytest.approx(0.0, abs=1e-3)

    def test_confidence_combines_live_and_prior_hands(self):
        t = OpponentTracker()
        t.seed_priors({1: {"hands_observed": 60, "vpip": 0.3}})

        t.start_hand(1, [{"id": 1, "name": "p"}], dealer_position=0,
                     small_blind=1, big_blind=2)
        opp = t.opponents[1]
        opp.hands_played = 30
        opp.confidence = 0.3

        # (60 + 30) / 100 = 0.9
        assert self._features(t)[11] == pytest.approx(0.9, abs=1e-3)


class TestSnapshot:
    def test_snapshot_includes_hands_observed_and_rates(self):
        t = OpponentTracker()
        t.start_hand(1, [{"id": 1, "name": "p"}], dealer_position=0,
                     small_blind=1, big_blind=2)
        opp = t.opponents[1]
        opp.hands_played = 42
        opp.vpip = 0.35
        opp.pfr = 0.18
        opp.af = 2.1

        snap = t.snapshot_for_registry()
        assert 1 in snap
        assert snap[1]["hands_observed"] == 42
        assert snap[1]["vpip"] == pytest.approx(0.35)
        assert snap[1]["pfr"] == pytest.approx(0.18)
        assert snap[1]["af"] == pytest.approx(2.1)

    def test_snapshot_empty_when_no_opponents(self):
        t = OpponentTracker()
        assert t.snapshot_for_registry() == {}


class TestSeedSnapshotRoundTrip:
    def test_snapshot_keys_match_seed_keys(self):
        """A snapshot from one tracker should be valid input to seed_priors
        on another tracker — round trips through the registry's
        update_behavior_stats need this contract."""
        t1 = OpponentTracker()
        t1.start_hand(1, [{"id": 1, "name": "p"}], dealer_position=0,
                      small_blind=1, big_blind=2)
        opp = t1.opponents[1]
        opp.hands_played = 25
        opp.vpip = 0.4
        opp.pfr = 0.25

        snap = t1.snapshot_for_registry()
        t2 = OpponentTracker()
        t2.seed_priors(snap)
        assert t2.priors[1]["vpip"] == pytest.approx(0.4)
        assert t2.priors[1]["hands_observed"] == 25
