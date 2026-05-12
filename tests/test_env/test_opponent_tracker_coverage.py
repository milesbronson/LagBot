"""
Happy-path coverage tests for OpponentTracker.

Each tracked metric gets at least one focused positive-path test that pins
its expected counts and computed percentages on a hand-crafted scenario.
These complement the adversarial tests in test_opponent_tracker_bugs.py:
the bug tests prove specific defects don't recur, these tests prove the
overall happy-path math still works (regression guard).

Metrics covered:
    Per-action / per-hand:
        - VPIP                              (C-1)
        - PFR                               (C-2)
        - AF                                (C-3)
        - Confidence                        (C-4)
        - hands_played accumulation         (C-5)
    Preflop sequence:
        - 3-bet opportunities + count       (C-6)
        - 3-bet frequency                   (C-7)
        - Squeeze opportunities + attempts  (C-8)
        - Fold-to-3-bet-after-raise         (C-9)
    Flop:
        - C-bet opportunities + made        (C-10)
        - Fold-to-cbet                      (C-11)
    Showdown:
        - WTSD count + %                    (C-12)
        - W$SD                              (C-13)
        - WWSF                              (C-14)
    Observation features:
        - 12 features per slot, order       (C-15)
        - Neutral defaults for unknown opp  (C-16)
        - Zero-pad for empty slots          (C-17)
"""

import pytest

from src.poker_env.opponent_tracker import OpponentTracker, Action, Street


# --------------------------------------------------------------------- helpers


def _players(n):
    return [{'id': i, 'name': chr(ord('A') + i), 'stack': 1000} for i in range(n)]


def _act(tracker, *, pid, action, street=Street.PREFLOP, amount=0,
         pot=10, stack_before=1000, stack_after=1000, position=0):
    tracker.record_action(
        player_id=pid, player_name=chr(ord('A') + pid),
        action=action, amount=amount,
        pot_size=pot, stack_before=stack_before, stack_after=stack_after,
        street=street, position=position,
    )


def _end(tracker, winners, winnings=None, stacks=None):
    winnings = winnings or {pid: 0 for pid in winners}
    stacks = stacks or {}
    tracker.end_hand(winners=winners, winnings=winnings, final_stacks=stacks)


# --------------------------------------------------------------------- C-1..C-5: basic per-hand counters


class TestBasicCounters:
    def test_c1_vpip_credits_player_who_voluntarily_invests(self):
        """VPIP should fire for CALL and for RAISE, not for FOLD or CHECK."""
        tracker = OpponentTracker()
        # 3 hands, A folds preflop in 1, calls in 1, raises in 1
        for h, action in enumerate([Action.FOLD, Action.CALL, Action.RAISE]):
            tracker.start_hand(h + 1, _players(2), dealer_position=0,
                               small_blind=1, big_blind=2)
            _act(tracker, pid=0, action=action, amount=10)
            _act(tracker, pid=1, action=Action.FOLD)
            _end(tracker, winners=[0 if action != Action.FOLD else 1])

        # A voluntarily invested in 2 of 3 hands
        assert tracker.opponents[0].vpip == pytest.approx(2 / 3, abs=1e-3)
        # B always folded
        assert tracker.opponents[1].vpip == 0.0

    def test_c2_pfr_credits_only_raises(self):
        """PFR fires for RAISE/ALL_IN, not for CALL."""
        tracker = OpponentTracker()
        # 4 hands: A raises in 2, calls in 1, folds in 1
        scenarios = [Action.RAISE, Action.RAISE, Action.CALL, Action.FOLD]
        for h, action in enumerate(scenarios):
            tracker.start_hand(h + 1, _players(2), dealer_position=0,
                               small_blind=1, big_blind=2)
            _act(tracker, pid=0, action=action, amount=10)
            _act(tracker, pid=1, action=Action.FOLD)
            _end(tracker, winners=[0])

        # PFR = 2/4 = 0.5; VPIP = 3/4 = 0.75
        assert tracker.opponents[0].pfr == pytest.approx(0.5, abs=1e-3)
        assert tracker.opponents[0].vpip == pytest.approx(0.75, abs=1e-3)

    def test_c3_aggression_factor_is_bets_raises_over_calls(self):
        """AF = (bets + raises + all_ins) / calls, capped at 1.0 when no calls."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        # A: 2 raises preflop, 1 call on flop -> AF = 2/1 = 2.0
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.RAISE, amount=30)
        _act(tracker, pid=0, action=Action.RAISE, amount=90, pot=130)
        _act(tracker, pid=1, action=Action.CALL, amount=60, pot=190)
        _act(tracker, pid=0, action=Action.CALL, amount=20, pot=210, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.CHECK, street=Street.FLOP)
        _end(tracker, winners=[0])

        assert tracker.opponents[0].af == pytest.approx(2.0, abs=1e-3)

    def test_c4_confidence_caps_at_one(self):
        """Confidence = min(hands_played / 100, 1.0)."""
        tracker = OpponentTracker()
        for h in range(150):
            tracker.start_hand(h + 1, _players(2), dealer_position=0,
                               small_blind=1, big_blind=2)
            _act(tracker, pid=0, action=Action.FOLD)
            _end(tracker, winners=[1])

        assert tracker.opponents[0].hands_played == 150
        assert tracker.opponents[0].confidence == 1.0

    def test_c5_hands_played_increments_only_when_player_acted(self):
        """A player who took no actions in a hand doesn't get hands_played++."""
        tracker = OpponentTracker()
        # 3 players seated, but player 2 takes no action this hand
        tracker.start_hand(1, _players(3), dealer_position=0,
                           small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.FOLD)
        # No action from pid=2 (e.g. sat out)
        _end(tracker, winners=[0])

        assert tracker.opponents[0].hands_played == 1
        assert tracker.opponents[1].hands_played == 1
        assert tracker.opponents[2].hands_played == 0


# --------------------------------------------------------------------- C-6..C-9: preflop sequence


class TestThreeBetCoverage:
    def test_c6_three_bet_opportunity_only_when_facing_a_raise(self):
        """The opener (first raiser) never has a 3-bet opportunity."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(3), dealer_position=0,
                           small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=2, action=Action.FOLD)
        _end(tracker, winners=[0])

        assert tracker.opponents[0].three_bet_opportunities == 0
        # B and C both faced A's raise without having raised first.
        assert tracker.opponents[1].three_bet_opportunities == 1
        assert tracker.opponents[2].three_bet_opportunities == 1
        # None of them actually 3-bet.
        assert tracker.opponents[0].three_bet_count == 0
        assert tracker.opponents[1].three_bet_count == 0
        assert tracker.opponents[2].three_bet_count == 0

    def test_c7_three_bet_frequency_over_multiple_hands(self):
        """3-bet% = three_bet_count / three_bet_opportunities."""
        tracker = OpponentTracker()
        # Hand 1: A raises, B 3-bets (B: 1 opp, 1 count)
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.RAISE, amount=30)
        _act(tracker, pid=0, action=Action.FOLD)
        _end(tracker, winners=[1])

        # Hand 2: A raises, B folds (B: 1 opp, 0 count)
        tracker.start_hand(2, _players(2), dealer_position=1, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.FOLD)
        _end(tracker, winners=[0])

        # Hand 3: A raises, B folds (B: 1 opp, 0 count)
        tracker.start_hand(3, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.FOLD)
        _end(tracker, winners=[0])

        b = tracker.opponents[1]
        assert b.three_bet_opportunities == 3
        assert b.three_bet_count == 1
        assert b.three_bet_frequency == pytest.approx(1 / 3, abs=1e-3)
        assert b.three_bet_percent == pytest.approx(1 / 3, abs=1e-3)


class TestSqueezeCoverage:
    def test_c8_squeeze_requires_raise_plus_caller(self):
        """Squeeze: raise → call(s) → re-raise. Distinct from a plain 3-bet."""
        tracker = OpponentTracker()
        # 4 players: A raises, B calls, C re-raises (squeeze), D folds, A & B fold.
        tracker.start_hand(1, _players(4), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=2, action=Action.RAISE, amount=40)
        _act(tracker, pid=3, action=Action.FOLD)
        _act(tracker, pid=0, action=Action.FOLD)
        _act(tracker, pid=1, action=Action.FOLD)
        _end(tracker, winners=[2])

        c = tracker.opponents[2]
        assert c.squeeze_opportunities == 1
        assert c.squeeze_attempts == 1
        assert c.squeeze_percent == pytest.approx(1.0, abs=1e-3)
        # B faced a raise but no caller in front -> no squeeze opp
        assert tracker.opponents[1].squeeze_opportunities == 0
        # A never faced a raise -> no squeeze opp
        assert tracker.opponents[0].squeeze_opportunities == 0


class TestFoldTo3BetAfterRaiseCoverage:
    def test_c9_fold_to_3bet_after_raise_basic(self):
        """A raises, B 3-bets, A folds -> A is credited fold-to-3bet-after-raise."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.RAISE, amount=30)
        _act(tracker, pid=0, action=Action.FOLD)
        _end(tracker, winners=[1])

        a = tracker.opponents[0]
        assert a.raised_preflop == 1
        assert a.faced_3bet_after_raise == 1
        assert a.folded_to_3bet_after_raise == 1
        assert a.fold_to_3bet_after_raise_percent == pytest.approx(1.0, abs=1e-3)

    def test_c9b_call_response_to_3bet_does_not_count_as_fold(self):
        """A raises, B 3-bets, A calls -> faced 3-bet but didn't fold to it."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.RAISE, amount=30)
        _act(tracker, pid=0, action=Action.CALL, amount=20)
        _act(tracker, pid=0, action=Action.CHECK, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.CHECK, street=Street.FLOP)
        _end(tracker, winners=[0])

        a = tracker.opponents[0]
        assert a.faced_3bet_after_raise == 1
        assert a.folded_to_3bet_after_raise == 0
        assert a.fold_to_3bet_after_raise_percent == 0.0


# --------------------------------------------------------------------- C-10..C-11: flop c-bet


class TestCbetCoverage:
    def test_c10_cbet_made_when_aggressor_opens_flop(self):
        """A raises pf, B calls, A opens flop with bet -> A is credited c-bet."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=0, action=Action.RAISE, amount=15, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.FOLD, street=Street.FLOP)
        _end(tracker, winners=[0])

        a = tracker.opponents[0]
        assert a.flop_cbet_opportunities == 1
        assert a.flop_cbet_made == 1
        assert a.flop_cbet_percent == pytest.approx(1.0, abs=1e-3)
        assert a.cbet_percent == pytest.approx(1.0, abs=1e-3)
        # B was not the aggressor; no opportunity, no made
        assert tracker.opponents[1].flop_cbet_opportunities == 0
        assert tracker.opponents[1].flop_cbet_made == 0

    def test_c10b_cbet_opportunity_only_no_make_on_check(self):
        """A raises pf, B calls, A checks flop -> opportunity but no c-bet made."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=0, action=Action.CHECK, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.CHECK, street=Street.FLOP)
        _act(tracker, pid=0, action=Action.CHECK, street=Street.TURN)
        _act(tracker, pid=1, action=Action.CHECK, street=Street.TURN)
        _act(tracker, pid=0, action=Action.CHECK, street=Street.RIVER)
        _act(tracker, pid=1, action=Action.CHECK, street=Street.RIVER)
        _end(tracker, winners=[0])

        a = tracker.opponents[0]
        assert a.flop_cbet_opportunities == 1
        assert a.flop_cbet_made == 0
        assert a.flop_cbet_percent == 0.0

    def test_c11_fold_to_cbet(self):
        """A raises pf, B calls, A bets flop, B folds -> B is credited."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=0, action=Action.RAISE, amount=15, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.FOLD, street=Street.FLOP)
        _end(tracker, winners=[0])

        b = tracker.opponents[1]
        assert b.faced_flop_cbet == 1
        assert b.folded_to_flop_cbet == 1
        assert b.fold_to_flop_cbet_percent == pytest.approx(1.0, abs=1e-3)
        assert b.fold_to_cbet_percent == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------- C-12..C-14: showdown


class TestShowdownCoverage:
    def test_c12_wtsd_counts_all_players_who_reach_showdown(self):
        """Both winner and loser reach showdown; both should be credited."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.CALL, amount=2)
        _act(tracker, pid=1, action=Action.CHECK)
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            _act(tracker, pid=1, action=Action.CHECK, street=street)
            _act(tracker, pid=0, action=Action.CHECK, street=street)
        _end(tracker, winners=[0], winnings={0: 4, 1: -2}, stacks={0: 1002, 1: 998})

        assert tracker.opponents[0].went_to_showdown == 1
        assert tracker.opponents[1].went_to_showdown == 1
        assert tracker.opponents[0].wtsd_percent == pytest.approx(1.0, abs=1e-3)
        assert tracker.opponents[1].wtsd_percent == pytest.approx(1.0, abs=1e-3)

    def test_c13_wsd_winner_only(self):
        """W$SD = showdown_wins / went_to_showdown. Loser stays at 0."""
        tracker = OpponentTracker()
        # Hand 1: A wins at showdown
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.CALL, amount=2)
        _act(tracker, pid=1, action=Action.CHECK)
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            _act(tracker, pid=1, action=Action.CHECK, street=street)
            _act(tracker, pid=0, action=Action.CHECK, street=street)
        _end(tracker, winners=[0], winnings={0: 4, 1: -2}, stacks={0: 1002, 1: 998})

        # Hand 2: B wins at showdown
        tracker.start_hand(2, _players(2), dealer_position=1, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.CALL, amount=2)
        _act(tracker, pid=1, action=Action.CHECK)
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            _act(tracker, pid=0, action=Action.CHECK, street=street)
            _act(tracker, pid=1, action=Action.CHECK, street=street)
        _end(tracker, winners=[1], winnings={0: -2, 1: 4}, stacks={0: 1000, 1: 1000})

        a, b = tracker.opponents[0], tracker.opponents[1]
        assert a.went_to_showdown == 2
        assert b.went_to_showdown == 2
        assert a.showdown_wins == 1
        assert b.showdown_wins == 1
        assert a.win_at_showdown_percent == pytest.approx(0.5, abs=1e-3)
        assert b.win_at_showdown_percent == pytest.approx(0.5, abs=1e-3)

    def test_c14_wwsf_won_when_saw_flop(self):
        """WWSF = won_when_saw_flop / saw_flop_count, including non-showdown wins."""
        tracker = OpponentTracker()
        # Hand 1: A and B see flop; A wins (no showdown because B folds flop).
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.RAISE, amount=10)
        _act(tracker, pid=1, action=Action.CALL, amount=10)
        _act(tracker, pid=0, action=Action.RAISE, amount=15, street=Street.FLOP)
        _act(tracker, pid=1, action=Action.FOLD, street=Street.FLOP)
        _end(tracker, winners=[0], winnings={0: 25, 1: -10}, stacks={0: 1025, 1: 990})

        # Hand 2: A and B see flop; B wins at showdown.
        tracker.start_hand(2, _players(2), dealer_position=1, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.CALL, amount=2)
        _act(tracker, pid=1, action=Action.CHECK)
        for street in [Street.FLOP, Street.TURN, Street.RIVER]:
            _act(tracker, pid=0, action=Action.CHECK, street=street)
            _act(tracker, pid=1, action=Action.CHECK, street=street)
        _end(tracker, winners=[1], winnings={0: -2, 1: 4}, stacks={0: 1023, 1: 994})

        a, b = tracker.opponents[0], tracker.opponents[1]
        assert a.saw_flop_count == 2
        assert b.saw_flop_count == 2
        assert a.won_when_saw_flop == 1
        assert b.won_when_saw_flop == 1
        assert a.wwsf_percent == pytest.approx(0.5, abs=1e-3)
        assert b.wwsf_percent == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------- C-15..C-17: observation features


class TestObservationFeaturesCoverage:
    def test_c15_observation_features_layout_for_known_opponent(self):
        """All 12 features land in the right slot, in expected order."""
        tracker = OpponentTracker()
        # Initialize an opponent via the real init path, then manually pin
        # the per-stat values so the feature layout is deterministic.
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.FOLD)
        _act(tracker, pid=1, action=Action.FOLD)
        _end(tracker, winners=[0])

        # Now manually populate known values on opp 1 for layout assertion
        opp = tracker.opponents[1]
        opp.vpip = 0.30
        opp.pfr = 0.20
        opp.af = 1.5            # / 3.0 -> 0.5 in features
        opp.three_bet_percent = 0.10
        opp.cbet_percent = 0.60
        opp.fold_to_cbet_percent = 0.40
        opp.went_to_showdown_percent = 0.25
        opp.win_at_showdown_percent = 0.55
        opp.wwsf_percent = 0.45
        opp.fold_to_3bet_after_raise_percent = 0.65
        opp.squeeze_percent = 0.05
        opp.confidence = 0.7

        features = tracker.get_observation_features(
            hero_id=0, opponent_ids=[1], max_opponents=9,
            features_per_opponent=12,
        )

        assert len(features) == 9 * 12
        # Slot 0 (opp id=1)
        slot0 = features[:12]
        assert slot0 == pytest.approx(
            [0.30, 0.20, 0.5, 0.10, 0.60, 0.40, 0.25, 0.55, 0.45, 0.65, 0.05, 0.7],
            abs=1e-3,
        )
        # Slots 1..8 zero-padded
        assert features[12:] == [0.0] * (8 * 12)

    def test_c16_neutral_defaults_for_unknown_opponent(self):
        """A player_id in opponent_ids but not in tracker.opponents uses defaults."""
        tracker = OpponentTracker()
        features = tracker.get_observation_features(
            hero_id=0, opponent_ids=[99], max_opponents=9,
            features_per_opponent=12,
        )

        # First slot should be the neutral defaults.
        assert features[:12] == [0.3, 0.2, 0.33, 0.1, 0.5, 0.5, 0.2, 0.4, 0.4, 0.5, 0.05, 0.0]
        # Rest zero-padded.
        assert features[12:] == [0.0] * (8 * 12)

    def test_c17_zero_pad_for_missing_slots(self):
        """Empty opponent slots are zero-padded across all 12 dims each."""
        tracker = OpponentTracker()
        features = tracker.get_observation_features(
            hero_id=0, opponent_ids=[], max_opponents=9,
            features_per_opponent=12,
        )

        assert len(features) == 9 * 12
        assert features == [0.0] * (9 * 12)

    def test_c17b_clipping_caps_features_at_one(self):
        """Outlier values are clipped at 1.0 (except confidence which is already capped)."""
        tracker = OpponentTracker()
        tracker.start_hand(1, _players(2), dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, action=Action.FOLD)
        _act(tracker, pid=1, action=Action.FOLD)
        _end(tracker, winners=[0])

        opp = tracker.opponents[1]
        opp.vpip = 5.0
        opp.pfr = 9.0
        opp.af = 100.0  # / 3.0 -> 33.3, clipped to 1.0
        opp.three_bet_percent = 2.0
        opp.cbet_percent = 2.0
        opp.fold_to_cbet_percent = 2.0
        opp.went_to_showdown_percent = 2.0
        opp.win_at_showdown_percent = 2.0
        opp.wwsf_percent = 2.0
        opp.fold_to_3bet_after_raise_percent = 2.0
        opp.squeeze_percent = 2.0
        opp.confidence = 0.5  # untouched in clip path

        features = tracker.get_observation_features(
            hero_id=0, opponent_ids=[1], max_opponents=9,
            features_per_opponent=12,
        )

        slot0 = features[:12]
        # All values 0..10 clipped at 1.0, confidence stays at 0.5.
        assert slot0[:11] == [1.0] * 11
        assert slot0[11] == pytest.approx(0.5, abs=1e-3)
