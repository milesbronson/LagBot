"""
Bug-witnessing tests for OpponentTracker.

Each test in this file targets a specific bug identified in
working_docs/audit_2026-05-10.md §2 and tracked in
working_docs/tests_plan_opponent_tracker_2026-05-10.md.

Tests are written to assert *desired* behavior. When first run, the P0/P1
tests should FAIL — that proves the bug is real and reproducible. After
the corresponding fix in opponent_tracker.py, the test should pass.
"""

import pytest
from src.poker_env.opponent_tracker import OpponentTracker, Action, Street
from src.poker_env.texas_holdem_env import TexasHoldemEnv


# --------------------------------------------------------------------- helpers


def _three_players():
    return [
        {'id': 0, 'name': 'A', 'stack': 1000},
        {'id': 1, 'name': 'B', 'stack': 1000},
        {'id': 2, 'name': 'C', 'stack': 1000},
    ]


def _heads_up():
    return [
        {'id': 0, 'name': 'A', 'stack': 1000},
        {'id': 1, 'name': 'B', 'stack': 1000},
    ]


def _four_players():
    return [
        {'id': 0, 'name': 'A', 'stack': 1000},
        {'id': 1, 'name': 'B', 'stack': 1000},
        {'id': 2, 'name': 'C', 'stack': 1000},
        {'id': 3, 'name': 'D', 'stack': 1000},
    ]


def _act(tracker, *, pid, name, action, street=Street.PREFLOP, amount=0,
         pot=10, stack_before=1000, stack_after=1000, position=0):
    tracker.record_action(
        player_id=pid, player_name=name,
        action=action, amount=amount,
        pot_size=pot, stack_before=stack_before, stack_after=stack_after,
        street=street, position=position,
    )


# --------------------------------------------------------------------- P0 tests


class TestThreeBetLogic:
    """T1, T2, T3 — 3-bet detection."""

    def test_t1_three_bet_opportunity_counts_when_facing_a_raise(self):
        """
        T1: When A raises preflop and B+C fold without 3-betting, B and C
        each had a 3-bet opportunity (they faced a raise and could re-raise).
        Currently this is undercounted — opportunity is only credited when
        the player themselves raised.
        """
        tracker = OpponentTracker()
        players = _three_players()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10)
        _act(tracker, pid=1, name='B', action=Action.FOLD)
        _act(tracker, pid=2, name='C', action=Action.FOLD)

        tracker.end_hand(winners=[0], winnings={0: 3, 1: -1, 2: -2},
                         final_stacks={0: 1003, 1: 999, 2: 998})

        assert tracker.opponents[1].three_bet_opportunities == 1, (
            "B faced a raise and could have 3-bet; that's an opportunity")
        assert tracker.opponents[2].three_bet_opportunities == 1, (
            "C also faced A's raise; also an opportunity")
        assert tracker.opponents[1].three_bet_count == 0
        assert tracker.opponents[2].three_bet_count == 0
        assert tracker.opponents[0].three_bet_opportunities == 0, (
            "A acted first; never faced a raise; no opportunity")

    def test_t2_three_bet_count_requires_earlier_raise_from_someone_else(self):
        """
        T2: Heads-up. A raises, B re-raises (3-bet). B should be credited
        with a 3-bet; A should not.
        """
        tracker = OpponentTracker()
        players = _heads_up()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10)
        _act(tracker, pid=1, name='B', action=Action.RAISE, amount=30)
        _act(tracker, pid=0, name='A', action=Action.FOLD)

        tracker.end_hand(winners=[1], winnings={0: -10, 1: 10},
                         final_stacks={0: 990, 1: 1010})

        assert tracker.opponents[1].three_bet_count == 1, (
            "B re-raised after A's open; that's a 3-bet")
        assert tracker.opponents[0].three_bet_count == 0, (
            "A's first action was an open raise, not a 3-bet")
        assert tracker.opponents[1].three_bet_opportunities == 1

    def test_t3_four_bet_does_not_count_as_three_bet(self):
        """
        T3: A raises, B re-raises (3-bet), A re-raises again (4-bet), B folds.
        A's second raise is a 4-bet — it should NOT increment three_bet_count
        for A. Current code counts any player with >=2 raises in one hand as
        having 3-bet.
        """
        tracker = OpponentTracker()
        players = _heads_up()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10)   # open
        _act(tracker, pid=1, name='B', action=Action.RAISE, amount=30)   # 3-bet
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=90)   # 4-bet
        _act(tracker, pid=1, name='B', action=Action.FOLD)

        tracker.end_hand(winners=[0], winnings={0: 30, 1: -30},
                         final_stacks={0: 1030, 1: 970})

        assert tracker.opponents[0].three_bet_count == 0, (
            "A's second raise is a 4-bet, not a 3-bet")
        assert tracker.opponents[1].three_bet_count == 1, (
            "B's only raise was the 3-bet")


class TestFoldToCbet:
    def test_t4_fold_to_cbet_across_players(self):
        """
        T4: A raises preflop, B calls. Flop: A bets (c-bet), B folds.
        B faced a c-bet and folded; B's counters should increment.
        Current code requires the same player to both bet AND fold the flop
        which is impossible — counters are always equal or zero.
        """
        tracker = OpponentTracker()
        players = _heads_up()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        # Preflop: A raises, B calls
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.CALL, amount=10, street=Street.PREFLOP)
        # Flop: A c-bets, B folds
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=15, street=Street.FLOP)
        _act(tracker, pid=1, name='B', action=Action.FOLD, street=Street.FLOP)

        tracker.end_hand(winners=[0], winnings={0: 20, 1: -10},
                         final_stacks={0: 1020, 1: 990})

        assert tracker.opponents[1].faced_flop_cbet == 1, (
            "B faced A's c-bet on the flop")
        assert tracker.opponents[1].folded_to_flop_cbet == 1, (
            "B folded to that c-bet")
        assert tracker.opponents[0].faced_flop_cbet == 0, (
            "A made the c-bet; didn't face one")


class TestCbetLogic:
    def test_t5_cbet_only_counts_when_preflop_aggressor_opens_flop(self):
        """
        T5: A raises preflop, B and C call. On the flop, B bets (a 'donk
        bet' — B was not the preflop aggressor). A raises in response.
        A's flop raise is NOT a c-bet because A didn't OPEN the flop
        betting. A had a c-bet opportunity (they were the preflop raiser
        who saw a flop) but they didn't take it.
        """
        tracker = OpponentTracker()
        players = _three_players()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        # Preflop: A raises, B + C call
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.CALL, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=2, name='C', action=Action.CALL, amount=10, street=Street.PREFLOP)
        # Flop: B donk-bets, A raises, C folds, B folds
        _act(tracker, pid=1, name='B', action=Action.RAISE, amount=15, street=Street.FLOP)
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=45, street=Street.FLOP)
        _act(tracker, pid=2, name='C', action=Action.FOLD, street=Street.FLOP)
        _act(tracker, pid=1, name='B', action=Action.FOLD, street=Street.FLOP)

        tracker.end_hand(winners=[0], winnings={0: 45, 1: -25, 2: -10},
                         final_stacks={0: 1045, 1: 975, 2: 990})

        assert tracker.opponents[0].flop_cbet_opportunities == 1, (
            "A was preflop aggressor and saw a flop — they had a c-bet opportunity")
        assert tracker.opponents[0].flop_cbet_made == 0, (
            "A did not open flop betting (B did); A's raise was a response, not a c-bet")
        assert tracker.opponents[1].flop_cbet_made == 0, (
            "B was not the preflop aggressor")


class TestWentToShowdown:
    def test_t6_wtsd_counts_showdown_losers(self):
        """
        T6: Heads-up. Preflop A raises, B calls. Flop check-check, turn
        check-check, river check-check. Showdown: A wins, B loses.
        Both players went to showdown. Current code excludes B because
        they didn't win and didn't fold.
        """
        tracker = OpponentTracker()
        players = _heads_up()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        # Preflop: A raises, B calls
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.CALL, amount=10, street=Street.PREFLOP)
        # Flop: check-check
        _act(tracker, pid=1, name='B', action=Action.CHECK, street=Street.FLOP)
        _act(tracker, pid=0, name='A', action=Action.CHECK, street=Street.FLOP)
        # Turn: check-check
        _act(tracker, pid=1, name='B', action=Action.CHECK, street=Street.TURN)
        _act(tracker, pid=0, name='A', action=Action.CHECK, street=Street.TURN)
        # River: check-check
        _act(tracker, pid=1, name='B', action=Action.CHECK, street=Street.RIVER)
        _act(tracker, pid=0, name='A', action=Action.CHECK, street=Street.RIVER)

        # A wins at showdown
        tracker.end_hand(winners=[0], winnings={0: 10, 1: -10},
                         final_stacks={0: 1010, 1: 990})

        assert tracker.opponents[0].went_to_showdown == 1
        assert tracker.opponents[1].went_to_showdown == 1, (
            "B reached showdown and lost; should still count as WTSD")
        assert tracker.opponents[0].showdown_wins == 1
        assert tracker.opponents[1].showdown_wins == 0


class TestPosition:
    """T7 — position-relative-to-button. Runs against the env."""

    def test_t7_position_rotates_with_button(self):
        """
        T7: Play multiple hands. The dealer button rotates. Each player's
        recorded position (relative to button) should change each hand.

        Today the env's _calculate_player_positions stores {player_id: list_index},
        which never changes between hands. We assert positions DO rotate.
        """
        env = TexasHoldemEnv(num_players=3, starting_stack=1000, track_opponents=True)
        positions_by_hand = []

        for _ in range(3):
            env.reset()
            hand = env.opponent_tracker.current_hand
            positions_by_hand.append(dict(hand.players_positions))
            # Burn the hand quickly so the next reset advances the button
            done = False
            steps = 0
            while not done and steps < 200:
                obs, reward, terminated, truncated, info = env.step(0)  # everyone folds
                done = terminated or truncated
                steps += 1

        # Each player's position should differ across the three hands (button rotates)
        for pid in range(3):
            positions_for_pid = [positions_by_hand[i].get(pid) for i in range(3)]
            assert len(set(positions_for_pid)) > 1, (
                f"Player {pid} had constant position {positions_for_pid} across "
                f"3 hands — button is not rotating positions")


# --------------------------------------------------------------------- P1 tests


class TestSqueeze:
    def test_t8_squeeze_requires_earlier_raise_and_caller_from_others(self):
        """
        T8: 4 players. A raises preflop, B calls, C raises (squeeze!),
        D folds, A folds, B folds. C is credited with a squeeze.
        Negative case: another hand where there's no caller in front of
        the third aggressor — that's a 3-bet, not a squeeze.
        """
        tracker = OpponentTracker()
        players = _four_players()
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)

        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.CALL, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=2, name='C', action=Action.RAISE, amount=40, street=Street.PREFLOP)
        _act(tracker, pid=3, name='D', action=Action.FOLD, street=Street.PREFLOP)
        _act(tracker, pid=0, name='A', action=Action.FOLD, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.FOLD, street=Street.PREFLOP)

        tracker.end_hand(winners=[2], winnings={0: -10, 1: -10, 2: 20, 3: 0},
                         final_stacks={0: 990, 1: 990, 2: 1020, 3: 1000})

        assert tracker.opponents[2].squeeze_opportunities == 1, (
            "C faced a raise + a call before acting — squeeze opportunity")
        assert tracker.opponents[2].squeeze_attempts == 1, (
            "C raised in that spot — squeeze attempt")
        assert tracker.opponents[0].squeeze_opportunities == 0, (
            "A was the opener; no squeeze possible")
        assert tracker.opponents[1].squeeze_opportunities == 0, (
            "B faced a raise but no prior caller; not a squeeze spot")


class TestFoldTo3BetAfterRaising:
    def test_t9_fold_to_3bet_is_per_hand_not_lifetime(self):
        """
        T9: Heads-up. Hand 1: A raises, B re-raises, A folds.
        Hand 2: A folds preflop (no raise).
        Hand 3: A calls preflop (no raise).

        A's faced_3bet_after_raise should be 1 (only hand 1).
        Current code: any subsequent hand where len(preflop_actions)>=2
        and len(preflop_raises)==1 fires the branch because the lifetime
        gate `raised_preflop > 0` stays true forever.
        """
        tracker = OpponentTracker()
        players = _heads_up()

        # Hand 1: A raises, B 3-bets, A folds
        tracker.start_hand(1, players, dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, name='A', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.RAISE, amount=30, street=Street.PREFLOP)
        _act(tracker, pid=0, name='A', action=Action.FOLD, street=Street.PREFLOP)
        tracker.end_hand(winners=[1], winnings={0: -10, 1: 10},
                         final_stacks={0: 990, 1: 1010})

        # Hand 2: A folds preflop (no raise this hand)
        tracker.start_hand(2, players, dealer_position=1, small_blind=1, big_blind=2)
        _act(tracker, pid=1, name='B', action=Action.RAISE, amount=10, street=Street.PREFLOP)
        _act(tracker, pid=0, name='A', action=Action.FOLD, street=Street.PREFLOP)
        tracker.end_hand(winners=[1], winnings={0: -1, 1: 1},
                         final_stacks={0: 989, 1: 1011})

        # Hand 3: A calls preflop, no raise from A
        tracker.start_hand(3, players, dealer_position=0, small_blind=1, big_blind=2)
        _act(tracker, pid=0, name='A', action=Action.CALL, amount=2, street=Street.PREFLOP)
        _act(tracker, pid=1, name='B', action=Action.CHECK, street=Street.PREFLOP)
        _act(tracker, pid=0, name='A', action=Action.CHECK, street=Street.FLOP)
        _act(tracker, pid=1, name='B', action=Action.CHECK, street=Street.FLOP)
        tracker.end_hand(winners=[0], winnings={0: 1, 1: -1},
                         final_stacks={0: 990, 1: 1010})

        assert tracker.opponents[0].faced_3bet_after_raise == 1, (
            "A only faced a 3-bet-after-their-raise in hand 1")
        assert tracker.opponents[0].folded_to_3bet_after_raise == 1


# --------------------------------------------------------------------- Sanity tests


class TestSanity:
    """T13, T14 — should pass as a smoke test."""

    def test_t13_confidence_scales_with_hands_played(self):
        tracker = OpponentTracker()
        players = _heads_up()

        # Play 50 hands (player 1 folds each time)
        for h in range(50):
            tracker.start_hand(h + 1, players, dealer_position=0, small_blind=1, big_blind=2)
            _act(tracker, pid=1, name='B', action=Action.FOLD)
            tracker.end_hand(winners=[0], winnings={0: 1, 1: -1},
                             final_stacks={0: 1000 + h, 1: 1000 - h})

        assert tracker.opponents[1].hands_played == 50
        assert tracker.opponents[1].confidence == pytest.approx(0.5)

        # Play 50 more
        for h in range(50, 100):
            tracker.start_hand(h + 1, players, dealer_position=0, small_blind=1, big_blind=2)
            _act(tracker, pid=1, name='B', action=Action.FOLD)
            tracker.end_hand(winners=[0], winnings={0: 1, 1: -1},
                             final_stacks={0: 1000 + h, 1: 1000 - h})

        assert tracker.opponents[1].confidence == pytest.approx(1.0)

    def test_t14_unknown_opponent_returns_none(self):
        tracker = OpponentTracker()
        assert tracker.get_opponent_stats(opponent_id=99) is None
