"""Locks for the rules-v2 engine corrections (2026-08-17).

These encode REAL no-limit hold'em rules that the original engine got
wrong: heads-up button/blind assignment and action order, the big
blind's option, no phantom decisions on all-in run-outs, and min-raise
legality. If one of these fails, the engine has regressed to pre-v2
behavior — do not "fix" the test.
"""

from src.poker_env.game_state import GameState
from src.poker_env.player import Player
from src.poker_env.pot_manager import PotManager


def _heads_up_game():
    game = GameState(num_players=2, starting_stack=1000, small_blind=5, big_blind=10)
    game.start_new_hand()
    return game


def _three_player_game():
    game = GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)
    game.start_new_hand()
    return game


class TestHeadsUpButton:
    def test_button_posts_small_blind(self):
        game = _heads_up_game()
        btn = game.button_position
        other = 1 - btn
        assert game.players[btn].current_bet == 5, "button must post the SB"
        assert game.players[other].current_bet == 10, "non-button posts the BB"

    def test_button_acts_first_preflop(self):
        game = _heads_up_game()
        assert game.current_player_idx == game.button_position

    def test_big_blind_acts_first_postflop(self):
        game = _heads_up_game()
        btn = game.button_position
        # SB completes, BB checks its option -> flop
        game.execute_action(1)
        game.execute_action(1)
        assert game.is_betting_round_complete()
        game.advance_betting_round()
        assert game.current_player_idx == 1 - btn, "BB acts first postflop in HU"


class TestBigBlindOption:
    def test_bb_gets_option_after_limp_with_fold(self):
        game = _three_player_game()
        btn = game.button_position
        sb = (btn + 1) % 3
        bb = (sb + 1) % 3

        # Button folds, SB completes. The BB has NOT acted — the round
        # must stay open for the BB's option (check or raise).
        game.execute_action(0)  # button folds
        game.execute_action(1)  # SB calls
        assert not game.is_betting_round_complete(), \
            "BB must get its option after a limp+fold"
        assert game.current_player_idx == bb

        # BB exercises the option with a raise; SB must respond.
        game.execute_action(2, raise_amount=30)
        assert not game.is_betting_round_complete()
        assert game.current_player_idx == sb
        game.execute_action(1)  # SB calls the raise
        assert game.is_betting_round_complete()

    def test_bb_check_closes_the_round(self):
        game = _three_player_game()
        game.execute_action(0)  # button folds
        game.execute_action(1)  # SB completes
        game.execute_action(1)  # BB checks the option
        assert game.is_betting_round_complete()


class TestAllInRunout:
    def test_no_phantom_decisions_when_opponent_all_in_and_matched(self):
        game = _heads_up_game()
        # SB shoves, BB calls: both all-in (equal stacks) -> every later
        # street must report the round complete without any action.
        game.execute_action(2, raise_amount=game.get_current_player().stack)
        game.execute_action(1)
        assert game.is_betting_round_complete()
        game.advance_betting_round()  # flop
        assert game.is_betting_round_complete(), \
            "no live decision exists on an all-in run-out street"


class TestMinRaiseLegality:
    def test_tiny_reraise_downgraded_to_call(self):
        players = [Player(0, 1000, "P0"), Player(1, 1000, "P1")]
        pm = PotManager(small_blind=5, big_blind=10, min_raise_multiplier=1.0)
        pm.start_new_hand()
        pm.place_bet(players[0], 30)          # open to 30 (raise of 30)
        # 1 chip over the call is NOT a legal raise -> becomes a call.
        amount, action = pm.place_bet(players[1], 31)
        assert action == "call"
        assert pm.current_bet == 30, "an illegal tiny raise must not reopen betting"

    def test_short_all_in_does_not_shrink_min_raise(self):
        players = [Player(0, 1000, "P0"), Player(1, 43, "P1")]
        pm = PotManager(small_blind=5, big_blind=10, min_raise_multiplier=1.0)
        pm.start_new_hand()
        pm.place_bet(players[0], 30)
        min_raise_before = pm.min_raise
        # P1's all-in for 43 is a raise of 13 < min_raise: allowed (all-in)
        # but must not lower the ladder for everyone else.
        amount, action = pm.place_bet(players[1], 43)
        assert action == "all-in"
        assert pm.min_raise >= min_raise_before
