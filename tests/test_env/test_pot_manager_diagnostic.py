"""
Diagnostic tests for the 8 remaining game-logic failures, pinning each
suspected root cause documented in working_docs/bug_report_2026-05-11.md.

These tests call game-state primitives (place_bet, calculate_side_pots,
distribute_pots, env.step) directly with hand-rolled state. They avoid
asserting on whichever player is "first to act" — which is what the
TestAllInCurrentBetFix tests got wrong — by reading game.get_current_player()
at the point of action.

Tests are organised by bug ID from the report:

  PROD-1: env.step after terminal re-distributes the pot
  PROD-2: auto-rebuy creates chips silently on reset()
  DESIGN-1: sub-min all-in semantics (call vs reopen)
  DESIGN-2: uncalled-bet refund vs single-eligible side pot

Each test is intentionally tight and named with its bug ID so failures
point straight at the report.
"""

import pytest
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.poker_env.game_state import GameState
from src.poker_env.pot_manager import PotManager
from src.poker_env.player import Player


# ---------------------------------------------------------------------------
# PROD-1: env.step after terminal re-distributes the pot
# ---------------------------------------------------------------------------


class TestProd1TerminalStepReDistributes:
    """Pin: stepping the env after `terminated=True` re-runs determine_winners
    and pays out the same pot again. Stacks must not change once the hand is
    over until reset() is called."""

    def _drive_to_terminal(self):
        env = TexasHoldemEnv(num_players=3, starting_stack=2500)
        env.game_state.players[1].stack = 500
        env.game_state.players[2].stack = 800
        env.game_state.players[0].stack = 1200
        env.reset()
        all_in_idx = 2 + len(env.raise_bins)
        terminated = False
        for _ in range(3):
            _, _, terminated, _, _ = env.step(all_in_idx)
            if terminated:
                break
        assert terminated, "Setup failed: three all-ins should terminate the hand"
        return env

    def test_stacks_frozen_after_terminal(self):
        env = self._drive_to_terminal()
        stacks_at_terminal = [p.stack for p in env.game_state.players]

        for _ in range(5):
            env.step(1)

        stacks_after_extra_steps = [p.stack for p in env.game_state.players]
        assert stacks_after_extra_steps == stacks_at_terminal, (
            f"PROD-1: stacks changed after terminal step. "
            f"At terminal: {stacks_at_terminal}, after 5 extra steps: "
            f"{stacks_after_extra_steps}"
        )

    def test_total_chips_in_play_frozen_after_terminal(self):
        env = self._drive_to_terminal()
        total_at_terminal = sum(p.stack for p in env.game_state.players)

        for _ in range(5):
            env.step(1)

        total_after = sum(p.stack for p in env.game_state.players)
        assert total_after == total_at_terminal, (
            f"PROD-1: chips created after terminal. "
            f"At terminal: {total_at_terminal}, after 5 extra steps: {total_after}"
        )

    def test_terminal_step_returns_terminated_true(self):
        env = self._drive_to_terminal()
        _, _, terminated, _, _ = env.step(1)
        assert terminated, "Step after terminal should still report terminated=True"


# ---------------------------------------------------------------------------
# PROD-2: auto-rebuy creates chips on reset()
# ---------------------------------------------------------------------------


class TestProd2AutoRebuyAccounting:
    """Pin: when a player has stack=0 at reset() time, the env injects
    starting_stack chips. This is intentional for training continuity, but
    chip-conservation tests must include `total_buy_in` in their accounting,
    or the auto-rebuy must be opt-in.

    These tests document the current behaviour so a future config change is
    visible."""

    def test_busted_player_rebuy_on_reset_creates_chips(self):
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        env.reset()

        env.game_state.players[0].stack = 0
        env.game_state.players[1].stack = 0
        before_reset_stacks = sum(p.stack for p in env.game_state.players)

        env.reset()
        after_reset_stacks = sum(p.stack for p in env.game_state.players)
        after_reset_with_pot = after_reset_stacks + env.game_state.pot_manager.get_pot_total()

        # Two players rebought $1000 each — net chip injection of $2000.
        # New hand also posted blinds (chips moved from stack into pot).
        assert after_reset_with_pot == before_reset_stacks + 2 * 1000, (
            "PROD-2: total chips (stacks + pot) should grow by exactly 2 * starting_stack"
        )

    def test_total_buy_in_increments_on_rebuy(self):
        env = TexasHoldemEnv(num_players=2, starting_stack=1000)
        env.reset()
        p0 = env.game_state.players[0]
        buy_in_before = p0.total_buy_in

        p0.stack = 0
        env.reset()

        assert p0.total_buy_in == buy_in_before + 1000, (
            "PROD-2: rebuy should be reflected in total_buy_in for accounting"
        )

    def test_conservation_invariant_with_buy_ins(self):
        """The real invariant: sum(stacks) + sum(pots) ==
        sum(total_buy_in across players) at any between-hands moment."""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        env.reset()

        for _ in range(5):
            env.reset()
            done = False
            steps = 0
            while not done and steps < 50:
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                done = term or trunc
                steps += 1

            total_stacks = sum(p.stack for p in env.game_state.players)
            total_buy_in = sum(p.total_buy_in for p in env.game_state.players)
            assert total_stacks <= total_buy_in, (
                f"Stacks ({total_stacks}) > total buy-ins ({total_buy_in}) — "
                "chips created without recorded buy-in"
            )


# ---------------------------------------------------------------------------
# DESIGN-1: sub-min all-in semantics
# ---------------------------------------------------------------------------


class TestDesign1SubMinAllIn:
    """Pin: when stack < to_call, the player goes all-in for stack. When
    0 < (stack - to_call) < min_raise, the player CAN go all-in but it
    should not reopen action for previous raisers."""

    @pytest.fixture
    def pm(self):
        return PotManager(small_blind=5, big_blind=10)

    def test_stack_less_than_to_call_is_all_in(self, pm):
        pm.start_new_hand()
        pm.current_bet = 100  # someone has bet 100
        p = Player(0, 30, "Short")  # only 30 chips
        amount, action = pm.place_bet(p, 30)
        assert action == "all-in", f"Expected all-in (stack<to_call), got {action}"
        assert p.is_all_in
        assert p.stack == 0
        assert p.total_bet_this_hand == 30
        assert pm.current_bet == 100, (
            "current_bet must NOT update when all-in is less than to_call"
        )

    def test_stack_equals_to_call_is_all_in_call(self, pm):
        pm.start_new_hand()
        pm.current_bet = 100
        p = Player(0, 100, "Exact")
        amount, action = pm.place_bet(p, 100)
        assert action == "all-in"
        assert p.is_all_in
        assert pm.current_bet == 100, "current_bet must not change on call-sized all-in"

    def test_stack_just_over_to_call_is_all_in_short_raise(self, pm):
        """DESIGN-1 latent: stack=110, to_call=100, min_raise=10. The 10-chip
        raise IS exactly min_raise so this is a legal raise/all-in. But if
        we set min_raise=20, the 10-chip "raise" should be an all-in that
        does NOT reopen action.

        Current implementation always sets current_bet and last_raise_amount
        regardless of whether the raise meets min_raise. This test pins
        current behaviour; if/when we implement the no-reopen rule, the
        assertion below for last_raise_amount will need updating."""
        pm.start_new_hand()
        pm.current_bet = 100
        pm.min_raise = 20  # min raise is 20
        p = Player(0, 110, "Squeeze")
        amount, action = pm.place_bet(p, 110)
        assert action == "all-in"
        assert p.is_all_in
        assert pm.current_bet == 110, (
            "current_bet IS updated currently (no-reopen rule not yet implemented)"
        )


# ---------------------------------------------------------------------------
# DESIGN-2: uncalled-bet refund vs single-eligible side pot
# ---------------------------------------------------------------------------


class TestDesign2UncalledBetRefund:
    """Pin: when one player's total bet exceeds everyone else's, the
    uncalled excess should refund to that player (not form a single-eligible
    side pot). Chip outcome is identical when the over-bettor is in the
    only-eligible group, but the pot structure differs."""

    @pytest.fixture
    def pm(self):
        return PotManager(small_blind=5, big_blind=10)

    def test_overbet_creates_uncontested_side_pot_currently(self, pm):
        """Documents current behaviour: 2 pots."""
        pm.start_new_hand()
        players = [Player(0, 0, "P0"), Player(1, 850, "P1")]
        players[0].total_bet_this_hand = 1000
        players[1].total_bet_this_hand = 150

        pots = pm.calculate_side_pots(players)
        assert len(pots) == 2, "Current behaviour: 2 pots (main + uncontested side)"
        assert pots[0].amount == 300, "Main pot = 150 * 2 = 300"
        assert pots[1].amount == 850, "Side pot = uncalled 850"
        assert set(pots[1].eligible_players) == {0}, "Only P0 eligible for side pot"

    def test_overbet_chip_outcome_is_correct_after_refund(self, pm):
        """After the PROD-3 fix, distribute_pots refunds the uncalled excess
        directly to the over-bettor's stack and produces a single contested
        pot. P0 (best hand) wins the matched 300; the 700 uncalled excess
        is refunded to P0's stack, not paid out via winnings. Total chips
        returned to P0 == 300 (winnings) + 700 (refund) == 1000."""
        pm.start_new_hand()
        players = [Player(0, 0, "P0"), Player(1, 850, "P1")]
        players[0].total_bet_this_hand = 1000
        players[1].total_bet_this_hand = 150
        p0_stack_before = players[0].stack

        winnings = pm.distribute_pots(players, hand_ranks={0: 100, 1: 200})
        assert winnings[0] == 300, "P0 wins the matched pot of 300"
        assert winnings[1] == 0
        # The 700 (1000 - 150 second-highest * 2 ... wait the excess is
        # 1000 - 150 = 850). 850 refunded directly to P0's stack.
        assert players[0].stack == p0_stack_before + 850, "P0 refunded 850 to stack"
        # Total chip outcome for P0: 850 (refund) + 300 (winnings) = 1150
        # — same outcome as the pre-fix two-pots-to-one-player path.

    def test_overbet_with_p1_winning_p1_only_gets_matched_portion(self, pm):
        """If P1 had the better hand but couldn't match P0's all-in, P1 still
        only gets the matched portion. P0's 850 uncalled excess is refunded
        directly to P0's stack (not via winnings)."""
        pm.start_new_hand()
        players = [Player(0, 0, "P0"), Player(1, 850, "P1")]
        players[0].total_bet_this_hand = 1000
        players[1].total_bet_this_hand = 150
        p0_stack_before = players[0].stack

        winnings = pm.distribute_pots(players, hand_ranks={0: 200, 1: 100})
        assert winnings[1] == 300, "P1 wins matched 300"
        assert winnings[0] == 0, "P0 wins nothing via the contested pot"
        assert players[0].stack == p0_stack_before + 850, "P0 refunded 850 to stack"


# ---------------------------------------------------------------------------
# Side-pot calculation sanity (regression coverage)
# ---------------------------------------------------------------------------


class TestSidePotCalculationSanity:
    """Targeted side-pot tests independent of env.step. These should be
    GREEN today and stay green after PROD-1 / DESIGN-2 fixes."""

    @pytest.fixture
    def pm(self):
        return PotManager(small_blind=5, big_blind=10)

    def test_three_way_equal_one_pot(self, pm):
        pm.start_new_hand()
        players = [Player(i, 0, f"P{i}") for i in range(3)]
        for p in players:
            p.total_bet_this_hand = 100

        pots = pm.calculate_side_pots(players)
        assert len(pots) == 1
        assert pots[0].amount == 300
        assert set(pots[0].eligible_players) == {0, 1, 2}

    def test_three_way_layered_three_pots(self, pm):
        pm.start_new_hand()
        players = [Player(0, 0, "Short"), Player(1, 0, "Mid"), Player(2, 0, "Deep")]
        players[0].total_bet_this_hand = 50
        players[1].total_bet_this_hand = 150
        players[2].total_bet_this_hand = 300

        pots = pm.calculate_side_pots(players)
        assert len(pots) == 3
        assert pots[0].amount == 150  # 50 * 3
        assert pots[1].amount == 200  # 100 * 2
        assert pots[2].amount == 150  # 150 * 1
        assert set(pots[0].eligible_players) == {0, 1, 2}
        assert set(pots[1].eligible_players) == {1, 2}
        assert set(pots[2].eligible_players) == {2}

    def test_folded_player_contribution_stays_in_pot(self, pm):
        """If a player puts chips in and then folds, those chips remain in
        the pot and are eligible for distribution to the remaining players."""
        pm.start_new_hand()
        players = [Player(0, 0, "P0"), Player(1, 0, "P1"), Player(2, 0, "P2")]
        players[0].total_bet_this_hand = 100
        players[1].total_bet_this_hand = 100
        players[2].total_bet_this_hand = 50
        players[2].is_active = False  # folded

        pots = pm.calculate_side_pots(players)
        total_in_pots = sum(p.amount for p in pots)
        total_contributed = sum(p.total_bet_this_hand for p in players)
        assert total_in_pots == total_contributed, (
            f"Folded player's chips were lost: contributed={total_contributed}, "
            f"in pots={total_in_pots}"
        )


# ---------------------------------------------------------------------------
# Position-awareness: the bugged TestAllInCurrentBetFix scenarios, rewritten
# correctly. These should be GREEN today and pin the actual production
# behaviour the original tests were trying to verify.
# ---------------------------------------------------------------------------


class TestAllInCurrentBetFixRewritten:
    """The original TestAllInCurrentBetFix tests assumed P0 is first to act.
    In a 3-player game after button rotation, P1 is first to act preflop.
    These versions read get_current_player() to act on the right player."""

    def _make_game(self):
        return GameState(num_players=3, starting_stack=1000, small_blind=5, big_blind=10)

    def test_all_in_as_raise_updates_current_bet(self):
        game = self._make_game()
        game.start_new_hand()

        actor = game.get_current_player()
        actor_starting_stack = actor.stack
        to_call_before = game.pot_manager.current_bet - actor.current_bet

        # Total bet = whole stack; this is an all-in raise (stack > to_call).
        total_bet = actor_starting_stack + actor.current_bet
        game.execute_action(2, raise_amount=total_bet)

        assert actor.is_all_in, "Actor should be all-in"
        assert actor.stack == 0
        assert game.pot_manager.current_bet == actor.current_bet, (
            f"current_bet ({game.pot_manager.current_bet}) should equal "
            f"actor.current_bet ({actor.current_bet}) after an all-in raise"
        )

    def test_all_in_as_call_does_not_update_current_bet(self):
        game = self._make_game()
        game.start_new_hand()
        # Make sure SB current_bet < BB so a sub-stack actor can call exactly.
        # Preflop: P2=SB(5), P0=BB(10), P1 acts first with stack=1000.
        # Set P1.stack = 10 so calling the BB uses their whole stack.
        actor = game.get_current_player()
        actor.stack = 10

        current_bet_before = game.pot_manager.current_bet
        game.execute_action(1)  # call

        assert actor.is_all_in, "Actor should be all-in after calling with last chips"
        assert game.pot_manager.current_bet == current_bet_before, (
            "current_bet should NOT change on a call-sized all-in"
        )

    def test_three_player_all_in_sequence(self):
        game = self._make_game()
        game.start_new_hand()

        # P1 raises to 50 (raise of 50 from current_bet=10 means total_bet=50)
        p1 = game.get_current_player()
        game.execute_action(2, raise_amount=50)
        assert game.pot_manager.current_bet == 50
        assert p1.current_bet == 50

        # P2 (SB) goes all-in for their remaining stack
        p2 = game.get_current_player()
        p2_total_bet_after_allin = p2.current_bet + p2.stack
        game.execute_action(2, raise_amount=p2_total_bet_after_allin)
        assert p2.is_all_in
        assert game.pot_manager.current_bet == p2.current_bet
