"""
Comprehensive test cases for all-in scenarios and side pot calculation
Based on real hand history that revealed bugs in the all-in logic
"""

import pytest
from src.poker_env.pot_manager import PotManager, Pot
from src.poker_env.player import Player
from src.poker_env.game_state import GameState, BettingRound


class TestAllInBasics:
    """Test basic all-in mechanics"""
    
    @pytest.fixture
    def pot_manager(self):
        return PotManager(small_blind=5, big_blind=10)
    
    @pytest.fixture
    def two_players(self):
        return [
            Player(0, 1000, "Alice"),
            Player(1, 1000, "Bob")
        ]
    
    def test_all_in_action_detection(self, pot_manager, two_players):
        """Test that all-in is correctly detected when player bets all remaining stack"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        # Player has exactly 50 chips, should go all-in
        two_players[0].stack = 50
        amount, action = pot_manager.place_bet(two_players[0], 50)
        
        assert amount == 50, "All-in amount not registered correctly"
        assert action == "all-in", "Action should be 'all-in'"
        assert two_players[0].is_all_in, "Player should be marked as all-in"
        assert two_players[0].stack == 0, "Stack should be 0 after all-in"
        assert two_players[0].total_bet_this_hand == 50, "total_bet_this_hand not updated"
    
    def test_all_in_updates_total_bet_this_hand(self, pot_manager, two_players):
        """CRITICAL: Ensure all-in properly updates total_bet_this_hand"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 5
        
        # Player bets 5 chips first (call)
        amount1, _ = pot_manager.place_bet(two_players[0], 5)
        assert two_players[0].total_bet_this_hand == 5
        
        # Later goes all-in for remaining 100
        two_players[0].stack = 100
        amount2, action = pot_manager.place_bet(two_players[0], 100)
        
        # total_bet_this_hand should now be 5 + 100 = 105
        assert two_players[0].total_bet_this_hand == 105, \
            f"Expected 105, got {two_players[0].total_bet_this_hand}"
        assert action == "all-in"
    
    def test_all_in_amount_goes_to_pot(self, pot_manager, two_players):
        """Ensure all-in amount is added to pot"""
        pot_manager.start_new_hand()
        pot_manager.current_bet = 10
        
        two_players[0].stack = 50
        initial_pot = pot_manager.pots[0].amount
        
        amount, action = pot_manager.place_bet(two_players[0], 50)
        
        final_pot = pot_manager.pots[0].amount
        assert final_pot == initial_pot + 50, \
            f"Pot should increase by 50, got increase of {final_pot - initial_pot}"


class TestHeadsUpAllIn:
    """Test all-in scenarios in heads-up play (2 players)"""
    
    @pytest.fixture
    def pot_manager(self):
        return PotManager(small_blind=5, big_blind=10)
    
    def test_headsup_action_sequence_scenario_1(self, pot_manager):
        """
        Scenario: Heads-up, button goes all-in after opponent raises
        This mimics your hand #3 situation
        """
        players = [
            Player(0, 1000, "Player_0"),
            Player(1, 1000, "Player_1_BTN")
        ]
        
        pot_manager.start_new_hand()
        
        # Blinds
        pot_manager.post_blinds(players[0], players[1])
        assert players[0].total_bet_this_hand == 5
        assert players[1].total_bet_this_hand == 10
        assert pot_manager.pots[0].amount == 15
        
        # Player 0 calls big blind
        to_call = pot_manager.current_bet - players[0].current_bet
        amount1, action1 = pot_manager.place_bet(players[0], to_call)
        assert action1 == "call"
        assert players[0].total_bet_this_hand == 10
        assert pot_manager.pots[0].amount == 20
        
        # Player 1 raises
        raise_amount = 20
        amount2, action2 = pot_manager.place_bet(players[1], raise_amount)
        assert action2 == "raise"
        assert players[1].current_bet == 30
        assert players[1].total_bet_this_hand == 30
        assert pot_manager.pots[0].amount == 40
        
        # Player 0 calls the raise
        to_call2 = pot_manager.current_bet - players[0].current_bet
        amount3, action3 = pot_manager.place_bet(players[0], to_call2)
        assert action3 == "call"
        assert players[0].total_bet_this_hand == 30
        assert pot_manager.pots[0].amount == 60
        
        print(f"After preflop betting:")
        print(f"  Player 0: stack={players[0].stack}, total_bet={players[0].total_bet_this_hand}")
        print(f"  Player 1: stack={players[1].stack}, total_bet={players[1].total_bet_this_hand}")
        print(f"  Pot: {pot_manager.pots[0].amount}")
    
    def test_headsup_river_all_in(self, pot_manager):
        """
        CRITICAL: Reproduce the exact failure from hand #3
        Player goes all-in on river, both players' final contributions need to be correct
        """
        players = [
            Player(0, 950, "Player_0"),  # After losing chips earlier in hand
            Player(1, 950, "Player_1_BTN")
        ]
        
        pot_manager.start_new_hand()
        
        # Simulate preflop and streets already played
        # Current state: both at 950 chips, pot is 100
        pot_manager.pots[0].amount = 100
        
        # River action: Player 0 is first to act, checks
        pot_manager.current_bet = 0
        players[0].reset_current_bet()
        players[1].reset_current_bet()
        
        amount1, action1 = pot_manager.place_bet(players[0], 0)
        assert action1 == "check"
        
        # Player 1 bets 100
        amount2, action2 = pot_manager.place_bet(players[1], 100)
        assert action2 == "raise"
        assert players[1].current_bet == 100
        assert players[1].total_bet_this_hand == 100
        
        # Player 0 goes all-in for remaining 950
        to_call = pot_manager.current_bet - players[0].current_bet
        amount3, action3 = pot_manager.place_bet(players[0], to_call + 950)
        
        print(f"\nRiver all-in sequence:")
        print(f"  Action 3: {action3}, amount={amount3}")
        print(f"  Player 0: stack={players[0].stack}, current_bet={players[0].current_bet}, total_bet={players[0].total_bet_this_hand}")
        print(f"  Player 1: stack={players[1].stack}, current_bet={players[1].current_bet}, total_bet={players[1].total_bet_this_hand}")
        
        assert action3 == "all-in", f"Expected 'all-in', got '{action3}'"
        assert players[0].stack == 0, f"Player 0 stack should be 0, got {players[0].stack}"
        assert players[0].total_bet_this_hand == 950, \
            f"Player 0 total_bet_this_hand should be 950, got {players[0].total_bet_this_hand}"
        
        # Critical: Pot must include both players' all-in amounts
        expected_pot = 100 + 100 + 950  # original + P1 bet + P0 all-in
        assert pot_manager.pots[0].amount == expected_pot, \
            f"Pot should be {expected_pot}, got {pot_manager.pots[0].amount}"


class TestSidePotCalculation:
    """Test side pot calculation with all-ins"""
    
    @pytest.fixture
    def pot_manager(self):
        return PotManager(small_blind=5, big_blind=10)
    
    @pytest.fixture
    def three_players(self):
        return [
            Player(0, 100, "ShortStack"),
            Player(1, 500, "MediumStack"),
            Player(2, 1000, "LargeStack")
        ]
    
    def test_single_all_in_side_pot(self, pot_manager, three_players):
        """
        Test scenario:
        - Player 0: all-in for 100
        - Player 1: bets 100
        - Player 2: bets 100
        
        Should create: Main pot (300) with all 3 eligible, side pot doesn't exist
        """
        pot_manager.start_new_hand()
        
        three_players[0].total_bet_this_hand = 100
        three_players[1].total_bet_this_hand = 100
        three_players[2].total_bet_this_hand = 100
        
        pots = pot_manager.calculate_side_pots(three_players)
        
        assert len(pots) == 1, "Should have 1 pot when everyone puts in same amount"
        assert pots[0].amount == 300, f"Main pot should be 300, got {pots[0].amount}"
        assert len(pots[0].eligible_players) == 3
    
    def test_multiple_all_in_levels_side_pots(self, pot_manager, three_players):
        """
        CRITICAL: Test the exact scenario from your hand
        
        - Player 0: goes all-in for 850 (has $850 remaining after earlier streets)
        - Player 1: calls 850
        
        Then at showdown:
        - Main pot: 1700 (850 from each)
        - No side pots (both all-in at same level)
        """
        pot_manager.start_new_hand()
        
        # Preflop: 15 chips already in pot (blinds and initial action)
        pot_manager.pots[0].amount = 100  # Pot before river
        
        # River betting
        three_players[0].total_bet_this_hand = 850  # All-in
        three_players[1].total_bet_this_hand = 850  # Called all-in
        three_players[2].total_bet_this_hand = 0    # Folded earlier
        
        pots = pot_manager.calculate_side_pots(three_players)
        
        assert len(pots) == 1, f"Should have 1 pot, got {len(pots)}"
        assert pots[0].amount == 1700, f"Pot should be 1700, got {pots[0].amount}"
        assert 0 in pots[0].eligible_players
        assert 1 in pots[0].eligible_players
        assert 2 not in pots[0].eligible_players, "Folded player shouldn't be eligible"
    
    def test_stacked_all_ins(self, pot_manager, three_players):
        """
        CRITICAL: Multiple all-ins at different levels
        
        - Player 0: all-in for 50
        - Player 1: all-in for 150  
        - Player 2: bets 300
        
        Should create:
        - Main pot (150):   50 from each player = 150 (all eligible)
        - Side pot (300):   100 from P1 and P2 = 200 (P1, P2 eligible)
        - Side pot 2 (150): 150 from P2 = 150 (P2 eligible)
        """
        pot_manager.start_new_hand()
        
        three_players[0].total_bet_this_hand = 50
        three_players[1].total_bet_this_hand = 150
        three_players[2].total_bet_this_hand = 300
        
        pots = pot_manager.calculate_side_pots(three_players)
        
        assert len(pots) == 3, f"Should have 3 pots, got {len(pots)}"
        
        # Main pot: 50 from each
        assert pots[0].amount == 150, f"Main pot should be 150, got {pots[0].amount}"
        assert set(pots[0].eligible_players) == {0, 1, 2}
        
        # Side pot 1: 100 from P1 and P2
        assert pots[1].amount == 200, f"Side pot 1 should be 200, got {pots[1].amount}"
        assert set(pots[1].eligible_players) == {1, 2}
        
        # Side pot 2: 150 from P2
        assert pots[2].amount == 150, f"Side pot 2 should be 150, got {pots[2].amount}"
        assert set(pots[2].eligible_players) == {2}



class TestPotDistribution:
    """Test pot distribution with side pots"""
    
    @pytest.fixture
    def pot_manager(self):
        return PotManager(small_blind=5, big_blind=10)
    
    def test_distribute_single_pot_winner(self):
        """Test distributing single pot to winner"""
        pot_manager = PotManager(5, 10)
        players = [
            Player(0, 0, "Alice"),
            Player(1, 0, "Bob")
        ]
        
        players[0].total_bet_this_hand = 500
        players[1].total_bet_this_hand = 500
        
        hand_ranks = {0: 1, 1: 2}  # Player 0 has better hand
        winnings = pot_manager.distribute_pots(players, hand_ranks)
        
        assert winnings[0] == 1000, f"Player 0 should win 1000, got {winnings[0]}"
        assert winnings[1] == 0, f"Player 1 should win 0, got {winnings[1]}"
    
    def test_distribute_side_pots_multiple_all_ins(self):
        """
        CRITICAL: Test distribution with stacked all-ins
        
        Setup:
        - Player 0: all-in for 100 (best hand rank 1)
        - Player 1: all-in for 200 (bad hand rank 3)
        - Player 2: bets 300 (middle hand rank 2)
        
        Expected:
        - Main pot (300): 3-way (100 x 3), goes to Player 0 (rank 1)
        - Side pot 1 (200): 2-way (100 x 2), goes to Player 0 (rank 1 vs 3)
        - Side pot 2 (100): 1-way, goes to Player 2 (only eligible)
        """
        pot_manager = PotManager(5, 10)
        players = [
            Player(0, 0, "ShortStack"),
            Player(1, 0, "MediumStack"),
            Player(2, 0, "DeepStack")
        ]
        
        players[0].total_bet_this_hand = 100
        players[1].total_bet_this_hand = 200
        players[2].total_bet_this_hand = 300
        
        hand_ranks = {
            0: 1,   # Best
            1: 3,   # Worst
            2: 2    # Middle
        }
        
        p2_stack_before = players[2].stack
        winnings = pot_manager.distribute_pots(players, hand_ranks)

        print(f"\nMulti-way pot distribution:")
        print(f"  Player 0 winnings: {winnings[0]}")
        print(f"  Player 1 winnings: {winnings[1]}")
        print(f"  Player 2 winnings: {winnings[2]}")
        print(f"  Total distributed: {sum(winnings.values())}")

        # PROD-3 fix: P2's 100-chip excess over P1 is uncalled, refunded to
        # P2's stack before distribution. After refund: P0=100, P1=200, P2=200.
        # Main pot (300) goes to P0; side pot (200, eligible P1/P2) goes to P2.
        assert winnings[0] == 300, f"P0 should win 300, got {winnings[0]}"
        assert winnings[1] == 0, f"P1 should win 0, got {winnings[1]}"
        assert winnings[2] == 200, f"P2 should win 200, got {winnings[2]}"
        assert sum(winnings.values()) == 500, "Distributed pot total after refund"
        # The 100-chip refund is applied directly to P2's stack.
        assert players[2].stack == p2_stack_before + 100, "P2 refund of 100"
    
    def test_distribute_split_pot_main_and_side(self):
        """
        Test split pot scenarios with side pots
        
        Setup:
        - Player 0: all-in for 100 (hand rank 1 - ties with P1)
        - Player 1: bets 200 (hand rank 1 - ties with P0)
        - Player 2: bets 300 (hand rank 3 - worst)
        
        Expected:
        - Main pot (300): Split between P0 and P1 = 150 each
        - Side pot (200): Split between P1 and P2... wait, P2 didn't go all-in
          Actually: P1 and P2 eligible, P1 wins with better rank
        """
        pot_manager = PotManager(5, 10)
        players = [
            Player(0, 0, "Player0"),
            Player(1, 0, "Player1"),
            Player(2, 0, "Player2")
        ]
        
        players[0].total_bet_this_hand = 100
        players[1].total_bet_this_hand = 200
        players[2].total_bet_this_hand = 300
        
        hand_ranks = {
            0: 1,   # Tie
            1: 1,   # Tie
            2: 3    # Worst
        }
        
        p2_stack_before = players[2].stack
        winnings = pot_manager.distribute_pots(players, hand_ranks)

        print(f"\nSplit pot distribution:")
        print(f"  Player 0: {winnings[0]}")
        print(f"  Player 1: {winnings[1]}")
        print(f"  Player 2: {winnings[2]}")

        # PROD-3 fix: P2's 100-chip excess over P1 is refunded before
        # distribution. After refund: P0=100, P1=200, P2=200.
        # Main pot (300): P0 and P1 tie, 150 each.
        # Side pot (200, eligible P1/P2): P1 wins (rank 1 vs 3).
        assert winnings[0] == 150, f"P0 should win 150, got {winnings[0]}"
        assert winnings[1] == 350, f"P1 should win 350, got {winnings[1]}"
        assert winnings[2] == 0, f"P2 should win 0 (refund handles excess), got {winnings[2]}"
        assert sum(winnings.values()) == 500
        assert players[2].stack == p2_stack_before + 100, "P2 refund of 100"

class TestAllInCurrentBetFix:
    """Test suite for all-in current_bet updates"""
    
    @pytest.fixture
    def game(self):
        """Create game with 3 players"""
        return GameState(
            num_players=3,
            starting_stack=1000,
            small_blind=5,
            big_blind=10
        )
    
    def test_all_in_as_raise_updates_current_bet(self, game):
        """Scenario 1: Player goes all-in with a RAISE.

        In a 3-player hand after the first button rotation, the actor
        first-to-act is the player after the BB (UTG). Use
        get_current_player() rather than hard-coding P0.
        """
        game.start_new_hand()

        actor = game.get_current_player()
        actor_starting_stack = actor.stack
        # Total bet for an all-in raise = actor's existing current_bet + stack.
        total_bet = actor.current_bet + actor_starting_stack
        game.execute_action(2, raise_amount=total_bet)

        assert actor.is_all_in, "Actor should be all-in"
        assert actor.stack == 0
        assert game.pot_manager.current_bet == actor.current_bet, (
            f"current_bet ({game.pot_manager.current_bet}) should equal "
            f"actor.current_bet ({actor.current_bet}) after an all-in raise"
        )
        # Other players still haven't acted.
        assert not game.is_betting_round_complete()
    
    def test_all_in_as_call_does_not_update_current_bet(self, game):
        """Scenario 2: Player goes all-in with a CALL (exact amount).

        Set the first-to-act player's stack equal to the BB so calling
        their last chip is an all-in. current_bet must NOT change.
        """
        game.start_new_hand()

        actor = game.get_current_player()
        actor.stack = game.pot_manager.current_bet - actor.current_bet
        current_bet_before = game.pot_manager.current_bet

        game.execute_action(1)  # call

        assert actor.is_all_in, "Actor should be all-in after calling with last chips"
        assert game.pot_manager.current_bet == current_bet_before, (
            f"current_bet should not change on a call-sized all-in: "
            f"was {current_bet_before}, is {game.pot_manager.current_bet}"
        )
    
    def test_all_in_short_stack_goes_all_in(self, game):
        """Scenario 3: Player has stack < to_call.

        Real poker rule: player goes all-in for their remaining stack (this
        does not reopen action for previous raisers). The original test
        expected a FOLD, which is incorrect. Verifies the all-in branch in
        place_bet handles stack < to_call correctly.
        """
        game.start_new_hand()

        actor = game.get_current_player()
        actor.stack = 3  # less than the BB call
        to_call = game.pot_manager.current_bet - actor.current_bet
        assert to_call > actor.stack, "Setup: stack must be less than to_call"

        game.execute_action(2, raise_amount=actor.current_bet + actor.stack)

        assert actor.is_all_in, "Actor should be all-in, not folded"
        assert actor.is_active, "All-in player remains active (can still win)"
        assert actor.stack == 0
    
    def test_three_player_all_in_sequence(self, game):
        """Integration: 3-player all-in sequence using actual current player
        each step (the original test confused position assignments).
        """
        game.start_new_hand()

        # Step 1: UTG raises to 50 total.
        utg = game.get_current_player()
        game.execute_action(2, raise_amount=50)
        assert game.pot_manager.current_bet == 50
        assert utg.current_bet == 50

        # Step 2: next actor goes all-in. raise_amount is TOTAL bet.
        actor2 = game.get_current_player()
        all_in_total = actor2.current_bet + actor2.stack
        game.execute_action(2, raise_amount=all_in_total)
        assert actor2.is_all_in
        assert game.pot_manager.current_bet == actor2.current_bet

        # Step 3: third actor calls the all-in.
        actor3 = game.get_current_player()
        game.execute_action(1)
        assert game.pot_manager.current_bet == actor2.current_bet, (
            "Calling an all-in should not move current_bet"
        )

        # Step 4: UTG (who raised to 50) must respond to the bigger all-in.
        game.execute_action(1)  # call

        assert game.is_betting_round_complete()



class TestCompleteHandScenarios:
    """Test complete hand scenarios to catch integration bugs"""
    
    def test_hand_3_reproduction(self):
        """
        REPRODUCE YOUR EXACT HAND #3 BUG
        
        Your hand history showed:
        - Both players end up winning chips from the pot
        - This is impossible - one player wins everything
        """
        pot_manager = PotManager(5, 10)
        players = [
            Player(0, 1000, "Player_0"),
            Player(1, 1000, "Player_1_BTN")
        ]
        
        pot_manager.start_new_hand()
        
        # === PREFLOP ===
        pot_manager.post_blinds(players[0], players[1])
        
        # Player 0: calls big blind
        to_call = pot_manager.current_bet - players[0].current_bet
        pot_manager.place_bet(players[0], to_call)
        
        # Player 1: raises 50% pot (5 more)
        pot_manager.place_bet(players[1], 10)
        
        # Player 0: calls the raise
        to_call = pot_manager.current_bet - players[0].current_bet
        pot_manager.place_bet(players[0], to_call)
        
        # End of preflop
        current_pot = pot_manager.pots[0].amount
        print(f"\nEnd of preflop:")
        print(f"  Pot: {current_pot}")
        print(f"  P0: stack={players[0].stack}, total_bet={players[0].total_bet_this_hand}")
        print(f"  P1: stack={players[1].stack}, total_bet={players[1].total_bet_this_hand}")
        
        assert current_pot == 40, f"Preflop pot should be 40, got {current_pot}"
        
        # === FLOP ===
        pot_manager.start_new_betting_round(players)
        
        # Player 0: bets 30
        pot_manager.place_bet(players[0], 30)
        
        # Player 1: calls
        to_call = pot_manager.current_bet - players[1].current_bet
        pot_manager.place_bet(players[1], to_call)
        
        # === TURN ===
        pot_manager.start_new_betting_round(players)
        
        # Player 0: checks
        pot_manager.place_bet(players[0], 0)
        
        # Player 1: bets 100
        pot_manager.place_bet(players[1], 100)
        
        # Player 0: calls
        to_call = pot_manager.current_bet - players[0].current_bet
        pot_manager.place_bet(players[0], to_call)
        
        # === RIVER ===
        pot_manager.start_new_betting_round(players)
        
        # Player 0: goes all-in for remaining 850
        all_in_amount = players[0].stack
        _, action = pot_manager.place_bet(players[0], all_in_amount)
        
        print(f"\nRiver all-in:")
        print(f"  Action: {action}")
        print(f"  P0: stack={players[0].stack}, total_bet={players[0].total_bet_this_hand}")
        print(f"  P1: stack={players[1].stack}, total_bet={players[1].total_bet_this_hand}")
        print(f"  Pot total: {pot_manager.pots[0].amount}")
        
        assert action == "all-in", f"Should be all-in, got {action}"
        assert players[0].stack == 0
        
        # === SHOWDOWN ===
        # Only calculate pots, don't distribute yet
        pots = pot_manager.calculate_side_pots(players)
        
        print(f"\nSide pots calculated:")
        for i, pot in enumerate(pots):
            print(f"  Pot {i}: amount={pot.amount}, eligible={pot.eligible_players}")
        
        total_in_pots = sum(pot.amount for pot in pots)
        assert total_in_pots == pot_manager.pots[0].amount, \
            f"Pots don't add up: {total_in_pots} != {pot_manager.pots[0].amount}"
        print()
        # calculate_side_pots is pure — it produces the layered structure
        # without applying the uncalled-bet refund. The 2-pot structure here
        # reflects P0 over-betting (850 unmatched). distribute_pots will then
        # refund the 850 to P0 and produce a single contested pot.
        assert len(pots) == 2, f"Should have 2 pots (matched + uncontested), got {len(pots)}"
        assert pots[0].amount == 300 and set(pots[0].eligible_players) == {0, 1}
        assert pots[1].amount == 850 and pots[1].eligible_players == [0]

        # Capture chip totals before distribution to verify conservation.
        chips_before = players[0].stack + players[1].stack + pot_manager.pots[0].amount

        # Distribute to a winner (player 0 wins)
        hand_ranks = {0: 100, 1: 200}
        winnings = pot_manager.distribute_pots(players, hand_ranks)

        print(f"\nWinnings:")
        print(f"  P0: {winnings[0]} (matched-pot winnings)")
        print(f"  P1: {winnings[1]}")
        print(f"  P0 stack after refund: {players[0].stack}")

        # PROD-3 fix: P0's 850-chip over-bet is refunded directly to P0's stack
        # before distribute_pots calculates winnings. So winnings shows only the
        # matched 300-chip pot; the 850 lives in P0's stack as a refund.
        assert winnings[0] == 300, "Player 0 wins the contested 300 pot"
        assert winnings[1] == 0, "Player 1 wins nothing"
        # P0's stack started at 0 (all-in) and got the 850 refund applied
        # during distribute_pots.
        assert players[0].stack == 850, f"P0 stack should be 850 after refund, got {players[0].stack}"

        # Chip conservation: refund + winnings + remaining stacks == chips_before
        total_to_p0 = players[0].stack + winnings[0]  # 850 refund + 300 winnings
        total_to_p1 = players[1].stack + winnings[1]
        assert total_to_p0 + total_to_p1 == chips_before, \
            f"Chip conservation violated: {total_to_p0 + total_to_p1} != {chips_before}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])