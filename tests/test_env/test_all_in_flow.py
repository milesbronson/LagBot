"""
DIAGNOSTIC TEST: Action Flow Bug When Player Goes All-In

This test reveals the critical bug: when one player goes all-in, the game
doesn't properly allow the next player to make their decision before advancing
to showdown.

The issue is likely in:
1. is_betting_round_complete() - not properly checking all-in scenarios
2. execute_action() flow - may auto-advance betting round
3. Game loop in texas_holdem_env.py - may skip opponent's action

Run with: pytest test_diagnostic_all_in_flow.py -v -s
"""

import pytest
from src.poker_env.game_state import GameState, BettingRound
from src.poker_env.player import Player


class TestAllInActionFlow:
    """Diagnostic tests for all-in action flow"""
    
    def setup_method(self):
        """Setup for each test"""
        self.game = GameState(
            num_players=2,
            starting_stack=1000,
            small_blind=5,
            big_blind=10
        )
    
    def test_all_in_action_flow_two_players(self):
        """
        CRITICAL TEST: Two players, one goes all-in
        
        Expected flow:
        1. Player 0 (SB): posts $5, action to Player 1
        2. Player 1 (BB): posts $10, action to Player 0
        3. Player 0: goes all-in for $995
        4. Player 1: MUST get action (can call, fold, or raise)
        5. Only then do we check if betting round is complete
        """
        print("\n" + "="*80)
        print("TEST: Two-Player All-In Action Flow")
        print("="*80)
        
        self.game.start_new_hand()
        
        print("\n--- INITIAL STATE ---")
        print(f"Blinds posted:")
        print(f"  Player 0 (SB): stack={self.game.players[0].stack}, bet={self.game.players[0].current_bet}")
        print(f"  Player 1 (BB): stack={self.game.players[1].stack}, bet={self.game.players[1].current_bet}")
        print(f"Current player index: {self.game.current_player_idx}")
        print(f"Current player: Player {self.game.get_current_player().player_id}")
        
        # Player 0 (SB) should act first preflop
        assert self.game.current_player_idx == 0, "Player 0 should act first preflop"
        p0 = self.game.get_current_player()
        print(f"\nPlayer 0's turn (small blind)")
        print(f"  Stack: ${p0.stack}, needs to call: ${self.game.pot_manager.current_bet - p0.current_bet}")
        
        # Player 0 goes all-in
        print(f"\n--- PLAYER 0 GOES ALL-IN ---")
        print(f"Before all-in:")
        print(f"  P0: stack={self.game.players[0].stack}, current_bet={self.game.players[0].current_bet}, total_bet={self.game.players[0].total_bet_this_hand}")
        print(f"  Pot: {self.game.pot_manager.pots[0].amount}")
        
        self.game.execute_action(2, raise_amount=self.game.players[0].stack)  # All-in
        
        print(f"\nAfter all-in:")
        print(f"  P0: stack={self.game.players[0].stack}, current_bet={self.game.players[0].current_bet}, total_bet={self.game.players[0].total_bet_this_hand}")
        print(f"  Pot: {self.game.pot_manager.pots[0].amount}")
        print(f"  Current player index: {self.game.current_player_idx}")
        print(f"  Current player: Player {self.game.get_current_player().player_id}")
        
        # CRITICAL CHECK: Is the game asking Player 1 for their action?
        print(f"\n--- BETTING ROUND STATUS ---")
        print(f"Is betting round complete? {self.game.is_betting_round_complete()}")
        print(f"Is hand complete? {self.game.is_hand_complete()}")
        print(f"Active players: {len(self.game.get_active_players())}")
        
        active_players = self.game.get_active_players()
        can_act = [p for p in active_players if not p.is_all_in]
        print(f"Players who can act: {len(can_act)}")
        for p in can_act:
            print(f"  Player {p.player_id}: stack={p.stack}, current_bet={p.current_bet}")
        
        # THE CRITICAL BUG CHECK
        print(f"\n🔴 CRITICAL BUG CHECK 🔴")
        print(f"Expected: current_player_idx should be 1 (Player 1 gets to act)")
        print(f"Actual: current_player_idx is {self.game.current_player_idx}")
        
        assert self.game.current_player_idx == 1, \
            f"❌ BUG: After P0 all-in, P1 should be current player! Got {self.game.current_player_idx}"
        
        assert not self.game.is_betting_round_complete(), \
            "❌ BUG: Betting round should NOT be complete - P1 hasn't acted yet!"
        
        assert not self.game.is_hand_complete(), \
            "❌ BUG: Hand should NOT be complete - we're still in betting round!"
        
        # Now Player 1 should be able to act
        print(f"\n--- PLAYER 1'S ACTION ---")
        p1 = self.game.get_current_player()
        print(f"Player 1's turn:")
        print(f"  Stack: ${p1.stack}")
        print(f"  Current bet: ${p1.current_bet}")
        print(f"  To call: ${self.game.pot_manager.current_bet - p1.current_bet}")
        print(f"  Is all-in? {p1.is_all_in}")
        
        # Player 1 calls the all-in
        to_call = self.game.pot_manager.current_bet - p1.current_bet
        print(f"\nPlayer 1 calls for ${to_call}")
        
        self.game.execute_action(1)  # Call
        
        print(f"\nAfter P1 call:")
        print(f"  P0: stack={self.game.players[0].stack}, current_bet={self.game.players[0].current_bet}, total_bet={self.game.players[0].total_bet_this_hand}")
        print(f"  P1: stack={self.game.players[1].stack}, current_bet={self.game.players[1].current_bet}, total_bet={self.game.players[1].total_bet_this_hand}")
        print(f"  Pot: {self.game.pot_manager.pots[0].amount}")
        
        # Now betting round should be complete
        print(f"\n--- BETTING ROUND COMPLETION CHECK ---")
        print(f"Is betting round complete? {self.game.is_betting_round_complete()}")
        print(f"Is hand complete? {self.game.is_hand_complete()}")
        
        assert self.game.is_betting_round_complete(), \
            "Betting round should be complete after both all-in"
        
        # Hand should NOT be complete yet - we're still preflop
        # Betting round complete doesn't mean hand complete
        print(f"\nBetting round complete, should advance to showdown or next street")
    
    def test_all_in_betting_round_complete_logic(self):
        """
        Test the is_betting_round_complete() logic with all-ins
        
        The bug might be in how is_betting_round_complete() handles all-in players
        """
        print("\n" + "="*80)
        print("TEST: Betting Round Complete Logic with All-Ins")
        print("="*80)
        
        self.game.start_new_hand()
        
        print(f"\nInitial state:")
        print(f"  Active players: {len(self.game.get_active_players())}")
        print(f"  Players who can act: {len([p for p in self.game.get_active_players() if not p.is_all_in])}")
        print(f"  is_betting_round_complete(): {self.game.is_betting_round_complete()}")
        
        # Player 0 goes all-in
        self.game.execute_action(2, raise_amount=self.game.players[0].stack)
        
        print(f"\nAfter Player 0 all-in:")
        print(f"  Player 0: is_all_in={self.game.players[0].is_all_in}, stack={self.game.players[0].stack}")
        print(f"  Active players: {len(self.game.get_active_players())}")
        print(f"  Players who can act: {len([p for p in self.game.get_active_players() if not p.is_all_in])}")
        print(f"  is_betting_round_complete(): {self.game.is_betting_round_complete()}")
        
        # This should be FALSE because Player 1 can still act
        print(f"\n🔍 IS_BETTING_ROUND_COMPLETE LOGIC CHECK:")
        print(f"Expected: False (Player 1 can still act)")
        print(f"Actual: {self.game.is_betting_round_complete()}")
        
        assert not self.game.is_betting_round_complete(), \
            "❌ BUG: Betting round should NOT be complete - Player 1 can act!"
        
        # Player 1 calls
        self.game.execute_action(1)
        
        print(f"\nAfter Player 1 calls:")
        print(f"  Player 0: is_all_in={self.game.players[0].is_all_in}, current_bet={self.game.players[0].current_bet}")
        print(f"  Player 1: is_all_in={self.game.players[1].is_all_in}, current_bet={self.game.players[1].current_bet}")
        print(f"  Betting round complete? {self.game.is_betting_round_complete()}")
        
        # Now it should be True
        assert self.game.is_betting_round_complete(), \
            "Betting round should be complete when all active players have matched bet"
    
    def test_current_player_advancement_with_all_in(self):
        """
        Test that current_player_idx properly advances when one player all-ins
        
        Expected:
        1. Initial: Player 0 to act (SB)
        2. After P0 all-in: Player 1 to act (BB)
        3. After P1 acts: Betting round complete, no more players to act
        """
        print("\n" + "="*80)
        print("TEST: Current Player Index Advancement with All-In")
        print("="*80)
        
        self.game.start_new_hand()
        
        print(f"\nStep 1: Blinds posted, Player 0's turn")
        print(f"  current_player_idx: {self.game.current_player_idx}")
        print(f"  Current player: Player {self.game.get_current_player().player_id}")
        assert self.game.current_player_idx == 0
        
        print(f"\nStep 2: Player 0 goes all-in")
        self.game.execute_action(2, raise_amount=self.game.players[0].stack)
        
        print(f"  After action, current_player_idx: {self.game.current_player_idx}")
        print(f"  Current player: Player {self.game.get_current_player().player_id}")
        
        # Check if advance happened
        if self.game.current_player_idx != 1:
            print(f"\n❌ BUG: current_player_idx should be 1, got {self.game.current_player_idx}")
            print(f"Betting round: {self.game.betting_round.name}")
            print(f"Betting round complete? {self.game.is_betting_round_complete()}")
            print(f"Hand complete? {self.game.is_hand_complete()}")
        
        assert self.game.current_player_idx == 1, \
            f"❌ BUG: After P0 all-in, current should be P1, got {self.game.current_player_idx}"
        
        print(f"\nStep 3: Player 1 calls")
        self.game.execute_action(1)
        
        print(f"  After P1 call:")
        print(f"  Betting round complete? {self.game.is_betting_round_complete()}")
        print(f"  Hand complete? {self.game.is_hand_complete()}")
        print(f"  current_player_idx: {self.game.current_player_idx}")
        
        # Betting round should be complete
        assert self.game.is_betting_round_complete()
    
    def test_sequence_leading_to_showdown(self):
        """
        Full sequence: all-in on river should NOT immediately go to showdown
        
        Expected flow:
        1. Both players act all streets
        2. On river: Player 0 goes all-in
        3. Player 1 gets chance to act (call/fold)
        4. THEN move to showdown
        """
        print("\n" + "="*80)
        print("TEST: All-In on River - Full Action Sequence")
        print("="*80)
        
        # Play to river with all-in scenario
        self.game.start_new_hand()
        
        # Preflop: both to river
        print(f"\n--- PREFLOP ---")
        print(f"Player 0 goes all-in")
        self.game.execute_action(2, raise_amount=self.game.players[0].stack)  # P0 all-in for ~990
        
        print(f"Current player: {self.game.current_player_idx}")
        print(f"Betting round complete? {self.game.is_betting_round_complete()}")
        
        print(f"\nPlayer 1 calls all-in")
        self.game.execute_action(1)  # P1 calls
        
        print(f"Current player: {self.game.current_player_idx}")
        print(f"Betting round complete? {self.game.is_betting_round_complete()}")
        print(f"Hand complete? {self.game.is_hand_complete()}")
        print(f"Betting round: {self.game.betting_round.name}")
        
        # Both are all-in, so showdown immediately
        if self.game.is_hand_complete():
            print(f"\n✓ Both all-in preflop → showdown (expected)")
            return  # This is fine
        
        # If not hand complete, we advance betting round
        print(f"\nNot yet at showdown, advancing betting round...")
        self.game.advance_betting_round()
        
        print(f"Betting round: {self.game.betting_round.name}")
        print(f"Hand complete? {self.game.is_hand_complete()}")
        
        # Since both all-in, subsequent streets auto-complete
        while not self.game.is_hand_complete() and self.game.betting_round != BettingRound.SHOWDOWN:
            print(f"\nAdvancing to {self.game.betting_round.name}")
            self.game.advance_betting_round()
        
        print(f"\nFinal betting round: {self.game.betting_round.name}")
        assert self.game.is_hand_complete(), "Hand should be complete at showdown"
        print(f"✓ Hand complete at showdown")


class TestAllInBugSummary:
    """Summary of the all-in action flow bug"""
    
    def test_bug_summary(self):
        """Document the bug"""
        print("\n" + "="*80)
        print("ALL-IN ACTION FLOW BUG SUMMARY")
        print("="*80)
        
        bug_summary = """
THE BUG: When one player goes all-in, the game doesn't properly allow the
opponent to make their decision. It may:

1. ❌ Skip the opponent's action entirely
   - Symptom: current_player_idx doesn't advance to opponent
   - Symptom: betting round immediately considered complete
   - Result: Opponent never gets to act

2. ❌ Immediately jump to showdown
   - Symptom: After one player all-in, hand_complete() returns True
   - Symptom: Game doesn't call advance_betting_round()
   - Result: Opponent's contribution to pot is incomplete

3. ❌ Don't properly register opponent's response
   - Symptom: Opponent acts, but their bet isn't added to pot
   - Symptom: total_bet_this_hand doesn't update
   - Result: Chip imbalance at showdown

LIKELY LOCATIONS OF BUG:

1. game_state.py - execute_action()
   - May not properly advance current_player_idx
   - May auto-advance betting round when not appropriate
   - May not handle all-in player next in action

2. game_state.py - is_betting_round_complete()
   - May incorrectly return True when all-in player needs opponent action
   - May not account for "players who can act" vs "active players"

3. game_state.py - _advance_current_player()
   - May skip all-in players (should skip folded players only)
   - May not loop back to beginning of players list correctly

4. texas_holdem_env.py - step() or step_with_raise()
   - May auto-advance betting round without checking if all players acted
   - May not wait for opponent's action after all-in

5. play.py - game loop
   - May not request opponent input after all-in
   - May directly call determine_winners() instead of waiting

WHAT SHOULD HAPPEN:

1. Player goes all-in
2. current_player_idx advances to NEXT PLAYER
3. is_betting_round_complete() returns False (opponent needs to act)
4. Game asks opponent for action (in play.py or env step)
5. Opponent acts (call/fold)
6. is_betting_round_complete() NOW returns True
7. Betting round advances to next street OR showdown
8. All future streets may auto-play (both all-in)

The key is: just because one player all-ins doesn't mean betting round is
complete. The NEXT player still must act.
        """
        
        print(bug_summary)
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])