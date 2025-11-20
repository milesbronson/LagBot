"""
IN-DEPTH TEST: Current_Bet Not Resetting Between Rounds

ISSUE FROM HAND #4:
- Preflop: P0 (BTN/BB) bets $10, P1 (SB) calls
- Flop: P0 current_bet shows $0 but pot_manager.current_bet shows $10
- Result: P0 can't raise (thinks they already bet)

ROOT CAUSE: current_bet not being reset for new betting round

This test encapsulates the exact bug step-by-step
"""

import pytest
from src.poker_env.game_state import GameState, BettingRound


class TestCurrentBetResetBug:
    """Test that current_bet resets properly between rounds"""
    
    @pytest.fixture
    def game(self):
        """Create heads-up game (2 players for this scenario)"""
        return GameState(
            num_players=2,
            starting_stack=2000,
            small_blind=5,
            big_blind=10
        )
    
    def test_current_bet_reset_preflop_to_flop(self, game):
        """
        STEP-BY-STEP reproduction of Hand #4 bug
        
        Preflop:
        - P0 (BTN): posts $10 big blind
        - P1 (SB): posts $5 small blind
        - P1 acts first (in heads-up)
        - P1 calls (matches BB)
        - P0 checks
        
        Flop:
        - current_bet MUST reset to $0
        - P0 should be able to raise
        """
        game.start_new_hand()
        
        print("\n" + "="*80)
        print("HEADS-UP: BB and Raising Issue")
        print("="*80)
        
        p0 = game.players[0]
        p1 = game.players[1]
        
        # === PREFLOP ===
        print("\n" + "="*80)
        print("PREFLOP")
        print("="*80)
        
        print(f"\nAfter blinds:")
        print(f"  P0 (BTN): stack={p0.stack}, current_bet={p0.current_bet}")
        print(f"  P1 (SB): stack={p1.stack}, current_bet={p1.current_bet}")
        print(f"  pot_manager.current_bet = {game.pot_manager.current_bet}")
        print(f"  Current player: {game.current_player_idx}")
        
        assert p0.current_bet == 10, "P0 should have BB of 10"
        assert p1.current_bet == 5, "P1 should have SB of 5"
        assert game.pot_manager.current_bet == 10, "current_bet should be 10 (BB)"
        
        # In heads-up, SB acts first
        assert game.current_player_idx == 1, "P1 (SB) should act first heads-up"
        
        # Step 1: P1 calls the BB
        print(f"\nStep 1: P1 (SB) calls BB")
        to_call = game.pot_manager.current_bet - p1.current_bet
        print(f"  to_call: {to_call}")
        game.execute_action(1)  # Call
        
        print(f"  After P1 call:")
        print(f"    P1: current_bet={p1.current_bet}")
        print(f"    pot_manager.current_bet = {game.pot_manager.current_bet}")
        assert p1.current_bet == 10, "P1 should match BB"
        
        # Step 2: P0 checks
        print(f"\nStep 2: P0 (BB) checks")
        game.execute_action(1)  # Check
        
        print(f"  After P0 check:")
        print(f"    P0: current_bet={p0.current_bet}")
        print(f"    Betting round complete: {game.is_betting_round_complete()}")
        
        # Verify preflop is complete
        assert game.is_betting_round_complete(), "Betting round should be complete"
        assert p0.total_bet_this_hand == 10
        assert p1.total_bet_this_hand == 10
        
        print(f"\n✓ PREFLOP COMPLETE")
        
        # === MOVE TO FLOP ===
        print("\n" + "="*80)
        print("ADVANCE TO FLOP")
        print("="*80)
        
        game.advance_betting_round()
        
        print(f"\nAfter advance_betting_round():")
        print(f"  Betting round: {game.betting_round.name}")
        print(f"  Community cards: {game.community_cards}")
        
        # === FLOP ===
        print("\n" + "="*80)
        print("FLOP - CRITICAL CHECK")
        print("="*80)
        
        print(f"\nFlop state MUST have reset current_bet:")
        print(f"  P0: stack={p0.stack}, current_bet={p0.current_bet}")
        print(f"  P1: stack={p1.stack}, current_bet={p1.current_bet}")
        print(f"  pot_manager.current_bet = {game.pot_manager.current_bet}")
        print(f"  Current player: {game.current_player_idx}")
        
        # CRITICAL ASSERTIONS
        print(f"\n🔴 CRITICAL CHECK 🔴")
        
        # Both players' current_bet must be 0 on new round
        assert p0.current_bet == 0, \
            f"❌ BUG: P0.current_bet should be 0 (new round), got {p0.current_bet}"
        assert p1.current_bet == 0, \
            f"❌ BUG: P1.current_bet should be 0 (new round), got {p1.current_bet}"
        
        # pot_manager.current_bet must be 0
        assert game.pot_manager.current_bet == 0, \
            f"❌ BUG: pot_manager.current_bet should be 0, got {game.pot_manager.current_bet}"
        
        print(f"✓ All current_bets reset to 0")
        
        # P0 acts first on flop (in heads-up, BTN acts first post-flop)
        assert game.current_player_idx == 0, "P0 should act first on flop"
        
        # Step 3: P0 checks
        print(f"\nStep 3: P0 checks on flop")
        game.execute_action(1)  # Check
        
        print(f"  After P0 check:")
        print(f"    P0: current_bet={p0.current_bet}")
        assert p0.current_bet == 0, "P0 check shouldn't change current_bet"
        
        # Step 4: P1 raises
        print(f"\nStep 4: P1 raises to 20")
        print(f"  Before P1 action:")
        print(f"    pot_manager.current_bet = {game.pot_manager.current_bet}")
        print(f"    min_raise = {game.pot_manager.min_raise}")
        
        game.execute_action(2, raise_amount=20)  # Raise 20
        
        print(f"  After P1 raise:")
        print(f"    P1: current_bet={p1.current_bet}")
        print(f"    pot_manager.current_bet = {game.pot_manager.current_bet}")
        
        assert p1.current_bet == 20
        assert game.pot_manager.current_bet == 20
        
        # CRITICAL: P0 should now be able to respond
        print(f"\nStep 5: P0 should be able to raise")
        print(f"  Current player: {game.current_player_idx}")
        print(f"  Betting round complete: {game.is_betting_round_complete()}")
        
        assert game.current_player_idx == 0, "P0 should be next to act"
        assert not game.is_betting_round_complete(), "Round not complete, P0 hasn't matched"
        
        # P0 can now raise
        print(f"\n  P0 raises to 50")
        game.execute_action(2, raise_amount=30)  # Raise 30 more
        
        print(f"  After P0 raise:")
        print(f"    P0: current_bet={p0.current_bet}")
        print(f"    pot_manager.current_bet = {game.pot_manager.current_bet}")
        
        assert p0.current_bet == 50
        assert game.pot_manager.current_bet == 50
        
        print(f"\n✓ FLOP COMPLETE - BB WAS ABLE TO RAISE")
    
    def test_multiple_rounds_current_bet_reset(self, game):
        """
        Test that current_bet resets correctly through multiple rounds
        (Preflop → Flop → Turn → River)
        """
        game.start_new_hand()
        
        print("\n" + "="*80)
        print("MULTI-ROUND RESET TEST")
        print("="*80)
        
        p0, p1 = game.players[0], game.players[1]
        
        for round_num in range(4):
            round_names = ["PREFLOP", "FLOP", "TURN", "RIVER"]
            
            print(f"\n{'='*80}")
            print(f"{round_names[round_num]}")
            print(f"{'='*80}")
            
            print(f"\nAt start of {round_names[round_num]}:")
            print(f"  P0: current_bet={p0.current_bet}")
            print(f"  P1: current_bet={p1.current_bet}")
            print(f"  pot_manager.current_bet = {game.pot_manager.current_bet}")
            
            # Except preflop, current_bet should be 0 at start
            if round_num > 0:
                assert p0.current_bet == 0, f"Round {round_num}: P0 current_bet should be 0"
                assert p1.current_bet == 0, f"Round {round_num}: P1 current_bet should be 0"
                assert game.pot_manager.current_bet == 0, f"Round {round_num}: pot_manager.current_bet should be 0"
            
            # Simulate betting for this round
            if round_num == 0:
                # Preflop: P1 calls, P0 checks
                game.execute_action(1)  # P1 call
                game.execute_action(1)  # P0 check
            else:
                # Other rounds: P0 checks, P1 bets, P0 calls
                game.execute_action(1)  # P0 check
                game.execute_action(2, raise_amount=10)  # P1 bet
                game.execute_action(1)  # P0 call
            
            # Verify round is complete
            assert game.is_betting_round_complete(), f"Round {round_num} should be complete"
            
            # Advance if not at river
            if round_num < 3:
                game.advance_betting_round()
                print(f"\n✓ Advanced to {round_names[round_num + 1]}")
        
        print(f"\n✓ ALL ROUNDS: current_bet reset correctly")
    
    def test_bb_can_raise_after_preflop(self, game):
        """
        CRITICAL: Reproduce exact Hand #4 issue
        
        Verify big blind can raise on flop after calling/checking preflop
        """
        game.start_new_hand()
        
        print("\n" + "="*80)
        print("CRITICAL: BB CAN RAISE AFTER PREFLOP")
        print("="*80)
        
        p0, p1 = game.players[0], game.players[1]
        
        # Preflop: standard action
        print(f"\nPreflop: P1 calls, P0 checks")
        game.execute_action(1)  # P1 call
        game.execute_action(1)  # P0 check
        
        game.advance_betting_round()
        
        # Flop: P0 (BB) should be able to raise
        print(f"\nFlop: P0 checks, P1 bets, P0 RAISES")
        
        # P0 checks
        game.execute_action(1)  # Check
        print(f"  P0 checked: current_bet={p0.current_bet}, pot_mgr.current_bet={game.pot_manager.current_bet}")
        
        # P1 bets
        game.execute_action(2, raise_amount=20)
        print(f"  P1 bet 20: current_bet={p1.current_bet}, pot_mgr.current_bet={game.pot_manager.current_bet}")
        
        # P0 MUST be able to raise now
        print(f"\n  P0's turn - checking if can raise:")
        print(f"    P0.current_bet: {p0.current_bet}")
        print(f"    to_call: {game.pot_manager.current_bet - p0.current_bet}")
        print(f"    P0 can act: {p0.can_act()}")
        
        # Check P0 can act
        assert p0.can_act(), "P0 should be able to act (not all-in, not folded)"
        
        # Check they're not all-in
        assert not p0.is_all_in, "P0 should not be all-in"
        
        # Check betting round not complete (P0 hasn't matched P1's bet yet)
        assert not game.is_betting_round_complete(), \
            "Betting round should NOT be complete - P0 hasn't matched P1's bet"
        
        # Execute the raise
        print(f"\n  P0 raises to 60")
        game.execute_action(2, raise_amount=40)
        print(f"  After raise: current_bet={p0.current_bet}, pot_mgr.current_bet={game.pot_manager.current_bet}")
        
        assert p0.current_bet == 60
        assert game.pot_manager.current_bet == 60
        
        print(f"\n✓ BB WAS ABLE TO RAISE ON FLOP")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])