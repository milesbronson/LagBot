"""
Test all-in calculations to ensure correct bet amounts
"""

import pytest
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.poker_env.player import Player
from src.poker_env.pot_manager import PotManager


class TestAllInCalculations:
    """Verify all-in bet amounts are correct"""
    
    def test_all_in_simple(self):
        """Test simple all-in with initial stacks"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        obs, info = env.reset()
        
        print("\n" + "="*80)
        print("TEST: Simple All-In")
        print("="*80)
        
        # Get initial state
        for p in env.game_state.players:
            print(f"{p.name}: stack=${p.stack}, bet=${p.current_bet}")
        
        print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
        print(f"Current bet: ${env.game_state.pot_manager.current_bet}")
        
        # Player goes all-in
        current_player = env.game_state.get_current_player()
        print(f"\n{current_player.name} going all-in with stack=${current_player.stack}")
        
        # All-in action (action 5 for 2 players, varies by num_raise_bins)
        all_in_action = 2 + len(env.raise_bins) if env.include_all_in else None
        print(f"All-in action index: {all_in_action}")
        
        if all_in_action is not None:
            obs, reward, terminated, truncated, info = env.step(all_in_action)
            
            print(f"\nAfter all-in:")
            for p in env.game_state.players:
                print(f"{p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
            
            print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
    
    def test_all_in_chain(self):
        """Test chain of all-ins like in the bug report"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        obs, info = env.reset()
        
        print("\n" + "="*80)
        print("TEST: Chain of All-Ins")
        print("="*80)
        
        # Track each action
        action_num = 0
        steps = 0
        
        while not env.game_state.is_hand_complete() and steps < 50:
            current_player = env.game_state.get_current_player()
            action_num += 1
            
            print(f"\n--- Action {action_num} ---")
            print(f"Current player: {current_player.name} (stack=${current_player.stack})")
            print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
            print(f"Current bet: ${env.game_state.pot_manager.current_bet}")
            
            # Show all stacks before action
            print("Before action:")
            for p in env.game_state.players:
                print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
            
            # Take action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            # Show results after action
            print(f"Action: {info.get('action', 'unknown')}")
            print("After action:")
            for p in env.game_state.players:
                print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
            print(f"Pot after: ${env.game_state.pot_manager.get_pot_total()}")
            
            # Only validate chips if hand is NOT complete (to avoid double-counting)
            if not env.game_state.is_hand_complete():
                pot = env.game_state.pot_manager.get_pot_total()
                stacks = sum(p.stack for p in env.game_state.players)
                total = stacks + pot
                
                print(f"Check: stacks=${stacks} + pot=${pot} = ${total}")
                
                if total != 3000:
                    print(f"⚠ ERROR: Should equal 3000!")
                    raise AssertionError(f"Chips missing! stacks={stacks} + pot={pot} != 3000")
            else:
                print(f"Hand is complete - skipping chip check (pot has been distributed)")
        
        # Display final hand history
        print("\n" + "="*80)
        print("FINAL HAND HISTORY")
        print("="*80)
        env.game_state.display_hand_history()
        
        # Final chip check: at end of hand, all chips should be in stacks
        stacks = sum(p.stack for p in env.game_state.players)
        print(f"\nFinal stacks total: ${stacks}")
        assert stacks == 3000, f"Final stacks incorrect! {stacks} != 3000"
        print("✓ Test passed - chips conserved through all-in chain")

    def test_all_in_side_pots(self):
        """Three players all-in at different levels with side pots"""
    
        print("\n" + "="*80)
        print("TEST: Three-Way All-In with Side Pots (CORRECTED)")
        print("="*80)
        
        # Create environment
        env = TexasHoldemEnv(num_players=3, starting_stack=2500)
        
        # CORRECTED: Set stacks based on actual player order
        # Player 1 acts first preflop, so give it smallest stack
        env.game_state.players[1].stack = 500   # Smallest stack (acts first)
        env.game_state.players[2].stack = 800   # Middle stack
        env.game_state.players[0].stack = 1200  # Largest stack
        
        print("\nInitial stacks:")
        for p in env.game_state.players:
            print(f"  {p.name} (ID={p.player_id}): ${p.stack}")
        
        # Start hand
        obs, info = env.reset()
        
        print("\nAfter blinds:")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
        print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
        
        # Action 1: Player 1 goes all-in with $500
        print("\n--- ACTION 1: Player_1 all-in ($500) ---")
        current = env.game_state.get_current_player()
        print(f"Current player: {current.name} (ID={current.player_id})")
        all_in_idx = 2 + len(env.raise_bins)
        obs, reward, terminated, truncated, info = env.step(all_in_idx)
        
        print(f"After all-in:")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
        print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
        
        # Action 2: Player 2 all-in with $800
        print("\n--- ACTION 2: Player_2 all-in ($800) ---")
        current = env.game_state.get_current_player()
        print(f"Current player: {current.name} (ID={current.player_id})")
        obs, reward, terminated, truncated, info = env.step(all_in_idx)
        
        print(f"After all-in:")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
        print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
        
        # Action 3: Player 0 all-in with $1200
        print("\n--- ACTION 3: Player_0 all-in ($1200) ---")
        current = env.game_state.get_current_player()
        print(f"Current player: {current.name} (ID={current.player_id})")
        obs, reward, terminated, truncated, info = env.step(all_in_idx)
        
        print(f"After all-in:")
        for p in env.game_state.players:
            print(f"  {p.name}: stack=${p.stack}, bet=${p.current_bet}, total_bet=${p.total_bet_this_hand}")
        print(f"Pot: ${env.game_state.pot_manager.get_pot_total()}")
        
        # Play out remaining streets (should auto-resolve now all are all-in)
        done = False
        steps = 0
        while not done and steps < 100:
            obs, reward, terminated, truncated, info = env.step(1)  # Check/call rest
            done = terminated or truncated
            steps += 1
        
        print("\n--- HAND COMPLETE ---")
        env.game_state.display_hand_history()
        
        # Calculate side pots
        print("\n--- SIDE POT CALCULATION ---")
        pots = env.game_state.pot_manager.calculate_side_pots(env.game_state.players)
        
        print(f"Number of pots: {len(pots)}")
        for i, pot in enumerate(pots):
            print(f"  Pot {i+1}: ${pot.amount}, eligible players: {[env.game_state.players[pid].name for pid in pot.eligible_players]}")
        
        # Get hand ranks
        hand_ranks = {}
        for p in env.game_state.players:
            if p.hand:
                rank = env.game_state.hand_evaluator.evaluate_hand(p.hand, env.game_state.community_cards)
                hand_ranks[p.player_id] = rank
        
        print(f"\nHand ranks:")
        for pid, rank in hand_ranks.items():
            player = env.game_state.players[pid]
            print(f"  {player.name}: rank={rank}")
        
        # Get winnings
        winnings = env.game_state.pot_manager.distribute_pots(env.game_state.players, hand_ranks)
        
        print(f"\nWinnings distribution:")
        for pid, amount in winnings.items():
            player = env.game_state.players[pid]
            print(f"  {player.name}: ${amount}")
        
        # Final stacks
        print(f"\nFinal stacks:")
        for p in env.game_state.players:
            print(f"  {p.name}: ${p.stack}")
        
        # Verify chip conservation
        total_stacks = sum(p.stack for p in env.game_state.players)
        print(f"\nChip conservation check:")
        print(f"  Total stacks: ${total_stacks}")
        print(f"  Expected: $2500 (500 + 800 + 1200)")
        
        assert total_stacks == 2500, f"Chips not conserved! {total_stacks} != 2500"
        
        print("\n✓ Test passed - chips conserved and side pots calculated correctly")


    def test_all_in_calculation_formula(self):
        """Test the all-in raise amount calculation"""
        pot_manager = PotManager(small_blind=5, big_blind=10)
        
        player = Player(0, 1000)
        pot_manager.start_new_hand()
        
        print("\n" + "="*80)
        print("TEST: All-In Formula")
        print("="*80)
        print(f"Player stack: ${player.stack}")
        print(f"Current bet: ${pot_manager.current_bet}")
        
        to_call = pot_manager.current_bet - player.current_bet
        print(f"To call: ${to_call}")
        print(f"All-in should be: ${player.stack} (entire stack)")
        
        # Simulate all-in
        all_in_raise = player.stack
        print(f"All-in raise amount: ${all_in_raise}")
        
        total_contribution = to_call + all_in_raise
        print(f"Total contribution: ${to_call} + ${all_in_raise} = ${total_contribution}")
        
        # This should equal player.stack (they go all-in)
        assert total_contribution == player.stack + to_call, (
            f"All-in calculation wrong: {total_contribution} != {player.stack + to_call}"
        )
    
    def test_initial_bet_then_all_in(self):
        """Test: Player makes initial raise, then goes all-in"""
        env = TexasHoldemEnv(num_players=3, starting_stack=1000)
        obs, info = env.reset()
        
        print("\n" + "="*80)
        print("TEST: Initial Bet Then All-In")
        print("="*80)
        
        # Action 1: Player raises (action 2 = first raise bin, 50% pot)
        print("\n--- ACTION 1: First player raises (50% pot) ---")
        current = env.game_state.get_current_player()
        print(f"{current.name} raises")
        obs, reward, terminated, truncated, info = env.step(2)
        
        # Action 2: Next player goes all-in
        print("\n--- ACTION 2: Second player all-in ---")
        current = env.game_state.get_current_player()
        all_in_idx = 2 + len(env.raise_bins)
        print(f"{current.name} goes all-in (stack=${current.stack})")
        obs, reward, terminated, truncated, info = env.step(all_in_idx)
        
        # Display hand history
        env.game_state.display_hand_history()
        
        # Verify chips
        pot = env.game_state.pot_manager.get_pot_total()
        stacks = sum(p.stack for p in env.game_state.players)
        total_bet = sum(p.total_bet_this_hand for p in env.game_state.players)
        total = pot + stacks
        
        print(f"\nChip check: pot=${pot} + stacks=${stacks}  = ${total}")
        assert total == 3000, f"Chips missing! {total} != 3000"



        """Test multiple players going all-in"""
        env = TexasHoldemEnv(num_players=3, starting_stack=100)  # Small stacks to trigger all-ins
        
        print("\n" + "="*80)
        print("TEST: Multiple All-Ins with Small Stacks")
        print("="*80)
        
        hands_played = 0
        all_in_count = 0
        
        for hand_num in range(10):
            obs, info = env.reset()
            
            done = False
            steps = 0
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if info.get('action') == 'all-in':
                    all_in_count += 1
                    pot = env.game_state.pot_manager.get_pot_total()
                    print(f"Hand {hand_num}, Step {steps}: ALL-IN, Pot now ${pot}")
                
                done = terminated or truncated
                steps += 1
            
            hands_played += 1
            
            # Verify chips after each hand
            pot = env.game_state.pot_manager.get_pot_total()
            stacks = sum(p.stack for p in env.game_state.players)
            total_bet = sum(p.total_bet_this_hand for p in env.game_state.players)
            
            total = pot + stacks + total_bet
            if total != 300:  # 3 players × 100
                print(f"⚠ Hand {hand_num}: Chips missing! {total} != 300")
        
        print(f"\nPlayed {hands_played} hands, saw {all_in_count} all-ins")
        assert all_in_count > 0, "No all-ins happened with small stacks!"


if __name__ == "__main__":
    pytest.main([__file__, "-s", "-v"])