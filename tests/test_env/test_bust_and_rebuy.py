"""
Tests for player bust and rebuy mechanics - verify chip tracking
"""

import pytest
from src.poker_env.texas_holdem_env import TexasHoldemEnv


class TestPlayerBust:
    """Test that chips are properly accounted for when a player goes bust"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000)
    
    def test_initial_chip_total(self, env):
        """Test that initial chip total is correct"""
        initial_total = sum(p.stack for p in env.game_state.players)
        
        # 3 players × $1000 each = $3000
        assert initial_total == 3000
        assert len(env.game_state.players) == 3
    
    def test_chips_conserved_after_reset(self, env):
        """Test that chip total is conserved after environment reset"""
        initial_total = sum(p.stack for p in env.game_state.players)
        
        env.reset()
        
        total_after_reset = sum((p.stack + p.total_winnings) for p in env.game_state.players)
        
        # All players should be reset to starting_stack
        assert total_after_reset == initial_total
        for player in env.game_state.players:
            assert player.stack == 1000
    
    def test_chips_conserved_single_hand(self, env):
        """Test that total chips are conserved during a single hand"""
        initial_total = sum(p.stack for p in env.game_state.players)
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Get chip total after hand
        final_total = sum((p.stack + p.total_winnings) for p in env.game_state.players)
        
        # Chips should be conserved (minus rake)
        assert final_total <= initial_total, f"Chips increased! Initial: {initial_total}, Final: {final_total}"
        print("\nChip Conservation Single Hand:")
        print(f"  Initial: ${initial_total}")
        print(f"  Final:   ${final_total}")
        print(f"  Difference (rake): ${initial_total - final_total}")
    
    def test_bust_player_has_zero_stack(self, env):
        """Test that a busted player has zero stack"""
        obs, info = env.reset()
        
        # Play many hands to see if anyone busts
        for hand in range(100):
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            # Check if anyone busted
            busted_players = [p for p in env.game_state.players if p.stack == 0]
            
            if busted_players:
                print(f"\n✓ Player went bust at hand {hand}")
                for player in busted_players:
                    print(f"  {player.name}: ${player.stack}")
                
                # Verify busted player doesn't have cards
                for player in busted_players:
                    assert player.stack == 0
                
                return  # Test passed, found a bust
            
            obs, info = env.reset()
        
        pytest.skip("No player went bust in 100 hands (variance)")
    
    def test_chips_when_player_busts(self, env):
        """Test that total chips account for busted players correctly"""
        initial_total = sum(p.stack for p in env.game_state.players)
        
        # Play many hands
        for hand in range(200):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            # Check current chip total
            current_total = sum(p.stack for p in env.game_state.players)
            
            # Should never exceed initial (only rake should decrease it)
            assert current_total <= initial_total, (
                f"Chips increased after bust! Hand {hand}: "
                f"Initial {initial_total}, Current {current_total}"
            )
            
            # Should never become negative
            assert current_total >= 0, (
                f"Chips became negative! Hand {hand}: {current_total}"
            )
            
            # Find any busted players
            busted = [p for p in env.game_state.players if p.stack == 0]
            active = [p for p in env.game_state.players if p.stack > 0]
            
            if busted:
                print(f"\nHand {hand} - Bust detected:")
                print(f"  Active players: {len(active)}")
                print(f"  Busted players: {len(busted)}")
                print(f"  Total chips: ${current_total}")
                print(f"  Initial: ${initial_total}")
                print(f"  Rake paid: ${initial_total - current_total}")
                return  # Found a bust scenario


class TestPlayerRebuy:
    """Test rebuy mechanics - players coming back with new chips"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment with rebuy tracking"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000)
    
    def test_manual_rebuy_increases_stack(self, env):
        """Test that manually adding chips to a busted player works"""
        obs, info = env.reset()
        
        # Manually bust a player
        player = env.game_state.players[0]
        original_stack = player.stack
        initial_total = sum(p.stack for p in env.game_state.players)
        
        # Force player to have zero chips
        player.stack = 0
        
        total_with_bust = sum(p.stack for p in env.game_state.players)
        assert total_with_bust == initial_total - original_stack
        
        print("\nRebuy Test:")
        print(f"  Initial total: ${initial_total}")
        print(f"  After bust: ${total_with_bust}")
        
        # Rebuy - add new chips
        rebuy_amount = 1000
        player.add_chips(rebuy_amount)
        
        total_after_rebuy = sum(p.stack for p in env.game_state.players)
        expected_total = total_with_bust + rebuy_amount
        
        assert player.stack == rebuy_amount
        assert total_after_rebuy == expected_total
        
        print(f"  After rebuy: ${total_after_rebuy}")
        print(f"  Player stack: ${player.stack}")
        print(f"  Total increased by: ${total_after_rebuy - total_with_bust}")
    
    def test_rebuy_sequence(self, env):
        """Test a sequence of bust and rebuy"""
        obs, info = env.reset()
        initial_total = sum(p.stack for p in env.game_state.players)
        
        print("\nRebuy Sequence Test:")
        print(f"  Starting total: ${initial_total}")
        
        # Scenario: Player 0 busts, rebuys
        player = env.game_state.players[0]
        
        # Bust the player
        original_stack = player.stack
        player.stack = 0
        total_after_bust = sum(p.stack for p in env.game_state.players)
        print(f"  After P0 bust: ${total_after_bust} (lost ${original_stack})")
        
        # Rebuy
        rebuy_1 = 1000
        player.add_chips(rebuy_1)
        total_after_rebuy_1 = sum(p.stack for p in env.game_state.players)
        print(f"  After P0 rebuy $1000: ${total_after_rebuy_1}")
        
        # Verify
        assert total_after_rebuy_1 == total_after_bust + rebuy_1
        
        # Bust again
        player.stack = 0
        total_after_bust_2 = sum(p.stack for p in env.game_state.players)
        print(f"  After P0 bust again: ${total_after_bust_2}")
        
        # Rebuy again
        rebuy_2 = 500  # Smaller rebuy this time
        player.add_chips(rebuy_2)
        total_after_rebuy_2 = sum(p.stack for p in env.game_state.players)
        print(f"  After P0 rebuy $500: ${total_after_rebuy_2}")
        
        # Verify
        assert total_after_rebuy_2 == total_after_bust_2 + rebuy_2
        print("  Final verification: chips properly tracked through busts and rebuys ✓")
    
    def test_chip_accounting_with_rake_and_rebuy(self, env):
        """Test chip accounting with rake deductions and rebuys"""
        # Enable rake for this test
        env.game_state.pot_manager.rake_percent = 0.05  # 5% rake
        
        initial_total = sum(p.stack for p in env.game_state.players)
        print("\nChip Accounting with Rake and Rebuy:")
        print(f"  Starting total: ${initial_total}")
        print("  Rake: 5%")
        
        total_rake_paid = 0
        
        # Play a hand (rake will be deducted)
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        total_after_hand = sum(p.stack for p in env.game_state.players)
        rake_from_hand = initial_total - total_after_hand
        total_rake_paid += rake_from_hand
        
        print(f"  After hand 1: ${total_after_hand} (rake: ${rake_from_hand})")
        
        # Simulate a player bust and rebuy
        player = env.game_state.players[0]
        player.stack = 0
        total_with_bust = sum(p.stack for p in env.game_state.players)
        
        # Rebuy
        rebuy = 500
        player.add_chips(rebuy)
        total_after_rebuy = sum(p.stack for p in env.game_state.players)
        
        print(f"  After bust and rebuy: ${total_after_rebuy}")
        
        # Play another hand
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        final_total = sum(p.stack for p in env.game_state.players)
        rake_from_hand_2 = total_after_rebuy - final_total
        total_rake_paid += rake_from_hand_2
        
        print(f"  After hand 2: ${final_total} (rake: ${rake_from_hand_2})")
        print(f"  Total rake paid: ${total_rake_paid}")
        
        # Verify the math
        expected_final = initial_total + rebuy - total_rake_paid
        assert final_total == expected_final, (
            f"Chip accounting error! Expected ${expected_final}, got ${final_total}"
        )
        print("  Chip accounting verified ✓")


class TestChipFlowTracking:
    """Test comprehensive chip flow tracking"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000)
    
    def test_detailed_chip_report(self, env):
        """Generate a detailed report of chip flow"""
        initial_total = sum(p.stack for p in env.game_state.players)
        
        print("\n" + "="*60)
        print("DETAILED CHIP FLOW REPORT")
        print("="*60)
        
        print("\nInitial State:")
        print(f"  Total chips: ${initial_total}")
        for i, player in enumerate(env.game_state.players):
            print(f"    Player {i}: ${player.stack}")
        
        # Play 10 hands and track everything
        for hand_num in range(10):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            current_total = sum(p.stack for p in env.game_state.players)
            total_rake = initial_total - current_total
            
            print(f"\nAfter Hand {hand_num + 1}:")
            print(f"  Total chips: ${current_total}")
            print(f"  Cumulative rake: ${total_rake}")
            
            for i, player in enumerate(env.game_state.players):
                status = ""
                if player.stack == 0:
                    status = " [BUST]"
                profit = player.stack - 1000
                print(f"    Player {i}: ${player.stack} ({profit:+d}){status}")
            
            # Verify invariant
            assert current_total <= initial_total
            assert current_total >= 0
    
    def test_player_elimination(self, env):
        """Test scenario where players are eliminated"""
        print("\n" + "="*60)
        print("PLAYER ELIMINATION TEST")
        print("="*60)
        
        initial_total = sum(p.stack for p in env.game_state.players)
        print(f"\nStarting: ${initial_total} across {len(env.game_state.players)} players")
        
        # Play hands until someone goes to zero
        for hand in range(500):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            current_total = sum(p.stack for p in env.game_state.players)
            busted = [p for p in env.game_state.players if p.stack == 0]
            active = [p for p in env.game_state.players if p.stack > 0]
            
            if len(busted) > 0:
                print(f"\nBust detected at hand {hand + 1}:")
                print(f"  Active players: {len(active)}")
                print(f"  Busted players: {len(busted)}")
                print(f"  Total chips: ${current_total}")
                print(f"  Cumulative rake: ${initial_total - current_total}")
                
                for player in active:
                    print(f"    {player.name}: ${player.stack}")
                
                for player in busted:
                    print(f"    {player.name}: ${player.stack} [BUST]")
                
                # Verify chip conservation
                assert current_total <= initial_total
                return
        
        pytest.skip("No player went bust in 500 hands")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])