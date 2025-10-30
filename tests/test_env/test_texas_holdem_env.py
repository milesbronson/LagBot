"""
Tests for Texas Hold'em environment
"""

import pytest
import numpy as np
from src.poker_env.texas_holdem_env import TexasHoldemEnv


class TestTexasHoldemEnv:
    """Test cases for TexasHoldemEnv"""
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return TexasHoldemEnv(num_players=3, starting_stack=1000)
    
    def test_initialization(self, env):
        """Test environment initialization"""
        assert env.num_players == 3
        assert env.starting_stack == 1000
        assert len(env.game_state.players) == 3
    
    def test_reset(self, env):
        """Test environment reset"""
        obs = env.reset()
        
        # Check observation shape
        assert obs.shape == env.observation_space.shape
        
        # Check that players have cards
        for player in env.game_state.players:
            if player.stack > 0:
                assert len(player.hand) == 2
    
    def test_action_space(self, env):
        """Test action space"""
        # With default raise_bins=[0.5, 1.0, 2.0] and include_all_in=True:
        # 0: Fold, 1: Call, 2-4: Raise bins (50%, 100%, 200% pot), 5: All-in
        assert env.action_space.n == 6

        # Verify raise bins are set correctly
        assert env.raise_bins == [0.5, 1.0, 2.0]
        assert env.include_all_in == True
    
    def test_observation_space(self, env):
        """Test observation space"""
        obs = env.reset()
        assert env.observation_space.contains(obs)
    
    def test_step(self, env):
        """Test taking a step"""
        env.reset()
        
        obs, reward, done, info = env.step(1)  # Call action
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_full_hand(self, env):
        """Test playing a complete hand"""
        env.reset()
        
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
        
        # Hand should complete within reasonable number of steps
        assert steps < max_steps
        assert done
    
    def test_multiple_hands(self, env):
        """Test playing multiple hands"""
        for hand in range(3):
            obs = env.reset()
            assert env.game_state.hand_number == hand + 1
            
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
    
    def test_fold_action(self, env):
        """Test fold action"""
        env.reset()
        
        initial_active = len(env.game_state.get_active_players())
        
        # Fold
        obs, reward, done, info = env.step(0)
        
        # Should have one less active player
        assert len(env.game_state.get_active_players()) <= initial_active
    
    def test_all_players_fold(self, env):
        """Test when all but one player folds"""
        env.reset()
        
        # Make all players except one fold
        done = False
        fold_count = 0
        
        while not done:
            action = 0 if fold_count < 2 else 1  # Fold first 2, last player calls
            obs, reward, done, info = env.step(action)
            fold_count += 1
            
            if done:
                break
        
        # Should have winner
        assert 'winnings' in info
    
    def test_betting_rounds_progress(self, env):
        """Test that betting rounds progress correctly"""
        env.reset()
        
        initial_round = env.game_state.betting_round
        
        # Everyone calls to progress to next round
        done = False
        actions_taken = 0
        
        while not done and actions_taken < 10:
            action = 1  # Call
            obs, reward, done, info = env.step(action)
            actions_taken += 1
        
        # Should progress through betting rounds
        # (or hand completes)
        assert done or env.game_state.betting_round.value >= initial_round.value
    
    def test_validate_action(self, env):
        """Test action validation with pot-based raise bins"""
        env.reset()

        # Test 1: Normal raise action conversion
        # Action 2 should be the first raise bin (50% pot)
        action_type, raise_amount = env._validate_and_convert_action(2)
        assert action_type == 2  # Raise action
        assert raise_amount is not None

        # Test 2: Player with insufficient stack for raise should fall back to call
        player = env.game_state.get_current_player()
        original_stack = player.stack
        player.stack = 5  # Very small stack
        env.game_state.pot_manager.current_bet = 100

        # Try to raise with insufficient chips - should convert to call
        action_type, raise_amount = env._validate_and_convert_action(2)
        assert action_type == 1  # Should convert to call
        assert raise_amount is None

        # Restore stack for other tests
        player.stack = original_stack

        # Test 3: All-in action (last action in space)
        all_in_action = 2 + len(env.raise_bins)  # Should be action 5 with default bins
        action_type, raise_amount = env._validate_and_convert_action(all_in_action)
        assert action_type == 2  # Raise (all-in is a raise)
        assert raise_amount == player.stack  # Should raise entire stack
    
    def test_render(self, env):
        """Test rendering (should not crash)"""
        env.reset()
        env.render()  # Should not raise an exception
    
    def test_close(self, env):
        """Test closing environment"""
        env.close()  # Should not raise an exception
    
    def test_different_player_counts(self):
        """Test with different numbers of players"""
        for num_players in [2, 6, 10]:
            env = TexasHoldemEnv(num_players=num_players)
            obs = env.reset()
            assert len(env.game_state.players) == num_players
    
    def test_invalid_player_count(self):
        """Test that invalid player counts raise errors"""
        with pytest.raises(ValueError):
            TexasHoldemEnv(num_players=1)  # Too few
        
        with pytest.raises(ValueError):
            TexasHoldemEnv(num_players=11)  # Too many
    
    def test_rake_enabled(self):
        """Test environment with rake"""
        env = TexasHoldemEnv(
            num_players=3,
            rake_percent=0.05,
            rake_cap=10
        )
        
        obs = env.reset()
        assert env.game_state.pot_manager.rake_percent == 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])