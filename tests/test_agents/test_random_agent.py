"""
Tests for random agents
"""

import pytest
import numpy as np
from src.agents.random_agent import RandomAgent, WeightedRandomAgent, CallAgent


class TestRandomAgent:
    """Test cases for RandomAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a random agent"""
        return RandomAgent()
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "RandomAgent"
        assert agent.hands_played == 0
    
    def test_select_action(self, agent):
        """Test action selection"""
        obs = np.zeros(10)
        action = agent.select_action(obs)
        
        assert action in [0, 1, 2]
    
    def test_select_action_with_valid_actions(self, agent):
        """Test action selection with valid actions list"""
        obs = np.zeros(10)
        valid_actions = [1, 2]  # Only call and raise
        
        action = agent.select_action(obs, valid_actions)
        
        assert action in valid_actions
    
    def test_reset(self, agent):
        """Test reset increments hands played"""
        initial_hands = agent.hands_played
        agent.reset()
        
        assert agent.hands_played == initial_hands + 1
    
    def test_get_stats(self, agent):
        """Test getting agent statistics"""
        stats = agent.get_stats()
        
        assert 'name' in stats
        assert 'hands_played' in stats
        assert 'total_winnings' in stats


class TestWeightedRandomAgent:
    """Test cases for WeightedRandomAgent"""
    
    def test_initialization(self):
        """Test weighted agent initialization"""
        agent = WeightedRandomAgent(
            fold_weight=0.1,
            call_weight=0.5,
            raise_weight=0.4
        )
        
        # Weights should sum to 1
        assert abs(sum(agent.weights) - 1.0) < 0.001
    
    def test_select_action(self):
        """Test weighted action selection"""
        agent = WeightedRandomAgent(
            fold_weight=0.0,  # Never fold
            call_weight=0.5,
            raise_weight=0.5
        )
        
        obs = np.zeros(10)
        
        # Take multiple actions to verify distribution
        actions = [agent.select_action(obs) for _ in range(100)]
        
        # Should never fold
        assert 0 not in actions
        # Should have both calls and raises
        assert 1 in actions
        assert 2 in actions
    
    def test_always_fold_agent(self):
        """Test agent that always folds"""
        agent = WeightedRandomAgent(
            fold_weight=1.0,
            call_weight=0.0,
            raise_weight=0.0
        )
        
        obs = np.zeros(10)
        actions = [agent.select_action(obs) for _ in range(10)]
        
        # Should always fold
        assert all(a == 0 for a in actions)


class TestCallAgent:
    """Test cases for CallAgent"""
    
    @pytest.fixture
    def agent(self):
        """Create a call agent"""
        return CallAgent()
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.name == "CallAgent"
    
    def test_always_calls(self, agent):
        """Test that agent always calls"""
        obs = np.zeros(10)
        
        actions = [agent.select_action(obs) for _ in range(10)]
        
        # Should always call (action 1)
        assert all(a == 1 for a in actions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])