"""
Random agent for baseline testing
"""

import numpy as np
import random
from src.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that takes random actions
    Useful as a baseline for testing
    """
    
    def __init__(self, name: str = "RandomAgent"):
        """
        Initialize random agent
        
        Args:
            name: Agent name
        """
        super().__init__(name)
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """
        Select a random action
        
        Args:
            observation: Current game observation (unused)
            valid_actions: List of valid action indices (if None, assumes all actions valid)
            
        Returns:
            Random action index
        """
        if valid_actions is None:
            # Default actions: 0=fold, 1=check/call, 2=raise
            valid_actions = [0, 1, 2]
        
        return random.choice(valid_actions)


class WeightedRandomAgent(BaseAgent):
    """
    Agent that takes weighted random actions
    Useful for testing with different playstyles
    """
    
    def __init__(self, 
                 name: str = "WeightedRandomAgent",
                 fold_weight: float = 0.2,
                 call_weight: float = 0.5,
                 raise_weight: float = 0.3):
        """
        Initialize weighted random agent
        
        Args:
            name: Agent name
            fold_weight: Probability weight for folding
            call_weight: Probability weight for calling
            raise_weight: Probability weight for raising
        """
        super().__init__(name)
        
        # Normalize weights
        total = fold_weight + call_weight + raise_weight
        self.weights = [fold_weight / total, call_weight / total, raise_weight / total]
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """
        Select a weighted random action
        
        Args:
            observation: Current game observation (unused)
            valid_actions: List of valid action indices
            
        Returns:
            Weighted random action index
        """
        if valid_actions is None:
            valid_actions = [0, 1, 2]
        
        # Filter weights for valid actions
        valid_weights = [self.weights[i] for i in valid_actions]
        
        # Renormalize
        total = sum(valid_weights)
        valid_weights = [w / total for w in valid_weights]
        
        return np.random.choice(valid_actions, p=valid_weights)


class CallAgent(BaseAgent):
    """
    Agent that always calls
    Useful for testing calling station strategy
    """
    
    def __init__(self, name: str = "CallAgent"):
        """
        Initialize call agent
        
        Args:
            name: Agent name
        """
        super().__init__(name)
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """
        Always call (or check if possible)
        
        Args:
            observation: Current game observation (unused)
            valid_actions: List of valid action indices
            
        Returns:
            Call/check action (1)
        """
        return 1  # Always call/check