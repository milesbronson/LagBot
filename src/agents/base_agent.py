"""
Base agent class for poker players
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for poker agents
    """
    
    def __init__(self, name: str = "Agent"):
        """
        Initialize the agent
        
        Args:
            name: Agent name
        """
        self.name = name
        self.hands_played = 0
        self.total_winnings = 0
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, valid_actions: list) -> int:
        """
        Select an action given an observation
        
        Args:
            observation: Current game observation
            valid_actions: List of valid action indices
            
        Returns:
            Selected action index
        """
        pass
    
    def update(self, observation: np.ndarray, action: int, reward: float, 
               next_observation: np.ndarray, done: bool):
        """
        Update agent after taking an action (for learning agents)
        
        Args:
            observation: Previous observation
            action: Action taken
            reward: Reward received
            next_observation: New observation
            done: Whether episode is done
        """
        pass
    
    def reset(self):
        """Reset agent for a new hand"""
        self.hands_played += 1
    
    def get_stats(self) -> dict:
        """
        Get agent statistics
        
        Returns:
            Dictionary of statistics
        """
        return {
            'name': self.name,
            'hands_played': self.hands_played,
            'total_winnings': self.total_winnings
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"