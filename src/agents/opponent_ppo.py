"""
OpponentPPO - A loaded PPO model used as an opponent

Key difference from before:
- Gets the FULL observation including opponent tracking
- Same observation space as the training agent
- Can adapt based on opponent statistics

This means:
- Generation 0 was trained on basic observation
- When used as opponent in Gen 1, it gets Gen 1's full observation
- Gen 0 learns to adapt to the opponent stats in real-time
"""

from typing import Optional
import numpy as np
import torch
from stable_baselines3 import PPO
from src.agents.base_agent import BaseAgent


class OpponentPPO(BaseAgent):
    """
    A previously trained PPO model loaded and used as a fixed opponent.

    Key points:
    - Loads a saved PPO model from a .zip file
    - Receives the SAME observation as the training agent
    - Gets updated observation with opponent stats and action history
    - Plays deterministically (for consistency)
    - Cannot learn - acts as a fixed baseline opponent
    - Supports GPU acceleration (CUDA/MPS)

    Example usage:
        ```python
        opponent_ppo = OpponentPPO('models/generation_0/final_model.zip')

        # Gets same obs as training agent, including opponent stats
        action = opponent_ppo.select_action(obs_with_opponent_stats)
        ```
    """

    def __init__(self, model_path: str, name: str = "OpponentPPO", deterministic: bool = False, device: str = "cpu"):
        """
        Load a trained PPO model to use as opponent.

        Args:
            model_path: Path to saved PPO model (.zip file)
            name: Name for this opponent
            deterministic: If True, always pick best action. If False, sample from policy.
                          False allows for more varied play
            device: Device to load model on ('auto', 'cpu', 'cuda', 'mps')
        """
        super().__init__(name)

        self.model_path = model_path
        self.deterministic = deterministic
        self.model = None
        self.load_success = False

        # Auto-detect device if not specified
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self._load_model()

    def _load_model(self):
        """Load the PPO model from disk"""
        try:
            self.model = PPO.load(self.model_path, device=self.device)
            self.load_success = True
            print(f"✓ Loaded opponent PPO from: {self.model_path} (device: {self.device})")
        except Exception as e:
            print(f"✗ Error loading opponent PPO from {self.model_path}: {e}")
            self.model = None
            self.load_success = False
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """Select action using the loaded PPO policy.

        `valid_actions` is accepted to match BaseAgent's signature (and the
        way EvalGate / autoplay loops call every agent uniformly) but is
        ignored: the PPO policy was trained against the full discrete action
        space, and the env itself coerces illegal actions on step()."""
        if self.model is None:
            # Fallback: default to call
            return 1
        
        try:
            action, _states = self.model.predict(
                observation, 
                deterministic=self.deterministic
            )
            return int(action)
        except Exception as e:
            print(f"Error getting action from opponent PPO: {e}")
            return 1  # Default to call
    
    def select_action_deterministic(self, observation: np.ndarray) -> int:
        """
        Select best action deterministically.
        Useful for consistent, optimal play.
        """
        if self.model is None:
            return 1
        
        try:
            action, _ = self.model.predict(observation, deterministic=True)
            return int(action)
        except Exception as e:
            print(f"Error getting deterministic action: {e}")
            return 1
    
    def is_loaded(self) -> bool:
        """Check if model loaded successfully"""
        return self.load_success and self.model is not None
    
    def get_model(self):
        """Get the underlying Stable Baselines3 PPO model"""
        return self.model
    
    def __repr__(self):
        status = "✓ loaded" if self.load_success else "✗ failed to load"
        return f"OpponentPPO({self.name}, {status}, {self.model_path})"

