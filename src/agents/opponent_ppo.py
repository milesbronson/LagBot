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
    
    Example usage:
        ```python
        opponent_ppo = OpponentPPO('models/generation_0/final_model.zip')
        
        # Gets same obs as training agent, including opponent stats
        action = opponent_ppo.select_action(obs_with_opponent_stats)
        ```
    """
    
    def __init__(self, model_path: str, name: str = "OpponentPPO", deterministic: bool = False):
        """
        Load a trained PPO model to use as opponent.
        
        Args:
            model_path: Path to saved PPO model (.zip file)
            name: Name for this opponent
            deterministic: If True, always pick best action. If False, sample from policy.
                          False allows for more varied play
        """
        super().__init__(name)
        
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = None
        self.load_success = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the PPO model from disk"""
        try:
            self.model = PPO.load(self.model_path)
            self.load_success = True
            print(f"✓ Loaded opponent PPO from: {self.model_path}")
        except Exception as e:
            print(f"✗ Error loading opponent PPO from {self.model_path}: {e}")
            self.model = None
            self.load_success = False
    
    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action using the loaded PPO policy.
        
        CRITICAL: This observation should be the SAME observation that the
        training agent receives, including:
        - Hole cards
        - Community cards
        - Stack sizes
        - Pot info
        - Opponent statistics (if included)
        - Action history (if included)
        
        Args:
            observation: Full game observation (same format as training agent)
            
        Returns:
            Action index (0=fold, 1=call, 2=raise)
        """
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
    
    def select_action_stochastic(self, observation: np.ndarray) -> int:
        """
        Select action by sampling from policy (stochastic).
        Useful for more varied play.
        """
        if self.model is None:
            return 1
        
        try:
            action, _ = self.model.predict(observation, deterministic=False)
            return int(action)
        except Exception as e:
            print(f"Error getting stochastic action: {e}")
            return 1
    
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


class OpponentPPOEnsemble:
    """
    Use multiple opponent PPO versions in a round-robin fashion.
    
    Useful for:
    - Training against multiple previous generations
    - Avoiding overfitting to a single opponent
    - Creating more varied training
    
    Example:
        ```python
        opponents = [
            OpponentPPO('models/gen0/final_model.zip'),
            OpponentPPO('models/gen1/final_model.zip'),
        ]
        ensemble = OpponentPPOEnsemble(opponents)
        
        # Alternates between gen0 and gen1
        action = ensemble.select_action(obs)
        ```
    """
    
    def __init__(self, opponents: list, strategy: str = "round_robin"):
        """
        Args:
            opponents: List of OpponentPPO instances
            strategy: How to select opponent
                     - "round_robin": Cycle through opponents
                     - "random": Random selection each call
                     - "best": Always use best performing
        """
        self.opponents = [o for o in opponents if o.is_loaded()]
        self.strategy = strategy
        self.call_count = 0
        
        if not self.opponents:
            print("Warning: No valid opponents in ensemble!")
    
    def select_action(self, observation: np.ndarray) -> int:
        """Select opponent and get action"""
        if not self.opponents:
            return 1  # Default to call
        
        if self.strategy == "round_robin":
            opponent = self.opponents[self.call_count % len(self.opponents)]
            self.call_count += 1
            return opponent.select_action(observation)
        
        elif self.strategy == "random":
            import random
            opponent = random.choice(self.opponents)
            return opponent.select_action(observation)
        
        else:
            # Default to first
            return self.opponents[0].select_action(observation)
    
    def __repr__(self):
        return f"OpponentPPOEnsemble({len(self.opponents)} opponents, {self.strategy})"


# Utility function to find and load latest opponent
def load_latest_opponent_ppo(models_dir: str = "models") -> Optional[OpponentPPO]:
    """
    Find the most recent trained PPO model and load it as opponent.
    
    Args:
        models_dir: Directory containing trained models
        
    Returns:
        OpponentPPO instance, or None if not found
        
    Example:
        ```python
        opponent = load_latest_opponent_ppo()
        if opponent and opponent.is_loaded():
            print(f"Loaded opponent: {opponent}")
        ```
    """
    from pathlib import Path
    
    models_path = Path(models_dir)
    if not models_path.exists():
        return None
    
    # Find all final_model.zip files
    final_models = list(models_path.glob("*/final_model.zip"))
    
    if not final_models:
        return None
    
    # Get the most recent
    latest_path = max(final_models, key=lambda p: p.stat().st_mtime)
    
    return OpponentPPO(str(latest_path), name=f"Previous({latest_path.parent.name})")