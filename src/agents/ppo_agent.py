"""
PPO agent using Stable Baselines3
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    PPO-based poker agent using Stable Baselines3
    """
    
    def __init__(
        self, 
        env,
        name: str = "PPOAgent",
        learning_rate: float = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        tensorboard_log: str = "./logs/",
        **kwargs
    ):
        """
        Initialize PPO agent
        
        Args:
            env: Gym environment
            name: Agent name
            learning_rate: Learning rate
            n_steps: Number of steps to run per update
            batch_size: Minibatch size
            n_epochs: Number of epochs for optimization
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: Clipping parameter
            tensorboard_log: Tensorboard log directory
            **kwargs: Additional PPO parameters
        """
        super().__init__(name)
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            tensorboard_log=tensorboard_log,
            verbose=1,
            **kwargs
        )
    
    def select_action(self, observation: np.ndarray, valid_actions: list = None) -> int:
        """
        Select action using PPO policy
        
        Args:
            observation: Current game observation
            valid_actions: List of valid action indices (not used, PPO handles this)
            
        Returns:
            Selected action index
        """
        action, _ = self.model.predict(observation, deterministic=False)
        return int(action)
    
    def select_action_deterministic(self, observation: np.ndarray) -> int:
        """
        Select action deterministically (for evaluation)
        
        Args:
            observation: Current game observation
            
        Returns:
            Selected action index
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def train(self, total_timesteps: int, callback=None):
        """
        Train the agent
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional training callback
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
    
    def save(self, path: str):
        """
        Save the model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a saved model
        
        Args:
            path: Path to load the model from
        """
        self.model = PPO.load(path)
        print(f"Model loaded from {path}")
    
    @classmethod
    def load_agent(cls, path: str, env, name: str = "PPOAgent"):
        """
        Load a PPO agent from a saved model
        
        Args:
            path: Path to the saved model
            env: Gym environment
            name: Agent name
            
        Returns:
            Loaded PPOAgent instance
        """
        agent = cls(env, name=name)
        agent.load(path)
        return agent


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress
    """
    
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        """
        Initialize callback
        
        Args:
            save_freq: Save model every N steps
            save_path: Path to save models
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """
        Called at each step
        
        Returns:
            True to continue training
        """
        # Save model periodically
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/model_{self.n_calls}_steps"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model to {model_path}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout
        """
        # Log custom metrics if needed
        pass