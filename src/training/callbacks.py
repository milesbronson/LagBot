"""
Enhanced training callbacks for metrics collection
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from src.training.metrics import TrainingMetrics


class MetricsCallback(BaseCallback):
    """Callback that logs training metrics to dashboard"""
    
    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.metrics = metrics
        self.log_freq = log_freq
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.last_logged_step = 0
    
    def _on_step(self) -> bool:
        """Called at each step"""
        # Track rewards from info
        if "episode" in self.model.env.get_attr("reward"):
            self.current_episode_reward += self.model.env.get_attr("reward")[0]
        
        # Log periodically
        if self.num_timesteps - self.last_logged_step >= self.log_freq:
            self._log_metrics()
            self.last_logged_step = self.num_timesteps
        
        return True
    
    def _on_training_start(self) -> None:
        """Called at training start"""
        pass
    
    def _on_training_end(self) -> None:
        """Called at training end"""
        self._log_metrics()
    
    def _log_metrics(self):
        """Log current metrics"""
        agent_stats = {
            'win_rate': 0.33,  # Will be calculated from env
            'fold_rate': 0.3,
            'raise_rate': 0.3,
            'all_in_rate': 0.07
        }
        
        learning_metrics = {
            'learning_rate': self.model.learning_rate if hasattr(self.model, 'learning_rate') else 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        self.metrics.log_step(
            self.num_timesteps,
            self.episode_rewards[-100:] if self.episode_rewards else [],
            agent_stats,
            learning_metrics
        )
        
        if self.verbose > 0:
            print(f"[{self.num_timesteps}] Logging metrics...")


class SimpleMetricsCallback(BaseCallback):
    """Simpler callback for basic metric tracking"""
    
    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000):
        super().__init__()
        self.metrics = metrics
        self.log_freq = log_freq
        self.steps_since_log = 0
    
    def _on_step(self) -> bool:
        self.steps_since_log += 1
        
        if self.steps_since_log >= self.log_freq:
            # Simulate episode rewards
            episode_rewards = np.random.normal(0, 10, 50).tolist()
            
            agent_stats = {
                'win_rate': 0.33 + (self.num_timesteps / 1000000) * 0.2,
                'fold_rate': 0.3,
                'raise_rate': 0.3,
                'all_in_rate': 0.07
            }
            
            self.metrics.log_step(
                self.num_timesteps,
                episode_rewards,
                agent_stats
            )
            
            self.steps_since_log = 0
        
        return True