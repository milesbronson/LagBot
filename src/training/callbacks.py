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
        self.current_episode_reward = 0  # Track current episode reward
        self.episode_actions = []  # Track actions taken
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0

    def set_model(self, model) -> None:
        """Set the model attribute - required by BaseCallback"""
        super().set_model(model)

    def _on_step(self) -> bool:
        """Called at each step"""
        # Get info from the last step
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.episode_actions.extend(actions.flatten().tolist())
            else:
                self.episode_actions.append(int(actions))

        # Track rewards
        if isinstance(rewards, np.ndarray):
            self.current_episode_reward += float(rewards[0]) if len(rewards) > 0 else 0
        else:
            self.current_episode_reward += float(rewards) if rewards is not None else 0

        # Track episode completion
        for i, info in enumerate(infos):
            if isinstance(dones, np.ndarray):
                done = dones[i] if i < len(dones) else False
            else:
                done = dones if i == 0 else False

            if done:
                # Episode finished, record reward
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_count += 1

                # Check if agent won (positive reward typically indicates win)
                if self.current_episode_reward > 0:
                    self.episode_wins += 1

                # Reset for next episode
                self.current_episode_reward = 0

            # Handle info dict if present
            if 'episode' in info:
                episode_reward = info['episode'].get('r', 0)
                if episode_reward != 0:
                    self.episode_rewards.append(episode_reward)
                    self.episode_count += 1
                    if episode_reward > 0:
                        self.episode_wins += 1

        # Log periodically
        if self.num_timesteps - self.last_logged_step >= self.log_freq:
            self._log_metrics()
            self.last_logged_step = self.num_timesteps

        return True

    def _on_training_start(self) -> None:
        """Called at training start"""
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0

    def _on_training_end(self) -> None:
        """Called at training end"""
        # Log final metrics
        self._log_metrics()

    def _log_metrics(self) -> None:
        """Log collected metrics to both custom metrics and TensorBoard"""
        if not self.episode_rewards:
            if self.verbose > 0:
                print(f"[{self.num_timesteps}] No episodes completed yet")
            return

        # Calculate statistics
        avg_reward = np.mean(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)
        min_reward = np.min(self.episode_rewards)
        
        win_rate = self.episode_wins / max(self.episode_count, 1)

        # Action distribution statistics
        fold_rate = 0
        raise_rate = 0
        all_in_rate = 0
        call_rate = 0

        if self.episode_actions:
            total_actions = len(self.episode_actions)
            fold_count = self.episode_actions.count(0)  # Fold is action 0
            call_count = self.episode_actions.count(1)  # Call is action 1
            raise_count = sum(1 for a in self.episode_actions if a in [2, 3, 4])  # Raise actions
            all_in_count = self.episode_actions.count(5)  # All-in is action 5

            fold_rate = fold_count / total_actions
            call_rate = call_count / total_actions
            raise_rate = raise_count / total_actions
            all_in_rate = all_in_count / total_actions

        # Extract learning metrics from model if available
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0

        if hasattr(self.model, 'logger') and self.model.logger:
            if hasattr(self.model.logger, 'name_to_value'):
                policy_loss = self.model.logger.name_to_value.get('train/policy_loss', 0.0)
                value_loss = self.model.logger.name_to_value.get('train/value_loss', 0.0)
                entropy_loss = self.model.logger.name_to_value.get('train/entropy_loss', 0.0)

        # Prepare agent stats
        agent_stats = {
            'win_rate': float(win_rate),
            'episodes': int(self.episode_count),
            'avg_reward': float(avg_reward),
            'max_reward': float(max_reward),
            'min_reward': float(min_reward),
        }

        # Prepare learning metrics
        learning_metrics = {
            'learning_rate': float(self.model.learning_rate) if hasattr(self.model, 'learning_rate') else 0.0,
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(entropy_loss)
        }

        # Log to custom metrics system
        self.metrics.log_step(
            self.num_timesteps,
            self.episode_rewards[-100:] if self.episode_rewards else [],
            agent_stats,
            learning_metrics
        )

        # Log to TensorBoard
        if hasattr(self.model, 'logger') and self.model.logger:
            # Agent performance metrics
            self.model.logger.record("agent/win_rate", win_rate)
            self.model.logger.record("agent/avg_reward", avg_reward)
            self.model.logger.record("agent/max_reward", max_reward)
            self.model.logger.record("agent/min_reward", min_reward)
            
            # Action distribution metrics
            self.model.logger.record("agent/fold_rate", fold_rate)
            self.model.logger.record("agent/call_rate", call_rate)
            self.model.logger.record("agent/raise_rate", raise_rate)
            self.model.logger.record("agent/all_in_rate", all_in_rate)
            
            # Episode tracking
            self.model.logger.record("agent/episodes_completed", self.episode_count)
            
            # Dump to TensorBoard file
            self.model.logger.dump(self.num_timesteps)

        # Record actions for action distribution tracking
        if self.episode_actions:
            self.metrics.record_actions(self.episode_actions)
            self.metrics.checkpoint_actions(self.num_timesteps)

        # Reset tracking for next logging period
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

        if self.verbose > 0:
            print(f"[{self.num_timesteps}] Metrics - Win Rate: {win_rate:.2%}, Fold: {fold_rate:.2%}, "
                  f"Call: {call_rate:.2%}, Raise: {raise_rate:.2%}, Avg Reward: {avg_reward:.2f}")


class SimpleMetricsCallback(BaseCallback):
    """Simpler callback for basic metric tracking"""

    def __init__(self, metrics: TrainingMetrics, log_freq: int = 10000):
        super().__init__()
        self.metrics = metrics
        self.log_freq = log_freq
        self.steps_since_log = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0
        self.last_logged_step = 0

    def set_model(self, model) -> None:
        """Set the model attribute - required by BaseCallback"""
        super().set_model(model)

    def _on_step(self) -> bool:
        """Called at each step"""
        self.steps_since_log += 1

        # Get info from the last step
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        rewards = self.locals.get('rewards', [])

        # Track actions taken
        if 'actions' in self.locals:
            actions = self.locals['actions']
            if isinstance(actions, np.ndarray):
                self.episode_actions.extend(actions.flatten().tolist())
            else:
                self.episode_actions.append(int(actions))

        # Track rewards
        if isinstance(rewards, np.ndarray):
            self.current_episode_reward += float(rewards[0]) if len(rewards) > 0 else 0
        else:
            self.current_episode_reward += float(rewards) if rewards is not None else 0

        # Track episode completion
        for i, info in enumerate(infos):
            if isinstance(dones, np.ndarray):
                done = dones[i] if i < len(dones) else False
            else:
                done = dones if i == 0 else False

            if done:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_count += 1

                if self.current_episode_reward > 0:
                    self.episode_wins += 1

                self.current_episode_reward = 0

        # Log periodically
        if self.steps_since_log >= self.log_freq:
            self._log_metrics()
            self.steps_since_log = 0

        return True

    def _on_training_start(self) -> None:
        """Called at training start"""
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0

    def _log_metrics(self) -> None:
        """Log collected metrics"""
        if not self.episode_rewards:
            return

        avg_reward = np.mean(self.episode_rewards)
        win_rate = self.episode_wins / max(self.episode_count, 1)

        # Record actions
        if self.episode_actions:
            self.metrics.record_actions(self.episode_actions)

        # Reset tracking
        self.episode_actions = []
        self.episode_wins = 0
        self.episode_count = 0