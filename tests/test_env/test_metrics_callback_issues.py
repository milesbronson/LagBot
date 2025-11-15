"""
Tests for MetricsCallback - fixed version
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.training.callbacks import MetricsCallback, SimpleMetricsCallback
from src.training.metrics import TrainingMetrics


class TestMetricsCallback:
    """Tests for MetricsCallback class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PPO model with proper attributes"""
        model = Mock()
        model.num_timesteps = 0
        model.env = Mock()
        model.learning_rate = 0.0003
        model.logger = Mock()
        model.logger.name_to_value = {
            'train/policy_loss': 0.05,
            'train/value_loss': 0.02,
            'train/entropy_loss': 0.001
        }
        model.env.get_attr = Mock(return_value=[{}])
        return model
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock TrainingMetrics"""
        metrics = Mock(spec=TrainingMetrics)
        metrics.log_step = Mock()
        metrics.record_actions = Mock()
        metrics.checkpoint_actions = Mock()
        return metrics
    
    def test_initialization(self, mock_metrics):
        """Test callback initializes with required attributes"""
        callback = MetricsCallback(
            metrics=mock_metrics,
            log_freq=10000,
            verbose=1
        )
        
        assert callback.metrics == mock_metrics
        assert callback.log_freq == 10000
        assert callback.verbose == 1
        assert callback.episode_rewards == []
        assert callback.current_episode_reward == 0
        assert callback.episode_actions == []
        assert callback.episode_wins == 0
        assert callback.episode_count == 0
    
    def test_set_model(self, mock_model, mock_metrics):
        """Test set_model properly sets the model"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        assert callback.model == mock_model
    
    def test_on_step_with_no_episodes(self, mock_model, mock_metrics):
        """Test _on_step returns True and doesn't log if no episodes"""
        callback = MetricsCallback(mock_metrics, log_freq=10000)
        callback.set_model(mock_model)
        
        # Setup locals like Stable Baselines3 would
        callback.locals = {
            'infos': [{}],
            'dones': np.array([False]),
            'rewards': np.array([0.0]),
            'actions': np.array([1])
        }
        callback.num_timesteps = 100
        
        result = callback._on_step()
        
        assert result is True
    
    def test_on_step_tracks_rewards(self, mock_model, mock_metrics):
        """Test _on_step properly accumulates rewards"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        callback.locals = {
            'infos': [{}],
            'dones': np.array([False]),
            'rewards': np.array([5.0]),
            'actions': np.array([1])
        }
        callback.num_timesteps = 10
        
        callback._on_step()
        assert callback.current_episode_reward == 5.0
        
        callback.locals['rewards'] = np.array([3.0])
        callback._on_step()
        assert callback.current_episode_reward == 8.0
    
    def test_on_step_tracks_actions(self, mock_model, mock_metrics):
        """Test _on_step tracks actions correctly"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        callback.locals = {
            'infos': [{}],
            'dones': np.array([False]),
            'rewards': np.array([0.0]),
            'actions': np.array([0, 1, 2])
        }
        callback.num_timesteps = 10
        
        callback._on_step()
        
        assert callback.episode_actions == [0, 1, 2]
    
    def test_on_step_episode_completion(self, mock_model, mock_metrics):
        """Test _on_step handles episode completion"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        # Simulate episode end
        callback.locals = {
            'infos': [{'episode': {'r': 10.0}}],
            'dones': np.array([True]),
            'rewards': np.array([10.0]),
            'actions': np.array([1])
        }
        callback.num_timesteps = 10
        
        callback._on_step()
        
        assert callback.episode_count == 1
        assert len(callback.episode_rewards) == 1
        assert callback.episode_rewards[0] == 10.0
    
    def test_on_step_win_tracking(self, mock_model, mock_metrics):
        """Test _on_step tracks wins correctly"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        # Positive reward = win
        callback.locals = {
            'infos': [{'episode': {'r': 5.0}}],
            'dones': np.array([True]),
            'rewards': np.array([5.0]),
            'actions': np.array([1])
        }
        callback.num_timesteps = 10
        
        callback._on_step()
        
        assert callback.episode_wins == 1
        assert callback.episode_count == 1
    
    def test_on_step_logging_at_frequency(self, mock_model, mock_metrics):
        """Test _on_step logs at specified frequency"""
        callback = MetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        # Setup episode completion
        callback.locals = {
            'infos': [{'episode': {'r': 5.0}}],
            'dones': np.array([True]),
            'rewards': np.array([5.0]),
            'actions': np.array([1])
        }
        
        # Not at logging step yet
        callback.num_timesteps = 50
        callback._on_step()
        mock_metrics.log_step.assert_not_called()
        
        # At logging step
        callback.num_timesteps = 101
        callback._on_step()
        mock_metrics.log_step.assert_called_once()
    
    def test_on_training_start_resets_state(self, mock_model, mock_metrics):
        """Test _on_training_start resets all tracking"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        # Pollute state
        callback.episode_rewards = [1, 2, 3]
        callback.current_episode_reward = 10
        callback.episode_actions = [0, 1, 2]
        callback.episode_wins = 5
        callback.episode_count = 3
        
        callback._on_training_start()
        
        assert callback.episode_rewards == []
        assert callback.current_episode_reward == 0
        assert callback.episode_actions == []
        assert callback.episode_wins == 0
        assert callback.episode_count == 0
    
    def test_on_training_end_logs_metrics(self, mock_model, mock_metrics):
        """Test _on_training_end calls log metrics"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        callback.episode_rewards = [5.0]
        callback.episode_actions = [1, 2, 0]
        callback.episode_count = 1
        callback.episode_wins = 1
        callback.num_timesteps = 1000
        
        callback._on_training_end()
        
        mock_metrics.log_step.assert_called()
    
    def test_multiple_episodes_tracking(self, mock_model, mock_metrics):
        """Test tracking multiple completed episodes"""
        callback = MetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        # First episode
        callback.locals = {
            'infos': [{'episode': {'r': 10.0}}],
            'dones': np.array([True]),
            'rewards': np.array([10.0]),
            'actions': np.array([1])
        }
        callback.num_timesteps = 10
        callback._on_step()
        
        # Second episode
        callback.locals = {
            'infos': [{'episode': {'r': -5.0}}],
            'dones': np.array([True]),
            'rewards': np.array([-5.0]),
            'actions': np.array([0])
        }
        callback.num_timesteps = 20
        callback._on_step()
        
        assert callback.episode_count == 2
        assert len(callback.episode_rewards) == 2
        assert callback.episode_rewards == [10.0, -5.0]
        assert callback.episode_wins == 1  # Only first episode was a win


class TestSimpleMetricsCallback:
    """Tests for SimpleMetricsCallback class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PPO model"""
        model = Mock()
        model.num_timesteps = 0
        return model
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock TrainingMetrics"""
        metrics = Mock(spec=TrainingMetrics)
        metrics.log_step = Mock()
        metrics.record_actions = Mock()
        return metrics
    
    def test_simple_initialization(self, mock_metrics):
        """Test SimpleMetricsCallback initializes"""
        callback = SimpleMetricsCallback(mock_metrics, log_freq=1000)
        
        assert callback.metrics == mock_metrics
        assert callback.log_freq == 1000
        assert callback.current_episode_reward == 0
    
    def test_simple_set_model(self, mock_model, mock_metrics):
        """Test set_model for SimpleMetricsCallback"""
        callback = SimpleMetricsCallback(mock_metrics)
        callback.set_model(mock_model)
        
        assert callback.model == mock_model
    
    def test_simple_on_step_basic(self, mock_model, mock_metrics):
        """Test SimpleMetricsCallback _on_step"""
        callback = SimpleMetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        callback.locals = {
            'infos': [{}],
            'dones': np.array([False]),
            'rewards': np.array([1.0]),
            'actions': np.array([1])
        }
        
        result = callback._on_step()
        assert result is True