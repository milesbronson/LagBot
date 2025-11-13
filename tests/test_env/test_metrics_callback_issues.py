import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.training.callbacks import MetricsCallback
from src.training.metrics import TrainingMetrics


class TestMetricsCallback:
    """Tests to identify issues with MetricsCallback"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PPO model"""
        model = Mock()
        model.num_timesteps = 0
        model.env = Mock()
        model.learning_rate = 0.0003
        model.env.get_attr = Mock(return_value=[{}])
        return model
    
    @pytest.fixture
    def mock_metrics(self):
        """Create mock TrainingMetrics"""
        return Mock(spec=TrainingMetrics)
    
    def test_initialization(self, mock_metrics):
        """Test callback initializes"""
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
    
    def test_on_step_increments_counter(self, mock_model, mock_metrics):
        """Test _on_step increments step counter"""
        callback = MetricsCallback(mock_metrics, log_freq=1000)
        callback.set_model(mock_model)
        
        mock_model.num_timesteps = 100
        result = callback._on_step()
        
        assert result is True
    
    def test_issue_env_get_attr_call(self, mock_model, mock_metrics):
        """
        ISSUE: Calling model.env.get_attr("reward") assumes:
        1. Environment is wrapped and accessible
        2. "reward" is an attribute that can be retrieved
        This will fail in actual training
        """
        callback = MetricsCallback(mock_metrics, log_freq=1000)
        callback.set_model(mock_model)
        
        # This is what the code tries to do:
        # if "episode" in self.model.env.get_attr("reward"):
        
        # Problem: get_attr returns list of values, not dict
        mock_model.env.get_attr = Mock(return_value=[0.5])
        
        # This line will fail:
        # "episode" in [0.5] raises TypeError
        with pytest.raises(TypeError):
            if "episode" in mock_model.env.get_attr("reward"):
                pass
    
    def test_issue_get_attr_incorrect_usage(self, mock_model, mock_metrics):
        """
        ISSUE: get_attr returns list, but code checks dict
        get_attr("reward") returns: [value1, value2, ...]
        But code checks: "episode" in result
        """
        callback = MetricsCallback(mock_metrics, log_freq=1000)
        callback.set_model(mock_model)
        
        # get_attr returns a list
        mock_model.env.get_attr = Mock(return_value=[10.0, 5.0])
        
        # This will fail - can't do "x" in [10.0, 5.0]
        try:
            if "episode" in mock_model.env.get_attr("reward"):
                pass
            assert False, "Should have raised TypeError"
        except TypeError as e:
            assert "in" in str(e) or "string" in str(e)
    
    def test_issue_reward_tracking_missing(self, mock_model, mock_metrics):
        """
        ISSUE: No mechanism to actually get episode rewards
        The callback never receives reward values from steps
        current_episode_reward is incremented but never initialized properly
        """
        callback = MetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        # Run multiple steps
        for i in range(100):
            mock_model.num_timesteps = i + 1
            callback._on_step()
        
        # episode_rewards is never populated
        assert callback.episode_rewards == []
        # current_episode_reward is never reset between episodes
        assert callback.current_episode_reward == 0
    
    def test_issue_log_metrics_uses_fake_data(self, mock_model, mock_metrics):
        """
        ISSUE: _log_metrics uses hardcoded fake stats
        win_rate, fold_rate, etc. are always the same
        """
        callback = MetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        # Trigger logging
        mock_model.num_timesteps = 100
        callback.steps_since_log = 100
        callback._on_step()
        
        # Check what was logged
        if mock_metrics.log_step.called:
            call_args = mock_metrics.log_step.call_args
            agent_stats = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get('agent_stats')
            
            # These are hardcoded fake values
            assert agent_stats['win_rate'] == 0.33
            assert agent_stats['fold_rate'] == 0.3
            assert agent_stats['all_in_rate'] == 0.07
    
    def test_issue_no_info_dict_handling(self, mock_model, mock_metrics):
        """
        ISSUE: Callback never receives 'info' dict from step()
        Real rewards come in info dict, but callback has no access to it
        """
        callback = MetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        # There's no mechanism to pass info to callback
        # Stable Baselines3 callbacks don't automatically receive info
        # This is a fundamental design issue
        
        assert not hasattr(callback, 'last_info')
        assert not hasattr(callback, 'episode_rewards_dict')
    
    def test_issue_no_env_reference(self, mock_model, mock_metrics):
        """
        ISSUE: No direct reference to environment to get stats
        Callback can't access game_state, player stats, etc.
        """
        callback = MetricsCallback(mock_metrics, log_freq=100)
        callback.set_model(mock_model)
        
        # Callback has no way to access:
        # - env.game_state.players
        # - win/loss statistics
        # - action distributions
        # - hand history
        
        # It only has access to model, which doesn't expose these
        assert not hasattr(callback, 'env')


class TestMetricsCallbackSummary:
    """Summary of identified issues"""
    
    def test_summary_of_issues(self):
        """Document all issues with MetricsCallback"""
        
        issues = {
            "1_get_attr_type_error": {
                "problem": "model.env.get_attr('reward') returns list, but code checks for 'episode' in it",
                "line": "if 'episode' in self.model.env.get_attr('reward')",
                "error": "TypeError: argument of type 'list' is not iterable"
            },
            "2_no_reward_tracking": {
                "problem": "No mechanism to capture actual episode rewards from environment",
                "consequence": "episode_rewards stays empty, metrics are never populated"
            },
            "3_hardcoded_fake_stats": {
                "problem": "Agent stats (win_rate, fold_rate) are hardcoded constants",
                "consequence": "No real poker statistics logged"
            },
            "4_no_info_dict_access": {
                "problem": "Callback has no access to info dict returned by step()",
                "consequence": "Can't get hand results, rewards, etc."
            },
            "5_no_env_access": {
                "problem": "Callback doesn't have reference to actual environment",
                "consequence": "Can't access game_state, players, action history, etc."
            },
            "6_wrong_design": {
                "problem": "Trying to get data that's not available to callbacks",
                "solution": "Need to log stats directly from environment or pass them to callback"
            }
        }
        
        for issue_id, details in issues.items():
            print(f"\n{issue_id}:")
            print(f"  Problem: {details['problem']}")
            if 'consequence' in details:
                print(f"  Consequence: {details['consequence']}")
            if 'solution' in details:
                print(f"  Solution: {details['solution']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])