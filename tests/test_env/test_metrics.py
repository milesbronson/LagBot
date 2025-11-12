"""
Tests for updated metrics.py with action distribution tracking
"""

import pytest
import os
import json
import tempfile
from src.training.metrics import TrainingMetrics, DashboardData


class TestTrainingMetrics:
    """Test TrainingMetrics class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for metrics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def metrics(self, temp_dir):
        """Create TrainingMetrics instance"""
        return TrainingMetrics("test_run", save_dir=temp_dir)
    
    def test_initialization(self, metrics):
        """Test metrics initialization"""
        assert metrics.run_name == "test_run"
        assert metrics.total_actions == 0
        assert metrics.action_counts == {}
        assert metrics.action_history['timesteps'] == []
        assert metrics.action_history['distributions'] == []
    
    def test_record_single_action(self, metrics):
        """Test recording a single action"""
        metrics.record_actions([0], num_actions=6)
        
        assert metrics.total_actions == 1
        assert metrics.action_counts[0] == 1
    
    def test_record_multiple_actions(self, metrics):
        """Test recording multiple actions"""
        actions = [0, 1, 1, 2, 0, 1]
        metrics.record_actions(actions, num_actions=6)
        
        assert metrics.total_actions == 6
        assert metrics.action_counts[0] == 2
        assert metrics.action_counts[1] == 3
        assert metrics.action_counts[2] == 1
    
    def test_record_batch_actions(self, metrics):
        """Test recording actions in batches"""
        metrics.record_actions([0, 1], num_actions=6)
        assert metrics.total_actions == 2
        
        metrics.record_actions([1, 2], num_actions=6)
        assert metrics.total_actions == 4
        assert metrics.action_counts[1] == 2
    
    def test_record_actions_out_of_range(self, metrics):
        """Test that out-of-range actions are ignored"""
        metrics.record_actions([0, 10, 1], num_actions=6)
        
        assert metrics.total_actions == 2
        assert metrics.action_counts[0] == 1
        assert metrics.action_counts[1] == 1
        assert 10 not in metrics.action_counts
    
    def test_record_step(self, metrics):
        """Test recording training step"""
        rewards = [10.5, 12.3, 11.8]
        metrics.record_step(
            timestep=1000,
            episode_rewards=rewards,
            agent_stats={'win_rate': 0.35, 'fold_rate': 0.3},
            learning_metrics={'learning_rate': 0.0003}
        )
        
        assert len(metrics.metrics['timesteps']) == 1
        assert metrics.metrics['timesteps'][0] == 1000
        assert len(metrics.metrics['rewards']) == 1
        assert metrics.metrics['rewards'][0] == pytest.approx(11.53, rel=0.01)
        assert metrics.metrics['win_rate'][-1] == 0.35
        assert metrics.metrics['learning_rate'][-1] == 0.0003
    
    def test_record_step_100_episode_average(self, metrics):
        """Test 100-episode moving average"""
        rewards = list(range(50, 150))  # 100 rewards
        metrics.record_step(timestep=1000, episode_rewards=rewards)
        
        # Average should be mean of all 100
        expected_avg = sum(rewards) / len(rewards)
        assert metrics.metrics['avg_reward_100'][0] == pytest.approx(expected_avg)
    
    def test_record_step_less_than_100_episodes(self, metrics):
        """Test average with less than 100 episodes"""
        rewards = list(range(10, 30))  # 20 rewards
        metrics.record_step(timestep=1000, episode_rewards=rewards)
        
        # Should average all 20
        expected_avg = sum(rewards) / len(rewards)
        assert metrics.metrics['avg_reward_100'][0] == pytest.approx(expected_avg)
    
    def test_checkpoint_actions(self, metrics):
        """Test action distribution checkpoint"""
        metrics.record_actions([0, 0, 1, 1, 1, 2], num_actions=6)
        metrics.checkpoint_actions(timestep=1000, num_actions=6)
        
        assert len(metrics.action_history['timesteps']) == 1
        assert metrics.action_history['timesteps'][0] == 1000
        
        dist = metrics.action_history['distributions'][0]
        assert dist[0] == pytest.approx(33.33, rel=0.01)  # 2/6
        assert dist[1] == pytest.approx(50.0, rel=0.01)   # 3/6
        assert dist[2] == pytest.approx(16.67, rel=0.01)  # 1/6
        assert dist[3] == 0.0
    
    def test_checkpoint_actions_zero_total(self, metrics):
        """Test checkpoint with no actions recorded"""
        metrics.checkpoint_actions(timestep=1000, num_actions=6)
        
        dist = metrics.action_history['distributions'][0]
        # All actions should be 0%
        assert all(v == 0.0 for v in dist.values())
    
    def test_checkpoint_multiple_times(self, metrics):
        """Test multiple checkpoints"""
        # First checkpoint
        metrics.record_actions([0, 1, 1], num_actions=6)
        metrics.checkpoint_actions(timestep=100, num_actions=6)
        
        # Second checkpoint
        metrics.record_actions([2, 2, 2], num_actions=6)
        metrics.checkpoint_actions(timestep=200, num_actions=6)
        
        assert len(metrics.action_history['timesteps']) == 2
        assert metrics.action_history['timesteps'] == [100, 200]
        
        # First distribution: 0:33%, 1:67%
        assert metrics.action_history['distributions'][0][0] == pytest.approx(33.33, rel=0.01)
        
        # Second distribution: 0:16.67%, 1:33%, 2:50%
        assert metrics.action_history['distributions'][1][2] == pytest.approx(50.0, rel=0.01)
    
    def test_metrics_saved_to_json(self, metrics, temp_dir):
        """Test that metrics are saved to JSON"""
        metrics.record_step(
            timestep=1000,
            episode_rewards=[10.0, 20.0]
        )
        
        metrics_file = os.path.join(metrics.run_dir, 'metrics.json')
        assert os.path.exists(metrics_file)
        
        with open(metrics_file, 'r') as f:
            saved = json.load(f)
        
        assert saved['run_name'] == 'test_run'
        assert saved['timesteps'][0] == 1000
    
    def test_action_history_saved_to_json(self, metrics, temp_dir):
        """Test that action history is saved to JSON"""
        metrics.record_actions([0, 1, 1], num_actions=6)
        metrics.checkpoint_actions(timestep=100, num_actions=6)
        
        actions_file = os.path.join(metrics.run_dir, 'action_history.json')
        assert os.path.exists(actions_file)
        
        with open(actions_file, 'r') as f:
            saved = json.load(f)
        
        assert saved['timesteps'][0] == 100
        assert '0' in saved['distributions'][0]
    
    def test_get_summary(self, metrics):
        """Test summary generation"""
        metrics.record_step(
            timestep=1000,
            episode_rewards=[10.0, 20.0, 30.0],
            agent_stats={'win_rate': 0.4, 'fold_rate': 0.3, 'raise_rate': 0.2},
            learning_metrics={}
        )
        
        summary = metrics.get_summary()
        
        assert summary['run_name'] == 'test_run'
        assert summary['total_timesteps'] == 1000
        assert summary['current_reward'] == pytest.approx(20.0)
        assert summary['avg_reward_100'] == pytest.approx(20.0)
        assert summary['win_rate'] == 0.4
    
    def test_get_summary_empty(self, metrics):
        """Test summary on empty metrics"""
        summary = metrics.get_summary()
        assert summary == {}


class TestDashboardData:
    """Test DashboardData class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory with test data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test run
            run_dir = os.path.join(tmpdir, 'test_run')
            os.makedirs(run_dir)
            
            # Save test metrics
            metrics_data = {
                'timesteps': [100, 200],
                'avg_reward_100': [5.0, 10.0],
                'episodes': [10, 20],
                'win_rate': [0.3, 0.4],
                'fold_rate': [0.3, 0.3],
                'raise_rate': [0.2, 0.2],
                'all_in_rate': [0.0, 0.0]
            }
            with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f)
            
            # Save test action history
            actions_data = {
                'timesteps': [100, 200],
                'distributions': [
                    {'0': 33.3, '1': 66.7, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0},
                    {'0': 20.0, '1': 50.0, '2': 30.0, '3': 0.0, '4': 0.0, '5': 0.0}
                ]
            }
            with open(os.path.join(run_dir, 'action_history.json'), 'w') as f:
                json.dump(actions_data, f)
            
            yield tmpdir
    
    def test_initialization(self, temp_dir):
        """Test DashboardData initialization"""
        dashboard = DashboardData(metrics_dir=temp_dir)
        assert dashboard.metrics_dir == temp_dir
    
    def test_load_all_runs(self, temp_dir):
        """Test loading all runs"""
        dashboard = DashboardData(metrics_dir=temp_dir)
        runs = dashboard.load_all_runs()
        
        assert 'test_run' in runs
        assert runs['test_run']['timesteps'] == [100, 200]
        assert runs['test_run']['avg_reward_100'] == [5.0, 10.0]
    
    def test_load_action_history(self, temp_dir):
        """Test loading action history"""
        dashboard = DashboardData(metrics_dir=temp_dir)
        history = dashboard.load_action_history('test_run')
        
        assert history['timesteps'] == [100, 200]
        assert len(history['distributions']) == 2
        assert history['distributions'][0]['0'] == 33.3
    
    def test_load_action_history_missing(self, temp_dir):
        """Test loading action history for non-existent run"""
        dashboard = DashboardData(metrics_dir=temp_dir)
        history = dashboard.load_action_history('nonexistent')
        
        assert history['timesteps'] == []
        assert history['distributions'] == []
    
    def test_load_all_runs_empty_dir(self):
        """Test loading from empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dashboard = DashboardData(metrics_dir=tmpdir)
            runs = dashboard.load_all_runs()
            
            assert runs == {}
    
    def test_get_run_comparison(self, temp_dir):
        """Test run comparison"""
        dashboard = DashboardData(metrics_dir=temp_dir)
        comparison = dashboard.get_run_comparison()
        
        assert comparison['total_runs'] == 1
        assert comparison['best_run'] == 'test_run'
        assert 'test_run' in comparison['runs']
        assert comparison['runs']['test_run']['current_avg_reward'] == 10.0
        assert comparison['runs']['test_run']['best_avg_reward'] == 10.0
    
    def test_get_run_comparison_multiple_runs(self):
        """Test comparison with multiple runs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create run 1
            run1_dir = os.path.join(tmpdir, 'run1')
            os.makedirs(run1_dir)
            with open(os.path.join(run1_dir, 'metrics.json'), 'w') as f:
                json.dump({
                    'timesteps': [1000],
                    'avg_reward_100': [15.0],
                    'episodes': [100],
                    'win_rate': [0.4]
                }, f)
            
            # Create run 2
            run2_dir = os.path.join(tmpdir, 'run2')
            os.makedirs(run2_dir)
            with open(os.path.join(run2_dir, 'metrics.json'), 'w') as f:
                json.dump({
                    'timesteps': [1000],
                    'avg_reward_100': [20.0],
                    'episodes': [100],
                    'win_rate': [0.5]
                }, f)
            
            dashboard = DashboardData(metrics_dir=tmpdir)
            comparison = dashboard.get_run_comparison()
            
            assert comparison['total_runs'] == 2
            assert comparison['best_run'] == 'run2'  # run2 has higher reward
            assert comparison['runs']['run1']['current_avg_reward'] == 15.0
            assert comparison['runs']['run2']['current_avg_reward'] == 20.0


class TestMetricsIntegration:
    """Integration tests for metrics workflow"""
    
    def test_full_training_workflow(self):
        """Test complete training workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = TrainingMetrics("integration_test", save_dir=tmpdir)
            
            # Simulate training steps
            for step in range(3):
                actions = [i % 6 for i in range(100)]
                rewards = [10 + step for _ in range(10)]
                
                metrics.record_actions(actions, num_actions=6)
                metrics.record_step(
                    timestep=(step + 1) * 1000,
                    episode_rewards=rewards,
                    agent_stats={'win_rate': 0.3 + step * 0.05}
                )
                metrics.checkpoint_actions((step + 1) * 1000, num_actions=6)
            
            # Verify files exist
            metrics_file = os.path.join(metrics.run_dir, 'metrics.json')
            actions_file = os.path.join(metrics.run_dir, 'action_history.json')
            
            assert os.path.exists(metrics_file)
            assert os.path.exists(actions_file)
            
            # Load and verify
            dashboard = DashboardData(metrics_dir=tmpdir)
            runs = dashboard.load_all_runs()
            history = dashboard.load_action_history('integration_test')
            
            assert 'integration_test' in runs
            assert len(history['timesteps']) == 3
    
    def test_metrics_persistence(self):
        """Test that metrics persist across instances"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First instance
            metrics1 = TrainingMetrics("persist_test", save_dir=tmpdir)
            metrics1.record_actions([0, 1, 1], num_actions=6)
            metrics1.checkpoint_actions(100, num_actions=6)
            
            # Load from disk
            dashboard = DashboardData(metrics_dir=tmpdir)
            runs = dashboard.load_all_runs()
            
            assert 'persist_test' in runs
            
            # Second instance reads same data
            metrics2 = TrainingMetrics("persist_test", save_dir=tmpdir)
            history = dashboard.load_action_history("persist_test")
            
            assert history['timesteps'][0] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])