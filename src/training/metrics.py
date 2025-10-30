"""
Training metrics collection and dashboard data generation
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


class TrainingMetrics:
    """Collects high-level training statistics"""
    
    def __init__(self, run_name: str, save_dir: str = "metrics"):
        self.run_name = run_name
        self.save_dir = save_dir
        self.run_dir = os.path.join(save_dir, run_name)
        
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.metrics = {
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'timesteps': [],
            'rewards': [],
            'avg_reward_100': [],
            'win_rate': [],
            'fold_rate': [],
            'raise_rate': [],
            'all_in_rate': [],
            'learning_rate': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'episodes': []
        }
    
    def log_step(self, 
                 timestep: int,
                 episode_rewards: List[float],
                 agent_stats: Dict[str, Any] = None,
                 learning_metrics: Dict[str, Any] = None):
        """Log metrics from a training step"""
        
        self.metrics['timesteps'].append(timestep)
        self.metrics['episodes'].append(len(episode_rewards))
        
        if episode_rewards:
            self.metrics['rewards'].append(float(np.mean(episode_rewards)))
            # Last 100 episodes average
            last_100 = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            self.metrics['avg_reward_100'].append(float(np.mean(last_100)))
        
        if agent_stats:
            self.metrics['win_rate'].append(agent_stats.get('win_rate', 0))
            self.metrics['fold_rate'].append(agent_stats.get('fold_rate', 0))
            self.metrics['raise_rate'].append(agent_stats.get('raise_rate', 0))
            self.metrics['all_in_rate'].append(agent_stats.get('all_in_rate', 0))
        
        if learning_metrics:
            self.metrics['learning_rate'].append(learning_metrics.get('learning_rate', 0))
            self.metrics['policy_loss'].append(learning_metrics.get('policy_loss', 0))
            self.metrics['value_loss'].append(learning_metrics.get('value_loss', 0))
            self.metrics['entropy'].append(learning_metrics.get('entropy', 0))
        
        self._save()
    
    def _save(self):
        """Save metrics to JSON"""
        metrics_file = os.path.join(self.run_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get high-level summary"""
        if not self.metrics['avg_reward_100']:
            return {}
        
        return {
            'run_name': self.run_name,
            'total_timesteps': self.metrics['timesteps'][-1] if self.metrics['timesteps'] else 0,
            'current_reward': self.metrics['rewards'][-1] if self.metrics['rewards'] else 0,
            'avg_reward_100': self.metrics['avg_reward_100'][-1] if self.metrics['avg_reward_100'] else 0,
            'best_reward_100': max(self.metrics['avg_reward_100']) if self.metrics['avg_reward_100'] else 0,
            'win_rate': self.metrics['win_rate'][-1] if self.metrics['win_rate'] else 0,
            'fold_rate': self.metrics['fold_rate'][-1] if self.metrics['fold_rate'] else 0,
            'raise_rate': self.metrics['raise_rate'][-1] if self.metrics['raise_rate'] else 0,
        }


class DashboardData:
    """Generate dashboard-ready data across multiple runs"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.metrics_dir = metrics_dir
    
    def load_all_runs(self) -> Dict[str, Dict]:
        """Load all training run metrics"""
        runs = {}
        
        if not os.path.exists(self.metrics_dir):
            return runs
        
        for run_name in os.listdir(self.metrics_dir):
            run_path = os.path.join(self.metrics_dir, run_name)
            metrics_file = os.path.join(run_path, 'metrics.json')
            
            if os.path.isfile(metrics_file):
                with open(metrics_file, 'r') as f:
                    runs[run_name] = json.load(f)
        
        return runs
    
    def get_run_comparison(self) -> Dict[str, Any]:
        """Compare all active runs"""
        runs = self.load_all_runs()
        
        comparison = {
            'runs': {},
            'best_run': None,
            'total_runs': len(runs),
            'timestamp': datetime.now().isoformat()
        }
        
        best_reward = -float('inf')
        
        for run_name, metrics in runs.items():
            if not metrics.get('avg_reward_100'):
                continue
            
            current_avg = metrics['avg_reward_100'][-1]
            best_avg = max(metrics['avg_reward_100'])
            
            comparison['runs'][run_name] = {
                'current_avg_reward': current_avg,
                'best_avg_reward': best_avg,
                'total_timesteps': metrics['timesteps'][-1] if metrics['timesteps'] else 0,
                'total_episodes': metrics['episodes'][-1] if metrics['episodes'] else 0,
                'win_rate': metrics['win_rate'][-1] if metrics['win_rate'] else 0,
                'status': 'training'
            }
            
            if current_avg > best_reward:
                best_reward = current_avg
                comparison['best_run'] = run_name
        
        return comparison
    
    def export_csv(self, output_file: str = "training_summary.csv"):
        """Export summary to CSV"""
        runs = self.load_all_runs()
        
        with open(output_file, 'w') as f:
            f.write("Run,Timesteps,Episodes,CurrentReward,AvgReward100,BestReward100,WinRate\n")
            
            for run_name, metrics in sorted(runs.items()):
                if not metrics.get('timesteps'):
                    continue
                
                ts = metrics['timesteps'][-1]
                eps = metrics['episodes'][-1]
                curr_reward = metrics['rewards'][-1] if metrics['rewards'] else 0
                avg_100 = metrics['avg_reward_100'][-1] if metrics['avg_reward_100'] else 0
                best_100 = max(metrics['avg_reward_100']) if metrics['avg_reward_100'] else 0
                wr = metrics['win_rate'][-1] if metrics['win_rate'] else 0
                
                f.write(f"{run_name},{ts},{eps},{curr_reward:.2f},{avg_100:.2f},{best_100:.2f},{wr:.2f}\n")
        
        print(f"Exported to {output_file}")


if __name__ == "__main__":
    # Test
    metrics = TrainingMetrics("test_run")
    
    for step in range(0, 100000, 10000):
        rewards = np.random.normal(0, 10, 100).tolist()
        agent_stats = {
            'win_rate': 0.33 + step/1000000,
            'fold_rate': 0.3,
            'raise_rate': 0.3,
            'all_in_rate': 0.07
        }
        learning_metrics = {
            'learning_rate': 0.0003,
            'policy_loss': 0.5 - step/200000,
            'value_loss': 0.3 - step/300000,
            'entropy': 1.0 - step/100000
        }
        metrics.log_step(step, rewards, agent_stats, learning_metrics)
    
    print(metrics.get_summary())
    
    dashboard = DashboardData()
    comparison = dashboard.get_run_comparison()
    print(json.dumps(comparison, indent=2))
    dashboard.export_csv()