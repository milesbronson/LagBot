"""
Training dashboard - real-time visualization with action distribution
"""

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import numpy as np
from src.training.metrics import DashboardData


class TrainingDashboard:
    """Interactive dashboard for monitoring training with action tracking"""
    
    def __init__(self, metrics_dir: str = "metrics"):
        self.dashboard = DashboardData(metrics_dir)
        self.metrics_dir = metrics_dir
        
        # Action names for poker
        self.action_names = {
            0: "Fold",
            1: "Check/Call",
            2: "Raise 50%",
            3: "Raise 100%",
            4: "Raise 200%",
            5: "All-in"
        }
    
    def set_action_names(self, names: dict):
        """Set custom action names"""
        self.action_names = names
    
    def plot_single_run(self, run_name: str, save_path: str = None):
        """Plot metrics for a single training run"""
        runs = self.dashboard.load_all_runs()
        
        if run_name not in runs:
            print(f"Run '{run_name}' not found")
            return
        
        metrics = runs[run_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Dashboard: {run_name}', fontsize=16, fontweight='bold')
        
        # Rewards
        ax = axes[0, 0]
        if metrics['timesteps']:
            ax.plot(metrics['timesteps'], metrics['rewards'], label='Episode Reward', alpha=0.5)
            ax.plot(metrics['timesteps'], metrics['avg_reward_100'], label='Avg Reward (100 ep)', linewidth=2)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Reward')
            ax.set_title('Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Action Distribution
        ax = axes[0, 1]
        if metrics['fold_rate'] and metrics['raise_rate'] and metrics['all_in_rate']:
            ax.plot(metrics['timesteps'], metrics['fold_rate'], label='Fold Rate', marker='o', markersize=3)
            ax.plot(metrics['timesteps'], metrics['raise_rate'], label='Raise Rate', marker='s', markersize=3)
            ax.plot(metrics['timesteps'], metrics['all_in_rate'], label='All-in Rate', marker='^', markersize=3)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Rate')
            ax.set_title('Action Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Loss Curves
        ax = axes[1, 0]
        if metrics['policy_loss']:
            ax.plot(metrics['timesteps'], metrics['policy_loss'], label='Policy Loss', linewidth=2)
            if metrics['value_loss']:
                ax.plot(metrics['timesteps'], metrics['value_loss'], label='Value Loss', linewidth=2)
            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Stats Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
        TRAINING SUMMARY
        
        Total Timesteps: {metrics['timesteps'][-1]:,}
        Total Episodes: {metrics['episodes'][-1]:,}
        
        Current Reward: {metrics['rewards'][-1]:.2f}
        Avg Reward (100): {metrics['avg_reward_100'][-1]:.2f}
        Best Reward (100): {max(metrics['avg_reward_100']):.2f}
        
        Win Rate: {metrics['win_rate'][-1]:.1%}
        Fold Rate: {metrics['fold_rate'][-1]:.1%}
        Raise Rate: {metrics['raise_rate'][-1]:.1%}
        
        Start Time: {metrics['start_time']}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def plot_action_distribution(self, action_history: dict, save_path: str = None):
        """Plot action distribution over training time"""
        if not action_history or 'timesteps' not in action_history:
            print("No action history to plot")
            return
        
        timesteps = action_history['timesteps']
        distributions = action_history['distributions']
        
        if not timesteps:
            return
        
        # Extract data by action
        action_data = {i: [] for i in range(len(self.action_names))}
        for dist in distributions:
            for action_id in range(len(self.action_names)):
                action_data[action_id].append(dist.get(action_id, 0))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Action Distribution Over Training', fontsize=14, fontweight='bold')
        
        # Stacked area chart
        ax1.stackplot(
            timesteps,
            *[action_data[i] for i in range(len(self.action_names))],
            labels=[self.action_names.get(i, f"Action {i}") for i in range(len(self.action_names))],
            alpha=0.8
        )
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Action Distribution (Stacked Area)')
        ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Line plot (overlaid)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.action_names)))
        for action_id in range(len(self.action_names)):
            ax2.plot(
                timesteps,
                action_data[action_id],
                label=self.action_names.get(action_id, f"Action {action_id}"),
                marker='o',
                markersize=4,
                color=colors[action_id],
                linewidth=2
            )
        
        ax2.set_xlabel('Timesteps')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Action Distribution (Line Plot)')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def plot_combined_dashboard(self, metrics: dict, action_history: dict = None, 
                               save_path: str = None):
        """Plot combined dashboard with metrics and action distribution"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Reward curve
        ax1 = fig.add_subplot(gs[0, :])
        if metrics.get('timesteps'):
            ax1.plot(metrics['timesteps'], metrics.get('avg_reward_100', []), 
                    label='Avg Reward (100 ep)', linewidth=2, color='blue')
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Reward')
            ax1.set_title('Learning Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. Action distribution (stacked)
        ax2 = fig.add_subplot(gs[1, :])
        if action_history and action_history.get('timesteps'):
            timesteps = action_history['timesteps']
            distributions = action_history['distributions']
            action_data = {i: [] for i in range(len(self.action_names))}
            for dist in distributions:
                for action_id in range(len(self.action_names)):
                    action_data[action_id].append(dist.get(action_id, 0))
            
            ax2.stackplot(
                timesteps,
                *[action_data[i] for i in range(len(self.action_names))],
                labels=[self.action_names.get(i, f"Action {i}") for i in range(len(self.action_names))],
                alpha=0.8
            )
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Percentage (%)')
            ax2.set_title('Action Distribution Over Time')
            ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
        
        # 3. Final action distribution (pie)
        ax3 = fig.add_subplot(gs[2, 0])
        if action_history and action_history.get('distributions'):
            final_dist = action_history['distributions'][-1]
            actions = list(range(len(self.action_names)))
            percentages = [final_dist.get(i, 0) for i in actions]
            labels = [self.action_names.get(i, f"Action {i}") for i in actions]
            
            ax3.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Current Action Distribution')
        
        # 4. Action diversity (entropy)
        ax4 = fig.add_subplot(gs[2, 1])
        if action_history and action_history.get('distributions'):
            entropy_values = []
            for dist in action_history['distributions']:
                probs = np.array([dist.get(i, 0) / 100 for i in range(len(self.action_names))])
                probs = probs[probs > 0]
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropy_values.append(entropy)
            
            ax4.plot(action_history['timesteps'], entropy_values, 
                    marker='o', markersize=4, linewidth=2, color='purple')
            ax4.set_xlabel('Timesteps')
            ax4.set_ylabel('Entropy (Diversity)')
            ax4.set_title('Action Distribution Diversity')
            ax4.grid(True, alpha=0.3)
            max_entropy = np.log(len(self.action_names))
            ax4.axhline(y=max_entropy, color='r', linestyle='--', alpha=0.5, label='Max Diversity')
            ax4.legend()
        
        fig.suptitle('Enhanced Training Dashboard with Action Distribution', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def plot_comparison(self, save_path: str = None):
        """Compare multiple runs"""
        comparison = self.dashboard.get_run_comparison()
        
        if not comparison['runs']:
            print("No runs to compare")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Training Runs Comparison', fontsize=16, fontweight='bold')
        
        run_names = list(comparison['runs'].keys())
        
        # Best rewards
        ax = axes[0]
        best_rewards = [comparison['runs'][r]['best_avg_reward'] for r in run_names]
        current_rewards = [comparison['runs'][r]['current_avg_reward'] for r in run_names]
        
        x = np.arange(len(run_names))
        width = 0.35
        ax.bar(x - width/2, best_rewards, width, label='Best Avg Reward', alpha=0.8)
        ax.bar(x + width/2, current_rewards, width, label='Current Avg Reward', alpha=0.8)
        ax.set_xlabel('Run')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Win rates
        ax = axes[1]
        win_rates = [comparison['runs'][r]['win_rate'] for r in run_names]
        ax.bar(run_names, win_rates, alpha=0.8, color='green')
        ax.set_xlabel('Run')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Comparison')
        ax.set_xticklabels(run_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig
    
    def generate_html_report(self, output_file: str = "training_report.html"):
        """Generate HTML report"""
        comparison = self.dashboard.get_run_comparison()
        runs = self.dashboard.load_all_runs()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Poker Bot Training Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
                h1 { color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }
                h2 { color: #666; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; background-color: white; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #0066cc; color: white; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metric { display: inline-block; margin: 10px 20px; padding: 10px; background-color: white; border-radius: 5px; }
                .metric-label { font-weight: bold; color: #666; }
                .metric-value { font-size: 24px; color: #0066cc; }
                .best { background-color: #ffffcc; }
                .summary { background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <h1>🎰 Poker Bot Training Report</h1>
            <p>Generated: """ + datetime.now().isoformat() + """</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-label">Total Runs</div>
                    <div class="metric-value">""" + str(comparison['total_runs']) + """</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Best Run</div>
                    <div class="metric-value">""" + (comparison['best_run'] or 'N/A') + """</div>
                </div>
            </div>
            
            <h2>Run Comparison</h2>
            <table>
                <tr>
                    <th>Run Name</th>
                    <th>Timesteps</th>
                    <th>Episodes</th>
                    <th>Current Avg Reward</th>
                    <th>Best Avg Reward</th>
                    <th>Win Rate</th>
                </tr>
        """
        
        for run_name in sorted(comparison['runs'].keys()):
            run_data = comparison['runs'][run_name]
            best_class = 'best' if run_name == comparison['best_run'] else ''
            html += f"""
                <tr class="{best_class}">
                    <td><strong>{run_name}</strong></td>
                    <td>{run_data['total_timesteps']:,}</td>
                    <td>{run_data['total_episodes']:,}</td>
                    <td>{run_data['current_avg_reward']:.2f}</td>
                    <td>{run_data['best_avg_reward']:.2f}</td>
                    <td>{run_data['win_rate']:.1%}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Individual Run Details</h2>
        """
        
        for run_name, metrics in sorted(runs.items()):
            if metrics.get('timesteps'):
                html += f"""
                <h3>{run_name}</h3>
                <div class="summary">
                    <div class="metric">
                        <div class="metric-label">Total Timesteps</div>
                        <div class="metric-value">{metrics['timesteps'][-1]:,}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Current Reward</div>
                        <div class="metric-value">{metrics['rewards'][-1]:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Avg (100 eps)</div>
                        <div class="metric-value">{metrics['avg_reward_100'][-1]:.2f}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value">{metrics['win_rate'][-1]:.1%}</div>
                    </div>
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Report generated: {output_file}")


def main():
    """Generate all dashboards"""
    dashboard = TrainingDashboard()
    
    runs = dashboard.dashboard.load_all_runs()
    
    # Plot each run
    for run_name in runs.keys():
        dashboard.plot_single_run(run_name, f"dashboard_{run_name}.png")
    
    # Comparison
    if len(runs) > 1:
        dashboard.plot_comparison("dashboard_comparison.png")
    
    # HTML report
    dashboard.generate_html_report()
    
    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()