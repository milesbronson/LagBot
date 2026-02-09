#!/usr/bin/env python3
"""
Analyze per-opponent profit data from training runs
"""

import argparse
import sys
from src.training.opponent_profit_tracker import load_opponent_profit_data
import matplotlib.pyplot as plt
import numpy as np


def plot_opponent_profits(run_name: str, save_path: str = None):
    """Plot cumulative profit against each opponent over time"""
    data = load_opponent_profit_data(run_name)

    if data is None:
        print(f"No opponent profit data found for run: {run_name}")
        print(f"Make sure the run has opponent_profits.json file")
        return

    # Extract data
    opponent_results = data['opponent_results']
    history = data['history']

    if not history['timesteps']:
        print("No history data available yet")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Opponent Profit Analysis: {run_name}', fontsize=16, fontweight='bold')

    # Plot 1: Cumulative profit over time
    timesteps = history['timesteps']
    colors = plt.cm.tab10(np.linspace(0, 1, len(opponent_results)))

    for idx, (opp_id, opp_data) in enumerate(opponent_results.items()):
        opp_name = opp_data['name']
        opp_type = opp_data['type']

        # Extract this opponent's profit history
        profits = [snapshot.get(opp_id, 0) for snapshot in history['opponent_profits']]

        ax1.plot(timesteps, profits,
                label=f"{opp_name} ({opp_type})",
                linewidth=2,
                color=colors[idx],
                marker='o',
                markersize=4)

    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Cumulative Profit (normalized)', fontsize=12)
    ax1.set_title('Profit vs Each Opponent Over Time', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Final profit summary (bar chart)
    opponent_names = []
    final_profits = []
    opponent_types = []

    for opp_id, opp_data in opponent_results.items():
        opponent_names.append(opp_data['name'])
        final_profits.append(opp_data['total_profit'])
        opponent_types.append(opp_data['type'])

    if not opponent_names:
        # No data yet
        ax2.text(0.5, 0.5, 'No opponent profit data yet\n(Training just started)',
                ha='center', va='center', transform=ax2.transAxes,
                fontsize=14, style='italic')
        ax2.set_xlabel('Opponent', fontsize=12)
        ax2.set_ylabel('Total Profit (normalized)', fontsize=12)
        ax2.set_title('Total Profit Against Each Opponent', fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        return

    # Sort by profit
    sorted_data = sorted(zip(opponent_names, final_profits, opponent_types),
                        key=lambda x: x[1],
                        reverse=True)
    opponent_names, final_profits, opponent_types = zip(*sorted_data)

    bars = ax2.bar(range(len(opponent_names)), final_profits, color=colors[:len(opponent_names)])

    # Color bars: green for positive, red for negative
    for bar, profit in zip(bars, final_profits):
        if profit > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    ax2.set_xlabel('Opponent', fontsize=12)
    ax2.set_ylabel('Total Profit (normalized)', fontsize=12)
    ax2.set_title('Total Profit Against Each Opponent', fontsize=14)
    ax2.set_xticks(range(len(opponent_names)))
    ax2.set_xticklabels([f"{name}\n({t})" for name, t in zip(opponent_names, opponent_types)],
                        rotation=0, ha='center')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for i, (bar, profit) in enumerate(zip(bars, final_profits)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{profit:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()


def print_opponent_summary(run_name: str):
    """Print text summary of opponent matchups"""
    data = load_opponent_profit_data(run_name)

    if data is None:
        print(f"No opponent profit data found for run: {run_name}")
        return

    opponent_results = data['opponent_results']

    print("\n" + "="*90)
    print(f"OPPONENT PROFIT ANALYSIS: {run_name}")
    print("="*90)

    if not opponent_results:
        print("No data yet")
        return

    # Sort by total profit (descending)
    sorted_opponents = sorted(
        opponent_results.items(),
        key=lambda x: x[1]['total_profit'],
        reverse=True
    )

    print(f"\n{'Opponent':<20} {'Type':<15} {'Hands':<8} {'Total Profit':<15} "
          f"{'Avg/Hand':<12} {'Win Rate':<10}")
    print("-"*90)

    for opp_id, stats in sorted_opponents:
        print(f"{stats['name']:<20} "
              f"{stats['type']:<15} "
              f"{stats['hands_played']:<8} "
              f"{stats['total_profit']:>14.4f} "
              f"{stats['avg_profit']:>11.6f} "
              f"{stats['win_rate']:>9.1%}")

    # Find best and worst matchups
    best = max(opponent_results.items(), key=lambda x: x[1]['total_profit'])
    worst = min(opponent_results.items(), key=lambda x: x[1]['total_profit'])

    print("\n" + "="*90)
    print(f"🏆 Best Matchup:  {best[1]['name']} ({best[1]['type']}) "
          f"→ +{best[1]['total_profit']:.4f} profit")
    print(f"⚠️  Worst Matchup: {worst[1]['name']} ({worst[1]['type']}) "
          f"→ {worst[1]['total_profit']:.4f} profit")
    print("="*90)

    # Analysis
    print("\n📊 INSIGHTS:")
    total_profit = sum(opp['total_profit'] for opp in opponent_results.values())
    print(f"   • Total cumulative profit: {total_profit:.4f}")

    profitable_count = sum(1 for opp in opponent_results.values() if opp['total_profit'] > 0)
    print(f"   • Profitable matchups: {profitable_count}/{len(opponent_results)}")

    if len(opponent_results) >= 2:
        profit_std = np.std([opp['total_profit'] for opp in opponent_results.values()])
        print(f"   • Profit variance: {profit_std:.4f} "
              f"({'high' if profit_std > 0.1 else 'low'} - "
              f"{'exploiting some opponents much better' if profit_std > 0.1 else 'consistent performance'})")


def main():
    parser = argparse.ArgumentParser(description="Analyze per-opponent profit data")
    parser.add_argument("run_name", help="Training run name (e.g., deep_arch_3M_clean)")
    parser.add_argument("--plot", action="store_true", help="Generate profit plots")
    parser.add_argument("--save", type=str, help="Save plot to file instead of showing")
    args = parser.parse_args()

    # Print text summary
    print_opponent_summary(args.run_name)

    # Generate plot if requested
    if args.plot:
        plot_opponent_profits(args.run_name, save_path=args.save)


if __name__ == "__main__":
    main()
