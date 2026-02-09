"""
Track profit/loss against individual opponents
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np


class OpponentProfitTracker:
    """Track how much money the agent makes/loses against each opponent"""

    def __init__(self, run_name: str, save_dir: str = "metrics"):
        self.run_name = run_name
        self.save_dir = save_dir
        self.run_dir = os.path.join(save_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # opponent_id -> list of results
        self.opponent_results = defaultdict(lambda: {
            'name': 'Unknown',
            'type': 'unknown',  # 'call', 'random', 'ppo_gen_X', 'tight', etc.
            'hands_played': 0,
            'total_profit': 0.0,
            'profits': [],  # Individual hand results
            'win_count': 0,
            'loss_count': 0,
            'avg_profit': 0.0,
            'win_rate': 0.0
        })

        # Track opponent types seen
        self.opponent_types = {}  # opponent_id -> (name, type)

        # History snapshots for graphing
        self.history = {
            'timesteps': [],
            'opponent_profits': []  # List of {opponent_id: cumulative_profit}
        }

    def register_opponent(self, opponent_id: int, name: str, opponent_type: str):
        """Register an opponent with their type (call, random, ppo_gen_1, etc.)"""
        self.opponent_types[opponent_id] = (name, opponent_type)
        self.opponent_results[opponent_id]['name'] = name
        self.opponent_results[opponent_id]['type'] = opponent_type

    def record_hand_result(self, opponent_id: int, profit: float):
        """
        Record the result of a hand against a specific opponent

        Args:
            opponent_id: ID of the opponent (1 or 2 for 3-player game)
            profit: Amount won/lost in this hand (normalized by starting stack)
        """
        stats = self.opponent_results[opponent_id]

        # Update statistics
        stats['hands_played'] += 1
        stats['total_profit'] += profit
        stats['profits'].append(profit)

        if profit > 0:
            stats['win_count'] += 1
        elif profit < 0:
            stats['loss_count'] += 1

        # Update derived stats
        stats['avg_profit'] = stats['total_profit'] / stats['hands_played']
        stats['win_rate'] = stats['win_count'] / stats['hands_played']

    def checkpoint(self, timestep: int):
        """Save a snapshot of current cumulative profits for graphing"""
        snapshot = {
            opp_id: stats['total_profit']
            for opp_id, stats in self.opponent_results.items()
        }

        self.history['timesteps'].append(timestep)
        self.history['opponent_profits'].append(snapshot)

        self._save()

    def _save(self):
        """Save opponent profit data to JSON"""
        # Convert defaultdict to regular dict for JSON serialization
        data = {
            'opponent_results': {
                str(opp_id): {
                    'name': stats['name'],
                    'type': stats['type'],
                    'hands_played': stats['hands_played'],
                    'total_profit': float(stats['total_profit']),
                    'avg_profit': float(stats['avg_profit']),
                    'win_count': stats['win_count'],
                    'loss_count': stats['loss_count'],
                    'win_rate': float(stats['win_rate']),
                    # Don't save full profit list (too large)
                }
                for opp_id, stats in self.opponent_results.items()
            },
            'history': self.history
        }

        output_file = os.path.join(self.run_dir, 'opponent_profits.json')
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of performance against each opponent"""
        summary = {}

        for opp_id, stats in self.opponent_results.items():
            if stats['hands_played'] > 0:
                summary[stats['name']] = {
                    'type': stats['type'],
                    'hands_played': stats['hands_played'],
                    'total_profit': stats['total_profit'],
                    'avg_profit_per_hand': stats['avg_profit'],
                    'win_rate': stats['win_rate'],
                    'profit_per_100_hands': stats['avg_profit'] * 100
                }

        return summary

    def get_best_matchup(self) -> tuple:
        """Return (opponent_name, profit) for most profitable opponent"""
        if not self.opponent_results:
            return None, 0.0

        best_opp = max(
            self.opponent_results.items(),
            key=lambda x: x[1]['total_profit']
        )

        return best_opp[1]['name'], best_opp[1]['total_profit']

    def get_worst_matchup(self) -> tuple:
        """Return (opponent_name, profit) for least profitable opponent"""
        if not self.opponent_results:
            return None, 0.0

        worst_opp = min(
            self.opponent_results.items(),
            key=lambda x: x[1]['total_profit']
        )

        return worst_opp[1]['name'], worst_opp[1]['total_profit']

    def print_summary(self):
        """Print a formatted summary of opponent matchups"""
        print("\n" + "="*70)
        print("OPPONENT PROFIT ANALYSIS")
        print("="*70)

        summary = self.get_summary()

        if not summary:
            print("No data yet")
            return

        # Sort by total profit (descending)
        sorted_opponents = sorted(
            summary.items(),
            key=lambda x: x[1]['total_profit'],
            reverse=True
        )

        print(f"\n{'Opponent':<20} {'Type':<15} {'Hands':<8} {'Total $':<10} {'Avg $/Hand':<12} {'Win Rate':<10}")
        print("-"*85)

        for opp_name, stats in sorted_opponents:
            print(f"{opp_name:<20} "
                  f"{stats['type']:<15} "
                  f"{stats['hands_played']:<8} "
                  f"{stats['total_profit']:>9.2f} "
                  f"{stats['avg_profit_per_hand']:>11.4f} "
                  f"{stats['win_rate']:>9.1%}")

        best_name, best_profit = self.get_best_matchup()
        worst_name, worst_profit = self.get_worst_matchup()

        print("\n" + "="*70)
        print(f"Best Matchup:  {best_name} (+{best_profit:.2f})")
        print(f"Worst Matchup: {worst_name} ({worst_profit:.2f})")
        print("="*70)


def load_opponent_profit_data(run_name: str, save_dir: str = "metrics") -> Dict:
    """Load opponent profit data from saved JSON"""
    file_path = os.path.join(save_dir, run_name, 'opponent_profits.json')

    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        return json.load(f)
