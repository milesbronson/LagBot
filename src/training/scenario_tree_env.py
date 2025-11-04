"""
Scenario tree environment wrapper - reduces variance through hand replays
"""

import copy
import random
from typing import Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from treys import Deck
from src.training.opponent_tracker import OpponentTracker


class ScenarioTreeEnv(gym.Wrapper):
    """
    Wraps TexasHoldemEnv to replay hands from decision points with different deck outcomes.
    
    This reduces variance by averaging rewards across multiple runouts of the same scenario.
    Keeps Gymnasium interface clean while internally doing scenario replays.
    """
    
    def __init__(self, env, fixed_opponent, learning_agent, replay_stages=None, replays=10):
        """
        Args:
            env: Base TexasHoldemEnv
            fixed_opponent: Non-learning opponent (CallAgent, RandomAgent, etc)
            learning_agent: Agent being trained
            replay_stages: Which stages to replay from ['flop', 'turn', 'river']
            replays: How many times to replay each scenario
        """
        super().__init__(env)
        self.fixed_opponent = fixed_opponent
        self.learning_agent = learning_agent
        self.replay_stages = replay_stages or ['flop', 'turn', 'river']
        self.replays = replays
        
        # Opponent tracking
        self.opponent_tracker = OpponentTracker()
        self.learning_agent_id = 0  # Assume learning agent is player 0
        self.opponent_id = 1
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action with scenario replay if hand ends.
        
        Returns Gymnasium format: (obs, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If hand is done, replay scenario and average rewards
        if terminated:
            avg_reward = self._replay_and_average()
            reward = avg_reward
            info['replayed'] = True
            info['replay_count'] = self.replays
        
        return obs, reward, terminated, truncated, info
    
    def reset(self) -> np.ndarray:
        """Reset environment and opponent tracker"""
        self.opponent_tracker = OpponentTracker()
        return self.env.reset()
    
    def _save_decision_point(self) -> Dict[str, Any]:
        """
        Save game state at current decision point.
        Captures: hole cards, stacks, bets, community cards
        Does NOT save: deck (we want different runouts)
        """
        gs = self.env.game_state
        
        saved = {
            'hole_cards': [p.hand.copy() for p in gs.players],
            'community_cards': gs.community_cards.copy(),
            'stacks': [p.stack for p in gs.players],
            'current_bets': [p.current_bet for p in gs.players],
            'total_bets': [p.total_bet_this_hand for p in gs.players],
            'button_position': gs.button_position,
            'betting_round': gs.betting_round,
            'current_player_idx': gs.current_player_idx,
            'pot_manager_state': {
                'current_bet': gs.pot_manager.current_bet,
                'min_raise': gs.pot_manager.min_raise,
            }
        }
        return saved
    
    def _restore_state(self, saved: Dict[str, Any]):
        """Restore environment to saved state"""
        gs = self.env.game_state
        
        # Restore players
        for i, player in enumerate(gs.players):
            player.hand = saved['hole_cards'][i].copy()
            player.stack = saved['stacks'][i]
            player.current_bet = saved['current_bets'][i]
            player.total_bet_this_hand = saved['total_bets'][i]
        
        # Restore game state
        gs.community_cards = saved['community_cards'].copy()
        gs.button_position = saved['button_position']
        gs.betting_round = saved['betting_round']
        gs.current_player_idx = saved['current_player_idx']
        
        # Restore pot manager
        gs.pot_manager.current_bet = saved['pot_manager_state']['current_bet']
        gs.pot_manager.min_raise = saved['pot_manager_state']['min_raise']
    
    def _replay_from_decision_point(self, saved_state: Dict[str, Any]) -> float:
        """
        Replay scenario from decision point with fresh deck (different community cards).
        
        Returns reward from playing out the hand.
        """
        # Restore to decision point
        self._restore_state(saved_state)
        
        # Create fresh deck (this is the key - different cards each replay)
        gs = self.env.game_state
        gs.deck = Deck().cards
        random.shuffle(gs.deck)
        
        # Play out rest of hand
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            current_player = gs.get_current_player()
            obs = self.env._get_observation()
            
            # Determine which agent acts
            if current_player.player_id == self.learning_agent_id:
                action = self.learning_agent.select_action(obs)
            else:
                action = self.fixed_opponent.select_action(obs)
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            steps += 1
        
        return reward
    
    def _replay_and_average(self) -> float:
        """
        Save current decision point and replay N times with different decks.
        Average the rewards across all replays.
        
        Returns averaged reward.
        """
        # Save state at decision point
        saved = self._save_decision_point()
        
        # Replay multiple times
        rewards = []
        for i in range(self.replays):
            reward = self._replay_from_decision_point(saved)
            rewards.append(reward)
        
        # Average
        avg_reward = float(np.mean(rewards))
        
        return avg_reward
    
    def update_opponent_stats(self, action: int, info: Dict[str, Any]):
        """Update opponent tracker with action taken"""
        # This gets called by training loop to track opponent behavior
        pass
    
    def get_opponent_stats(self) -> Dict[str, float]:
        """Get current opponent statistics"""
        return self.opponent_tracker.get_stats()
    
    def get_opponent_stats_vector(self) -> np.ndarray:
        """Get opponent stats as vector for observation"""
        return self.opponent_tracker.get_stats_vector()
    
    def get_opponent_type(self) -> str:
        """Get classified opponent type"""
        return self.opponent_tracker.get_player_type()
    
    def get_exploits(self) -> Dict[str, str]:
        """Get current exploits against opponent"""
        return self.opponent_tracker.get_exploits()
    
    def print_opponent_summary(self):
        """Print summary of opponent tendencies"""
        stats = self.opponent_tracker.get_stats()
        exploits = self.opponent_tracker.get_exploits()
        
        print("\n" + "="*60)
        print(f"OPPONENT ANALYSIS: {self.opponent_tracker.get_player_type()}")
        print("="*60)
        print(f"Hands played: {stats['hands_played']}")
        print(f"VPIP: {stats['VPIP']:.1%}")
        print(f"PFR: {stats['PFR']:.1%}")
        print(f"AF: {stats['AF']:.2f}")
        print(f"Fold to 3-Bet: {stats['fold_to_3bet']:.1%}")
        print(f"C-Bet %: {stats['cbet_pct']:.1%}")
        print(f"Fold to C-Bet: {stats['fold_to_cbet']:.1%}")
        print(f"Go to Showdown: {stats['go2sd']:.1%}")
        print(f"Win at Showdown: {stats['wafsd']:.1%}")
        print("\nExploits:")
        for key, value in exploits.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")