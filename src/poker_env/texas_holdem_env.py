"""
OpenAI Gym environment with pot-based raise actions + all-in
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from src.poker_env.game_state import GameState, BettingRound
from src.poker_env.hand_evaluator import HandEvaluator


class TexasHoldemEnv(gym.Env):
    """
    Texas Hold'em with pot-based raise bins + all-in action.
    
    Action Space: Discrete(2 + len(raise_bins) + 1)
    - 0: Fold
    - 1: Check/Call  
    - 2 to N-1: Raise by bin[0], bin[1], ... (pot-based percentages)
    - N: All-in
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        num_players: int = 6,
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        rake_percent: float = 0.0,
        rake_cap: int = 0,
        min_raise_multiplier: float = 1.0,
        raise_bins: Optional[List[float]] = None,
        include_all_in: bool = True
    ):
        """
        Args:
            raise_bins: List of pot percentages (e.g., [0.5, 1.0, 2.0])
            include_all_in: If True, add all-in as last action
        """
        super().__init__()
        
        if not 2 <= num_players <= 10:
            raise ValueError("Number of players must be between 2 and 10")
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.raise_bins = raise_bins if raise_bins else [0.5, 1.0, 2.0]
        self.include_all_in = include_all_in
        
        self.game_state = GameState(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            rake_percent=rake_percent,
            rake_cap=rake_cap,
            min_raise_multiplier=min_raise_multiplier,
            raise_bins=self.raise_bins
        )
        
        # Action space: Fold, Call, + raise bins + all-in
        num_raise_actions = len(self.raise_bins)
        num_all_in = 1 if include_all_in else 0
        self.action_space = spaces.Discrete(2 + num_raise_actions + num_all_in)
        
        obs_size = 7 * 4 + 4 + 4
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.current_agent = 0
        
    def set_raise_bins(self, raise_bins: List[float]):
        """Update raise bins and action space"""
        self.raise_bins = sorted(raise_bins)
        self.game_state.pot_manager.set_raise_bins(self.raise_bins)
        num_raise_actions = len(self.raise_bins)
        num_all_in = 1 if self.include_all_in else 0
        self.action_space = spaces.Discrete(2 + num_raise_actions + num_all_in)
        
    def get_raise_bins(self) -> List[float]:
        """Get current raise bin percentages"""
        return self.raise_bins.copy()
    
    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Reset for new hand"""
        if seed is not None:
            np.random.seed(seed)
        
        for player in self.game_state.players:
            if player.stack <= 0:
                player.stack = self.starting_stack
        
        self.game_state.start_new_hand()
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action"""
        current_player = self.game_state.get_current_player()
        starting_stack = current_player.stack + current_player.total_bet_this_hand
        
        action, raise_amount = self._validate_and_convert_action(action)
        action_type = self.game_state.execute_action(action, raise_amount)
        
        if self.game_state.is_betting_round_complete():
            if not self.game_state.is_hand_complete():
                self.game_state.advance_betting_round()
        
        done = self.game_state.is_hand_complete()
        reward = 0.0
        info = {'action': action_type, 'raise_bins': self.raise_bins}
        
        if done:
            winnings = self.game_state.determine_winners()
            reward = float(current_player.stack - starting_stack)
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        terminated = done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info
    
    def _validate_and_convert_action(self, action: int) -> Tuple[int, Optional[int]]:
        """Convert raw action to (action_type, raise_amount)"""
        if action == 0:
            return 0, None
        elif action == 1:
            return 1, None
        else:
            # Check if this is all-in action
            last_action_idx = 2 + len(self.raise_bins)
            if self.include_all_in and action == last_action_idx:
                # All-in action
                player = self.game_state.get_current_player()
                return 2, player.stack  # Raise by entire stack
            
            # Otherwise it's a raise bin action
            bin_idx = action - 2
            if bin_idx >= len(self.raise_bins):
                return 1, None  # Invalid, default to call
            
            player = self.game_state.get_current_player()
            pot = self.game_state.pot_manager.get_pot_total()
            to_call = self.game_state.pot_manager.current_bet - player.current_bet
            
            raise_chips = int(pot * self.raise_bins[bin_idx])
            raise_chips = self.game_state.pot_manager._round_to_big_blind(raise_chips)
            
            if raise_chips < self.game_state.pot_manager.min_raise:
                raise_chips = self.game_state.pot_manager.min_raise
            
            if raise_chips > player.stack:
                if player.stack > to_call:
                    raise_chips = player.stack
                else:
                    return 1, None
            
            return 2, to_call + raise_chips
    
    def get_valid_actions(self) -> List[int]:
        """Get valid actions for current player"""
        player = self.game_state.get_current_player()
        pot = self.game_state.pot_manager.get_pot_total()
        to_call = self.game_state.pot_manager.current_bet - player.current_bet
        
        valid = [0, 1]
        
        for i, bin_pct in enumerate(self.raise_bins):
            raise_amt = int(pot * bin_pct)
            raise_amt = self.game_state.pot_manager._round_to_big_blind(raise_amt)
            
            if to_call + raise_amt <= player.stack + to_call:
                valid.append(2 + i)
        
        # Add all-in if available and player has chips
        if self.include_all_in and player.stack > 0:
            all_in_idx = 2 + len(self.raise_bins)
            valid.append(all_in_idx)
        
        return valid
    
    def get_action_description(self, action: int) -> str:
        """Human-readable action name"""
        if action == 0:
            return "Fold"
        elif action == 1:
            return "Check/Call"
        else:
            last_idx = 2 + len(self.raise_bins)
            if self.include_all_in and action == last_idx:
                return "All-in"
            
            idx = action - 2
            if idx < len(self.raise_bins):
                return f"Raise {self.raise_bins[idx]*100:.0f}% pot"
        return f"Action {action}"
    
    def step_with_raise(self, action: int, raise_amount: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action with custom raise amount (for humans)"""
        current_player = self.game_state.get_current_player()
        starting_stack = current_player.stack + current_player.total_bet_this_hand
        
        action_type = self.game_state.execute_action(action, raise_amount)
        
        if self.game_state.is_betting_round_complete():
            if not self.game_state.is_hand_complete():
                self.game_state.advance_betting_round()
        
        done = self.game_state.is_hand_complete()
        reward = 0.0
        info = {'action': action_type}
        
        if done:
            winnings = self.game_state.determine_winners()
            reward = float(current_player.stack - starting_stack)
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        terminated = done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector"""
        player = self.game_state.get_current_player()
        
        hole = self._encode_cards(player.hand)
        comm = self._encode_cards(
            self.game_state.community_cards + [0]*(5-len(self.game_state.community_cards))
        )
        
        stack = player.stack / self.starting_stack
        pot = self.game_state.pot_manager.get_pot_total() / self.starting_stack
        bet = player.current_bet / self.starting_stack
        call = (self.game_state.pot_manager.current_bet - player.current_bet) / self.starting_stack
        
        active = len(self.game_state.get_active_players()) / self.num_players
        pos = self.game_state.current_player_idx / self.num_players
        rnd = self.game_state.betting_round.value / 4
        btn = self.game_state.button_position / self.num_players
        
        return np.concatenate([hole, comm, [stack, pot, bet, call], [active, pos, rnd, btn]]).astype(np.float32)
    
    def _encode_cards(self, cards: list) -> np.ndarray:
        """Encode cards"""
        enc = []
        for c in cards:
            if c == 0:
                enc.extend([0, 0, 0, 0])
            else:
                r = (c >> 8) & 0xFF
                s = (c >> 12) & 0xF
                enc.extend([r/14.0, s/4.0, 1.0, 0.0])
        return np.array(enc)
    
    def render(self, mode='human'):
        """Render game state"""
        if mode != 'human':
            return
        
        print("\n" + "="*60)
        print(f"Hand #{self.game_state.hand_number} - {self.game_state.betting_round.name}")
        print("="*60)
        
        if self.game_state.community_cards:
            comm = " ".join([HandEvaluator.card_to_string(c) for c in self.game_state.community_cards])
            print(f"Community: {comm}")
        else:
            print("Community: (none yet)")
        
        print(f"Pot: ${self.game_state.pot_manager.get_pot_total()}")
        print(f"Bet: ${self.game_state.pot_manager.current_bet}, Min Raise: ${self.game_state.pot_manager.min_raise}")
        print()
        
        for i, p in enumerate(self.game_state.players):
            mk = "→ " if i == self.game_state.current_player_idx else "  "
            bn = "(BTN) " if i == self.game_state.button_position else ""
            st = ""
            
            if not p.is_active:
                st = " [FOLDED]"
            elif p.is_all_in:
                st = " [ALL-IN]"
            
            cards = " ".join([HandEvaluator.card_to_string(c) for c in p.hand]) if i == self.current_agent and p.hand else ("## ##" if p.is_active else "-- --")
            print(f"{mk}{bn}{p.name}: ${p.stack} (Bet: ${p.current_bet}) [{cards}]{st}")
        
        print("="*60 + "\n")
    
    def close(self):
        pass