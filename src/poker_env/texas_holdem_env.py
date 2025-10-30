"""
OpenAI Gym environment for Texas Hold'em Poker
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional

from src.poker_env.game_state import GameState, BettingRound
from src.poker_env.hand_evaluator import HandEvaluator


class TexasHoldemEnv(gym.Env):
    """
    Texas Hold'em Poker environment compatible with OpenAI Gym
    
    Observation Space:
        - Player's hole cards (2 cards encoded)
        - Community cards (5 cards encoded, padded with zeros)
        - Player's stack
        - Player's current bet
        - Pot size
        - Current bet to call
        - Number of active players
        - Position information
        - Betting round
        
    Action Space:
        Discrete(3):
            0: Fold
            1: Check/Call
            2: Raise (minimum raise)
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
        min_raise_multiplier: float = 1.0
    ):
        """
        Initialize the Texas Hold'em environment
        
        Args:
            num_players: Number of players at the table (2-10)
            starting_stack: Starting chip stack
            small_blind: Small blind amount
            big_blind: Big blind amount
            rake_percent: Rake percentage (0.0 to 1.0)
            rake_cap: Maximum rake per hand
            min_raise_multiplier: Multiplier for minimum raise (e.g., 2.0 for 2x rule)
        """
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Initialize game state
        self.game_state = GameState(
            num_players=num_players,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            rake_percent=rake_percent,
            rake_cap=rake_cap,
            min_raise_multiplier=min_raise_multiplier
        )
        
        # Define action space: [Fold, Check/Call, Raise]
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # Cards: 2 hole + 5 community = 7 cards * 4 (suit encoding)
        # Stack info: player stack, pot size, current bet, call amount
        # Game info: num active players, position, betting round, button position
        obs_size = (
            7 * 4 +  # Cards (one-hot encoded by suit/rank)
            4 +      # Stack information
            4        # Game information
        )
        
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.current_agent = 0  # Which agent is learning (0 by default)
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new hand
        
        Returns:
            Initial observation
        """
        # Check if any players are out of chips
        for player in self.game_state.players:
            if player.stack <= 0:
                player.stack = self.starting_stack
        
        self.game_state.start_new_hand()
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute an action in the environment
        
        Args:
            action: Action to take (0=fold, 1=check/call, 2=raise)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        current_player = self.game_state.get_current_player()
        starting_stack = current_player.stack + current_player.total_bet_this_hand
        
        # Validate action
        action = self._validate_action(action)
        
        # Execute action
        action_type = self.game_state.execute_action(action)
        
        # Check if betting round is complete
        if self.game_state.is_betting_round_complete():
            if not self.game_state.is_hand_complete():
                self.game_state.advance_betting_round()
        
        # Check if hand is complete
        done = self.game_state.is_hand_complete()
        
        # Calculate reward
        reward = 0.0
        info = {'action': action_type}
        
        if done:
            # Determine winners and distribute chips
            winnings = self.game_state.determine_winners()
            
            # Reward is change in stack
            final_stack = current_player.stack
            reward = float(final_stack - starting_stack)
            
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def step_with_raise(self, action: int, raise_amount: Optional[int] = None) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute an action with custom raise amount
        
        Args:
            action: Action to take (0=fold, 1=check/call, 2=raise)
            raise_amount: Custom raise amount (if None, uses minimum)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        current_player = self.game_state.get_current_player()
        starting_stack = current_player.stack + current_player.total_bet_this_hand
        
        # Validate action
        action = self._validate_action(action)
        
        # Execute action with custom raise amount
        action_type = self.game_state.execute_action(action, raise_amount)
        
        # Check if betting round is complete
        if self.game_state.is_betting_round_complete():
            if not self.game_state.is_hand_complete():
                self.game_state.advance_betting_round()
        
        # Check if hand is complete
        done = self.game_state.is_hand_complete()
        
        # Calculate reward
        reward = 0.0
        info = {'action': action_type}
        
        if done:
            # Determine winners and distribute chips
            winnings = self.game_state.determine_winners()
            
            # Reward is change in stack
            final_stack = current_player.stack
            reward = float(final_stack - starting_stack)
            
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _validate_action(self, action: int) -> int:
        """
        Validate and potentially modify an action
        
        Args:
            action: Proposed action
            
        Returns:
            Valid action
        """
        current_player = self.game_state.get_current_player()
        to_call = self.game_state.pot_manager.current_bet - current_player.current_bet
        
        # If player wants to raise but doesn't have enough chips, convert to call
        if action == 2:  # Raise
            min_raise = self.game_state.pot_manager.min_raise
            if current_player.stack < to_call + min_raise:
                action = 1  # Convert to call/all-in
        
        # If current bet is 0, can't fold (should check instead)
        if action == 0 and to_call == 0:
            action = 1  # Convert to check
        
        return action
    
    def get_valid_raise_range(self) -> Tuple[int, int]:
        """
        Get valid raise range for current player
        
        Returns:
            Tuple of (min_raise, max_raise)
        """
        current_player = self.game_state.get_current_player()
        return self.game_state.pot_manager.get_valid_raise_range(current_player)
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation
        
        Returns:
            Observation array
        """
        current_player = self.game_state.get_current_player()
        
        # Encode cards
        hole_cards_encoded = self._encode_cards(current_player.hand)
        community_cards_encoded = self._encode_cards(
            self.game_state.community_cards + [0] * (5 - len(self.game_state.community_cards))
        )
        
        # Stack information (normalized)
        player_stack = current_player.stack / self.starting_stack
        pot_size = self.game_state.pot_manager.get_pot_total() / self.starting_stack
        current_bet = current_player.current_bet / self.starting_stack
        to_call = (self.game_state.pot_manager.current_bet - current_player.current_bet) / self.starting_stack
        
        # Game information
        num_active = len(self.game_state.get_active_players()) / self.num_players
        position = self.game_state.current_player_idx / self.num_players
        betting_round = self.game_state.betting_round.value / 4  # Normalize to 0-1
        button_pos = self.game_state.button_position / self.num_players
        
        # Combine all features
        observation = np.concatenate([
            hole_cards_encoded,
            community_cards_encoded,
            [player_stack, pot_size, current_bet, to_call],
            [num_active, position, betting_round, button_pos]
        ]).astype(np.float32)
        
        return observation
    
    def _encode_cards(self, cards: list) -> np.ndarray:
        """
        Encode cards as numerical features
        
        Args:
            cards: List of card integers (treys format)
            
        Returns:
            Encoded card features
        """
        encoded = []
        
        for card in cards:
            if card == 0:  # Empty card slot
                encoded.extend([0, 0, 0, 0])
            else:
                # Extract rank and suit from treys card format
                # Treys uses bit representation: rank in bits 0-3, suit in bits 12-15
                rank = (card >> 8) & 0xFF  # Simplified rank extraction
                suit = (card >> 12) & 0xF
                
                # Normalize rank (2-14) and suit (0-3)
                rank_norm = rank / 14.0
                suit_norm = suit / 4.0
                
                # Create simple encoding
                encoded.extend([rank_norm, suit_norm, 1.0, 0.0])  # Card present indicator
        
        return np.array(encoded)
    
    def render(self, mode='human'):
        """
        Render the current game state
        
        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            return
        
        print("\n" + "="*60)
        print(f"Hand #{self.game_state.hand_number} - {self.game_state.betting_round.name}")
        print("="*60)
        
        # Show community cards
        if self.game_state.community_cards:
            community_str = " ".join([
                HandEvaluator.card_to_string(c) 
                for c in self.game_state.community_cards
            ])
            print(f"Community Cards: {community_str}")
        else:
            print("Community Cards: (none yet)")
        
        print(f"Pot: ${self.game_state.pot_manager.get_pot_total()}")
        print(f"Current Bet: ${self.game_state.pot_manager.current_bet}")
        print(f"Min Raise: ${self.game_state.pot_manager.min_raise}")
        print()
        
        # Show all players
        for i, player in enumerate(self.game_state.players):
            marker = "→ " if i == self.game_state.current_player_idx else "  "
            button = "(BTN) " if i == self.game_state.button_position else ""
            status = ""
            
            if not player.is_active:
                status = " [FOLDED]"
            elif player.is_all_in:
                status = " [ALL-IN]"
            
            # Only show hole cards for current player
            if i == self.current_agent and player.hand:
                cards_str = " ".join([HandEvaluator.card_to_string(c) for c in player.hand])
            else:
                cards_str = "## ##" if player.is_active else "-- --"
            
            print(f"{marker}{button}{player.name}: ${player.stack} (Bet: ${player.current_bet}) [{cards_str}]{status}")
        
        print("="*60 + "\n")
    
    def close(self):
        """Clean up environment resources"""
        pass


# Example usage
if __name__ == "__main__":
    # Create environment
    env = TexasHoldemEnv(num_players=3, min_raise_multiplier=2.0)
    
    # Reset environment
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Render initial state
    env.render()
    
    # Play a few random actions
    done = False
    step_count = 0
    
    while not done and step_count < 20:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        print(f"\nStep {step_count + 1}: Action={action}, Reward={reward:.2f}")
        env.render()
        
        step_count += 1
    
    print(f"\nHand complete! Final info: {info}")