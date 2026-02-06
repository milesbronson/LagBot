"""
OpenAI Gym environment with pot-based raise actions + all-in + opponent tracking
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List

from src.poker_env.game_state import GameState, BettingRound
from src.poker_env.hand_evaluator import HandEvaluator
from src.poker_env.opponent_tracker import OpponentTracker, Action, Street


class TexasHoldemEnv(gym.Env):
    """
    Texas Hold'em with pot-based raise bins + all-in action.
    
    Action Space: Discrete(2 + len(raise_bins) + 1)
    - 0: Fold
    - 1: Check/Call  
    - 2 to N-1: Raise by bin[0], bin[1], ... (pot-based percentages)
    - N: All-in
    
    Observation Space: 32 base dims + 36 opponent stats (9 opponents × 4 features)
    """
    
    metadata = {'render.modes': ['human']}
    
    # Opponent tracking constants
    MAX_OPPONENTS = 9
    FEATURES_PER_OPPONENT = 8  # VPIP, PFR, AF, 3bet%, cbet%, fold_to_cbet%, showdown%, confidence
    
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
        include_all_in: bool = True,
        reset_stacks_every_n_timesteps: Optional[int] = None,
        track_opponents: bool = True
    ):
        """
        Args:
            raise_bins: List of pot percentages (e.g., [0.5, 1.0, 2.0])
            include_all_in: If True, add all-in as last action
            track_opponents: If True, include opponent stats in observation (68 dims vs 32)
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
        self.reset_stacks_every_n_timesteps = reset_stacks_every_n_timesteps
        self.timesteps_since_reset = 0
        self.total_timesteps = 0
        self.track_opponents = track_opponents
        
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
        
        # Observation space: base (36) + opponent stats (72 if tracking, was 36)
        base_obs_size = 7 * 4 + 4 + 4  # 36 (7 cards × 4 features + 8 game state)
        opponent_obs_size = self.MAX_OPPONENTS * self.FEATURES_PER_OPPONENT if track_opponents else 0  # 9 × 8 = 72
        obs_size = base_obs_size + opponent_obs_size  # 36 + 72 = 108 total
        
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.learning_agent_id = 0
        self.opponent_tracker = OpponentTracker(max_history_hands=1000)
        self.player_positions = {}
    
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
                player.record_buy_in(self.starting_stack)  
                player.stack = self.starting_stack
        
        self.game_state.start_new_hand()
        
        # Start opponent tracking for this hand
        players = [{'id': p.player_id, 'name': p.name, 'stack': p.stack} 
                   for p in self.game_state.players]
        self.opponent_tracker.start_hand(
            hand_number=self.game_state.hand_number,
            players=players,
            dealer_position=self.game_state.button_position,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        self._calculate_player_positions()
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action"""
        current_player = self.game_state.get_current_player()
        starting_stack = current_player.stack + current_player.total_bet_this_hand
        
        # Store pre-action state for tracking
        stack_before = current_player.stack + current_player.total_bet_this_hand
        pot_before = self.game_state.pot_manager.get_pot_total()
        street_before = self._betting_round_to_street(self.game_state.betting_round)
        position = self.game_state.current_player_idx
        
        action_int, raise_amount = self._validate_and_convert_action(action)
        action_type_str = self.game_state.execute_action(action_int, raise_amount)
        
        # Record action with opponent tracker
        action_enum = self._string_to_action_enum(action_type_str)
        action_amount = self._calculate_action_amount(current_player, action_int, raise_amount)
        
        self.opponent_tracker.record_action(
            player_id=current_player.player_id,
            player_name=current_player.name,
            action=action_enum,
            amount=action_amount,
            pot_size=pot_before,
            stack_before=stack_before,
            stack_after=current_player.stack,
            street=street_before,
            position=position
        )

        if self.game_state.is_betting_round_complete():
            if not self.game_state.is_hand_complete():
                self.game_state.advance_betting_round()
        
        done = self.game_state.is_hand_complete()
        reward = 0.0
        info = {'action': action_type_str, 'raise_bins': self.raise_bins}
        
        # Track timesteps and reset stacks when limit hit
        self.timesteps_since_reset += 1
        self.total_timesteps += 1
        
        if self.reset_stacks_every_n_timesteps is not None:
            if self.timesteps_since_reset >= self.reset_stacks_every_n_timesteps:
                for player in self.game_state.players:
                    player.stack = self.starting_stack
                self.timesteps_since_reset = 0
                print(f"[RESET] Timestep {self.total_timesteps}")
        
        if done:
            winnings = self.game_state.determine_winners()
            # CRITICAL FIX: Always calculate reward for the agent being trained (player 0)
            # not for whoever's turn it was when the hand ended
            learning_agent = self.game_state.players[self.learning_agent_id]
            agent_starting_stack = learning_agent.starting_stack_this_hand
            reward = (learning_agent.stack - agent_starting_stack) / self.game_state.big_blind
            final_stacks = {p.player_id: p.stack for p in self.game_state.players}
            
            # End hand tracking
            winner_ids = [pid for pid, amt in winnings.items() if amt > 0]
            self.opponent_tracker.end_hand(
                winners=winner_ids,
                winnings=winnings,
                final_stacks=final_stacks
            )
            
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        terminated = done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info
    
    def _calculate_player_positions(self):
        """Record player positions (0=dealer, 1=SB, 2=BB, etc)"""
        positions = {}
        for i, player in enumerate(self.game_state.players):
            positions[player.player_id] = i
        self.opponent_tracker.record_positions(positions)
    
    def _betting_round_to_street(self, betting_round: BettingRound) -> Street:
        """Convert BettingRound enum to Street enum"""
        mapping = {
            BettingRound.PREFLOP: Street.PREFLOP,
            BettingRound.FLOP: Street.FLOP,
            BettingRound.TURN: Street.TURN,
            BettingRound.RIVER: Street.RIVER,
            BettingRound.SHOWDOWN: Street.RIVER,
        }
        return mapping.get(betting_round, Street.PREFLOP)
    
    def _string_to_action_enum(self, action_str: str) -> Action:
        """Convert action string to Action enum"""
        action_str_lower = action_str.lower()
        if 'fold' in action_str_lower:
            return Action.FOLD
        elif 'check' in action_str_lower:
            return Action.CHECK
        elif 'call' in action_str_lower:
            return Action.CALL
        elif 'all' in action_str_lower or 'all-in' in action_str_lower:
            return Action.ALL_IN
        elif 'raise' in action_str_lower:
            return Action.RAISE
        elif 'bet' in action_str_lower:
            return Action.BET
        else:
            return Action.CHECK

    def _validate_and_convert_action(self, action: int) -> Tuple[int, Optional[int]]:
        """Convert raw action to (action_type, raise_amount)

        Returns:
            (action_type, total_bet_amount) where:
            - action_type: 0=fold, 1=call, 2=raise
            - total_bet_amount: for raises, the TOTAL amount to bet (to_call + raise_chips)
        """
        if action == 0:
            return 0, None
        elif action == 1:
            return 1, None
        else:
            # Check if this is all-in action
            last_action_idx = 2 + len(self.raise_bins)
            if self.include_all_in and action == last_action_idx:
                player = self.game_state.get_current_player()
                return 2, player.stack

            # Otherwise it's a raise bin action
            bin_idx = action - 2
            if bin_idx >= len(self.raise_bins):
                return 1, None  # Invalid, default to call

            player = self.game_state.get_current_player()
            pot = self.game_state.pot_manager.get_pot_total()
            to_call = self.game_state.pot_manager.current_bet - player.current_bet

            # Calculate raise amount (just the raise portion, not including to_call)
            raise_chips = int(pot * self.raise_bins[bin_idx])
            raise_chips = self.game_state.pot_manager._round_to_big_blind(raise_chips)
            if raise_chips < self.game_state.pot_manager.min_raise:
                raise_chips = self.game_state.pot_manager.min_raise

            # BUG FIX: Check if player has enough for to_call + raise_chips
            total_needed = to_call + raise_chips

            if total_needed > player.stack:
                # Player doesn't have enough for this raise size
                if player.stack > to_call:
                    # Player can go all-in (which is more than a call)
                    return 2, player.stack
                else:
                    # Player can only afford to call
                    return 1, None

            return 2, total_needed
    
    def _calculate_action_amount(self, current_player, action_type: int, raise_amount: Optional[int]) -> int:
        """Calculate actual amount contributed in this action

        NOTE: This is called BEFORE the action is executed, so we estimate the amount.
        For raises, raise_amount is the TOTAL bet amount (to_call + raise_chips).
        """
        to_call = self.game_state.pot_manager.current_bet - current_player.current_bet

        if action_type == 0:  # Fold
            return 0
        elif action_type == 1:  # Call/Check
            return to_call
        else:  # Raise (action_type == 2)
            # raise_amount is the total bet amount, limited by stack
            if raise_amount is None:
                return self.game_state.pot_manager.min_raise
            # Return the actual chips that will go into pot (capped by player stack)
            return min(raise_amount, current_player.stack)

    def get_valid_actions(self) -> List[int]:
        """Get valid actions for current player"""
        player = self.game_state.get_current_player()
        pot = self.game_state.pot_manager.get_pot_total()
        to_call = self.game_state.pot_manager.current_bet - player.current_bet

        valid = [0, 1]

        for i, bin_pct in enumerate(self.raise_bins):
            raise_amt = int(pot * bin_pct)
            raise_amt = self.game_state.pot_manager._round_to_big_blind(raise_amt)

            # Check if player has enough chips for to_call + raise_amt
            # BUG FIX: Was "to_call + raise_amt <= player.stack + to_call" which is wrong
            if to_call + raise_amt <= player.stack:
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
    
    def step_with_raise(self, action: int, raise_amount: Optional[int] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
            # FIX: Calculate reward for the agent being trained (player 0)
            learning_agent = self.game_state.players[self.learning_agent_id]
            agent_starting_stack = learning_agent.starting_stack_this_hand
            reward = float(learning_agent.stack - agent_starting_stack)
            info['winnings'] = winnings
            info['hand_complete'] = True
        
        terminated = done
        truncated = False
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector (32 base + 36 opponent stats if tracking)"""
        player = self.game_state.get_current_player()
        
        # Base observation (32 dims)
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
        
        base_obs = np.concatenate([hole, comm, [stack, pot, bet, call], [active, pos, rnd, btn]]).astype(np.float32)
        
        # Add opponent stats if tracking enabled
        if self.track_opponents:
            opponent_features = self._get_opponent_features(player.player_id)
            return np.concatenate([base_obs, opponent_features])
        
        return base_obs
    
    def _get_opponent_features(self, hero_id: int) -> np.ndarray:
        """Get opponent stats for observation space (36 dims: 9 opponents × 4 features)"""
        opponent_ids = [p.player_id for p in self.game_state.players if p.player_id != hero_id]
        
        features = self.opponent_tracker.get_observation_features(
            hero_id=hero_id,
            opponent_ids=opponent_ids,
            max_opponents=self.MAX_OPPONENTS,
            features_per_opponent=self.FEATURES_PER_OPPONENT
        )
        
        return np.array(features, dtype=np.float32)
    
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
            
            cards = " ".join([HandEvaluator.card_to_string(c) for c in p.hand]) if i == self.learning_agent_id and p.hand else ("## ##" if p.is_active else "-- --")
            print(f"{mk}{bn}{p.name}: ${p.stack} (Bet: ${p.current_bet}) [{cards}]{st}")
        
        # Show opponent stats if available
        if self.track_opponents:
            stats = self.opponent_tracker.get_all_opponent_stats()
            if any(s.get('hands_played', 0) > 0 for s in stats.values() if s):
                print("\n📊 Opponent Stats:")
                for pid, s in stats.items():
                    if s and s.get('hands_played', 0) > 0:
                        print(f"  P{pid}: VPIP={s['vpip']:.1%} PFR={s['pfr']:.1%} AF={s['af']:.2f}")
        
        print("="*60 + "\n")
    
    def close(self):
        pass