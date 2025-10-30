"""
Game state management for Texas Hold'em
"""

from enum import Enum
from typing import List, Optional
import random
from treys import Card, Deck

from src.poker_env.player import Player
from src.poker_env.pot_manager import PotManager
from src.poker_env.hand_evaluator import HandEvaluator


class BettingRound(Enum):
    """Enum for different betting rounds"""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4


class GameState:
    """Manages the complete state of a poker game"""
    
    def __init__(
        self,
        num_players: int,
        starting_stack: int,
        small_blind: int,
        big_blind: int,
        rake_percent: float = 0.0,
        rake_cap: int = 0,
        min_raise_multiplier: float = 1.0,
        raise_bins: list = None
    ):
        """
        Initialize game state
        
        Args:
            num_players: Number of players (2-10)
            starting_stack: Starting chip stack
            small_blind: Small blind amount
            big_blind: Big blind amount
            rake_percent: Rake percentage (0.0 to 1.0)
            rake_cap: Maximum rake per hand
            min_raise_multiplier: Multiplier for minimum raise
            raise_bins: List of pot percentages for raise sizes
        """
        if not 2 <= num_players <= 10:
            raise ValueError("Number of players must be between 2 and 10")
            
        self.players = [
            Player(player_id=i, stack=starting_stack)
            for i in range(num_players)
        ]
        
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.min_raise_multiplier = min_raise_multiplier
        
        self.pot_manager = PotManager(
            small_blind,
            big_blind,
            rake_percent,
            rake_cap,
            min_raise_multiplier,
            raise_bins
        )
        self.hand_evaluator = HandEvaluator()
        
        self.deck: List[int] = []
        self.community_cards: List[int] = []
        self.button_position = 0
        self.current_player_idx = 0
        self.betting_round = BettingRound.PREFLOP
        self.hand_number = 0
        
        self.last_aggressor_idx: Optional[int] = None
        self.num_actions_this_round = 0
        
    def start_new_hand(self):
        """Start a new hand"""
        self.hand_number += 1
        
        for player in self.players:
            player.reset_for_new_hand()
        
        active_players = [p for p in self.players if p.stack > 0]
        if len(active_players) < 2:
            raise ValueError("Not enough players with chips to start a hand")
        
        self.deck = Deck().cards
        random.shuffle(self.deck)
        self.community_cards = []
        
        self.pot_manager.start_new_hand()
        
        for player in self.players:
            if player.stack > 0:
                player.deal_hand([self.deck.pop(), self.deck.pop()])
        
        self.button_position = (self.button_position + 1) % len(self.players)
        
        sb_idx = self._get_next_active_player(self.button_position)
        bb_idx = self._get_next_active_player(sb_idx)
        
        self.pot_manager.post_blinds(
            self.players[sb_idx],
            self.players[bb_idx]
        )
        
        self.current_player_idx = self._get_next_active_player(bb_idx)
        self.betting_round = BettingRound.PREFLOP
        self.last_aggressor_idx = bb_idx
        self.num_actions_this_round = 0
        
    def _get_next_active_player(self, start_idx: int) -> int:
        """Get the next active player who can act"""
        idx = (start_idx + 1) % len(self.players)
        checked_count = 0
        
        while checked_count < len(self.players):
            if self.players[idx].can_act():
                return idx
            idx = (idx + 1) % len(self.players)
            checked_count += 1
            
        return start_idx
    
    def get_current_player(self) -> Player:
        """Get the player whose turn it is"""
        return self.players[self.current_player_idx]
    
    def get_active_players(self) -> List[Player]:
        """Get all players still active in the hand"""
        return [p for p in self.players if p.is_active]
    
    def is_betting_round_complete(self) -> bool:
        """Check if the current betting round is complete"""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return True
        
        players_who_can_act = [p for p in active_players if not p.is_all_in]
        
        if not players_who_can_act:
            return True
        
        if self.num_actions_this_round < len(players_who_can_act):
            return False
        
        current_bet = self.pot_manager.current_bet
        for player in players_who_can_act:
            if player.current_bet != current_bet:
                return False
        
        return True
    
    def advance_betting_round(self):
        """Move to the next betting round"""
        if self.betting_round == BettingRound.PREFLOP:
            self._burn_card()
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
            self.betting_round = BettingRound.FLOP
            
        elif self.betting_round == BettingRound.FLOP:
            self._burn_card()
            self.community_cards.append(self.deck.pop())
            self.betting_round = BettingRound.TURN
            
        elif self.betting_round == BettingRound.TURN:
            self._burn_card()
            self.community_cards.append(self.deck.pop())
            self.betting_round = BettingRound.RIVER
            
        elif self.betting_round == BettingRound.RIVER:
            self.betting_round = BettingRound.SHOWDOWN
            
        self.pot_manager.start_new_betting_round(self.players)
        self.current_player_idx = self._get_next_active_player(self.button_position)
        self.last_aggressor_idx = None
        self.num_actions_this_round = 0
    
    def _burn_card(self):
        """Burn a card from the deck"""
        if self.deck:
            self.deck.pop()
    
    def execute_action(self, action: int, raise_amount: Optional[int] = None):
        """
        Execute a player action
        
        Args:
            action: Action code (0=fold, 1=check/call, 2=raise)
            raise_amount: Amount to raise (if action is raise)
        """
        player = self.get_current_player()
        
        if action == 0:
            player.fold()
            action_type = "fold"
            
        elif action == 1:
            to_call = self.pot_manager.current_bet - player.current_bet
            _, action_type = self.pot_manager.place_bet(player, to_call)
            
        elif action == 2:
            if raise_amount is None:
                raise_amount = self.pot_manager.min_raise
            
            to_call = self.pot_manager.current_bet - player.current_bet
            total_bet = to_call + raise_amount
            _, action_type = self.pot_manager.place_bet(player, total_bet)
            
            if action_type == "raise":
                self.last_aggressor_idx = self.current_player_idx
        
        self.num_actions_this_round += 1
        self.current_player_idx = self._get_next_active_player(self.current_player_idx)
        
        return action_type
    
    def determine_winners(self) -> dict:
        """Determine winners and distribute pots"""
        active_players = self.get_active_players()
        
        hand_ranks = {}
        for player in active_players:
            if player.hand:
                rank = self.hand_evaluator.evaluate_hand(player.hand, self.community_cards)
                hand_ranks[player.player_id] = rank
        
        winnings = self.pot_manager.distribute_pots(self.players, hand_ranks)
        
        for player_id, amount in winnings.items():
            self.players[player_id].add_chips(amount)
        
        return winnings
    
    def is_hand_complete(self) -> bool:
        """Check if the hand is complete"""
        active_players = self.get_active_players()
        
        if len(active_players) <= 1:
            return True
        
        if self.betting_round == BettingRound.SHOWDOWN:
            return True
        
        return False
    
    def add_player(self, stack: int) -> int:
        """Add a new player to the game"""
        player_id = len(self.players)
        self.players.append(Player(player_id=player_id, stack=stack))
        return player_id
    
    def remove_player(self, player_id: int):
        """Remove a player from the game"""
        if 0 <= player_id < len(self.players):
            self.players[player_id].is_sitting_out = True
            self.players[player_id].is_active = False