"""
Player class representing a poker player at the table
"""

from typing import Optional, List


class Player:
    """
    Represents a player in the poker game
    """
    
    def __init__(self, player_id: int, stack: int, name: Optional[str] = None):
        """
        Initialize a player
        
        Args:
            player_id: Unique identifier for the player
            stack: Starting chip stack
            name: Optional player name
        """
        self.player_id = player_id
        self.name = name or f"Player_{player_id}"
        self.stack = stack
        self.hand: List[int] = []  # Cards in hand (using treys card representation)
        self.current_bet = 0
        self.total_bet_this_hand = 0
        self.is_active = True  # Still in the hand
        self.is_all_in = False
        self.is_sitting_out = False  # Temporarily not playing
        
    def deal_hand(self, cards: List[int]):
        """Deal hole cards to the player"""
        self.hand = cards
        
    def bet(self, amount: int) -> int:
        """
        Player makes a bet
        
        Args:
            amount: Amount to bet
            
        Returns:
            Actual amount bet (may be less if all-in)
        """
        actual_bet = min(amount, self.stack)
        self.stack -= actual_bet
        self.current_bet += actual_bet
        self.total_bet_this_hand += actual_bet
        
        if self.stack == 0:
            self.is_all_in = True
            
        return actual_bet
    
    def add_chips(self, amount: int):
        """Add chips to player's stack"""
        self.stack += amount
        
    def fold(self):
        """Player folds their hand"""
        self.is_active = False
        
    def reset_for_new_hand(self):
        """Reset player state for a new hand"""
        self.hand = []
        self.current_bet = 0
        self.total_bet_this_hand = 0
        self.is_active = True if not self.is_sitting_out else False
        self.is_all_in = False
        
    def reset_current_bet(self):
        """Reset current bet for new betting round"""
        self.current_bet = 0
        
    def can_act(self) -> bool:
        """Check if player can take an action"""
        return self.is_active and not self.is_all_in
    
    def __repr__(self):
        return (f"Player({self.name}, stack={self.stack}, "
                f"bet={self.current_bet}, active={self.is_active})")