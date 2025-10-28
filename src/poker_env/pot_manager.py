"""
Pot and betting management including side pots
"""

from typing import List, Dict, Tuple
from src.poker_env.player import Player


class Pot:
    """Represents a single pot (main or side pot)"""
    
    def __init__(self):
        self.amount = 0
        self.eligible_players: List[int] = []  # Player IDs eligible to win this pot
        
    def add_chips(self, amount: int):
        """Add chips to the pot"""
        self.amount += amount
        
    def __repr__(self):
        return f"Pot(amount={self.amount}, eligible={len(self.eligible_players)})"


class PotManager:
    """
    Manages betting, pots, and side pots for a poker game
    """
    
    def __init__(self, small_blind: int, big_blind: int, rake_percent: float = 0.0, rake_cap: int = 0):
        """
        Initialize the pot manager
        
        Args:
            small_blind: Small blind amount
            big_blind: Big blind amount
            rake_percent: Percentage of pot to take as rake (0.0 to 1.0)
            rake_cap: Maximum rake amount per hand
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rake_percent = rake_percent
        self.rake_cap = rake_cap
        
        self.pots: List[Pot] = []
        self.current_bet = 0
        self.min_raise = big_blind
        
    def start_new_hand(self):
        """Reset pot manager for a new hand"""
        self.pots = [Pot()]
        self.current_bet = 0
        self.min_raise = self.big_blind
        
    def post_blinds(self, small_blind_player: Player, big_blind_player: Player):
        """
        Post small and big blinds
        
        Args:
            small_blind_player: Player in small blind position
            big_blind_player: Player in big blind position
        """
        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)
        
        self.pots[0].add_chips(sb_amount + bb_amount)
        self.current_bet = self.big_blind
        
    def place_bet(self, player: Player, amount: int) -> Tuple[int, str]:
        """
        Place a bet for a player
        
        Args:
            player: The player making the bet
            amount: Amount to bet
            
        Returns:
            Tuple of (actual_amount_bet, action_type)
            action_type can be: "fold", "check", "call", "raise", "all-in"
        """
        if amount == 0 and self.current_bet == player.current_bet:
            return 0, "check"
        
        if amount < self.current_bet - player.current_bet:
            # Not enough to call, must be folding
            player.fold()
            return 0, "fold"
        
        # Calculate how much more the player needs to put in
        to_call = self.current_bet - player.current_bet
        actual_bet = player.bet(amount)
        
        # Add to pot
        self.pots[0].add_chips(actual_bet)
        
        if player.is_all_in:
            action = "all-in"
        elif actual_bet == to_call:
            action = "call"
        elif actual_bet > to_call:
            # This is a raise
            raise_amount = actual_bet - to_call
            self.current_bet = player.current_bet
            self.min_raise = max(self.min_raise, raise_amount)
            action = "raise"
        else:
            action = "call"  # Partial call (shouldn't happen in proper play)
            
        return actual_bet, action
    
    def start_new_betting_round(self, players: List[Player]):
        """
        Start a new betting round (flop, turn, or river)
        
        Args:
            players: List of all players
        """
        # Reset current bets for all players
        for player in players:
            player.reset_current_bet()
        
        self.current_bet = 0
        self.min_raise = self.big_blind
        
    def calculate_side_pots(self, players: List[Player]) -> List[Pot]:
        """
        Calculate main pot and side pots
        
        Args:
            players: List of all players
            
        Returns:
            List of pots (main pot + side pots)
        """
        # Get all players who contributed to the pot
        contributing_players = [p for p in players if p.total_bet_this_hand > 0]
        
        if not contributing_players:
            return [Pot()]
        
        # Sort players by total bet amount
        contributing_players.sort(key=lambda p: p.total_bet_this_hand)
        
        pots = []
        remaining_players = contributing_players.copy()
        previous_bet_level = 0
        
        while remaining_players:
            # Get the smallest bet level among remaining players
            min_bet = remaining_players[0].total_bet_this_hand
            
            # Create a pot for this level
            pot = Pot()
            pot_contribution = min_bet - previous_bet_level
            
            # Each remaining player contributes to this pot
            for player in remaining_players:
                pot.add_chips(pot_contribution)
                pot.eligible_players.append(player.player_id)
            
            pots.append(pot)
            
            # Remove players who are all-in at this level
            remaining_players = [p for p in remaining_players if p.total_bet_this_hand > min_bet]
            previous_bet_level = min_bet
        
        return pots
    
    def distribute_pots(self, players: List[Player], hand_ranks: Dict[int, int]) -> Dict[int, int]:
        """
        Distribute pots to winners, applying rake if configured
        
        Args:
            players: List of all players
            hand_ranks: Dictionary mapping player_id to hand rank (lower is better)
            
        Returns:
            Dictionary mapping player_id to amount won
        """
        # Calculate side pots
        pots = self.calculate_side_pots(players)
        
        winnings: Dict[int, int] = {p.player_id: 0 for p in players}
        
        for pot in pots:
            # Find eligible players for this pot
            eligible_ranks = {
                pid: hand_ranks.get(pid, 9999) 
                for pid in pot.eligible_players 
                if pid in hand_ranks
            }
            
            if not eligible_ranks:
                continue
            
            # Find winner(s) - lowest rank wins
            best_rank = min(eligible_ranks.values())
            winners = [pid for pid, rank in eligible_ranks.items() if rank == best_rank]
            
            # Apply rake before distributing
            pot_after_rake = pot.amount
            if self.rake_percent > 0 and len(eligible_ranks) > 1:  # Only rake multi-way pots
                rake = min(int(pot.amount * self.rake_percent), self.rake_cap)
                pot_after_rake -= rake
            
            # Split pot among winners
            amount_per_winner = pot_after_rake // len(winners)
            remainder = pot_after_rake % len(winners)
            
            for i, winner_id in enumerate(winners):
                winnings[winner_id] += amount_per_winner
                if i < remainder:  # Distribute remainder chips
                    winnings[winner_id] += 1
        
        return winnings
    
    def get_pot_total(self) -> int:
        """Get total amount in all pots"""
        return sum(pot.amount for pot in self.pots)
    
    def __repr__(self):
        return f"PotManager(pots={len(self.pots)}, total={self.get_pot_total()}, current_bet={self.current_bet})"