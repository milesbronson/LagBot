"""
Pot and betting management with pot-based raise bins + all-in action
"""

from typing import List, Dict, Tuple
from src.poker_env.player import Player


class Pot:
    """Represents a single pot (main or side pot)"""
    
    def __init__(self):
        self.amount = 0
        self.eligible_players: List[int] = []
        
    def add_chips(self, amount: int):
        """Add chips to the pot"""
        self.amount += amount
        
    def __repr__(self):
        return f"Pot(amount={self.amount}, eligible={len(self.eligible_players)})"


class PotManager:
    """Manages betting, pots, side pots, and raise bin calculations"""
    
    def __init__(self, small_blind: int, big_blind: int, rake_percent: float = 0.0, 
                 rake_cap: int = 0, min_raise_multiplier: float = 1.0, 
                 raise_bins: List[float] = None, include_all_in: bool = True):
        """
        Args:
            raise_bins: List of pot percentages for raise sizes (e.g., [0.5, 1.0, 2.0])
            include_all_in: If True, always include all-in as an option
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.rake_percent = rake_percent
        self.rake_cap = rake_cap
        self.min_raise_multiplier = min_raise_multiplier
        self.raise_bins = sorted(raise_bins if raise_bins else [0.5, 1.0, 2.0])
        self.include_all_in = include_all_in
        
        self.pots: List[Pot] = []
        self.current_bet = 0
        self.min_raise = big_blind
        self.last_raise_amount = 0
        
    def set_raise_bins(self, raise_bins: List[float]):
        """Update raise bin percentages"""
        self.raise_bins = sorted(raise_bins)
        
    def get_raise_bins(self) -> List[float]:
        """Get current raise bin percentages"""
        return self.raise_bins.copy()
        
    def calculate_raise_amounts(self, player: 'Player', current_pot: int) -> List[int]:
        """Calculate actual raise amounts based on pot percentages"""
        raise_amounts = []
        
        for bin_percent in self.raise_bins:
            raise_amount = int(current_pot * bin_percent)
            raise_amount = self._round_to_big_blind(raise_amount)
            
            if raise_amount < self.min_raise:
                raise_amount = self.min_raise
            
            if raise_amount <= player.stack:
                raise_amounts.append(raise_amount)
        
        # Always include all-in as an option if player has chips and option enabled
        if self.include_all_in and player.stack > 0:
            raise_amounts.append(player.stack)
        
        return sorted(list(set(raise_amounts)))
    
    def _round_to_big_blind(self, amount: int) -> int:
        """Round amount to nearest big blind"""
        if self.big_blind == 0:
            return amount
        remainder = amount % self.big_blind
        if remainder < self.big_blind / 2:
            return amount - remainder
        else:
            return amount - remainder + self.big_blind
        
    def start_new_hand(self):
        """Reset pot manager for a new hand"""
        self.pots = [Pot()]
        self.current_bet = 0
        self.min_raise = self.big_blind
        self.last_raise_amount = 0
        
    def post_blinds(self, small_blind_player: Player, big_blind_player: Player):
        """Post small and big blinds"""
        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)
        
        self.pots[0].add_chips(sb_amount + bb_amount)
        self.current_bet = self.big_blind
        self.last_raise_amount = self.big_blind
        
    def place_bet(self, player: Player, amount: int) -> Tuple[int, str]:
        """
        Place a bet for a player.
        
        FIX: Properly updates current_bet for raises, but NOT for calls/all-ins that 
        just match or go short of the to_call amount.
        """
        # Check if this is a check
        if amount == 0 and self.current_bet == player.current_bet:
            return 0, "check"
        
        to_call = self.current_bet - player.current_bet
        
        # Fold if amount is less than required call
        if amount < to_call:
            player.fold()
            return 0, "fold"
        
        # Place the bet
        actual_bet = player.bet(amount)
        self.pots[0].add_chips(actual_bet)
        
        # Determine action and update current_bet ONLY for raises
        if actual_bet > to_call:
            # This is a raise - update current_bet
            raise_amount = actual_bet - to_call
            self.current_bet = player.current_bet  # ← UPDATE for raises only
            self.last_raise_amount = raise_amount
            self.min_raise = int(raise_amount * self.min_raise_multiplier)
            action = "all-in" if player.is_all_in else "raise"
        else:
            # This is a call (actual_bet == to_call) - don't update current_bet
            action = "all-in" if player.is_all_in else "call"
        
        return actual_bet, action
    
    def start_new_betting_round(self, players: List[Player]):
        """Start a new betting round"""
        for player in players:
            player.reset_current_bet()
        
        self.current_bet = 0
        self.min_raise = int(self.big_blind * self.min_raise_multiplier)
        self.last_raise_amount = 0
        
    def get_valid_raise_range(self, player: Player) -> Tuple[int, int]:
        """Get valid raise range for a player"""
        to_call = self.current_bet - player.current_bet
        min_raise_amount = self.min_raise
        max_raise_amount = to_call + player.stack
        
        if max_raise_amount < to_call + min_raise_amount:
            return 0, 0
        
        return min_raise_amount, max_raise_amount
    
    def calculate_side_pots(self, players: List[Player]) -> List[Pot]:
        """Calculate main pot and side pots"""
        contributing_players = [p for p in players if p.total_bet_this_hand > 0]
        
        if not contributing_players:
            return [Pot()]
        
        contributing_players.sort(key=lambda p: p.total_bet_this_hand)
        
        pots = []
        remaining_players = contributing_players.copy()
        previous_bet_level = 0
        
        while remaining_players:
            min_bet = remaining_players[0].total_bet_this_hand
            pot = Pot()
            pot_contribution = min_bet - previous_bet_level
            
            for player in remaining_players:
                pot.add_chips(pot_contribution)
                pot.eligible_players.append(player.player_id)
            
            pots.append(pot)
            remaining_players = [p for p in remaining_players if p.total_bet_this_hand > min_bet]
            previous_bet_level = min_bet
        
        return pots
    
    def distribute_pots(self, players: List[Player], hand_ranks: Dict[int, int]) -> Dict[int, int]:
        """Distribute pots to winners"""
        pots = self.calculate_side_pots(players)
        winnings: Dict[int, int] = {p.player_id: 0 for p in players}
        
        for pot in pots:
            eligible_ranks = {
                pid: hand_ranks.get(pid, 9999) 
                for pid in pot.eligible_players 
                if pid in hand_ranks
            }
            
            if not eligible_ranks:
                continue
            
            best_rank = min(eligible_ranks.values())
            winners = [pid for pid, rank in eligible_ranks.items() if rank == best_rank]
            
            pot_after_rake = pot.amount
            if self.rake_percent > 0 and len(eligible_ranks) > 1:
                rake = min(int(pot.amount * self.rake_percent), self.rake_cap)
                pot_after_rake -= rake
            
            amount_per_winner = pot_after_rake // len(winners)
            remainder = pot_after_rake % len(winners)
            
            for i, winner_id in enumerate(winners):
                winnings[winner_id] += amount_per_winner
                if i < remainder:
                    winnings[winner_id] += 1
        
        return winnings
    
    def get_pot_total(self) -> int:
        """Get total amount in all pots"""
        return sum(pot.amount for pot in self.pots)
    
    def __repr__(self):
        return f"PotManager(pots={len(self.pots)}, total={self.get_pot_total()}, current_bet={self.current_bet})"