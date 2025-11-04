"""
Opponent tracker - collects real poker HUD statistics
"""

from typing import Dict
import numpy as np


class OpponentTracker:
    """Tracks opponent statistics to enable exploitative play"""
    
    def __init__(self):
        # Basic counts
        self.hands_played = 0
        
        # VPIP tracking
        self.hands_vpip = 0
        
        # PFR tracking
        self.hands_pfr = 0
        
        # Aggression tracking
        self.bets_made = 0
        self.raises_made = 0
        self.calls_made = 0
        
        # 3-Bet tracking
        self.hands_faced_3bet = 0
        self.hands_folded_to_3bet = 0
        
        # C-Bet tracking
        self.cbet_situations = 0
        self.cbets_made = 0
        self.cbet_faced = 0
        self.cbet_folded_to = 0
        
        # Showdown tracking
        self.showdown_count = 0
        self.showdown_wins = 0
    
    def record_hand_start(self):
        """Called at start of hand"""
        self.hands_played += 1
    
    def record_preflop_action(self, entered_pot: bool, raised: bool):
        """Record preflop action"""
        if entered_pot:
            self.hands_vpip += 1
        if raised:
            self.hands_pfr += 1
    
    def record_aggression(self, action: str):
        """Record aggressive action (bet/raise/call)"""
        if action == "bet" or action == "raise":
            self.bets_made += 1
        elif action == "raise":
            self.raises_made += 1
        elif action == "call":
            self.calls_made += 1
    
    def record_3bet_response(self, faced: bool, folded: bool):
        """Record 3-bet response"""
        if faced:
            self.hands_faced_3bet += 1
            if folded:
                self.hands_folded_to_3bet += 1
    
    def record_cbet_action(self, is_cbet_situation: bool, made_bet: bool, 
                          faced_bet: bool, folded_to_bet: bool):
        """Record continuation bet action"""
        if is_cbet_situation:
            self.cbet_situations += 1
            if made_bet:
                self.cbets_made += 1
        
        if faced_bet:
            self.cbet_faced += 1
            if folded_to_bet:
                self.cbet_folded_to += 1
    
    def record_showdown(self, went_to_showdown: bool, won: bool):
        """Record showdown result"""
        if went_to_showdown:
            self.showdown_count += 1
            if won:
                self.showdown_wins += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get all statistics as dictionary"""
        stats = {
            'hands_played': self.hands_played,
            'VPIP': self._safe_divide(self.hands_vpip, self.hands_played),
            'PFR': self._safe_divide(self.hands_pfr, self.hands_played),
            'AF': self._calculate_af(),
            'fold_to_3bet': self._safe_divide(self.hands_folded_to_3bet, self.hands_faced_3bet),
            'cbet_pct': self._safe_divide(self.cbets_made, self.cbet_situations),
            'fold_to_cbet': self._safe_divide(self.cbet_folded_to, self.cbet_faced),
            'go2sd': self._safe_divide(self.showdown_count, self.hands_played),
            'wafsd': self._safe_divide(self.showdown_wins, self.showdown_count),
        }
        return stats
    
    def get_stats_vector(self) -> np.ndarray:
        """Get stats as normalized numpy vector for neural network"""
        stats = self.get_stats()
        vector = np.array([
            stats['VPIP'],
            stats['PFR'],
            stats['AF'] / 4.0,  # Normalize AF (typical range 0-4)
            stats['fold_to_3bet'],
            stats['cbet_pct'],
            stats['fold_to_cbet'],
            stats['go2sd'],
            stats['wafsd'],
        ], dtype=np.float32)
        return vector
    
    def get_player_type(self) -> str:
        """Classify player type based on stats"""
        stats = self.get_stats()
        
        if stats['hands_played'] < 10:
            return 'unknown'
        
        vpip = stats['VPIP']
        pfr = stats['PFR']
        af = stats['AF']
        fold_3bet = stats['fold_to_3bet']
        
        # Tight-Aggressive (TAG)
        if vpip < 0.25 and pfr > 0.12 and af > 2.0 and fold_3bet > 0.4:
            return 'TAG'
        
        # Loose-Aggressive (LAG)
        if vpip > 0.30 and pfr > 0.20 and af > 2.5 and fold_3bet < 0.4:
            return 'LAG'
        
        # Tight-Passive (Nit)
        if vpip < 0.15 and pfr < 0.10 and af < 1.5 and fold_3bet > 0.7:
            return 'Nit'
        
        # Loose-Passive (Fish)
        if vpip > 0.40 and pfr < 0.25 and af < 1.0:
            return 'Fish'
        
        return 'Unknown'
    
    def get_exploits(self) -> Dict[str, str]:
        """Get exploitative strategies based on stats"""
        stats = self.get_stats()
        exploits = {}
        
        # VPIP exploits
        if stats['VPIP'] > 0.40:
            exploits['VPIP'] = 'Play wider range, tighten value requirements'
        elif stats['VPIP'] < 0.15:
            exploits['VPIP'] = 'Steal blinds more often, widen aggression'
        
        # Fold to 3-bet exploits
        if stats['fold_to_3bet'] > 0.50:
            exploits['fold_to_3bet'] = '3-bet aggressively, widen range'
        elif stats['fold_to_3bet'] < 0.30:
            exploits['fold_to_3bet'] = '3-bet only for value, use premium hands'
        
        # C-Bet exploits
        if stats['cbet_pct'] > 0.80:
            exploits['cbet_pct'] = 'Check-raise more, float more'
        elif stats['cbet_pct'] < 0.50:
            exploits['cbet_pct'] = 'Take initiative, bet flops yourself'
        
        # Fold to C-Bet exploits
        if stats['fold_to_cbet'] > 0.60:
            exploits['fold_to_cbet'] = 'C-bet often with many bluffs'
        elif stats['fold_to_cbet'] < 0.40:
            exploits['fold_to_cbet'] = 'C-bet only with strong hands or draws'
        
        # Go to Showdown exploits
        if stats['go2sd'] > 0.35:
            exploits['go2sd'] = 'Value bet more, bluff less'
        elif stats['go2sd'] < 0.20:
            exploits['go2sd'] = 'Apply more pressure, more bluffs'
        
        # WAFSD exploits
        if stats['wafsd'] > 0.55:
            exploits['wafsd'] = "Don't bluff, value bet more"
        elif stats['wafsd'] < 0.45:
            exploits['wafsd'] = 'Bluff more, call down more'
        
        return exploits
    
    def _calculate_af(self) -> float:
        """Calculate aggression factor: (bets + raises) / calls"""
        aggression = self.bets_made + self.raises_made
        if self.calls_made == 0:
            return float(aggression) if aggression > 0 else 1.0
        return float(aggression) / float(self.calls_made)
    
    @staticmethod
    def _safe_divide(numerator: int, denominator: int) -> float:
        """Safe division that returns 0 if denominator is 0"""
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)
    
    def reset(self):
        """Reset all tracking for new session"""
        self.__init__()
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"OpponentTracker({self.get_player_type()}, "
                f"VPIP={stats['VPIP']:.1%}, "
                f"PFR={stats['PFR']:.1%}, "
                f"AF={stats['AF']:.2f})")