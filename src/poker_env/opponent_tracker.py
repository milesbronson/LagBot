"""
Opponent Tracking System for Texas Hold'em RL Bot

Tracks opponent statistics, hand history, position, and stack information
to enable exploitative play and opponent modeling.

Positions are tracked as integers (0-9) representing seat numbers relative to dealer:
0 = Dealer/Button
1 = Small Blind
2 = Big Blind
3-9 = Other seats (position relative to dealer)
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from datetime import datetime


class Action(Enum):
    """Poker actions"""
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"
    ALL_IN = "all_in"
    POST_BLIND = "post_blind"


class Street(Enum):
    """Betting streets"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


@dataclass
class ActionRecord:
    """Single action in a hand"""
    street: Street
    player_id: int
    player_name: str
    action: Action
    amount: int
    pot_size: int
    stack_before: int
    stack_after: int
    position: int  # Position as integer (0 = dealer, 1 = SB, 2 = BB, etc)
    timestamp: float


@dataclass
class StackRatio:
    """Stack sizes relative to blinds"""
    player_stack: float           # Player stack / big blind
    opponent_stack: float         # Opponent stack / big blind
    pot_size: float              # Pot size / big blind
    bb: int                       # Big blind amount


@dataclass
class OpponentProfile:
    """Statistics for a single opponent"""
    player_id: int
    player_name: str
    
    # Basic stats
    hands_played: int = 0
    
    # Preflop stats
    vpip: float = 0.0  # Voluntarily put in pot %
    pfr: float = 0.0   # Preflop raise %
    af: float = 0.0    # Aggression factor
    
    # 3-bet stats
    three_bet_faced: int = 0
    three_bet_folded: int = 0
    three_bet_percent: float = 0.0
    
    # C-bet stats
    cbet_opportunities: int = 0
    cbets_made: int = 0
    cbet_percent: float = 0.0
    
    # Fold stats
    fold_to_cbet_faced: int = 0
    fold_to_cbet_count: int = 0
    fold_to_cbet_percent: float = 0.0
    
    # Showdown stats
    went_to_showdown: int = 0
    showdown_wins: int = 0
    went_to_showdown_percent: float = 0.0
    win_at_showdown_percent: float = 0.0
    
    # 3-Bet frequency (3bet % of opened pots)
    three_bet_count: int = 0
    three_bet_opportunities: int = 0
    three_bet_frequency: float = 0.0
    
    # WTSD: Went To Showdown
    wtsd_percent: float = 0.0
    
    # W$SD: Won Money At Showdown
    won_at_showdown: int = 0
    money_won_at_showdown: int = 0
    
    # WWSF: Won When Saw Flop
    saw_flop_count: int = 0
    won_when_saw_flop: int = 0
    wwsf_percent: float = 0.0
    
    # Fold to 3-Bet After Raising
    raised_preflop: int = 0
    faced_3bet_after_raise: int = 0
    folded_to_3bet_after_raise: int = 0
    fold_to_3bet_after_raise_percent: float = 0.0
    
    # Preflop Squeeze
    squeeze_opportunities: int = 0
    squeeze_attempts: int = 0
    squeeze_percent: float = 0.0
    
    # Flop C-Bet
    flop_cbet_opportunities: int = 0
    flop_cbet_made: int = 0
    flop_cbet_percent: float = 0.0
    
    # Fold to Flop C-Bet
    faced_flop_cbet: int = 0
    folded_to_flop_cbet: int = 0
    fold_to_flop_cbet_percent: float = 0.0
    
    # Position-based tracking
    position_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # Recent actions for temporal modeling (last N actions)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Last update time
    last_update: Optional[float] = None
    
    # Confidence in stats (hands_played)
    confidence: float = 0.0
    
    def recalculate_metrics(self):
        """Recalculate all derived percentages from raw counts"""
        # VPIP and PFR stay as calculated elsewhere
        
        # WTSD: Went To Showdown %
        self.wtsd_percent = self.went_to_showdown / max(self.hands_played, 1)
        
        # W$SD: Won Money At Showdown
        if self.went_to_showdown > 0:
            self.win_at_showdown_percent = self.showdown_wins / self.went_to_showdown
        
        # WWSF: Won When Saw Flop
        if self.saw_flop_count > 0:
            self.wwsf_percent = self.won_when_saw_flop / self.saw_flop_count
        
        # 3-Bet Frequency
        if self.three_bet_opportunities > 0:
            self.three_bet_frequency = self.three_bet_count / self.three_bet_opportunities
        
        # Fold to 3-Bet After Raising
        if self.faced_3bet_after_raise > 0:
            self.fold_to_3bet_after_raise_percent = self.folded_to_3bet_after_raise / self.faced_3bet_after_raise
        
        # Squeeze %
        if self.squeeze_opportunities > 0:
            self.squeeze_percent = self.squeeze_attempts / self.squeeze_opportunities
        
        # Flop C-Bet %
        if self.flop_cbet_opportunities > 0:
            self.flop_cbet_percent = self.flop_cbet_made / self.flop_cbet_opportunities
        
        # Fold to Flop C-Bet %
        if self.faced_flop_cbet > 0:
            self.fold_to_flop_cbet_percent = self.folded_to_flop_cbet / self.faced_flop_cbet
    
    def get_player_type(self) -> str:
        """Classify opponent based on stats"""
        if self.hands_played < 10:
            return "UNKNOWN"
        
        vpip = self.vpip
        pfr = self.pfr
        af = self.af
        
        # Check TIGHT_AGGRESSIVE first (tight but aggressive)
        if vpip < 0.25 and af > 2.0:
            return "TIGHT_AGGRESSIVE"
        elif vpip < 0.15 and pfr < 0.12:
            return "VERY_TIGHT"
        elif vpip < 0.25 and pfr < 0.20:
            return "TIGHT"
        elif vpip > 0.50 and pfr > 0.35:
            return "LOOSE_AGGRESSIVE"
        elif vpip > 0.50 and af < 1.0:
            return "LOOSE_PASSIVE"
        elif vpip < 0.25 and af > 2.0:
            return "TIGHT_AGGRESSIVE"
        else:
            return "BALANCED"
    
    def _recalculate_stats(self):
        """Recalculate VPIP, PFR, and AF from position stats and recent actions"""
        if self.hands_played == 0:
            return
        
        vpip_total = 0
        pfr_total = 0
        
        for pos_stats in self.position_stats.values():
            vpip_total += pos_stats.get('vpip_count', 0)
            pfr_total += pos_stats.get('pfr_count', 0)
        
        self.vpip = vpip_total / self.hands_played if self.hands_played > 0 else 0
        self.pfr = pfr_total / self.hands_played if self.hands_played > 0 else 0
        
        # AF: Aggression Factor = (bets + raises) / calls
        bets_raises = sum(
            1 for a in self.recent_actions 
            if a['action'] in ['bet', 'raise', 'all_in']
        )
        calls = sum(
            1 for a in self.recent_actions 
            if a['action'] == 'call'
        )
        self.af = bets_raises / max(calls, 1) if calls > 0 else 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'player_id': self.player_id,
            'player_name': self.player_name,
            'hands_played': self.hands_played,
            'vpip': round(self.vpip, 3),
            'pfr': round(self.pfr, 3),
            'af': round(self.af, 3),
            'three_bet_frequency': round(self.three_bet_frequency, 3),
            'wtsd': round(self.wtsd_percent, 3),
            'wsd': round(self.win_at_showdown_percent, 3),
            'wwsf': round(self.wwsf_percent, 3),
            'fold_to_3bet_after_raise': round(self.fold_to_3bet_after_raise_percent, 3),
            'squeeze_percent': round(self.squeeze_percent, 3),
            'flop_cbet': round(self.flop_cbet_percent, 3),
            'fold_to_flop_cbet': round(self.fold_to_flop_cbet_percent, 3),
            'cbet_percent': round(self.cbet_percent, 3),
            'fold_to_cbet_percent': round(self.fold_to_cbet_percent, 3),
            'went_to_showdown_percent': round(self.went_to_showdown_percent, 3),
            'player_type': self.get_player_type(),
            'confidence': round(self.confidence, 3),
        }


@dataclass
class HandRecord:
    """Complete record of a poker hand"""
    hand_number: int
    dealer_position: int
    small_blind: int
    big_blind: int
    timestamp: float
    
    # Actions by street
    actions: List[ActionRecord] = field(default_factory=list)
    
    # Final results
    winner_ids: List[int] = field(default_factory=list)
    winnings: Dict[int, int] = field(default_factory=dict)
    stacks_at_end: Dict[int, int] = field(default_factory=dict)
    
    # Community cards
    community_cards: Tuple = field(default_factory=tuple)
    
    # Participant info
    players_in_hand: List[int] = field(default_factory=list)
    players_positions: Dict[int, int] = field(default_factory=dict)  # player_id -> position_int
    num_players: int = 0  # Total number of players in this hand
    
    def add_action(self, action_record: ActionRecord):
        """Add action to hand"""
        self.actions.append(action_record)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'hand_number': self.hand_number,
            'dealer_position': self.dealer_position,
            'blinds': (self.small_blind, self.big_blind),
            'timestamp': self.timestamp,
            'num_players': self.num_players,
            'actions': len(self.actions),
            'winner_ids': self.winner_ids,
            'players_in_hand': self.players_in_hand,
        }


class OpponentTracker:
    """
    Multi-opponent tracking system for Texas Hold'em.
    
    Tracks statistics, hand history, position, and stack information
    for all opponents at the table.
    """
    
    def __init__(self, max_history_hands: int = 1000):
        self.opponents: Dict[int, OpponentProfile] = {}
        self.hand_history: deque = deque(maxlen=max_history_hands)
        self.current_hand: Optional[HandRecord] = None
        self.hand_number: int = 0
        self.bb: int = 0
        
    def start_hand(self, hand_number: int, players: List[Dict], dealer_position: int,
                   small_blind: int, big_blind: int):
        """Start tracking a new hand"""
        self.hand_number = hand_number
        self.bb = big_blind
        
        # Create hand record
        self.current_hand = HandRecord(
            hand_number=hand_number,
            dealer_position=dealer_position,
            small_blind=small_blind,
            big_blind=big_blind,
            timestamp=datetime.now().timestamp(),
            players_in_hand=[p['id'] for p in players],
            num_players=len(players)
        )
        
        # Initialize opponents if not seen before
        for player in players:
            player_id = player['id']
            if player_id not in self.opponents:
                self.opponents[player_id] = OpponentProfile(
                    player_id=player_id,
                    player_name=player.get('name', f'Player_{player_id}')
                )
    
    def record_action(self, player_id: int, player_name: str, action: Action,
                     amount: int, pot_size: int, stack_before: int, stack_after: int,
                     street: Street, position: int):
        """Record a player action"""
        if self.current_hand is None:
            return
        
        # Create action record
        action_record = ActionRecord(
            street=street,
            player_id=player_id,
            player_name=player_name,
            action=action,
            amount=amount,
            pot_size=pot_size,
            stack_before=stack_before,
            stack_after=stack_after,
            position=position,
            timestamp=datetime.now().timestamp()
        )
        
        self.current_hand.add_action(action_record)
        
        # Update opponent profile
        if player_id in self.opponents:
            opponent = self.opponents[player_id]
            opponent.recent_actions.append({
                'action': action.value,
                'street': street.value,
                'position': position,  # position is already int
                'amount': amount,
                'timestamp': action_record.timestamp
            })
    
    def record_community_cards(self, cards: Tuple):
        """Record community cards for the hand"""
        if self.current_hand is not None:
            self.current_hand.community_cards = cards
    
    def record_positions(self, positions: Dict[int, int]):
        """Record player positions for the hand (position as integers)"""
        if self.current_hand is not None:
            self.current_hand.players_positions = positions
    
    def end_hand(self, winners: List[int], winnings: Dict[int, int],
                final_stacks: Dict[int, int]):
        """Complete hand tracking and update statistics"""
        if self.current_hand is None:
            return
        
        self.current_hand.winner_ids = winners
        self.current_hand.winnings = winnings
        self.current_hand.stacks_at_end = final_stacks
        
        # Store hand
        self.hand_history.append(self.current_hand)
        
        # Update opponent statistics
        self._update_opponent_stats(self.current_hand)
        
        self.current_hand = None
    
    def _update_opponent_stats(self, hand: HandRecord):
        """Calculate updated statistics for all opponents in hand"""
        
        for player_id in hand.players_in_hand:
            if player_id not in self.opponents:
                continue
            
            opponent = self.opponents[player_id]
            position = hand.players_positions.get(player_id, -1)  # -1 = unknown position
            
            # Get actions by this player
            player_actions = [a for a in hand.actions if a.player_id == player_id]
            
            if not player_actions:
                continue
            
            # Increment hands played
            opponent.hands_played += 1
            opponent.last_update = datetime.now().timestamp()
            opponent.confidence = min(opponent.hands_played / 100, 1.0)
            
            # Track position stats
            pos_key = position
            if pos_key not in opponent.position_stats:
                opponent.position_stats[pos_key] = {
                    'hands': 0,
                    'vpip_count': 0,
                    'pfr_count': 0,
                }
            opponent.position_stats[pos_key]['hands'] += 1
            
            # Preflop actions
            preflop_actions = [a for a in player_actions if a.street == Street.PREFLOP]
            flop_actions = [a for a in player_actions if a.street == Street.FLOP]
            
            # 1. VPIP: Voluntarily put money in pot
            if preflop_actions:
                if any(a.action in [Action.CALL, Action.RAISE, Action.BET, Action.ALL_IN] for a in preflop_actions):
                    opponent.position_stats[pos_key]['vpip_count'] += 1
                
                # 2. PFR: Preflop raise
                if any(a.action in [Action.RAISE, Action.ALL_IN] for a in preflop_actions):
                    opponent.position_stats[pos_key]['pfr_count'] += 1
                
                # 3. 3-Bet Frequency: Track if player raised to a raise
                preflop_raises = [a for a in preflop_actions if a.action in [Action.RAISE, Action.ALL_IN]]
                if preflop_raises:
                    opponent.raised_preflop += 1
                    opponent.three_bet_opportunities += 1
                    # If they raised when someone else raised, it's a 3-bet
                    if len(preflop_raises) >= 2:
                        opponent.three_bet_count += 1
            
            # 4. WTSD: Went To Showdown
            if player_id in hand.winner_ids or (hand.actions and any(a.action == Action.FOLD for a in player_actions) and len(player_actions) >= 5):
                # Simple heuristic: if they got to river or were in at showdown
                if len(flop_actions) > 0:  # Made it to at least flop
                    opponent.went_to_showdown += 1
                    
                    # 5. W$SD: Won Money At Showdown
                    if player_id in hand.winner_ids and hand.winnings.get(player_id, 0) > 0:
                        opponent.showdown_wins += 1
                        opponent.won_at_showdown += 1
                        opponent.money_won_at_showdown += hand.winnings.get(player_id, 0)
            
            # 6. WWSF: Won When Saw Flop
            if flop_actions:
                opponent.saw_flop_count += 1
                if player_id in hand.winner_ids:
                    opponent.won_when_saw_flop += 1
            
            # 7. Fold to 3-Bet After Raising: Track folding to 3-bet
            if opponent.raised_preflop > 0:
                # Check if faced a re-raise (3-bet)
                if len(preflop_actions) >= 2:
                    preflop_raises = [a for a in preflop_actions if a.action in [Action.RAISE, Action.ALL_IN]]
                    if len(preflop_raises) == 1:  # Raised once
                        # Check if faced re-raise
                        opponent.faced_3bet_after_raise += 1
                        if preflop_actions[-1].action == Action.FOLD:
                            opponent.folded_to_3bet_after_raise += 1
            
            # 8. Preflop Squeeze: Difficult to detect perfectly, simplified version
            # Squeeze = raise when there's already a raise and at least one caller
            if preflop_actions and any(a.action == Action.RAISE for a in preflop_actions):
                # Check if it looks like a squeeze (raise after other action)
                action_types = [a.action for a in preflop_actions]
                if len([a for a in preflop_actions if a.action in [Action.CALL, Action.BET]]) > 0:
                    opponent.squeeze_opportunities += 1
                    if action_types[-1] == Action.RAISE:
                        opponent.squeeze_attempts += 1
            
            # 9. Flop C-Bet: Continuation bet on flop
            if len(flop_actions) > 0 and len(preflop_actions) > 0:
                # If they raised preflop and bet flop, it's a c-bet opportunity
                if any(a.action in [Action.RAISE, Action.BET] for a in preflop_actions):
                    opponent.flop_cbet_opportunities += 1
                    if any(a.action in [Action.BET, Action.RAISE] for a in flop_actions):
                        opponent.flop_cbet_made += 1
            
            # 10. Fold to Flop C-Bet: Track folding to flop bet
            if flop_actions and any(a.action in [Action.BET, Action.RAISE] for a in flop_actions):
                # Check if they faced a bet and folded
                fold_actions = [a for a in flop_actions if a.action == Action.FOLD]
                if fold_actions:
                    opponent.faced_flop_cbet += 1
                    opponent.folded_to_flop_cbet += 1
            
            # Calculate all percentages
            opponent.recalculate_metrics()
            opponent._recalculate_stats()
    
    def get_opponent_features(self, opponent_id: int, num_seats: int = 6,
                             recent_hand_window: int = 10) -> List[float]:
        """
        Get observation features for a specific opponent.
        
        Returns normalized features suitable for RL model.
        Format: [VPIP, PFR, AF, hands_played_norm, position_diversity, 
                 recent_action_type_1, recent_action_type_2, ...]
        """
        if opponent_id not in self.opponents:
            return [0.0] * (5 + recent_hand_window)
        
        opponent = self.opponents[opponent_id]
        features = []
        
        # Basic stats (normalized 0-1)
        features.append(min(opponent.vpip, 1.0))
        features.append(min(opponent.pfr, 1.0))
        features.append(min(opponent.af / 3.0, 1.0))  # Normalize AF (cap at 3)
        features.append(min(opponent.hands_played / 100, 1.0))  # Hands played
        
        # Position diversity (how spread across positions)
        position_count = len(opponent.position_stats)
        features.append(min(position_count / num_seats, 1.0))
        
        # Recent action encoding
        recent_actions = list(opponent.recent_actions)[-recent_hand_window:]
        action_types = {
            'fold': 0.0,
            'check': 0.2,
            'call': 0.4,
            'bet': 0.6,
            'raise': 0.8,
            'all_in': 1.0,
        }
        
        for i in range(recent_hand_window):
            if i < len(recent_actions):
                action_value = action_types.get(recent_actions[i]['action'], 0.0)
                features.append(action_value)
            else:
                features.append(0.0)  # Pad with zeros
        
        return features
    
    def get_all_opponents_features(self, learning_agent_id: int, num_seats: int = 6) -> Dict[int, List[float]]:
        """
        Get features for all opponents (for observation space).
        
        Returns dict mapping opponent_id -> feature vector.
        """
        features = {}
        for opponent_id, opponent in self.opponents.items():
            if opponent_id != learning_agent_id:
                features[opponent_id] = self.get_opponent_features(opponent_id, num_seats)
        return features
    
    def get_observation_features(self, hero_id: int, opponent_ids: List[int], 
                              max_opponents: int = 9, features_per_opponent: int = 4) -> List[float]:
        """
        Get fixed-size opponent features for RL observation space.
        
        Handles variable player counts by zero-padding missing opponent slots.
        This ensures a consistent observation space size regardless of:
        - Table size (2-10 players)
        - Number of opponents who have folded
        - Players who joined/left
        
        Args:
            hero_id: The learning agent's player ID (excluded from features)
            opponent_ids: List of opponent player IDs in seat order
            max_opponents: Maximum opponents to encode (default 9 for 10-player max)
            features_per_opponent: Features per opponent (default 4: VPIP, PFR, AF, confidence)
        
        Returns:
            List of floats with length = max_opponents * features_per_opponent (36 by default)
            
        Feature layout per opponent slot:
            [0] VPIP (0-1): Voluntarily put money in pot %
            [1] PFR (0-1): Preflop raise %
            [2] AF (0-1): Aggression factor / 3.0 (normalized, capped at 1.0)
            [3] Confidence (0-1): min(hands_played / 100, 1.0)
        
        Padding explanation:
            With variable players (2-10), observation must be fixed size for PPO.
            Solution: Always allocate 9 opponent slots (max opponents at 10-player table).
            
            Example with 3 players (hero + 2 opponents):
            [vpip1, pfr1, af1, conf1,   # Opponent 1 - real data
            vpip2, pfr2, af2, conf2,   # Opponent 2 - real data
            0.0, 0.0, 0.0, 0.0,        # Slot 3 - zero-padded
            0.0, 0.0, 0.0, 0.0,        # Slot 4 - zero-padded
            ...                         # Slots 5-9 - zero-padded
            ]
            
            The neural network learns that zeros indicate "no opponent in this slot".
        """
        features = []
        
        for i in range(max_opponents):
            if i < len(opponent_ids):
                pid = opponent_ids[i]
                if pid in self.opponents:
                    opp = self.opponents[pid]
                    features.extend([
                        min(opp.vpip, 1.0),
                        min(opp.pfr, 1.0),
                        min(opp.af / 3.0, 1.0),  # Normalize AF (typical range 0-3)
                        opp.confidence
                    ])
                else:
                    # Opponent in game but not yet tracked (new player)
                    # Use neutral defaults
                    features.extend([0.0, 0.0, 0.33, 0.0])
            else:
                # Zero-pad empty opponent slots
                features.extend([0.0] * features_per_opponent)
        
        return features
    
    def get_opponent_stats(self, opponent_id: int) -> Optional[Dict]:
        """Get human-readable stats for an opponent"""
        if opponent_id not in self.opponents:
            return None
        
        opponent = self.opponents[opponent_id]
        return opponent.to_dict()
    
    def get_all_opponent_stats(self) -> Dict[int, Dict]:
        """Get stats for all opponents"""
        return {pid: opp.to_dict() for pid, opp in self.opponents.items()}
    
    def get_recent_hands(self, opponent_id: int, limit: int = 20) -> List[Dict]:
        """Get recent hands involving a specific opponent"""
        relevant_hands = []
        for hand in reversed(self.hand_history):
            if opponent_id in hand.players_in_hand:
                relevant_hands.append(hand.to_dict())
                if len(relevant_hands) >= limit:
                    break
        return relevant_hands
    
    def export_stats(self, filepath: str):
        """Export all opponent stats to JSON"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_hands': self.hand_number,
            'opponents': self.get_all_opponent_stats(),
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_exploitable_opponents(self, threshold_hands: int = 20) -> List[Dict]:
        """
        Get list of opponents with exploitable weaknesses.
        
        Returns list sorted by exploitability score.
        """
        exploitable = []
        
        for opponent_id, opponent in self.opponents.items():
            if opponent.hands_played < threshold_hands:
                continue  # Not enough data
            
            exploits = []
            
            # Very loose (VPIP > 50%)
            if opponent.vpip > 0.50:
                exploits.append({
                    'type': 'loose_vpip',
                    'severity': opponent.vpip - 0.50,
                    'action': 'Tighten range against this opponent'
                })
            
            # Very tight (VPIP < 15%)
            if opponent.vpip < 0.15:
                exploits.append({
                    'type': 'tight_vpip',
                    'severity': 0.15 - opponent.vpip,
                    'action': 'Steal more blinds/aggress in position'
                })
            
            # Not aggressive (AF < 0.8)
            if opponent.af < 0.8:
                exploits.append({
                    'type': 'passive',
                    'severity': 0.8 - opponent.af,
                    'action': 'Value bet more, bluff less'
                })
            
            # Very aggressive (AF > 3.0)
            if opponent.af > 3.0:
                exploits.append({
                    'type': 'aggressive',
                    'severity': opponent.af - 3.0,
                    'action': 'Check-raise more, call down tighter'
                })
            
            if exploits:
                total_severity = sum(e['severity'] for e in exploits)
                exploitable.append({
                    'opponent_id': opponent_id,
                    'opponent_name': opponent.player_name,
                    'hands': opponent.hands_played,
                    'exploitability_score': total_severity,
                    'exploits': exploits,
                    'stats': opponent.to_dict()
                })
        
        # Sort by exploitability
        exploitable.sort(key=lambda x: x['exploitability_score'], reverse=True)
        return exploitable
    
    def get_player_count_for_hand(self, hand_number: int) -> int:
        """Get the number of players in a specific hand"""
        for hand in self.hand_history:
            if hand.hand_number == hand_number:
                return hand.num_players
        return 0
    
    def get_average_player_count(self) -> float:
        """Get average number of players across all hands"""
        if not self.hand_history:
            return 0
        return sum(h.num_players for h in self.hand_history) / len(self.hand_history)
    
    def get_hand_count_by_player_count(self) -> Dict[int, int]:
        """Get count of hands grouped by number of players
        
        Returns: {num_players: count_of_hands}
        Example: {6: 45, 4: 30, 5: 25} means 45 hands with 6 players, etc.
        """
        counts = {}
        for hand in self.hand_history:
            num = hand.num_players
            counts[num] = counts.get(num, 0) + 1
        return dict(sorted(counts.items()))


class StackRatioTracker:
    """Tracks stack sizes relative to blinds"""
    
    @staticmethod
    def get_stack_ratios(player_stack: int, opponent_stack: int, bb: int,
                        pot_size: int) -> StackRatio:
        """Get normalized stack ratios"""
        return StackRatio(
            player_stack=player_stack / bb if bb > 0 else 0,
            opponent_stack=opponent_stack / bb if bb > 0 else 0,
            pot_size=pot_size / bb if bb > 0 else 0,
            bb=bb
        )
    
    @staticmethod
    def classify_stack_depth(stack_ratio: float) -> str:
        """Classify stack depth"""
        if stack_ratio < 10:
            return "SHALLOW"
        elif stack_ratio < 40:
            return "MEDIUM"
        elif stack_ratio < 100:
            return "DEEP"
        else:
            return "VERY_DEEP"