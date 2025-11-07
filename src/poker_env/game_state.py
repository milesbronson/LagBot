"""
Game state management for Texas Hold'em with hand history tracking
"""

from enum import Enum
from typing import List, Optional, Dict, Any
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


class Street(Enum):
    """Poker streets for history tracking"""
    BLINDS = "Blinds"
    PREFLOP = "Preflop"
    FLOP = "Flop"
    TURN = "Turn"
    RIVER = "River"
    SHOWDOWN = "Showdown"


class HandHistory:
    """Complete record of a poker hand"""
    
    def __init__(self, hand_number: int):
        self.hand_number = hand_number
        self.actions: List[Dict[str, Any]] = []
        self.hole_cards: Dict[int, List[int]] = {}
        self.community_cards_by_street: Dict[str, List[int]] = {
            'flop': [],
            'turn': [],
            'river': []
        }
        self.button_position = 0
        self.small_blind = 0
        self.big_blind = 0
        self.final_stacks: Dict[int, int] = {}
        self.winnings: Dict[int, int] = {}
        self.hand_ranks: Dict[int, int] = {}
    
    def record_blind_action(self, player_id: int, player_name: str, blind_type: str, amount: int):
        """Record blind posting"""
        self.actions.append({
            'street': 'Blinds',
            'player_id': player_id,
            'player_name': player_name,
            'action': blind_type.lower(),
            'amount': amount,
            'total_bet': amount
        })
    
    def record_hole_cards(self, player_id: int, cards: List[int]):
        """Record hole cards"""
        if cards:
            self.hole_cards[player_id] = cards.copy()
    
    def record_action(self, street_name: str, player_id: int, player_name: str,
                     action: str, amount: int = 0, total_bet: int = 0):
        """Record a player action"""
        self.actions.append({
            'street': street_name,
            'player_id': player_id,
            'player_name': player_name,
            'action': action.lower(),
            'amount': amount,
            'total_bet': total_bet
        })
    
    def record_community_cards(self, street: str, cards: List[int]):
        """Record community cards"""
        if street in self.community_cards_by_street and cards:
            self.community_cards_by_street[street] = cards.copy()
    
    def record_results(self, button_pos: int, sb: int, bb: int, active_players: List[int],
                      final_stacks: Dict[int, int], winnings: Dict[int, int], 
                      hand_ranks: Dict[int, int]):
        """Record hand results"""
        self.button_position = button_pos
        self.small_blind = sb
        self.big_blind = bb
        self.final_stacks = final_stacks.copy()
        self.winnings = winnings.copy()
        self.hand_ranks = hand_ranks.copy()
    
    def display(self):
        """Display complete hand history to console"""
        print("\n" + "="*80)
        print(f"HAND #{self.hand_number}")
        print("="*80)
        
        # Setup
        print(f"\n🎰 TABLE SETUP:")
        print(f"   Button: Position {self.button_position} | Blinds: ${self.small_blind}/${self.big_blind}")
        
        # Hole cards
        if self.hole_cards:
            print(f"\n🃏 HOLE CARDS:")
            for pid in sorted(self.hole_cards.keys()):
                cards = self.hole_cards[pid]
                card_str = " ".join([HandEvaluator.card_to_string(c) for c in cards])
                print(f"   Player {pid}: {card_str}")
        
        # Actions by street
        if self.actions:
            print(f"\n⚡ ACTION SEQUENCE:")
            current_street = None
            
            for action_record in self.actions:
                street = action_record['street']
                
                # Print street header with community cards
                if street != current_street:
                    current_street = street
                    
                    if street == 'Flop' and self.community_cards_by_street['flop']:
                        cards_str = " ".join([HandEvaluator.card_to_string(c) 
                                            for c in self.community_cards_by_street['flop']])
                        print(f"\n   --- {street}: [{cards_str}] ---")
                    elif street == 'Turn' and self.community_cards_by_street['turn']:
                        cards_str = " ".join([HandEvaluator.card_to_string(c) 
                                            for c in self.community_cards_by_street['turn']])
                        print(f"\n   --- {street}: [{cards_str}] ---")
                    elif street == 'River' and self.community_cards_by_street['river']:
                        cards_str = " ".join([HandEvaluator.card_to_string(c) 
                                            for c in self.community_cards_by_street['river']])
                        print(f"\n   --- {street}: [{cards_str}] ---")
                    else:
                        print(f"\n   --- {street} ---")
                
                # Print action
                self._print_action(action_record)
        
        # Results
        print(f"\n🏆 RESULTS:")
        
        winners_found = False
        for pid in sorted(self.winnings.keys()):
            if self.winnings[pid] > 0:
                winners_found = True
                hand_desc = self._get_hand_description(pid)
                if hand_desc:
                    print(f"   ✓ Player {pid} wins ${self.winnings[pid]} with {hand_desc}")
                else:
                    print(f"   ✓ Player {pid} wins ${self.winnings[pid]} (opponents folded)")
        
        if not winners_found:
            print(f"   ⚠ No clear winner")
        
        # Final stacks
        print(f"\n💰 FINAL STACKS:")
        for pid in sorted(self.final_stacks.keys()):
            stack = self.final_stacks[pid]
            if pid in self.winnings and self.winnings[pid] != 0:
                change = self.winnings[pid]
                sign = "+" if change > 0 else ""
                print(f"   Player {pid}: ${stack} ({sign}${change})")
            else:
                print(f"   Player {pid}: ${stack}")
        
        print("="*80 + "\n")
    
    def _print_action(self, action_record: Dict[str, Any]):
        """Print a single action"""
        name = action_record['player_name']
        action = action_record['action']
        amount = action_record['amount']
        total_bet = action_record['total_bet']
        
        if action == 'small blind':
            print(f"   {name} posts small blind ${amount}")
        elif action == 'big blind':
            print(f"   {name} posts big blind ${amount}")
        elif action == 'fold':
            print(f"   {name} folds")
        elif action == 'check':
            print(f"   {name} checks")
        elif action == 'call':
            print(f"   {name} calls ${amount}")
        elif action == 'bet':
            print(f"   {name} bets ${amount}")
        elif action == 'raise':
            print(f"   {name} raises to ${total_bet}")
        elif action == 'all-in':
            print(f"   {name} goes all-in for ${amount}")
        else:
            print(f"   {name}: {action} ${amount}")
    
    def _get_hand_description(self, player_id: int) -> str:
        """Get human-readable hand description"""
        if player_id not in self.hand_ranks:
            return ""
        
        rank = self.hand_ranks[player_id]
        evaluator = HandEvaluator()
        rank_class = evaluator.get_rank_class(rank)
        return evaluator.class_to_string(rank_class)


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
        """Initialize game state"""
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
        
        # Hand history
        self.hand_history: Optional[HandHistory] = None
    
    def start_new_hand(self):
        """Start a new hand"""
        self.hand_number += 1
        self.hand_history = HandHistory(self.hand_number)
        
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
                hole_cards = [self.deck.pop(), self.deck.pop()]
                player.deal_hand(hole_cards)
                self.hand_history.record_hole_cards(player.player_id, hole_cards)
        
        self.button_position = (self.button_position + 1) % len(self.players)
        
        sb_idx = self._get_next_active_player(self.button_position)
        bb_idx = self._get_next_active_player(sb_idx)
        
        # Record blinds
        self.hand_history.record_blind_action(
            self.players[sb_idx].player_id,
            self.players[sb_idx].name,
            'small blind',
            self.small_blind
        )
        self.hand_history.record_blind_action(
            self.players[bb_idx].player_id,
            self.players[bb_idx].name,
            'big blind',
            self.big_blind
        )
        
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
            self.hand_history.record_community_cards('flop', self.community_cards)
            
        elif self.betting_round == BettingRound.FLOP:
            self._burn_card()
            self.community_cards.append(self.deck.pop())
            self.betting_round = BettingRound.TURN
            self.hand_history.record_community_cards('turn', self.community_cards)
            
        elif self.betting_round == BettingRound.TURN:
            self._burn_card()
            self.community_cards.append(self.deck.pop())
            self.betting_round = BettingRound.RIVER
            self.hand_history.record_community_cards('river', self.community_cards)
            
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
    
    def execute_action(self, action: int, raise_amount: Optional[int] = None) -> str:
        """Execute a player action and record in history"""
        player = self.get_current_player()
        
        if action == 0:
            player.fold()
            action_type = "fold"
            self.hand_history.record_action(
                self.betting_round.name,
                player.player_id,
                player.name,
                'fold'
            )
            
        elif action == 1:
            to_call = self.pot_manager.current_bet - player.current_bet
            _, action_type = self.pot_manager.place_bet(player, to_call)
            
            if action_type == "check":
                self.hand_history.record_action(
                    self.betting_round.name,
                    player.player_id,
                    player.name,
                    'check'
                )
            elif action_type == "call":
                self.hand_history.record_action(
                    self.betting_round.name,
                    player.player_id,
                    player.name,
                    'call',
                    amount=to_call
                )
            
        elif action == 2:
            if raise_amount is None:
                raise_amount = self.pot_manager.min_raise
            
            to_call = self.pot_manager.current_bet - player.current_bet
            total_bet = to_call + raise_amount
            _, action_type = self.pot_manager.place_bet(player, total_bet)
            
            if action_type == "raise":
                self.hand_history.record_action(
                    self.betting_round.name,
                    player.player_id,
                    player.name,
                    'raise',
                    amount=raise_amount,
                    total_bet=player.current_bet
                )
                self.last_aggressor_idx = self.current_player_idx
            elif action_type == "all-in":
                self.hand_history.record_action(
                    self.betting_round.name,
                    player.player_id,
                    player.name,
                    'all-in',
                    amount=total_bet
                )
        
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
        
        # Record results in hand history
        self.hand_history.record_results(
            self.button_position,
            self.small_blind,
            self.big_blind,
            [p.player_id for p in active_players],
            {p.player_id: p.stack for p in self.players},
            winnings,
            hand_ranks
        )
        
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
    
    def display_hand_history(self):
        """Display complete hand history"""
        if self.hand_history:
            self.hand_history.display()
    
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