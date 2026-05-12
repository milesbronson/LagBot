"""
Utilities for serializing game state to JSON format.
"""
from typing import Dict, List, Optional
from src.poker_env.game_state import GameState, Player, BettingRound
from backend.utils.card_converter import convert_cards_for_frontend


def serialize_player(player: Player, is_human: bool, show_hole_cards: bool = False) -> Dict:
    hole_cards = None
    if show_hole_cards or is_human:
        if player.hand and len(player.hand) == 2:
            hole_cards = convert_cards_for_frontend(player.hand)

    return {
        "player_id": player.player_id,
        "name": "You" if is_human else player.name,
        "stack": player.stack,
        "bet": player.current_bet,
        "is_active": player.is_active,
        "is_all_in": player.is_all_in,
        "is_folded": not player.is_active and not player.is_all_in,
        "hole_cards": hole_cards,
        "is_human": is_human,
        "is_dealer": False,
        "is_small_blind": False,
        "is_big_blind": False,
    }


def _get_position_indices(game_state: GameState):
    num_players = len(game_state.players)
    dealer_idx = game_state.button_position
    sb_idx = (dealer_idx + 1) % num_players
    bb_idx = (dealer_idx + 2) % num_players
    if num_players == 2:
        sb_idx = dealer_idx
        bb_idx = (dealer_idx + 1) % num_players
    return dealer_idx, sb_idx, bb_idx


def serialize_game_state(
    game_state: GameState,
    human_player_id: int,
    valid_actions: Optional[List[int]] = None,
    hand_complete: bool = False,
    winner_info: Optional[Dict] = None
) -> Dict:
    community_cards = []
    if game_state.community_cards:
        community_cards = convert_cards_for_frontend(game_state.community_cards)

    dealer_idx, sb_idx, bb_idx = _get_position_indices(game_state)

    players = []
    for player in game_state.players:
        is_human = player.player_id == human_player_id
        show_cards = hand_complete or is_human
        player_data = serialize_player(player, is_human, show_cards)

        player_data["is_dealer"] = player.player_id == dealer_idx
        player_data["is_small_blind"] = player.player_id == sb_idx
        player_data["is_big_blind"] = player.player_id == bb_idx

        players.append(player_data)

    pot_total = game_state.pot_manager.get_pot_total()
    current_bet = game_state.pot_manager.current_bet
    min_raise = game_state.pot_manager.min_raise

    human_player = game_state.players[human_player_id]
    is_human_turn = (
        not hand_complete and
        game_state.current_player_idx == human_player_id and
        human_player.is_active
    )

    return {
        "hand_number": game_state.hand_number,
        "betting_round": game_state.betting_round.name,
        "pot": pot_total,
        "current_bet": current_bet,
        "min_raise": min_raise,
        "community_cards": community_cards,
        "players": players,
        "current_player_idx": game_state.current_player_idx,
        "is_human_turn": is_human_turn,
        "valid_actions": valid_actions or [],
        "hand_complete": hand_complete,
        "winner_info": winner_info,
        "small_blind": game_state.small_blind,
        "big_blind": game_state.big_blind,
    }
