"""
Utilities for serializing game state to JSON format.
"""
from typing import Dict, List, Optional
from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.poker_env.game_state import GameState, Player, BettingRound
from backend.utils.card_converter import convert_cards_for_frontend


def serialize_player(player: Player, is_human: bool, show_hole_cards: bool = False) -> Dict:
    """
    Serialize player data for frontend.

    Args:
        player: Player object
        is_human: Whether this is the human player
        show_hole_cards: Whether to reveal hole cards (showdown or human player)

    Returns:
        Dictionary with player data
    """
    hole_cards = None
    if show_hole_cards or is_human:
        if player.hole_cards and len(player.hole_cards) == 2:
            hole_cards = convert_cards_for_frontend(player.hole_cards)

    return {
        "player_id": player.player_id,
        "name": f"Player {player.player_id}" if not is_human else "You",
        "stack": player.stack,
        "bet": player.bet,
        "is_active": player.is_active,
        "is_all_in": player.is_all_in,
        "is_folded": player.folded,
        "hole_cards": hole_cards,
        "is_human": is_human,
        "is_dealer": False,  # Set by caller
        "is_small_blind": False,  # Set by caller
        "is_big_blind": False,  # Set by caller
    }


def serialize_game_state(
    game_state: GameState,
    human_player_id: int,
    valid_actions: Optional[List[int]] = None,
    hand_complete: bool = False,
    winner_info: Optional[Dict] = None
) -> Dict:
    """
    Serialize complete game state for frontend.

    Args:
        game_state: Current game state
        human_player_id: ID of human player
        valid_actions: List of valid action IDs
        hand_complete: Whether hand is complete
        winner_info: Winner information for showdown

    Returns:
        Dictionary with complete game state
    """
    # Convert community cards
    community_cards = []
    if game_state.community_cards:
        community_cards = convert_cards_for_frontend(game_state.community_cards)

    # Serialize players
    players = []
    for player in game_state.players:
        is_human = player.player_id == human_player_id
        show_cards = hand_complete or is_human
        player_data = serialize_player(player, is_human, show_cards)

        # Add position markers
        player_data["is_dealer"] = player.player_id == game_state.dealer_idx
        player_data["is_small_blind"] = player.player_id == game_state.small_blind_idx
        player_data["is_big_blind"] = player.player_id == game_state.big_blind_idx

        players.append(player_data)

    # Get pot info
    pot_total = game_state.pot_manager.get_pot_total()
    current_bet = game_state.pot_manager.current_bet
    min_raise = game_state.pot_manager.min_raise

    # Determine if it's human's turn
    is_human_turn = (
        not hand_complete and
        game_state.current_player_idx == human_player_id and
        game_state.players[human_player_id].is_active and
        not game_state.players[human_player_id].folded
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
