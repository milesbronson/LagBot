"""
Pydantic response models for API endpoints.
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class PlayerResponse(BaseModel):
    """Player information."""
    player_id: int
    name: str
    stack: int
    bet: int
    is_active: bool
    is_all_in: bool
    is_folded: bool
    hole_cards: Optional[List[str]] = None
    is_human: bool
    is_dealer: bool
    is_small_blind: bool
    is_big_blind: bool


class GameStateResponse(BaseModel):
    """Complete game state."""
    hand_number: int
    betting_round: str
    pot: int
    current_bet: int
    min_raise: int
    community_cards: List[str]
    players: List[PlayerResponse]
    current_player_idx: int
    is_human_turn: bool
    valid_actions: List[int]
    hand_complete: bool
    winner_info: Optional[Dict[str, Any]] = None
    small_blind: int
    big_blind: int


class NewGameResponse(BaseModel):
    """Response for new game creation."""
    session_id: str
    state: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
