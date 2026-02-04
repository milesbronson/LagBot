"""
REST API routes for poker game.
"""
from fastapi import APIRouter, HTTPException
from backend.models.requests import NewGameRequest, ActionRequest
from backend.models.responses import NewGameResponse, ErrorResponse
from backend.services.game_manager import game_manager

router = APIRouter()


@router.post("/game/new", response_model=NewGameResponse)
async def create_game(request: NewGameRequest):
    """
    Create a new game session with bot opponents.

    Args:
        request: New game configuration

    Returns:
        Session ID and initial game state
    """
    try:
        session = game_manager.create_session(
            num_opponents=request.num_opponents,
            opponent_type=request.opponent_type,
            starting_stack=request.starting_stack,
            small_blind=request.small_blind,
            big_blind=request.big_blind
        )

        # Start first hand
        state = session.start_hand()

        return {
            "session_id": session.session_id,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/{session_id}/action")
async def submit_action(session_id: str, request: ActionRequest):
    """
    Submit player action and execute bot responses.

    Args:
        session_id: Session identifier
        request: Action details

    Returns:
        Updated game state
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = await session.execute_human_action(
            request.action_type,
            request.raise_amount
        )
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/game/{session_id}/new-hand")
async def start_new_hand(session_id: str):
    """
    Start a new hand in the current session.

    Args:
        session_id: Session identifier

    Returns:
        Initial state for new hand
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        state = session.start_hand()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/game/{session_id}/state")
async def get_game_state(session_id: str):
    """
    Get current game state.

    Args:
        session_id: Session identifier

    Returns:
        Current game state
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        valid_actions = []
        if session.env.game_state.current_player_idx == session.human_player_id:
            valid_actions = session.env.get_valid_actions()

        from backend.utils.state_serializer import serialize_game_state
        state = serialize_game_state(
            session.env.game_state,
            session.human_player_id,
            valid_actions=valid_actions,
            hand_complete=session.env.game_state.is_hand_complete()
        )
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/game/{session_id}/opponent-stats/{player_id}")
async def get_opponent_stats(session_id: str, player_id: int):
    """
    Get detailed statistics for an opponent.

    Args:
        session_id: Session identifier
        player_id: Player ID

    Returns:
        Opponent statistics
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        stats = session.get_opponent_stats(player_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/game/{session_id}")
async def delete_game(session_id: str):
    """
    Delete a game session.

    Args:
        session_id: Session identifier

    Returns:
        Success message
    """
    session = game_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    game_manager.delete_session(session_id)
    return {"message": "Session deleted successfully"}
