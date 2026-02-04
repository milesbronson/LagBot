"""
WebSocket handler for real-time game state updates.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.services.game_manager import game_manager

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time game state updates.

    Args:
        websocket: WebSocket connection
        session_id: Game session identifier
    """
    await websocket.accept()

    session = game_manager.get_session(session_id)
    if not session:
        await websocket.send_json({
            "type": "error",
            "message": "Session not found"
        })
        await websocket.close()
        return

    # Add WebSocket to session
    session.add_websocket(websocket)

    try:
        # Send initial state
        from backend.utils.state_serializer import serialize_game_state
        valid_actions = []
        if session.env.game_state.current_player_idx == session.human_player_id:
            valid_actions = session.env.get_valid_actions()

        initial_state = serialize_game_state(
            session.env.game_state,
            session.human_player_id,
            valid_actions=valid_actions,
            hand_complete=False
        )

        await websocket.send_json({
            "type": "connected",
            "state": initial_state
        })

        # Keep connection open and listen for messages
        while True:
            data = await websocket.receive_text()
            # Could handle client-side messages here if needed
            # For now, just keep connection alive

    except WebSocketDisconnect:
        session.remove_websocket(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        session.remove_websocket(websocket)
