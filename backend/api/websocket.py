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
        # Send the REAL current state — hardcoding hand_complete=False here
        # used to strand reconnecting clients in "waiting" with no modal —
        # plus this hand's action log so a client that connected after the
        # opening bot actions (always the case on hand #1) can backfill it.
        await websocket.send_json({
            "type": "connected",
            "state": session._current_state_snapshot(),
            "hand_actions": session._hand_actions,
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
