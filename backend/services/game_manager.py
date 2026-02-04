"""
Game session manager for handling multiple game sessions.
"""
import uuid
from typing import Dict, Optional
from backend.services.game_session import GameSession


class GameManager:
    """
    Manages multiple game sessions.
    Singleton pattern for global access.
    """

    def __init__(self):
        self.sessions: Dict[str, GameSession] = {}

    def create_session(
        self,
        num_opponents: int = 2,
        opponent_type: str = "trained",
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10
    ) -> GameSession:
        """
        Create a new game session.

        Args:
            num_opponents: Number of bot opponents
            opponent_type: Type of opponents
            starting_stack: Starting chip stack
            small_blind: Small blind amount
            big_blind: Big blind amount

        Returns:
            GameSession instance
        """
        session_id = str(uuid.uuid4())
        session = GameSession(
            session_id=session_id,
            num_opponents=num_opponents,
            opponent_type=opponent_type,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[GameSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            GameSession if found, None otherwise
        """
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str):
        """
        Delete a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_active_sessions(self) -> int:
        """Get count of active sessions."""
        return len(self.sessions)


# Global game manager instance
game_manager = GameManager()
