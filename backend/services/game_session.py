"""
GameSession wrapper for TexasHoldemEnv with bot agent management.
"""
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.opponent_ppo import OpponentPPO
from src.agents.random_agent import CallAgent, RandomAgent
from backend.utils.state_serializer import serialize_game_state


class GameSession:
    """
    Wraps TexasHoldemEnv for web interface access.
    Manages bot agents and provides JSON-serializable game state.
    """

    def __init__(
        self,
        session_id: str,
        num_opponents: int = 2,
        opponent_type: str = "trained",
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
    ):
        """
        Initialize game session.

        Args:
            session_id: Unique session identifier
            num_opponents: Number of bot opponents (1-9)
            opponent_type: Type of opponents ('trained', 'call', 'random', 'mixed')
            starting_stack: Starting chip stack
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.session_id = session_id
        self.num_opponents = num_opponents
        self.opponent_type = opponent_type
        self.human_player_id = 0  # Human is always player 0

        # Create environment
        self.env = TexasHoldemEnv(
            num_players=num_opponents + 1,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            track_opponents=True,
            raise_bins=[0.5, 1.0, 2.0],
            include_all_in=True
        )

        # Load bot agents
        self.bot_agents = self._load_bot_agents(opponent_type, num_opponents)

        # WebSocket connections for broadcasting
        self.websocket_connections = []

    def _load_bot_agents(self, opponent_type: str, num_opponents: int) -> List:
        """
        Load bot agents based on opponent type.

        Args:
            opponent_type: 'trained', 'call', 'random', or 'mixed'
            num_opponents: Number of opponents

        Returns:
            List of agent objects with select_action(obs) method
        """
        agents = []

        if opponent_type == "trained":
            # Try to load trained PPO models
            models_dir = Path("models/ppo")
            if models_dir.exists():
                model_files = sorted(models_dir.glob("ppo_poker_gen_*.zip"))
                if model_files:
                    # Use the latest trained model for all opponents
                    latest_model = model_files[-1]
                    for i in range(num_opponents):
                        try:
                            agent = OpponentPPO.load(str(latest_model))
                            agents.append(agent)
                        except Exception as e:
                            print(f"Failed to load trained model: {e}, using CallAgent")
                            agents.append(CallAgent())
                else:
                    # No trained models, use call agents
                    agents = [CallAgent() for _ in range(num_opponents)]
            else:
                # No models directory, use call agents
                agents = [CallAgent() for _ in range(num_opponents)]

        elif opponent_type == "call":
            agents = [CallAgent() for _ in range(num_opponents)]

        elif opponent_type == "random":
            agents = [RandomAgent() for _ in range(num_opponents)]

        elif opponent_type == "mixed":
            # Mix of agent types
            for i in range(num_opponents):
                if i % 3 == 0:
                    agents.append(RandomAgent())
                else:
                    agents.append(CallAgent())
        else:
            # Default to call agents
            agents = [CallAgent() for _ in range(num_opponents)]

        return agents

    def start_hand(self) -> Dict:
        """
        Start a new hand.

        Returns:
            Dictionary with initial game state
        """
        obs, info = self.env.reset()

        # Get valid actions for human player
        valid_actions = self.env.get_valid_actions()

        return serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=False
        )

    async def execute_human_action(
        self,
        action_type: int,
        raise_amount: Optional[int] = None
    ) -> Dict:
        """
        Execute human player action, then run bot actions with delays.

        Args:
            action_type: Action ID (0=fold, 1=call, 2+=raise/all-in)
            raise_amount: Raise amount for raise actions

        Returns:
            Dictionary with updated game state
        """
        # Execute human action
        obs, reward, terminated, truncated, info = self.env.step(action_type)
        done = terminated or truncated

        # Broadcast state after human action
        if not done:
            await self._broadcast_current_state()

        # Execute bot actions with delays for better UX
        while not done and self.env.game_state.current_player_idx != self.human_player_id:
            current_player_idx = self.env.game_state.current_player_idx
            current_player = self.env.game_state.players[current_player_idx]

            # Skip folded or all-in players
            if current_player.folded or current_player.is_all_in:
                break

            # Add delay for bot to "think"
            await asyncio.sleep(0.5)

            # Get bot action
            bot_idx = current_player_idx - 1  # Adjust for human at index 0
            if bot_idx >= 0 and bot_idx < len(self.bot_agents):
                bot_agent = self.bot_agents[bot_idx]
                bot_action = bot_agent.select_action(obs)
            else:
                # Fallback to call
                bot_action = 1

            # Execute bot action
            obs, reward, terminated, truncated, info = self.env.step(bot_action)
            done = terminated or truncated

            # Broadcast state after bot action
            await self._broadcast_current_state()

            # Check if betting round changed or hand complete
            if done or self.env.game_state.is_betting_round_complete():
                if not done and not self.env.game_state.is_hand_complete():
                    self.env.game_state.advance_betting_round()
                    await self._broadcast_current_state()
                break

        # Get winner info if hand complete
        winner_info = None
        if done:
            winner_info = info.get('winnings', {})

        # Get valid actions
        valid_actions = []
        if not done and self.env.game_state.current_player_idx == self.human_player_id:
            valid_actions = self.env.get_valid_actions()

        return serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=done,
            winner_info=winner_info
        )

    async def _broadcast_current_state(self):
        """Broadcast current game state to all connected WebSocket clients."""
        valid_actions = []
        is_human_turn = (
            self.env.game_state.current_player_idx == self.human_player_id and
            not self.env.game_state.is_hand_complete()
        )

        if is_human_turn:
            valid_actions = self.env.get_valid_actions()

        state = serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=False
        )

        # Send to all connected WebSocket clients
        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json({"type": "state_update", "state": state})
            except Exception as e:
                print(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

    def add_websocket(self, websocket):
        """Add WebSocket connection for broadcasting."""
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket):
        """Remove WebSocket connection."""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    def get_opponent_stats(self, opponent_id: int) -> Dict:
        """
        Get detailed statistics for an opponent.

        Args:
            opponent_id: Player ID of opponent

        Returns:
            Dictionary with opponent statistics
        """
        if opponent_id == self.human_player_id:
            return {}

        stats = self.env.opponent_tracker.get_opponent_stats(opponent_id)
        return stats

    def get_hand_history(self) -> List[Dict]:
        """
        Get hand history from opponent tracker.

        Returns:
            List of hand history entries
        """
        # Get recent hands from opponent tracker
        # This would need to be implemented in the OpponentTracker
        return []
