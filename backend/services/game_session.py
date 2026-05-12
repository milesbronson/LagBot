"""
GameSession wrapper for TexasHoldemEnv with bot agent management.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

from src.poker_env.texas_holdem_env import TexasHoldemEnv
from src.agents.opponent_ppo import OpponentPPO
from src.agents.random_agent import CallAgent, RandomAgent
from backend.utils.state_serializer import serialize_game_state
from backend.utils.card_converter import convert_cards_for_frontend


class GameSession:
    def __init__(
        self,
        session_id: str,
        num_opponents: int = 2,
        opponent_type: str = "trained",
        starting_stack: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
    ):
        self.session_id = session_id
        self.num_opponents = num_opponents
        self.opponent_type = opponent_type
        self.human_player_id = 0

        self.env = TexasHoldemEnv(
            num_players=num_opponents + 1,
            starting_stack=starting_stack,
            small_blind=small_blind,
            big_blind=big_blind,
            track_opponents=True,
            raise_bins=[0.5, 1.0, 2.0],
            include_all_in=True
        )

        self.bot_agents = self._load_bot_agents(opponent_type, num_opponents)
        self.websocket_connections = []
        self._hand_actions: List[Dict] = []
        self._hand_starting_stacks: Dict[int, int] = {}

    def _find_latest_model(self) -> Optional[Path]:
        models_dir = Path("models")
        if not models_dir.exists():
            return None

        best_path = None
        best_steps = -1
        for zip_file in models_dir.rglob("model_*_steps.zip"):
            try:
                steps = int(zip_file.stem.split("_")[1])
                if steps > best_steps:
                    best_steps = steps
                    best_path = zip_file
            except (ValueError, IndexError):
                continue

        if best_path is None:
            for zip_file in models_dir.rglob("*.zip"):
                return zip_file

        return best_path

    def _load_bot_agents(self, opponent_type: str, num_opponents: int) -> Dict[int, object]:
        agents: Dict[int, object] = {}

        if opponent_type == "trained":
            model_path = self._find_latest_model()
            if model_path:
                for i in range(num_opponents):
                    player_id = i + 1
                    try:
                        agent = OpponentPPO(str(model_path))
                        agents[player_id] = agent
                    except Exception as e:
                        print(f"Failed to load model for player {player_id}: {e}")
                        agents[player_id] = CallAgent()
            else:
                for i in range(num_opponents):
                    agents[i + 1] = CallAgent()

        elif opponent_type == "call":
            for i in range(num_opponents):
                agents[i + 1] = CallAgent()

        elif opponent_type == "random":
            for i in range(num_opponents):
                agents[i + 1] = RandomAgent()

        elif opponent_type == "mixed":
            for i in range(num_opponents):
                player_id = i + 1
                if i % 3 == 0:
                    agents[player_id] = RandomAgent()
                else:
                    agents[player_id] = CallAgent()
        else:
            for i in range(num_opponents):
                agents[i + 1] = CallAgent()

        return agents

    def _snapshot_starting_stacks(self):
        self._hand_starting_stacks = {
            p.player_id: p.starting_stack_this_hand
            for p in self.env.game_state.players
        }

    def _record_action(self, player_id: int, player_name: str, action: str, amount: int):
        self._hand_actions.append({
            "player_id": player_id,
            "player_name": player_name,
            "action_type": action,
            "amount": amount,
            "betting_round": self.env.game_state.betting_round.name,
            "pot_after": self.env.game_state.pot_manager.get_pot_total(),
        })

    async def _save_hand_to_db(self, winner_info: Dict[int, int]):
        try:
            from backend.db.hand_history import save_hand

            community = []
            if self.env.game_state.community_cards:
                community = convert_cards_for_frontend(self.env.game_state.community_cards)

            players = []
            for p in self.env.game_state.players:
                hole_cards = None
                if p.hand and len(p.hand) == 2:
                    hole_cards = convert_cards_for_frontend(p.hand)
                players.append({
                    "player_id": p.player_id,
                    "player_name": "You" if p.player_id == self.human_player_id else p.name,
                    "hole_cards": hole_cards,
                    "starting_stack": self._hand_starting_stacks.get(p.player_id, 0),
                    "ending_stack": p.stack,
                    "is_human": p.player_id == self.human_player_id,
                    "position": None,
                })

            await save_hand(
                session_id=self.session_id,
                hand_number=self.env.game_state.hand_number,
                num_players=len(self.env.game_state.players),
                small_blind=self.env.game_state.small_blind,
                big_blind=self.env.game_state.big_blind,
                community_cards=community,
                pot=self.env.game_state.pot_manager.get_pot_total(),
                winner_info=winner_info,
                players=players,
                actions=self._hand_actions,
            )
        except Exception as e:
            print(f"Failed to save hand to DB: {e}")

    async def start_hand(self) -> Dict:
        self._hand_actions = []
        obs, info = self.env.reset()
        self._snapshot_starting_stacks()
        done = False
        logger.info(f"[{self.session_id}] Hand started. Current player: {self.env.game_state.current_player_idx}, Human: {self.human_player_id}")

        if self.env.game_state.current_player_idx != self.human_player_id:
            obs, done, info = await self._run_bot_loop(obs, done, info)

        winner_info = None
        if done:
            winner_info = info.get('winnings', {})
            logger.info(f"[{self.session_id}] Hand complete during start. Winners: {winner_info}")
            await self._save_hand_to_db(winner_info)
            await self._broadcast_current_state(hand_complete=True, winner_info=winner_info)

        valid_actions = []
        if not done and self.env.game_state.current_player_idx == self.human_player_id:
            valid_actions = self.env.get_valid_actions()

        state = serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=done,
            winner_info=winner_info
        )
        logger.info(f"[{self.session_id}] start_hand returning hand_complete={done}, is_human_turn={state.get('is_human_turn')}")
        return state

    async def execute_human_action(
        self,
        action_type: int,
        raise_amount: Optional[int] = None
    ) -> Dict:
        logger.info(f"[{self.session_id}] Human action: type={action_type}, raise_amount={raise_amount}")

        if action_type == 0:
            obs, reward, terminated, truncated, info = self.env.step_with_amount(0)
        elif action_type == 1:
            obs, reward, terminated, truncated, info = self.env.step_with_amount(1)
        elif raise_amount is not None:
            obs, reward, terminated, truncated, info = self.env.step_with_amount(2, raise_amount)
        else:
            obs, reward, terminated, truncated, info = self.env.step(action_type)
        done = terminated or truncated

        human_action = {
            "player_id": self.human_player_id,
            "player_name": "You",
            "action": info.get("action", "unknown"),
            "amount": self.env.game_state.players[self.human_player_id].current_bet,
        }
        self._record_action(
            self.human_player_id, "You",
            info.get("action", "unknown"),
            self.env.game_state.players[self.human_player_id].current_bet,
        )
        logger.info(f"[{self.session_id}] Human did: {info.get('action', 'unknown')}, done={done}")

        if not done:
            await self._broadcast_current_state(last_action=human_action)

        obs, done, info = await self._run_bot_loop(obs, done, info)

        winner_info = None
        if done:
            winner_info = info.get('winnings', {})
            logger.info(f"[{self.session_id}] Hand complete. Winners: {winner_info}")
            await self._save_hand_to_db(winner_info)
            await self._broadcast_current_state(hand_complete=True, winner_info=winner_info)

        valid_actions = []
        if not done and self.env.game_state.current_player_idx == self.human_player_id:
            valid_actions = self.env.get_valid_actions()

        state = serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=done,
            winner_info=winner_info
        )
        logger.info(f"[{self.session_id}] execute_human_action returning hand_complete={done}, is_human_turn={state.get('is_human_turn')}")
        return state

    async def _run_bot_loop(self, obs, done: bool, info: dict):
        while not done and self.env.game_state.current_player_idx != self.human_player_id:
            current_player_id = self.env.game_state.current_player_idx
            player_name = self.env.game_state.players[current_player_id].name

            bot_agent = self.bot_agents.get(current_player_id)
            if bot_agent is None:
                bot_action = 1
            else:
                bot_action = bot_agent.select_action(obs)

            await asyncio.sleep(0.5)

            obs, reward, terminated, truncated, info = self.env.step(bot_action)
            done = terminated or truncated

            action_str = info.get("action", "unknown")
            bet_amount = self.env.game_state.players[current_player_id].current_bet
            self._record_action(current_player_id, player_name, action_str, bet_amount)
            logger.info(f"[{self.session_id}] Bot {player_name} (id={current_player_id}): {action_str}, done={done}")

            last_action = {
                "player_id": current_player_id,
                "player_name": player_name,
                "action": action_str,
                "amount": bet_amount,
            }
            await self._broadcast_current_state(last_action=last_action, hand_complete=done,
                                                 winner_info=info.get('winnings') if done else None)

        return obs, done, info

    async def _broadcast_current_state(self, last_action: Optional[Dict] = None,
                                        hand_complete: bool = False,
                                        winner_info: Optional[Dict] = None):
        valid_actions = []
        if not hand_complete and self.env.game_state.current_player_idx == self.human_player_id:
            valid_actions = self.env.get_valid_actions()

        state = serialize_game_state(
            self.env.game_state,
            self.human_player_id,
            valid_actions=valid_actions,
            hand_complete=hand_complete,
            winner_info=winner_info
        )

        message = {"type": "state_update", "state": state}
        if last_action:
            message["last_action"] = last_action

        logger.debug(f"[{self.session_id}] Broadcasting: hand_complete={hand_complete}, is_human_turn={state.get('is_human_turn')}")

        disconnected = []
        for ws in self.websocket_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.websocket_connections.remove(ws)

    def add_websocket(self, websocket):
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    def get_opponent_stats(self, opponent_id: int) -> Dict:
        if opponent_id == self.human_player_id:
            return {}
        return self.env.opponent_tracker.get_opponent_stats(opponent_id)
