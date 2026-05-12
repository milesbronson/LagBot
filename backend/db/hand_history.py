from typing import Dict, List, Optional
import asyncpg
from backend.db.database import get_pool


async def save_hand(
    session_id: str,
    hand_number: int,
    num_players: int,
    small_blind: int,
    big_blind: int,
    community_cards: List[str],
    pot: int,
    winner_info: Dict[int, int],
    players: List[Dict],
    actions: List[Dict],
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            winner_ids = [pid for pid, amt in winner_info.items() if amt > 0]
            winner_amounts = [winner_info[pid] for pid in winner_ids]

            hand_id = await conn.fetchval(
                """
                INSERT INTO hands (session_id, hand_number, num_players, small_blind,
                                   big_blind, community_cards, pot, winner_ids, winner_amounts)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
                """,
                session_id, hand_number, num_players, small_blind, big_blind,
                community_cards, pot, winner_ids, winner_amounts,
            )

            for p in players:
                await conn.execute(
                    """
                    INSERT INTO hand_players (hand_id, player_id, player_name, hole_cards,
                                              starting_stack, ending_stack, is_human, position)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    hand_id, p["player_id"], p["player_name"], p.get("hole_cards"),
                    p["starting_stack"], p["ending_stack"], p["is_human"], p.get("position"),
                )

            for i, a in enumerate(actions):
                await conn.execute(
                    """
                    INSERT INTO hand_actions (hand_id, action_order, player_id, player_name,
                                              betting_round, action_type, amount, pot_after)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    hand_id, i, a["player_id"], a["player_name"],
                    a.get("betting_round"), a["action_type"], a.get("amount", 0),
                    a.get("pot_after"),
                )

    return hand_id


async def get_session_hands(session_id: str, limit: int = 20) -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, hand_number, num_players, small_blind, big_blind,
                   community_cards, pot, winner_ids, winner_amounts, created_at
            FROM hands
            WHERE session_id = $1
            ORDER BY hand_number DESC
            LIMIT $2
            """,
            session_id, limit,
        )

        hands = []
        for row in rows:
            hand = dict(row)
            hand["created_at"] = hand["created_at"].isoformat()

            actions = await conn.fetch(
                """
                SELECT action_order, player_id, player_name, betting_round,
                       action_type, amount, pot_after
                FROM hand_actions
                WHERE hand_id = $1
                ORDER BY action_order
                """,
                row["id"],
            )
            hand["actions"] = [dict(a) for a in actions]

            players = await conn.fetch(
                """
                SELECT player_id, player_name, hole_cards, starting_stack,
                       ending_stack, is_human, position
                FROM hand_players
                WHERE hand_id = $1
                ORDER BY player_id
                """,
                row["id"],
            )
            hand["players"] = [dict(p) for p in players]

            hands.append(hand)

        return hands


async def get_all_hands(limit: int = 50) -> List[Dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, session_id, hand_number, num_players, pot,
                   winner_ids, winner_amounts, created_at
            FROM hands
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [
            {**dict(r), "created_at": r["created_at"].isoformat()}
            for r in rows
        ]
