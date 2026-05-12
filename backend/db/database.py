import os
from typing import Optional
import asyncpg
from pathlib import Path

_pool: Optional[asyncpg.Pool] = None

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://lagbot:lagbot@localhost:5432/lagbot"
)


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        await _init_schema(_pool)
    return _pool


async def _init_schema(pool: asyncpg.Pool):
    schema_path = Path(__file__).parent / "schema.sql"
    schema_sql = schema_path.read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
