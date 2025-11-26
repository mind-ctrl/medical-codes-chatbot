"""
Database Connection Management
Async connection pool for PostgreSQL with pgvector
"""

import asyncpg
from typing import Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)


class Database:
    """Database connection pool manager"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Create connection pool"""
        logger.info("Creating database connection pool...")
        self.pool = await asyncpg.create_pool(
            settings.NEON_DATABASE_URL,
            min_size=5,
            max_size=settings.DB_POOL_SIZE,
            command_timeout=60
        )
        logger.info(f"Connection pool created (max size: {settings.DB_POOL_SIZE})")

    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def fetch(self, query: str, *args):
        """
        Fetch multiple rows

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of records
        """
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        """
        Fetch single row

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single record or None
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        """
        Fetch single value

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value or None
        """
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args):
        """
        Execute query without return

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query execution result
        """
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def executemany(self, query: str, args):
        """
        Execute query with multiple parameter sets

        Args:
            query: SQL query
            args: List of parameter tuples

        Returns:
            Query execution result
        """
        async with self.pool.acquire() as conn:
            return await conn.executemany(query, args)


# Global database instance
db = Database()
