"""Storage backends for evaluation results.

Available backends:
- PostgreSQL: Persistent storage with arbiter schema (requires DATABASE_URL)
- Redis: Fast caching with TTL (requires REDIS_URL)

Setup:
    1. Set DATABASE_URL and/or REDIS_URL in .env
    2. For PostgreSQL: Run migrations with `alembic upgrade head`
    3. Use storage backends in evaluate() or batch_evaluate() calls

Example:
    >>> from arbiter_ai import evaluate, batch_evaluate
    >>> from arbiter_ai.storage import PostgresStorage, RedisStorage
    >>>
    >>> # Single evaluation with PostgreSQL storage
    >>> async with PostgresStorage() as storage:
    ...     result = await evaluate(
    ...         output="Paris is the capital of France",
    ...         reference="The capital of France is Paris",
    ...         evaluators=["semantic"],
    ...         storage=storage,
    ...         storage_metadata={"user_id": "user123"}
    ...     )
    ...     # Result ID available at result.metadata["storage_result_id"]
    >>>
    >>> # Batch evaluation with Redis caching
    >>> async with RedisStorage(ttl=3600) as storage:
    ...     results = await batch_evaluate(
    ...         items=[{"output": "...", "reference": "..."}],
    ...         storage=storage,
    ...         storage_metadata={"experiment": "v1"}
    ...     )
    ...     # Batch ID available at results.metadata["storage_batch_id"]
"""

from arbiter_ai.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
    StorageError,
)

__all__ = [
    "StorageBackend",
    "StorageError",
    "ConnectionError",
    "SaveError",
    "RetrievalError",
    "PostgresStorage",
    "RedisStorage",
]


def __getattr__(name: str) -> type:
    """Lazy import for optional storage backends.

    This allows importing from arbiter_ai.storage without requiring
    optional dependencies (asyncpg, redis) to be installed.
    """
    if name == "PostgresStorage":
        try:
            from arbiter_ai.storage.postgres import PostgresStorage

            return PostgresStorage
        except ImportError as e:
            raise ImportError(
                "PostgresStorage requires asyncpg. Install with: pip install arbiter[postgres]"
            ) from e
    elif name == "RedisStorage":
        try:
            from arbiter_ai.storage.redis import RedisStorage

            return RedisStorage
        except ImportError as e:
            raise ImportError(
                "RedisStorage requires redis. Install with: pip install arbiter[redis]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
