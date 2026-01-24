"""Cache backend implementations for CachingMiddleware.

This module provides pluggable cache backends:
- CacheBackend: Protocol defining the cache interface
- MemoryCacheBackend: In-memory cache with TTL and LRU eviction
- RedisCacheBackend: Distributed Redis cache for multi-process scenarios
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from ..logging import get_logger

logger = get_logger("middleware")

__all__ = [
    "CacheBackend",
    "CacheEntry",
    "MemoryCacheBackend",
    "RedisCacheBackend",
]


class CacheBackend(Protocol):
    """Protocol for cache backend implementations.

    Backends must implement async get/set operations with optional TTL.
    Used by CachingMiddleware for pluggable cache storage.

    This protocol enables structural typing, allowing any class that
    implements these methods to be used as a cache backend without
    explicit inheritance.
    """

    async def get(self, key: str) -> Optional[str]:
        """Retrieve cached value by key.

        Args:
            key: Cache key (SHA256 hash string)

        Returns:
            Cached JSON string if found and not expired, None otherwise
        """
        ...

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store value in cache with optional TTL.

        Args:
            key: Cache key (SHA256 hash string)
            value: JSON-serialized result to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        ...

    async def delete(self, key: str) -> None:
        """Remove key from cache.

        Args:
            key: Cache key to remove
        """
        ...

    def size(self) -> int:
        """Return current number of cached entries.

        Returns:
            Number of entries in cache, or -1 if unknown (e.g., Redis)
        """
        ...

    async def clear(self) -> None:
        """Remove all cached entries."""
        ...


@dataclass
class CacheEntry:
    """Entry in the memory cache with expiration tracking."""

    value: str
    expires_at: Optional[float]  # Unix timestamp, None = never expires


class MemoryCacheBackend:
    """In-memory cache backend with TTL and LRU eviction.

    Thread-safe for single-process async use. Uses OrderedDict for
    LRU eviction when max_size is exceeded.

    Example:
        >>> backend = MemoryCacheBackend(max_size=100)
        >>> await backend.set("key", "value", ttl=300)
        >>> cached = await backend.get("key")
    """

    def __init__(self, max_size: int = 100) -> None:
        """Initialize memory cache backend.

        Args:
            max_size: Maximum number of entries before LRU eviction
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    async def get(self, key: str) -> Optional[str]:
        """Retrieve value if exists and not expired."""
        if key not in self._cache:
            return None

        entry = self._cache[key]

        # Check TTL expiration
        if entry.expires_at is not None and time.time() > entry.expires_at:
            del self._cache[key]
            return None

        # Move to end for LRU tracking
        self._cache.move_to_end(key)
        return entry.value

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store value with optional TTL."""
        expires_at = time.time() + ttl if ttl is not None else None

        # If key exists, remove it first to update position
        if key in self._cache:
            del self._cache[key]

        # Evict oldest entries if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[key] = CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)

    async def clear(self) -> None:
        """Clear all entries."""
        self._cache.clear()


class RedisCacheBackend:
    """Redis cache backend for distributed caching with TTL.

    Provides shared cache across multiple processes/instances.
    Follows patterns from arbiter_ai/storage/redis.py for async
    operations and connection lifecycle management.

    Example:
        >>> backend = RedisCacheBackend(redis_url="redis://localhost:6379")
        >>> async with backend:
        ...     await backend.set("key", "value", ttl=300)
        ...     cached = await backend.get("key")
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_prefix: str = "arbiter:cache:",
        default_ttl: int = 3600,
    ) -> None:
        """Initialize Redis cache backend.

        Args:
            redis_url: Redis connection string (defaults to REDIS_URL env var)
            key_prefix: Prefix for all cache keys in Redis
            default_ttl: Default TTL in seconds if not specified per-key

        Raises:
            ValueError: If redis_url not provided and REDIS_URL not set
        """
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        if not self._redis_url:
            raise ValueError(
                "Redis URL required: provide redis_url or set REDIS_URL env var"
            )

        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._client: Optional[Any] = None  # redis.Redis[str]
        self._connected = False

    def _make_key(self, key: str) -> str:
        """Generate prefixed Redis key."""
        return f"{self._key_prefix}{key}"

    async def connect(self) -> None:
        """Establish Redis connection."""
        if self._connected:
            return

        # redis_url is validated in __init__, safe to assert
        assert self._redis_url is not None

        try:
            import redis.asyncio as redis

            self._client = await redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._client.ping()
            self._connected = True
            logger.info("Redis cache backend connected")

        except ImportError:
            raise ImportError(
                "Redis backend requires redis package. "
                "Install with: pip install redis"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Redis cache backend disconnected")

    async def __aenter__(self) -> "RedisCacheBackend":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def get(self, key: str) -> Optional[str]:
        """Retrieve value from Redis."""
        if not self._connected:
            await self.connect()

        assert self._client is not None  # Guaranteed after connect()

        try:
            redis_key = self._make_key(key)
            result: Optional[str] = await self._client.get(redis_key)
            return result
        except Exception as e:
            logger.warning(f"Redis get failed for {key}: {e}")
            return None

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Store value in Redis with TTL."""
        if not self._connected:
            await self.connect()

        assert self._client is not None  # Guaranteed after connect()

        try:
            redis_key = self._make_key(key)
            effective_ttl = ttl if ttl is not None else self._default_ttl
            await self._client.setex(redis_key, effective_ttl, value)
        except Exception as e:
            logger.warning(f"Redis set failed for {key}: {e}")

    async def delete(self, key: str) -> None:
        """Remove key from Redis."""
        if not self._connected:
            await self.connect()

        assert self._client is not None  # Guaranteed after connect()

        try:
            redis_key = self._make_key(key)
            await self._client.delete(redis_key)
        except Exception as e:
            logger.warning(f"Redis delete failed for {key}: {e}")

    def size(self) -> int:
        """Return approximate cache size.

        Note: Returns -1 for Redis as getting exact size with prefix
        would require scanning all keys, which is expensive.
        """
        return -1

    async def clear(self) -> None:
        """Clear all cache entries with this backend's prefix."""
        if not self._connected:
            await self.connect()

        assert self._client is not None  # Guaranteed after connect()

        try:
            pattern = f"{self._key_prefix}*"
            cursor: int = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning(f"Redis clear failed: {e}")
