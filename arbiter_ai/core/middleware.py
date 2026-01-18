"""Middleware system for adding cross-cutting functionality to Arbiter.

This module provides a flexible middleware pipeline that allows you to
add functionality like logging, metrics, caching, and rate limiting
without modifying core Arbiter code.

## Middleware Pattern:

Middleware components form a chain where each component can:
1. Process the request before passing it on
2. Modify the request or context
3. Handle the response after processing
4. Short-circuit the chain if needed

## Built-in Middleware:

- **LoggingMiddleware**: Logs all evaluation operations
- **MetricsMiddleware**: Collects performance metrics
- **CachingMiddleware**: Caches evaluation results
- **RateLimitingMiddleware**: Limits request rate

## Usage:

    >>> from arbiter import evaluate, MiddlewarePipeline
    >>> from arbiter_ai.core.middleware import LoggingMiddleware, MetricsMiddleware
    >>>
    >>> # Create middleware pipeline
    >>> pipeline = MiddlewarePipeline([
    ...     LoggingMiddleware(log_level="DEBUG"),
    ...     MetricsMiddleware(),
    ... ])
    >>>
    >>> # Use with evaluate()
    >>> result = await evaluate(
    ...     output="...",
    ...     reference="...",
    ...     middleware=pipeline
    ... )
    >>>
    >>> # Access metrics
    >>> metrics = pipeline.get_middleware(MetricsMiddleware)
    >>> print(metrics.get_metrics())

## Custom Middleware:

    >>> class MyMiddleware(Middleware):
    ...     async def process(self, output, reference, next_handler, context):
    ...         # Pre-processing
    ...         print(f"Evaluating: {output[:50]}...")
    ...
    ...         # Call next in chain
    ...         result = await next_handler(output, reference)
    ...
    ...         # Post-processing
    ...         print(f"Completed with score: {result.overall_score}")
    ...
    ...         return result
"""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Union

from .logging import get_logger
from .models import ComparisonResult, EvaluationResult
from .type_defs import MiddlewareContext

# Type alias for results that can flow through middleware
MiddlewareResult = Union[EvaluationResult, ComparisonResult]

logger = get_logger("middleware")

__all__ = [
    "Middleware",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    "MiddlewarePipeline",
    "MiddlewareResult",
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    "monitor",
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


class Middleware(ABC):
    """Abstract base class for all middleware components.

    Middleware allows you to intercept and modify the evaluation process
    without changing core Arbiter logic. Each middleware can inspect or
    modify the request, add to the context, and process the response.

    The middleware pattern enables:
    - Logging and debugging
    - Performance monitoring
    - Caching and optimization
    - Security and rate limiting
    - Request/response transformation

    Example:
        >>> class TimingMiddleware(Middleware):
        ...     async def process(self, output, reference, next_handler, context):
        ...         start = time.time()
        ...         result = await next_handler(output, reference)
        ...         elapsed = time.time() - start
        ...         print(f"Evaluation took {elapsed:.2f} seconds")
        ...         return result
    """

    @abstractmethod
    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Process the request through this middleware.

        This is the main method that each middleware must implement. It
        receives the request, can perform pre-processing, must call the
        next handler, and can perform post-processing.

        Args:
            output: The LLM output to be evaluated. Middleware can modify
                this before passing it to the next handler.
            reference: Reference text for comparison (may be None for
                reference-free evaluation).
            next_handler: Async callable representing the next middleware
                in the chain or the final evaluate handler. Must be called
                to continue processing.
            context: Mutable dictionary shared between all middleware in
                the pipeline. Used to pass data between middleware components.
                Common keys include 'evaluators', 'metrics', 'config'.

        Returns:
            MiddlewareResult (EvaluationResult or ComparisonResult) from
            the evaluation process. Middleware can inspect or modify this
            before returning.

        Raises:
            Any exception from the evaluation process. Middleware can
            catch and handle exceptions or let them propagate.

        Example:
            >>> async def process(self, output, reference, next_handler, context):
            ...     # Pre-processing
            ...     context['start_time'] = time.time()
            ...
            ...     # Must call next handler
            ...     result = await next_handler(output, reference)
            ...
            ...     # Post-processing
            ...     context['duration'] = time.time() - context['start_time']
            ...
            ...     return result
        """


class LoggingMiddleware(Middleware):
    """Middleware that logs all evaluation operations.

    Provides detailed logging of the evaluation process including:
    - Output and reference text (truncated)
    - Configuration (evaluators, metrics)
    - Processing time
    - Results (scores, pass/fail status)
    - Errors with full context

    Useful for debugging, monitoring, and understanding how
    evaluations are performed.

    Example:
        >>> # Basic usage
        >>> middleware = LoggingMiddleware()
        >>>
        >>> # With custom log level
        >>> middleware = LoggingMiddleware(log_level="DEBUG")
        >>>
        >>> # In pipeline
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(LoggingMiddleware())
    """

    def __init__(self, log_level: str = "INFO"):
        """Initialize logging middleware with specified level.

        Args:
            log_level: Logging level as string. Valid values are:
                - "DEBUG": Detailed information for debugging
                - "INFO": General informational messages (default)
                - "WARNING": Warning messages
                - "ERROR": Error messages only
                Case-insensitive.
        """
        self.log_level = getattr(logging, log_level.upper())

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Log the evaluation process."""
        start_time = time.time()

        logger.log(self.log_level, f"Starting evaluation for output: {output[:100]}...")
        if reference:
            logger.log(self.log_level, f"Reference: {reference[:100]}...")

        metrics_list = context.get("metrics", [])
        logger.log(
            self.log_level,
            f"Context: evaluators={context.get('evaluators', [])}, "
            f"metrics={len(metrics_list) if metrics_list else 0}",
        )

        try:
            result: MiddlewareResult = await next_handler(output, reference)

            elapsed = time.time() - start_time
            logger.log(self.log_level, f"Evaluation completed in {elapsed:.2f}s")

            # Handle both EvaluationResult and ComparisonResult
            if isinstance(result, EvaluationResult):
                logger.log(
                    self.log_level,
                    f"Result: overall_score={result.overall_score:.3f}, "
                    f"passed={result.passed}, "
                    f"num_scores={len(result.scores)}",
                )
            elif isinstance(result, ComparisonResult):
                logger.log(
                    self.log_level,
                    f"Result: winner={result.winner}, "
                    f"confidence={result.confidence:.3f}",
                )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"Evaluation failed after {elapsed:.2f}s: {type(e).__name__}: {e!s}"
            )
            raise


class MetricsMiddleware(Middleware):
    """Middleware that collects detailed metrics about evaluations.

    Tracks comprehensive metrics including:
    - Total requests and success rate
    - Processing time statistics
    - Score distributions
    - Token usage and costs
    - Error rates and types

    Metrics are accumulated across all requests and can be retrieved
    for analysis or monitoring dashboards.

    Example:
        >>> metrics_mw = MetricsMiddleware()
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(metrics_mw)
        >>>
        >>> # Process some requests
        >>> for output, ref in test_cases:
        ...     await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Get metrics
        >>> stats = metrics_mw.get_metrics()
        >>> print(f"Average time: {stats['average_time']:.2f}s")
        >>> print(f"Average score: {stats['average_score']:.3f}")
    """

    def __init__(self) -> None:
        """Initialize metrics collection with zero counters.

        Creates a metrics dictionary that tracks:
        - total_requests: Number of evaluate() calls
        - total_time: Cumulative processing time
        - average_score: Running average of overall scores
        - errors: Count of failed requests
        - llm_calls: Total LLM API calls made
        - tokens_used: Total tokens consumed
        """
        self.metrics = {
            "total_requests": 0,
            "total_time": 0.0,
            "average_score": 0.0,
            "errors": 0,
            "llm_calls": 0,
            "tokens_used": 0,
            "passed_count": 0,
        }

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Collect metrics about the evaluation."""
        start_time = time.time()
        self.metrics["total_requests"] += 1

        try:
            # Track LLM calls via context
            initial_llm_calls = context.get("llm_calls", 0)

            result: MiddlewareResult = await next_handler(output, reference)

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["total_time"] += elapsed

            # Update average score based on result type
            total = self.metrics["total_requests"]
            old_avg = self.metrics["average_score"]
            if isinstance(result, EvaluationResult):
                score_value = result.overall_score
                # Track pass/fail
                if result.passed:
                    self.metrics["passed_count"] += 1
            else:  # isinstance(result, ComparisonResult)
                score_value = result.confidence

            self.metrics["average_score"] = (
                old_avg * (total - 1) + float(score_value)
            ) / total

            # Track LLM calls
            final_llm_calls = context.get("llm_calls", 0)
            if isinstance(final_llm_calls, (int, float)) and isinstance(
                initial_llm_calls, (int, float)
            ):
                self.metrics["llm_calls"] += int(final_llm_calls - initial_llm_calls)

            # Track tokens
            self.metrics["tokens_used"] += result.total_tokens

            return result

        except Exception:
            self.metrics["errors"] += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics with calculated averages."""
        metrics = self.metrics.copy()

        # Calculate averages
        if metrics["total_requests"] > 0:
            metrics["avg_time_per_request"] = (
                metrics["total_time"] / metrics["total_requests"]
            )
            metrics["avg_llm_calls_per_request"] = (
                metrics["llm_calls"] / metrics["total_requests"]
            )
            metrics["pass_rate"] = metrics["passed_count"] / metrics["total_requests"]

        return metrics


class CachingMiddleware(Middleware):
    """Caches evaluation results with pluggable backends and TTL support.

    Supports multiple cache backends (memory, Redis) with automatic
    cost savings tracking. Uses SHA256 for deterministic cache keys.

    Cache Key Components:
        - Output text (hashed)
        - Reference text (hashed, if provided)
        - Criteria (if provided)
        - Evaluator names (sorted)
        - Metric names (sorted)
        - Model and temperature configuration

    Example:
        >>> # Memory backend (default)
        >>> cache = CachingMiddleware(max_size=200, ttl=3600)
        >>>
        >>> # Redis backend
        >>> cache = CachingMiddleware(
        ...     backend="redis",
        ...     redis_url="redis://localhost:6379",
        ...     ttl=3600
        ... )
        >>>
        >>> pipeline = MiddlewarePipeline([cache])
        >>> result = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Check stats including cost savings
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Estimated savings: ${stats['estimated_savings_usd']:.4f}")
    """

    def __init__(
        self,
        backend: Literal["memory", "redis"] = "memory",
        max_size: int = 100,
        ttl: Optional[int] = None,
        redis_url: Optional[str] = None,
        redis_key_prefix: str = "arbiter:cache:",
    ) -> None:
        """Initialize caching middleware.

        Args:
            backend: Cache backend type ("memory" or "redis")
            max_size: Maximum entries for memory backend (ignored for Redis)
            ttl: Time-to-live in seconds (None = no expiration for memory,
                 defaults to 3600 for Redis)
            redis_url: Redis connection URL (required if backend="redis",
                       falls back to REDIS_URL env var)
            redis_key_prefix: Key prefix for Redis entries

        Raises:
            ValueError: If backend="redis" and no Redis URL available
        """
        self._backend_type = backend
        self._max_size = max_size
        self._ttl = ttl

        # Initialize backend
        if backend == "memory":
            self._backend: Union[MemoryCacheBackend, RedisCacheBackend] = (
                MemoryCacheBackend(max_size=max_size)
            )
        elif backend == "redis":
            self._backend = RedisCacheBackend(
                redis_url=redis_url,
                key_prefix=redis_key_prefix,
                default_ttl=ttl or 3600,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'memory' or 'redis'")

        # Statistics
        self._hits = 0
        self._misses = 0
        self._estimated_savings_usd = 0.0

    def _generate_cache_key(
        self,
        output: str,
        reference: Optional[str],
        context: MiddlewareContext,
    ) -> str:
        """Generate deterministic SHA256 cache key from inputs.

        Includes all factors that affect evaluation results:
        - Output and reference text
        - Criteria (for custom evaluators)
        - Evaluator and metric configuration
        - Model and temperature settings

        Args:
            output: LLM output text
            reference: Reference text (may be None)
            context: Middleware context with configuration

        Returns:
            SHA256 hex digest as cache key
        """
        # Extract and normalize context values
        evaluators_list = context.get("evaluators", [])
        evaluators = ",".join(sorted(str(e) for e in evaluators_list))

        metrics_list = context.get("metrics", [])
        metrics = ",".join(sorted(str(m) for m in metrics_list))

        # Include criteria for CustomCriteriaEvaluator
        criteria = context.get("criteria", "")

        # Model configuration
        model = context.get("model", "default")
        temperature = context.get("temperature", 0.7)

        # Build canonical key components
        key_parts = [
            f"output:{output}",
            f"reference:{reference or ''}",
            f"criteria:{criteria}",
            f"evaluators:{evaluators}",
            f"metrics:{metrics}",
            f"model:{model}",
            f"temperature:{temperature}",
        ]

        # Create deterministic hash
        canonical_string = "|".join(key_parts)
        return hashlib.sha256(canonical_string.encode("utf-8")).hexdigest()

    def _calculate_result_cost(self, result: MiddlewareResult) -> float:
        """Calculate the cost of a cached result's LLM interactions.

        Args:
            result: Cached evaluation or comparison result

        Returns:
            Estimated cost in USD
        """
        from .cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        total_cost = 0.0

        for interaction in result.interactions:
            if interaction.cost is not None:
                total_cost += interaction.cost
            else:
                total_cost += calc.calculate_cost(
                    model=interaction.model,
                    input_tokens=interaction.input_tokens,
                    output_tokens=interaction.output_tokens,
                    cached_tokens=interaction.cached_tokens,
                )

        return total_cost

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Check cache before processing, cache results after.

        On cache hit:
        - Returns cached result immediately
        - Tracks estimated cost savings

        On cache miss:
        - Calls next handler
        - Caches result with optional TTL
        """
        cache_key = self._generate_cache_key(output, reference, context)

        # Attempt cache retrieval
        cached_json = await self._backend.get(cache_key)

        if cached_json is not None:
            self._hits += 1
            logger.debug(f"Cache hit for key: {cache_key[:16]}...")

            # Deserialize cached result
            cached_data = json.loads(cached_json)
            result_type = cached_data.get("_result_type", "evaluation")

            if result_type == "comparison":
                result: MiddlewareResult = ComparisonResult.model_validate(
                    cached_data["result"]
                )
            else:
                result = EvaluationResult.model_validate(cached_data["result"])

            # Track cost savings
            saved_cost = self._calculate_result_cost(result)
            self._estimated_savings_usd += saved_cost

            return result

        # Cache miss - execute evaluation
        self._misses += 1
        logger.debug(f"Cache miss for key: {cache_key[:16]}...")

        result = await next_handler(output, reference)

        # Serialize and cache result
        result_type_str = (
            "comparison" if isinstance(result, ComparisonResult) else "evaluation"
        )

        # Exclude computed fields to avoid validation errors on deserialize
        result_dict = result.model_dump(
            mode="json", exclude={"interactions": {"__all__": {"total_tokens"}}}
        )

        cache_data = {
            "_result_type": result_type_str,
            "result": result_dict,
        }

        await self._backend.set(
            cache_key,
            json.dumps(cache_data, sort_keys=True),
            ttl=self._ttl,
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with hits, misses, hit_rate, size, max_size,
            and estimated_savings_usd
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "size": self._backend.size(),
            "max_size": self._max_size,
            "estimated_savings_usd": self._estimated_savings_usd,
        }

    async def clear(self) -> None:
        """Clear all cached entries and reset statistics."""
        await self._backend.clear()
        self._hits = 0
        self._misses = 0
        self._estimated_savings_usd = 0.0

    async def close(self) -> None:
        """Close backend connections (for Redis)."""
        if hasattr(self._backend, "close"):
            await self._backend.close()

    async def __aenter__(self) -> "CachingMiddleware":
        """Async context manager entry."""
        if hasattr(self._backend, "__aenter__"):
            await self._backend.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Async context manager exit."""
        await self.close()

    # Backward compatibility properties
    @property
    def cache(self) -> Dict[str, Any]:
        """Legacy access to cache contents (memory backend only).

        Returns dict mapping cache keys to serialized values.
        For full access, use get_stats() instead.
        """
        if isinstance(self._backend, MemoryCacheBackend):
            return {k: v.value for k, v in self._backend._cache.items()}
        return {}

    @property
    def hits(self) -> int:
        """Number of cache hits."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of cache misses."""
        return self._misses

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size


class RateLimitingMiddleware(Middleware):
    """Rate limits evaluation requests.

    Prevents excessive API usage by enforcing a maximum number of
    requests per minute. Useful for staying within API quotas and
    preventing accidental DoS of LLM services.

    Example:
        >>> rate_limiter = RateLimitingMiddleware(max_requests_per_minute=30)
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(rate_limiter)
        >>>
        >>> # Will raise error if rate exceeded
        >>> try:
        ...     for i in range(100):
        ...         await evaluate(output, ref, middleware=pipeline)
        ... except RuntimeError as e:
        ...     print(f"Rate limit hit: {e}")
    """

    def __init__(self, max_requests_per_minute: int = 60):
        """Initialize rate limiting.

        Args:
            max_requests_per_minute: Maximum requests allowed per minute
        """
        self.max_requests = max_requests_per_minute
        self.requests: List[float] = []

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Check rate limit before processing."""
        now = time.time()

        # Remove old requests (older than 60 seconds)
        self.requests = [t for t in self.requests if now - t < 60]

        # Check rate limit
        if len(self.requests) >= self.max_requests:
            wait_time = 60 - (now - self.requests[0])
            raise RuntimeError(
                f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
            )

        # Add current request
        self.requests.append(now)

        result = await next_handler(output, reference)
        return result  # type: ignore[no-any-return]


class MiddlewarePipeline:
    """Manages the middleware pipeline.

    Orchestrates multiple middleware components into a single processing
    chain. Middleware are executed in the order they are added.

    Example:
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(LoggingMiddleware())
        >>> pipeline.add(MetricsMiddleware())
        >>> pipeline.add(CachingMiddleware())
        >>>
        >>> # Use with evaluation
        >>> result = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Get specific middleware
        >>> metrics = pipeline.get_middleware(MetricsMiddleware)
        >>> print(metrics.get_metrics())
    """

    def __init__(self, middleware: Optional[List[Middleware]] = None) -> None:
        """Initialize pipeline with optional middleware list.

        Args:
            middleware: Optional list of middleware to add initially
        """
        self.middleware: List[Middleware] = middleware or []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining
        """
        self.middleware.append(middleware)
        return self

    def get_middleware(self, middleware_type: type) -> Optional[Middleware]:
        """Get middleware instance by type.

        Args:
            middleware_type: Class type of middleware to retrieve

        Returns:
            First middleware instance of the given type, or None
        """
        for mw in self.middleware:
            if isinstance(mw, middleware_type):
                return mw
        return None

    async def execute(
        self,
        output: str,
        reference: Optional[str],
        final_handler: Callable[[str, Optional[str]], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> EvaluationResult:
        """Execute the middleware pipeline.

        Args:
            output: Output text to evaluate
            reference: Reference text (may be None)
            final_handler: The actual evaluation function
            context: Shared context between middleware

        Returns:
            EvaluationResult from the pipeline
        """
        if context is None:
            context = {}

        # Build the chain
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> MiddlewareResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final handler
                result: MiddlewareResult = await final_handler(
                    current_output, current_reference
                )
                return result

            # Call current middleware
            current = self.middleware[index]
            return await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

        result = await chain(0, output, reference)
        # Type narrow to EvaluationResult for this method's return type
        if not isinstance(result, EvaluationResult):
            raise TypeError(
                f"Expected EvaluationResult from pipeline, got {type(result).__name__}"
            )
        return result

    async def execute_comparison(
        self,
        output_a: str,
        output_b: str,
        criteria: Optional[str],
        reference: Optional[str],
        final_handler: Callable[[str, str, Optional[str], Optional[str]], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> ComparisonResult:
        """Execute middleware pipeline for pairwise comparison.

        This method adapts the pairwise comparison signature to work with
        the existing middleware infrastructure. Middleware can check the
        context for `is_pairwise_comparison=True` to detect and handle
        pairwise operations specially if needed.

        The adapter works by:
        1. Packaging both outputs into context for middleware access
        2. Passing a formatted string to middleware for logging/tracking
        3. Calling the final pairwise comparison handler
        4. Returning the ComparisonResult

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            criteria: Optional comparison criteria
            reference: Optional reference context
            final_handler: The actual comparison function
            context: Shared context between middleware

        Returns:
            ComparisonResult from the pipeline

        Example:
            >>> pipeline = MiddlewarePipeline([
            ...     LoggingMiddleware(),
            ...     MetricsMiddleware()
            ... ])
            >>> result = await pipeline.execute_comparison(
            ...     output_a="First output",
            ...     output_b="Second output",
            ...     criteria="accuracy, clarity",
            ...     reference="Reference text",
            ...     final_handler=compare_impl
            ... )
        """
        if context is None:
            context = {}

        # Mark this as a pairwise comparison for middleware
        context["is_pairwise_comparison"] = True
        context["pairwise_data"] = {
            "output_a": output_a,
            "output_b": output_b,
            "criteria": criteria,
        }

        # Create formatted output for middleware logging
        # This allows existing middleware to work without modification
        formatted_output = (
            f"PAIRWISE COMPARISON:\n"
            f"Output A: {output_a[:100]}...\n"
            f"Output B: {output_b[:100]}..."
        )

        # Build the chain - adapter pattern
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> MiddlewareResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final pairwise handler
                # Use original outputs, not formatted version
                result = await final_handler(output_a, output_b, criteria, reference)
                return result  # type: ignore[no-any-return]

            # Call current middleware with formatted output
            current = self.middleware[index]

            # Middleware processes the formatted output but we preserve pairwise data
            result = await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

            return result

        # Execute the chain - the formatted output is for middleware visibility only
        result = await chain(0, formatted_output, reference)

        # Type narrow to ComparisonResult for this method's return type
        if not isinstance(result, ComparisonResult):
            raise TypeError(
                f"Expected ComparisonResult from pipeline, got {type(result).__name__}"
            )
        return result


@asynccontextmanager
async def monitor(
    include_logging: bool = True, include_metrics: bool = True, log_level: str = "INFO"
) -> AsyncIterator[Dict[str, Any]]:
    """Context manager for monitoring evaluations.

    Convenient helper for creating a pipeline with logging and metrics
    middleware. Automatically logs final metrics when done.

    Args:
        include_logging: Whether to include logging middleware
        include_metrics: Whether to include metrics middleware
        log_level: Logging level (if logging enabled)

    Yields:
        Dictionary with 'pipeline' and 'metrics' keys

    Example:
        >>> async with monitor() as ctx:
        ...     pipeline = ctx['pipeline']
        ...     for output, ref in test_cases:
        ...         await evaluate(output, ref, middleware=pipeline)
        ...
        ...     # Metrics automatically logged at end
        ...     metrics = ctx['metrics']
        ...     if metrics:
        ...         print(metrics.get_metrics())
    """
    pipeline = MiddlewarePipeline()
    metrics_middleware = None

    if include_logging:
        pipeline.add(LoggingMiddleware(log_level))

    if include_metrics:
        metrics_middleware = MetricsMiddleware()
        pipeline.add(metrics_middleware)

    data = {"pipeline": pipeline, "metrics": metrics_middleware}

    yield data

    # After completion, log final metrics
    if metrics_middleware:
        final_metrics = metrics_middleware.get_metrics()
        logger.info(f"Session metrics: {final_metrics}")
