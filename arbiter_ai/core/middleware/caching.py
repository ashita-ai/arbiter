"""Caching middleware for evaluation results.

This module provides CachingMiddleware for caching evaluation results
with pluggable backends (memory, Redis) and TTL support.
"""

import hashlib
import json
from typing import Any, Callable, Dict, Literal, Optional, Union

from ..logging import get_logger
from ..models import ComparisonResult, EvaluationResult
from ..type_defs import MiddlewareContext
from .base import Middleware, MiddlewareResult
from .cache_backends import MemoryCacheBackend, RedisCacheBackend

logger = get_logger("middleware")

__all__ = ["CachingMiddleware"]


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
        from ..cost_calculator import get_cost_calculator

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
