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

from ..logging import get_logger
from .base import Middleware, MiddlewarePipeline, MiddlewareResult
from .cache_backends import (
    CacheBackend,
    CacheEntry,
    MemoryCacheBackend,
    RedisCacheBackend,
)
from .caching import CachingMiddleware
from .logging_mw import LoggingMiddleware
from .metrics import MetricsMiddleware
from .monitor import monitor
from .rate_limiting import RateLimitingMiddleware

# Module-level logger for backward compatibility
logger = get_logger("middleware")

__all__ = [
    # Base classes
    "Middleware",
    "MiddlewarePipeline",
    "MiddlewareResult",
    # Cache backends
    "CacheBackend",
    "CacheEntry",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    # Middleware implementations
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    # Utilities
    "monitor",
]
