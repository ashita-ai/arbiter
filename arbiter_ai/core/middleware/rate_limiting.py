"""Rate limiting middleware for Arbiter evaluations.

This module provides RateLimitingMiddleware for enforcing request
rate limits to prevent API quota exhaustion.
"""

import time
from typing import Any, Callable, List, Optional

from ..type_defs import MiddlewareContext
from .base import Middleware, MiddlewareResult

__all__ = ["RateLimitingMiddleware"]


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
