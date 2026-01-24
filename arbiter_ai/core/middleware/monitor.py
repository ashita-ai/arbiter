"""Monitor context manager for evaluation sessions.

This module provides a convenient context manager for creating
middleware pipelines with logging and metrics.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict

from ..logging import get_logger
from .base import MiddlewarePipeline
from .logging_mw import LoggingMiddleware
from .metrics import MetricsMiddleware

logger = get_logger("middleware")

__all__ = ["monitor"]


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

    data: Dict[str, Any] = {"pipeline": pipeline, "metrics": metrics_middleware}

    yield data

    # After completion, log final metrics
    if metrics_middleware:
        final_metrics = metrics_middleware.get_metrics()
        logger.info(f"Session metrics: {final_metrics}")
