"""Logging middleware for Arbiter evaluations.

This module provides LoggingMiddleware for detailed logging of the
evaluation process including inputs, outputs, timing, and errors.
"""

import logging
import time
from typing import Any, Callable, Optional

from ..logging import get_logger
from ..models import ComparisonResult, EvaluationResult
from ..type_defs import MiddlewareContext
from .base import Middleware, MiddlewareResult

logger = get_logger("middleware")

__all__ = ["LoggingMiddleware"]


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
