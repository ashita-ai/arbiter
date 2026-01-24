"""Metrics middleware for collecting evaluation statistics.

This module provides MetricsMiddleware for tracking comprehensive
metrics about evaluation performance, scores, and resource usage.
"""

import time
from typing import Any, Callable, Dict, Optional

from ..models import EvaluationResult
from ..type_defs import MiddlewareContext
from .base import Middleware, MiddlewareResult

__all__ = ["MetricsMiddleware"]


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
        self.metrics: Dict[str, Any] = {
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
