"""Performance monitoring and observability for Arbiter evaluation operations.

This module provides comprehensive monitoring capabilities for tracking
the performance and behavior of evaluation operations. It collects
detailed metrics about LLM calls, evaluator execution, and overall
processing time.

## Key Features:

- **Detailed Metrics**: Track timing, token usage, and call counts
- **Performance Analysis**: Identify bottlenecks and optimization opportunities
- **Error Tracking**: Capture and categorize failures for debugging
- **Integration**: Optional Logfire integration for production monitoring
- **Context Managers**: Easy instrumentation of operations

## Usage:

    >>> # Basic monitoring
    >>> monitor = PerformanceMonitor()
    >>> async with monitor.track_operation("evaluate") as metrics:
    ...     result = await evaluate(output, reference)
    >>> print(metrics.to_dict())

    >>> # Global monitoring
    >>> from arbiter import get_global_monitor
    >>> monitor = get_global_monitor()
    >>>
    >>> # Context manager for specific operations
    >>> async with monitor_context("batch_evaluation"):
    ...     await process_batch(outputs, references)

## Metrics Collected:

- **Timing**: Total duration, LLM time, evaluator time
- **LLM Usage**: Call count, tokens used, tokens per second
- **Evaluators**: Which evaluators ran, how long they took
- **Scores**: Score distributions and averages
- **Errors**: Detailed error information for debugging

## Integration with Logfire:

Set the LOGFIRE_TOKEN environment variable to enable production monitoring:

    export LOGFIRE_TOKEN=your_token_here

This will send detailed traces and metrics to Logfire for analysis.
"""

import logging
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import logfire

logger = logging.getLogger("arbiter.monitoring")

# Configure logfire if token is available
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

__all__ = [
    "PerformanceMetrics",
    "PerformanceMonitor",
    "get_global_monitor",
    "monitor",
    "QueryMetrics",
    "QueryMonitor",
    "get_query_monitor",
    "track_query",
]


@dataclass
class PerformanceMetrics:
    """Comprehensive metrics for a single evaluation operation.

    This dataclass collects all performance-related data during the
    execution of an evaluate() call. It tracks timing, resource usage,
    and quality metrics to provide full observability.

    Example:
        >>> metrics = PerformanceMetrics(start_time=time.time())
        >>> # ... perform operations ...
        >>> metrics.llm_calls += 1
        >>> metrics.tokens_used += 150
        >>> metrics.end_time = time.time()
        >>> metrics.finalize()
        >>> print(f"Total time: {metrics.total_duration:.2f}s")

    Attributes:
        start_time: Unix timestamp when operation began
        end_time: Unix timestamp when operation completed
        total_duration: Total seconds elapsed
        llm_calls: Number of LLM API calls made
        llm_time: Total seconds spent in LLM calls
        tokens_used: Total tokens consumed
        tokens_per_second: Token generation rate
        evaluator_calls: Number of evaluator executions
        evaluator_time: Total seconds in evaluator execution
        evaluators_used: List of evaluator names called
        scores_computed: Number of scores calculated
        average_score: Average score across metrics
        errors: List of error details
    """

    # Timing metrics
    start_time: float
    end_time: float = 0.0
    total_duration: float = 0.0

    # LLM metrics
    llm_calls: int = 0
    llm_time: float = 0.0
    tokens_used: int = 0
    tokens_per_second: float = 0.0

    # Evaluator metrics
    evaluator_calls: int = 0
    evaluator_time: float = 0.0
    evaluators_used: List[str] = field(default_factory=list)

    # Score metrics
    scores_computed: int = 0
    average_score: float = 0.0
    score_distribution: Dict[str, float] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self) -> None:
        """Calculate derived metrics after operation completes."""
        self.total_duration = self.end_time - self.start_time

        if self.tokens_used > 0 and self.llm_time > 0:
            self.tokens_per_second = self.tokens_used / self.llm_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting."""
        return {
            "timing": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration": round(self.total_duration, 3),
                "llm_time": round(self.llm_time, 3),
                "evaluator_time": round(self.evaluator_time, 3),
            },
            "llm": {
                "calls": self.llm_calls,
                "tokens_used": self.tokens_used,
                "tokens_per_second": round(self.tokens_per_second, 1),
            },
            "evaluators": {
                "calls": self.evaluator_calls,
                "evaluators_used": list(set(self.evaluators_used)),
                "avg_time_per_call": (
                    round(self.evaluator_time / self.evaluator_calls, 3)
                    if self.evaluator_calls > 0
                    else 0
                ),
            },
            "scores": {
                "count": self.scores_computed,
                "average": round(self.average_score, 3),
                "distribution": self.score_distribution,
            },
            "errors": {"count": len(self.errors), "details": self.errors},
        }


class PerformanceMonitor:
    """Monitor for tracking evaluation performance.

    Provides context managers and tracking methods for monitoring
    evaluation operations.

    Example:
        >>> monitor = PerformanceMonitor()
        >>>
        >>> async with monitor.track_operation("evaluate") as metrics:
        ...     result = await evaluate(output, reference)
        ...     metrics.scores_computed = len(result.scores)
        >>>
        >>> print(monitor.get_summary())
    """

    def __init__(self) -> None:
        """Initialize monitor with empty metrics."""
        self.operations: List[PerformanceMetrics] = []
        self._current_metrics: Optional[PerformanceMetrics] = None

    @asynccontextmanager
    async def track_operation(
        self, operation_name: str
    ) -> AsyncIterator[PerformanceMetrics]:
        """Track a single operation.

        Args:
            operation_name: Name of the operation being tracked

        Yields:
            PerformanceMetrics object to populate during operation

        Example:
            >>> async with monitor.track_operation("evaluate") as metrics:
            ...     result = await evaluate(output, reference)
            ...     metrics.llm_calls = 1
            ...     metrics.tokens_used = result.total_tokens
        """
        metrics = PerformanceMetrics(start_time=time.time())
        self._current_metrics = metrics

        try:
            yield metrics
        finally:
            metrics.end_time = time.time()
            metrics.finalize()
            self.operations.append(metrics)
            self._current_metrics = None

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the currently active metrics object."""
        return self._current_metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all operations.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.operations:
            return {"total_operations": 0}

        total_time = sum(m.total_duration for m in self.operations)
        total_llm_calls = sum(m.llm_calls for m in self.operations)
        total_tokens = sum(m.tokens_used for m in self.operations)
        total_errors = sum(len(m.errors) for m in self.operations)

        return {
            "total_operations": len(self.operations),
            "total_time": round(total_time, 3),
            "total_llm_calls": total_llm_calls,
            "total_tokens": total_tokens,
            "total_errors": total_errors,
            "avg_time_per_operation": round(total_time / len(self.operations), 3),
            "avg_tokens_per_operation": (
                total_tokens // len(self.operations) if self.operations else 0
            ),
        }

    def reset(self) -> None:
        """Reset all tracked operations."""
        self.operations.clear()
        self._current_metrics = None


# Global monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


@asynccontextmanager
async def monitor(
    operation_name: str = "evaluation",
) -> AsyncIterator[PerformanceMetrics]:
    """Convenient context manager for monitoring operations.

    Args:
        operation_name: Name of the operation to track

    Yields:
        PerformanceMetrics object

    Example:
        >>> async with monitor("batch_evaluate") as metrics:
        ...     for output, ref in pairs:
        ...         result = await evaluate(output, ref)
        ...         metrics.scores_computed += len(result.scores)
    """
    global_monitor = get_global_monitor()
    async with global_monitor.track_operation(operation_name) as metrics:
        yield metrics


# Query Performance Monitoring


@dataclass
class QueryMetrics:
    """Metrics for a single database query execution.

    Tracks timing and metadata for database operations to identify
    slow queries and performance bottlenecks.

    Example:
        >>> metrics = QueryMetrics(
        ...     operation="save_result",
        ...     duration=0.045,
        ...     timestamp=time.time()
        ... )
        >>> print(f"Query took {metrics.duration:.3f}s")

    Attributes:
        operation: Name of the database operation (e.g., "save_result", "get_result")
        duration: Query execution time in seconds
        timestamp: Unix timestamp when query was executed
        success: Whether the query completed successfully
        error: Error message if query failed
    """

    operation: str
    duration: float
    timestamp: float
    success: bool = True
    error: Optional[str] = None


SlowQueryCallback = Callable[[QueryMetrics], None]


class QueryMonitor:
    """Monitor for tracking database query performance.

    Collects query execution metrics and provides percentile-based
    performance analysis. Supports optional slow query alerting.

    Example:
        >>> monitor = QueryMonitor(slow_query_threshold=0.1)
        >>> monitor.on_slow_query = lambda m: print(f"Slow: {m.operation}")
        >>>
        >>> with monitor.track("save_result"):
        ...     await storage.save_result(result)
        >>>
        >>> stats = monitor.get_statistics()
        >>> print(f"p95: {stats['percentiles']['p95']:.3f}s")

    Attributes:
        slow_query_threshold: Duration in seconds above which queries are
            considered slow and trigger the callback.
        on_slow_query: Optional callback invoked for slow queries.
    """

    def __init__(
        self,
        slow_query_threshold: float = 1.0,
        max_history: int = 10000,
    ) -> None:
        """Initialize query monitor.

        Args:
            slow_query_threshold: Threshold in seconds for slow query alerts
            max_history: Maximum number of query metrics to retain
        """
        self.slow_query_threshold = slow_query_threshold
        self.max_history = max_history
        self._queries: List[QueryMetrics] = []
        self._on_slow_query: Optional[SlowQueryCallback] = None

    @property
    def on_slow_query(self) -> Optional[SlowQueryCallback]:
        """Get the slow query callback."""
        return self._on_slow_query

    @on_slow_query.setter
    def on_slow_query(self, callback: Optional[SlowQueryCallback]) -> None:
        """Set the slow query callback."""
        self._on_slow_query = callback

    def record(self, metrics: QueryMetrics) -> None:
        """Record a query execution.

        Args:
            metrics: Query metrics to record
        """
        self._queries.append(metrics)

        # Trim history if needed
        if len(self._queries) > self.max_history:
            self._queries = self._queries[-self.max_history :]

        # Check for slow query
        if metrics.duration >= self.slow_query_threshold:
            logger.warning(
                f"Slow query detected: {metrics.operation} "
                f"took {metrics.duration:.3f}s (threshold: {self.slow_query_threshold}s)"
            )
            if self._on_slow_query:
                self._on_slow_query(metrics)

    @contextmanager
    def track(self, operation: str) -> Iterator[None]:
        """Track a database query execution.

        Args:
            operation: Name of the operation being tracked

        Example:
            >>> with monitor.track("get_result"):
            ...     result = await conn.fetchrow(query)
        """
        start = time.perf_counter()
        error: Optional[str] = None
        success = True

        try:
            yield
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = time.perf_counter() - start
            metrics = QueryMetrics(
                operation=operation,
                duration=duration,
                timestamp=time.time(),
                success=success,
                error=error,
            )
            self.record(metrics)

    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get query performance statistics.

        Args:
            operation: Optional filter for specific operation type

        Returns:
            Dictionary with count, timing stats, and percentiles

        Example:
            >>> stats = monitor.get_statistics("save_result")
            >>> print(f"Count: {stats['count']}, p95: {stats['percentiles']['p95']:.3f}s")
        """
        queries = self._queries
        if operation:
            queries = [q for q in queries if q.operation == operation]

        if not queries:
            return {
                "count": 0,
                "success_rate": 0.0,
                "timing": {"min": 0.0, "max": 0.0, "avg": 0.0},
                "percentiles": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            }

        durations = sorted([q.duration for q in queries])
        successful = sum(1 for q in queries if q.success)

        return {
            "count": len(queries),
            "success_rate": successful / len(queries),
            "timing": {
                "min": round(durations[0], 6),
                "max": round(durations[-1], 6),
                "avg": round(sum(durations) / len(durations), 6),
            },
            "percentiles": {
                "p50": round(self._percentile(durations, 50), 6),
                "p95": round(self._percentile(durations, 95), 6),
                "p99": round(self._percentile(durations, 99), 6),
            },
        }

    def get_operations(self) -> List[str]:
        """Get list of unique operation names tracked.

        Returns:
            List of operation names
        """
        return list(set(q.operation for q in self._queries))

    def get_slow_queries(self, threshold: Optional[float] = None) -> List[QueryMetrics]:
        """Get queries exceeding the threshold.

        Args:
            threshold: Duration threshold in seconds (uses instance threshold if None)

        Returns:
            List of slow query metrics
        """
        threshold = threshold or self.slow_query_threshold
        return [q for q in self._queries if q.duration >= threshold]

    def reset(self) -> None:
        """Clear all recorded queries."""
        self._queries.clear()

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Calculate percentile from sorted data.

        Args:
            sorted_data: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Value at the given percentile
        """
        if not sorted_data:
            return 0.0

        k = (len(sorted_data) - 1) * (percentile / 100.0)
        f = int(k)
        c = f + 1

        if c >= len(sorted_data):
            return sorted_data[-1]

        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


# Global query monitor instance
_query_monitor: Optional[QueryMonitor] = None


def get_query_monitor() -> QueryMonitor:
    """Get the global query monitor instance.

    Returns:
        Global QueryMonitor instance

    Example:
        >>> monitor = get_query_monitor()
        >>> stats = monitor.get_statistics()
    """
    global _query_monitor
    if _query_monitor is None:
        _query_monitor = QueryMonitor()
    return _query_monitor


@contextmanager
def track_query(operation: str) -> Iterator[None]:
    """Track a database query using the global monitor.

    This is a convenience function that uses the global query monitor.

    Args:
        operation: Name of the database operation

    Example:
        >>> with track_query("save_result"):
        ...     await conn.execute(insert_query)
    """
    monitor = get_query_monitor()
    with monitor.track(operation):
        yield
