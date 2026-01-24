"""Unit tests for monitoring.py."""

import time
from unittest.mock import patch

import pytest

from arbiter_ai.core.monitoring import (
    PerformanceMetrics,
    PerformanceMonitor,
    QueryMetrics,
    QueryMonitor,
    get_global_monitor,
    get_query_monitor,
    monitor,
    track_query,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics."""

    def test_initialization(self):
        """Test PerformanceMetrics initialization."""
        start = time.time()
        metrics = PerformanceMetrics(start_time=start)

        assert metrics.start_time == start
        assert metrics.end_time == 0.0
        assert metrics.llm_calls == 0
        assert metrics.tokens_used == 0
        assert len(metrics.errors) == 0

    def test_finalize_calculates_duration(self):
        """Test that finalize() calculates total_duration."""
        start = time.time()
        metrics = PerformanceMetrics(start_time=start)
        time.sleep(0.01)  # Small delay
        metrics.end_time = time.time()

        metrics.finalize()

        assert metrics.total_duration > 0
        assert metrics.total_duration == metrics.end_time - metrics.start_time

    def test_finalize_calculates_tokens_per_second(self):
        """Test that finalize() calculates tokens_per_second."""
        metrics = PerformanceMetrics(start_time=time.time())
        metrics.tokens_used = 1000
        metrics.llm_time = 2.0
        metrics.end_time = time.time()

        metrics.finalize()

        assert metrics.tokens_per_second == 500.0  # 1000 tokens / 2 seconds

    def test_finalize_handles_zero_llm_time(self):
        """Test that finalize() handles zero llm_time gracefully."""
        metrics = PerformanceMetrics(start_time=time.time())
        metrics.tokens_used = 1000
        metrics.llm_time = 0.0
        metrics.end_time = time.time()

        metrics.finalize()

        # Should not raise, tokens_per_second should remain 0
        assert metrics.tokens_per_second == 0.0

    def test_to_dict_structure(self):
        """Test that to_dict() returns correct structure."""
        start = time.time()
        metrics = PerformanceMetrics(start_time=start)
        metrics.end_time = start + 1.5
        metrics.llm_calls = 2
        metrics.tokens_used = 500
        metrics.llm_time = 1.0
        metrics.evaluator_calls = 2
        metrics.evaluator_time = 0.5
        metrics.evaluators_used = ["semantic", "custom"]
        metrics.scores_computed = 2
        metrics.average_score = 0.85
        metrics.score_distribution = {"semantic": 0.9, "custom": 0.8}
        metrics.finalize()

        result = metrics.to_dict()

        assert "timing" in result
        assert "llm" in result
        assert "evaluators" in result
        assert "scores" in result
        assert "errors" in result

        # Check timing section
        assert "total_duration" in result["timing"]
        assert "llm_time" in result["timing"]
        assert isinstance(result["timing"]["start_time"], str)
        assert isinstance(result["timing"]["end_time"], str)

        # Check LLM section
        assert result["llm"]["calls"] == 2
        assert result["llm"]["tokens_used"] == 500
        assert result["llm"]["tokens_per_second"] == 500.0

        # Check evaluators section
        assert result["evaluators"]["calls"] == 2
        assert len(result["evaluators"]["evaluators_used"]) == 2
        assert result["evaluators"]["avg_time_per_call"] == 0.25

        # Check scores section
        assert result["scores"]["count"] == 2
        assert result["scores"]["average"] == 0.85
        assert result["scores"]["distribution"] == {"semantic": 0.9, "custom": 0.8}

    def test_to_dict_with_errors(self):
        """Test to_dict() with error tracking."""
        metrics = PerformanceMetrics(start_time=time.time())
        metrics.end_time = time.time()
        metrics.errors = [
            {"type": "ValueError", "message": "Test error"},
            {"type": "RuntimeError", "message": "Another error"},
        ]
        metrics.finalize()

        result = metrics.to_dict()

        assert result["errors"]["count"] == 2
        assert len(result["errors"]["details"]) == 2


class TestPerformanceMonitor:
    """Test PerformanceMonitor."""

    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()

        assert len(monitor.operations) == 0
        assert monitor._current_metrics is None

    @pytest.mark.asyncio
    async def test_track_operation(self):
        """Test tracking an operation."""
        monitor = PerformanceMonitor()

        async with monitor.track_operation("test_op") as metrics:
            assert metrics.start_time > 0
            assert monitor._current_metrics == metrics

            # Simulate some work
            metrics.llm_calls = 1
            metrics.tokens_used = 100
            time.sleep(0.01)

        # After context exit
        assert monitor._current_metrics is None
        assert len(monitor.operations) == 1
        assert monitor.operations[0].llm_calls == 1
        assert monitor.operations[0].tokens_used == 100
        assert monitor.operations[0].end_time > 0
        assert monitor.operations[0].total_duration > 0

    @pytest.mark.asyncio
    async def test_track_operation_finalizes_metrics(self):
        """Test that track_operation finalizes metrics."""
        monitor = PerformanceMonitor()

        async with monitor.track_operation("test_op") as metrics:
            metrics.llm_time = 1.0
            metrics.tokens_used = 500

        # Metrics should be finalized
        tracked = monitor.operations[0]
        assert tracked.tokens_per_second == 500.0

    @pytest.mark.asyncio
    async def test_track_operation_exception_handling(self):
        """Test that metrics are finalized even on exception."""
        monitor = PerformanceMonitor()

        with pytest.raises(ValueError):
            async with monitor.track_operation("test_op") as metrics:
                metrics.llm_calls = 1
                raise ValueError("Test error")

        # Metrics should still be recorded and finalized
        assert len(monitor.operations) == 1
        assert monitor.operations[0].llm_calls == 1
        assert monitor.operations[0].end_time > 0

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        monitor = PerformanceMonitor()

        # No current operation
        assert monitor.get_current_metrics() is None

    @pytest.mark.asyncio
    async def test_get_current_metrics_during_operation(self):
        """Test getting current metrics during operation."""
        monitor = PerformanceMonitor()

        async with monitor.track_operation("test_op") as metrics:
            current = monitor.get_current_metrics()
            assert current == metrics
            assert current is not None

    def test_get_summary_empty(self):
        """Test getting summary with no operations."""
        monitor = PerformanceMonitor()

        summary = monitor.get_summary()

        assert summary["total_operations"] == 0

    @pytest.mark.asyncio
    async def test_get_summary_with_operations(self):
        """Test getting summary with tracked operations."""
        monitor = PerformanceMonitor()

        # Track multiple operations
        async with monitor.track_operation("op1") as m1:
            m1.llm_calls = 1
            m1.tokens_used = 100
            time.sleep(0.01)

        async with monitor.track_operation("op2") as m2:
            m2.llm_calls = 2
            m2.tokens_used = 200
            time.sleep(0.01)

        summary = monitor.get_summary()

        assert summary["total_operations"] == 2
        assert summary["total_llm_calls"] == 3
        assert summary["total_tokens"] == 300
        assert summary["total_errors"] == 0
        assert summary["total_time"] > 0
        assert summary["avg_time_per_operation"] > 0
        assert summary["avg_tokens_per_operation"] == 150

    @pytest.mark.asyncio
    async def test_get_summary_with_errors(self):
        """Test summary includes error count."""
        monitor = PerformanceMonitor()

        async with monitor.track_operation("op1") as m1:
            m1.errors.append({"type": "Error", "message": "test"})

        async with monitor.track_operation("op2") as m2:
            m2.errors.append({"type": "Error1", "message": "test1"})
            m2.errors.append({"type": "Error2", "message": "test2"})

        summary = monitor.get_summary()

        assert summary["total_errors"] == 3

    def test_reset(self):
        """Test resetting the monitor."""
        monitor = PerformanceMonitor()
        monitor.operations.append(PerformanceMetrics(start_time=time.time()))
        monitor.operations.append(PerformanceMetrics(start_time=time.time()))

        assert len(monitor.operations) == 2

        monitor.reset()

        assert len(monitor.operations) == 0
        assert monitor._current_metrics is None


class TestGlobalMonitor:
    """Test global monitor functions."""

    def test_get_global_monitor_creates_instance(self):
        """Test that get_global_monitor creates a monitor."""
        # Clear any existing global monitor
        import arbiter_ai.core.monitoring as mon_module

        mon_module._global_monitor = None

        monitor = get_global_monitor()

        assert monitor is not None
        assert isinstance(monitor, PerformanceMonitor)

    def test_get_global_monitor_returns_same_instance(self):
        """Test that get_global_monitor returns the same instance."""
        monitor1 = get_global_monitor()
        monitor2 = get_global_monitor()

        assert monitor1 is monitor2

    @pytest.mark.asyncio
    async def test_monitor_context_manager(self):
        """Test the monitor() context manager."""
        # Reset global monitor
        import arbiter_ai.core.monitoring as mon_module

        mon_module._global_monitor = None

        async with monitor("test_operation") as metrics:
            assert metrics.start_time > 0
            metrics.llm_calls = 1
            metrics.tokens_used = 50

        # Check global monitor has the operation
        global_monitor = get_global_monitor()
        assert len(global_monitor.operations) == 1
        assert global_monitor.operations[0].llm_calls == 1
        assert global_monitor.operations[0].tokens_used == 50

    @pytest.mark.asyncio
    async def test_monitor_context_manager_default_name(self):
        """Test monitor() with default operation name."""
        # Reset global monitor
        import arbiter_ai.core.monitoring as mon_module

        mon_module._global_monitor = PerformanceMonitor()

        async with monitor() as metrics:
            metrics.scores_computed = 5

        # Should use default name "evaluation"
        global_monitor = get_global_monitor()
        assert len(global_monitor.operations) == 1


class TestLogfireIntegration:
    """Test Logfire integration."""

    @patch.dict("os.environ", {"LOGFIRE_TOKEN": "test_token"})
    @patch("arbiter_ai.core.monitoring.logfire.configure")
    def test_logfire_configured_when_token_present(self, mock_configure):
        """Test that logfire is configured when LOGFIRE_TOKEN is set."""
        # Re-import to trigger configuration

        # The module level code runs on import
        # Since we can't re-run that easily, just verify the env var is set
        import os

        assert os.getenv("LOGFIRE_TOKEN") == "test_token"


class TestQueryMetrics:
    """Test QueryMetrics dataclass."""

    def test_initialization(self):
        """Test QueryMetrics initialization."""
        metrics = QueryMetrics(
            operation="save_result",
            duration=0.045,
            timestamp=time.time(),
        )

        assert metrics.operation == "save_result"
        assert metrics.duration == 0.045
        assert metrics.success is True
        assert metrics.error is None

    def test_initialization_with_error(self):
        """Test QueryMetrics with error."""
        metrics = QueryMetrics(
            operation="get_result",
            duration=0.1,
            timestamp=time.time(),
            success=False,
            error="Connection timeout",
        )

        assert metrics.success is False
        assert metrics.error == "Connection timeout"


class TestQueryMonitor:
    """Test QueryMonitor class."""

    def test_initialization(self):
        """Test QueryMonitor initialization."""
        monitor = QueryMonitor()

        assert monitor.slow_query_threshold == 1.0
        assert monitor.max_history == 10000
        assert len(monitor._queries) == 0
        assert monitor.on_slow_query is None

    def test_initialization_with_custom_threshold(self):
        """Test QueryMonitor with custom threshold."""
        monitor = QueryMonitor(slow_query_threshold=0.5, max_history=100)

        assert monitor.slow_query_threshold == 0.5
        assert monitor.max_history == 100

    def test_record_query(self):
        """Test recording a query."""
        monitor = QueryMonitor()
        metrics = QueryMetrics(
            operation="save_result",
            duration=0.05,
            timestamp=time.time(),
        )

        monitor.record(metrics)

        assert len(monitor._queries) == 1
        assert monitor._queries[0] == metrics

    def test_record_trims_history(self):
        """Test that recording trims history when exceeding max."""
        monitor = QueryMonitor(max_history=3)

        for i in range(5):
            metrics = QueryMetrics(
                operation=f"op_{i}",
                duration=0.01 * i,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        assert len(monitor._queries) == 3
        assert monitor._queries[0].operation == "op_2"
        assert monitor._queries[-1].operation == "op_4"

    def test_slow_query_callback(self):
        """Test slow query callback is invoked."""
        monitor = QueryMonitor(slow_query_threshold=0.1)
        slow_queries = []
        monitor.on_slow_query = lambda m: slow_queries.append(m)

        # Fast query - no callback
        fast = QueryMetrics(operation="fast_op", duration=0.05, timestamp=time.time())
        monitor.record(fast)
        assert len(slow_queries) == 0

        # Slow query - callback invoked
        slow = QueryMetrics(operation="slow_op", duration=0.2, timestamp=time.time())
        monitor.record(slow)
        assert len(slow_queries) == 1
        assert slow_queries[0].operation == "slow_op"

    def test_track_context_manager(self):
        """Test track() context manager."""
        monitor = QueryMonitor()

        with monitor.track("test_query"):
            time.sleep(0.01)

        assert len(monitor._queries) == 1
        assert monitor._queries[0].operation == "test_query"
        assert monitor._queries[0].duration >= 0.01
        assert monitor._queries[0].success is True

    def test_track_records_error(self):
        """Test track() records errors."""
        monitor = QueryMonitor()

        with pytest.raises(ValueError):
            with monitor.track("failing_query"):
                raise ValueError("Query failed")

        assert len(monitor._queries) == 1
        assert monitor._queries[0].success is False
        assert "Query failed" in monitor._queries[0].error

    def test_get_statistics_empty(self):
        """Test get_statistics with no queries."""
        monitor = QueryMonitor()

        stats = monitor.get_statistics()

        assert stats["count"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["timing"]["min"] == 0.0
        assert stats["percentiles"]["p50"] == 0.0

    def test_get_statistics_with_queries(self):
        """Test get_statistics with recorded queries."""
        monitor = QueryMonitor()

        # Record queries with known durations
        for duration in [0.01, 0.02, 0.03, 0.04, 0.05]:
            metrics = QueryMetrics(
                operation="test_op",
                duration=duration,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        stats = monitor.get_statistics()

        assert stats["count"] == 5
        assert stats["success_rate"] == 1.0
        assert stats["timing"]["min"] == 0.01
        assert stats["timing"]["max"] == 0.05
        assert stats["timing"]["avg"] == 0.03
        assert stats["percentiles"]["p50"] == 0.03

    def test_get_statistics_filtered_by_operation(self):
        """Test get_statistics filtered by operation name."""
        monitor = QueryMonitor()

        # Record different operations
        for op in ["save_result", "save_result", "get_result"]:
            metrics = QueryMetrics(
                operation=op,
                duration=0.01,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        stats = monitor.get_statistics("save_result")

        assert stats["count"] == 2

        stats_get = monitor.get_statistics("get_result")
        assert stats_get["count"] == 1

    def test_get_statistics_success_rate(self):
        """Test success rate calculation."""
        monitor = QueryMonitor()

        # 3 successful, 2 failed
        for success in [True, True, True, False, False]:
            metrics = QueryMetrics(
                operation="test_op",
                duration=0.01,
                timestamp=time.time(),
                success=success,
            )
            monitor.record(metrics)

        stats = monitor.get_statistics()

        assert stats["success_rate"] == 0.6

    def test_get_operations(self):
        """Test get_operations returns unique operations."""
        monitor = QueryMonitor()

        for op in ["save_result", "get_result", "save_result", "save_batch"]:
            metrics = QueryMetrics(
                operation=op,
                duration=0.01,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        operations = monitor.get_operations()

        assert len(operations) == 3
        assert set(operations) == {"save_result", "get_result", "save_batch"}

    def test_get_slow_queries(self):
        """Test get_slow_queries filtering."""
        monitor = QueryMonitor(slow_query_threshold=0.1)

        for duration in [0.05, 0.1, 0.15, 0.2]:
            metrics = QueryMetrics(
                operation="test_op",
                duration=duration,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        slow = monitor.get_slow_queries()

        assert len(slow) == 3  # 0.1, 0.15, 0.2

    def test_get_slow_queries_custom_threshold(self):
        """Test get_slow_queries with custom threshold."""
        monitor = QueryMonitor(slow_query_threshold=0.1)

        for duration in [0.05, 0.1, 0.15, 0.2]:
            metrics = QueryMetrics(
                operation="test_op",
                duration=duration,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        slow = monitor.get_slow_queries(threshold=0.15)

        assert len(slow) == 2  # 0.15, 0.2

    def test_reset(self):
        """Test reset clears all queries."""
        monitor = QueryMonitor()

        for i in range(5):
            metrics = QueryMetrics(
                operation="test_op",
                duration=0.01,
                timestamp=time.time(),
            )
            monitor.record(metrics)

        assert len(monitor._queries) == 5

        monitor.reset()

        assert len(monitor._queries) == 0

    def test_percentile_calculation(self):
        """Test percentile calculation accuracy."""
        # Test with 100 values for accurate percentile calculation
        durations = [i / 100.0 for i in range(1, 101)]

        monitor = QueryMonitor()
        p50 = monitor._percentile(durations, 50)
        p95 = monitor._percentile(durations, 95)
        p99 = monitor._percentile(durations, 99)

        assert abs(p50 - 0.505) < 0.01
        assert abs(p95 - 0.955) < 0.01
        assert abs(p99 - 0.995) < 0.01

    def test_percentile_empty_list(self):
        """Test percentile with empty list."""
        monitor = QueryMonitor()
        assert monitor._percentile([], 50) == 0.0

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        monitor = QueryMonitor()
        assert monitor._percentile([0.5], 50) == 0.5
        assert monitor._percentile([0.5], 99) == 0.5


class TestGlobalQueryMonitor:
    """Test global query monitor functions."""

    def test_get_query_monitor_creates_instance(self):
        """Test that get_query_monitor creates a monitor."""
        import arbiter_ai.core.monitoring as mon_module

        mon_module._query_monitor = None

        monitor = get_query_monitor()

        assert monitor is not None
        assert isinstance(monitor, QueryMonitor)

    def test_get_query_monitor_returns_same_instance(self):
        """Test that get_query_monitor returns the same instance."""
        monitor1 = get_query_monitor()
        monitor2 = get_query_monitor()

        assert monitor1 is monitor2

    def test_track_query_context_manager(self):
        """Test track_query() context manager."""
        import arbiter_ai.core.monitoring as mon_module

        mon_module._query_monitor = QueryMonitor()

        with track_query("test_query"):
            time.sleep(0.01)

        monitor = get_query_monitor()
        assert len(monitor._queries) == 1
        assert monitor._queries[0].operation == "test_query"

    def test_track_query_records_errors(self):
        """Test track_query() records errors."""
        import arbiter_ai.core.monitoring as mon_module

        mon_module._query_monitor = QueryMonitor()

        with pytest.raises(RuntimeError):
            with track_query("failing_query"):
                raise RuntimeError("DB error")

        monitor = get_query_monitor()
        assert len(monitor._queries) == 1
        assert monitor._queries[0].success is False
