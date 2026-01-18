"""Tests for structured logging integration.

Tests verify that:
- configure_logging() works correctly
- get_logger() returns loggers with the arbiter.* namespace
- Log messages are emitted at appropriate levels
- Silent by default (WARNING level)
"""

import logging
from io import StringIO
from unittest.mock import AsyncMock, patch

import pytest

from arbiter_ai import configure_logging, get_logger
from arbiter_ai.core.logging import ARBITER_LOGGER_NAME


class TestConfigureLogging:
    """Tests for configure_logging() function."""

    def setup_method(self) -> None:
        """Reset logging state before each test."""
        # Clear any existing handlers on the arbiter logger
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def teardown_method(self) -> None:
        """Clean up logging state after each test."""
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_configure_logging_default_level(self) -> None:
        """Default level should be WARNING (silent by default)."""
        configure_logging()
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert logger.level == logging.WARNING

    def test_configure_logging_debug_level(self) -> None:
        """Can configure DEBUG level."""
        configure_logging(level=logging.DEBUG)
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert logger.level == logging.DEBUG

    def test_configure_logging_info_level(self) -> None:
        """Can configure INFO level."""
        configure_logging(level=logging.INFO)
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert logger.level == logging.INFO

    def test_configure_logging_adds_handler(self) -> None:
        """configure_logging() should add a handler."""
        configure_logging()
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_configure_logging_no_duplicate_handlers(self) -> None:
        """Calling configure_logging() multiple times should not add duplicate handlers."""
        configure_logging()
        configure_logging()
        configure_logging()
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert len(logger.handlers) == 1

    def test_configure_logging_custom_format(self) -> None:
        """Can configure custom log format."""
        custom_format = "%(name)s - %(message)s"
        configure_logging(format=custom_format)
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert logger.handlers[0].formatter is not None
        assert logger.handlers[0].formatter._fmt == custom_format

    def test_configure_logging_custom_handler(self) -> None:
        """Can configure custom handler."""
        stream = StringIO()
        custom_handler = logging.StreamHandler(stream)
        configure_logging(level=logging.INFO, handler=custom_handler)

        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.info("test message")

        output = stream.getvalue()
        assert "test message" in output

    def test_configure_logging_replaces_handler_when_custom_provided(self) -> None:
        """Custom handler should replace existing handlers."""
        # First configure with default
        configure_logging()
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        assert len(logger.handlers) == 1

        # Now configure with custom handler
        stream = StringIO()
        custom_handler = logging.StreamHandler(stream)
        configure_logging(handler=custom_handler)

        # Should have replaced the handler
        assert len(logger.handlers) == 1
        assert logger.handlers[0] is custom_handler


class TestGetLogger:
    """Tests for get_logger() function."""

    def test_get_logger_returns_logger(self) -> None:
        """get_logger() should return a Logger instance."""
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_get_logger_namespace_prefix(self) -> None:
        """Loggers should have the arbiter.* namespace."""
        logger = get_logger("api")
        assert logger.name == "arbiter.api"

        logger = get_logger("llm")
        assert logger.name == "arbiter.llm"

        logger = get_logger("evaluators.semantic")
        assert logger.name == "arbiter.evaluators.semantic"

    def test_get_logger_inherits_from_root(self) -> None:
        """Child loggers should inherit from root arbiter logger."""
        # Configure root logger
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        configure_logging(level=logging.DEBUG, handler=handler)

        # Get child logger and log
        child_logger = get_logger("test_child")
        child_logger.debug("child message")

        # Child message should appear in root handler's stream
        output = stream.getvalue()
        assert "child message" in output


class TestLoggerNamespaces:
    """Tests for logger namespace conventions."""

    def test_api_logger_namespace(self) -> None:
        """API module should use arbiter.api namespace."""
        from arbiter_ai.api import logger

        assert logger.name == "arbiter.api"

    def test_middleware_logger_namespace(self) -> None:
        """Middleware module should use arbiter.middleware namespace."""
        from arbiter_ai.core.middleware import logger

        assert logger.name == "arbiter.middleware"

    def test_llm_client_logger_namespace(self) -> None:
        """LLM client module should use arbiter.llm namespace."""
        from arbiter_ai.core.llm_client import logger

        assert logger.name == "arbiter.llm"

    def test_evaluators_base_logger_namespace(self) -> None:
        """Evaluators base module should use arbiter.evaluators namespace."""
        from arbiter_ai.evaluators.base import logger

        assert logger.name == "arbiter.evaluators"


class TestLoggingOutput:
    """Tests for logging output content."""

    def setup_method(self) -> None:
        """Set up logging capture."""
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        configure_logging(level=logging.DEBUG, handler=self.handler)

    def teardown_method(self) -> None:
        """Clean up logging state."""
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_debug_log_content(self) -> None:
        """DEBUG logs should include detailed information."""
        logger = get_logger("test")
        logger.debug("Token count: %d", 1234)

        output = self.stream.getvalue()
        assert "Token count: 1234" in output
        assert "DEBUG" in output

    def test_info_log_content(self) -> None:
        """INFO logs should include operation information."""
        logger = get_logger("test")
        logger.info("Evaluation complete: score=%.2f", 0.85)

        output = self.stream.getvalue()
        assert "Evaluation complete: score=0.85" in output
        assert "INFO" in output

    def test_warning_log_content(self) -> None:
        """WARNING logs should include issue information."""
        logger = get_logger("test")
        logger.warning("Retry attempt %d of %d", 2, 3)

        output = self.stream.getvalue()
        assert "Retry attempt 2 of 3" in output
        assert "WARNING" in output

    def test_error_log_content(self) -> None:
        """ERROR logs should include error details."""
        logger = get_logger("test")
        logger.error("API call failed: %s", "timeout")

        output = self.stream.getvalue()
        assert "API call failed: timeout" in output
        assert "ERROR" in output


class TestSilentByDefault:
    """Tests verifying logging is silent by default."""

    def setup_method(self) -> None:
        """Reset to default state."""
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def teardown_method(self) -> None:
        """Clean up."""
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    def test_debug_silent_by_default(self) -> None:
        """DEBUG logs should not appear when using default configuration."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        configure_logging()  # Default WARNING level
        root_logger = logging.getLogger(ARBITER_LOGGER_NAME)
        root_logger.addHandler(handler)

        # Log at DEBUG level
        child_logger = get_logger("test")
        child_logger.debug("This should not appear")

        output = stream.getvalue()
        assert "This should not appear" not in output

    def test_info_silent_by_default(self) -> None:
        """INFO logs should not appear when using default configuration."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        configure_logging()  # Default WARNING level
        root_logger = logging.getLogger(ARBITER_LOGGER_NAME)
        root_logger.addHandler(handler)

        # Log at INFO level
        child_logger = get_logger("test")
        child_logger.info("This should not appear")

        output = stream.getvalue()
        assert "This should not appear" not in output

    def test_warning_visible_by_default(self) -> None:
        """WARNING logs should appear when using default configuration."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))

        configure_logging()  # Default WARNING level
        root_logger = logging.getLogger(ARBITER_LOGGER_NAME)
        # Clear and add our test handler
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

        # Log at WARNING level
        child_logger = get_logger("test")
        child_logger.warning("This should appear")

        output = stream.getvalue()
        assert "This should appear" in output


class TestAPILogging:
    """Tests for logging in API functions."""

    def setup_method(self) -> None:
        """Set up logging capture."""
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        configure_logging(level=logging.DEBUG, handler=self.handler)

    def teardown_method(self) -> None:
        """Clean up."""
        logger = logging.getLogger(ARBITER_LOGGER_NAME)
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)

    @pytest.mark.asyncio
    async def test_evaluate_logs_start(self) -> None:
        """evaluate() should log at start."""
        from arbiter_ai import evaluate

        # Mock the internal evaluation to avoid actual API calls
        with patch("arbiter_ai.api._evaluate_impl") as mock_impl:
            mock_result = AsyncMock()
            mock_result.overall_score = 0.85
            mock_result.passed = True
            mock_result.total_tokens = 100
            mock_result.processing_time = 1.0
            mock_result.evaluator_names = ["semantic"]
            mock_result.interactions = []
            mock_impl.return_value = mock_result

            try:
                await evaluate(
                    output="test output",
                    reference="test reference",
                    evaluators=["semantic"],
                )
            except Exception:
                pass  # We expect this to fail without real API, that's fine

        output = self.stream.getvalue()
        # Check that we logged the start
        assert "INFO:arbiter.api:" in output or "Starting evaluation" in output

    @pytest.mark.asyncio
    async def test_batch_evaluate_logs_start(self) -> None:
        """batch_evaluate() should log at start."""
        from arbiter_ai import batch_evaluate

        with patch("arbiter_ai.api._batch_evaluate_impl") as mock_impl:
            mock_result = AsyncMock()
            mock_result.successful_items = 2
            mock_result.total_items = 2
            mock_result.failed_items = 0
            mock_result.total_tokens = 200
            mock_result.processing_time = 2.0
            mock_impl.return_value = mock_result

            try:
                await batch_evaluate(
                    items=[
                        {"output": "test1", "reference": "ref1"},
                        {"output": "test2", "reference": "ref2"},
                    ],
                    evaluators=["semantic"],
                )
            except Exception:
                pass

        output = self.stream.getvalue()
        assert "arbiter.api" in output
