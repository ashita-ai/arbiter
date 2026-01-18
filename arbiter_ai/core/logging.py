"""Structured logging configuration for Arbiter.

This module provides a centralized logging configuration for the Arbiter
evaluation framework. All Arbiter loggers use the 'arbiter.*' namespace,
making it easy to configure logging levels and handlers.

## Namespace Structure:

- arbiter.api: Main API functions (evaluate, compare, batch_evaluate)
- arbiter.llm: LLM client operations (API calls, retries, tokens)
- arbiter.evaluators.*: Individual evaluators (semantic, custom_criteria, etc.)
- arbiter.middleware: Middleware pipeline operations
- arbiter.storage: Storage backend operations

## Quick Start:

    >>> import logging
    >>> from arbiter import configure_logging
    >>>
    >>> # Enable debug logging for all Arbiter components
    >>> configure_logging(level=logging.DEBUG)
    >>>
    >>> # Or configure specific components
    >>> logging.getLogger("arbiter.llm").setLevel(logging.DEBUG)
    >>> logging.getLogger("arbiter.api").setLevel(logging.INFO)

## Default Behavior:

Arbiter is silent by default (WARNING level). This means you won't see any
logs unless you explicitly configure logging or an actual warning/error occurs.

## Log Levels Guide:

| Level   | When Used                                      |
|---------|------------------------------------------------|
| DEBUG   | Token counts, prompt lengths, timing details   |
| INFO    | Evaluation start/complete, batch progress      |
| WARNING | Retries, fallbacks, deprecated features        |
| ERROR   | Failed evaluations, API errors                 |
"""

import logging
from typing import Optional

__all__ = ["configure_logging", "get_logger"]

# Root logger for all Arbiter components
ARBITER_LOGGER_NAME = "arbiter"


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the Arbiter namespace prefix.

    This is the recommended way to create loggers within Arbiter modules.
    It ensures all loggers use the 'arbiter.*' namespace for consistent
    configuration.

    Args:
        name: Logger name suffix (e.g., 'api', 'llm', 'evaluators.semantic')

    Returns:
        Logger instance with name 'arbiter.{name}'

    Example:
        >>> logger = get_logger("api")
        >>> logger.name
        'arbiter.api'
        >>>
        >>> logger = get_logger("evaluators.semantic")
        >>> logger.name
        'arbiter.evaluators.semantic'
    """
    return logging.getLogger(f"{ARBITER_LOGGER_NAME}.{name}")


def configure_logging(
    level: int = logging.WARNING,
    format: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> None:
    """Configure Arbiter logging.

    Sets up logging for all Arbiter components with the specified level,
    format, and handler. This is a convenience function for quick setup.
    For more fine-grained control, configure individual loggers directly.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Default WARNING
            means Arbiter is silent unless warnings or errors occur.
        format: Custom log format string. Default shows level, logger name,
            and message in a compact format suitable for CLI output.
        handler: Custom handler for log output. Default is StreamHandler
            which writes to stderr.

    Example:
        >>> import logging
        >>> from arbiter import configure_logging
        >>>
        >>> # Enable debug logging
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Custom format with timestamps
        >>> configure_logging(
        ...     level=logging.INFO,
        ...     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ... )
        >>>
        >>> # Log to file
        >>> file_handler = logging.FileHandler("arbiter.log")
        >>> configure_logging(level=logging.DEBUG, handler=file_handler)

    Note:
        This function configures the root 'arbiter' logger. Child loggers
        (arbiter.api, arbiter.llm, etc.) inherit this configuration unless
        explicitly overridden.
    """
    logger = logging.getLogger(ARBITER_LOGGER_NAME)
    logger.setLevel(level)

    # Only add handler if none exist (avoid duplicate handlers on repeated calls)
    if not logger.handlers:
        h = handler or logging.StreamHandler()
        fmt = format or "%(levelname)s:%(name)s:%(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)
    elif handler is not None:
        # If a custom handler is provided, replace existing handlers
        logger.handlers.clear()
        fmt = format or "%(levelname)s:%(name)s:%(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    elif format is not None:
        # Update format on existing handlers
        fmt_obj = logging.Formatter(format)
        for h in logger.handlers:
            h.setFormatter(fmt_obj)
