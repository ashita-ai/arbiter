"""Environment variable configuration for Arbiter.

This module provides functions to read default configuration values from
environment variables. This allows setting project-wide defaults without
modifying code.

## Environment Variables:

| Variable                  | Default      | Description                     |
|---------------------------|--------------|---------------------------------|
| `ARBITER_DEFAULT_MODEL`   | `gpt-4o-mini`| Default LLM model for evals     |
| `ARBITER_DEFAULT_THRESHOLD`| `0.7`       | Default pass/fail threshold     |
| `ARBITER_TIMEOUT`         | (none)       | Default timeout in seconds      |
| `ARBITER_MAX_CONCURRENCY` | `10`         | Default batch concurrency       |

## Usage:

    >>> import os
    >>> os.environ["ARBITER_DEFAULT_MODEL"] = "claude-3-5-sonnet"
    >>>
    >>> from arbiter_ai.core.config import get_default_model
    >>> get_default_model()
    'claude-3-5-sonnet'
    >>>
    >>> # Or use with evaluate() - env vars are used when params are None
    >>> from arbiter_ai import evaluate
    >>> result = await evaluate(output="...", reference="...")
    >>> # Uses claude-3-5-sonnet from environment
"""

import os
from typing import Optional

from .exceptions import ConfigurationError

__all__ = [
    "get_default_model",
    "get_default_threshold",
    "get_default_timeout",
    "get_default_max_concurrency",
]


def get_default_model() -> str:
    """Get default model from environment or fallback.

    Reads from ARBITER_DEFAULT_MODEL environment variable.

    Returns:
        Model name string

    Example:
        >>> import os
        >>> os.environ["ARBITER_DEFAULT_MODEL"] = "gpt-4o"
        >>> get_default_model()
        'gpt-4o'
    """
    return os.getenv("ARBITER_DEFAULT_MODEL", "gpt-4o-mini")


def get_default_threshold() -> float:
    """Get default threshold from environment or fallback.

    Reads from ARBITER_DEFAULT_THRESHOLD environment variable.

    Returns:
        Threshold as float between 0.0 and 1.0

    Raises:
        ConfigurationError: If threshold value is invalid

    Example:
        >>> import os
        >>> os.environ["ARBITER_DEFAULT_THRESHOLD"] = "0.8"
        >>> get_default_threshold()
        0.8
    """
    value = os.getenv("ARBITER_DEFAULT_THRESHOLD", "0.7")
    try:
        threshold = float(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid ARBITER_DEFAULT_THRESHOLD: '{value}' is not a valid number",
            details={"env_var": "ARBITER_DEFAULT_THRESHOLD", "value": value},
        )

    if not 0.0 <= threshold <= 1.0:
        raise ConfigurationError(
            f"Invalid ARBITER_DEFAULT_THRESHOLD: {threshold} must be between 0.0 and 1.0",
            details={"env_var": "ARBITER_DEFAULT_THRESHOLD", "value": threshold},
        )

    return threshold


def get_default_timeout() -> Optional[float]:
    """Get default timeout from environment or None.

    Reads from ARBITER_TIMEOUT environment variable.

    Returns:
        Timeout in seconds as float, or None if not set

    Raises:
        ConfigurationError: If timeout value is invalid

    Example:
        >>> import os
        >>> os.environ["ARBITER_TIMEOUT"] = "30"
        >>> get_default_timeout()
        30.0
    """
    value = os.getenv("ARBITER_TIMEOUT")
    if value is None:
        return None

    try:
        timeout = float(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid ARBITER_TIMEOUT: '{value}' is not a valid number",
            details={"env_var": "ARBITER_TIMEOUT", "value": value},
        )

    if timeout <= 0:
        raise ConfigurationError(
            f"Invalid ARBITER_TIMEOUT: {timeout} must be positive",
            details={"env_var": "ARBITER_TIMEOUT", "value": timeout},
        )

    return timeout


def get_default_max_concurrency() -> int:
    """Get default max concurrency from environment or fallback.

    Reads from ARBITER_MAX_CONCURRENCY environment variable.

    Returns:
        Max concurrency as positive integer

    Raises:
        ConfigurationError: If concurrency value is invalid

    Example:
        >>> import os
        >>> os.environ["ARBITER_MAX_CONCURRENCY"] = "20"
        >>> get_default_max_concurrency()
        20
    """
    value = os.getenv("ARBITER_MAX_CONCURRENCY", "10")
    try:
        concurrency = int(value)
    except ValueError:
        raise ConfigurationError(
            f"Invalid ARBITER_MAX_CONCURRENCY: '{value}' is not a valid integer",
            details={"env_var": "ARBITER_MAX_CONCURRENCY", "value": value},
        )

    if concurrency <= 0:
        raise ConfigurationError(
            f"Invalid ARBITER_MAX_CONCURRENCY: {concurrency} must be positive",
            details={"env_var": "ARBITER_MAX_CONCURRENCY", "value": concurrency},
        )

    return concurrency
