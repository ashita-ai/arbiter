"""Exception hierarchy for Arbiter evaluation framework.

This module defines all custom exceptions used throughout Arbiter.
All exceptions inherit from ArbiterError for easy catching.

## Exception Hierarchy:

```
ArbiterError (base)
├── ConfigurationError - Invalid configuration
├── ModelProviderError - LLM API failures
│   ├── RateLimitError - Rate limit exceeded
│   ├── AuthenticationError - Invalid API credentials
│   ├── ModelNotFoundError - Model does not exist
│   └── ContextLengthError - Input exceeds context length
├── EvaluatorError - Evaluator execution failures
│   ├── EvaluatorNotFoundError - Evaluator does not exist
│   └── EvaluatorConfigError - Invalid evaluator configuration
├── StorageError - Storage backend failures
├── PluginError - Plugin loading/execution failures
├── ValidationError - Input validation failures
├── TimeoutError - Operation timeout
├── CostLimitError - Cost threshold exceeded
└── CircuitBreakerOpenError - Circuit breaker blocking requests
```

## Usage:

    >>> from arbiter_ai.core.exceptions import (
    ...     ArbiterError,
    ...     RateLimitError,
    ...     EvaluatorNotFoundError,
    ... )
    >>>
    >>> try:
    ...     result = await evaluate(output, reference)
    ... except RateLimitError as e:
    ...     print(f"Rate limited, retry after: {e.retry_after}s")
    ... except EvaluatorNotFoundError as e:
    ...     print(f"Unknown evaluator: {e.evaluator_name}")
    ... except ArbiterError as e:
    ...     print(f"Arbiter error: {e}")
"""

from typing import Any, Dict, List, Optional

__all__ = [
    # Base
    "ArbiterError",
    # Configuration
    "ConfigurationError",
    # Model Provider errors
    "ModelProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "ContextLengthError",
    # Evaluator errors
    "EvaluatorError",
    "EvaluatorNotFoundError",
    "EvaluatorConfigError",
    # Other errors
    "StorageError",
    "PluginError",
    "ValidationError",
    "TimeoutError",
    "CostLimitError",
    "CircuitBreakerOpenError",
]


class ArbiterError(Exception):
    """Base exception for all Arbiter errors.

    All custom exceptions in Arbiter inherit from this class.
    This allows catching all Arbiter-specific errors with a single
    except clause.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize error with message and optional details.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context like
                error codes, failed values, etc.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(ArbiterError):
    """Raised when configuration is invalid.

    Examples:
        - Invalid model name
        - Out-of-range temperature value
        - Missing required API keys
        - Incompatible configuration options

    Example:
        >>> raise ConfigurationError(
        ...     "Invalid model name",
        ...     details={"model": "invalid-model", "valid_models": ["gpt-4", "gpt-3.5"]}
        ... )
    """


class ModelProviderError(ArbiterError):
    """Raised when LLM API calls fail.

    This exception covers all LLM provider errors including:
        - API authentication failures
        - Rate limiting
        - API errors (500, 503, etc.)
        - Network connectivity issues
        - Invalid requests

    This is a retryable error type - operations will automatically
    retry when this exception is raised.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        provider: Optional provider name (openai, anthropic, etc.)
        model: Optional model name that was being used

    Example:
        >>> raise ModelProviderError(
        ...     "OpenAI API error",
        ...     details={"status_code": 500},
        ...     provider="openai",
        ...     model="gpt-4o"
        ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize error with message and optional context.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
            provider: Optional provider name
            model: Optional model name
        """
        super().__init__(message, details)
        self.provider = provider
        self.model = model


class RateLimitError(ModelProviderError):
    """Raised when rate limit is exceeded.

    This exception indicates the LLM provider has rate limited the request.
    Clients should wait before retrying.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)

    Example:
        >>> raise RateLimitError(
        ...     "Rate limit exceeded",
        ...     retry_after=60.0,
        ...     provider="openai"
        ... )
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        """Initialize rate limit error.

        Args:
            message: Error message
            details: Optional dictionary with additional context
            provider: Optional provider name
            model: Optional model name
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, details, provider, model)
        self.retry_after = retry_after


class AuthenticationError(ModelProviderError):
    """Raised when API authentication fails.

    This exception indicates invalid or missing API credentials.

    Example:
        >>> raise AuthenticationError(
        ...     "Invalid API key",
        ...     provider="anthropic"
        ... )
    """


class ModelNotFoundError(ModelProviderError):
    """Raised when the requested model does not exist.

    This exception indicates the specified model name is not valid
    for the provider.

    Example:
        >>> raise ModelNotFoundError(
        ...     "Model 'gpt-5' not found",
        ...     provider="openai",
        ...     model="gpt-5"
        ... )
    """


class ContextLengthError(ModelProviderError):
    """Raised when input exceeds model context length.

    Attributes:
        max_tokens: Maximum tokens allowed by the model (if known)
        requested_tokens: Number of tokens in the request (if known)

    Example:
        >>> raise ContextLengthError(
        ...     "Input exceeds context length",
        ...     max_tokens=128000,
        ...     requested_tokens=150000
        ... )
    """

    def __init__(
        self,
        message: str = "Input exceeds model context length",
        details: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        requested_tokens: Optional[int] = None,
    ):
        """Initialize context length error.

        Args:
            message: Error message
            details: Optional dictionary with additional context
            provider: Optional provider name
            model: Optional model name
            max_tokens: Maximum tokens allowed by the model
            requested_tokens: Number of tokens in the request
        """
        super().__init__(message, details, provider, model)
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens


class EvaluatorError(ArbiterError):
    """Raised when evaluator execution fails.

    Examples:
        - Evaluator initialization failure
        - Invalid evaluator configuration
        - Evaluator processing errors
        - Score calculation failures

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        evaluator: Optional name of the evaluator that failed

    Example:
        >>> raise EvaluatorError(
        ...     "Semantic evaluator failed",
        ...     evaluator="semantic",
        ...     details={"reason": "embeddings unavailable"}
        ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        evaluator: Optional[str] = None,
    ):
        """Initialize evaluator error.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
            evaluator: Optional name of the evaluator
        """
        super().__init__(message, details)
        self.evaluator = evaluator


class EvaluatorNotFoundError(EvaluatorError):
    """Raised when the requested evaluator does not exist.

    Attributes:
        evaluator_name: Name of the evaluator that was not found
        available: List of available evaluator names

    Example:
        >>> raise EvaluatorNotFoundError(
        ...     evaluator_name="invalid_evaluator",
        ...     available=["semantic", "custom_criteria", "factuality"]
        ... )
    """

    def __init__(
        self,
        evaluator_name: str,
        available: List[str],
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize evaluator not found error.

        Args:
            evaluator_name: Name of the evaluator that was not found
            available: List of available evaluator names
            details: Optional dictionary with additional context
        """
        message = (
            f"Evaluator '{evaluator_name}' not found. "
            f"Available: {', '.join(sorted(available))}"
        )
        super().__init__(message, details, evaluator=evaluator_name)
        self.evaluator_name = evaluator_name
        self.available = available


class EvaluatorConfigError(EvaluatorError):
    """Raised when evaluator configuration is invalid.

    Example:
        >>> raise EvaluatorConfigError(
        ...     "Invalid threshold value",
        ...     evaluator="custom_criteria",
        ...     details={"threshold": -0.5, "valid_range": "0.0-1.0"}
        ... )
    """


class StorageError(ArbiterError):
    """Raised when storage operations fail.

    Examples:
        - File system errors
        - Database connection failures
        - Serialization errors
        - Storage quota exceeded

    Example:
        >>> raise StorageError(
        ...     "Failed to save evaluation result",
        ...     details={"backend": "redis", "reason": "connection timeout"}
        ... )
    """


class PluginError(ArbiterError):
    """Raised when plugin operations fail.

    Examples:
        - Plugin not found
        - Plugin loading failure
        - Invalid plugin interface
        - Plugin execution error

    Example:
        >>> raise PluginError(
        ...     "Failed to load storage plugin",
        ...     details={"plugin": "s3-storage", "reason": "missing dependencies"}
        ... )
    """


class ValidationError(ArbiterError):
    """Raised when input validation fails.

    Examples:
        - Empty output text
        - Invalid reference format
        - Out-of-range parameters
        - Missing required fields

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        field: Optional name of the field that failed validation

    Example:
        >>> raise ValidationError(
        ...     "Output text cannot be empty",
        ...     field="output"
        ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
            field: Optional name of the field that failed validation
        """
        super().__init__(message, details)
        self.field = field


class TimeoutError(ArbiterError):
    """Raised when operations exceed time limits.

    This is a retryable error type - operations may automatically
    retry when this exception is raised, depending on retry configuration.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        timeout: The timeout value that was exceeded (in seconds)

    Examples:
        - LLM API call timeout
        - Evaluation timeout
        - Storage operation timeout

    Example:
        >>> raise TimeoutError(
        ...     "Evaluation timed out",
        ...     timeout=30.0
        ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        """Initialize timeout error.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
            timeout: The timeout value that was exceeded (in seconds)
        """
        super().__init__(message, details)
        self.timeout = timeout


class CostLimitError(ArbiterError):
    """Raised when cost threshold is exceeded.

    This exception indicates that the estimated or actual cost of an
    operation would exceed the configured cost limit.

    Attributes:
        message: Human-readable error message
        details: Optional dictionary with additional error context
        limit: The cost limit that was exceeded
        estimated_cost: The estimated cost of the operation
        currency: Currency for the cost values (default: USD)

    Example:
        >>> raise CostLimitError(
        ...     "Operation would exceed cost limit",
        ...     limit=1.00,
        ...     estimated_cost=1.50
        ... )
    """

    def __init__(
        self,
        message: str = "Cost limit exceeded",
        details: Optional[Dict[str, Any]] = None,
        limit: Optional[float] = None,
        estimated_cost: Optional[float] = None,
        currency: str = "USD",
    ):
        """Initialize cost limit error.

        Args:
            message: Error message describing what went wrong
            details: Optional dictionary with additional context
            limit: The cost limit that was exceeded
            estimated_cost: The estimated cost of the operation
            currency: Currency for the cost values
        """
        super().__init__(message, details)
        self.limit = limit
        self.estimated_cost = estimated_cost
        self.currency = currency


class CircuitBreakerOpenError(ArbiterError):
    """Raised when circuit breaker is open and blocking requests.

    This exception indicates that too many failures have occurred and
    the circuit breaker has entered the OPEN state to prevent cascading
    failures. Requests are temporarily blocked until the circuit enters
    HALF_OPEN state for recovery testing.

    This is a temporary error - clients should wait and retry after
    the circuit breaker timeout period.

    Examples:
        - LLM provider experiencing outage
        - Multiple consecutive API failures
        - Provider rate limiting across multiple requests

    Example:
        >>> raise CircuitBreakerOpenError(
        ...     "Circuit breaker is open",
        ...     details={
        ...         "failure_count": 5,
        ...         "retry_after": 60,
        ...         "last_failure": "API timeout"
        ...     }
        ... )
    """
