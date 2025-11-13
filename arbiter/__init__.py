"""Arbiter: Streaming LLM evaluation framework with semantic comparison.

Arbiter provides composable evaluation primitives for assessing LLM outputs
through multiple metrics including semantic similarity, factuality, consistency,
and custom criteria. Built on PydanticAI with optional streaming support.

## Quick Start:

    >>> from arbiter import evaluate
    >>>
    >>> # Simple evaluation with automatic client management
    >>> result = await evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris",
    ...     evaluators=["semantic"],
    ...     model="gpt-4o"
    ... )
    >>> print(f"Score: {result.overall_score}")
    >>>
    >>> # Or use evaluator directly for more control
    >>> from arbiter import SemanticEvaluator, LLMManager
    >>> client = await LLMManager.get_client(model="gpt-4o")
    >>> evaluator = SemanticEvaluator(client)
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris"
    ... )
    >>> print(f"Score: {score.value}")

## Key Features:

- **Semantic Comparison**: Milvus-backed vector similarity for output comparison
- **Multiple Metrics**: Factuality, consistency, semantic similarity, and more
- **Composable**: Build custom evaluation pipelines with middleware
- **Streaming Ready**: Optional ByteWax integration for streaming data
- **Full Observability**: Complete audit trails and performance metrics

## Main Components:

- `evaluate()`: Primary async function for evaluation
- `SemanticEvaluator`: Evaluator with semantic comparison
- `EvaluationResult`: Contains scores, metrics, and metadata
- `Config`: Configuration object for models and behavior

For more information, see the documentation at:
https://docs.arbiter.ai/
"""

from dotenv import load_dotenv

# Core API
from .api import compare, evaluate

# Core components
from .core import (
    ArbiterError,
    BaseEvaluator,
    CachingMiddleware,
    ComparisonResult,
    ConfigurationError,
    ConnectionMetrics,
    EvaluationResult,
    EvaluatorError,
    LLMClient,
    LLMInteraction,
    LLMManager,
    LoggingMiddleware,
    Metric,
    MetricsMiddleware,
    MetricType,
    Middleware,
    MiddlewarePipeline,
    ModelProviderError,
    PerformanceMetrics,
    PerformanceMonitor,
    PluginError,
    Provider,
    RateLimitingMiddleware,
    RetryConfig,
    Score,
    StorageBackend,
    StorageError,
    StorageType,
    TimeoutError,
    ValidationError,
    get_global_monitor,
    monitor,
)

# Evaluators
from .evaluators import (
    BasePydanticEvaluator,
    CustomCriteriaEvaluator,
    PairwiseComparisonEvaluator,
    SemanticEvaluator,
)

# Load environment variables
load_dotenv()

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Main API
    "evaluate",
    "compare",
    # Evaluators
    "SemanticEvaluator",
    "CustomCriteriaEvaluator",
    "PairwiseComparisonEvaluator",
    "BasePydanticEvaluator",
    # Core Models
    "EvaluationResult",
    "ComparisonResult",
    "Score",
    "Metric",
    "LLMInteraction",
    # Interfaces
    "BaseEvaluator",
    "StorageBackend",
    # LLM Client
    "LLMClient",
    "LLMManager",
    "Provider",
    "ConnectionMetrics",
    # Middleware
    "Middleware",
    "MiddlewarePipeline",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "CachingMiddleware",
    "RateLimitingMiddleware",
    # Monitoring
    "PerformanceMetrics",
    "PerformanceMonitor",
    "get_global_monitor",
    "monitor",
    # Configuration
    "RetryConfig",
    # Types
    "Provider",
    "MetricType",
    "StorageType",
    # Exceptions
    "ArbiterError",
    "ConfigurationError",
    "ModelProviderError",
    "EvaluatorError",
    "ValidationError",
    "StorageError",
    "PluginError",
    "TimeoutError",
]
