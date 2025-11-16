# Arbiter Implementation Specification

**Version:** 1.0
**Status:** Active
**Purpose:** Implementation requirements, coding standards, and development guidelines
**Last Updated:** 2025-11-16

---

## Table of Contents

1. [Coding Standards](#coding-standards)
2. [Component Implementation](#component-implementation)
3. [Testing Requirements](#testing-requirements)
4. [Documentation Standards](#documentation-standards)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Performance Requirements](#performance-requirements)
7. [Deployment Configuration](#deployment-configuration)
8. [Development Workflow](#development-workflow)

---

## Coding Standards

### Type Annotations

**Requirement:** All functions MUST have complete type annotations

```python
# ✅ CORRECT - Complete type annotations
async def evaluate(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: List[Union[str, BaseEvaluator]] = None,
    model: str = "gpt-4o-mini",
    threshold: float = 0.7,
    **kwargs: Any
) -> EvaluationResult:
    """Evaluate LLM output."""
    ...

# ❌ INCORRECT - Missing type annotations
async def evaluate(output, reference=None, **kwargs):
    """Evaluate LLM output."""
    ...
```

**Mypy Configuration:**
```ini
# mypy.ini
[mypy]
python_version = 3.10
strict = True
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_any_unimported = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
```

**Type Hint Guidelines:**

```python
# Use Protocol for duck typing
from typing import Protocol

class Middleware(Protocol):
    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[..., Awaitable[EvaluationResult]],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        ...

# Use TypedDict for structured dicts
from typing import TypedDict

class EvaluateRequest(TypedDict, total=False):
    output: str                                    # Required
    reference: Optional[str]                       # Optional
    evaluators: List[Union[str, BaseEvaluator]]   # Optional

# Use Generic for reusable types
from typing import Generic, TypeVar

T = TypeVar('T')

class Result(Generic[T]):
    value: T
    error: Optional[Exception]

# Use Literal for constrained strings
from typing import Literal

Provider = Literal["openai", "anthropic", "google", "groq"]

def get_client(provider: Provider) -> LLMClient:
    ...
```

---

### Async/Await Patterns

**Requirement:** All I/O operations MUST be async

```python
# ✅ CORRECT - Async I/O
async def evaluate(output: str, ...) -> EvaluationResult:
    client = await LLMManager.get_client(model="gpt-4o-mini")
    result = await evaluator.evaluate(output, reference)
    await client.close()
    return result

# ❌ INCORRECT - Blocking I/O
def evaluate(output: str, ...) -> EvaluationResult:
    response = requests.post(url, json=data)  # Blocks event loop
    return result
```

**Async Best Practices:**

```python
# Pattern 1: Context manager for resource cleanup
async with LLMManager.get_client(model="gpt-4o-mini") as client:
    evaluator = SemanticEvaluator(client)
    result = await evaluator.evaluate(output, reference)
# client.close() called automatically

# Pattern 2: Manual cleanup in finally
client = None
try:
    client = await LLMManager.get_client(model="gpt-4o-mini")
    result = await evaluator.evaluate(output, reference)
finally:
    if client:
        await client.close()

# Pattern 3: Concurrent execution with asyncio.gather
results = await asyncio.gather(
    evaluator1.evaluate(output1, reference1),
    evaluator2.evaluate(output2, reference2),
    evaluator3.evaluate(output3, reference3),
    return_exceptions=True  # Don't fail all on one error
)

# Pattern 4: Sequential with error handling
try:
    result1 = await evaluator1.evaluate(output1, reference1)
    result2 = await evaluator2.evaluate(output2, reference2)
except TimeoutError:
    logger.error("Evaluation timed out")
    raise
```

---

### Error Handling

**Requirement:** Use specific exception types, avoid bare `except`

```python
# ✅ CORRECT - Specific exceptions
try:
    result = await agent.run(user_prompt)
except asyncio.TimeoutError:
    logger.error(f"LLM request timed out after {timeout}s")
    raise TimeoutError(f"Evaluation timed out after {timeout}s")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        retry_after = e.response.headers.get("Retry-After", 60)
        raise RateLimitError(f"Rate limited, retry after {retry_after}s")
    else:
        raise ModelProviderError(f"Provider error: {e.response.status_code}")
except ValidationError as e:
    logger.error(f"Invalid response: {e}")
    raise EvaluatorError(f"Malformed LLM response: {e}")

# ❌ INCORRECT - Bare except
try:
    result = await agent.run(user_prompt)
except:  # Catches KeyboardInterrupt, SystemExit!
    logger.error("Something went wrong")
    raise
```

**Exception Hierarchy:**

```python
# Custom exception base
class ArbiterError(Exception):
    """Base exception for all Arbiter errors."""
    pass

# Configuration errors
class ConfigurationError(ArbiterError):
    """Invalid configuration (API keys, model names)."""
    pass

# Provider errors
class ModelProviderError(ArbiterError):
    """LLM provider API errors."""
    pass

class TimeoutError(ModelProviderError):
    """LLM request timeout."""
    pass

class RateLimitError(ModelProviderError):
    """Rate limit exceeded."""
    pass

class CircuitBreakerOpen(ModelProviderError):
    """Circuit breaker is open."""
    pass

# Evaluator errors
class EvaluatorError(ArbiterError):
    """Evaluation failed."""
    pass

class ValidationError(ArbiterError):
    """Input validation failed."""
    pass

# Usage
if not output:
    raise ValidationError("Output cannot be empty")

if provider not in SUPPORTED_PROVIDERS:
    raise ConfigurationError(
        f"Unsupported provider: {provider}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )
```

---

### Logging Standards

**Requirement:** Use structured logging with consistent format

```python
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ✅ CORRECT - Structured logging
def log_evaluation_start(
    request_id: str,
    evaluator: str,
    model: str,
    output_length: int
) -> None:
    logger.info(
        "Evaluation started",
        extra={
            "request_id": request_id,
            "evaluator": evaluator,
            "model": model,
            "output_length": output_length
        }
    )

# ✅ CORRECT - Error logging with context
def log_evaluation_error(
    request_id: str,
    error: Exception,
    context: Dict[str, Any]
) -> None:
    logger.error(
        f"Evaluation failed: {error}",
        extra={
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **context
        },
        exc_info=True  # Include stack trace
    )

# ❌ INCORRECT - Unstructured logging
logger.info(f"Starting evaluation for {model}")  # Hard to parse
```

**Log Levels:**

```python
# DEBUG: Detailed diagnostic information
logger.debug(f"System prompt: {system_prompt[:100]}...")

# INFO: General informational messages
logger.info(f"Evaluation completed: score={score:.2f}")

# WARNING: Potentially problematic situations
logger.warning(f"Large input truncated: {len(output)} -> 50000 chars")

# ERROR: Error events that might allow application to continue
logger.error(f"LLM request failed: {error}", exc_info=True)

# CRITICAL: Very severe error events
logger.critical(f"Connection pool exhausted, system degraded")
```

**Logging Configuration:**

```python
# config/logging.py
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
        },
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "arbiter.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": "DEBUG"
        }
    },
    "loggers": {
        "arbiter": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

---

### Code Formatting

**Tools:**
- **Black:** Automatic code formatting (line length 88)
- **Ruff:** Fast Python linter (replaces flake8, isort, etc.)

```bash
# Format code
black arbiter/ tests/

# Check formatting (CI)
black --check arbiter/ tests/

# Lint code
ruff check arbiter/ tests/

# Fix auto-fixable issues
ruff check --fix arbiter/ tests/

# Sort imports
ruff check --select I --fix arbiter/ tests/
```

**Configuration:**

```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
]

[tool.ruff.lint.isort]
known-first-party = ["arbiter"]
```

---

## Component Implementation

### LLMClient Implementation

**File:** `arbiter/core/llm_client.py`

**Implementation Requirements:**

```python
class LLMClient:
    """LLM client implementation checklist.

    REQUIRED INITIALIZATION:
    - ✅ Store provider, model, http_client, retry_config
    - ✅ Initialize PydanticAI Model instance
    - ✅ Get or create circuit breaker for provider

    REQUIRED METHODS:
    - ✅ create_agent(response_type, system_prompt, retries) -> Agent
    - ✅ close() -> None

    IMPLEMENTATION CHECKLIST:
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        http_client: httpx.AsyncClient,
        retry_config: RetryConfig = RetryConfig.STANDARD,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        # ✅ 1. Validate inputs
        if not model:
            raise ConfigurationError("Model name is required")

        if provider not in Provider.__members__.values():
            raise ConfigurationError(f"Unsupported provider: {provider}")

        # ✅ 2. Store configuration
        self.provider = provider
        self.model_name = model
        self.http_client = http_client
        self.retry_config = retry_config

        # ✅ 3. Initialize circuit breaker (shared per provider)
        self.circuit_breaker = circuit_breaker or CircuitBreaker.get_or_create(provider)

        # ✅ 4. Create PydanticAI model
        self.model = Model(provider=provider.value, model_name=model)

    async def create_agent(
        self,
        response_type: Type[BaseModel],
        system_prompt: str,
        retries: Optional[int] = None
    ) -> Agent:
        """Create PydanticAI agent.

        IMPLEMENTATION CHECKLIST:
        - ✅ Check circuit breaker state (fail fast if open)
        - ✅ Validate response_type is Pydantic BaseModel
        - ✅ Validate system_prompt is non-empty and <2000 chars
        - ✅ Use retry_config or override with retries param
        - ✅ Create Agent with model and response_type
        - ✅ Return configured agent
        """
        # ✅ 1. Check circuit breaker
        if self.circuit_breaker.is_open():
            raise CircuitBreakerOpen(
                f"Circuit breaker open for provider {self.provider}"
            )

        # ✅ 2. Validate inputs
        if not issubclass(response_type, BaseModel):
            raise ValidationError("response_type must be Pydantic BaseModel")

        if not system_prompt or len(system_prompt) > 2000:
            raise ValidationError("system_prompt must be 1-2000 chars")

        # ✅ 3. Create agent
        agent = Agent(
            model=self.model,
            result_type=response_type,
            system_prompt=system_prompt,
            retries=retries or self.retry_config.max_retries
        )

        return agent

    async def close(self) -> None:
        """Close HTTP client.

        IMPLEMENTATION CHECKLIST:
        - ✅ Close HTTPX client (drains connection pool)
        - ✅ Do NOT close circuit breaker (shared resource)
        """
        await self.http_client.aclose()
```

**Testing Checklist:**

```python
# tests/unit/test_llm_client.py

def test_init_validates_provider():
    """Test that init rejects invalid providers."""
    with pytest.raises(ConfigurationError):
        LLMClient(provider="invalid", ...)

def test_init_validates_model():
    """Test that init rejects empty model names."""
    with pytest.raises(ConfigurationError):
        LLMClient(provider="openai", model="", ...)

async def test_create_agent_checks_circuit_breaker():
    """Test that create_agent fails fast if circuit open."""
    client.circuit_breaker.state = CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        await client.create_agent(...)

async def test_create_agent_validates_response_type():
    """Test that create_agent rejects non-Pydantic types."""
    with pytest.raises(ValidationError):
        await client.create_agent(response_type=dict, ...)

async def test_close_drains_connections():
    """Test that close drains HTTP client connections."""
    await client.close()
    assert client.http_client.is_closed
```

---

### Evaluator Implementation

**Base Class:** `arbiter/evaluators/base.py`

**Implementation Requirements:**

```python
class BasePydanticEvaluator(ABC):
    """Template method pattern for evaluators.

    SUBCLASS IMPLEMENTATION CHECKLIST:
    - ✅ Implement name property (lowercase, alphanumeric + underscores)
    - ✅ Implement _get_system_prompt() (return <2000 char string)
    - ✅ Implement _get_user_prompt() (include output, handle None reference/criteria)
    - ✅ Implement _get_response_type() (return Pydantic BaseModel with score field)
    - ✅ Implement _compute_score() (transform response to Score object)
    - ✅ Register evaluator in registry (arbiter/core/registry.py)
    - ✅ Export in __init__.py (arbiter/evaluators/__init__.py)
    - ✅ Add to __all__ in top-level __init__.py (arbiter/__init__.py)
    """

    def __init__(self, llm_client: LLMClient):
        """Initialize evaluator.

        IMPLEMENTATION:
        - ✅ Store llm_client reference
        - ✅ Initialize empty interactions list
        - ✅ No other mutable state
        """
        self.llm_client = llm_client
        self.interactions: List[LLMInteraction] = []

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> Score:
        """Evaluate output (template method).

        IMPLEMENTATION (DO NOT OVERRIDE):
        1. ✅ Validate inputs (output not empty)
        2. ✅ Get system_prompt from subclass
        3. ✅ Get user_prompt from subclass
        4. ✅ Get response_type from subclass
        5. ✅ Create agent via llm_client
        6. ✅ Call agent.run() with retry logic
        7. ✅ Transform response via _compute_score()
        8. ✅ Track interaction (tokens, latency, purpose)
        9. ✅ Return Score object
        """
        # ✅ 1. Validate inputs
        if not output:
            raise ValidationError("Output cannot be empty")

        # ✅ 2-4. Get prompts and response type
        try:
            system_prompt = self._get_system_prompt()
            user_prompt = self._get_user_prompt(output, reference, criteria)
            response_type = self._get_response_type()
        except Exception as e:
            raise EvaluatorError(f"Prompt generation failed: {e}")

        # ✅ 5-6. Call LLM
        start_time = time.time()
        try:
            agent = await self.llm_client.create_agent(
                response_type=response_type,
                system_prompt=system_prompt
            )

            result = await agent.run(user_prompt)
            response = result.data

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            raise TimeoutError(f"LLM request timed out after {elapsed:.1f}s")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After", 60)
                raise RateLimitError(f"Rate limited, retry after {retry_after}s")
            else:
                raise ModelProviderError(f"Provider error: {e.response.status_code}")

        except ValidationError as e:
            raise EvaluatorError(f"Malformed LLM response: {e}")

        # ✅ 7. Compute score
        try:
            score = await self._compute_score(response)
        except Exception as e:
            raise EvaluatorError(f"Score computation failed: {e}")

        # ✅ 8. Track interaction
        elapsed = time.time() - start_time
        interaction = LLMInteraction(
            prompt=user_prompt,
            response=str(response),
            model=self.llm_client.model_name,
            tokens_used=result.usage().total_tokens if hasattr(result, 'usage') else 0,
            latency=elapsed,
            purpose=f"{self.name}_evaluation",
            metadata={
                "evaluator": self.name,
                "has_reference": reference is not None,
                "has_criteria": criteria is not None
            }
        )
        self.interactions.append(interaction)

        # ✅ 9. Return score
        return score
```

**Example Implementation:**

```python
# arbiter/evaluators/my_evaluator.py

from typing import Type, Optional
from pydantic import BaseModel, Field
from arbiter.core.models import Score
from arbiter.evaluators.base import BasePydanticEvaluator

class MyEvaluatorResponse(BaseModel):
    """Response model for MyEvaluator.

    REQUIREMENTS:
    - ✅ Must have 'score' field (0.0-1.0)
    - ✅ Should have 'confidence' field (0.0-1.0)
    - ✅ Should have 'explanation' field
    - ✅ Can have domain-specific fields
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Evaluation score")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence")
    explanation: str = Field(..., description="Human-readable explanation")

    # Domain-specific fields
    strengths: List[str] = Field(default_factory=list, description="What was good")
    weaknesses: List[str] = Field(default_factory=list, description="What needs improvement")


class MyEvaluator(BasePydanticEvaluator):
    """My custom evaluator.

    Purpose: [Describe what this evaluator assesses]

    Requires:
    - reference: [Yes/No, and why]
    - criteria: [Yes/No, and why]

    Returns:
    - Score with value 0.0-1.0
    - Metadata: {strengths, weaknesses, ...}
    """

    @property
    def name(self) -> str:
        """Evaluator name (must be unique)."""
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        """Return system prompt for LLM.

        GUIDELINES:
        - Define expert role clearly
        - Specify output format expectations
        - Emphasize objectivity
        - Keep under 2000 chars
        """
        return """You are an expert evaluator assessing [WHAT YOU EVALUATE].

Your task is to:
1. Analyze the output carefully
2. Identify strengths and weaknesses
3. Assign a score from 0.0 to 1.0
4. Provide clear explanations

Be objective and consistent."""

    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        """Return user prompt with evaluation task.

        GUIDELINES:
        - Include all relevant inputs
        - Provide clear instructions
        - Handle missing reference/criteria gracefully
        """
        prompt = f"Output to evaluate:\n{output}\n\n"

        if reference:
            prompt += f"Reference:\n{reference}\n\n"

        if criteria:
            prompt += f"Evaluation criteria:\n{criteria}\n\n"

        prompt += "Provide your evaluation with score, confidence, explanation, strengths, and weaknesses."

        return prompt

    def _get_response_type(self) -> Type[BaseModel]:
        """Return Pydantic response model."""
        return MyEvaluatorResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Transform LLM response to Score object.

        GUIDELINES:
        - Extract score value (required)
        - Extract confidence (use default if missing)
        - Build explanation from response
        - Add metadata dict with domain-specific fields
        """
        resp = cast(MyEvaluatorResponse, response)

        # Build explanation
        explanation = resp.explanation

        if resp.strengths:
            explanation += f"\n\nStrengths:\n" + "\n".join(f"- {s}" for s in resp.strengths)

        if resp.weaknesses:
            explanation += f"\n\nWeaknesses:\n" + "\n".join(f"- {w}" for w in resp.weaknesses)

        # Build metadata
        metadata = {
            "strengths": resp.strengths,
            "weaknesses": resp.weaknesses,
            "strength_count": len(resp.strengths),
            "weakness_count": len(resp.weaknesses)
        }

        return Score(
            name=self.name,
            value=resp.score,
            confidence=resp.confidence,
            explanation=explanation,
            metadata=metadata
        )
```

**Registration:**

```python
# arbiter/core/registry.py

def _initialize_builtin_evaluators() -> None:
    """Initialize registry with built-in evaluators."""
    from ..evaluators import (
        SemanticEvaluator,
        CustomCriteriaEvaluator,
        FactualityEvaluator,
        GroundednessEvaluator,
        RelevanceEvaluator,
        MyEvaluator,  # ✅ Import
    )

    AVAILABLE_EVALUATORS["semantic"] = SemanticEvaluator
    AVAILABLE_EVALUATORS["custom_criteria"] = CustomCriteriaEvaluator
    AVAILABLE_EVALUATORS["factuality"] = FactualityEvaluator
    AVAILABLE_EVALUATORS["groundedness"] = GroundednessEvaluator
    AVAILABLE_EVALUATORS["relevance"] = RelevanceEvaluator
    AVAILABLE_EVALUATORS["my_evaluator"] = MyEvaluator  # ✅ Register
```

**Export:**

```python
# arbiter/evaluators/__init__.py

from .my_evaluator import MyEvaluator, MyEvaluatorResponse  # ✅ Import

__all__ = [
    "BasePydanticEvaluator",
    "SemanticEvaluator",
    "MyEvaluator",  # ✅ Export
    ...
]
```

---

### Middleware Implementation

**Base Protocol:** `arbiter/core/middleware.py`

**Implementation Requirements:**

```python
class Middleware(Protocol):
    """Protocol for middleware components.

    IMPLEMENTATION CHECKLIST:
    - ✅ Implement process(output, reference, next_handler, context) method
    - ✅ Call next_handler with (potentially modified) arguments
    - ✅ Return (potentially modified) result from next_handler
    - ✅ Handle errors appropriately (catch, transform, or propagate)
    - ✅ Cleanup resources in finally block if acquired
    - ✅ Thread-safe if stateful (use locks)
    """

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[..., Awaitable[EvaluationResult]],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Process evaluation request."""
        ...
```

**Example Implementation:**

```python
class TimingMiddleware:
    """Middleware that tracks evaluation timing.

    IMPLEMENTATION CHECKLIST:
    - ✅ Stateless (no instance variables modified)
    - ✅ Calls next_handler in try block
    - ✅ Records metrics in finally block
    - ✅ Propagates exceptions
    - ✅ Thread-safe (no shared mutable state)
    """

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[..., Awaitable[EvaluationResult]],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Process evaluation with timing."""

        # ✅ 1. Pre-processing
        start_time = time.time()
        request_id = context.get("request_id", "unknown")

        try:
            # ✅ 2. Call next handler
            result = await next_handler(output, reference)

            # ✅ 3. Post-processing (success)
            elapsed = time.time() - start_time
            logger.info(
                f"Evaluation completed in {elapsed:.2f}s",
                extra={"request_id": request_id, "elapsed": elapsed}
            )

            return result

        except Exception as e:
            # ✅ 4. Post-processing (error)
            elapsed = time.time() - start_time
            logger.error(
                f"Evaluation failed after {elapsed:.2f}s: {e}",
                extra={"request_id": request_id, "elapsed": elapsed}
            )
            raise

        finally:
            # ✅ 5. Cleanup (always runs)
            # Record metric even if failed
            metrics.histogram("evaluation_duration_seconds", time.time() - start_time)
```

---

## Testing Requirements

### Test Coverage Requirements

**Minimum Coverage:** 80% (Current: 93%)

**Coverage by Component:**

```yaml
core:
  llm_client.py: ">=90%"
  middleware.py: ">=90%"
  registry.py: ">=95%"
  circuit_breaker.py: ">=85%"

evaluators:
  base.py: ">=95%"
  semantic.py: ">=85%"
  custom_criteria.py: ">=85%"
  factuality.py: ">=90%"
  groundedness.py: ">=90%"
  relevance.py: ">=90%"

api:
  api.py: ">=85%"
```

**Running Tests:**

```bash
# Run all tests with coverage
pytest tests/ --cov=arbiter --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_semantic.py -v

# Run tests with markers
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests

# Generate HTML coverage report
pytest tests/ --cov=arbiter --cov-report=html
open htmlcov/index.html
```

---

### Unit Test Requirements

**Location:** `tests/unit/`

**Naming Convention:** `test_<module>.py`

**Test Structure:**

```python
# tests/unit/test_my_evaluator.py

import pytest
from unittest.mock import AsyncMock, MagicMock

from arbiter.evaluators.my_evaluator import MyEvaluator, MyEvaluatorResponse
from tests.conftest import MockAgentResult  # Shared fixtures


@pytest.fixture
def evaluator(mock_llm_client):
    """Create MyEvaluator instance with mocked client."""
    return MyEvaluator(llm_client=mock_llm_client)


class TestMyEvaluator:
    """Test suite for MyEvaluator.

    REQUIRED TESTS:
    - ✅ test_name_property
    - ✅ test_system_prompt
    - ✅ test_user_prompt_with_reference
    - ✅ test_user_prompt_without_reference
    - ✅ test_response_type
    - ✅ test_compute_score
    - ✅ test_evaluate_success
    - ✅ test_evaluate_error_handling
    - ✅ test_interaction_tracking
    """

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "my_evaluator"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert len(prompt) < 2000
        assert "evaluator" in prompt.lower()

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Test output"
        reference = "Test reference"
        prompt = evaluator._get_user_prompt(output, reference, None)

        assert output in prompt
        assert reference in prompt
        assert "evaluate" in prompt.lower()

    def test_user_prompt_without_reference(self, evaluator):
        """Test user prompt generation without reference."""
        output = "Test output"
        prompt = evaluator._get_user_prompt(output, None, None)

        assert output in prompt
        # Should handle missing reference gracefully

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == MyEvaluatorResponse

    @pytest.mark.asyncio
    async def test_compute_score(self, evaluator):
        """Test score computation."""
        response = MyEvaluatorResponse(
            score=0.85,
            confidence=0.9,
            explanation="Good quality",
            strengths=["Clear", "Accurate"],
            weaknesses=["Too brief"]
        )

        score = await evaluator._compute_score(response)

        assert score.name == "my_evaluator"
        assert score.value == 0.85
        assert score.confidence == 0.9
        assert "Good quality" in score.explanation
        assert score.metadata["strength_count"] == 2
        assert score.metadata["weakness_count"] == 1

    @pytest.mark.asyncio
    async def test_evaluate_success(self, evaluator, mock_agent):
        """Test successful evaluation."""
        # Setup mock
        mock_response = MyEvaluatorResponse(
            score=0.9,
            confidence=0.88,
            explanation="Excellent",
            strengths=["A", "B"],
            weaknesses=[]
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Execute
        score = await evaluator.evaluate(
            output="Test output",
            reference="Test reference"
        )

        # Verify
        assert score.value == 0.9
        assert score.confidence == 0.88
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].purpose == "my_evaluator_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(output="Test", reference="Ref")

        assert "my_evaluator" in str(exc_info.value) or "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = MyEvaluatorResponse(
            score=0.9,
            confidence=0.9,
            explanation="Test",
            strengths=[],
            weaknesses=[]
        )
        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(output="Test", reference="Ref")

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"  # From mock
        assert interaction.purpose == "my_evaluator_evaluation"
        assert interaction.metadata["evaluator"] == "my_evaluator"
        assert interaction.metadata["has_reference"] is True
```

---

### Integration Test Requirements

**Location:** `tests/integration/`

**Purpose:** Test real LLM API calls (requires API keys)

**Markers:**

```python
# tests/integration/test_semantic_integration.py

import pytest
import os

# Skip if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_semantic_evaluator_real_api():
    """Test SemanticEvaluator with real OpenAI API.

    REQUIREMENTS:
    - ✅ Use real LLM API (not mocked)
    - ✅ Test realistic scenarios
    - ✅ Verify response structure
    - ✅ Check token usage tracking
    - ✅ Validate interaction logging
    """
    from arbiter import SemanticEvaluator, LLMManager

    # Real API call
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = SemanticEvaluator(client)

    score = await evaluator.evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris"
    )

    # Verify
    assert 0.0 <= score.value <= 1.0
    assert score.value > 0.8  # Should be semantically similar
    assert score.confidence > 0.0
    assert len(score.explanation) > 0

    # Verify interaction tracking
    assert len(evaluator.interactions) == 1
    interaction = evaluator.interactions[0]
    assert interaction.tokens_used > 0
    assert interaction.latency > 0
    assert interaction.model == "gpt-4o-mini"

    await client.close()
```

---

### Test Fixtures

**Location:** `tests/conftest.py`

**Shared Fixtures:**

```python
# tests/conftest.py

import pytest
from unittest.mock import MagicMock, AsyncMock
from arbiter.core.llm_client import LLMClient
from arbiter.core.types import Provider


class MockAgentResult:
    """Mock PydanticAI agent result."""

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data

    def usage(self):
        """Mock usage data."""
        class Usage:
            total_tokens = 100
            prompt_tokens = 60
            completion_tokens = 40
        return Usage()


@pytest.fixture
def mock_llm_client():
    """Create mocked LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.provider = Provider.OPENAI
    client.model_name = "gpt-4o-mini"
    return client


@pytest.fixture
def mock_agent():
    """Create mocked PydanticAI agent."""
    agent = MagicMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
async def real_llm_client():
    """Create real LLMClient (for integration tests)."""
    from arbiter import LLMManager
    client = await LLMManager.get_client(model="gpt-4o-mini")
    yield client
    await client.close()
```

---

## Documentation Standards

### Docstring Format

**Standard:** Google-style docstrings

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Short one-line summary.

    Longer description if needed. Explain what the function does,
    not how it does it. Focus on behavior and contracts.

    Args:
        param1: Description of param1
        param2: Description of param2. Defaults to 10.

    Returns:
        True if successful, False otherwise

    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is not an integer

    Example:
        >>> result = function_name("test", 20)
        >>> print(result)
        True

    Note:
        Additional notes about usage, performance, or caveats
    """
    ...
```

**Class Docstrings:**

```python
class MyClass:
    """Short one-line summary of class.

    Longer description of class purpose, responsibilities, and usage.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Example:
        >>> obj = MyClass(param="value")
        >>> result = obj.method()
    """

    def __init__(self, param: str):
        """Initialize MyClass.

        Args:
            param: Description of param
        """
        self.attr1 = param
```

---

### README Guidelines

**Required Sections:**
1. **Installation** - How to install
2. **Quick Start** - 3-line example
3. **Features** - What it does
4. **Documentation** - Links to docs
5. **Examples** - Links to example files
6. **Contributing** - How to contribute
7. **License** - MIT license

**Example Structure:**

```markdown
# Project Name

Brief description (1-2 sentences)

## Installation

\`\`\`bash
pip install arbiter
\`\`\`

## Quick Start

\`\`\`python
from arbiter import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    evaluators=["semantic"]
)
\`\`\`

## Features

- Feature 1
- Feature 2

## Documentation

- [Design Spec](DESIGN_SPEC.md)
- [Architecture](ARCHITECTURE.md)
- [Implementation Spec](IMPLEMENTATION_SPEC.md)

## Examples

- [Basic Evaluation](examples/basic_evaluation.py)
- [Custom Criteria](examples/custom_criteria_example.py)

## License

MIT
```

---

## Error Handling Patterns

### Input Validation

```python
def validate_input(output: str, reference: Optional[str]) -> None:
    """Validate evaluation inputs.

    VALIDATION CHECKLIST:
    - ✅ Output not empty
    - ✅ Output not too long (>50K chars)
    - ✅ Reference not too long if provided
    - ✅ Output is string type
    """
    if not output:
        raise ValidationError("Output cannot be empty")

    if not isinstance(output, str):
        raise ValidationError(f"Output must be string, got {type(output)}")

    if len(output) > 50_000:
        logger.warning(f"Large output truncated: {len(output)} -> 50000 chars")
        output = output[:50_000]

    if reference and len(reference) > 50_000:
        logger.warning(f"Large reference truncated: {len(reference)} -> 50000 chars")
        reference = reference[:50_000]
```

### Retry Logic

```python
async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> T:
    """Retry function with exponential backoff.

    IMPLEMENTATION CHECKLIST:
    - ✅ Retry only on retriable errors (timeout, 5xx, 429)
    - ✅ Don't retry on client errors (4xx except 429)
    - ✅ Respect Retry-After header for 429
    - ✅ Exponential backoff with jitter
    - ✅ Max delay cap
    """
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return await func()

        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Respect Retry-After
                retry_after = int(e.response.headers.get("Retry-After", delay))
                delay = min(retry_after, max_delay)
                logger.warning(f"Rate limited, retrying in {delay}s")

            elif 500 <= e.response.status_code < 600:
                # Server error, retry
                logger.warning(f"Server error {e.response.status_code}, retrying in {delay}s")

            else:
                # Client error, don't retry
                raise

            if attempt == max_retries - 1:
                raise

        # Exponential backoff with jitter
        await asyncio.sleep(delay + random.uniform(0, 0.1 * delay))
        delay = min(delay * backoff_factor, max_delay)

    raise Exception("Retry logic error: should not reach here")
```

---

## Performance Requirements

### Framework Overhead

**Target:** <50ms (p95)

**Measurement:**

```python
import time

start = time.time()
result = await evaluate(output="test", evaluators=["semantic"])
framework_overhead = time.time() - start - sum(i.latency for i in result.interactions)

# framework_overhead should be < 0.050 (50ms) at p95
```

### Memory Usage

**Target:** <500MB per 1000 evaluations

**Monitoring:**

```python
import psutil
import os

process = psutil.Process(os.getpid())

memory_before = process.memory_info().rss / 1024 / 1024  # MB

# Run 1000 evaluations
for i in range(1000):
    result = await evaluate(output=f"test {i}", evaluators=["semantic"])

memory_after = process.memory_info().rss / 1024 / 1024  # MB
memory_used = memory_after - memory_before

# memory_used should be < 500 MB
```

---

## Deployment Configuration

### Environment Variables

```bash
# Required (at least one provider)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...
MISTRAL_API_KEY=...
COHERE_API_KEY=...

# Optional configuration
ARBITER_LOG_LEVEL=INFO              # DEBUG, INFO, WARNING, ERROR
ARBITER_POOL_SIZE=50                # Connection pool size
ARBITER_TIMEOUT=30                  # Default timeout (seconds)
ARBITER_MAX_RETRIES=3               # Max retry attempts
ARBITER_CACHE_TTL=3600              # Cache TTL (seconds)
ARBITER_RATE_LIMIT=100              # Max requests per minute
```

### Configuration Files

```python
# config/production.py

from arbiter.core.retry import RetryConfig
from arbiter.core.middleware import CachingMiddleware, RateLimitingMiddleware

PRODUCTION_CONFIG = {
    "retry_config": RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0,
        backoff_factor=2.0
    ),
    "middleware": [
        RateLimitingMiddleware(max_requests=100, time_window=60),
        CachingMiddleware(max_size=1000, ttl_seconds=3600)
    ],
    "pool_config": {
        "max_connections": 50,
        "max_keepalive_connections": 10,
        "keepalive_expiry": 300
    }
}
```

---

## Development Workflow

### Git Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-evaluator

# 2. Make changes
# ... edit files ...

# 3. Run quality checks
black arbiter/ tests/
ruff check --fix arbiter/ tests/
mypy arbiter/
pytest tests/ --cov=arbiter

# 4. Commit changes
git add arbiter/evaluators/my_evaluator.py tests/unit/test_my_evaluator.py
git commit -m "Add MyEvaluator for [purpose]

- Implement template method pattern
- Add comprehensive tests (>80% coverage)
- Register in evaluator registry
- Add example file

Closes #123"

# 5. Push and create PR
git push origin feature/my-evaluator
gh pr create --title "Add MyEvaluator" --body "..."
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml

repos:
  - repo: https://github.com/psf/black
    rev: 25.0.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.0
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml

name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run formatters
        run: |
          black --check arbiter/ tests/
          ruff check arbiter/ tests/

      - name: Run type checker
        run: mypy arbiter/

      - name: Run tests
        run: pytest tests/ --cov=arbiter --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

---

## Related Documentation

- **DESIGN_SPEC.md** - Vision and features
- **ARCHITECTURE.md** - System architecture and interfaces
- **PROJECT_TODO.md** - Current development tasks
- **ROADMAP.md** - Development phases

---

**Last Updated:** 2025-11-16
**Next Review:** 2026-01-15 (Post v1.0 release)
