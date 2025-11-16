# Arbiter Architecture Specification

**Version:** 1.0
**Status:** Active
**Purpose:** Comprehensive architectural specification for production deployment
**Last Updated:** 2025-11-16

---

## Table of Contents

1. [System Context](#system-context)
2. [Component Architecture](#component-architecture)
3. [Interface Contracts](#interface-contracts)
4. [State Management](#state-management)
5. [Concurrency Model](#concurrency-model)
6. [Failure Modes & Recovery](#failure-modes--recovery)
7. [Data Flow](#data-flow)
8. [Integration Patterns](#integration-patterns)
9. [Security Architecture](#security-architecture)
10. [Deployment Architecture](#deployment-architecture)

---

## System Context

### External Dependencies

```yaml
external_systems:
  llm_providers:
    - name: OpenAI
      protocol: HTTPS/REST
      auth: Bearer token (API key)
      endpoints:
        - https://api.openai.com/v1/chat/completions
      rate_limits: Tier-based (varies by account)
      retry_policy: Respect Retry-After header

    - name: Anthropic
      protocol: HTTPS/REST
      auth: X-API-Key header
      endpoints:
        - https://api.anthropic.com/v1/messages
      rate_limits: Request-based limits
      retry_policy: Exponential backoff

    - name: Google (Gemini)
      protocol: HTTPS/REST
      auth: API key in query params
      endpoints:
        - https://generativelanguage.googleapis.com/v1beta/models
      rate_limits: QPM-based limits
      retry_policy: Standard backoff

    - name: Groq, Mistral, Cohere
      protocol: HTTPS/REST
      auth: Bearer token or API key
      rate_limits: Provider-specific
      retry_policy: Exponential backoff

  optional_dependencies:
    - name: FAISS (local)
      purpose: Fast similarity search
      install: pip install arbiter[scale]
      fallback: LLM-based similarity

    - name: Redis (future)
      purpose: Distributed caching
      status: Deferred to v2.0
```

### System Boundaries

```
┌─────────────────────────────────────────────────────────┐
│                     USER APPLICATION                     │
│  (AI pipelines, evaluation scripts, Loom workflows)     │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ import arbiter
                     │ await evaluate(...)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    ARBITER FRAMEWORK                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │              Public API (api.py)                     │ │
│ │  evaluate(), compare(), LLMManager.get_client()     │ │
│ └───────────────────┬─────────────────────────────────┘ │
│                     │                                    │
│ ┌───────────────────▼─────────────────────────────────┐ │
│ │         Middleware Pipeline                          │ │
│ │  Logging → Metrics → Caching → Rate Limiting        │ │
│ └───────────────────┬─────────────────────────────────┘ │
│                     │                                    │
│ ┌───────────────────▼─────────────────────────────────┐ │
│ │         Evaluators (Business Logic)                  │ │
│ │  Semantic, CustomCriteria, Factuality, etc.         │ │
│ └───────────────────┬─────────────────────────────────┘ │
│                     │                                    │
│ ┌───────────────────▼─────────────────────────────────┐ │
│ │    LLM Client + Connection Pool                     │ │
│ │  PydanticAI integration, retry logic, pooling       │ │
│ └───────────────────┬─────────────────────────────────┘ │
└─────────────────────┼─────────────────────────────────┘
                      │ HTTPS/REST
                      │ API keys, timeouts, retries
                      ▼
┌─────────────────────────────────────────────────────────┐
│              EXTERNAL LLM PROVIDERS                      │
│  OpenAI, Anthropic, Google, Groq, Mistral, Cohere      │
└─────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Public API Layer (`arbiter/api.py`)

**Responsibility:** Simple, stateless entry points for evaluation operations

#### `evaluate()` Function

```python
async def evaluate(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: List[Union[str, BaseEvaluator]] = None,
    model: str = "gpt-4o-mini",
    provider: Optional[str] = None,
    threshold: float = 0.7,
    middleware: Optional[List[Middleware]] = None,
    **kwargs
) -> EvaluationResult:
    """Evaluate LLM output against reference or criteria.

    State Management:
    - Creates temporary LLM client if not provided
    - Instantiates evaluators from registry
    - Cleans up client after evaluation

    Error Handling:
    - ValidationError: Invalid inputs (empty output, unknown evaluator)
    - ConfigurationError: Missing API keys, invalid provider
    - ModelProviderError: LLM API failures (after retries)
    - TimeoutError: LLM timeout (after 60s)

    Performance Characteristics:
    - Framework overhead: <50ms (p95)
    - Total time: Dominated by LLM API latency (1-5s typical)
    - Memory: ~5MB per evaluation + response size

    Concurrency:
    - Thread-safe for concurrent calls
    - Each call gets isolated client or shared from pool
    - No shared mutable state between calls
    """
```

**Interface Contract:**

```python
class EvaluateRequest(TypedDict, total=False):
    """Typed request for evaluate() function."""
    output: str                                    # Required
    reference: Optional[str]                       # Optional
    criteria: Optional[str]                        # Optional
    evaluators: List[Union[str, BaseEvaluator]]   # Optional, default ["semantic"]
    model: str                                     # Optional, default "gpt-4o-mini"
    provider: Optional[str]                        # Optional, auto-detect from model
    threshold: float                               # Optional, default 0.7
    middleware: Optional[List[Middleware]]         # Optional, default None

class EvaluateResponse(BaseModel):
    """Typed response from evaluate() function."""
    # Inputs (echoed back)
    output: str
    reference: Optional[str]
    criteria: Optional[str]

    # Results
    scores: List[Score]                            # One per evaluator
    overall_score: float                           # Average of all scores
    passed: bool                                   # overall_score >= threshold

    # Metadata
    metrics: List[Metric]                          # Performance metrics
    evaluator_names: List[str]                     # Names of evaluators used
    total_tokens: int                              # Sum of all LLM tokens
    processing_time: float                         # Total wall-clock time (seconds)
    interactions: List[LLMInteraction]             # Complete audit trail

    # Quality indicators
    model_config = ConfigDict(frozen=True)         # Immutable after creation
```

#### `compare()` Function

```python
async def compare(
    output_a: str,
    output_b: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    model: str = "gpt-4o-mini",
    aspects: Optional[List[str]] = None,
    **kwargs
) -> ComparisonResult:
    """Compare two LLM outputs for A/B testing.

    State Management:
    - Uses PairwiseComparisonEvaluator internally
    - Creates temporary client, cleans up after

    Error Handling:
    - Same as evaluate()
    - Additional: ValidationError if output_a == output_b

    Performance:
    - Single LLM call for comparison
    - Faster than running evaluate() twice
    """
```

---

### 2. Middleware Pipeline (`arbiter/core/middleware.py`)

**Responsibility:** Cross-cutting concerns via chain of responsibility pattern

#### Middleware Interface

```python
class Middleware(Protocol):
    """Protocol for middleware components.

    Execution Model:
    1. Pre-processing: Middleware receives request
    2. Delegation: Calls next_handler with (potentially modified) request
    3. Post-processing: Middleware receives result from downstream
    4. Return: Returns (potentially modified) result

    State Management:
    - Middleware should be stateless OR thread-safe
    - Context dict allows passing state between middleware
    - Cleanup in finally blocks if resources acquired

    Error Handling:
    - Middleware can catch/transform exceptions
    - Middleware can add context to exceptions
    - Middleware must call next_handler in try block
    """

    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[..., Awaitable[EvaluationResult]],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Process evaluation with next middleware in chain.

        Args:
            output: LLM output to evaluate
            reference: Optional reference text
            next_handler: Next middleware or final evaluator
            context: Shared context dict for passing state

        Returns:
            EvaluationResult from downstream, potentially modified

        Raises:
            Any exception from downstream (can be caught/transformed)
        """
        ...
```

#### Built-in Middleware

**LoggingMiddleware:**
```python
class LoggingMiddleware:
    """Logs evaluation requests and responses.

    State: Stateless (uses module-level logger)

    Logged Information:
    - Pre: request_id, evaluator, model, output_length, reference_length
    - Post: request_id, status, score, tokens, duration
    - Error: request_id, error_type, error_message

    Configuration:
    - log_level: INFO (default), DEBUG, WARNING
    - include_output: bool (default False, PII risk)
    """
```

**MetricsMiddleware:**
```python
class MetricsMiddleware:
    """Tracks evaluation metrics via PerformanceMonitor.

    State: References global PerformanceMonitor (thread-safe)

    Metrics Tracked:
    - evaluations_total: counter with labels {evaluator, model, status}
    - evaluation_duration_seconds: histogram
    - evaluation_score: histogram
    - llm_tokens_total: counter with labels {provider, model}

    Configuration:
    - monitor: PerformanceMonitor instance (default: global)
    """
```

**CachingMiddleware:**
```python
class CachingMiddleware:
    """LRU cache for evaluation results.

    State: Internal LRU cache (thread-safe via locks)

    Cache Key: hash(output + reference + criteria + evaluator + model)
    Cache Value: EvaluationResult (frozen/immutable)
    TTL: Configurable (default 3600s)

    Eviction: LRU when max_size reached

    Configuration:
    - max_size: int (default 1000)
    - ttl_seconds: int (default 3600)

    Thread Safety: Uses threading.RLock for cache access
    """
```

**RateLimitingMiddleware:**
```python
class RateLimitingMiddleware:
    """Token bucket rate limiting.

    State: Token bucket with refill rate (thread-safe)

    Algorithm:
    - Bucket capacity: max_requests
    - Refill rate: max_requests / time_window
    - On request: Try to consume 1 token
    - If bucket empty: Wait or raise RateLimitError

    Configuration:
    - max_requests: int (default 100)
    - time_window: int seconds (default 60)
    - wait_on_limit: bool (default True)

    Thread Safety: Uses asyncio.Lock for bucket access
    """
```

#### Middleware Pipeline Execution

```python
class MiddlewarePipeline:
    """Chains middleware in specified order.

    Execution Order:
    1. LoggingMiddleware (first - logs everything)
    2. MetricsMiddleware (early - tracks everything)
    3. RateLimitingMiddleware (before expensive operations)
    4. CachingMiddleware (check cache before LLM calls)
    5. [Final Handler: Evaluator]

    Reverse Order on Return:
    - CachingMiddleware (store result)
    - RateLimitingMiddleware (no-op on return)
    - MetricsMiddleware (record final metrics)
    - LoggingMiddleware (log completion)

    Error Propagation:
    - Exceptions bubble up through middleware chain
    - Each middleware can catch/transform/rethrow
    - Finally blocks ensure cleanup
    """

    def __init__(self, middleware: List[Middleware]):
        self.middleware = middleware

    async def execute(
        self,
        output: str,
        reference: Optional[str],
        final_handler: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """Execute middleware chain.

        Implementation:
        - Build nested async function chain
        - Each middleware wraps the next
        - Final handler is innermost function
        """
```

---

### 3. Evaluator Layer (`arbiter/evaluators/`)

**Responsibility:** Domain-specific evaluation logic via template method pattern

#### BaseEvaluator Protocol

```python
class BaseEvaluator(Protocol):
    """Protocol that all evaluators must implement.

    Lifecycle:
    1. __init__(llm_client: LLMClient) - Store client reference
    2. evaluate(output, reference, criteria) - Public method
    3. _get_system_prompt() - Template method (subclass)
    4. _get_user_prompt() - Template method (subclass)
    5. _get_response_type() - Template method (subclass)
    6. [LLM API call via PydanticAI]
    7. _compute_score() - Template method (subclass)
    8. Return Score object

    State Management:
    - llm_client: Immutable reference (set in __init__)
    - interactions: List[LLMInteraction] (mutable, appended to)
    - No other mutable state

    Thread Safety:
    - Safe for concurrent evaluate() calls
    - Each call has isolated state (local variables)
    - interactions list requires external synchronization if shared
    """

    @property
    def name(self) -> str:
        """Unique evaluator name (e.g., 'semantic', 'factuality')."""
        ...

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None
    ) -> Score:
        """Evaluate output. Thread-safe."""
        ...

    def get_interactions(self) -> List[LLMInteraction]:
        """Get LLM interaction history. Returns copy."""
        ...

    def clear_interactions(self) -> None:
        """Clear interaction history. NOT thread-safe."""
        ...
```

#### BasePydanticEvaluator (Abstract Base)

```python
class BasePydanticEvaluator(ABC):
    """Template method pattern implementation.

    Enforces:
    - All evaluators use PydanticAI for structured outputs
    - Consistent interaction tracking
    - Standard error handling
    - Score computation contract

    Subclass Requirements:
    1. Implement 4 abstract methods
    2. Return Pydantic model from _get_response_type()
    3. Response model MUST have 'score' field (0.0-1.0)
    4. Response model SHOULD have 'confidence' and 'explanation'

    Constraints:
    - System prompt: <2000 chars (token limits)
    - User prompt: Must include output text
    - Response type: Must be Pydantic BaseModel
    - Score value: Must be 0.0-1.0 inclusive

    Error Handling:
    - ValidationError: Invalid inputs → raise immediately, no retry
    - Timeout: LLM timeout → raise TimeoutError after 60s
    - ModelProviderError: LLM API error → retry with backoff (max 3)
    - All exceptions include request context (model, evaluator, output_length)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.interactions: List[LLMInteraction] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name. Must be lowercase, alphanumeric + underscores."""

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return system prompt for LLM.

        Guidelines:
        - Define expert role clearly
        - Specify output format expectations
        - Emphasize objectivity and consistency
        - Keep under 2000 chars
        """

    @abstractmethod
    def _get_user_prompt(
        self,
        output: str,
        reference: Optional[str],
        criteria: Optional[str]
    ) -> str:
        """Return user prompt with evaluation task.

        Guidelines:
        - Include all relevant inputs
        - Provide clear evaluation instructions
        - Specify what to analyze
        - Handle missing reference/criteria gracefully
        """

    @abstractmethod
    def _get_response_type(self) -> Type[BaseModel]:
        """Return Pydantic model for structured response.

        Required Fields:
        - score: float = Field(ge=0.0, le=1.0)  # REQUIRED

        Recommended Fields:
        - confidence: float = Field(default=0.85, ge=0.0, le=1.0)
        - explanation: str

        Domain-Specific Fields:
        - Add as needed (e.g., factual_claims, missing_points)
        """

    @abstractmethod
    async def _compute_score(self, response: BaseModel) -> Score:
        """Transform LLM response to Score object.

        Responsibilities:
        - Extract score value (0.0-1.0)
        - Extract confidence if available
        - Build explanation from response
        - Add metadata dict with domain-specific fields

        Returns:
            Score(
                name=self.name,
                value=response.score,
                confidence=response.confidence,
                explanation=response.explanation,
                metadata={...}
            )
        """
```

#### Evaluator Registry

```python
# Global registry: name -> class
AVAILABLE_EVALUATORS: Dict[str, Type[BaseEvaluator]] = {}

def register_evaluator(name: str, evaluator_class: Type[BaseEvaluator]) -> None:
    """Register custom evaluator.

    Validation:
    - Name must be unique (not already registered)
    - Class must inherit from BaseEvaluator
    - Name must match evaluator.name property

    Thread Safety: NOT thread-safe (registration at module load only)
    """

def get_evaluator_class(name: str) -> Optional[Type[BaseEvaluator]]:
    """Lookup evaluator class by name.

    Thread Safety: Thread-safe (read-only access)
    """

def validate_evaluator_name(name: str) -> None:
    """Validate evaluator exists.

    Raises:
        ValidationError: If name not registered
        - Error message includes available evaluators
        - Suggests using register_evaluator() for custom
    """
```

---

### 4. LLM Client Layer (`arbiter/core/llm_client.py`)

**Responsibility:** Provider-agnostic LLM access with connection pooling and retry

#### LLMClient Interface

```python
class LLMClient:
    """PydanticAI-based LLM client with connection pooling.

    Lifecycle:
    1. Async context manager: async with LLMManager.get_client(...) as client
    2. Or manual: client = await LLMManager.get_client(...); await client.close()

    State Management:
    - model: Model instance (PydanticAI)
    - provider: Provider enum
    - http_client: HTTPX AsyncClient (connection pool)
    - circuit_breaker: CircuitBreaker instance (shared per provider)

    Connection Pool:
    - Default limits: 50 connections, 10 per host
    - Configurable via LLMClientPool
    - Shared across LLMClient instances for same provider

    Thread Safety:
    - Safe for concurrent create_agent() calls
    - HTTPX client handles concurrent requests
    - Circuit breaker uses atomic operations
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        http_client: httpx.AsyncClient,
        retry_config: RetryConfig = RetryConfig.STANDARD,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.provider = provider
        self.model_name = model
        self.http_client = http_client
        self.retry_config = retry_config
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # Create PydanticAI model
        self.model = Model(provider=provider, model_name=model)

    async def create_agent(
        self,
        response_type: Type[BaseModel],
        system_prompt: str,
        retries: Optional[int] = None
    ) -> Agent:
        """Create PydanticAI agent for structured output.

        Circuit Breaker Integration:
        - Checks circuit state before creating agent
        - If OPEN: Raises CircuitBreakerOpen immediately
        - If HALF_OPEN: Allows request, monitors result
        - If CLOSED: Normal operation

        Retry Logic:
        - Uses RetryConfig for backoff strategy
        - Retries on: Timeout, 429 Rate Limit, 503 Service Unavailable
        - Does NOT retry: 400 Bad Request, 401 Unauthorized, 404 Not Found
        - Records failures in circuit breaker

        Returns:
            Agent configured for structured output
        """

    async def close(self) -> None:
        """Close HTTP client connections.

        Cleanup:
        - Closes HTTPX client (drains connection pool)
        - Does NOT close circuit breaker (shared across clients)
        """
```

#### LLMClientPool (Connection Pool Manager)

```python
class LLMClientPool:
    """Manages connection pools per provider.

    Design:
    - One connection pool per provider
    - Pools are shared across all LLMClient instances
    - Lazy initialization (pool created on first use)
    - Automatic health checks and cleanup

    State:
    - _pools: Dict[Provider, httpx.AsyncClient] (class-level, shared)
    - _locks: Dict[Provider, asyncio.Lock] (for pool initialization)
    - _health_check_tasks: Dict[Provider, asyncio.Task] (background tasks)

    Configuration:
    - max_connections: 50 (total pool size)
    - max_keepalive_connections: 10 (idle connections)
    - keepalive_expiry: 300s (close idle after 5 min)
    - timeout: 30s (connect, read, write)

    Thread Safety:
    - Locks ensure one pool per provider
    - HTTPX client is internally thread-safe
    """

    _pools: Dict[Provider, httpx.AsyncClient] = {}
    _locks: Dict[Provider, asyncio.Lock] = {}

    @classmethod
    async def get_pool(cls, provider: Provider) -> httpx.AsyncClient:
        """Get or create connection pool for provider.

        Algorithm:
        1. Check if pool exists
        2. If not, acquire lock for provider
        3. Double-check pool (another coroutine may have created)
        4. Create pool if still missing
        5. Start health check task
        6. Return pool

        Returns:
            Shared HTTPX AsyncClient for provider
        """

    @classmethod
    async def close_all(cls) -> None:
        """Close all connection pools.

        Use Cases:
        - Application shutdown
        - Test cleanup
        - Pool reconfiguration

        Algorithm:
        1. Cancel all health check tasks
        2. Close all HTTPX clients (drains connections)
        3. Clear pools dict
        """
```

#### Circuit Breaker

```python
class CircuitBreaker:
    """Prevents cascade failures via circuit breaker pattern.

    States:
    - CLOSED: Normal operation, all requests allowed
    - OPEN: Failure threshold exceeded, reject all requests
    - HALF_OPEN: Testing recovery, allow limited requests

    State Transitions:
    - CLOSED → OPEN: 5 consecutive failures within 30s
    - OPEN → HALF_OPEN: After 60s timeout
    - HALF_OPEN → CLOSED: 1 successful request
    - HALF_OPEN → OPEN: 1 failed request (timeout doubles to 120s)

    Thresholds:
    - failure_threshold: 5 consecutive failures
    - failure_window: 30 seconds
    - recovery_timeout: 60 seconds (doubles on retry failure)
    - half_open_max_requests: 1

    Thread Safety:
    - Uses asyncio.Lock for state transitions
    - Atomic operations for counters
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        failure_window: int = 30,
        recovery_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failures: List[float] = []  # Timestamps of recent failures
        self.last_failure_time: Optional[float] = None
        self.lock = asyncio.Lock()

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection.

        Algorithm:
        1. Check current state
        2. If OPEN: Raise CircuitBreakerOpen
        3. If HALF_OPEN: Limit concurrent requests to 1
        4. Execute function
        5. On success: Record success, transition HALF_OPEN → CLOSED
        6. On failure: Record failure, check thresholds, maybe transition

        Raises:
            CircuitBreakerOpen: Circuit is open, rejecting requests
            Original exception: From wrapped function (if circuit allows)
        """
```

---

## Interface Contracts

### API Contracts (OpenAPI Schema)

```yaml
openapi: 3.0.0
info:
  title: Arbiter Evaluation API
  version: 1.0.0
  description: Production-grade LLM evaluation infrastructure

paths:
  /evaluate:
    post:
      summary: Evaluate LLM output
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [output]
              properties:
                output:
                  type: string
                  description: LLM output to evaluate
                  minLength: 1
                  maxLength: 50000
                reference:
                  type: string
                  description: Reference text for comparison
                  nullable: true
                criteria:
                  type: string
                  description: Evaluation criteria
                  nullable: true
                evaluators:
                  type: array
                  description: Evaluator names to use
                  items:
                    type: string
                    enum: [semantic, custom_criteria, factuality, groundedness, relevance, pairwise]
                  default: [semantic]
                model:
                  type: string
                  description: LLM model to use
                  default: gpt-4o-mini
                threshold:
                  type: number
                  description: Pass/fail threshold
                  minimum: 0.0
                  maximum: 1.0
                  default: 0.7

      responses:
        200:
          description: Evaluation successful
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EvaluationResult'

        400:
          description: Invalid request
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

        429:
          description: Rate limit exceeded
          headers:
            Retry-After:
              schema:
                type: integer
                description: Seconds to wait before retry
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

        500:
          description: Internal server error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

        503:
          description: Service unavailable (circuit breaker open)
          headers:
            Retry-After:
              schema:
                type: integer
                description: Seconds until circuit breaker retry
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'

components:
  schemas:
    EvaluationResult:
      type: object
      required: [output, scores, overall_score, passed, total_tokens, processing_time]
      properties:
        output:
          type: string
        reference:
          type: string
          nullable: true
        criteria:
          type: string
          nullable: true
        scores:
          type: array
          items:
            $ref: '#/components/schemas/Score'
        overall_score:
          type: number
          minimum: 0.0
          maximum: 1.0
        passed:
          type: boolean
        total_tokens:
          type: integer
          minimum: 0
        processing_time:
          type: number
          description: Seconds
        interactions:
          type: array
          items:
            $ref: '#/components/schemas/LLMInteraction'

    Score:
      type: object
      required: [name, value, confidence, explanation]
      properties:
        name:
          type: string
          description: Evaluator name
        value:
          type: number
          minimum: 0.0
          maximum: 1.0
        confidence:
          type: number
          minimum: 0.0
          maximum: 1.0
        explanation:
          type: string
        metadata:
          type: object
          additionalProperties: true

    LLMInteraction:
      type: object
      required: [model, tokens_used, latency, purpose]
      properties:
        prompt:
          type: string
        response:
          type: string
        model:
          type: string
        tokens_used:
          type: integer
        latency:
          type: number
        timestamp:
          type: string
          format: date-time
        purpose:
          type: string
        metadata:
          type: object
          additionalProperties: true

    Error:
      type: object
      required: [error_type, message]
      properties:
        error_type:
          type: string
          enum: [ValidationError, ConfigurationError, ModelProviderError, TimeoutError, CircuitBreakerOpen]
        message:
          type: string
        details:
          type: object
          additionalProperties: true
```

---

## State Management

### Component State Lifecycle

```yaml
LLMClient:
  creation:
    method: LLMManager.get_client(provider, model)
    state: model, http_client, circuit_breaker initialized

  usage:
    method: create_agent(response_type, system_prompt)
    state: No new state created (stateless operation)

  cleanup:
    method: await client.close()
    state: http_client closed, connections drained

  shared_state:
    - Circuit breaker (shared per provider)
    - Connection pool (shared per provider)

Evaluator:
  creation:
    method: evaluator_class(llm_client)
    state: llm_client reference stored, interactions list initialized

  usage:
    method: await evaluator.evaluate(output, reference)
    state: interactions list appended with new LLMInteraction

  cleanup:
    method: evaluator.clear_interactions()
    state: interactions list cleared (manual operation)

  shared_state:
    - None (each evaluator instance is independent)

Middleware:
  creation:
    method: middleware_class(**config)
    state: Configuration stored (e.g., cache, rate limiter)

  usage:
    method: await middleware.process(output, reference, next_handler, context)
    state: Middleware may update internal state (cache, buckets)

  cleanup:
    method: No explicit cleanup (relies on garbage collection)

  shared_state:
    - CachingMiddleware: LRU cache (thread-safe)
    - RateLimitingMiddleware: Token bucket (thread-safe)
    - MetricsMiddleware: PerformanceMonitor (global singleton)

ConnectionPool:
  creation:
    method: LLMClientPool.get_pool(provider)
    state: HTTPX AsyncClient created with configured limits

  usage:
    method: Transparent (used by LLMClient for requests)
    state: Connections acquired/released automatically

  cleanup:
    method: await LLMClientPool.close_all()
    state: All pools closed, connections drained

  shared_state:
    - Pool per provider (class-level dictionary)
```

### Concurrency Guarantees

```python
# Thread-safe operations:
- evaluate() function: Multiple concurrent calls safe
- LLMClient.create_agent(): Safe for concurrent calls
- Middleware.process(): Safe if middleware is stateless or uses locks
- ConnectionPool.get_pool(): Safe (uses asyncio.Lock)
- CircuitBreaker.call(): Safe (uses asyncio.Lock for state transitions)

# NOT thread-safe (require external synchronization):
- Evaluator.clear_interactions(): Modifies mutable list
- Registry operations: register_evaluator() at runtime (module load is safe)

# Async-safe patterns:
- Use asyncio.gather() for concurrent evaluations
- Use asyncio.create_task() for background operations
- Use asyncio.Lock for shared mutable state
- Use threading.RLock for synchronous shared state (CachingMiddleware)
```

---

## Concurrency Model

### Async/Await Architecture

**Design Philosophy:**
- Async-first design for I/O-bound LLM operations
- Non-blocking HTTP requests via HTTPX
- Connection pooling for efficient resource usage
- No CPU-bound operations in hot paths

**Execution Model:**

```python
# Single evaluation (sequential)
result = await evaluate(output="...", evaluators=["semantic"])
# 1. Create/get LLM client (async, uses pool)
# 2. Create middleware pipeline (sync)
# 3. Execute middleware chain (async)
# 4. Call LLM API (async, awaits response)
# 5. Compute score (sync, CPU-bound but fast)
# 6. Return result

# Concurrent evaluations (parallel I/O)
results = await asyncio.gather(
    evaluate(output="output1", evaluators=["semantic"]),
    evaluate(output="output2", evaluators=["factuality"]),
    evaluate(output="output3", evaluators=["groundedness"])
)
# - All 3 LLM API calls happen concurrently
# - Connection pool manages HTTP connections
# - Each evaluation has isolated state
# - Results collected when all complete

# Concurrent evaluations with same evaluator (safe)
semantic_evaluator = SemanticEvaluator(client)
scores = await asyncio.gather(
    semantic_evaluator.evaluate(output="output1", reference="ref1"),
    semantic_evaluator.evaluate(output="output2", reference="ref2")
)
# - Safe: evaluate() method has no shared mutable state
# - Caution: interactions list is shared (appended to)
# - Recommendation: Create separate evaluator instances for true isolation
```

### Connection Pool Management

```python
class LLMClientPool:
    """Connection pool sizing and tuning.

    Configuration:
    - max_connections: 50
      Rationale: Support ~50 concurrent evaluations
      Tuning: Increase for higher concurrency (100+)

    - max_keepalive_connections: 10
      Rationale: Reuse connections for repeated requests
      Tuning: Increase for sustained high throughput

    - keepalive_expiry: 300 seconds
      Rationale: Balance connection reuse vs provider limits
      Tuning: Decrease to 60s if hitting provider connection limits

    - timeout: 30 seconds
      Rationale: LLM APIs typically respond in 1-10s
      Tuning: Increase to 60s for complex evaluations

    Resource Limits:
    - Each connection: ~50KB memory (TCP buffers, SSL state)
    - 50 connections: ~2.5MB total (negligible)
    - Bottleneck: Provider rate limits, not connection pool
    """
```

---

## Failure Modes & Recovery

### Comprehensive Failure Mode Catalog

#### FM-001: LLM API Timeout

**Scenario:** LLM provider doesn't respond within timeout (default 30s)

**Detection:**
```python
try:
    result = await agent.run(user_prompt)
except asyncio.TimeoutError:
    # HTTP timeout from HTTPX
```

**Immediate Actions:**
1. Cancel HTTP request and close connection
2. Log error with request_id, model, provider, elapsed_time
3. Emit metric: `llm_timeout_total{provider="openai", model="gpt-4o-mini"}`
4. Record failure in circuit breaker

**Recovery Procedure:**
```yaml
retry_strategy:
  max_retries: 3
  backoff: exponential
  delays: [1s, 2s, 4s]
  total_timeout: 90s (30s * 3 + overhead)

recovery_steps:
  attempt_1:
    delay: 1s
    action: Retry with same timeout (30s)
  attempt_2:
    delay: 2s
    action: Retry with increased timeout (45s)
  attempt_3:
    delay: 4s
    action: Retry with maximum timeout (60s)
  final_failure:
    action: Raise TimeoutError to caller
    message: "LLM request timed out after 3 retries (90s total)"
```

**User Impact:**
- Evaluation fails after ~90 seconds total
- User receives TimeoutError exception
- Error message includes total elapsed time and retry count

**Monitoring:**
```yaml
alerts:
  - name: HighTimeoutRate
    condition: llm_timeout_total rate > 5% over 5 minutes
    severity: warning
    action: Check provider status, consider timeout increase
```

---

#### FM-002: LLM Rate Limit (429)

**Scenario:** Provider returns HTTP 429 Rate Limit Exceeded

**Detection:**
```python
try:
    result = await agent.run(user_prompt)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:
        # Rate limit hit
```

**Immediate Actions:**
1. Extract `Retry-After` header (default 60s if missing)
2. Log warning with retry time, provider, model
3. Emit metric: `llm_rate_limited_total{provider="openai"}`
4. Record non-failure in circuit breaker (rate limit ≠ failure)

**Recovery Procedure:**
```yaml
retry_strategy:
  respect_retry_after: true
  max_wait: 300s (5 minutes)

recovery_steps:
  extract_retry_after:
    sources: [Retry-After header, X-RateLimit-Reset header]
    fallback: 60s

  wait:
    duration: min(retry_after, max_wait)
    backpressure: Warn caller if retry_after > 60s

  retry:
    action: Retry request after wait
    counts_toward: retry limit (max 3 total retries)

  final_failure:
    condition: All 3 retries exhausted with 429
    action: Raise RateLimitError to caller
    message: "Rate limit exceeded after 3 retries, consider reducing request rate"
```

**User Impact:**
- Evaluation delayed by Retry-After duration (typically 60s)
- Transparent retry (user doesn't see error unless all retries fail)
- Total delay: 60s + 120s + 240s = 420s worst case (7 minutes)

**Monitoring:**
```yaml
alerts:
  - name: HighRateLimitRate
    condition: llm_rate_limited_total rate > 1% over 5 minutes
    severity: warning
    action: Review rate limiting configuration, consider tier upgrade

  - name: RateLimitBackpressure
    condition: retry_after > 120s
    severity: critical
    action: Immediate review of request patterns, possible abuse
```

**Degradation Strategy:**
```python
# Optional: Implement client-side rate limiting
rate_limiter = RateLimitingMiddleware(
    max_requests=50,  # Per minute
    time_window=60,
    wait_on_limit=True  # Block rather than fail
)
```

---

#### FM-003: Connection Pool Exhausted

**Scenario:** All connections in pool are in use, new request waits >10s

**Detection:**
```python
# HTTPX AsyncClient internal timeout
try:
    response = await http_client.get(url)
except httpx.PoolTimeout:
    # Connection pool exhausted
```

**Immediate Actions:**
1. Log warning with pool utilization metrics:
   - Total connections: 50
   - Active connections: 50
   - Idle connections: 0
   - Waiting requests: N
2. Emit metric: `pool_exhausted_total{provider="openai"}`
3. Wait up to 10s for connection release

**Recovery Procedure:**
```yaml
immediate_recovery:
  wait_for_connection:
    timeout: 10s
    action: Block until connection available

  timeout_handling:
    condition: No connection after 10s
    action: Raise ConnectionError
    message: "Connection pool exhausted, consider increasing pool size"

long_term_recovery:
  pool_resizing:
    trigger: pool_exhausted_total > 10 per hour
    action: Increase max_connections from 50 to 100
    validation: Monitor memory usage (<100MB total)

  request_throttling:
    trigger: Sustained pool exhaustion
    action: Enable RateLimitingMiddleware
    config: max_requests=50/min (align with pool size)
```

**User Impact:**
- Evaluation delayed up to 10 seconds waiting for connection
- If timeout, evaluation fails with ConnectionError
- Recommendation: Implement request queuing or backpressure

**Monitoring:**
```yaml
metrics:
  - name: connection_pool_utilization
    type: gauge
    labels: [provider, state]
    description: Current connections (active/idle/total)

  - name: connection_pool_wait_time
    type: histogram
    buckets: [0.1, 0.5, 1.0, 5.0, 10.0]
    description: Time waiting for connection

alerts:
  - name: PoolUtilizationHigh
    condition: connection_pool_utilization{state="active"} / total > 0.9 for 2 minutes
    severity: warning
    action: Consider pool size increase or request throttling
```

---

#### FM-004: Circuit Breaker Open

**Scenario:** 5 consecutive failures within 30 seconds triggers circuit breaker

**Detection:**
```python
try:
    result = await circuit_breaker.call(lambda: agent.run(prompt))
except CircuitBreakerOpen:
    # Circuit is open, rejecting requests
```

**Immediate Actions:**
1. Open circuit for 60 seconds
2. Reject all new requests with `CircuitBreakerOpen` exception
3. Emit metric: `circuit_breaker_opened{provider="openai", model="gpt-4o-mini"}`
4. Emit alert: Circuit breaker opened (critical severity)

**State Transitions:**
```yaml
states:
  CLOSED:
    description: Normal operation
    request_handling: All requests allowed
    failure_tracking: Record last 5 failures with timestamps
    transition_condition: 5 failures within 30 seconds
    next_state: OPEN

  OPEN:
    description: Failing fast to prevent cascade
    request_handling: Reject all requests immediately
    error: CircuitBreakerOpen("Provider ${provider} unhealthy, retry after ${timeout}s")
    timeout: 60 seconds (first open), 120s (subsequent)
    transition_condition: Timeout expires
    next_state: HALF_OPEN

  HALF_OPEN:
    description: Testing recovery
    request_handling: Allow 1 request to test
    success_action: Transition to CLOSED
    failure_action: Transition to OPEN (double timeout)
```

**Recovery Procedure:**
```yaml
automatic_recovery:
  step_1:
    state: OPEN
    duration: 60s
    action: Reject all requests, wait for recovery timeout

  step_2:
    state: HALF_OPEN
    action: Allow 1 test request
    timeout: 30s

  step_3_success:
    condition: Test request succeeds
    action: Close circuit, resume normal operation
    notification: Emit metric circuit_breaker_closed

  step_3_failure:
    condition: Test request fails
    action: Reopen circuit
    new_timeout: 120s (double previous)
    max_timeout: 600s (10 minutes)

manual_intervention:
  triggers:
    - Circuit remains open > 5 minutes
    - Multiple circuit open events in 1 hour
  actions:
    - Check provider status page
    - Verify API key validity
    - Review recent changes (code, config)
    - Consider provider failover (if available)
```

**User Impact:**
- All evaluations fail fast for 60+ seconds
- Users receive `CircuitBreakerOpen` exception with retry time
- Prevents cascade failures and wasted resources
- Allows system to recover without overwhelming failing provider

**Monitoring:**
```yaml
metrics:
  - name: circuit_breaker_state
    type: gauge
    labels: [provider, model]
    values: {CLOSED: 0, OPEN: 1, HALF_OPEN: 2}

  - name: circuit_breaker_transitions
    type: counter
    labels: [provider, from_state, to_state]

alerts:
  - name: CircuitBreakerOpened
    condition: circuit_breaker_state == 1
    severity: critical
    action: Immediate investigation of provider health

  - name: FrequentCircuitBreaker
    condition: circuit_breaker_transitions{to_state="OPEN"} > 3 per hour
    severity: critical
    action: Systemic issue investigation, consider degradation
```

---

#### FM-005: Invalid API Key

**Scenario:** Provider returns HTTP 401 Unauthorized

**Detection:**
```python
try:
    result = await agent.run(user_prompt)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        # Invalid or missing API key
```

**Immediate Actions:**
1. Log error with provider and attempted key (last 4 chars only)
2. Emit metric: `llm_auth_failed_total{provider="openai"}`
3. Do NOT retry (auth errors are not transient)
4. Raise ConfigurationError immediately

**Recovery Procedure:**
```yaml
no_automatic_recovery:
  reason: Authentication failures require manual intervention

error_message:
  template: |
    Invalid API key for provider '{provider}'.

    Troubleshooting:
    1. Check environment variable '{env_var}' is set
    2. Verify API key is valid (last 4 chars: {last4})
    3. Check API key has required permissions
    4. Verify API key hasn't expired

    Provider documentation: {docs_url}

manual_recovery:
  step_1: Verify API key in provider dashboard
  step_2: Update environment variable
  step_3: Restart application to reload env vars
  step_4: Test with simple evaluation
```

**User Impact:**
- Evaluation fails immediately (no retry)
- Clear error message with troubleshooting steps
- No wasted retry attempts or delays

**Monitoring:**
```yaml
alerts:
  - name: AuthenticationFailure
    condition: llm_auth_failed_total > 0
    severity: critical
    action: Immediate config verification, possible security incident
```

---

#### FM-006: Model Not Found (404)

**Scenario:** Provider returns HTTP 404 for unknown model

**Detection:**
```python
try:
    result = await agent.run(user_prompt)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 404:
        # Model doesn't exist or access denied
```

**Immediate Actions:**
1. Log error with provider, attempted model name
2. Emit metric: `llm_model_not_found_total{provider="openai", model="gpt-5"}`
3. Do NOT retry (404 is not transient)
4. Raise ConfigurationError with supported models list

**Error Message:**
```python
raise ConfigurationError(
    f"Model '{model}' not found for provider '{provider}'.\n"
    f"\n"
    f"Supported models for {provider}:\n"
    f"  - gpt-4o\n"
    f"  - gpt-4o-mini\n"
    f"  - gpt-4-turbo\n"
    f"\n"
    f"Check provider documentation for current model availability."
)
```

---

#### FM-007: Malformed Response

**Scenario:** LLM returns invalid JSON or missing required fields

**Detection:**
```python
try:
    response = ResponseModel(**llm_response)
except ValidationError as e:
    # Pydantic validation failed
```

**Immediate Actions:**
1. Log error with:
   - Raw LLM response (truncated to 500 chars)
   - Expected schema
   - Validation errors
2. Emit metric: `llm_malformed_response_total{provider, model}`
3. Retry (malformed responses may be transient)

**Recovery Procedure:**
```yaml
retry_strategy:
  max_retries: 3
  backoff: exponential [1s, 2s, 4s]

  retry_actions:
    attempt_1:
      action: Retry with same prompt
      rationale: May be transient LLM issue

    attempt_2:
      action: Retry with simplified prompt
      modification: Remove complex instructions

    attempt_3:
      action: Retry with explicit schema in prompt
      modification: Add "Respond with JSON: {schema}"

  final_failure:
    action: Raise EvaluatorError
    message: |
      LLM returned malformed response after 3 retries.

      Expected: {schema}
      Received: {response}
      Errors: {validation_errors}
```

**User Impact:**
- Transparent retry (3 attempts)
- Evaluation fails if all retries return malformed responses
- Error includes actual response for debugging

**Monitoring:**
```yaml
alerts:
  - name: HighMalformedResponseRate
    condition: llm_malformed_response_total rate > 5% over 10 minutes
    severity: warning
    action: Review prompts, check for provider API changes
```

---

### Failure Mode Summary Table

| Failure Mode | Detection | Retry? | User Impact | Recovery Time |
|--------------|-----------|--------|-------------|---------------|
| FM-001: Timeout | asyncio.TimeoutError | Yes (3x) | 90s delay | Automatic |
| FM-002: Rate Limit | HTTP 429 | Yes (respect Retry-After) | 60-420s delay | Automatic |
| FM-003: Pool Exhausted | httpx.PoolTimeout | Wait 10s | 0-10s delay | Automatic |
| FM-004: Circuit Breaker | CircuitBreakerOpen | No | 60-600s fast-fail | Automatic |
| FM-005: Invalid API Key | HTTP 401 | No | Immediate fail | Manual |
| FM-006: Model Not Found | HTTP 404 | No | Immediate fail | Manual |
| FM-007: Malformed Response | ValidationError | Yes (3x) | 7s delay | Automatic |

---

## Data Flow

### Evaluation Request Flow

```
USER
  │
  │ await evaluate(output, reference, evaluators=["semantic"])
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ PUBLIC API (api.py)                                         │
│ 1. Validate inputs (output not empty, evaluators exist)    │
│ 2. Get/create LLM client (via LLMManager)                 │
│ 3. Build middleware pipeline                               │
│ 4. For each evaluator:                                     │
│    - Instantiate from registry or use provided instance    │
│    - Execute via middleware pipeline                       │
│ 5. Aggregate results                                       │
│ 6. Return EvaluationResult                                 │
└────────────┬───────────────────────────────────────────────┘
             │
             │ middleware_pipeline.execute(output, reference, evaluator.evaluate)
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ MIDDLEWARE PIPELINE                                         │
│ LoggingMiddleware                                           │
│   ├─> Log: request_id, evaluator, model, input_lengths     │
│   └─> next(output, reference, context)                     │
│        │                                                    │
│        ▼                                                    │
│ MetricsMiddleware                                           │
│   ├─> Start timer                                          │
│   └─> next(output, reference, context)                     │
│        │                                                    │
│        ▼                                                    │
│ RateLimitingMiddleware                                      │
│   ├─> Check token bucket                                   │
│   ├─> If empty: wait or raise RateLimitError               │
│   └─> next(output, reference, context)                     │
│        │                                                    │
│        ▼                                                    │
│ CachingMiddleware                                           │
│   ├─> Compute cache key: hash(output+ref+criteria+eval)    │
│   ├─> Check cache                                          │
│   ├─> If hit: return cached result                         │
│   └─> If miss: next(output, reference, context)            │
│        │                                                    │
│        ▼                                                    │
│ [Final Handler: Evaluator]                                 │
└────────────┬───────────────────────────────────────────────┘
             │
             │ evaluator.evaluate(output, reference, criteria)
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ EVALUATOR (e.g., SemanticEvaluator)                        │
│ 1. Validate inputs                                         │
│ 2. system_prompt = _get_system_prompt()                    │
│ 3. user_prompt = _get_user_prompt(output, reference)       │
│ 4. response_type = _get_response_type()                    │
│ 5. Create PydanticAI agent                                 │
│ 6. Call LLM via agent.run(user_prompt)                     │
│ 7. response = await agent.run() # Structured output        │
│ 8. score = _compute_score(response)                        │
│ 9. Track interaction (tokens, latency, purpose)            │
│ 10. Return Score                                           │
└────────────┬───────────────────────────────────────────────┘
             │
             │ client.create_agent(response_type, system_prompt)
             │ agent.run(user_prompt)
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ LLM CLIENT                                                  │
│ 1. Check circuit breaker state                             │
│ 2. Get connection from pool                                │
│ 3. Build HTTP request:                                     │
│    - Headers: Authorization, Content-Type                  │
│    - Body: {model, messages, response_format}              │
│ 4. Execute with retry logic:                               │
│    - Try HTTP request                                      │
│    - If timeout: retry with backoff                        │
│    - If 429: wait Retry-After, retry                       │
│    - If 5xx: retry with backoff                            │
│    - If 4xx (not 429): fail immediately                    │
│ 5. Parse response                                          │
│ 6. Validate with Pydantic                                  │
│ 7. Track metrics (tokens, latency)                         │
│ 8. Update circuit breaker (success/failure)                │
│ 9. Return structured response                              │
└────────────┬───────────────────────────────────────────────┘
             │
             │ HTTPS POST
             │ Authorization: Bearer sk-...
             │ Body: {model, messages, ...}
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ EXTERNAL LLM PROVIDER (OpenAI, Anthropic, etc.)            │
│ 1. Authenticate request                                    │
│ 2. Validate payload                                        │
│ 3. Check rate limits                                       │
│ 4. Generate LLM response                                   │
│ 5. Return JSON response                                    │
└─────────────────────────────────────────────────────────────┘
```

### Error Flow (FM-001: Timeout)

```
LLM CLIENT
  │
  │ HTTP request to provider
  │ Timeout after 30s
  │
  ▼
┌─────────────────────────────────────────┐
│ RETRY LOGIC                             │
│ Attempt 1: FAIL (timeout)               │
│   ├─> Wait 1s                           │
│   └─> Retry                             │
│ Attempt 2: FAIL (timeout)               │
│   ├─> Wait 2s                           │
│   └─> Retry                             │
│ Attempt 3: FAIL (timeout)               │
│   ├─> Wait 4s                           │
│   └─> Give up                           │
│ Record failure in circuit breaker       │
│ Raise TimeoutError                      │
└───────────┬─────────────────────────────┘
            │
            │ TimeoutError exception
            │
            ▼
┌─────────────────────────────────────────┐
│ MIDDLEWARE PIPELINE (unwind)            │
│ CachingMiddleware: Don't cache error    │
│ RateLimitingMiddleware: No-op           │
│ MetricsMiddleware: Record error metric  │
│ LoggingMiddleware: Log error details    │
│ Propagate exception up                  │
└───────────┬─────────────────────────────┘
            │
            │ TimeoutError exception
            │
            ▼
┌─────────────────────────────────────────┐
│ PUBLIC API                              │
│ Cleanup: Close temporary client         │
│ Propagate exception to user             │
└───────────┬─────────────────────────────┘
            │
            │ TimeoutError exception
            │
            ▼
          USER
```

---

## Integration Patterns

### Loom Pipeline Integration

```python
# Loom configuration: arbiter_config.yaml
evaluations:
  - name: factuality_check
    type: arbiter.FactualityEvaluator
    threshold: 0.85
    fail_on_error: true

  - name: groundedness_check
    type: arbiter.GroundednessEvaluator
    threshold: 0.80
    fail_on_error: false  # Warning only

quality_gate:
  mode: all_pass  # Options: all_pass, any_pass, weighted
  weights:
    factuality_check: 0.6
    groundedness_check: 0.4
```

```python
# Loom implementation
from arbiter import FactualityEvaluator, GroundednessEvaluator, LLMManager

async def evaluate_step(llm_output: str, context: dict) -> dict:
    """Loom pipeline step for evaluation."""
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Run evaluations in parallel
    results = await asyncio.gather(
        FactualityEvaluator(client).evaluate(
            output=llm_output,
            reference=context.get("reference")
        ),
        GroundednessEvaluator(client).evaluate(
            output=llm_output,
            reference=context.get("sources")
        )
    )

    # Quality gate logic
    factuality_score, groundedness_score = results
    passed = (
        factuality_score.value >= 0.85 and
        groundedness_score.value >= 0.80
    )

    return {
        "passed": passed,
        "scores": {
            "factuality": factuality_score.value,
            "groundedness": groundedness_score.value
        },
        "explanations": {
            "factuality": factuality_score.explanation,
            "groundedness": groundedness_score.explanation
        }
    }
```

### LangChain Integration (Planned)

```python
from langchain.evaluation import load_evaluator
from arbiter import SemanticEvaluator, LLMManager

# Register Arbiter evaluator with LangChain
class ArbiterSemanticEvaluator:
    """LangChain wrapper for Arbiter evaluators."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = None

    async def _initialize(self):
        if not self.client:
            self.client = await LLMManager.get_client(model=self.model)
            self.evaluator = SemanticEvaluator(self.client)

    async def aevaluate(
        self,
        prediction: str,
        reference: str,
        **kwargs
    ) -> dict:
        """LangChain-compatible evaluate interface."""
        await self._initialize()

        score = await self.evaluator.evaluate(
            output=prediction,
            reference=reference
        )

        return {
            "score": score.value,
            "confidence": score.confidence,
            "reasoning": score.explanation
        }

# Usage in LangChain
evaluator = ArbiterSemanticEvaluator(model="gpt-4o-mini")
result = await evaluator.aevaluate(
    prediction="Paris is the capital of France",
    reference="The capital of France is Paris"
)
```

---

## Security Architecture

### API Key Management

```yaml
api_key_storage:
  primary: Environment variables
  format: PROVIDER_API_KEY (e.g., OPENAI_API_KEY)
  required_at: Runtime (import time)

  security_requirements:
    - NEVER commit to version control
    - NEVER log full keys (log last 4 chars only)
    - NEVER include in error messages
    - NEVER transmit except to provider API (HTTPS only)

  rotation_procedure:
    step_1: Generate new key in provider dashboard
    step_2: Update environment variable
    step_3: Restart application
    step_4: Verify with test evaluation
    step_5: Revoke old key

  secrets_management:
    development: .env files (gitignored)
    staging: Vault, AWS Secrets Manager, or equivalent
    production: Kubernetes secrets, HashiCorp Vault, AWS Secrets Manager
```

### Input Sanitization

```python
def validate_and_sanitize_input(output: str) -> str:
    """Validate and sanitize user inputs.

    Security Checks:
    1. Length limits (prevent DoS via large inputs)
    2. Character encoding (UTF-8 only)
    3. Null byte injection prevention
    4. Control character removal

    Prompt Injection Mitigation:
    - Log suspicious patterns (e.g., "Ignore previous instructions")
    - Emit security metric if detected
    - Don't block (false positives too high)
    - Monitor for abuse patterns
    """
    if not output:
        raise ValidationError("Output cannot be empty")

    if len(output) > 50_000:
        logger.warning(f"Large input truncated: {len(output)} -> 50000 chars")
        output = output[:50_000]

    # Remove null bytes
    output = output.replace("\x00", "")

    # Check for potential prompt injection
    suspicious_patterns = [
        "ignore previous",
        "disregard all",
        "new instructions",
        "system:",
        "you are now"
    ]

    for pattern in suspicious_patterns:
        if pattern in output.lower():
            metrics.increment("security.suspicious_input_pattern", tags={"pattern": pattern})
            logger.warning(f"Suspicious pattern detected: {pattern}")

    return output
```

### Rate Limiting

```python
class RateLimitingMiddleware:
    """Prevent abuse via rate limiting.

    Security Features:
    - Per-client rate limiting (future: require client ID)
    - Global rate limiting (protect infrastructure)
    - Adaptive rate limiting (detect abuse patterns)

    Current Implementation:
    - Token bucket algorithm
    - Default: 100 requests per minute
    - Configurable per deployment

    Future Enhancements:
    - Per-API-key rate limits
    - Per-IP rate limits (if deployed as service)
    - Dynamic rate limiting based on system load
    """
```

---

## Deployment Architecture

### Standalone Library (Current)

```yaml
deployment_model: pip_install
installation:
  - pip install arbiter
  - pip install arbiter[scale]  # With FAISS

runtime_requirements:
  python: ">=3.10"
  memory: ~50MB base + 5MB per concurrent evaluation
  cpu: Negligible (I/O bound, not CPU bound)
  network: HTTPS egress to LLM providers

environment_variables:
  required:
    - At least one provider API key (e.g., OPENAI_API_KEY)
  optional:
    - ARBITER_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR
    - ARBITER_POOL_SIZE: Connection pool max size
    - ARBITER_TIMEOUT: Default LLM timeout (seconds)

configuration:
  primary: Environment variables
  secondary: Code (RetryConfig, MiddlewareConfig, etc.)
  persistence: None (stateless library)
```

### Future: Service Deployment (v2.0+)

```yaml
deployment_model: containerized_service
container:
  image: arbiter:1.0
  base: python:3.10-slim
  size: ~200MB

kubernetes:
  deployment:
    replicas: 3
    resources:
      requests:
        memory: 256Mi
        cpu: 100m
      limits:
        memory: 512Mi
        cpu: 500m

  service:
    type: LoadBalancer
    port: 8080
    protocol: HTTP

  ingress:
    enabled: true
    tls: true
    rate_limiting: 1000 req/min per client

  autoscaling:
    min_replicas: 3
    max_replicas: 10
    target_cpu: 70%
    target_memory: 80%

observability:
  metrics:
    exporter: Prometheus
    port: 9090
    path: /metrics

  logging:
    format: JSON
    level: INFO
    destination: stdout
    aggregation: ELK, Datadog, or CloudWatch

  tracing:
    enabled: true
    backend: Jaeger, Zipkin, or OpenTelemetry
    sample_rate: 0.1  # 10% of requests
```

---

## Related Documentation

- **DESIGN_SPEC.md** - Vision, features, and competitive analysis
- **IMPLEMENTATION_SPEC.md** - Coding standards and component requirements
- **PROJECT_TODO.md** - Current development tasks
- **ROADMAP.md** - Development phases and timeline

---

**Last Updated:** 2025-11-16
**Next Review:** 2026-01-15 (Post v1.0 release)
