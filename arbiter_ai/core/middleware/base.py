"""Base middleware infrastructure for Arbiter.

This module provides the core abstractions for the middleware pattern:
- Middleware ABC for implementing middleware components
- MiddlewarePipeline for chaining middleware together
- Type definitions for middleware results and handlers
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

from ..logging import get_logger
from ..models import ComparisonResult, EvaluationResult
from ..type_defs import MiddlewareContext

# Type alias for results that can flow through middleware
MiddlewareResult = Union[EvaluationResult, ComparisonResult]

logger = get_logger("middleware")

__all__ = [
    "Middleware",
    "MiddlewarePipeline",
    "MiddlewareResult",
]


class Middleware(ABC):
    """Abstract base class for all middleware components.

    Middleware allows you to intercept and modify the evaluation process
    without changing core Arbiter logic. Each middleware can inspect or
    modify the request, add to the context, and process the response.

    The middleware pattern enables:
    - Logging and debugging
    - Performance monitoring
    - Caching and optimization
    - Security and rate limiting
    - Request/response transformation

    Example:
        >>> class TimingMiddleware(Middleware):
        ...     async def process(self, output, reference, next_handler, context):
        ...         start = time.time()
        ...         result = await next_handler(output, reference)
        ...         elapsed = time.time() - start
        ...         print(f"Evaluation took {elapsed:.2f} seconds")
        ...         return result
    """

    @abstractmethod
    async def process(
        self,
        output: str,
        reference: Optional[str],
        next_handler: Callable[[str, Optional[str]], Any],
        context: MiddlewareContext,
    ) -> MiddlewareResult:
        """Process the request through this middleware.

        This is the main method that each middleware must implement. It
        receives the request, can perform pre-processing, must call the
        next handler, and can perform post-processing.

        Args:
            output: The LLM output to be evaluated. Middleware can modify
                this before passing it to the next handler.
            reference: Reference text for comparison (may be None for
                reference-free evaluation).
            next_handler: Async callable representing the next middleware
                in the chain or the final evaluate handler. Must be called
                to continue processing.
            context: Mutable dictionary shared between all middleware in
                the pipeline. Used to pass data between middleware components.
                Common keys include 'evaluators', 'metrics', 'config'.

        Returns:
            MiddlewareResult (EvaluationResult or ComparisonResult) from
            the evaluation process. Middleware can inspect or modify this
            before returning.

        Raises:
            Any exception from the evaluation process. Middleware can
            catch and handle exceptions or let them propagate.

        Example:
            >>> async def process(self, output, reference, next_handler, context):
            ...     # Pre-processing
            ...     context['start_time'] = time.time()
            ...
            ...     # Must call next handler
            ...     result = await next_handler(output, reference)
            ...
            ...     # Post-processing
            ...     context['duration'] = time.time() - context['start_time']
            ...
            ...     return result
        """


class MiddlewarePipeline:
    """Manages the middleware pipeline.

    Orchestrates multiple middleware components into a single processing
    chain. Middleware are executed in the order they are added.

    Example:
        >>> pipeline = MiddlewarePipeline()
        >>> pipeline.add(LoggingMiddleware())
        >>> pipeline.add(MetricsMiddleware())
        >>> pipeline.add(CachingMiddleware())
        >>>
        >>> # Use with evaluation
        >>> result = await evaluate(output, ref, middleware=pipeline)
        >>>
        >>> # Get specific middleware
        >>> metrics = pipeline.get_middleware(MetricsMiddleware)
        >>> print(metrics.get_metrics())
    """

    def __init__(self, middleware: Optional[List[Middleware]] = None) -> None:
        """Initialize pipeline with optional middleware list.

        Args:
            middleware: Optional list of middleware to add initially
        """
        self.middleware: List[Middleware] = middleware or []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance to add

        Returns:
            Self for method chaining
        """
        self.middleware.append(middleware)
        return self

    def get_middleware(self, middleware_type: type) -> Optional[Middleware]:
        """Get middleware instance by type.

        Args:
            middleware_type: Class type of middleware to retrieve

        Returns:
            First middleware instance of the given type, or None
        """
        for mw in self.middleware:
            if isinstance(mw, middleware_type):
                return mw
        return None

    async def execute(
        self,
        output: str,
        reference: Optional[str],
        final_handler: Callable[[str, Optional[str]], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> EvaluationResult:
        """Execute the middleware pipeline.

        Args:
            output: Output text to evaluate
            reference: Reference text (may be None)
            final_handler: The actual evaluation function
            context: Shared context between middleware

        Returns:
            EvaluationResult from the pipeline
        """
        if context is None:
            context = {}

        # Build the chain
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> MiddlewareResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final handler
                result: MiddlewareResult = await final_handler(
                    current_output, current_reference
                )
                return result

            # Call current middleware
            current = self.middleware[index]
            return await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

        result = await chain(0, output, reference)
        # Type narrow to EvaluationResult for this method's return type
        if not isinstance(result, EvaluationResult):
            raise TypeError(
                f"Expected EvaluationResult from pipeline, got {type(result).__name__}"
            )
        return result

    async def execute_comparison(
        self,
        output_a: str,
        output_b: str,
        criteria: Optional[str],
        reference: Optional[str],
        final_handler: Callable[[str, str, Optional[str], Optional[str]], Any],
        context: Optional[MiddlewareContext] = None,
    ) -> ComparisonResult:
        """Execute middleware pipeline for pairwise comparison.

        This method adapts the pairwise comparison signature to work with
        the existing middleware infrastructure. Middleware can check the
        context for `is_pairwise_comparison=True` to detect and handle
        pairwise operations specially if needed.

        The adapter works by:
        1. Packaging both outputs into context for middleware access
        2. Passing a formatted string to middleware for logging/tracking
        3. Calling the final pairwise comparison handler
        4. Returning the ComparisonResult

        Args:
            output_a: First output to compare
            output_b: Second output to compare
            criteria: Optional comparison criteria
            reference: Optional reference context
            final_handler: The actual comparison function
            context: Shared context between middleware

        Returns:
            ComparisonResult from the pipeline

        Example:
            >>> pipeline = MiddlewarePipeline([
            ...     LoggingMiddleware(),
            ...     MetricsMiddleware()
            ... ])
            >>> result = await pipeline.execute_comparison(
            ...     output_a="First output",
            ...     output_b="Second output",
            ...     criteria="accuracy, clarity",
            ...     reference="Reference text",
            ...     final_handler=compare_impl
            ... )
        """
        if context is None:
            context = {}

        # Mark this as a pairwise comparison for middleware
        context["is_pairwise_comparison"] = True
        context["pairwise_data"] = {
            "output_a": output_a,
            "output_b": output_b,
            "criteria": criteria,
        }

        # Create formatted output for middleware logging
        # This allows existing middleware to work without modification
        formatted_output = (
            f"PAIRWISE COMPARISON:\n"
            f"Output A: {output_a[:100]}...\n"
            f"Output B: {output_b[:100]}..."
        )

        # Build the chain - adapter pattern
        async def chain(
            index: int, current_output: str, current_reference: Optional[str]
        ) -> MiddlewareResult:
            if index >= len(self.middleware):
                # End of middleware chain, call final pairwise handler
                # Use original outputs, not formatted version
                result = await final_handler(output_a, output_b, criteria, reference)
                return result  # type: ignore[no-any-return]

            # Call current middleware with formatted output
            current = self.middleware[index]

            # Middleware processes the formatted output but we preserve pairwise data
            result = await current.process(
                current_output,
                current_reference,
                lambda o, r: chain(index + 1, o, r),
                context,
            )

            return result

        # Execute the chain - the formatted output is for middleware visibility only
        result = await chain(0, formatted_output, reference)

        # Type narrow to ComparisonResult for this method's return type
        if not isinstance(result, ComparisonResult):
            raise TypeError(
                f"Expected ComparisonResult from pipeline, got {type(result).__name__}"
            )
        return result
