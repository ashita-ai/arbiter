"""Main API for Arbiter evaluation framework.

This module provides the primary entry points for evaluating LLM outputs.

## Quick Start:

    >>> from arbiter import evaluate
    >>> from arbiter.core import LLMManager
    >>>
    >>> # Evaluate semantic similarity
    >>> client = await LLMManager.get_client(model="gpt-4o")
    >>> result = await evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris",
    ...     evaluators=["semantic"],
    ...     llm_client=client
    ... )
    >>>
    >>> print(f"Overall Score: {result.overall_score:.2f}")
    >>> print(f"LLM Calls: {len(result.interactions)}")
    >>> print(f"Total Cost: ${result.total_llm_cost():.4f}")

## With Multiple Evaluators:

    >>> result = await evaluate(
    ...     output="The quick brown fox jumps over the lazy dog",
    ...     reference="A fast brown fox leaps above a sleepy canine",
    ...     evaluators=["semantic"],  # More evaluators coming soon
    ...     llm_client=client
    ... )
"""

import time
from typing import List, Optional, Union

from .core import LLMClient, LLMManager, Provider
from .core.exceptions import ValidationError
from .core.middleware import MiddlewarePipeline
from .core.models import EvaluationResult, LLMInteraction, Metric, Score
from .evaluators import SemanticEvaluator

__all__ = ["evaluate"]


async def evaluate(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
    middleware: Optional[MiddlewarePipeline] = None,
) -> EvaluationResult:
    """Evaluate an LLM output with automatic interaction tracking.

    This is the main entry point for Arbiter evaluations. It runs one or more
    evaluators on the output and returns comprehensive results including all
    LLM interactions for full transparency.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison-based evaluation
        criteria: Optional evaluation criteria for reference-free evaluation
        evaluators: List of evaluator names to run (default: ["semantic"])
        llm_client: Optional pre-configured LLM client (will create if not provided)
        model: Model to use if creating new client (default: "gpt-4o")
        provider: Provider to use if creating new client (default: OPENAI)
        threshold: Score threshold for pass/fail (default: 0.7)
        middleware: Optional middleware pipeline for cross-cutting concerns

    Returns:
        EvaluationResult with scores, metrics, and complete LLM interaction tracking

    Raises:
        ValidationError: If input validation fails
        ValueError: If evaluator name is not recognized
        EvaluatorError: If evaluation fails

    Example:
        >>> # Basic semantic evaluation
        >>> result = await evaluate(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     evaluators=["semantic"]
        ... )
        >>> print(f"Score: {result.overall_score:.2f}")
        >>> print(f"Passed: {result.passed}")
        >>>
        >>> # Check LLM usage
        >>> for interaction in result.interactions:
        ...     print(f"Purpose: {interaction.purpose}")
        ...     print(f"Latency: {interaction.latency:.2f}s")
        ...     print(f"Model: {interaction.model}")
        >>>
        >>> # Calculate cost
        >>> cost = result.total_llm_cost(cost_per_1k_tokens=0.03)
        >>> print(f"Total cost: ${cost:.4f}")
    """
    # Input validation
    if not output or not output.strip():
        raise ValidationError("output cannot be empty or whitespace")

    if reference is not None and not reference.strip():
        raise ValidationError("reference cannot be empty or whitespace if provided")

    if criteria is not None and not criteria.strip():
        raise ValidationError("criteria cannot be empty or whitespace if provided")

    # If middleware is provided, use it to wrap the evaluation
    if middleware:
        async def _evaluate_core(output: str, reference: Optional[str]) -> EvaluationResult:
            return await _evaluate_impl(
                output=output,
                reference=reference,
                criteria=criteria,
                evaluators=evaluators,
                llm_client=llm_client,
                model=model,
                provider=provider,
                threshold=threshold,
            )

        return await middleware.execute(output, reference, _evaluate_core)

    # No middleware, run directly
    return await _evaluate_impl(
        output=output,
        reference=reference,
        criteria=criteria,
        evaluators=evaluators,
        llm_client=llm_client,
        model=model,
        provider=provider,
        threshold=threshold,
    )


async def _evaluate_impl(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
) -> EvaluationResult:
    """Internal implementation of evaluation (called directly or via middleware)."""
    start_time = time.time()

    # Default to semantic evaluator if none specified
    if evaluators is None:
        evaluators = ["semantic"]

    # Create LLM client if not provided
    if llm_client is None:
        llm_client = await LLMManager.get_client(
            provider=provider, model=model, temperature=0.0
        )

    # Initialize evaluator instances
    evaluator_instances = []
    for evaluator_name in evaluators:
        if evaluator_name == "semantic":
            evaluator_instances.append(SemanticEvaluator(llm_client))
        else:
            raise ValueError(
                f"Unknown evaluator: {evaluator_name}. "
                f"Available: semantic"
            )

    # Run evaluations and collect scores
    scores: List[Score] = []
    metrics: List[Metric] = []
    all_interactions: List[LLMInteraction] = []
    evaluator_names: List[str] = []

    for evaluator in evaluator_instances:
        # Clear previous interactions
        evaluator.clear_interactions()

        # Run evaluation
        eval_start = time.time()
        score = await evaluator.evaluate(output, reference, criteria)
        eval_time = time.time() - eval_start

        # Collect results
        scores.append(score)
        evaluator_names.append(evaluator.name)

        # Collect interactions from this evaluator
        interactions = evaluator.get_interactions()
        all_interactions.extend(interactions)

        # Create metric metadata
        metric = Metric(
            name=evaluator.name,
            evaluator=evaluator.name,
            model=llm_client.model,
            processing_time=eval_time,
            tokens_used=sum(i.tokens_used for i in interactions),
            metadata={
                "interaction_count": len(interactions),
                "has_reference": reference is not None,
                "has_criteria": criteria is not None,
            },
        )
        metrics.append(metric)

    # Calculate overall score (average for now)
    overall_score = sum(s.value for s in scores) / len(scores) if scores else 0.0

    # Determine if passed threshold
    passed = overall_score >= threshold

    # Calculate totals
    total_tokens = sum(i.tokens_used for i in all_interactions)
    processing_time = time.time() - start_time

    return EvaluationResult(
        output=output,
        reference=reference,
        criteria=criteria,
        scores=scores,
        overall_score=overall_score,
        passed=passed,
        metrics=metrics,
        evaluator_names=evaluator_names,
        total_tokens=total_tokens,
        processing_time=processing_time,
        interactions=all_interactions,
        metadata={
            "evaluator_count": len(evaluator_instances),
            "threshold": threshold,
        },
    )
