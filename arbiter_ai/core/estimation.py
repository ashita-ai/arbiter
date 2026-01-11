"""Cost estimation for Arbiter evaluations without making API calls.

This module provides functions to estimate evaluation costs before running,
helping users budget for large batch evaluations.

## Key Features:

- **Accurate Token Counting**: Uses LiteLLM for provider-agnostic token counts
- **Cost Calculation**: Uses LiteLLM pricing for accurate cost estimates
- **Prompt Preview**: Shows what prompts would be sent (dry-run mode)
- **No API Calls**: All estimation is done locally

## Usage:

    >>> from arbiter_ai import estimate_evaluation_cost
    >>>
    >>> estimate = await estimate_evaluation_cost(
    ...     output="Text to evaluate...",
    ...     reference="Reference text...",
    ...     evaluators=["semantic"],
    ...     model="gpt-4o-mini"
    ... )
    >>> print(f"Estimated cost: ${estimate.total_cost:.6f}")
    >>> print(f"Estimated tokens: {estimate.total_tokens}")
"""

import logging
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel, ConfigDict, Field

from .cost_calculator import get_cost_calculator
from .registry import get_available_evaluators, get_evaluator_class

logger = logging.getLogger(__name__)

__all__ = [
    "CostEstimate",
    "BatchCostEstimate",
    "DryRunResult",
    "estimate_tokens",
    "estimate_evaluation_cost",
    "estimate_batch_cost",
    "get_prompt_preview",
]

# Fallback: chars per token when litellm fails
FALLBACK_CHARS_PER_TOKEN = 4

# Estimated output tokens per evaluator (based on response structure)
ESTIMATED_OUTPUT_TOKENS = {
    "semantic": 150,  # Score + explanation + key differences/similarities
    "custom_criteria": 200,  # Score + explanation + criteria met/not met
    "factuality": 250,  # Score + claims + verification details
    "groundedness": 200,  # Score + grounded/ungrounded statements
    "relevance": 150,  # Score + explanation + relevance factors
    "pairwise": 200,  # Winner + confidence + reasoning
}

DEFAULT_OUTPUT_TOKENS = 150


class CostEstimate(BaseModel):
    """Cost estimate for a single evaluation.

    Attributes:
        total_cost: Estimated total cost in USD
        total_tokens: Estimated total tokens (input + output)
        input_tokens: Estimated input tokens
        output_tokens: Estimated output tokens
        by_evaluator: Cost breakdown by evaluator name
        model: Model used for estimation
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cost: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    by_evaluator: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    model: str = ""


class BatchCostEstimate(BaseModel):
    """Cost estimate for a batch evaluation.

    Attributes:
        total_cost: Estimated total cost for all items
        total_tokens: Estimated total tokens for all items
        per_item_cost: Average cost per item
        per_item_tokens: Average tokens per item
        item_count: Number of items in batch
        by_evaluator: Cost breakdown by evaluator
        model: Model used for estimation
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cost: float
    total_tokens: int
    per_item_cost: float
    per_item_tokens: int
    item_count: int
    by_evaluator: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    model: str = ""


class DryRunResult(BaseModel):
    """Result of a dry-run evaluation (no API calls made).

    Attributes:
        dry_run: Always True for dry-run results
        evaluators: List of evaluators that would be used
        prompts: Preview of prompts by evaluator
        estimated_tokens: Total estimated tokens
        estimated_cost: Total estimated cost in USD
        model: Model that would be used
        validation: Input validation status
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dry_run: bool = True
    evaluators: List[str] = Field(default_factory=list)
    prompts: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    estimated_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    validation: Dict[str, bool] = Field(default_factory=dict)


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens in a text string using LiteLLM.

    Uses LiteLLM's provider-agnostic token counter which automatically
    selects the correct tokenizer for any supported model (OpenAI,
    Anthropic, Google, Cohere, etc.). Falls back to character-based
    estimation if LiteLLM fails.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection (default: gpt-4o-mini)

    Returns:
        Token count

    Example:
        >>> tokens = estimate_tokens("Hello, world!", model="gpt-4o-mini")
        >>> print(tokens)  # Accurate count: 4
    """
    if not text:
        return 0

    try:
        # litellm.token_counter is not in type stubs but exists at runtime
        return litellm.token_counter(model=model, text=text)  # type: ignore[attr-defined]
    except Exception as e:
        # Fallback to character-based estimation
        logger.warning(f"LiteLLM token counting failed: {e}, using fallback")
        return max(1, len(text) // FALLBACK_CHARS_PER_TOKEN)


class _DummyLLMClient:
    """Minimal LLMClient stub for prompt extraction.

    This allows evaluator instantiation without a real LLM client.
    Only used for extracting prompt templates, not for actual API calls.
    """

    def __init__(self) -> None:
        self.model = "gpt-4o-mini"
        self.provider = "openai"

    def create_agent(self, *args: Any, **kwargs: Any) -> None:
        """Stub - never called during prompt extraction."""
        pass


def _get_evaluator_prompts(
    evaluator_name: str,
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
) -> Dict[str, str]:
    """Get the prompts that would be used for an evaluator.

    Args:
        evaluator_name: Name of the evaluator
        output: Output text to evaluate
        reference: Optional reference text
        criteria: Optional evaluation criteria

    Returns:
        Dictionary with 'system' and 'user' prompts
    """
    evaluator_class = get_evaluator_class(evaluator_name)
    if evaluator_class is None:
        logger.warning(f"Evaluator class not found for '{evaluator_name}'")
        return {"system": "", "user": ""}

    # Create a minimal evaluator instance to get prompts
    # Use a dummy LLM client since we only need the prompt templates
    try:
        dummy_client = _DummyLLMClient()

        # Try model= kwarg first (newer pattern), fall back to positional client
        try:
            evaluator = evaluator_class(model="gpt-4o-mini")
        except TypeError as e:
            logger.debug(
                f"Evaluator {evaluator_name} doesn't accept model=, trying client: {e}"
            )
            evaluator = evaluator_class(dummy_client)  # type: ignore[arg-type]

        # Extract prompts from BasePydanticEvaluator subclasses
        if hasattr(evaluator, "_get_system_prompt") and hasattr(
            evaluator, "_get_user_prompt"
        ):
            system_prompt: str = evaluator._get_system_prompt()
            user_prompt: str = evaluator._get_user_prompt(output, reference, criteria)
            return {"system": system_prompt, "user": user_prompt}

        logger.warning(f"Evaluator {evaluator_name} lacks prompt methods")
        return {"system": "", "user": ""}
    except Exception as e:
        logger.error(f"Failed to extract prompts from {evaluator_name}: {e}")
        return {"system": "", "user": ""}


async def estimate_evaluation_cost(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> CostEstimate:
    """Estimate the cost of an evaluation without making API calls.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison
        criteria: Optional evaluation criteria
        evaluators: List of evaluator names (defaults to ["semantic"])
        model: Model to use for cost calculation

    Returns:
        CostEstimate with token and cost breakdown

    Example:
        >>> estimate = await estimate_evaluation_cost(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(f"Estimated cost: ${estimate.total_cost:.6f}")
    """
    if evaluators is None:
        evaluators = ["semantic"]

    # Validate evaluators
    available = get_available_evaluators()
    for name in evaluators:
        if name not in available:
            raise ValueError(
                f"Unknown evaluator '{name}'. Available: {', '.join(sorted(available))}"
            )

    calc = get_cost_calculator()
    await calc.ensure_loaded()

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    by_evaluator: Dict[str, Dict[str, Any]] = {}

    for evaluator_name in evaluators:
        # Get prompts to estimate input tokens
        prompts = _get_evaluator_prompts(evaluator_name, output, reference, criteria)
        system_tokens = estimate_tokens(prompts["system"], model)
        user_tokens = estimate_tokens(prompts["user"], model)
        input_tokens = system_tokens + user_tokens

        # Estimate output tokens based on evaluator type
        output_tokens = ESTIMATED_OUTPUT_TOKENS.get(
            evaluator_name, DEFAULT_OUTPUT_TOKENS
        )

        # Calculate cost
        cost = calc.calculate_cost(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost += cost

        by_evaluator[evaluator_name] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }

    return CostEstimate(
        total_cost=total_cost,
        total_tokens=total_input_tokens + total_output_tokens,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        by_evaluator=by_evaluator,
        model=model,
    )


async def estimate_batch_cost(
    items: List[Dict[str, Any]],
    evaluators: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> BatchCostEstimate:
    """Estimate the cost of a batch evaluation without making API calls.

    Args:
        items: List of items to evaluate, each with "output" and optional fields
        evaluators: List of evaluator names (defaults to ["semantic"])
        model: Model to use for cost calculation

    Returns:
        BatchCostEstimate with total and per-item breakdown

    Example:
        >>> items = [
        ...     {"output": "Paris is the capital", "reference": "Paris"},
        ...     {"output": "Berlin is the capital", "reference": "Berlin"},
        ... ]
        >>> estimate = await estimate_batch_cost(
        ...     items=items,
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(f"Total cost: ${estimate.total_cost:.6f}")
        >>> print(f"Per item: ${estimate.per_item_cost:.6f}")
    """
    if evaluators is None:
        evaluators = ["semantic"]

    if not items:
        return BatchCostEstimate(
            total_cost=0.0,
            total_tokens=0,
            per_item_cost=0.0,
            per_item_tokens=0,
            item_count=0,
            by_evaluator={},
            model=model,
        )

    total_cost = 0.0
    total_tokens = 0
    aggregated_by_evaluator: Dict[str, Dict[str, Any]] = {
        name: {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        for name in evaluators
    }

    for item in items:
        output = item.get("output", "")
        reference = item.get("reference")
        criteria = item.get("criteria")

        estimate = await estimate_evaluation_cost(
            output=output,
            reference=reference,
            criteria=criteria,
            evaluators=evaluators,
            model=model,
        )

        total_cost += estimate.total_cost
        total_tokens += estimate.total_tokens

        for name, data in estimate.by_evaluator.items():
            aggregated_by_evaluator[name]["input_tokens"] += data["input_tokens"]
            aggregated_by_evaluator[name]["output_tokens"] += data["output_tokens"]
            aggregated_by_evaluator[name]["cost"] += data["cost"]

    item_count = len(items)
    return BatchCostEstimate(
        total_cost=total_cost,
        total_tokens=total_tokens,
        per_item_cost=total_cost / item_count,
        per_item_tokens=total_tokens // item_count,
        item_count=item_count,
        by_evaluator=aggregated_by_evaluator,
        model=model,
    )


async def get_prompt_preview(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    model: str = "gpt-4o-mini",
) -> DryRunResult:
    """Get a preview of what an evaluation would do without making API calls.

    This is the core of dry-run mode. It shows exactly what prompts would be
    sent to the LLM, along with cost estimates and validation status.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison
        criteria: Optional evaluation criteria
        evaluators: List of evaluator names (defaults to ["semantic"])
        model: Model that would be used

    Returns:
        DryRunResult with prompts, estimates, and validation info

    Example:
        >>> preview = await get_prompt_preview(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(preview.prompts["semantic"]["user"])
        >>> print(f"Estimated cost: ${preview.estimated_cost:.6f}")
    """
    if evaluators is None:
        evaluators = ["semantic"]

    # Validate evaluators
    available = get_available_evaluators()
    valid_evaluators = []
    for name in evaluators:
        if name in available:
            valid_evaluators.append(name)

    # Get prompts for each evaluator
    prompts: Dict[str, Dict[str, str]] = {}
    for evaluator_name in valid_evaluators:
        prompts[evaluator_name] = _get_evaluator_prompts(
            evaluator_name, output, reference, criteria
        )

    # Get cost estimate
    estimate = await estimate_evaluation_cost(
        output=output,
        reference=reference,
        criteria=criteria,
        evaluators=valid_evaluators,
        model=model,
    )

    # Validation checks
    validation = {
        "output_valid": bool(output and output.strip()),
        "reference_valid": reference is None or bool(reference.strip()),
        "criteria_valid": criteria is None or bool(criteria.strip()),
        "evaluators_valid": len(valid_evaluators) == len(evaluators),
    }

    return DryRunResult(
        dry_run=True,
        evaluators=valid_evaluators,
        prompts=prompts,
        estimated_tokens=estimate.total_tokens,
        estimated_cost=estimate.total_cost,
        model=model,
        validation=validation,
    )
