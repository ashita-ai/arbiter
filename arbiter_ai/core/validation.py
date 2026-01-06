"""Input validation for Arbiter API functions.

This module provides validation functions that check inputs before making
expensive LLM API calls. Clear validation errors save users time and costs.

Example:
    >>> from arbiter_ai.core.validation import validate_evaluate_inputs
    >>> from arbiter_ai.core.exceptions import ValidationError
    >>>
    >>> try:
    ...     validate_evaluate_inputs(
    ...         output="",
    ...         reference=None,
    ...         criteria=None,
    ...         evaluators=["semantic"],
    ...         threshold=0.7,
    ...         model="gpt-4o"
    ...     )
    ... except ValidationError as e:
    ...     print(f"Validation failed: {e}")
"""

from typing import Any, Dict, List, Optional

from .exceptions import ValidationError
from .registry import get_available_evaluators

__all__ = [
    "validate_evaluate_inputs",
    "validate_compare_inputs",
    "validate_batch_evaluate_inputs",
]


# Maximum length for text inputs (100,000 characters)
MAX_TEXT_LENGTH = 100_000


def validate_evaluate_inputs(
    output: str,
    reference: Optional[str],
    criteria: Optional[str],
    evaluators: Optional[List[str]],
    threshold: float,
    model: str,
) -> None:
    """Validate inputs for evaluate() function.

    Performs comprehensive validation of all inputs before evaluation starts,
    catching errors early to save API costs and provide clear error messages.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison
        criteria: Optional evaluation criteria
        evaluators: Optional list of evaluator names
        threshold: Score threshold for pass/fail
        model: Model name for LLM client

    Raises:
        ValidationError: If any input is invalid with descriptive message

    Example:
        >>> validate_evaluate_inputs(
        ...     output="Paris is the capital of France",
        ...     reference="Paris is France's capital",
        ...     criteria=None,
        ...     evaluators=["semantic"],
        ...     threshold=0.7,
        ...     model="gpt-4o"
        ... )
        >>> # No exception raised - inputs are valid
    """
    # Output validation
    if not output or not output.strip():
        raise ValidationError("'output' cannot be empty")

    if len(output) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"'output' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
            f"got {len(output):,}"
        )

    # Evaluators validation
    if evaluators is not None:
        if len(evaluators) == 0:
            raise ValidationError(
                "'evaluators' cannot be empty - provide at least one evaluator"
            )

        available = get_available_evaluators()
        for evaluator_name in evaluators:
            if evaluator_name not in available:
                raise ValidationError(
                    f"Unknown evaluator '{evaluator_name}'. "
                    f"Available: {', '.join(sorted(available))}"
                )

    # Threshold validation
    if not 0.0 <= threshold <= 1.0:
        raise ValidationError(
            f"'threshold' must be between 0.0 and 1.0, got {threshold}"
        )

    # Model validation
    if not model or not model.strip():
        raise ValidationError("'model' cannot be empty")

    # Reference validation for semantic evaluator
    if evaluators and "semantic" in evaluators:
        if not reference or not reference.strip():
            raise ValidationError(
                "'reference' is required when using 'semantic' evaluator"
            )

        if len(reference) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"'reference' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
                f"got {len(reference):,}"
            )

    # Criteria validation for custom_criteria evaluator
    if evaluators and "custom_criteria" in evaluators:
        if not criteria or not criteria.strip():
            raise ValidationError(
                "'criteria' is required when using 'custom_criteria' evaluator"
            )

        if len(criteria) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"'criteria' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
                f"got {len(criteria):,}"
            )

    # Reference validation (if provided)
    if reference is not None and len(reference) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"'reference' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
            f"got {len(reference):,}"
        )

    # Criteria validation (if provided)
    if criteria is not None and len(criteria) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"'criteria' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
            f"got {len(criteria):,}"
        )


def validate_compare_inputs(
    output_a: str,
    output_b: str,
    criteria: Optional[str],
    model: str,
) -> None:
    """Validate inputs for compare() function.

    Validates that both outputs are non-empty and within length limits
    before performing pairwise comparison.

    Args:
        output_a: First LLM output to compare
        output_b: Second LLM output to compare
        criteria: Optional comparison criteria
        model: Model name for LLM client

    Raises:
        ValidationError: If any input is invalid with descriptive message

    Example:
        >>> validate_compare_inputs(
        ...     output_a="Paris is the capital of France",
        ...     output_b="The capital of France is Paris",
        ...     criteria="accuracy and clarity",
        ...     model="gpt-4o"
        ... )
        >>> # No exception raised - inputs are valid
    """
    # Output A validation
    if not output_a or not output_a.strip():
        raise ValidationError("'output_a' cannot be empty")

    if len(output_a) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"'output_a' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
            f"got {len(output_a):,}"
        )

    # Output B validation
    if not output_b or not output_b.strip():
        raise ValidationError("'output_b' cannot be empty")

    if len(output_b) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"'output_b' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
            f"got {len(output_b):,}"
        )

    # Model validation
    if not model or not model.strip():
        raise ValidationError("'model' cannot be empty")

    # Criteria validation (if provided)
    if criteria is not None:
        if not criteria.strip():
            raise ValidationError("'criteria' cannot be empty if provided")

        if len(criteria) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"'criteria' exceeds maximum length ({MAX_TEXT_LENGTH:,} chars), "
                f"got {len(criteria):,}"
            )


def validate_batch_evaluate_inputs(
    items: List[Dict[str, Any]],
    evaluators: Optional[List[str]],
    threshold: float,
    model: str,
    max_concurrency: int,
) -> None:
    """Validate inputs for batch_evaluate() function.

    Validates the batch items list and checks each item has required fields
    before starting batch evaluation.

    Args:
        items: List of evaluation items with "output" and optional fields
        evaluators: Optional list of evaluator names
        threshold: Score threshold for pass/fail
        model: Model name for LLM client
        max_concurrency: Maximum parallel evaluations

    Raises:
        ValidationError: If any input is invalid with descriptive message

    Example:
        >>> validate_batch_evaluate_inputs(
        ...     items=[
        ...         {"output": "Paris is the capital", "reference": "Paris"},
        ...         {"output": "Berlin is the capital", "reference": "Berlin"}
        ...     ],
        ...     evaluators=["semantic"],
        ...     threshold=0.7,
        ...     model="gpt-4o",
        ...     max_concurrency=10
        ... )
        >>> # No exception raised - inputs are valid
    """
    # Items validation
    if not items or len(items) == 0:
        raise ValidationError("'items' cannot be empty - provide at least one item")

    # Evaluators validation
    if evaluators is not None:
        if len(evaluators) == 0:
            raise ValidationError(
                "'evaluators' cannot be empty - provide at least one evaluator"
            )

        available = get_available_evaluators()
        for evaluator_name in evaluators:
            if evaluator_name not in available:
                raise ValidationError(
                    f"Unknown evaluator '{evaluator_name}'. "
                    f"Available: {', '.join(sorted(available))}"
                )

    # Threshold validation
    if not 0.0 <= threshold <= 1.0:
        raise ValidationError(
            f"'threshold' must be between 0.0 and 1.0, got {threshold}"
        )

    # Model validation
    if not model or not model.strip():
        raise ValidationError("'model' cannot be empty")

    # Max concurrency validation
    if max_concurrency <= 0:
        raise ValidationError(
            f"'max_concurrency' must be positive, got {max_concurrency}"
        )

    # Validate each item in the batch
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValidationError(
                f"Item at index {i} must be a dictionary, got {type(item).__name__}"
            )

        # Check required "output" field
        if "output" not in item:
            raise ValidationError(f"Item at index {i} missing required field 'output'")

        output = item["output"]
        if not output or not str(output).strip():
            raise ValidationError(f"Item at index {i}: 'output' cannot be empty")

        if len(str(output)) > MAX_TEXT_LENGTH:
            raise ValidationError(
                f"Item at index {i}: 'output' exceeds maximum length "
                f"({MAX_TEXT_LENGTH:,} chars), got {len(str(output)):,}"
            )

        # Validate optional reference field
        if "reference" in item:
            reference = item["reference"]
            if reference is not None and len(str(reference)) > MAX_TEXT_LENGTH:
                raise ValidationError(
                    f"Item at index {i}: 'reference' exceeds maximum length "
                    f"({MAX_TEXT_LENGTH:,} chars), got {len(str(reference)):,}"
                )

        # Validate optional criteria field
        if "criteria" in item:
            criteria = item["criteria"]
            if criteria is not None and len(str(criteria)) > MAX_TEXT_LENGTH:
                raise ValidationError(
                    f"Item at index {i}: 'criteria' exceeds maximum length "
                    f"({MAX_TEXT_LENGTH:,} chars), got {len(str(criteria)):,}"
                )
