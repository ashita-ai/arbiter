"""Core data models for Arbiter evaluation framework.

This module defines the primary data structures used throughout Arbiter:
- EvaluationResult: Complete result of an evaluation
- Score: Individual metric score
- Metric: Metadata about a computed metric
- LLMInteraction: Track individual LLM API calls for transparency
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TextIO

from pydantic import BaseModel, ConfigDict, Field, computed_field

__all__ = [
    "Score",
    "Metric",
    "LLMInteraction",
    "EvaluationResult",
    "ComparisonResult",
    "BatchEvaluationResult",
]


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime.

    Replacement for deprecated datetime.utcnow() which returns naive datetime.
    Uses timezone.utc for Python 3.10+ compatibility.
    """
    return datetime.now(timezone.utc)


def _get_interaction_cost(interaction: "LLMInteraction") -> float:
    """Calculate cost for a single LLM interaction.

    Helper function to avoid code duplication across cost calculation methods.
    Uses cached cost if available, otherwise calculates from token counts.

    Args:
        interaction: LLMInteraction with token data and optional cached cost

    Returns:
        Cost in USD for this interaction
    """
    # Use cached cost if available
    if interaction.cost is not None:
        return interaction.cost

    # Calculate on-the-fly using cost calculator
    from arbiter_ai.core.cost_calculator import get_cost_calculator

    calc = get_cost_calculator()
    return calc.calculate_cost(
        model=interaction.model,
        input_tokens=interaction.input_tokens,
        output_tokens=interaction.output_tokens,
        cached_tokens=interaction.cached_tokens,
    )


class Score(BaseModel):
    """Individual evaluation score for a specific metric.

    Represents a single numeric score with metadata about how it was
    computed and what it represents.

    Example:
        >>> score = Score(
        ...     name="semantic_similarity",
        ...     value=0.92,
        ...     confidence=0.95,
        ...     explanation="High semantic overlap between output and reference"
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str = Field(..., description="Name of the metric (e.g., 'factuality')")
    value: float = Field(..., ge=0.0, le=1.0, description="Score value between 0 and 1")
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in this score"
    )
    explanation: Optional[str] = Field(
        None, description="Human-readable explanation of the score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the score"
    )

    def __lt__(self, other: object) -> bool:
        """Less than comparison by value."""
        if not isinstance(other, Score):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: object) -> bool:
        """Less than or equal comparison by value."""
        if not isinstance(other, Score):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: object) -> bool:
        """Greater than comparison by value."""
        if not isinstance(other, Score):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison by value."""
        if not isinstance(other, Score):
            return NotImplemented
        return self.value >= other.value

    def __eq__(self, other: object) -> bool:
        """Equality comparison by value (ignores name)."""
        if not isinstance(other, Score):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        """Hash by name and value for use in sets/dicts."""
        return hash((self.name, self.value))

    def meets_threshold(self, threshold: float) -> bool:
        """Check if score meets or exceeds threshold."""
        return self.value >= threshold

    def is_close_to(self, other: "Score", tolerance: float = 0.05) -> bool:
        """Check if score is within tolerance of another score."""
        return abs(self.value - other.value) <= tolerance

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if confidence exceeds threshold."""
        if self.confidence is None:
            return False
        return self.confidence >= threshold

    def is_low_confidence(self, threshold: float = 0.5) -> bool:
        """Check if confidence is below threshold."""
        if self.confidence is None:
            return True  # No confidence = low confidence
        return self.confidence < threshold

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        conf = f" confidence={self.confidence:.2f}" if self.confidence else ""
        return f"<Score name={self.name!r} value={self.value:.2f}{conf}>"


class LLMInteraction(BaseModel):
    """Record of a single LLM API call during evaluation.

    Tracks all LLM interactions for complete observability and debugging.
    Provides automatic transparency into evaluation processes.

    This provides:
    - Complete audit trail of LLM usage
    - Detailed token and cost tracking (input/output/cached)
    - Debugging capabilities
    - Transparency in how evaluations were computed

    Example:
        >>> interaction = LLMInteraction(
        ...     prompt="Evaluate the factuality of this statement...",
        ...     response="Score: 0.85. The statement is mostly accurate...",
        ...     model="gpt-4o",
        ...     input_tokens=120,
        ...     output_tokens=30,
        ...     latency=1.2,
        ...     purpose="factuality_scoring",
        ...     cost=0.000045
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    prompt: str = Field(..., description="The prompt sent to the LLM")
    response: str = Field(..., description="The LLM's response")
    model: str = Field(..., description="Model used for this call")

    # Token tracking (new detailed fields)
    input_tokens: int = Field(default=0, ge=0, description="Input tokens consumed")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens generated")
    cached_tokens: int = Field(
        default=0,
        ge=0,
        description="Cached input tokens (e.g., Anthropic prompt caching)",
    )

    # Backward compatibility
    tokens_used: int = Field(
        default=0, ge=0, description="Total tokens (for backward compatibility)"
    )

    # Cost tracking
    cost: Optional[float] = Field(
        None, ge=0, description="Actual cost in USD (calculated using LiteLLM pricing)"
    )

    latency: float = Field(..., ge=0, description="Time taken for this call (seconds)")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When this call was made"
    )
    purpose: str = Field(
        ...,
        description="Purpose of this call (e.g., 'scoring', 'semantic_comparison', 'factuality_check')",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context about this call"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Total tokens across input and output.

        Returns:
            Sum of input_tokens and output_tokens

        Example:
            >>> interaction.total_tokens
            150
        """
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        return (
            f"<LLMInteraction model={self.model!r} "
            f"tokens={self.input_tokens}+{self.output_tokens} "
            f"latency={self.latency:.2f}s>"
        )


class Metric(BaseModel):
    """Metadata about a computed metric.

    Provides information about how a metric was computed, including
    the model used, processing time, and any relevant context.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str = Field(..., description="Name of the metric")
    evaluator: str = Field(..., description="Name of the evaluator that computed it")
    model: Optional[str] = Field(None, description="LLM model used (if applicable)")
    processing_time: float = Field(..., description="Time taken to compute (seconds)")
    tokens_used: int = Field(default=0, description="Tokens consumed (if applicable)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        model_str = f" model={self.model!r}" if self.model else ""
        return (
            f"<Metric name={self.name!r} evaluator={self.evaluator!r}"
            f"{model_str} time={self.processing_time:.2f}s>"
        )


class EvaluationResult(BaseModel):
    """Complete result of an evaluation operation.

    Contains all scores, metadata, and audit trail information from
    evaluating an LLM output against reference or criteria.

    This is the primary result object returned by all evaluation
    operations in Arbiter.

    Example:
        >>> result = EvaluationResult(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     scores=[
        ...         Score(name="semantic_similarity", value=0.95),
        ...         Score(name="factuality", value=1.0),
        ...     ],
        ...     overall_score=0.975,
        ...     passed=True,
        ...     partial=False,
        ...     errors={}
        ... )
        >>>
        >>> # Partial result example (some evaluators failed)
        >>> result = EvaluationResult(
        ...     output="text",
        ...     scores=[Score(name="semantic", value=0.8)],
        ...     overall_score=0.8,
        ...     passed=True,
        ...     partial=True,
        ...     errors={"factuality": "API timeout"}
        ... )
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Input data
    output: str = Field(..., description="The LLM output that was evaluated")
    reference: Optional[str] = Field(
        None, description="Reference text used for comparison (if applicable)"
    )
    criteria: Optional[str] = Field(
        None, description="Evaluation criteria used (if reference-free)"
    )

    # Results
    scores: List[Score] = Field(
        default_factory=list,
        description="Individual metric scores from successful evaluators only. Failed evaluators are excluded.",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score calculated as the average of successful evaluator scores only. "
        "Failed evaluators are excluded from the calculation. If partial=True, this represents "
        "the average of only the successful evaluations, not all requested evaluators.",
    )
    passed: bool = Field(..., description="Whether evaluation passed quality threshold")

    # Error handling
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Errors encountered during evaluation, keyed by evaluator name",
    )
    partial: bool = Field(
        default=False,
        description="Whether this is a partial result (some evaluators failed)",
    )

    # Metadata
    metrics: List[Metric] = Field(
        default_factory=list, description="Metadata about computed metrics"
    )
    evaluator_names: List[str] = Field(
        default_factory=list, description="Names of evaluators used"
    )
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When evaluation completed"
    )

    # LLM interaction tracking for complete transparency
    interactions: List[LLMInteraction] = Field(
        default_factory=list,
        description="All LLM API calls made during evaluation for full transparency",
    )

    # Audit trail
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_score(self, name: str) -> Optional[Score]:
        """Get score by metric name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Score object if found, None otherwise
        """
        for score in self.scores:
            if score.name == name:
                return score
        return None

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric metadata by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Metric object if found, None otherwise
        """
        for metric in self.metrics:
            if metric.name == name:
                return metric
        return None

    def get_interactions_by_purpose(self, purpose: str) -> List[LLMInteraction]:
        """Get all LLM interactions for a specific purpose.

        Args:
            purpose: Purpose to filter by (e.g., 'scoring', 'semantic_comparison')

        Returns:
            List of interactions matching the purpose

        Example:
            >>> # Get all scoring-related LLM calls
            >>> scoring_calls = result.get_interactions_by_purpose("scoring")
            >>> total_tokens = sum(i.tokens_used for i in scoring_calls)
        """
        return [i for i in self.interactions if i.purpose == purpose]

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost with accurate pricing.

        Uses LiteLLM's bundled pricing database for accurate cost calculation.
        Falls back to conservative estimates if pricing data unavailable.

        Args:
            use_actual_pricing: If True, use LiteLLM pricing; if False, use simple estimation

        Returns:
            Total cost in USD

        Example:
            >>> result = await evaluate(output, reference)
            >>> cost = await result.total_llm_cost()
            >>> print(f"Evaluation cost: ${cost:.6f}")
        """
        if not use_actual_pricing:
            # Simple fallback: average $0.02 per 1K tokens
            total_tokens = sum(i.total_tokens for i in self.interactions)
            return (total_tokens / 1000) * 0.02

        # Use cost calculator for accurate pricing
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        total = 0.0
        for interaction in self.interactions:
            total += _get_interaction_cost(interaction)

        return total

    async def cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown by evaluator and model.

        Returns:
            Dictionary with cost breakdowns including:
            - total: Total cost in USD
            - by_evaluator: Cost grouped by evaluator purpose
            - by_model: Cost grouped by model
            - token_breakdown: Token usage details

        Example:
            >>> breakdown = await result.cost_breakdown()
            >>> print(breakdown)
            {
                "total": 0.0105,
                "by_evaluator": {
                    "semantic": 0.0045,
                    "factuality": 0.0060
                },
                "by_model": {
                    "gpt-4o-mini": 0.0105
                },
                "token_breakdown": {
                    "input_tokens": 2000,
                    "output_tokens": 500,
                    "cached_tokens": 100
                }
            }
        """
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        by_evaluator: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        total = 0.0
        total_input = 0
        total_output = 0
        total_cached = 0

        for interaction in self.interactions:
            # Calculate cost using helper
            cost = _get_interaction_cost(interaction)

            # Aggregate by purpose (evaluator)
            purpose = interaction.purpose
            by_evaluator[purpose] = by_evaluator.get(purpose, 0.0) + cost

            # Aggregate by model
            model = interaction.model
            by_model[model] = by_model.get(model, 0.0) + cost

            # Track tokens
            total_input += interaction.input_tokens
            total_output += interaction.output_tokens
            total_cached += interaction.cached_tokens

            total += cost

        return {
            "total": round(total, 6),
            "by_evaluator": {k: round(v, 6) for k, v in by_evaluator.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "token_breakdown": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cached_tokens": total_cached,
                "total_tokens": total_input + total_output,
            },
        }

    def pretty_print(
        self, file: Optional[TextIO] = None, verbose: bool = False
    ) -> None:
        """Print human-readable evaluation summary to terminal.

        Args:
            file: Output stream (default: sys.stdout)
            verbose: If True, include detailed information like token breakdown and interactions

        Example:
            >>> result = await evaluate(output="Paris", reference="Paris is the capital")
            >>> result.pretty_print()
            Evaluation Results
            ==================
            Overall Score: 0.87 ✓ PASSED

            Scores:
              • semantic:   0.92 (confidence: 0.88)
              • factuality: 0.82 (confidence: 0.85)

            Time: 1.23s
            LLM Calls: 2

            >>> # Redirect to file
            >>> with open("results.txt", "w") as f:
            ...     result.pretty_print(file=f)
        """
        import sys

        out = file or sys.stdout

        # Header
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        print("\nEvaluation Results", file=out)
        print("==================", file=out)
        print(f"Overall Score: {self.overall_score:.2f} {status}", file=out)

        # Partial result warning
        if self.partial:
            print("⚠ PARTIAL RESULT (some evaluators failed)", file=out)

        # Individual scores
        if self.scores:
            print("\nScores:", file=out)
            for score in self.scores:
                conf = (
                    f" (confidence: {score.confidence:.2f})"
                    if score.confidence is not None
                    else ""
                )
                print(f"  • {score.name:12s} {score.value:.2f}{conf}", file=out)

        # Errors
        if self.errors:
            print("\nErrors:", file=out)
            for evaluator, error in self.errors.items():
                print(f"  • {evaluator}: {error}", file=out)

        # Basic stats
        print(f"\nTime: {self.processing_time:.2f}s", file=out)
        print(f"LLM Calls: {len(self.interactions)}", file=out)

        # Verbose output
        if verbose:
            print("\nToken Usage:", file=out)
            total_input = sum(i.input_tokens for i in self.interactions)
            total_output = sum(i.output_tokens for i in self.interactions)
            total_cached = sum(i.cached_tokens for i in self.interactions)
            print(f"  Input:  {total_input:,}", file=out)
            print(f"  Output: {total_output:,}", file=out)
            if total_cached > 0:
                print(f"  Cached: {total_cached:,}", file=out)
            print(f"  Total:  {total_input + total_output:,}", file=out)

            if self.interactions:
                print("\nInteractions:", file=out)
                for i, interaction in enumerate(self.interactions, 1):
                    print(
                        f"  {i}. {interaction.purpose} ({interaction.model}): "
                        f"{interaction.input_tokens}→{interaction.output_tokens} tokens, "
                        f"{interaction.latency:.2f}s",
                        file=out,
                    )

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        status = "passed" if self.passed else "failed"
        evaluators = [s.name for s in self.scores]
        return (
            f"<EvaluationResult score={self.overall_score:.2f} {status}=True "
            f"evaluators={evaluators} time={self.processing_time:.2f}s>"
        )

    def summary(self) -> str:
        """Return one-line summary of evaluation result.

        Useful for logging, status updates, and quick inspection.

        Returns:
            Single-line summary string

        Example:
            >>> result.summary()
            '✓ PASSED (0.87) | semantic: 0.92, factuality: 0.82 | 1.23s'
        """
        status = "✓ PASSED" if self.passed else "✗ FAILED"
        scores = ", ".join(f"{s.name}: {s.value:.2f}" for s in self.scores)
        return f"{status} ({self.overall_score:.2f}) | {scores} | {self.processing_time:.2f}s"

    def to_dict(
        self,
        exclude: Optional[set[str]] = None,
        include_timestamp: bool = True,
    ) -> Dict[str, Any]:
        """Export result as dictionary.

        Args:
            exclude: Field names to exclude from output
            include_timestamp: Whether to include timestamp (default True)

        Returns:
            Dictionary representation of the result

        Example:
            >>> data = result.to_dict()
            >>> data = result.to_dict(exclude={'interactions', 'metadata'})
        """
        data = self.model_dump(exclude=exclude or set())
        if not include_timestamp and "timestamp" in data:
            del data["timestamp"]
        return data

    def to_json(
        self,
        indent: Optional[int] = None,
        exclude: Optional[set[str]] = None,
    ) -> str:
        """Export result as JSON string.

        Args:
            indent: Indentation level for pretty printing
            exclude: Field names to exclude from output

        Returns:
            JSON string representation

        Example:
            >>> json_str = result.to_json(indent=2)
            >>> json_str = result.to_json(exclude={'interactions'})
        """
        import json

        data = self.to_dict(exclude=exclude)
        return json.dumps(data, indent=indent, default=str)


class ComparisonResult(BaseModel):
    """Result of comparing two LLM outputs.

    Used for pairwise comparison evaluation where two outputs are compared
    against each other to determine which is better, or if they're equivalent.

    This is different from EvaluationResult which evaluates a single output
    against reference or criteria. ComparisonResult compares two outputs
    directly against each other.

    Example:
        >>> comparison = ComparisonResult(
        ...     output_a="GPT-4 response",
        ...     output_b="Claude response",
        ...     winner="output_a",
        ...     confidence=0.85,
        ...     reasoning="Output A is more accurate and complete",
        ...     aspect_scores={
        ...         "accuracy": {"output_a": 0.9, "output_b": 0.8},
        ...         "clarity": {"output_a": 0.85, "output_b": 0.9},
        ...     }
        ... )
        >>> print(f"Winner: {comparison.winner}")
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Input data
    output_a: str = Field(..., description="First output being compared")
    output_b: str = Field(..., description="Second output being compared")
    reference: Optional[str] = Field(
        None, description="Optional reference context (e.g., user question)"
    )
    criteria: Optional[str] = Field(
        None, description="Optional criteria used for comparison"
    )

    # Comparison results
    winner: Literal["output_a", "output_b", "tie"] = Field(
        ...,
        description="Which output is better, or 'tie' if equivalent",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the comparison decision",
    )
    reasoning: str = Field(
        ..., description="Detailed explanation of why this winner was chosen"
    )
    aspect_scores: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Scores for each aspect/criterion, mapping aspect name to scores for output_a and output_b",
    )

    # Metadata
    total_tokens: int = Field(default=0, description="Total tokens used")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=_utc_now, description="When comparison completed"
    )

    # LLM interaction tracking
    interactions: List[LLMInteraction] = Field(
        default_factory=list,
        description="All LLM API calls made during comparison for full transparency",
    )

    # Audit trail
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_aspect_score(
        self, aspect: str, output: Literal["output_a", "output_b"]
    ) -> Optional[float]:
        """Get score for a specific aspect and output.

        Args:
            aspect: Name of the aspect (e.g., 'accuracy', 'clarity')
            output: Which output to get score for ('output_a' or 'output_b')

        Returns:
            Score value if found, None otherwise

        Example:
            >>> accuracy_a = comparison.get_aspect_score("accuracy", "output_a")
            >>> print(f"Output A accuracy: {accuracy_a}")
        """
        if aspect in self.aspect_scores:
            return self.aspect_scores[aspect].get(output)
        return None

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost with accurate pricing.

        Uses LiteLLM's bundled pricing database for accurate cost calculation.
        Falls back to conservative estimates if pricing data unavailable.

        Args:
            use_actual_pricing: If True, use LiteLLM pricing; if False, use simple estimation

        Returns:
            Total cost in USD

        Example:
            >>> result = await compare(output_a, output_b)
            >>> cost = await result.total_llm_cost()
            >>> print(f"Comparison cost: ${cost:.6f}")
        """
        if not use_actual_pricing:
            # Simple fallback: average $0.02 per 1K tokens
            total_tokens = sum(i.total_tokens for i in self.interactions)
            return (total_tokens / 1000) * 0.02

        # Use cost calculator for accurate pricing
        from arbiter_ai.core.cost_calculator import get_cost_calculator

        calc = get_cost_calculator()
        await calc.ensure_loaded()

        total = 0.0
        for interaction in self.interactions:
            total += _get_interaction_cost(interaction)

        return total

    def pretty_print(
        self, file: Optional[TextIO] = None, verbose: bool = False
    ) -> None:
        """Print human-readable comparison summary to terminal.

        Args:
            file: Output stream (default: sys.stdout)
            verbose: If True, include detailed aspect scores and reasoning

        Example:
            >>> result = await compare(output_a="Response A", output_b="Response B")
            >>> result.pretty_print()
            Comparison Results
            ==================
            Winner: output_a ✓
            Confidence: 0.85

            Reasoning:
            Output A is more accurate and provides better context.

            Time: 1.45s
            LLM Calls: 1

            >>> # With verbose output
            >>> result.pretty_print(verbose=True)
        """
        import sys

        out = file or sys.stdout

        # Header
        winner_symbol = "✓" if self.winner != "tie" else "="
        winner_display = (
            self.winner.replace("_", " ").title() if self.winner != "tie" else "Tie"
        )
        print("\nComparison Results", file=out)
        print("==================", file=out)
        print(f"Winner: {winner_display} {winner_symbol}", file=out)
        print(f"Confidence: {self.confidence:.2f}", file=out)

        # Reasoning (always shown, but truncated if not verbose)
        print("\nReasoning:", file=out)
        if verbose or len(self.reasoning) <= 200:
            print(self.reasoning, file=out)
        else:
            # Truncate long reasoning in non-verbose mode
            print(f"{self.reasoning[:200]}...", file=out)

        # Aspect scores
        if self.aspect_scores and verbose:
            print("\nAspect Scores:", file=out)
            for aspect, scores in self.aspect_scores.items():
                a_score = scores.get("output_a", 0.0)
                b_score = scores.get("output_b", 0.0)
                print(
                    f"  • {aspect:12s} A: {a_score:.2f}  B: {b_score:.2f}",
                    file=out,
                )

        # Basic stats
        print(f"\nTime: {self.processing_time:.2f}s", file=out)
        print(f"LLM Calls: {len(self.interactions)}", file=out)

        # Verbose token breakdown
        if verbose and self.interactions:
            print("\nToken Usage:", file=out)
            total_input = sum(i.input_tokens for i in self.interactions)
            total_output = sum(i.output_tokens for i in self.interactions)
            print(f"  Input:  {total_input:,}", file=out)
            print(f"  Output: {total_output:,}", file=out)
            print(f"  Total:  {total_input + total_output:,}", file=out)

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        return (
            f"<ComparisonResult winner={self.winner!r} "
            f"confidence={self.confidence:.2f} time={self.processing_time:.2f}s>"
        )

    def summary(self) -> str:
        """Return one-line summary of comparison result.

        Useful for logging, status updates, and quick inspection.

        Returns:
            Single-line summary string

        Example:
            >>> comparison.summary()
            'Winner: Output A (0.85 confidence) | 1.45s'
        """
        if self.winner == "tie":
            winner_str = "Tie"
        else:
            winner_str = self.winner.replace("_", " ").title()
        return f"Winner: {winner_str} ({self.confidence:.2f} confidence) | {self.processing_time:.2f}s"

    def to_dict(
        self,
        exclude: Optional[set[str]] = None,
        include_timestamp: bool = True,
    ) -> Dict[str, Any]:
        """Export result as dictionary.

        Args:
            exclude: Field names to exclude from output
            include_timestamp: Whether to include timestamp (default True)

        Returns:
            Dictionary representation of the result

        Example:
            >>> data = comparison.to_dict()
            >>> data = comparison.to_dict(exclude={'interactions', 'metadata'})
        """
        data = self.model_dump(exclude=exclude or set())
        if not include_timestamp and "timestamp" in data:
            del data["timestamp"]
        return data

    def to_json(
        self,
        indent: Optional[int] = None,
        exclude: Optional[set[str]] = None,
    ) -> str:
        """Export result as JSON string.

        Args:
            indent: Indentation level for pretty printing
            exclude: Field names to exclude from output

        Returns:
            JSON string representation

        Example:
            >>> json_str = comparison.to_json(indent=2)
            >>> json_str = comparison.to_json(exclude={'interactions'})
        """
        import json

        data = self.to_dict(exclude=exclude)
        return json.dumps(data, indent=indent, default=str)


class BatchEvaluationResult(BaseModel):
    """Result of batch evaluation operation.

    Contains results for all items, error tracking, and aggregate statistics.
    Provides efficient batch processing while maintaining individual result fidelity.

    Example:
        >>> batch_result = await batch_evaluate(
        ...     items=[
        ...         {"output": "Paris is the capital of France", "reference": "Paris is France's capital"},
        ...         {"output": "Tokyo is the capital of Japan", "reference": "Tokyo is Japan's capital"},
        ...         {"output": "Invalid output"},  # This might fail
        ...     ],
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(f"Success rate: {batch_result.successful_items}/{batch_result.total_items}")
        >>> print(f"Total cost: ${await batch_result.total_llm_cost():.4f}")
        >>>
        >>> # Access individual results
        >>> for i, result in enumerate(batch_result.results):
        ...     if result:  # Check if evaluation succeeded
        ...         print(f"Item {i}: score = {result.overall_score:.2f}")
        ...     else:
        ...         error = next(e for e in batch_result.errors if e['index'] == i)
        ...         print(f"Item {i}: failed - {error['error']}")
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Results (None for failed items, preserving order)
    results: List[Optional[EvaluationResult]] = Field(
        default_factory=list,
        description="Evaluation results in original order. None for failed items.",
    )

    # Error tracking
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Errors that occurred, each with 'index', 'item', and 'error' keys",
    )

    # Statistics
    total_items: int = Field(..., description="Total number of items in batch")
    successful_items: int = Field(..., description="Number of successful evaluations")
    failed_items: int = Field(..., description="Number of failed evaluations")

    # Timing and tokens
    processing_time: float = Field(..., description="Total processing time in seconds")
    total_tokens: int = Field(
        default=0, description="Total tokens across all evaluations"
    )

    # Metadata
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When batch completed",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata and context"
    )

    def get_result(self, index: int) -> Optional[EvaluationResult]:
        """Get result by original item index.

        Args:
            index: Zero-based index of the item in the original batch

        Returns:
            EvaluationResult if successful, None if failed or out of range

        Example:
            >>> result = batch_result.get_result(0)
            >>> if result:
            ...     print(f"Score: {result.overall_score}")
        """
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

    def get_error(self, index: int) -> Optional[Dict[str, Any]]:
        """Get error information for a failed item.

        Args:
            index: Zero-based index of the item in the original batch

        Returns:
            Error dict if item failed, None if successful or not found

        Example:
            >>> error = batch_result.get_error(2)
            >>> if error:
            ...     print(f"Failed: {error['error']}")
        """
        for error in self.errors:
            if error["index"] == index:
                return error
        return None

    async def total_llm_cost(self, use_actual_pricing: bool = True) -> float:
        """Calculate total LLM cost across all successful evaluations.

        Uses LiteLLM's bundled pricing database for accurate cost calculation.

        Args:
            use_actual_pricing: If True, use LiteLLM pricing; if False, use simple estimation

        Returns:
            Total cost in USD across all evaluations

        Example:
            >>> batch_result = await batch_evaluate(items=[...])
            >>> total_cost = await batch_result.total_llm_cost()
            >>> avg_cost_per_item = total_cost / batch_result.total_items
            >>> print(f"Average cost per item: ${avg_cost_per_item:.4f}")
        """
        # Collect costs from all successful results
        successful_results = [r for r in self.results if r is not None]
        if not successful_results:
            return 0.0

        # Calculate cost for each result in parallel
        import asyncio

        costs = await asyncio.gather(
            *[
                r.total_llm_cost(use_actual_pricing=use_actual_pricing)
                for r in successful_results
            ]
        )
        return sum(costs)

    async def cost_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost breakdown across all successful evaluations.

        Returns:
            Dictionary with aggregate cost breakdowns including:
            - total: Total cost across all items
            - per_item_average: Average cost per item
            - by_evaluator: Aggregated cost by evaluator
            - by_model: Aggregated cost by model
            - success_rate: Ratio of successful to total items

        Example:
            >>> breakdown = await batch_result.cost_breakdown()
            >>> print(f"Total: ${breakdown['total']:.4f}")
            >>> print(f"Per item: ${breakdown['per_item_average']:.4f}")
            >>> print(f"Success rate: {breakdown['success_rate']:.1%}")
        """
        successful_results = [r for r in self.results if r is not None]
        if not successful_results:
            return {
                "total": 0.0,
                "per_item_average": 0.0,
                "by_evaluator": {},
                "by_model": {},
                "success_rate": 0.0,
            }

        # Get breakdowns from each result
        import asyncio

        breakdowns = await asyncio.gather(
            *[r.cost_breakdown() for r in successful_results]
        )

        # Aggregate costs by evaluator and model
        by_evaluator: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        total = 0.0

        for breakdown in breakdowns:
            total += breakdown["total"]
            for evaluator, cost in breakdown["by_evaluator"].items():
                by_evaluator[evaluator] = by_evaluator.get(evaluator, 0.0) + cost
            for model, cost in breakdown["by_model"].items():
                by_model[model] = by_model.get(model, 0.0) + cost

        return {
            "total": round(total, 6),
            "per_item_average": (
                round(total / self.total_items, 6) if self.total_items > 0 else 0.0
            ),
            "by_evaluator": {k: round(v, 6) for k, v in by_evaluator.items()},
            "by_model": {k: round(v, 6) for k, v in by_model.items()},
            "success_rate": (
                self.successful_items / self.total_items
                if self.total_items > 0
                else 0.0
            ),
        }

    def pretty_print(
        self, file: Optional[TextIO] = None, verbose: bool = False
    ) -> None:
        """Print human-readable batch evaluation summary to terminal.

        Args:
            file: Output stream (default: sys.stdout)
            verbose: If True, include per-item results and detailed statistics

        Example:
            >>> result = await batch_evaluate(items=[...], evaluators=["semantic"])
            >>> result.pretty_print()
            Batch Evaluation Results
            ========================
            Success: 95/100 (95.0%)
            Failed:  5/100

            Statistics:
              Mean:   0.85
              Std:    0.12
              Median: 0.87
              Range:  0.45 - 0.98

            Total Time: 45.2s
            Total Tokens: 125,430

            >>> # With verbose output showing individual results
            >>> result.pretty_print(verbose=True)
        """
        import sys

        out = file or sys.stdout

        # Header
        print("\nBatch Evaluation Results", file=out)
        print("========================", file=out)
        success_pct = (
            (self.successful_items / self.total_items * 100)
            if self.total_items > 0
            else 0.0
        )
        print(
            f"Success: {self.successful_items}/{self.total_items} ({success_pct:.1f}%)",
            file=out,
        )
        if self.failed_items > 0:
            print(f"Failed:  {self.failed_items}/{self.total_items}", file=out)

        # Statistics from successful results
        successful_results = [r for r in self.results if r is not None]
        if successful_results:
            scores = [r.overall_score for r in successful_results]
            mean_score = sum(scores) / len(scores)

            # Calculate std dev
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance**0.5

            # Calculate median
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            if n % 2 == 0:
                median = (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            else:
                median = sorted_scores[n // 2]

            print("\nStatistics:", file=out)
            print(f"  Mean:   {mean_score:.2f}", file=out)
            print(f"  Std:    {std_dev:.2f}", file=out)
            print(f"  Median: {median:.2f}", file=out)
            print(
                f"  Range:  {min(scores):.2f} - {max(scores):.2f}",
                file=out,
            )

            # Pass rate (assuming threshold of 0.7 if any result has passed field)
            if successful_results and hasattr(successful_results[0], "passed"):
                passed_count = sum(1 for r in successful_results if r.passed)
                pass_rate = (passed_count / len(successful_results)) * 100
                print(f"\nPass Rate: {pass_rate:.1f}%", file=out)

        # Overall stats
        print(f"\nTotal Time: {self.processing_time:.1f}s", file=out)
        if self.total_tokens > 0:
            print(f"Total Tokens: {self.total_tokens:,}", file=out)

        # Verbose: show individual results
        if verbose and successful_results:
            print("\nIndividual Results:", file=out)
            for i, result in enumerate(self.results):
                if result is not None:
                    status = "✓" if result.passed else "✗"
                    print(
                        f"  [{i:3d}] {status} Score: {result.overall_score:.2f} "
                        f"({result.processing_time:.2f}s)",
                        file=out,
                    )
                else:
                    error = self.get_error(i)
                    error_msg = error["error"] if error else "Unknown error"
                    # Truncate long error messages
                    if len(error_msg) > 50:
                        error_msg = error_msg[:50] + "..."
                    print(f"  [{i:3d}] ✗ Failed: {error_msg}", file=out)

        # Show errors summary if not verbose
        elif self.errors and not verbose:
            print("\nErrors: (use verbose=True for details)", file=out)
            # Group errors by type
            error_types: Dict[str, int] = {}
            for error in self.errors:
                error_msg = str(error.get("error", "Unknown"))
                # Extract first line or first 50 chars
                error_key = error_msg.split("\n")[0][:50]
                error_types[error_key] = error_types.get(error_key, 0) + 1

            # Show top 5
            for error_type, count in list(error_types.items())[:5]:
                print(f"  • {error_type}: {count}x", file=out)

    def __repr__(self) -> str:
        """Return concise string representation for debugging."""
        return (
            f"<BatchEvaluationResult items={self.total_items} "
            f"success={self.successful_items} failed={self.failed_items} "
            f"time={self.processing_time:.1f}s>"
        )

    def summary(self) -> str:
        """Return one-line summary of batch results.

        Useful for logging, status updates, and quick inspection.

        Returns:
            Single-line summary string

        Example:
            >>> batch.summary()
            '95/100 passed (95.0%) | mean: 0.85, median: 0.87 | 45.2s'
        """
        pct = (
            (self.successful_items / self.total_items * 100)
            if self.total_items > 0
            else 0
        )

        # Calculate stats from successful results
        successful = [r for r in self.results if r is not None]
        if successful:
            scores = [r.overall_score for r in successful]
            mean = sum(scores) / len(scores)
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            median = (
                sorted_scores[n // 2]
                if n % 2
                else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
            )
            stats = f"mean: {mean:.2f}, median: {median:.2f}"
        else:
            stats = "no results"

        return f"{self.successful_items}/{self.total_items} passed ({pct:.1f}%) | {stats} | {self.processing_time:.1f}s"

    def to_dict(
        self,
        exclude: Optional[set[str]] = None,
        include_timestamp: bool = True,
    ) -> Dict[str, Any]:
        """Export result as dictionary.

        Args:
            exclude: Field names to exclude from output
            include_timestamp: Whether to include timestamp (default True)

        Returns:
            Dictionary representation of the result

        Example:
            >>> data = batch.to_dict()
            >>> data = batch.to_dict(exclude={'results', 'metadata'})
        """
        data = self.model_dump(exclude=exclude or set())
        if not include_timestamp and "timestamp" in data:
            del data["timestamp"]
        return data

    def to_json(
        self,
        path: Optional[str] = None,
        indent: Optional[int] = 2,
        exclude: Optional[set[str]] = None,
    ) -> Optional[str]:
        """Export result as JSON string or write to file.

        Args:
            path: Optional file path to write JSON to. If provided, writes to file
                and returns None. If not provided, returns JSON string.
            indent: Indentation level for pretty printing (default: 2)
            exclude: Field names to exclude from output

        Returns:
            JSON string if path is None, otherwise None (writes to file)

        Example:
            >>> # Get JSON string
            >>> json_str = batch.to_json()
            >>>
            >>> # Write to file
            >>> batch.to_json("results.json")
        """
        import json

        # Build export structure matching issue #40 spec
        successful = [r for r in self.results if r is not None]
        scores = [r.overall_score for r in successful] if successful else []

        export_data: Dict[str, Any] = {
            "summary": {
                "total_items": self.total_items,
                "successful": self.successful_items,
                "failed": self.failed_items,
                "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "processing_time": round(self.processing_time, 3),
            },
            "results": [
                (
                    {
                        "index": i,
                        "output": (
                            r.output[:200] + "..." if len(r.output) > 200 else r.output
                        ),
                        "reference": (
                            r.reference[:200] + "..."
                            if r.reference and len(r.reference) > 200
                            else r.reference
                        ),
                        "overall_score": round(r.overall_score, 4),
                        "passed": r.passed,
                        "scores": {s.name: round(s.value, 4) for s in r.scores},
                    }
                    if r is not None
                    else {"index": i, "error": self.get_error(i)}
                )
                for i, r in enumerate(self.results)
            ],
            "errors": self.errors,
            "metadata": self.metadata,
        }

        # Apply exclusions if specified
        if exclude:
            export_data = {k: v for k, v in export_data.items() if k not in exclude}

        json_str = json.dumps(export_data, indent=indent, default=str)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return None

        return json_str

    def to_records(self) -> List[Dict[str, Any]]:
        """Export results as list of flat dictionaries for tabular processing.

        Each record represents one evaluation item with flattened score columns.
        Failed items include error information instead of scores.

        Returns:
            List of dictionaries, one per item in original order

        Example:
            >>> records = batch.to_records()
            >>> for r in records:
            ...     print(f"Item {r['index']}: {r.get('overall_score', 'FAILED')}")
        """
        records: List[Dict[str, Any]] = []

        for i, result in enumerate(self.results):
            if result is not None:
                record: Dict[str, Any] = {
                    "index": i,
                    "output": result.output,
                    "reference": result.reference,
                    "overall_score": result.overall_score,
                    "passed": result.passed,
                    "processing_time": result.processing_time,
                }
                # Flatten scores as individual columns
                for score in result.scores:
                    record[f"{score.name}_score"] = score.value
                    if score.confidence is not None:
                        record[f"{score.name}_confidence"] = score.confidence
                records.append(record)
            else:
                # Failed item
                error_info = self.get_error(i)
                record = {
                    "index": i,
                    "output": (
                        error_info.get("item", {}).get("output") if error_info else None
                    ),
                    "reference": (
                        error_info.get("item", {}).get("reference")
                        if error_info
                        else None
                    ),
                    "overall_score": None,
                    "passed": None,
                    "processing_time": None,
                    "error": error_info.get("error") if error_info else "Unknown error",
                }
                records.append(record)

        return records

    def to_csv(self, path: str, include_header: bool = True) -> None:
        """Export results to CSV file.

        Creates a flat CSV with one row per evaluation item. Score columns
        are dynamically named based on evaluator names (e.g., semantic_score).

        Args:
            path: File path to write CSV to
            include_header: Whether to include header row (default: True)

        Example:
            >>> batch.to_csv("results.csv")
            >>> batch.to_csv("results.csv", include_header=False)
        """
        import csv

        records = self.to_records()
        if not records:
            # Write empty file with basic headers
            with open(path, "w", newline="", encoding="utf-8") as f:
                empty_writer = csv.writer(f)
                if include_header:
                    empty_writer.writerow(
                        [
                            "index",
                            "output",
                            "reference",
                            "overall_score",
                            "passed",
                            "error",
                        ]
                    )
            return

        # Collect all unique keys across all records for consistent columns
        all_keys: set[str] = set()
        for record in records:
            all_keys.update(record.keys())

        # Order columns: fixed columns first, then dynamic score columns, then error
        fixed_cols = [
            "index",
            "output",
            "reference",
            "overall_score",
            "passed",
            "processing_time",
        ]
        score_cols = sorted(
            [k for k in all_keys if k.endswith("_score") and k != "overall_score"]
        )
        confidence_cols = sorted([k for k in all_keys if k.endswith("_confidence")])
        other_cols = ["error"] if "error" in all_keys else []

        fieldnames = fixed_cols + score_cols + confidence_cols + other_cols

        with open(path, "w", newline="", encoding="utf-8") as f:
            dict_writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            if include_header:
                dict_writer.writeheader()
            dict_writer.writerows(records)

    def to_dataframe(self) -> "Any":
        """Export results as pandas DataFrame.

        Requires pandas to be installed. Returns a DataFrame with one row
        per evaluation item, suitable for analysis and visualization.

        Returns:
            pandas.DataFrame with evaluation results

        Raises:
            ImportError: If pandas is not installed

        Example:
            >>> df = batch.to_dataframe()
            >>> print(df.describe())
            >>> df.to_excel("results.xlsx")
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        records = self.to_records()
        return pd.DataFrame(records)
