"""Instruction following evaluator for validating LLM output adherence to constraints.

This evaluator measures how well an LLM output follows given instructions or
constraints. It is critical for agents, tool use, structured outputs, and any
pipeline with specific requirements.

## When to Use:

- Validating agent outputs follow specific formats (JSON, markdown, etc.)
- Checking structured output compliance
- Ensuring content constraints are met (required fields, length limits)
- Verifying tone/style requirements
- Testing LLM pipelines with explicit instructions

## Example:

    >>> evaluator = InstructionFollowingEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output='{"name": "Alice", "age": 30}',
    ...     criteria="Respond in valid JSON. Include 'name' and 'age' fields."
    ... )
    >>> print(f"Score: {score.value:.2f}")
    >>> print(f"Violations: {score.metadata.get('instructions_violated', [])}")

## Evaluation Aspects:

The evaluator assesses:
- Format requirements (JSON, markdown, bullet points, etc.)
- Content constraints (required fields, topics to include/exclude)
- Length/word count limits
- Tone/style requirements
- Any explicit instructions provided in criteria
"""

from typing import List, Literal, Optional, Type, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.models import Score
from .base import BasePydanticEvaluator

__all__ = ["InstructionFollowingEvaluator", "InstructionFollowingResponse"]


ViolationSeverity = Literal["none", "minor", "major", "critical"]


class InstructionFollowingResponse(BaseModel):
    """Structured response for instruction following evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Instruction adherence score (0=ignored instructions, 1=fully followed)"
        ),
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this instruction adherence assessment",
    )
    explanation: str = Field(
        ..., description="Detailed explanation of how well instructions were followed"
    )
    instructions_followed: List[str] = Field(
        default_factory=list,
        description="List of instructions that were successfully followed",
    )
    instructions_violated: List[str] = Field(
        default_factory=list,
        description="List of instructions that were not followed",
    )
    instructions_partially_met: List[str] = Field(
        default_factory=list,
        description="List of instructions that were only partially followed",
    )
    violation_severity: ViolationSeverity = Field(
        default="none",
        description=(
            "Overall severity of violations: "
            "'none' (all followed), "
            "'minor' (small deviations), "
            "'major' (significant issues), "
            "'critical' (fundamental violations)"
        ),
    )

    @field_validator("explanation")
    @classmethod
    def validate_explanation_quality(cls, v: str) -> str:
        """Ensure explanation is meaningful and not empty."""
        if len(v.strip()) < 1:
            raise ValueError("Explanation cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_instruction_analysis(self) -> "InstructionFollowingResponse":
        """Ensure violation severity matches instruction counts."""
        has_violations = len(self.instructions_violated) > 0
        has_partial = len(self.instructions_partially_met) > 0

        # If no violations or partial, severity should be 'none' or possibly 'minor'
        if (
            not has_violations
            and not has_partial
            and self.violation_severity
            not in (
                "none",
                "minor",
            )
        ):
            raise ValueError(
                f"Violation severity '{self.violation_severity}' is inconsistent "
                "with no violated or partially met instructions"
            )

        # If violations exist but severity is 'none', that's inconsistent
        if has_violations and self.violation_severity == "none":
            raise ValueError(
                "Violation severity cannot be 'none' when instructions are violated"
            )

        return self


class InstructionFollowingEvaluator(BasePydanticEvaluator):
    """Evaluates how well LLM output follows given instructions.

    This evaluator analyzes whether an output adheres to specified instructions
    or constraints. It is essential for validating agent behavior, structured
    outputs, and any LLM pipeline with explicit requirements.

    The evaluator:
    - Identifies each instruction/constraint in the criteria
    - Checks if the output follows each instruction
    - Categorizes instructions as followed, violated, or partially met
    - Assesses the severity of any violations
    - Provides detailed explanation of adherence/violations

    Example:
        >>> from arbiter_ai import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = InstructionFollowingEvaluator(client)
        >>>
        >>> # Check JSON format compliance
        >>> score = await evaluator.evaluate(
        ...     output='{"name": "Alice", "age": 30}',
        ...     criteria="Respond in valid JSON. Include 'name' and 'age' fields."
        ... )
        >>> print(f"Score: {score.value:.2f}")
        >>>
        >>> # Check multiple constraints
        >>> score = await evaluator.evaluate(
        ...     output="Here is a brief summary of the topic...",
        ...     criteria=(
        ...         "1. Keep response under 100 words. "
        ...         "2. Use bullet points. "
        ...         "3. Include a conclusion."
        ...     )
        ... )
        >>> print(f"Followed: {score.metadata.get('instructions_followed', [])}")
        >>> print(f"Violated: {score.metadata.get('instructions_violated', [])}")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "instruction_following"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining instruction following evaluation approach."""
        return """You are an expert evaluator specializing in assessing instruction adherence.

Your task is to analyze how well a given output follows specified instructions or constraints.

For each instruction/constraint, determine:
1. Whether it was FOLLOWED (fully complied with)
2. Whether it was VIOLATED (not followed)
3. Whether it was PARTIALLY MET (some compliance but not complete)

Consider these types of instructions:
- Format requirements (JSON, markdown, bullet points, numbered lists, etc.)
- Content constraints (required fields, topics to include/exclude, specific information)
- Length/word count limits (maximum or minimum lengths)
- Tone/style requirements (formal, casual, technical, etc.)
- Structural requirements (sections, headings, ordering)
- Explicit prohibitions (things to avoid or not mention)

Scoring guidelines:
- 1.0: All instructions fully followed
- 0.8-0.99: Minor deviations that don't affect core compliance
- 0.5-0.79: Some instructions followed, some violated or partially met
- 0.2-0.49: Significant violations but some attempt at compliance
- 0.0-0.19: Instructions largely ignored

Severity levels:
- "none": All instructions followed
- "minor": Small deviations (e.g., slightly over word limit, minor formatting issues)
- "major": Significant issues (e.g., wrong format, missing required content)
- "critical": Fundamental violations (e.g., completely wrong output type, safety violations)

Be thorough and precise in your analysis. List each distinct instruction and its status."""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not criteria:
            raise ValueError(
                "InstructionFollowingEvaluator requires criteria to be provided. "
                "The criteria parameter should contain the instructions to check against."
            )

        prompt_parts = [f"OUTPUT TO EVALUATE:\n{output}\n"]

        if reference:
            prompt_parts.append(
                f"REFERENCE/CONTEXT (for additional context, not instructions):\n"
                f"{reference}\n"
            )

        prompt_parts.append(f"INSTRUCTIONS TO CHECK:\n{criteria}\n")

        prompt_parts.append(
            """Analyze how well the output follows the specified instructions.

For your response:
1. List each instruction that was successfully FOLLOWED
2. List each instruction that was VIOLATED (not followed)
3. List each instruction that was PARTIALLY MET
4. Provide an overall score (0.0 to 1.0) reflecting instruction adherence
5. Assess the violation severity ("none", "minor", "major", or "critical")
6. Explain your reasoning in detail

Be specific about which parts of the output correspond to which instructions."""
        )

        return "\n".join(prompt_parts)

    def _get_response_type(self) -> Type[BaseModel]:
        """Use instruction following response model."""
        return InstructionFollowingResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from instruction following response."""
        instruction_response = cast(InstructionFollowingResponse, response)

        # Build detailed explanation
        explanation_parts = [instruction_response.explanation]

        if instruction_response.instructions_followed:
            explanation_parts.append(
                "\n\nInstructions Followed:\n- "
                + "\n- ".join(instruction_response.instructions_followed)
            )

        if instruction_response.instructions_partially_met:
            explanation_parts.append(
                "\n\nInstructions Partially Met:\n- "
                + "\n- ".join(instruction_response.instructions_partially_met)
            )

        if instruction_response.instructions_violated:
            explanation_parts.append(
                "\n\nInstructions Violated:\n- "
                + "\n- ".join(instruction_response.instructions_violated)
            )

        full_explanation = "".join(explanation_parts)

        return Score(
            name=self.name,
            value=instruction_response.score,
            confidence=instruction_response.confidence,
            explanation=full_explanation,
            metadata={
                "instructions_followed": instruction_response.instructions_followed,
                "instructions_violated": instruction_response.instructions_violated,
                "instructions_partially_met": (
                    instruction_response.instructions_partially_met
                ),
                "followed_count": len(instruction_response.instructions_followed),
                "violated_count": len(instruction_response.instructions_violated),
                "partial_count": len(instruction_response.instructions_partially_met),
                "violation_severity": instruction_response.violation_severity,
            },
        )
