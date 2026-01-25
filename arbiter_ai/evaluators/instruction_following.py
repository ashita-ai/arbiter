"""Instruction following evaluator for agent/pipeline validation.

This evaluator measures how well an LLM output adheres to given instructions
or constraints. Critical for agents, tool use, structured outputs, and any
pipeline with specific requirements.

## When to Use:

- Validating agent pipelines follow instructions
- Checking structured output compliance (JSON format, required fields)
- Verifying length/word count limits are respected
- Ensuring tone/style requirements are met
- Any scenario where explicit instructions must be followed

## Example:

    >>> evaluator = InstructionFollowingEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Here is the answer in JSON format: {...}",
    ...     criteria="Respond in valid JSON. Include 'name' and 'age' fields. Keep response under 100 words."
    ... )
    >>> print(f"Score: {score.value:.2f}")
    >>> print(f"Instructions followed: {score.metadata.get('instructions_followed', [])}")
    >>> print(f"Instructions violated: {score.metadata.get('instructions_violated', [])}")
"""

from typing import List, Literal, Optional, Type, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.models import Score
from .base import BasePydanticEvaluator

__all__ = ["InstructionFollowingEvaluator", "InstructionFollowingResponse"]


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
        description="Overall score (1.0 = fully followed, 0.0 = ignored instructions)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this evaluation",
    )
    explanation: str = Field(
        ..., description="Detailed explanation of the evaluation and reasoning"
    )
    instructions_followed: List[str] = Field(
        default_factory=list,
        description="List of instructions that the output successfully follows",
    )
    instructions_violated: List[str] = Field(
        default_factory=list,
        description="List of instructions that the output violates or ignores",
    )
    instructions_partially_met: List[str] = Field(
        default_factory=list,
        description="List of instructions that are only partially followed",
    )
    violation_severity: Literal["none", "minor", "major", "critical"] = Field(
        default="none",
        description="Overall severity of violations (none, minor, major, critical)",
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
        """Ensure instructions are identified for non-trivial assessments."""
        total_instructions = (
            len(self.instructions_followed)
            + len(self.instructions_violated)
            + len(self.instructions_partially_met)
        )
        # High confidence scores with non-extreme values should have instructions
        if (
            self.confidence > 0.7
            and total_instructions == 0
            and self.score not in (0.0, 1.0)
        ):
            raise ValueError(
                "High confidence evaluations require at least one instruction "
                "to be identified as followed, violated, or partially met"
            )
        return self

    @model_validator(mode="after")
    def validate_severity_consistency(self) -> "InstructionFollowingResponse":
        """Ensure violation severity is consistent with violations found."""
        has_violations = len(self.instructions_violated) > 0
        if has_violations and self.violation_severity == "none":
            raise ValueError(
                "violation_severity cannot be 'none' when instructions_violated is not empty"
            )
        if not has_violations and self.violation_severity != "none":
            # Allow partial violations to have non-none severity
            if len(self.instructions_partially_met) == 0:
                raise ValueError(
                    "violation_severity must be 'none' when no instructions "
                    "are violated or partially met"
                )
        return self


class InstructionFollowingEvaluator(BasePydanticEvaluator):
    """Evaluates how well outputs follow explicit instructions.

    This evaluator assesses whether LLM outputs adhere to given instructions
    or constraints. It is essential for validating agent pipelines, structured
    output compliance, and any scenario requiring instruction adherence.

    The evaluator analyzes:
    - Format requirements (JSON, markdown, bullet points, etc.)
    - Content constraints (required fields, topics to include/exclude)
    - Length/word count limits
    - Tone/style requirements
    - Any explicit instructions provided in criteria

    Example:
        >>> from arbiter_ai import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = InstructionFollowingEvaluator(client)
        >>>
        >>> score = await evaluator.evaluate(
        ...     output='{"name": "John", "age": 30}',
        ...     criteria="Respond in valid JSON. Include 'name' and 'age' fields."
        ... )
        >>>
        >>> print(f"Score: {score.value:.2f}")
        >>> print(f"Followed: {score.metadata.get('instructions_followed', [])}")
        >>> print(f"Violated: {score.metadata.get('instructions_violated', [])}")
        >>> print(f"Severity: {score.metadata.get('violation_severity', 'none')}")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "instruction_following"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining instruction following evaluation approach."""
        return """You are an expert evaluator specializing in assessing how well text outputs follow explicit instructions.

Your task is to evaluate whether a given output adheres to specified instructions or constraints. You should:

1. Parse and identify each individual instruction from the criteria
2. Systematically check each instruction against the output
3. Categorize each instruction as: followed, violated, or partially met
4. Assess the overall severity of any violations
5. Provide a score from 0.0 (completely ignores instructions) to 1.0 (fully follows all instructions)

Instruction types to look for:
- FORMAT: Output format requirements (JSON, markdown, lists, etc.)
- CONTENT: Required or prohibited content (specific fields, topics, keywords)
- LENGTH: Word count, character limits, conciseness requirements
- STYLE: Tone, formality, voice requirements
- STRUCTURE: Organization, ordering, hierarchy requirements

Severity levels:
- none: All instructions followed perfectly
- minor: Small deviations that don't significantly impact the output's usefulness
- major: Significant violations that reduce the output's usefulness
- critical: Fundamental violations that make the output unsuitable for its purpose

Be thorough, precise, and fair. Consider the intent behind each instruction, not just literal compliance.
Provide clear, actionable feedback about which instructions were followed or violated."""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not criteria:
            raise ValueError(
                "InstructionFollowingEvaluator requires criteria (instructions) to be provided. "
                "Use the criteria parameter to specify the instructions to check against."
            )

        prompt_parts = [f"OUTPUT TO EVALUATE:\n{output}\n"]

        if reference:
            prompt_parts.append(f"REFERENCE CONTEXT:\n{reference}\n")

        prompt_parts.append(f"INSTRUCTIONS TO CHECK:\n{criteria}\n")

        prompt_parts.append(
            """Evaluate how well the output follows the specified instructions. For each instruction:
1. Identify whether it was followed, violated, or partially met
2. Note any specific evidence from the output

Then provide:
- An overall score (0.0 to 1.0) based on instruction adherence
- Your confidence in this assessment
- A detailed explanation of your findings
- Lists of instructions followed, violated, and partially met
- The overall severity of any violations (none, minor, major, critical)"""
        )

        return "\n".join(prompt_parts)

    def _get_response_type(self) -> Type[BaseModel]:
        """Use instruction following response model."""
        return InstructionFollowingResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from instruction following response."""
        instruction_response = cast(InstructionFollowingResponse, response)

        return Score(
            name=self.name,
            value=instruction_response.score,
            confidence=instruction_response.confidence,
            explanation=instruction_response.explanation,
            metadata={
                "instructions_followed": instruction_response.instructions_followed,
                "instructions_violated": instruction_response.instructions_violated,
                "instructions_partially_met": instruction_response.instructions_partially_met,
                "instructions_followed_count": len(
                    instruction_response.instructions_followed
                ),
                "instructions_violated_count": len(
                    instruction_response.instructions_violated
                ),
                "instructions_partially_met_count": len(
                    instruction_response.instructions_partially_met
                ),
                "violation_severity": instruction_response.violation_severity,
            },
        )
