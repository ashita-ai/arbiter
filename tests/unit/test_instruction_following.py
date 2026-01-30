"""Unit tests for InstructionFollowingEvaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from arbiter_ai.core.exceptions import EvaluatorError
from arbiter_ai.evaluators.instruction_following import (
    InstructionFollowingEvaluator,
    InstructionFollowingResponse,
)
from tests.conftest import MockAgentResult


@pytest.fixture
def evaluator(mock_llm_client):
    """Create an InstructionFollowingEvaluator instance."""
    return InstructionFollowingEvaluator(llm_client=mock_llm_client)


class TestInstructionFollowingEvaluator:
    """Test suite for InstructionFollowingEvaluator."""

    def test_name_property(self, evaluator):
        """Test that evaluator has correct name."""
        assert evaluator.name == "instruction_following"

    def test_system_prompt(self, evaluator):
        """Test that system prompt is well-formed."""
        prompt = evaluator._get_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "instruction" in prompt.lower()
        assert "followed" in prompt.lower()
        assert "violated" in prompt.lower()

    def test_user_prompt_with_criteria(self, evaluator):
        """Test user prompt generation with criteria."""
        output = '{"name": "Alice", "age": 30}'
        criteria = "Respond in valid JSON. Include 'name' and 'age' fields."
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "OUTPUT" in prompt
        assert "INSTRUCTIONS" in prompt

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference context."""
        output = "Summary of the document"
        reference = "Original document content"
        criteria = "Summarize in under 50 words"
        prompt = evaluator._get_user_prompt(output, reference, criteria)

        assert output in prompt
        assert reference in prompt
        assert criteria in prompt
        assert "REFERENCE" in prompt or "CONTEXT" in prompt

    def test_user_prompt_requires_criteria(self, evaluator):
        """Test that user prompt requires criteria."""
        with pytest.raises(ValueError) as exc_info:
            evaluator._get_user_prompt("test output", None, None)

        assert "criteria" in str(exc_info.value).lower()

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == InstructionFollowingResponse

    @pytest.mark.asyncio
    async def test_compute_score_all_followed(self, evaluator):
        """Test score computation when all instructions are followed."""
        response = InstructionFollowingResponse(
            score=1.0,
            confidence=0.95,
            explanation="All instructions were followed correctly",
            instructions_followed=["Valid JSON format", "Contains name field"],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="none",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "instruction_following"
        assert score.value == 1.0
        assert score.confidence == 0.95
        assert "All instructions were followed" in score.explanation
        assert "Instructions Followed" in score.explanation
        assert score.metadata["followed_count"] == 2
        assert score.metadata["violated_count"] == 0
        assert score.metadata["violation_severity"] == "none"

    @pytest.mark.asyncio
    async def test_compute_score_with_violations(self, evaluator):
        """Test score computation with instruction violations."""
        response = InstructionFollowingResponse(
            score=0.4,
            confidence=0.88,
            explanation="Some instructions were not followed",
            instructions_followed=["Contains name field"],
            instructions_violated=["Invalid JSON format", "Missing age field"],
            instructions_partially_met=[],
            violation_severity="major",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "instruction_following"
        assert score.value == 0.4
        assert score.confidence == 0.88
        assert "Instructions Violated" in score.explanation
        assert score.metadata["followed_count"] == 1
        assert score.metadata["violated_count"] == 2
        assert score.metadata["violation_severity"] == "major"

    @pytest.mark.asyncio
    async def test_compute_score_with_partial(self, evaluator):
        """Test score computation with partially met instructions."""
        response = InstructionFollowingResponse(
            score=0.7,
            confidence=0.82,
            explanation="Mixed compliance with instructions",
            instructions_followed=["Used JSON format"],
            instructions_violated=[],
            instructions_partially_met=["Word limit almost met"],
            violation_severity="minor",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "instruction_following"
        assert score.value == 0.7
        assert "Instructions Partially Met" in score.explanation
        assert score.metadata["partial_count"] == 1
        assert score.metadata["violation_severity"] == "minor"

    @pytest.mark.asyncio
    async def test_evaluate_with_criteria(self, evaluator, mock_agent):
        """Test evaluation with criteria."""
        mock_response = InstructionFollowingResponse(
            score=0.92,
            confidence=0.9,
            explanation="Instructions well followed",
            instructions_followed=["JSON format", "Contains required fields"],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="none",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output='{"name": "Alice", "age": 30}',
            criteria="Respond in valid JSON. Include 'name' and 'age' fields.",
        )

        assert score.value == 0.92
        assert score.confidence == 0.9
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].purpose == "instruction_following_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_without_criteria_raises(self, evaluator, mock_agent):
        """Test that evaluation without criteria raises ValueError."""
        mock_agent.run = AsyncMock()
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(output='{"name": "Alice"}')

        assert "criteria" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("LLM API error"))
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError) as exc_info:
            await evaluator.evaluate(
                output="Test output",
                criteria="Test criteria",
            )

        assert "instruction_following" in str(exc_info.value)
        assert "Evaluation failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = InstructionFollowingResponse(
            score=0.85,
            confidence=0.88,
            explanation="Test explanation",
            instructions_followed=["Test instruction"],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="none",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        await evaluator.evaluate(
            output="Test output",
            criteria="Test criteria",
        )

        assert len(evaluator.interactions) == 1
        interaction = evaluator.interactions[0]
        assert interaction.model == "gpt-4o-mini"
        assert interaction.purpose == "instruction_following_evaluation"
        assert interaction.tokens_used == 100
        assert interaction.metadata["evaluator"] == "instruction_following"
        assert interaction.metadata["has_reference"] is False
        assert interaction.metadata["has_criteria"] is True

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference context."""
        mock_response = InstructionFollowingResponse(
            score=0.75,
            confidence=0.8,
            explanation="Instructions mostly followed",
            instructions_followed=["Summarized content"],
            instructions_violated=[],
            instructions_partially_met=["Length slightly over"],
            violation_severity="minor",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Brief summary of the document",
            reference="Full document text here",
            criteria="Summarize in under 10 words",
        )

        assert score.value == 0.75
        assert len(evaluator.interactions) == 1
        assert evaluator.interactions[0].metadata["has_reference"] is True


class TestInstructionFollowingResponse:
    """Test suite for InstructionFollowingResponse model."""

    def test_response_creation(self):
        """Test creating an InstructionFollowingResponse."""
        response = InstructionFollowingResponse(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            instructions_followed=["Instruction 1"],
            instructions_violated=["Instruction 2"],
            instructions_partially_met=[],
            violation_severity="minor",
        )

        assert response.score == 0.85
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert response.instructions_followed == ["Instruction 1"]
        assert response.instructions_violated == ["Instruction 2"]
        assert response.violation_severity == "minor"

    def test_response_defaults(self):
        """Test InstructionFollowingResponse default values."""
        response = InstructionFollowingResponse(
            score=0.8,
            explanation="Test",
        )

        assert response.score == 0.8
        assert response.confidence == 0.85  # Default
        assert response.instructions_followed == []
        assert response.instructions_violated == []
        assert response.instructions_partially_met == []
        assert response.violation_severity == "none"

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        InstructionFollowingResponse(score=0.0, explanation="test")
        InstructionFollowingResponse(score=1.0, explanation="test")
        InstructionFollowingResponse(score=0.5, explanation="test")

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            InstructionFollowingResponse(score=-0.1, explanation="test")

        with pytest.raises(Exception):  # Pydantic validation error
            InstructionFollowingResponse(score=1.1, explanation="test")

    def test_response_severity_validation(self):
        """Test that violation severity must be valid."""
        # Valid severities
        InstructionFollowingResponse(
            score=1.0, explanation="test", violation_severity="none"
        )
        InstructionFollowingResponse(
            score=0.8,
            explanation="test",
            violation_severity="minor",
            instructions_partially_met=["Some partial"],
        )
        InstructionFollowingResponse(
            score=0.5,
            explanation="test",
            violation_severity="major",
            instructions_violated=["Some violation"],
        )
        InstructionFollowingResponse(
            score=0.1,
            explanation="test",
            violation_severity="critical",
            instructions_violated=["Critical violation"],
        )

        # Invalid severity
        with pytest.raises(Exception):  # Pydantic validation error
            InstructionFollowingResponse(
                score=0.5, explanation="test", violation_severity="invalid"
            )

    def test_response_explanation_required(self):
        """Test that explanation cannot be empty."""
        with pytest.raises(Exception):  # Pydantic validation error
            InstructionFollowingResponse(score=0.5, explanation="")

        with pytest.raises(Exception):  # Pydantic validation error
            InstructionFollowingResponse(score=0.5, explanation="   ")

    def test_severity_consistency_validation(self):
        """Test that violation severity is consistent with instruction counts."""
        # Valid: no violations with 'none' severity
        InstructionFollowingResponse(
            score=1.0,
            explanation="All good",
            instructions_followed=["All followed"],
            instructions_violated=[],
            violation_severity="none",
        )

        # Invalid: violations exist but severity is 'none'
        with pytest.raises(ValueError) as exc_info:
            InstructionFollowingResponse(
                score=0.5,
                explanation="Has violations",
                instructions_followed=[],
                instructions_violated=["Something violated"],
                violation_severity="none",
            )
        assert "none" in str(exc_info.value).lower()

    def test_severity_allows_minor_without_violations(self):
        """Test that 'minor' severity is allowed even without explicit violations."""
        # This is valid because minor issues might not be explicitly listed
        response = InstructionFollowingResponse(
            score=0.9,
            explanation="Minor formatting issues",
            instructions_followed=["Main instruction"],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="minor",
        )
        assert response.violation_severity == "minor"

    def test_severity_disallows_major_without_any_issues(self):
        """Test that major/critical requires violations or partial compliance."""
        with pytest.raises(ValueError):
            InstructionFollowingResponse(
                score=0.9,
                explanation="Claims major issues but lists none",
                instructions_followed=["All followed"],
                instructions_violated=[],
                instructions_partially_met=[],
                violation_severity="major",
            )


class TestInstructionFollowingRegistry:
    """Test that InstructionFollowingEvaluator is registered correctly."""

    def test_evaluator_in_registry(self):
        """Test that evaluator is available in registry."""
        from arbiter_ai.core.registry import (
            get_available_evaluators,
            get_evaluator_class,
        )

        evaluators = get_available_evaluators()
        assert "instruction_following" in evaluators

        evaluator_class = get_evaluator_class("instruction_following")
        assert evaluator_class is InstructionFollowingEvaluator

    def test_evaluator_importable_from_package(self):
        """Test that evaluator is importable from main package."""
        from arbiter_ai import InstructionFollowingEvaluator as ImportedEvaluator

        assert ImportedEvaluator is InstructionFollowingEvaluator
