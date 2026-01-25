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
def mock_agent():
    """Create a mock PydanticAI agent."""
    agent = AsyncMock()
    return agent


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
        assert "expert evaluator" in prompt.lower()
        assert "instruction" in prompt.lower()
        assert "format" in prompt.lower()
        assert "severity" in prompt.lower()

    def test_user_prompt_with_criteria(self, evaluator):
        """Test user prompt generation with criteria (instructions)."""
        output = "Test output"
        criteria = "Respond in JSON format. Include 'name' field."
        prompt = evaluator._get_user_prompt(output, None, criteria)

        assert output in prompt
        assert criteria in prompt
        assert "OUTPUT TO EVALUATE" in prompt
        assert "INSTRUCTIONS TO CHECK" in prompt

    def test_user_prompt_with_reference(self, evaluator):
        """Test user prompt generation with reference."""
        output = "Test output"
        reference = "Reference context"
        criteria = "Follow the format shown in reference"
        prompt = evaluator._get_user_prompt(output, reference, criteria)

        assert output in prompt
        assert reference in prompt
        assert criteria in prompt
        assert "REFERENCE CONTEXT" in prompt

    def test_user_prompt_requires_criteria(self, evaluator):
        """Test that user prompt raises error without criteria."""
        with pytest.raises(ValueError, match="requires criteria"):
            evaluator._get_user_prompt("output", None, None)

    def test_response_type(self, evaluator):
        """Test that response type is correct."""
        response_type = evaluator._get_response_type()
        assert response_type == InstructionFollowingResponse

    @pytest.mark.asyncio
    async def test_compute_score(self, evaluator):
        """Test score computation from response."""
        response = InstructionFollowingResponse(
            score=0.75,
            confidence=0.9,
            explanation="The output follows most instructions but violates format",
            instructions_followed=["Include name field", "Keep under 100 words"],
            instructions_violated=["Respond in JSON format"],
            instructions_partially_met=[],
            violation_severity="major",
        )

        score = await evaluator._compute_score(response)

        assert score.name == "instruction_following"
        assert score.value == 0.75
        assert score.confidence == 0.9
        assert "follows most instructions" in score.explanation
        assert score.metadata["instructions_followed"] == [
            "Include name field",
            "Keep under 100 words",
        ]
        assert score.metadata["instructions_violated"] == ["Respond in JSON format"]
        assert score.metadata["instructions_partially_met"] == []
        assert score.metadata["instructions_followed_count"] == 2
        assert score.metadata["instructions_violated_count"] == 1
        assert score.metadata["instructions_partially_met_count"] == 0
        assert score.metadata["violation_severity"] == "major"

    @pytest.mark.asyncio
    async def test_evaluate_full_compliance(self, evaluator, mock_agent):
        """Test evaluation with full instruction compliance."""
        mock_response = InstructionFollowingResponse(
            score=1.0,
            confidence=0.95,
            explanation="All instructions followed perfectly",
            instructions_followed=[
                "Respond in JSON format",
                "Include 'name' and 'age' fields",
                "Keep response under 100 words",
            ],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="none",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)

        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output='{"name": "John", "age": 30}',
            criteria="Respond in JSON format. Include 'name' and 'age' fields. Keep response under 100 words.",
        )

        assert score.value == 1.0
        assert score.confidence == 0.95
        assert len(score.metadata["instructions_followed"]) == 3
        assert len(score.metadata["instructions_violated"]) == 0
        assert score.metadata["violation_severity"] == "none"

        assert mock_agent.run.called
        call_args = mock_agent.run.call_args[0][0]
        assert "JSON" in call_args
        assert "name" in call_args

        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        assert interactions[0].purpose == "instruction_following_evaluation"

    @pytest.mark.asyncio
    async def test_evaluate_with_violations(self, evaluator, mock_agent):
        """Test evaluation with instruction violations."""
        mock_response = InstructionFollowingResponse(
            score=0.4,
            confidence=0.85,
            explanation="Multiple critical violations detected",
            instructions_followed=["Mentioned the topic"],
            instructions_violated=["Respond in JSON format", "Include required fields"],
            instructions_partially_met=["Keep under 100 words"],
            violation_severity="critical",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="This is plain text without any JSON structure.",
            criteria="Respond in JSON format. Include required fields. Keep under 100 words.",
        )

        assert score.value == 0.4
        assert score.metadata["violation_severity"] == "critical"
        assert len(score.metadata["instructions_violated"]) == 2
        assert len(score.metadata["instructions_partially_met"]) == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_reference(self, evaluator, mock_agent):
        """Test evaluation with reference context."""
        mock_response = InstructionFollowingResponse(
            score=0.8,
            confidence=0.88,
            explanation="Output mostly follows the reference format",
            instructions_followed=["Follow reference format"],
            instructions_violated=[],
            instructions_partially_met=["Match tone"],
            violation_severity="minor",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        score = await evaluator.evaluate(
            output="Output text following format",
            reference="Example format to follow",
            criteria="Follow reference format. Match tone.",
        )

        assert score.value == 0.8
        call_args = mock_agent.run.call_args[0][0]
        assert "Example format" in call_args
        assert "REFERENCE CONTEXT" in call_args

    @pytest.mark.asyncio
    async def test_evaluate_error_handling(self, evaluator, mock_agent):
        """Test error handling during evaluation."""
        mock_agent.run = AsyncMock(side_effect=Exception("API error"))
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        with pytest.raises(EvaluatorError, match="Evaluation failed"):
            await evaluator.evaluate(
                output="Test output",
                criteria="Test criteria",
            )

    @pytest.mark.asyncio
    async def test_interaction_tracking(self, evaluator, mock_agent):
        """Test that LLM interactions are tracked."""
        mock_response = InstructionFollowingResponse(
            score=0.9,
            confidence=0.85,
            explanation="Instructions followed well",
            instructions_followed=["Format correct"],
            instructions_violated=[],
            instructions_partially_met=[],
            violation_severity="none",
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        evaluator._llm_client = evaluator.llm_client
        evaluator.llm_client.create_agent = MagicMock(return_value=mock_agent)

        evaluator.clear_interactions()

        await evaluator.evaluate(
            output="Test output",
            criteria="Test criteria",
        )

        interactions = evaluator.get_interactions()
        assert len(interactions) == 1
        interaction = interactions[0]
        assert interaction.purpose == "instruction_following_evaluation"
        assert interaction.model == "gpt-4o-mini"
        assert "Test output" in interaction.prompt
        assert interaction.metadata["evaluator"] == "instruction_following"

    def test_clear_interactions(self, evaluator):
        """Test clearing interactions."""
        from arbiter_ai.core.models import LLMInteraction

        interaction = LLMInteraction(
            prompt="test",
            response="test",
            model="test",
            tokens_used=10,
            latency=0.1,
            purpose="test",
        )
        evaluator.interactions.append(interaction)

        assert len(evaluator.interactions) == 1
        evaluator.clear_interactions()
        assert len(evaluator.interactions) == 0


class TestInstructionFollowingResponse:
    """Test InstructionFollowingResponse model."""

    def test_response_creation(self):
        """Test creating a response with all fields."""
        response = InstructionFollowingResponse(
            score=0.75,
            confidence=0.9,
            explanation="Test explanation",
            instructions_followed=["Instruction 1"],
            instructions_violated=["Instruction 2"],
            instructions_partially_met=["Instruction 3"],
            violation_severity="major",
        )

        assert response.score == 0.75
        assert response.confidence == 0.9
        assert response.explanation == "Test explanation"
        assert len(response.instructions_followed) == 1
        assert len(response.instructions_violated) == 1
        assert len(response.instructions_partially_met) == 1
        assert response.violation_severity == "major"

    def test_response_defaults(self):
        """Test response with defaults."""
        response = InstructionFollowingResponse(
            score=1.0,
            explanation="All instructions followed",
        )

        assert response.confidence == 0.85
        assert response.instructions_followed == []
        assert response.instructions_violated == []
        assert response.instructions_partially_met == []
        assert response.violation_severity == "none"

    def test_response_score_validation(self):
        """Test that score must be between 0 and 1."""
        response = InstructionFollowingResponse(
            score=1.0, explanation="All instructions followed"
        )
        assert response.score == 1.0

        response = InstructionFollowingResponse(
            score=0.0,
            explanation="No instructions followed",
            instructions_violated=["All instructions"],
            violation_severity="critical",
        )
        assert response.score == 0.0

        with pytest.raises(Exception):
            InstructionFollowingResponse(score=1.5, explanation="Test")

        with pytest.raises(Exception):
            InstructionFollowingResponse(score=-0.1, explanation="Test")

    def test_response_explanation_validation(self):
        """Test that explanation cannot be empty."""
        with pytest.raises(Exception, match="cannot be empty"):
            InstructionFollowingResponse(score=0.5, explanation="")

        with pytest.raises(Exception, match="cannot be empty"):
            InstructionFollowingResponse(score=0.5, explanation="   ")

    def test_response_severity_validation(self):
        """Test that severity must be valid literal."""
        response = InstructionFollowingResponse(
            score=0.5,
            explanation="Test",
            instructions_violated=["Test instruction"],
            violation_severity="minor",
        )
        assert response.violation_severity == "minor"

        with pytest.raises(Exception):
            InstructionFollowingResponse(
                score=0.5,
                explanation="Test",
                instructions_violated=["Test"],
                violation_severity="invalid",
            )

    def test_severity_consistency_with_violations(self):
        """Test that severity is consistent with violations."""
        with pytest.raises(ValueError, match="cannot be 'none'"):
            InstructionFollowingResponse(
                score=0.5,
                explanation="Test",
                instructions_followed=["One"],
                instructions_violated=["Another"],
                violation_severity="none",
            )

    def test_severity_consistency_without_violations(self):
        """Test that severity must be none when no violations."""
        with pytest.raises(ValueError, match="must be 'none'"):
            InstructionFollowingResponse(
                score=0.9,
                explanation="Test",
                instructions_followed=["Instruction"],
                instructions_violated=[],
                instructions_partially_met=[],
                violation_severity="minor",
            )

    def test_partial_violations_allow_non_none_severity(self):
        """Test that partial violations can have non-none severity."""
        response = InstructionFollowingResponse(
            score=0.7,
            explanation="Partial compliance",
            instructions_followed=["One"],
            instructions_violated=[],
            instructions_partially_met=["Another"],
            violation_severity="minor",
        )
        assert response.violation_severity == "minor"

    def test_instruction_analysis_validation(self):
        """Test that high confidence scores require instruction identification."""
        with pytest.raises(ValueError, match="require at least one instruction"):
            InstructionFollowingResponse(
                score=0.5,
                confidence=0.9,
                explanation="Test",
                instructions_followed=[],
                instructions_violated=[],
                instructions_partially_met=[],
            )

    def test_extreme_scores_dont_require_instructions(self):
        """Test that extreme scores (0.0, 1.0) don't require instructions."""
        response = InstructionFollowingResponse(
            score=1.0,
            confidence=0.95,
            explanation="Perfect compliance",
        )
        assert response.score == 1.0

        response = InstructionFollowingResponse(
            score=0.0,
            confidence=0.95,
            explanation="Complete non-compliance",
            instructions_violated=["All"],
            violation_severity="critical",
        )
        assert response.score == 0.0


class TestInstructionFollowingRegistration:
    """Test that evaluator is properly registered."""

    def test_evaluator_in_registry(self):
        """Test that evaluator is registered in the registry."""
        from arbiter_ai.core.registry import get_evaluator_class

        evaluator_class = get_evaluator_class("instruction_following")
        assert evaluator_class is not None
        assert evaluator_class == InstructionFollowingEvaluator

    def test_evaluator_in_available_list(self):
        """Test that evaluator appears in available evaluators."""
        from arbiter_ai.core.registry import get_available_evaluators

        available = get_available_evaluators()
        assert "instruction_following" in available

    def test_evaluator_importable_from_arbiter(self):
        """Test that evaluator can be imported from arbiter_ai."""
        from arbiter_ai import InstructionFollowingEvaluator as ImportedEvaluator

        assert ImportedEvaluator is InstructionFollowingEvaluator
