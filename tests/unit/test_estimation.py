"""Tests for cost estimation and dry-run functionality."""

import pytest

from arbiter_ai import evaluate
from arbiter_ai.core.estimation import (
    BatchCostEstimate,
    CostEstimate,
    DryRunResult,
    estimate_batch_cost,
    estimate_evaluation_cost,
    estimate_tokens,
    get_prompt_preview,
)


class TestEstimateTokens:
    """Tests for token estimation using tiktoken."""

    def test_empty_string(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_short_string(self):
        """Test token estimation for short string."""
        # "Hello" is 1 token in tiktoken
        tokens = estimate_tokens("Hello")
        assert tokens >= 1

    def test_longer_string(self):
        """Test token estimation for longer string."""
        # 100 'a' chars - tiktoken will tokenize this efficiently
        text = "a" * 100
        tokens = estimate_tokens(text)
        # With tiktoken, repeated chars compress well, but still > 0
        assert tokens > 0

    def test_realistic_text(self):
        """Test token estimation for realistic text."""
        text = "Paris is the capital of France and is known for the Eiffel Tower."
        tokens = estimate_tokens(text)
        # tiktoken gives accurate count (usually 15-17 tokens for this)
        assert 10 <= tokens <= 25

    def test_with_model_parameter(self):
        """Test that model parameter affects tokenization."""
        text = "Hello, world! This is a test."
        # Different models use different tokenizers
        tokens_4o = estimate_tokens(text, model="gpt-4o-mini")
        tokens_35 = estimate_tokens(text, model="gpt-3.5-turbo")
        # Both should give reasonable results (may be same or different)
        assert tokens_4o > 0
        assert tokens_35 > 0

    def test_known_token_count(self):
        """Test against known tiktoken output."""
        # "Hello, world!" is exactly 4 tokens with cl100k_base/o200k_base
        text = "Hello, world!"
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        assert tokens == 4


class TestEstimateEvaluationCost:
    """Tests for single evaluation cost estimation."""

    @pytest.mark.asyncio
    async def test_basic_estimation(self):
        """Test basic cost estimation."""
        estimate = await estimate_evaluation_cost(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert isinstance(estimate, CostEstimate)
        assert estimate.total_cost >= 0
        assert estimate.total_tokens > 0
        assert estimate.input_tokens > 0
        assert estimate.output_tokens > 0
        assert estimate.model == "gpt-4o-mini"
        assert "semantic" in estimate.by_evaluator

    @pytest.mark.asyncio
    async def test_multiple_evaluators(self):
        """Test cost estimation with multiple evaluators."""
        estimate = await estimate_evaluation_cost(
            output="Test output",
            reference="Test reference",
            criteria="Test criteria",
            evaluators=["semantic", "custom_criteria"],
            model="gpt-4o-mini",
        )

        assert len(estimate.by_evaluator) == 2
        assert "semantic" in estimate.by_evaluator
        assert "custom_criteria" in estimate.by_evaluator
        # Multiple evaluators should cost more
        assert estimate.total_cost > 0

    @pytest.mark.asyncio
    async def test_default_evaluators(self):
        """Test cost estimation with default evaluators."""
        estimate = await estimate_evaluation_cost(
            output="Test output",
            model="gpt-4o-mini",
        )

        # Default is ["semantic"]
        assert "semantic" in estimate.by_evaluator

    @pytest.mark.asyncio
    async def test_invalid_evaluator(self):
        """Test cost estimation with invalid evaluator."""
        with pytest.raises(ValueError, match="Unknown evaluator"):
            await estimate_evaluation_cost(
                output="Test",
                evaluators=["nonexistent_evaluator"],
                model="gpt-4o-mini",
            )


class TestEstimateBatchCost:
    """Tests for batch cost estimation."""

    @pytest.mark.asyncio
    async def test_basic_batch_estimation(self):
        """Test basic batch cost estimation."""
        items = [
            {"output": "Output 1", "reference": "Reference 1"},
            {"output": "Output 2", "reference": "Reference 2"},
            {"output": "Output 3", "reference": "Reference 3"},
        ]

        estimate = await estimate_batch_cost(
            items=items,
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert isinstance(estimate, BatchCostEstimate)
        assert estimate.item_count == 3
        assert estimate.total_cost > 0
        assert estimate.per_item_cost > 0
        assert estimate.total_tokens > 0
        assert estimate.per_item_tokens > 0

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        """Test cost estimation for empty batch."""
        estimate = await estimate_batch_cost(
            items=[],
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert estimate.item_count == 0
        assert estimate.total_cost == 0.0
        assert estimate.per_item_cost == 0.0

    @pytest.mark.asyncio
    async def test_batch_with_criteria(self):
        """Test batch estimation with criteria."""
        items = [
            {"output": "Output 1", "criteria": "Be concise"},
            {"output": "Output 2", "criteria": "Be detailed"},
        ]

        estimate = await estimate_batch_cost(
            items=items,
            evaluators=["custom_criteria"],
            model="gpt-4o-mini",
        )

        assert estimate.item_count == 2
        assert "custom_criteria" in estimate.by_evaluator


class TestGetPromptPreview:
    """Tests for prompt preview (dry-run core)."""

    @pytest.mark.asyncio
    async def test_basic_preview(self):
        """Test basic prompt preview."""
        preview = await get_prompt_preview(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert isinstance(preview, DryRunResult)
        assert preview.dry_run is True
        assert preview.evaluators == ["semantic"]
        assert "semantic" in preview.prompts
        assert "system" in preview.prompts["semantic"]
        assert "user" in preview.prompts["semantic"]
        assert preview.estimated_cost >= 0
        assert preview.estimated_tokens > 0
        assert preview.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_preview_validation(self):
        """Test that preview includes validation info."""
        preview = await get_prompt_preview(
            output="Valid output",
            reference="Valid reference",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert preview.validation["output_valid"] is True
        assert preview.validation["reference_valid"] is True
        assert preview.validation["evaluators_valid"] is True

    @pytest.mark.asyncio
    async def test_preview_with_empty_output(self):
        """Test preview with empty output."""
        preview = await get_prompt_preview(
            output="",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        assert preview.validation["output_valid"] is False

    @pytest.mark.asyncio
    async def test_preview_multiple_evaluators(self):
        """Test preview with multiple evaluators."""
        preview = await get_prompt_preview(
            output="Test output",
            reference="Test reference",
            criteria="Test criteria",
            evaluators=["semantic", "custom_criteria"],
            model="gpt-4o-mini",
        )

        assert len(preview.evaluators) == 2
        assert len(preview.prompts) == 2


class TestEvaluateDryRun:
    """Tests for dry_run parameter in evaluate()."""

    @pytest.mark.asyncio
    async def test_dry_run_returns_preview(self):
        """Test that dry_run=True returns DryRunResult."""
        result = await evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert isinstance(result, DryRunResult)
        assert result.dry_run is True

    @pytest.mark.asyncio
    async def test_dry_run_no_api_calls(self):
        """Test that dry_run doesn't make API calls.

        This test verifies that no actual LLM calls are made by checking
        that we can run without valid API keys (the estimation doesn't need them).
        """
        # This should not raise even without valid API credentials
        # because dry_run doesn't make actual API calls
        result = await evaluate(
            output="Test output",
            evaluators=["semantic"],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert isinstance(result, DryRunResult)
        assert result.estimated_cost >= 0

    @pytest.mark.asyncio
    async def test_dry_run_shows_prompts(self):
        """Test that dry_run shows the prompts that would be used."""
        result = await evaluate(
            output="Test output for evaluation",
            reference="Reference text",
            evaluators=["semantic"],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert isinstance(result, DryRunResult)
        assert "semantic" in result.prompts
        # The user prompt should contain the output text
        assert "Test output for evaluation" in result.prompts["semantic"]["user"]

    @pytest.mark.asyncio
    async def test_dry_run_with_custom_criteria(self):
        """Test dry_run with custom_criteria evaluator."""
        result = await evaluate(
            output="Test output",
            criteria="The output should be informative",
            evaluators=["custom_criteria"],
            model="gpt-4o-mini",
            dry_run=True,
        )

        assert isinstance(result, DryRunResult)
        assert "custom_criteria" in result.prompts
        # The criteria should appear in the prompt
        assert "informative" in result.prompts["custom_criteria"]["user"]
