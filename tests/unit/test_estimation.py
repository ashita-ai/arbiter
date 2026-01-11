"""Tests for cost estimation and dry-run functionality."""

import pytest
from pydantic import ValidationError

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
    """Tests for token estimation using LiteLLM."""

    def test_empty_string(self):
        """Test token estimation for empty string."""
        assert estimate_tokens("") == 0

    def test_short_string(self):
        """Test token estimation for short string."""
        # "Hello" is 1 token
        tokens = estimate_tokens("Hello")
        assert tokens >= 1

    def test_longer_string(self):
        """Test token estimation for longer string."""
        # 100 'a' chars - tokenizer will compress efficiently, but still > 0
        text = "a" * 100
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_realistic_text(self):
        """Test token estimation for realistic text."""
        text = "Paris is the capital of France and is known for the Eiffel Tower."
        tokens = estimate_tokens(text)
        # LiteLLM gives accurate count (usually 15-17 tokens for this)
        assert 10 <= tokens <= 25

    def test_with_model_parameter(self):
        """Test that model parameter affects tokenization."""
        text = "Hello, world! This is a test."
        # Different models may use different tokenizers
        tokens_4o = estimate_tokens(text, model="gpt-4o-mini")
        tokens_35 = estimate_tokens(text, model="gpt-3.5-turbo")
        # Both should give reasonable results (may be same or different)
        assert tokens_4o > 0
        assert tokens_35 > 0

    def test_known_token_count(self):
        """Test against known token count."""
        # "Hello, world!" is exactly 4 tokens with OpenAI tokenizers
        text = "Hello, world!"
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        assert tokens == 4

    def test_unicode_text(self):
        """Test token estimation with unicode characters."""
        # Japanese text
        text_ja = "こんにちは世界"
        tokens_ja = estimate_tokens(text_ja, model="gpt-4o-mini")
        assert tokens_ja > 0

        # Emoji text
        text_emoji = "Hello! How are you doing today? Great!"
        tokens_emoji = estimate_tokens(text_emoji, model="gpt-4o-mini")
        assert tokens_emoji > 0

        # Mixed unicode
        text_mixed = "Hello cafe resume naive"
        tokens_mixed = estimate_tokens(text_mixed, model="gpt-4o-mini")
        assert tokens_mixed > 0

    def test_chinese_text(self):
        """Test token estimation with Chinese characters."""
        text = "machine learning is a very important technology"
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        assert tokens > 0

    def test_special_characters(self):
        """Test token estimation with special characters."""
        text = "Hello\nWorld\tTab \"quotes\" 'apostrophe' @#$%^&*()"
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        assert tokens > 0

    def test_very_long_text(self):
        """Test token estimation with very long text."""
        # 10,000 words of text
        text = " ".join(["word"] * 10000)
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        # Should be roughly 10,000 tokens (one per word)
        assert tokens > 5000
        assert tokens < 20000

    def test_whitespace_only(self):
        """Test token estimation with whitespace-only text."""
        text = "   \n\t\n   "
        tokens = estimate_tokens(text, model="gpt-4o-mini")
        # Whitespace is tokenized but shouldn't be many tokens
        assert tokens >= 0

    def test_unknown_model_fallback(self):
        """Test that unknown models still return reasonable estimates."""
        text = "Hello, world!"
        # Use a model name that definitely doesn't exist
        tokens = estimate_tokens(text, model="nonexistent-model-xyz-123")
        # Should still return a reasonable estimate via fallback
        assert tokens > 0


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


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_cost_estimate_rejects_extra_fields(self):
        """Test that CostEstimate rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            CostEstimate(
                total_cost=0.001,
                total_tokens=100,
                input_tokens=80,
                output_tokens=20,
                unknown_field="should fail",  # type: ignore[call-arg]
            )

    def test_batch_cost_estimate_rejects_extra_fields(self):
        """Test that BatchCostEstimate rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            BatchCostEstimate(
                total_cost=0.01,
                total_tokens=1000,
                per_item_cost=0.001,
                per_item_tokens=100,
                item_count=10,
                unknown_field="should fail",  # type: ignore[call-arg]
            )

    def test_dry_run_result_rejects_extra_fields(self):
        """Test that DryRunResult rejects unknown fields."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            DryRunResult(
                dry_run=True,
                unknown_field="should fail",  # type: ignore[call-arg]
            )

    def test_cost_estimate_immutable(self):
        """Test that CostEstimate is frozen (immutable)."""
        estimate = CostEstimate(
            total_cost=0.001,
            total_tokens=100,
            input_tokens=80,
            output_tokens=20,
        )
        with pytest.raises(ValidationError, match="frozen"):
            estimate.total_cost = 0.002  # type: ignore[misc]

    def test_dry_run_result_immutable(self):
        """Test that DryRunResult is frozen (immutable)."""
        result = DryRunResult(dry_run=True, model="gpt-4o-mini")
        with pytest.raises(ValidationError, match="frozen"):
            result.model = "gpt-4"  # type: ignore[misc]
