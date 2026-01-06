"""Unit tests for input validation module."""

import pytest

from arbiter_ai.core.exceptions import ValidationError
from arbiter_ai.core.validation import (
    MAX_TEXT_LENGTH,
    validate_batch_evaluate_inputs,
    validate_compare_inputs,
    validate_evaluate_inputs,
)


class TestValidateEvaluateInputs:
    """Test suite for validate_evaluate_inputs() function."""

    def test_valid_inputs_semantic(self):
        """Test that valid inputs for semantic evaluator pass validation."""
        # Should not raise any exception
        validate_evaluate_inputs(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            criteria=None,
            evaluators=["semantic"],
            threshold=0.7,
            model="gpt-4o",
        )

    def test_valid_inputs_custom_criteria(self):
        """Test that valid inputs for custom_criteria evaluator pass validation."""
        validate_evaluate_inputs(
            output="Test output",
            reference=None,
            criteria="Accuracy and clarity",
            evaluators=["custom_criteria"],
            threshold=0.8,
            model="gpt-4o-mini",
        )

    def test_valid_inputs_multiple_evaluators(self):
        """Test that valid inputs for multiple evaluators pass validation."""
        validate_evaluate_inputs(
            output="Test output",
            reference="Reference text",
            criteria="Test criteria",
            evaluators=["semantic", "custom_criteria"],
            threshold=0.5,
            model="claude-3-5-sonnet",
        )

    def test_valid_inputs_no_evaluators(self):
        """Test that None evaluators is valid (defaults will be applied)."""
        validate_evaluate_inputs(
            output="Test output",
            reference="Reference text",
            criteria=None,
            evaluators=None,
            threshold=0.7,
            model="gpt-4o",
        )

    def test_empty_output(self):
        """Test that empty output raises ValidationError."""
        with pytest.raises(ValidationError, match="'output' cannot be empty"):
            validate_evaluate_inputs(
                output="",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_whitespace_only_output(self):
        """Test that whitespace-only output raises ValidationError."""
        with pytest.raises(ValidationError, match="'output' cannot be empty"):
            validate_evaluate_inputs(
                output="   \t\n  ",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_output_exceeds_max_length(self):
        """Test that output exceeding max length raises ValidationError."""
        long_output = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'output' exceeds maximum length"):
            validate_evaluate_inputs(
                output=long_output,
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_output_at_max_length(self):
        """Test that output at exactly max length is valid."""
        max_output = "x" * MAX_TEXT_LENGTH
        validate_evaluate_inputs(
            output=max_output,
            reference="Reference",
            criteria=None,
            evaluators=["semantic"],
            threshold=0.7,
            model="gpt-4o",
        )

    def test_empty_evaluators_list(self):
        """Test that empty evaluators list raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="'evaluators' cannot be empty - provide at least one evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=[],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_unknown_evaluator(self):
        """Test that unknown evaluator name raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Unknown evaluator 'not_real'. Available:"
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["not_real"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_unknown_evaluator_lists_available(self):
        """Test that error message lists available evaluators."""
        with pytest.raises(ValidationError) as exc_info:
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["nonexistent"],
                threshold=0.7,
                model="gpt-4o",
            )
        error_msg = str(exc_info.value)
        assert "semantic" in error_msg
        assert "custom_criteria" in error_msg
        assert "factuality" in error_msg

    def test_threshold_below_minimum(self):
        """Test that threshold below 0.0 raises ValidationError."""
        with pytest.raises(
            ValidationError, match="'threshold' must be between 0.0 and 1.0"
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=-0.1,
                model="gpt-4o",
            )

    def test_threshold_above_maximum(self):
        """Test that threshold above 1.0 raises ValidationError."""
        with pytest.raises(
            ValidationError, match="'threshold' must be between 0.0 and 1.0, got 1.5"
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=1.5,
                model="gpt-4o",
            )

    def test_threshold_at_boundaries(self):
        """Test that threshold at 0.0 and 1.0 is valid."""
        # Threshold = 0.0
        validate_evaluate_inputs(
            output="Test output",
            reference="Reference",
            criteria=None,
            evaluators=["semantic"],
            threshold=0.0,
            model="gpt-4o",
        )

        # Threshold = 1.0
        validate_evaluate_inputs(
            output="Test output",
            reference="Reference",
            criteria=None,
            evaluators=["semantic"],
            threshold=1.0,
            model="gpt-4o",
        )

    def test_empty_model(self):
        """Test that empty model raises ValidationError."""
        with pytest.raises(ValidationError, match="'model' cannot be empty"):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="",
            )

    def test_whitespace_only_model(self):
        """Test that whitespace-only model raises ValidationError."""
        with pytest.raises(ValidationError, match="'model' cannot be empty"):
            validate_evaluate_inputs(
                output="Test output",
                reference="Reference",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="   ",
            )

    def test_semantic_evaluator_requires_reference(self):
        """Test that semantic evaluator requires reference."""
        with pytest.raises(
            ValidationError,
            match="'reference' is required when using 'semantic' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference=None,
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_semantic_evaluator_empty_reference(self):
        """Test that semantic evaluator rejects empty reference."""
        with pytest.raises(
            ValidationError,
            match="'reference' is required when using 'semantic' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_semantic_evaluator_whitespace_reference(self):
        """Test that semantic evaluator rejects whitespace-only reference."""
        with pytest.raises(
            ValidationError,
            match="'reference' is required when using 'semantic' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference="   ",
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_reference_exceeds_max_length(self):
        """Test that reference exceeding max length raises ValidationError."""
        long_reference = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'reference' exceeds maximum length"):
            validate_evaluate_inputs(
                output="Test output",
                reference=long_reference,
                criteria=None,
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_custom_criteria_evaluator_requires_criteria(self):
        """Test that custom_criteria evaluator requires criteria."""
        with pytest.raises(
            ValidationError,
            match="'criteria' is required when using 'custom_criteria' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference=None,
                criteria=None,
                evaluators=["custom_criteria"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_custom_criteria_evaluator_empty_criteria(self):
        """Test that custom_criteria evaluator rejects empty criteria."""
        with pytest.raises(
            ValidationError,
            match="'criteria' is required when using 'custom_criteria' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference=None,
                criteria="",
                evaluators=["custom_criteria"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_custom_criteria_evaluator_whitespace_criteria(self):
        """Test that custom_criteria evaluator rejects whitespace-only criteria."""
        with pytest.raises(
            ValidationError,
            match="'criteria' is required when using 'custom_criteria' evaluator",
        ):
            validate_evaluate_inputs(
                output="Test output",
                reference=None,
                criteria="   ",
                evaluators=["custom_criteria"],
                threshold=0.7,
                model="gpt-4o",
            )

    def test_criteria_exceeds_max_length(self):
        """Test that criteria exceeding max length raises ValidationError."""
        long_criteria = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'criteria' exceeds maximum length"):
            validate_evaluate_inputs(
                output="Test output",
                reference=None,
                criteria=long_criteria,
                evaluators=["custom_criteria"],
                threshold=0.7,
                model="gpt-4o",
            )


class TestValidateCompareInputs:
    """Test suite for validate_compare_inputs() function."""

    def test_valid_inputs_basic(self):
        """Test that valid inputs pass validation."""
        validate_compare_inputs(
            output_a="Output A text",
            output_b="Output B text",
            criteria=None,
            model="gpt-4o",
        )

    def test_valid_inputs_with_criteria(self):
        """Test that valid inputs with criteria pass validation."""
        validate_compare_inputs(
            output_a="Output A text",
            output_b="Output B text",
            criteria="Accuracy and clarity",
            model="claude-3-5-sonnet",
        )

    def test_empty_output_a(self):
        """Test that empty output_a raises ValidationError."""
        with pytest.raises(ValidationError, match="'output_a' cannot be empty"):
            validate_compare_inputs(
                output_a="",
                output_b="Output B",
                criteria=None,
                model="gpt-4o",
            )

    def test_whitespace_only_output_a(self):
        """Test that whitespace-only output_a raises ValidationError."""
        with pytest.raises(ValidationError, match="'output_a' cannot be empty"):
            validate_compare_inputs(
                output_a="   \t\n  ",
                output_b="Output B",
                criteria=None,
                model="gpt-4o",
            )

    def test_output_a_exceeds_max_length(self):
        """Test that output_a exceeding max length raises ValidationError."""
        long_output = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'output_a' exceeds maximum length"):
            validate_compare_inputs(
                output_a=long_output,
                output_b="Output B",
                criteria=None,
                model="gpt-4o",
            )

    def test_empty_output_b(self):
        """Test that empty output_b raises ValidationError."""
        with pytest.raises(ValidationError, match="'output_b' cannot be empty"):
            validate_compare_inputs(
                output_a="Output A",
                output_b="",
                criteria=None,
                model="gpt-4o",
            )

    def test_whitespace_only_output_b(self):
        """Test that whitespace-only output_b raises ValidationError."""
        with pytest.raises(ValidationError, match="'output_b' cannot be empty"):
            validate_compare_inputs(
                output_a="Output A",
                output_b="   \t\n  ",
                criteria=None,
                model="gpt-4o",
            )

    def test_output_b_exceeds_max_length(self):
        """Test that output_b exceeding max length raises ValidationError."""
        long_output = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'output_b' exceeds maximum length"):
            validate_compare_inputs(
                output_a="Output A",
                output_b=long_output,
                criteria=None,
                model="gpt-4o",
            )

    def test_empty_model(self):
        """Test that empty model raises ValidationError."""
        with pytest.raises(ValidationError, match="'model' cannot be empty"):
            validate_compare_inputs(
                output_a="Output A",
                output_b="Output B",
                criteria=None,
                model="",
            )

    def test_whitespace_only_model(self):
        """Test that whitespace-only model raises ValidationError."""
        with pytest.raises(ValidationError, match="'model' cannot be empty"):
            validate_compare_inputs(
                output_a="Output A",
                output_b="Output B",
                criteria=None,
                model="   ",
            )

    def test_empty_criteria_when_provided(self):
        """Test that empty criteria raises ValidationError when provided."""
        with pytest.raises(
            ValidationError, match="'criteria' cannot be empty if provided"
        ):
            validate_compare_inputs(
                output_a="Output A",
                output_b="Output B",
                criteria="",
                model="gpt-4o",
            )

    def test_whitespace_only_criteria_when_provided(self):
        """Test that whitespace-only criteria raises ValidationError when provided."""
        with pytest.raises(
            ValidationError, match="'criteria' cannot be empty if provided"
        ):
            validate_compare_inputs(
                output_a="Output A",
                output_b="Output B",
                criteria="   ",
                model="gpt-4o",
            )

    def test_criteria_exceeds_max_length(self):
        """Test that criteria exceeding max length raises ValidationError."""
        long_criteria = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(ValidationError, match="'criteria' exceeds maximum length"):
            validate_compare_inputs(
                output_a="Output A",
                output_b="Output B",
                criteria=long_criteria,
                model="gpt-4o",
            )

    def test_none_criteria_is_valid(self):
        """Test that None criteria is valid (optional parameter)."""
        validate_compare_inputs(
            output_a="Output A",
            output_b="Output B",
            criteria=None,
            model="gpt-4o",
        )


class TestValidateBatchEvaluateInputs:
    """Test suite for validate_batch_evaluate_inputs() function."""

    def test_valid_inputs_basic(self):
        """Test that valid batch inputs pass validation."""
        validate_batch_evaluate_inputs(
            items=[
                {"output": "Output 1", "reference": "Reference 1"},
                {"output": "Output 2", "reference": "Reference 2"},
            ],
            evaluators=["semantic"],
            threshold=0.7,
            model="gpt-4o",
            max_concurrency=10,
        )

    def test_valid_inputs_minimal(self):
        """Test that minimal valid inputs pass validation."""
        validate_batch_evaluate_inputs(
            items=[{"output": "Output 1"}],
            evaluators=None,
            threshold=0.5,
            model="gpt-4o-mini",
            max_concurrency=5,
        )

    def test_valid_inputs_with_criteria(self):
        """Test that valid inputs with criteria pass validation."""
        validate_batch_evaluate_inputs(
            items=[
                {"output": "Output 1", "criteria": "Criteria 1"},
                {"output": "Output 2", "criteria": "Criteria 2"},
            ],
            evaluators=["custom_criteria"],
            threshold=0.8,
            model="claude-3-5-sonnet",
            max_concurrency=3,
        )

    def test_empty_items_list(self):
        """Test that empty items list raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="'items' cannot be empty - provide at least one item",
        ):
            validate_batch_evaluate_inputs(
                items=[],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_not_dict(self):
        """Test that non-dict item raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Item at index 0 must be a dictionary, got str"
        ):
            validate_batch_evaluate_inputs(
                items=["not a dict"],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_missing_output_key(self):
        """Test that item missing 'output' key raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Item at index 0 missing required field 'output'"
        ):
            validate_batch_evaluate_inputs(
                items=[{"reference": "Reference only"}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_empty_output(self):
        """Test that item with empty output raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Item at index 0: 'output' cannot be empty"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": ""}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_whitespace_only_output(self):
        """Test that item with whitespace-only output raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Item at index 0: 'output' cannot be empty"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "   \t\n  "}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_output_exceeds_max_length(self):
        """Test that item output exceeding max length raises ValidationError."""
        long_output = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(
            ValidationError,
            match="Item at index 0: 'output' exceeds maximum length",
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": long_output}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_reference_exceeds_max_length(self):
        """Test that item reference exceeding max length raises ValidationError."""
        long_reference = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(
            ValidationError,
            match="Item at index 1: 'reference' exceeds maximum length",
        ):
            validate_batch_evaluate_inputs(
                items=[
                    {"output": "Output 1", "reference": "Reference 1"},
                    {"output": "Output 2", "reference": long_reference},
                ],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_item_criteria_exceeds_max_length(self):
        """Test that item criteria exceeding max length raises ValidationError."""
        long_criteria = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(
            ValidationError,
            match="Item at index 0: 'criteria' exceeds maximum length",
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1", "criteria": long_criteria}],
                evaluators=["custom_criteria"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_empty_evaluators_list(self):
        """Test that empty evaluators list raises ValidationError."""
        with pytest.raises(
            ValidationError,
            match="'evaluators' cannot be empty - provide at least one evaluator",
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=[],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_unknown_evaluator(self):
        """Test that unknown evaluator raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Unknown evaluator 'unknown'. Available:"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=["unknown"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_threshold_out_of_range(self):
        """Test that threshold out of range raises ValidationError."""
        with pytest.raises(
            ValidationError, match="'threshold' must be between 0.0 and 1.0"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=["semantic"],
                threshold=2.0,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_empty_model(self):
        """Test that empty model raises ValidationError."""
        with pytest.raises(ValidationError, match="'model' cannot be empty"):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=["semantic"],
                threshold=0.7,
                model="",
                max_concurrency=10,
            )

    def test_negative_max_concurrency(self):
        """Test that negative max_concurrency raises ValidationError."""
        with pytest.raises(
            ValidationError, match="'max_concurrency' must be positive, got -1"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=-1,
            )

    def test_zero_max_concurrency(self):
        """Test that zero max_concurrency raises ValidationError."""
        with pytest.raises(
            ValidationError, match="'max_concurrency' must be positive, got 0"
        ):
            validate_batch_evaluate_inputs(
                items=[{"output": "Output 1"}],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=0,
            )

    def test_multiple_items_validation(self):
        """Test that validation catches error in second item."""
        with pytest.raises(
            ValidationError, match="Item at index 1: 'output' cannot be empty"
        ):
            validate_batch_evaluate_inputs(
                items=[
                    {"output": "Valid output"},
                    {"output": ""},
                    {"output": "Another valid output"},
                ],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )

    def test_mixed_valid_and_invalid_items(self):
        """Test that validation stops at first invalid item."""
        with pytest.raises(ValidationError, match="Item at index 2 must be a"):
            validate_batch_evaluate_inputs(
                items=[
                    {"output": "Valid 1"},
                    {"output": "Valid 2"},
                    "not a dict",
                    {"output": "Valid 4"},
                ],
                evaluators=["semantic"],
                threshold=0.7,
                model="gpt-4o",
                max_concurrency=10,
            )
