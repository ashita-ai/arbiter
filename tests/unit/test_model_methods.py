"""Unit tests for model class methods: __repr__, summary(), to_dict(), to_json().

Tests cover:
- __repr__ methods for all model classes (#74)
- summary() methods for result classes (#80)
- to_dict() and to_json() export methods (#69)
"""

import json
from datetime import datetime, timezone

import pytest

from arbiter_ai.core.models import (
    BatchEvaluationResult,
    ComparisonResult,
    EvaluationResult,
    LLMInteraction,
    Metric,
    Score,
)


class TestScoreRepr:
    """Tests for Score.__repr__()."""

    def test_repr_basic(self) -> None:
        """Test basic repr output."""
        score = Score(name="semantic", value=0.85)
        result = repr(score)
        assert "<Score" in result
        assert "name='semantic'" in result
        assert "value=0.85" in result

    def test_repr_with_confidence(self) -> None:
        """Test repr includes confidence when present."""
        score = Score(name="factuality", value=0.92, confidence=0.88)
        result = repr(score)
        assert "confidence=0.88" in result

    def test_repr_without_confidence(self) -> None:
        """Test repr excludes confidence when None."""
        score = Score(name="test", value=0.75, confidence=None)
        result = repr(score)
        assert "confidence" not in result


class TestLLMInteractionRepr:
    """Tests for LLMInteraction.__repr__()."""

    def test_repr_basic(self) -> None:
        """Test basic repr output."""
        interaction = LLMInteraction(
            prompt="Test prompt",
            response="Test response",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            latency=1.23,
            purpose="test",
        )
        result = repr(interaction)
        assert "<LLMInteraction" in result
        assert "model='gpt-4o-mini'" in result
        assert "tokens=100+50" in result
        assert "latency=1.23s" in result


class TestMetricRepr:
    """Tests for Metric.__repr__()."""

    def test_repr_basic(self) -> None:
        """Test basic repr output."""
        metric = Metric(
            name="semantic_similarity",
            evaluator="semantic",
            processing_time=0.45,
        )
        result = repr(metric)
        assert "<Metric" in result
        assert "name='semantic_similarity'" in result
        assert "evaluator='semantic'" in result
        assert "time=0.45s" in result

    def test_repr_with_model(self) -> None:
        """Test repr includes model when present."""
        metric = Metric(
            name="test",
            evaluator="test_eval",
            model="gpt-4o",
            processing_time=1.0,
        )
        result = repr(metric)
        assert "model='gpt-4o'" in result

    def test_repr_without_model(self) -> None:
        """Test repr excludes model when None."""
        metric = Metric(
            name="test",
            evaluator="test_eval",
            processing_time=1.0,
        )
        result = repr(metric)
        assert "model=" not in result


class TestEvaluationResultRepr:
    """Tests for EvaluationResult.__repr__()."""

    def test_repr_passed(self) -> None:
        """Test repr for passed evaluation."""
        result = EvaluationResult(
            output="test output",
            overall_score=0.87,
            passed=True,
            processing_time=1.23,
            scores=[
                Score(name="semantic", value=0.92),
                Score(name="factuality", value=0.82),
            ],
        )
        repr_str = repr(result)
        assert "<EvaluationResult" in repr_str
        assert "score=0.87" in repr_str
        assert "passed=True" in repr_str
        assert "['semantic', 'factuality']" in repr_str
        assert "time=1.23s" in repr_str

    def test_repr_failed(self) -> None:
        """Test repr for failed evaluation."""
        result = EvaluationResult(
            output="test",
            overall_score=0.45,
            passed=False,
            processing_time=0.5,
        )
        repr_str = repr(result)
        assert "failed=True" in repr_str


class TestEvaluationResultSummary:
    """Tests for EvaluationResult.summary()."""

    def test_summary_passed(self) -> None:
        """Test summary for passed evaluation."""
        result = EvaluationResult(
            output="test",
            overall_score=0.87,
            passed=True,
            processing_time=1.23,
            scores=[
                Score(name="semantic", value=0.92),
                Score(name="factuality", value=0.82),
            ],
        )
        summary = result.summary()
        assert "✓ PASSED" in summary
        assert "(0.87)" in summary
        assert "semantic: 0.92" in summary
        assert "factuality: 0.82" in summary
        assert "1.23s" in summary

    def test_summary_failed(self) -> None:
        """Test summary for failed evaluation."""
        result = EvaluationResult(
            output="test",
            overall_score=0.45,
            passed=False,
            processing_time=0.98,
            scores=[Score(name="semantic", value=0.45)],
        )
        summary = result.summary()
        assert "✗ FAILED" in summary
        assert "(0.45)" in summary

    def test_summary_is_one_line(self) -> None:
        """Verify summary is a single line."""
        result = EvaluationResult(
            output="test",
            overall_score=0.85,
            passed=True,
            processing_time=1.0,
        )
        summary = result.summary()
        assert "\n" not in summary


class TestEvaluationResultExport:
    """Tests for EvaluationResult.to_dict() and to_json()."""

    @pytest.fixture
    def eval_result(self) -> EvaluationResult:
        """Create a sample EvaluationResult."""
        return EvaluationResult(
            output="Test output",
            reference="Test reference",
            overall_score=0.85,
            passed=True,
            processing_time=1.5,
            scores=[Score(name="semantic", value=0.85)],
            metadata={"key": "value"},
        )

    def test_to_dict_basic(self, eval_result: EvaluationResult) -> None:
        """Test basic to_dict export."""
        data = eval_result.to_dict()
        assert isinstance(data, dict)
        assert data["output"] == "Test output"
        assert data["overall_score"] == 0.85
        assert data["passed"] is True
        assert "timestamp" in data

    def test_to_dict_exclude(self, eval_result: EvaluationResult) -> None:
        """Test to_dict with exclusions."""
        data = eval_result.to_dict(exclude={"metadata", "interactions"})
        assert "metadata" not in data
        assert "interactions" not in data
        assert "output" in data

    def test_to_dict_exclude_timestamp(self, eval_result: EvaluationResult) -> None:
        """Test to_dict with timestamp exclusion."""
        data = eval_result.to_dict(include_timestamp=False)
        assert "timestamp" not in data

    def test_to_json_basic(self, eval_result: EvaluationResult) -> None:
        """Test basic to_json export."""
        json_str = eval_result.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["output"] == "Test output"

    def test_to_json_indent(self, eval_result: EvaluationResult) -> None:
        """Test to_json with indentation."""
        json_str = eval_result.to_json(indent=2)
        assert "\n" in json_str
        assert "  " in json_str

    def test_to_json_exclude(self, eval_result: EvaluationResult) -> None:
        """Test to_json with exclusions."""
        json_str = eval_result.to_json(exclude={"metadata"})
        data = json.loads(json_str)
        assert "metadata" not in data


class TestComparisonResultRepr:
    """Tests for ComparisonResult.__repr__()."""

    def test_repr_basic(self) -> None:
        """Test basic repr output."""
        result = ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.85,
            reasoning="A is better",
            processing_time=1.45,
        )
        repr_str = repr(result)
        assert "<ComparisonResult" in repr_str
        assert "winner='output_a'" in repr_str
        assert "confidence=0.85" in repr_str
        assert "time=1.45s" in repr_str


class TestComparisonResultSummary:
    """Tests for ComparisonResult.summary()."""

    def test_summary_output_a_wins(self) -> None:
        """Test summary when output_a wins."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.85,
            reasoning="A is better",
            processing_time=1.45,
        )
        summary = result.summary()
        assert "Winner: Output A" in summary
        assert "(0.85 confidence)" in summary
        assert "1.45s" in summary

    def test_summary_output_b_wins(self) -> None:
        """Test summary when output_b wins."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_b",
            confidence=0.92,
            reasoning="B is better",
            processing_time=2.0,
        )
        summary = result.summary()
        assert "Winner: Output B" in summary

    def test_summary_tie(self) -> None:
        """Test summary for tie."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="tie",
            confidence=0.5,
            reasoning="Equal",
            processing_time=1.0,
        )
        summary = result.summary()
        assert "Winner: Tie" in summary

    def test_summary_is_one_line(self) -> None:
        """Verify summary is a single line."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.85,
            reasoning="Test",
            processing_time=1.0,
        )
        summary = result.summary()
        assert "\n" not in summary


class TestComparisonResultExport:
    """Tests for ComparisonResult.to_dict() and to_json()."""

    @pytest.fixture
    def comparison_result(self) -> ComparisonResult:
        """Create a sample ComparisonResult."""
        return ComparisonResult(
            output_a="Output A",
            output_b="Output B",
            winner="output_a",
            confidence=0.85,
            reasoning="A is better because...",
            processing_time=1.45,
        )

    def test_to_dict_basic(self, comparison_result: ComparisonResult) -> None:
        """Test basic to_dict export."""
        data = comparison_result.to_dict()
        assert isinstance(data, dict)
        assert data["winner"] == "output_a"
        assert data["confidence"] == 0.85

    def test_to_dict_exclude(self, comparison_result: ComparisonResult) -> None:
        """Test to_dict with exclusions."""
        data = comparison_result.to_dict(exclude={"output_a", "output_b"})
        assert "output_a" not in data
        assert "output_b" not in data
        assert "winner" in data

    def test_to_json_basic(self, comparison_result: ComparisonResult) -> None:
        """Test basic to_json export."""
        json_str = comparison_result.to_json()
        data = json.loads(json_str)
        assert data["winner"] == "output_a"


class TestBatchEvaluationResultRepr:
    """Tests for BatchEvaluationResult.__repr__()."""

    def test_repr_basic(self) -> None:
        """Test basic repr output."""
        result = BatchEvaluationResult(
            total_items=100,
            successful_items=95,
            failed_items=5,
            processing_time=45.2,
        )
        repr_str = repr(result)
        assert "<BatchEvaluationResult" in repr_str
        assert "items=100" in repr_str
        assert "success=95" in repr_str
        assert "failed=5" in repr_str
        assert "time=45.2s" in repr_str


class TestBatchEvaluationResultSummary:
    """Tests for BatchEvaluationResult.summary()."""

    def test_summary_with_results(self) -> None:
        """Test summary with successful results."""
        eval_results = [
            EvaluationResult(
                output="test",
                overall_score=0.85,
                passed=True,
                processing_time=0.5,
            ),
            EvaluationResult(
                output="test",
                overall_score=0.87,
                passed=True,
                processing_time=0.5,
            ),
        ]
        result = BatchEvaluationResult(
            results=eval_results,
            total_items=2,
            successful_items=2,
            failed_items=0,
            processing_time=1.0,
        )
        summary = result.summary()
        assert "2/2 passed (100.0%)" in summary
        assert "mean: 0.86" in summary
        assert "1.0s" in summary

    def test_summary_with_failures(self) -> None:
        """Test summary with some failures."""
        result = BatchEvaluationResult(
            results=[None, None],
            total_items=2,
            successful_items=0,
            failed_items=2,
            processing_time=0.5,
        )
        summary = result.summary()
        assert "0/2 passed (0.0%)" in summary
        assert "no results" in summary

    def test_summary_is_one_line(self) -> None:
        """Verify summary is a single line."""
        result = BatchEvaluationResult(
            total_items=10,
            successful_items=8,
            failed_items=2,
            processing_time=5.0,
        )
        summary = result.summary()
        assert "\n" not in summary


class TestBatchEvaluationResultExport:
    """Tests for BatchEvaluationResult.to_dict() and to_json()."""

    @pytest.fixture
    def batch_result(self) -> BatchEvaluationResult:
        """Create a sample BatchEvaluationResult."""
        return BatchEvaluationResult(
            results=[
                EvaluationResult(
                    output="test",
                    overall_score=0.85,
                    passed=True,
                    processing_time=0.5,
                )
            ],
            total_items=1,
            successful_items=1,
            failed_items=0,
            processing_time=0.5,
            metadata={"batch_id": "test123"},
        )

    def test_to_dict_basic(self, batch_result: BatchEvaluationResult) -> None:
        """Test basic to_dict export."""
        data = batch_result.to_dict()
        assert isinstance(data, dict)
        assert data["total_items"] == 1
        assert data["successful_items"] == 1

    def test_to_dict_exclude(self, batch_result: BatchEvaluationResult) -> None:
        """Test to_dict with exclusions."""
        data = batch_result.to_dict(exclude={"results", "metadata"})
        assert "results" not in data
        assert "metadata" not in data
        assert "total_items" in data

    def test_to_json_basic(self, batch_result: BatchEvaluationResult) -> None:
        """Test basic to_json export."""
        json_str = batch_result.to_json()
        data = json.loads(json_str)
        assert data["total_items"] == 1

    def test_to_json_indent(self, batch_result: BatchEvaluationResult) -> None:
        """Test to_json with indentation."""
        json_str = batch_result.to_json(indent=2)
        assert "\n" in json_str


class TestDatetimeSerialization:
    """Tests for datetime serialization in to_json()."""

    def test_evaluation_result_datetime(self) -> None:
        """Test that datetime is properly serialized in EvaluationResult."""
        result = EvaluationResult(
            output="test",
            overall_score=0.85,
            passed=True,
            processing_time=1.0,
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "2024-01-15" in data["timestamp"]

    def test_comparison_result_datetime(self) -> None:
        """Test that datetime is properly serialized in ComparisonResult."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.85,
            reasoning="Test",
            processing_time=1.0,
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "2024-01-15" in data["timestamp"]

    def test_batch_result_datetime(self) -> None:
        """Test that datetime is properly serialized in BatchEvaluationResult."""
        result = BatchEvaluationResult(
            total_items=1,
            successful_items=1,
            failed_items=0,
            processing_time=1.0,
            timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        )
        json_str = result.to_json()
        data = json.loads(json_str)
        assert "2024-01-15" in data["timestamp"]
