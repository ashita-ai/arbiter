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
        assert json_str is not None
        data = json.loads(json_str)
        # New format has summary/results/errors structure
        assert data["summary"]["total_items"] == 1

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
        assert json_str is not None
        data = json.loads(json_str)
        assert "summary" in data


class TestBatchExportMethods:
    """Tests for BatchEvaluationResult export methods (issue #40)."""

    @pytest.fixture
    def batch_with_results(self) -> BatchEvaluationResult:
        """Create a BatchEvaluationResult with mixed success/failure."""
        return BatchEvaluationResult(
            results=[
                EvaluationResult(
                    output="Paris is the capital of France",
                    reference="The capital of France is Paris",
                    overall_score=0.92,
                    passed=True,
                    processing_time=0.5,
                    scores=[
                        Score(name="semantic", value=0.92, confidence=0.88),
                        Score(name="factuality", value=0.95),
                    ],
                ),
                EvaluationResult(
                    output="Tokyo is in Japan",
                    reference="Tokyo is the capital of Japan",
                    overall_score=0.85,
                    passed=True,
                    processing_time=0.4,
                    scores=[Score(name="semantic", value=0.85)],
                ),
                None,  # Failed item
            ],
            errors=[
                {
                    "index": 2,
                    "item": {"output": "Bad input", "reference": "Expected"},
                    "error": "Rate limit exceeded",
                }
            ],
            total_items=3,
            successful_items=2,
            failed_items=1,
            processing_time=1.2,
            metadata={"evaluators": ["semantic"]},
        )

    @pytest.fixture
    def empty_batch(self) -> BatchEvaluationResult:
        """Create an empty BatchEvaluationResult."""
        return BatchEvaluationResult(
            results=[],
            total_items=0,
            successful_items=0,
            failed_items=0,
            processing_time=0.0,
        )

    # to_records() tests
    def test_to_records_basic(self, batch_with_results: BatchEvaluationResult) -> None:
        """Test to_records returns list of dicts."""
        records = batch_with_results.to_records()
        assert isinstance(records, list)
        assert len(records) == 3

    def test_to_records_successful_item(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_records includes correct fields for successful items."""
        records = batch_with_results.to_records()
        first = records[0]

        assert first["index"] == 0
        assert first["output"] == "Paris is the capital of France"
        assert first["overall_score"] == 0.92
        assert first["passed"] is True
        assert first["semantic_score"] == 0.92
        assert first["semantic_confidence"] == 0.88
        assert first["factuality_score"] == 0.95
        assert "error" not in first

    def test_to_records_failed_item(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_records includes error info for failed items."""
        records = batch_with_results.to_records()
        failed = records[2]

        assert failed["index"] == 2
        assert failed["overall_score"] is None
        assert failed["passed"] is None
        assert failed["error"] == "Rate limit exceeded"

    def test_to_records_empty(self, empty_batch: BatchEvaluationResult) -> None:
        """Test to_records with empty batch."""
        records = empty_batch.to_records()
        assert records == []

    # to_json() tests
    def test_to_json_returns_string(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_json returns JSON string when no path provided."""
        result = batch_with_results.to_json()
        assert isinstance(result, str)
        data = json.loads(result)
        assert "summary" in data
        assert "results" in data

    def test_to_json_summary_structure(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_json summary contains expected fields."""
        result = batch_with_results.to_json()
        assert result is not None
        data = json.loads(result)
        summary = data["summary"]

        assert summary["total_items"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert "mean_score" in summary
        assert summary["mean_score"] == pytest.approx(0.885, rel=0.01)

    def test_to_json_writes_file(
        self,
        batch_with_results: BatchEvaluationResult,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test to_json writes to file when path provided."""
        file_path = tmp_path / "results.json"  # type: ignore[operator]
        result = batch_with_results.to_json(str(file_path))

        assert result is None  # Returns None when writing to file
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)
        assert data["summary"]["total_items"] == 3

    def test_to_json_exclude(self, batch_with_results: BatchEvaluationResult) -> None:
        """Test to_json with exclude parameter."""
        result = batch_with_results.to_json(exclude={"errors", "metadata"})
        assert result is not None
        data = json.loads(result)
        assert "errors" not in data
        assert "metadata" not in data
        assert "summary" in data

    # to_csv() tests
    def test_to_csv_creates_file(
        self,
        batch_with_results: BatchEvaluationResult,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test to_csv creates a CSV file."""
        file_path = tmp_path / "results.csv"  # type: ignore[operator]
        batch_with_results.to_csv(str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        assert "index" in content
        assert "overall_score" in content

    def test_to_csv_content(
        self,
        batch_with_results: BatchEvaluationResult,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test to_csv contains expected data."""
        import csv

        file_path = tmp_path / "results.csv"  # type: ignore[operator]
        batch_with_results.to_csv(str(file_path))

        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]["overall_score"] == "0.92"
        assert rows[2]["error"] == "Rate limit exceeded"

    def test_to_csv_no_header(
        self,
        batch_with_results: BatchEvaluationResult,
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Test to_csv without header."""
        file_path = tmp_path / "results.csv"  # type: ignore[operator]
        batch_with_results.to_csv(str(file_path), include_header=False)

        content = file_path.read_text()
        assert "index" not in content.split("\n")[0]  # First line is data, not header

    def test_to_csv_empty_batch(
        self, empty_batch: BatchEvaluationResult, tmp_path: pytest.TempPathFactory
    ) -> None:
        """Test to_csv with empty batch."""
        file_path = tmp_path / "results.csv"  # type: ignore[operator]
        empty_batch.to_csv(str(file_path))

        assert file_path.exists()
        content = file_path.read_text()
        # Should have header but no data rows
        lines = [line for line in content.strip().split("\n") if line]
        assert len(lines) == 1  # Just header

    # to_dataframe() tests
    def test_to_dataframe_returns_dataframe(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_dataframe returns pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = batch_with_results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_to_dataframe_columns(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_dataframe has expected columns."""
        pytest.importorskip("pandas")

        df = batch_with_results.to_dataframe()
        assert "index" in df.columns
        assert "overall_score" in df.columns
        assert "semantic_score" in df.columns

    def test_to_dataframe_values(
        self, batch_with_results: BatchEvaluationResult
    ) -> None:
        """Test to_dataframe contains correct values."""
        pytest.importorskip("pandas")
        import pandas as pd

        df = batch_with_results.to_dataframe()

        # First row should have score
        assert df.iloc[0]["overall_score"] == 0.92

        # Last row should have None/NaN for score
        assert pd.isna(df.iloc[2]["overall_score"])

    def test_to_dataframe_without_pandas(
        self, batch_with_results: BatchEvaluationResult, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test to_dataframe raises ImportError without pandas."""
        import sys

        # Hide pandas
        monkeypatch.setitem(sys.modules, "pandas", None)

        with pytest.raises(ImportError, match="pandas is required"):
            batch_with_results.to_dataframe()


class TestBatchEvaluationResultFiltering:
    """Tests for BatchEvaluationResult filtering methods (#73)."""

    @pytest.fixture
    def batch_with_varied_results(self) -> BatchEvaluationResult:
        """Create a BatchEvaluationResult with varied scores for filtering tests."""
        return BatchEvaluationResult(
            results=[
                EvaluationResult(
                    output="High quality output",
                    reference="Reference text",
                    overall_score=0.95,
                    passed=True,
                    processing_time=0.3,
                    scores=[Score(name="semantic", value=0.95, confidence=0.9)],
                ),
                EvaluationResult(
                    output="Medium quality output",
                    reference="Reference text",
                    overall_score=0.75,
                    passed=True,
                    processing_time=0.4,
                    scores=[Score(name="semantic", value=0.75, confidence=0.85)],
                ),
                EvaluationResult(
                    output="Low quality output",
                    reference="Reference text",
                    overall_score=0.45,
                    passed=False,
                    processing_time=0.35,
                    scores=[Score(name="semantic", value=0.45, confidence=0.8)],
                ),
                None,  # Failed item
                EvaluationResult(
                    output="Borderline output",
                    reference="Reference text",
                    overall_score=0.70,
                    passed=False,
                    processing_time=0.5,
                    scores=[Score(name="semantic", value=0.70, confidence=0.75)],
                ),
            ],
            errors=[
                {
                    "index": 3,
                    "item": {"output": "Error output", "reference": "Ref"},
                    "error": "Evaluation timed out",
                }
            ],
            total_items=5,
            successful_items=4,
            failed_items=1,
            processing_time=2.0,
        )

    @pytest.fixture
    def empty_batch(self) -> BatchEvaluationResult:
        """Create an empty BatchEvaluationResult."""
        return BatchEvaluationResult(
            results=[],
            total_items=0,
            successful_items=0,
            failed_items=0,
            processing_time=0.0,
        )

    # filter() tests
    def test_filter_by_passed_true(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter returns only passed results."""
        passed = batch_with_varied_results.filter(passed=True)
        assert len(passed) == 2
        assert all(r.passed for r in passed)
        assert passed[0].overall_score == 0.95
        assert passed[1].overall_score == 0.75

    def test_filter_by_passed_false(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter returns only failed results."""
        failed = batch_with_varied_results.filter(passed=False)
        assert len(failed) == 2
        assert all(not r.passed for r in failed)

    def test_filter_by_min_score(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter with min_score threshold."""
        high_quality = batch_with_varied_results.filter(min_score=0.9)
        assert len(high_quality) == 1
        assert high_quality[0].overall_score == 0.95

    def test_filter_by_max_score(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter with max_score threshold."""
        low_quality = batch_with_varied_results.filter(max_score=0.5)
        assert len(low_quality) == 1
        assert low_quality[0].overall_score == 0.45

    def test_filter_by_score_range(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter with both min and max score."""
        mid_range = batch_with_varied_results.filter(min_score=0.6, max_score=0.8)
        assert len(mid_range) == 2
        # Should include 0.75 and 0.70
        scores = {r.overall_score for r in mid_range}
        assert scores == {0.75, 0.70}

    def test_filter_combined_criteria(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter with combined passed and score criteria."""
        needs_review = batch_with_varied_results.filter(passed=False, min_score=0.6)
        assert len(needs_review) == 1
        assert needs_review[0].overall_score == 0.70
        assert not needs_review[0].passed

    def test_filter_no_criteria_returns_all_valid(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter without criteria returns all non-None results."""
        all_valid = batch_with_varied_results.filter()
        assert len(all_valid) == 4  # 5 total minus 1 None

    def test_filter_no_matches(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test filter returns empty list when no matches."""
        impossible = batch_with_varied_results.filter(min_score=0.99)
        assert impossible == []

    def test_filter_empty_batch(self, empty_batch: BatchEvaluationResult) -> None:
        """Test filter on empty batch returns empty list."""
        result = empty_batch.filter(passed=True)
        assert result == []

    def test_filter_excludes_none_results(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test that filter always excludes None results."""
        # Even with no filter criteria, None should be excluded
        all_results = batch_with_varied_results.filter()
        assert None not in all_results
        assert len(all_results) == 4

    # slice() tests
    def test_slice_basic(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test slice returns correct range."""
        first_three = batch_with_varied_results.slice(0, 3)
        assert len(first_three) == 3
        assert first_three[0].overall_score == 0.95  # type: ignore[union-attr]
        assert first_three[1].overall_score == 0.75  # type: ignore[union-attr]
        assert first_three[2].overall_score == 0.45  # type: ignore[union-attr]

    def test_slice_includes_none(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test slice includes None values for failed items."""
        middle = batch_with_varied_results.slice(2, 5)
        assert len(middle) == 3
        assert middle[1] is None  # The failed item at index 3

    def test_slice_out_of_range(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test slice with out-of-range indices works like list slicing."""
        result = batch_with_varied_results.slice(10, 20)
        assert result == []

    def test_slice_partial_range(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test slice with partial out-of-range returns available items."""
        result = batch_with_varied_results.slice(3, 100)
        assert len(result) == 2  # Only items at index 3 and 4

    def test_slice_empty_batch(self, empty_batch: BatchEvaluationResult) -> None:
        """Test slice on empty batch returns empty list."""
        result = empty_batch.slice(0, 10)
        assert result == []

    # get_failed_items() tests
    def test_get_failed_items_returns_copy(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_failed_items returns a copy of errors."""
        failed = batch_with_varied_results.get_failed_items()
        assert len(failed) == 1
        # Modify the returned list and verify original is unchanged
        failed.append({"index": 99, "error": "test"})
        assert len(batch_with_varied_results.errors) == 1

    def test_get_failed_items_structure(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_failed_items returns correct error structure."""
        failed = batch_with_varied_results.get_failed_items()
        assert failed[0]["index"] == 3
        assert failed[0]["error"] == "Evaluation timed out"
        assert "item" in failed[0]

    def test_get_failed_items_empty(self, empty_batch: BatchEvaluationResult) -> None:
        """Test get_failed_items on batch with no errors."""
        failed = empty_batch.get_failed_items()
        assert failed == []

    # get_results_with_indices() tests
    def test_get_results_with_indices_basic(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_results_with_indices returns indexed tuples."""
        indexed = batch_with_varied_results.get_results_with_indices()
        assert len(indexed) == 5
        assert all(isinstance(item, tuple) for item in indexed)
        assert all(len(item) == 2 for item in indexed)

    def test_get_results_with_indices_correct_indices(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_results_with_indices has correct index values."""
        indexed = batch_with_varied_results.get_results_with_indices()
        for i, (idx, _) in enumerate(indexed):
            assert idx == i

    def test_get_results_with_indices_includes_none(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_results_with_indices includes None entries."""
        indexed = batch_with_varied_results.get_results_with_indices()
        idx, result = indexed[3]
        assert idx == 3
        assert result is None

    def test_get_results_with_indices_empty(
        self, empty_batch: BatchEvaluationResult
    ) -> None:
        """Test get_results_with_indices on empty batch."""
        indexed = empty_batch.get_results_with_indices()
        assert indexed == []

    def test_get_results_with_indices_preserves_order(
        self, batch_with_varied_results: BatchEvaluationResult
    ) -> None:
        """Test get_results_with_indices maintains original order."""
        indexed = batch_with_varied_results.get_results_with_indices()
        # Check specific results at expected positions
        _, first = indexed[0]
        assert first is not None
        assert first.overall_score == 0.95
        _, last = indexed[4]
        assert last is not None
        assert last.overall_score == 0.70
