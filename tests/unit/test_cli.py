"""Unit tests for CLI module.

Tests cover:
- evaluate command
- batch command
- compare command
- list-evaluators command
- cost command
- Input file loading
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from arbiter_ai.cli import _load_input_file, app

runner = CliRunner()


class TestEvaluateCommand:
    """Tests for the evaluate command."""

    @patch("arbiter_ai.evaluate")
    def test_evaluate_basic(self, mock_evaluate: MagicMock) -> None:
        """Test basic evaluate command."""
        # Create mock result
        mock_result = MagicMock()
        mock_result.overall_score = 0.85
        mock_result.passed = True
        mock_result.scores = []
        mock_result.interactions = []
        mock_result.processing_time = 1.0
        mock_result.total_llm_cost = AsyncMock(return_value=0.001)

        mock_evaluate.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--output",
                "Test output",
                "--reference",
                "Test reference",
                "--evaluators",
                "semantic",
            ],
        )

        assert result.exit_code == 0
        assert "0.85" in result.output
        mock_evaluate.assert_called_once()

    @patch("arbiter_ai.evaluate")
    def test_evaluate_json_format(self, mock_evaluate: MagicMock) -> None:
        """Test evaluate with JSON output format."""
        mock_result = MagicMock()
        mock_result.overall_score = 0.85
        mock_result.passed = True
        mock_result.scores = []
        mock_result.total_llm_cost = AsyncMock(return_value=0.001)

        mock_evaluate.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--output",
                "Test",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        # Should be valid JSON
        output_data = json.loads(result.output.strip())
        assert output_data["overall_score"] == 0.85

    @patch("arbiter_ai.evaluate")
    def test_evaluate_quiet_format(self, mock_evaluate: MagicMock) -> None:
        """Test evaluate with quiet output format."""
        mock_result = MagicMock()
        mock_result.overall_score = 0.92

        mock_evaluate.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--output",
                "Test",
                "--format",
                "quiet",
            ],
        )

        assert result.exit_code == 0
        assert "0.92" in result.output


class TestBatchCommand:
    """Tests for the batch command."""

    @patch("arbiter_ai.batch_evaluate")
    def test_batch_with_jsonl_file(self, mock_batch: MagicMock, tmp_path: Path) -> None:
        """Test batch command with JSONL input."""
        # Create test input file
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            '{"output": "test1", "reference": "ref1"}\n'
            '{"output": "test2", "reference": "ref2"}\n'
        )

        # Mock batch result
        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"summary": {"total_items": 2}}'
        mock_result.summary.return_value = "2/2 passed"

        mock_batch.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "batch",
                "--file",
                str(input_file),
                "--evaluators",
                "semantic",
            ],
        )

        assert result.exit_code == 0
        mock_batch.assert_called_once()

    @patch("arbiter_ai.batch_evaluate")
    def test_batch_with_output_file(
        self, mock_batch: MagicMock, tmp_path: Path
    ) -> None:
        """Test batch command writing to output file."""
        input_file = tmp_path / "input.jsonl"
        input_file.write_text('{"output": "test"}\n')

        output_file = tmp_path / "output.json"

        mock_result = MagicMock()
        mock_result.to_json.return_value = '{"summary": {}}'
        mock_result.summary.return_value = "1/1"

        mock_batch.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "batch",
                "--file",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestCompareCommand:
    """Tests for the compare command."""

    @patch("arbiter_ai.compare")
    def test_compare_basic(self, mock_compare: MagicMock) -> None:
        """Test basic compare command."""
        mock_result = MagicMock()
        mock_result.winner = "output_a"
        mock_result.confidence = 0.85
        mock_result.reasoning = "A is better"
        mock_result.aspect_scores = {}

        mock_compare.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "compare",
                "--output-a",
                "Response A",
                "--output-b",
                "Response B",
            ],
        )

        assert result.exit_code == 0
        assert "Output A" in result.output
        mock_compare.assert_called_once()

    @patch("arbiter_ai.compare")
    def test_compare_tie(self, mock_compare: MagicMock) -> None:
        """Test compare command with tie result."""
        mock_result = MagicMock()
        mock_result.winner = "tie"
        mock_result.confidence = 0.5
        mock_result.reasoning = "Equal quality"
        mock_result.aspect_scores = {}

        mock_compare.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "compare",
                "--output-a",
                "A",
                "--output-b",
                "B",
            ],
        )

        assert result.exit_code == 0
        assert "Tie" in result.output

    @patch("arbiter_ai.compare")
    def test_compare_json_format(self, mock_compare: MagicMock) -> None:
        """Test compare with JSON output format."""
        mock_result = MagicMock()
        mock_result.winner = "output_b"
        mock_result.confidence = 0.9
        mock_result.reasoning = "B is better"
        mock_result.aspect_scores = {"clarity": {"output_a": 0.7, "output_b": 0.9}}

        mock_compare.return_value = mock_result

        result = runner.invoke(
            app,
            [
                "compare",
                "--output-a",
                "A",
                "--output-b",
                "B",
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0
        output_data = json.loads(result.output.strip())
        assert output_data["winner"] == "output_b"


class TestListEvaluatorsCommand:
    """Tests for the list-evaluators command."""

    def test_list_evaluators(self) -> None:
        """Test list-evaluators command."""
        result = runner.invoke(app, ["list-evaluators"])

        assert result.exit_code == 0
        assert "semantic" in result.output.lower()


class TestCostCommand:
    """Tests for the cost command."""

    @patch("arbiter_ai.core.cost_calculator.get_cost_calculator")
    def test_cost_basic(self, mock_get_calc: MagicMock) -> None:
        """Test basic cost command."""
        mock_calc = MagicMock()
        mock_calc.calculate_cost.return_value = 0.00045
        mock_get_calc.return_value = mock_calc

        result = runner.invoke(
            app,
            [
                "cost",
                "--model",
                "gpt-4o-mini",
                "--input-tokens",
                "1000",
                "--output-tokens",
                "500",
            ],
        )

        assert result.exit_code == 0
        assert "0.00045" in result.output
        mock_calc.calculate_cost.assert_called_once_with(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
            cached_tokens=0,
        )


class TestLoadInputFile:
    """Tests for the _load_input_file helper."""

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Test loading JSONL file."""
        file_path = tmp_path / "test.jsonl"
        file_path.write_text('{"output": "a"}\n' '{"output": "b"}\n')

        items = _load_input_file(file_path)
        assert len(items) == 2
        assert items[0]["output"] == "a"
        assert items[1]["output"] == "b"

    def test_load_json_array(self, tmp_path: Path) -> None:
        """Test loading JSON array file."""
        file_path = tmp_path / "test.json"
        file_path.write_text('[{"output": "a"}, {"output": "b"}]')

        items = _load_input_file(file_path)
        assert len(items) == 2

    def test_load_json_single_object(self, tmp_path: Path) -> None:
        """Test loading JSON single object file."""
        file_path = tmp_path / "test.json"
        file_path.write_text('{"output": "single"}')

        items = _load_input_file(file_path)
        assert len(items) == 1
        assert items[0]["output"] == "single"

    def test_load_csv(self, tmp_path: Path) -> None:
        """Test loading CSV file."""
        file_path = tmp_path / "test.csv"
        file_path.write_text("output,reference\na,ref_a\nb,ref_b\n")

        items = _load_input_file(file_path)
        assert len(items) == 2
        assert items[0]["output"] == "a"
        assert items[0]["reference"] == "ref_a"

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading nonexistent file via CLI."""
        file_path = tmp_path / "nonexistent.jsonl"

        result = runner.invoke(
            app,
            ["batch", "--file", str(file_path)],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_load_unsupported_format(self, tmp_path: Path) -> None:
        """Test loading unsupported file format via CLI."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("some text")

        result = runner.invoke(
            app,
            ["batch", "--file", str(file_path)],
        )
        assert result.exit_code != 0
