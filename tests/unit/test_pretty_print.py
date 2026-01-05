"""Unit tests for pretty_print() methods in result models.

Tests terminal-friendly output formatting for:
- EvaluationResult.pretty_print()
- ComparisonResult.pretty_print()
- BatchEvaluationResult.pretty_print()
"""

from io import StringIO

from arbiter_ai.core.models import (
    BatchEvaluationResult,
    ComparisonResult,
    EvaluationResult,
    LLMInteraction,
    Score,
)


class TestEvaluationResultPrettyPrint:
    """Test EvaluationResult.pretty_print() method."""

    def test_basic_output(self):
        """Test basic pretty print output with passing result."""
        result = EvaluationResult(
            output="Paris is the capital of France",
            reference="Paris",
            scores=[
                Score(name="semantic", value=0.92, confidence=0.88),
                Score(name="factuality", value=0.82, confidence=0.85),
            ],
            overall_score=0.87,
            passed=True,
            processing_time=1.23,
            interactions=[
                LLMInteraction(
                    prompt="test",
                    response="test",
                    model="gpt-4o-mini",
                    input_tokens=100,
                    output_tokens=50,
                    latency=0.5,
                    purpose="scoring",
                ),
                LLMInteraction(
                    prompt="test2",
                    response="test2",
                    model="gpt-4o-mini",
                    input_tokens=120,
                    output_tokens=60,
                    latency=0.73,
                    purpose="factuality",
                ),
            ],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        # Check header
        assert "Evaluation Results" in text
        assert "==================" in text

        # Check overall score and status
        assert "Overall Score: 0.87 ✓ PASSED" in text

        # Check individual scores
        assert "Scores:" in text
        assert "semantic" in text
        assert "0.92" in text
        assert "confidence: 0.88" in text
        assert "factuality" in text
        assert "0.82" in text

        # Check stats
        assert "Time: 1.23s" in text
        assert "LLM Calls: 2" in text

        # Unicode symbols present
        assert "✓" in text
        assert "•" in text

    def test_failed_result(self):
        """Test pretty print output with failed result."""
        result = EvaluationResult(
            output="Wrong answer",
            reference="Correct answer",
            scores=[Score(name="semantic", value=0.35, confidence=0.9)],
            overall_score=0.35,
            passed=False,
            processing_time=0.8,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        assert "Overall Score: 0.35 ✗ FAILED" in text
        assert "✗" in text

    def test_partial_result_with_errors(self):
        """Test pretty print with partial result showing errors."""
        result = EvaluationResult(
            output="Test output",
            scores=[Score(name="semantic", value=0.85, confidence=0.9)],
            overall_score=0.85,
            passed=True,
            partial=True,
            errors={"factuality": "API timeout", "groundedness": "Rate limit exceeded"},
            processing_time=2.5,
            interactions=[
                LLMInteraction(
                    prompt="test",
                    response="test",
                    model="gpt-4o-mini",
                    input_tokens=100,
                    output_tokens=50,
                    latency=1.0,
                    purpose="semantic",
                )
            ],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        # Check partial warning
        assert "⚠ PARTIAL RESULT" in text
        assert "some evaluators failed" in text

        # Check errors section
        assert "Errors:" in text
        assert "factuality: API timeout" in text
        assert "groundedness: Rate limit exceeded" in text

    def test_verbose_output(self):
        """Test verbose mode shows detailed token breakdown and interactions."""
        result = EvaluationResult(
            output="Test output",
            scores=[Score(name="semantic", value=0.92)],
            overall_score=0.92,
            passed=True,
            processing_time=1.5,
            interactions=[
                LLMInteraction(
                    prompt="test prompt",
                    response="test response",
                    model="gpt-4o-mini",
                    input_tokens=1200,
                    output_tokens=300,
                    cached_tokens=500,
                    latency=0.8,
                    purpose="semantic_scoring",
                ),
                LLMInteraction(
                    prompt="test prompt 2",
                    response="test response 2",
                    model="claude-3-5-sonnet",
                    input_tokens=800,
                    output_tokens=200,
                    cached_tokens=0,
                    latency=1.2,
                    purpose="factuality_check",
                ),
            ],
        )

        output = StringIO()
        result.pretty_print(file=output, verbose=True)
        text = output.getvalue()

        # Check token breakdown
        assert "Token Usage:" in text
        assert "Input:  2,000" in text
        assert "Output: 500" in text
        assert "Cached: 500" in text
        assert "Total:  2,500" in text

        # Check interactions
        assert "Interactions:" in text
        assert "1. semantic_scoring (gpt-4o-mini)" in text
        assert "1200→300 tokens" in text
        assert "0.80s" in text
        assert "2. factuality_check (claude-3-5-sonnet)" in text
        assert "800→200 tokens" in text
        assert "1.20s" in text

    def test_no_scores(self):
        """Test pretty print with no scores (all evaluators failed)."""
        result = EvaluationResult(
            output="Test output",
            scores=[],
            overall_score=0.0,
            passed=False,
            partial=True,
            errors={"semantic": "Model unavailable"},
            processing_time=0.1,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        assert "Overall Score: 0.00 ✗ FAILED" in text
        assert "Errors:" in text
        # Should not have Scores section if no scores
        assert "Scores:" not in text or text.count("•") == 0

    def test_output_to_stdout_default(self, capsys):
        """Test that output goes to stdout by default."""
        result = EvaluationResult(
            output="Test",
            scores=[Score(name="test", value=0.8)],
            overall_score=0.8,
            passed=True,
            processing_time=0.5,
            interactions=[],
        )

        result.pretty_print()  # No file parameter
        captured = capsys.readouterr()

        assert "Evaluation Results" in captured.out
        assert "Overall Score: 0.80 ✓ PASSED" in captured.out


class TestComparisonResultPrettyPrint:
    """Test ComparisonResult.pretty_print() method."""

    def test_basic_output_winner_a(self):
        """Test basic comparison output with output_a as winner."""
        result = ComparisonResult(
            output_a="GPT-4 response",
            output_b="Claude response",
            winner="output_a",
            confidence=0.85,
            reasoning="Output A is more accurate and provides better context with specific examples.",
            processing_time=1.45,
            interactions=[
                LLMInteraction(
                    prompt="compare",
                    response="winner: a",
                    model="gpt-4o-mini",
                    input_tokens=200,
                    output_tokens=100,
                    latency=1.45,
                    purpose="comparison",
                )
            ],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        # Check header
        assert "Comparison Results" in text
        assert "==================" in text

        # Check winner
        assert "Winner: Output A ✓" in text
        assert "Confidence: 0.85" in text

        # Check reasoning
        assert "Reasoning:" in text
        assert "Output A is more accurate" in text

        # Check stats
        assert "Time: 1.45s" in text
        assert "LLM Calls: 1" in text

    def test_tie_result(self):
        """Test comparison output with tie result."""
        result = ComparisonResult(
            output_a="Response A",
            output_b="Response B",
            winner="tie",
            confidence=0.92,
            reasoning="Both outputs are equally good with comparable quality.",
            processing_time=1.2,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        assert "Winner: Tie =" in text
        assert "Confidence: 0.92" in text

    def test_long_reasoning_truncated(self):
        """Test that long reasoning is truncated in non-verbose mode."""
        long_reasoning = "This is a very long reasoning " * 20  # > 200 chars
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_b",
            confidence=0.8,
            reasoning=long_reasoning,
            processing_time=1.0,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        # Should be truncated with "..."
        assert "..." in text
        # Check that it's actually truncated
        assert len(text) < len(long_reasoning)

    def test_verbose_shows_full_reasoning(self):
        """Test verbose mode shows full reasoning and aspect scores."""
        long_reasoning = "This is a very long reasoning " * 20
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.88,
            reasoning=long_reasoning,
            aspect_scores={
                "accuracy": {"output_a": 0.95, "output_b": 0.80},
                "clarity": {"output_a": 0.85, "output_b": 0.90},
                "completeness": {"output_a": 0.92, "output_b": 0.75},
            },
            processing_time=2.1,
            interactions=[
                LLMInteraction(
                    prompt="compare",
                    response="result",
                    model="gpt-4o",
                    input_tokens=500,
                    output_tokens=150,
                    latency=2.0,
                    purpose="comparison",
                )
            ],
        )

        output = StringIO()
        result.pretty_print(file=output, verbose=True)
        text = output.getvalue()

        # Full reasoning shown
        assert long_reasoning in text
        assert "..." not in text  # No truncation

        # Aspect scores shown
        assert "Aspect Scores:" in text
        assert "accuracy" in text
        assert "A: 0.95  B: 0.80" in text
        assert "clarity" in text
        assert "A: 0.85  B: 0.90" in text

        # Token usage shown
        assert "Token Usage:" in text
        assert "Input:  500" in text
        assert "Output: 150" in text

    def test_output_to_stdout_default(self, capsys):
        """Test that output goes to stdout by default."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.9,
            reasoning="A is better",
            processing_time=1.0,
            interactions=[],
        )

        result.pretty_print()
        captured = capsys.readouterr()

        assert "Comparison Results" in captured.out
        assert "Winner: Output A ✓" in captured.out


class TestBatchEvaluationResultPrettyPrint:
    """Test BatchEvaluationResult.pretty_print() method."""

    def test_basic_output(self):
        """Test basic batch evaluation output."""
        # Create successful results with scores from 0.55 to 1.00
        results = [
            EvaluationResult(
                output=f"output_{i}",
                scores=[Score(name="semantic", value=0.55 + i * 0.05)],
                overall_score=0.55 + i * 0.05,
                passed=True,
                processing_time=0.5,
                interactions=[],
            )
            for i in range(10)
        ]

        batch_result = BatchEvaluationResult(
            results=results,
            total_items=10,
            successful_items=10,
            failed_items=0,
            processing_time=5.2,
            total_tokens=1000,
            errors=[],
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        # Check header
        assert "Batch Evaluation Results" in text
        assert "========================" in text

        # Check success rate
        assert "Success: 10/10 (100.0%)" in text

        # Check statistics
        assert "Statistics:" in text
        assert "Mean:" in text
        assert "Std:" in text
        assert "Median:" in text
        assert "Range:" in text

        # Check totals
        assert "Total Time: 5.2s" in text
        assert "Total Tokens: 1,000" in text

    def test_with_failures(self):
        """Test batch output with some failures."""
        results = [
            EvaluationResult(
                output=f"output_{i}",
                scores=[Score(name="semantic", value=0.85)],
                overall_score=0.85,
                passed=True,
                processing_time=0.5,
                interactions=[],
            )
            for i in range(95)
        ]
        # Add 5 None results for failures
        results.extend([None] * 5)  # type: ignore

        batch_result = BatchEvaluationResult(
            results=results,
            total_items=100,
            successful_items=95,
            failed_items=5,
            processing_time=45.2,
            total_tokens=125430,
            errors=[
                {"index": 95, "item": {}, "error": "API timeout"},
                {"index": 96, "item": {}, "error": "Rate limit exceeded"},
                {"index": 97, "item": {}, "error": "API timeout"},
                {"index": 98, "item": {}, "error": "Invalid input"},
                {"index": 99, "item": {}, "error": "API timeout"},
            ],
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        assert "Success: 95/100 (95.0%)" in text
        assert "Failed:  5/100" in text

        # Error summary shown (not verbose)
        assert "Errors:" in text
        assert "use verbose=True for details" in text

    def test_verbose_shows_individual_results(self):
        """Test verbose mode shows individual results."""
        results = [
            EvaluationResult(
                output=f"output_{i}",
                scores=[Score(name="semantic", value=0.75 + i * 0.02)],
                overall_score=0.75 + i * 0.02,
                passed=(0.75 + i * 0.02) >= 0.7,
                processing_time=0.5 + i * 0.1,
                interactions=[],
            )
            for i in range(10)
        ]

        batch_result = BatchEvaluationResult(
            results=results,
            total_items=10,
            successful_items=10,
            failed_items=0,
            processing_time=10.0,
            total_tokens=5000,
            errors=[],
        )

        output = StringIO()
        batch_result.pretty_print(file=output, verbose=True)
        text = output.getvalue()

        # Check individual results header
        assert "Individual Results:" in text

        # Check for index markers and scores
        assert "[  0] ✓" in text
        assert "[  9] ✓" in text

        # Check pass/fail symbols
        assert text.count("✓") >= 10  # At least 10 for passed items

    def test_verbose_shows_failed_items(self):
        """Test verbose mode shows failed items with error messages."""
        results = [
            EvaluationResult(
                output="success",
                scores=[Score(name="semantic", value=0.85)],
                overall_score=0.85,
                passed=True,
                processing_time=0.5,
                interactions=[],
            ),
            None,  # Failed item
            EvaluationResult(
                output="success",
                scores=[Score(name="semantic", value=0.90)],
                overall_score=0.90,
                passed=True,
                processing_time=0.6,
                interactions=[],
            ),
        ]

        batch_result = BatchEvaluationResult(
            results=results,  # type: ignore
            total_items=3,
            successful_items=2,
            failed_items=1,
            processing_time=5.0,
            total_tokens=1000,
            errors=[{"index": 1, "item": {}, "error": "API connection failed"}],
        )

        output = StringIO()
        batch_result.pretty_print(file=output, verbose=True)
        text = output.getvalue()

        # Check individual results
        assert "[  0] ✓" in text
        assert "[  1] ✗ Failed: API connection failed" in text
        assert "[  2] ✓" in text

    def test_pass_rate_calculation(self):
        """Test pass rate is calculated correctly."""
        results = [
            EvaluationResult(
                output=f"output_{i}",
                scores=[Score(name="semantic", value=0.50 + i * 0.05)],
                overall_score=0.50 + i * 0.05,
                passed=(0.50 + i * 0.05) >= 0.7,
                processing_time=0.5,
                interactions=[],
            )
            for i in range(10)
        ]

        # 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        # Passed: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 = 6/10 = 60%

        batch_result = BatchEvaluationResult(
            results=results,
            total_items=10,
            successful_items=10,
            failed_items=0,
            processing_time=5.0,
            total_tokens=2000,
            errors=[],
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        # 6 results are >= 0.70 (indices 4-9)
        assert "Pass Rate: 60.0%" in text

    def test_statistics_calculations(self):
        """Test that statistics are calculated correctly."""
        # Create results with known distribution
        scores_list = [0.45, 0.65, 0.75, 0.85, 0.95]
        results = [
            EvaluationResult(
                output="test",
                scores=[Score(name="semantic", value=score)],
                overall_score=score,
                passed=score >= 0.7,
                processing_time=0.5,
                interactions=[],
            )
            for score in scores_list
        ]

        batch_result = BatchEvaluationResult(
            results=results,
            total_items=5,
            successful_items=5,
            failed_items=0,
            processing_time=2.5,
            total_tokens=1000,
            errors=[],
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        # Mean = (0.45 + 0.65 + 0.75 + 0.85 + 0.95) / 5 = 0.73
        assert "Mean:   0.73" in text

        # Median = 0.75 (middle value)
        assert "Median: 0.75" in text

        # Range
        assert "Range:  0.45 - 0.95" in text

    def test_error_grouping_non_verbose(self):
        """Test that errors are grouped by type in non-verbose mode."""
        results = [None] * 10  # All failed

        errors = [
            {"index": i, "item": {}, "error": "API timeout" if i < 7 else "Rate limit"}
            for i in range(10)
        ]

        batch_result = BatchEvaluationResult(
            results=results,  # type: ignore
            total_items=10,
            successful_items=0,
            failed_items=10,
            processing_time=5.0,
            total_tokens=0,
            errors=errors,
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        # Should show error summary
        assert "Errors:" in text
        assert "API timeout: 7x" in text
        assert "Rate limit: 3x" in text

    def test_output_to_stdout_default(self, capsys):
        """Test that output goes to stdout by default."""
        batch_result = BatchEvaluationResult(
            results=[
                EvaluationResult(
                    output="test",
                    scores=[Score(name="semantic", value=0.8)],
                    overall_score=0.8,
                    passed=True,
                    processing_time=0.5,
                    interactions=[],
                )
            ],
            total_items=1,
            successful_items=1,
            failed_items=0,
            processing_time=0.5,
            total_tokens=100,
            errors=[],
        )

        batch_result.pretty_print()
        captured = capsys.readouterr()

        assert "Batch Evaluation Results" in captured.out
        assert "Success: 1/1" in captured.out


class TestPrettyPrintEdgeCases:
    """Test edge cases for pretty_print methods."""

    def test_evaluation_result_no_confidence(self):
        """Test score display when confidence is None."""
        result = EvaluationResult(
            output="test",
            scores=[Score(name="semantic", value=0.85, confidence=None)],
            overall_score=0.85,
            passed=True,
            processing_time=0.5,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output)
        text = output.getvalue()

        # Should show score without confidence
        assert "semantic" in text
        assert "0.85" in text
        # Should NOT show confidence info
        assert "confidence:" not in text

    def test_batch_result_empty(self):
        """Test batch result with no items."""
        batch_result = BatchEvaluationResult(
            results=[],
            total_items=0,
            successful_items=0,
            failed_items=0,
            processing_time=0.0,
            total_tokens=0,
            errors=[],
        )

        output = StringIO()
        batch_result.pretty_print(file=output)
        text = output.getvalue()

        # Should handle division by zero gracefully
        assert "Batch Evaluation Results" in text
        # No crash on 0/0 percentage

    def test_comparison_no_aspect_scores(self):
        """Test comparison output when aspect_scores is empty."""
        result = ComparisonResult(
            output_a="A",
            output_b="B",
            winner="output_a",
            confidence=0.85,
            reasoning="A is better",
            aspect_scores={},  # Empty
            processing_time=1.0,
            interactions=[],
        )

        output = StringIO()
        result.pretty_print(file=output, verbose=True)
        text = output.getvalue()

        # Should not crash, just won't show aspect scores section
        assert "Comparison Results" in text
        # Aspect Scores section shouldn't appear if empty
        if "Aspect Scores:" in text:
            # If it appears, it should be empty
            assert text.count("•") == 0
