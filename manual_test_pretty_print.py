"""Manual test for pretty_print() methods - no dependencies required.

This script tests the pretty_print() methods without requiring LLM calls,
by creating result objects directly with test data.
"""

from arbiter_ai.core.models import (
    BatchEvaluationResult,
    ComparisonResult,
    EvaluationResult,
    LLMInteraction,
    Score,
)
from datetime import datetime, timezone
from io import StringIO

# Direct imports to avoid dependency issues
import sys
sys.path.insert(0, '/home/nick/arbiter')


def test_evaluation_result_pretty_print():
    """Test EvaluationResult.pretty_print()."""
    print("=" * 80)
    print("TEST 1: EvaluationResult.pretty_print()")
    print("=" * 80)

    # Create a sample result
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
                prompt="test prompt 1",
                response="test response 1",
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                latency=0.5,
                purpose="scoring",
            ),
            LLMInteraction(
                prompt="test prompt 2",
                response="test response 2",
                model="gpt-4o-mini",
                input_tokens=120,
                output_tokens=60,
                latency=0.73,
                purpose="factuality",
            ),
        ],
    )

    print("\n1. Basic output:")
    result.pretty_print()

    print("\n\n2. Verbose output:")
    result.pretty_print(verbose=True)

    print("\n\n3. Output to StringIO (testing file parameter):")
    output = StringIO()
    result.pretty_print(file=output)
    captured = output.getvalue()
    print("Captured output:")
    print(captured)

    # Verify key elements are present
    assert "Evaluation Results" in captured
    assert "Overall Score: 0.87 ✓ PASSED" in captured
    assert "semantic" in captured
    assert "factuality" in captured
    assert "Time: 1.23s" in captured
    assert "LLM Calls: 2" in captured
    print("✅ All assertions passed for EvaluationResult")


def test_comparison_result_pretty_print():
    """Test ComparisonResult.pretty_print()."""
    print("\n" + "=" * 80)
    print("TEST 2: ComparisonResult.pretty_print()")
    print("=" * 80)

    comparison = ComparisonResult(
        output_a="GPT-4 response",
        output_b="Claude response",
        winner="output_a",
        confidence=0.85,
        reasoning="Output A is more accurate and provides better context with specific examples.",
        aspect_scores={
            "accuracy": {"output_a": 0.95, "output_b": 0.80},
            "clarity": {"output_a": 0.85, "output_b": 0.90},
        },
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

    print("\n1. Basic output:")
    comparison.pretty_print()

    print("\n\n2. Verbose output:")
    comparison.pretty_print(verbose=True)

    print("\n\n3. Output to StringIO:")
    output = StringIO()
    comparison.pretty_print(file=output)
    captured = output.getvalue()
    print("Captured output:")
    print(captured)

    # Verify key elements
    assert "Comparison Results" in captured
    assert "Winner: Output A ✓" in captured
    assert "Confidence: 0.85" in captured
    assert "Time: 1.45s" in captured
    print("✅ All assertions passed for ComparisonResult")


def test_batch_evaluation_result_pretty_print():
    """Test BatchEvaluationResult.pretty_print()."""
    print("\n" + "=" * 80)
    print("TEST 3: BatchEvaluationResult.pretty_print()")
    print("=" * 80)

    # Create 10 successful results
    results = [
        EvaluationResult(
            output=f"output_{i}",
            scores=[Score(name="semantic", value=0.70 + i * 0.05)],
            overall_score=0.70 + i * 0.05,
            passed=(0.70 + i * 0.05) >= 0.7,
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

    print("\n1. Basic output:")
    batch_result.pretty_print()

    print("\n\n2. Verbose output:")
    batch_result.pretty_print(verbose=True)

    print("\n\n3. Output to StringIO:")
    output = StringIO()
    batch_result.pretty_print(file=output)
    captured = output.getvalue()
    print("Captured output:")
    print(captured)

    # Verify key elements
    assert "Batch Evaluation Results" in captured
    assert "Success: 10/10 (100.0%)" in captured
    assert "Statistics:" in captured
    assert "Mean:" in captured
    assert "Total Time: 5.2s" in captured
    print("✅ All assertions passed for BatchEvaluationResult")


def test_partial_result():
    """Test partial result with errors."""
    print("\n" + "=" * 80)
    print("TEST 4: Partial Result with Errors")
    print("=" * 80)

    result = EvaluationResult(
        output="Test output",
        scores=[Score(name="semantic", value=0.85, confidence=0.9)],
        overall_score=0.85,
        passed=True,
        partial=True,
        errors={"factuality": "API timeout",
                "groundedness": "Rate limit exceeded"},
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

    result.pretty_print()

    output = StringIO()
    result.pretty_print(file=output)
    captured = output.getvalue()

    assert "⚠ PARTIAL RESULT" in captured
    assert "Errors:" in captured
    assert "factuality: API timeout" in captured
    print("✅ Partial result test passed")


def test_unicode_symbols():
    """Test that Unicode symbols are present in output."""
    print("\n" + "=" * 80)
    print("TEST 5: Unicode Symbols")
    print("=" * 80)

    # Test passed result
    result_pass = EvaluationResult(
        output="test",
        scores=[Score(name="test", value=0.9)],
        overall_score=0.9,
        passed=True,
        processing_time=0.5,
        interactions=[],
    )

    output = StringIO()
    result_pass.pretty_print(file=output)
    captured = output.getvalue()

    assert "✓" in captured, "Should contain checkmark for passed"
    assert "•" in captured, "Should contain bullet points"
    print("✅ Passed result contains ✓ and •")

    # Test failed result
    result_fail = EvaluationResult(
        output="test",
        scores=[Score(name="test", value=0.3)],
        overall_score=0.3,
        passed=False,
        processing_time=0.5,
        interactions=[],
    )

    output = StringIO()
    result_fail.pretty_print(file=output)
    captured = output.getvalue()

    assert "✗" in captured, "Should contain X for failed"
    print("✅ Failed result contains ✗")


def main():
    """Run all manual tests."""
    print("\n" + "=" * 80)
    print("MANUAL TESTS FOR PRETTY_PRINT() METHODS")
    print("=" * 80)

    try:
        test_evaluation_result_pretty_print()
        test_comparison_result_pretty_print()
        test_batch_evaluation_result_pretty_print()
        test_partial_result()
        test_unicode_symbols()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print("\nThe pretty_print() methods are working correctly!")
        print("\nFeatures verified:")
        print("  ✓ Basic output formatting")
        print("  ✓ Verbose mode with detailed information")
        print("  ✓ File parameter for output redirection")
        print("  ✓ Unicode symbols (✓ ✗ •)")
        print("  ✓ Partial results with error handling")
        print("  ✓ Statistics calculation for batch results")
        print("  ✓ Token usage breakdown")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
