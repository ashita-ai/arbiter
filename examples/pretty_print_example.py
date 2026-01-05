"""Example demonstrating pretty_print() methods for terminal-friendly output.

This example shows how to use the pretty_print() method on evaluation results
to get nicely formatted, human-readable output in the terminal.
"""

import asyncio

from arbiter_ai import batch_evaluate, compare, evaluate


async def demo_evaluation_result_pretty_print():
    """Demonstrate EvaluationResult.pretty_print()."""
    print("\n" + "=" * 80)
    print("DEMO 1: EvaluationResult.pretty_print()")
    print("=" * 80)

    # Example 1: Basic evaluation
    print("\n1. Basic Evaluation (Passing):")
    result = await evaluate(
        output="Paris is the capital of France and is known for the Eiffel Tower.",
        reference="Paris is the capital of France.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )
    result.pretty_print()

    # Example 2: Evaluation with verbose output
    print("\n\n2. Same Evaluation (Verbose Mode):")
    result.pretty_print(verbose=True)

    # Example 3: Failed evaluation
    print("\n\n3. Failed Evaluation:")
    result_failed = await evaluate(
        output="Tokyo is the capital of China.",
        reference="Beijing is the capital of China.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )
    result_failed.pretty_print()

    # Example 4: Multiple evaluators
    print("\n\n4. Multiple Evaluators:")
    result_multi = await evaluate(
        output="The Earth orbits around the Sun in approximately 365 days.",
        reference="Earth takes 365.25 days to orbit the Sun.",
        evaluators=["semantic", "factuality"],
        model="gpt-4o-mini",
    )
    result_multi.pretty_print()

    # Example 5: Redirect output to file
    print("\n\n5. Saving to File:")
    with open("/tmp/evaluation_results.txt", "w") as f:
        result.pretty_print(file=f)
    print("✓ Results saved to /tmp/evaluation_results.txt")

    # Show the file contents
    print("\nFile contents:")
    with open("/tmp/evaluation_results.txt", "r") as f:
        print(f.read())


async def demo_comparison_result_pretty_print():
    """Demonstrate ComparisonResult.pretty_print()."""
    print("\n" + "=" * 80)
    print("DEMO 2: ComparisonResult.pretty_print()")
    print("=" * 80)

    # Example 1: Basic comparison
    print("\n1. Basic Comparison:")
    comparison = await compare(
        output_a="Paris is the capital of France, known for its art and culture.",
        output_b="Paris is France's capital city.",
        reference="What is the capital of France?",
        model="gpt-4o-mini",
    )
    comparison.pretty_print()

    # Example 2: Verbose comparison with aspect scores
    print("\n\n2. Verbose Comparison:")
    comparison.pretty_print(verbose=True)

    # Example 3: Tie result
    print("\n\n3. Tie Result:")
    comparison_tie = await compare(
        output_a="The quick brown fox jumps over the lazy dog.",
        output_b="A quick brown fox leaps over a lazy dog.",
        model="gpt-4o-mini",
    )
    comparison_tie.pretty_print()


async def demo_batch_evaluation_pretty_print():
    """Demonstrate BatchEvaluationResult.pretty_print()."""
    print("\n" + "=" * 80)
    print("DEMO 3: BatchEvaluationResult.pretty_print()")
    print("=" * 80)

    # Example 1: Successful batch
    print("\n1. Batch Evaluation (All Successful):")
    items = [
        {
            "output": f"Paris is the capital of France. Population: {i}M.",
            "reference": "Paris is the capital of France.",
        }
        for i in range(1, 11)
    ]
    batch_result = await batch_evaluate(
        items=items,
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )
    batch_result.pretty_print()

    # Example 2: Verbose batch output
    print("\n\n2. Batch Evaluation (Verbose Mode - showing first 5 items):")
    # Create a smaller batch for verbose demo
    items_small = items[:5]
    batch_result_small = await batch_evaluate(
        items=items_small,
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )
    batch_result_small.pretty_print(verbose=True)

    # Example 3: Batch with mixed pass/fail
    print("\n\n3. Batch Evaluation (Mixed Results):")
    items_mixed = [
        {
            "output": "Paris is the capital of France.",
            "reference": "Paris is the capital of France.",
        },
        {
            "output": "Wrong answer about capitals.",
            "reference": "Paris is the capital of France.",
        },
        {
            "output": "Tokyo is the capital of Japan.",
            "reference": "Tokyo is the capital of Japan.",
        },
        {
            "output": "Berlin is the capital of Spain.",  # Wrong
            "reference": "Madrid is the capital of Spain.",
        },
        {
            "output": "London is the capital of the UK.",
            "reference": "London is the capital of the United Kingdom.",
        },
    ]
    batch_result_mixed = await batch_evaluate(
        items=items_mixed,
        evaluators=["semantic"],
        model="gpt-4o-mini",
        threshold=0.8,  # Higher threshold to see some failures
    )
    batch_result_mixed.pretty_print()


async def demo_cost_information():
    """Demonstrate cost information in pretty_print."""
    print("\n" + "=" * 80)
    print("DEMO 4: Cost Information")
    print("=" * 80)

    print("\n1. Evaluation Cost:")
    result = await evaluate(
        output="Paris is the capital of France.",
        reference="Paris is France's capital.",
        evaluators=["semantic", "factuality"],
        model="gpt-4o-mini",
    )

    # Pretty print shows token usage
    result.pretty_print(verbose=True)

    # Get actual cost
    cost = await result.total_llm_cost()
    print(f"\nTotal Cost: ${cost:.6f}")

    # Get cost breakdown
    breakdown = await result.cost_breakdown()
    print("\nCost Breakdown:")
    print(f"  By Evaluator: {breakdown['by_evaluator']}")
    print(f"  By Model: {breakdown['by_model']}")
    print(f"  Token Breakdown: {breakdown['token_breakdown']}")


async def main():
    """Run all pretty_print demonstrations."""
    print("\n" + "=" * 80)
    print("ARBITER PRETTY PRINT DEMONSTRATION")
    print("Terminal-Friendly Output Examples")
    print("=" * 80)

    # Demo 1: EvaluationResult
    await demo_evaluation_result_pretty_print()

    # Demo 2: ComparisonResult
    await demo_comparison_result_pretty_print()

    # Demo 3: BatchEvaluationResult
    await demo_batch_evaluation_pretty_print()

    # Demo 4: Cost Information
    await demo_cost_information()

    print("\n" + "=" * 80)
    print("ALL DEMOS COMPLETE")
    print("=" * 80)
    print("\n✨ The pretty_print() method makes debugging and interactive use")
    print("   much more pleasant with clean, formatted terminal output!")
    print("\nKey Features:")
    print("  • Unicode symbols (✓ ✗ •) for visual clarity")
    print("  • Concise default output, detailed verbose mode")
    print("  • File redirection support for logging")
    print("  • Automatic statistics and cost tracking")
    print("\nUsage:")
    print("  result.pretty_print()              # Basic output to stdout")
    print("  result.pretty_print(verbose=True)  # Detailed output")
    print("  result.pretty_print(file=f)        # Write to file")


if __name__ == "__main__":
    asyncio.run(main())
