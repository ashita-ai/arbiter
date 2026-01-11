"""Cost Estimation and Dry Run - Budget Before You Evaluate

This example demonstrates Arbiter's cost estimation and dry-run features,
allowing you to preview evaluations and estimate costs before making API calls.

Key Features:
- Estimate costs without API calls
- Preview exact prompts that would be sent
- Batch cost estimation for large datasets
- Dry-run mode for debugging and validation

Requirements:
    No API key required for estimation/dry-run (no API calls made)

Run with:
    python examples/cost_estimation_example.py
"""

import asyncio

from arbiter_ai import (
    evaluate,
    estimate_evaluation_cost,
    estimate_batch_cost,
    get_prompt_preview,
)


async def main():
    """Demonstrate cost estimation and dry-run features."""

    print("Cost Estimation and Dry Run Example")
    print("=" * 50)

    # Example 1: Single evaluation cost estimate
    print("\n1. Single Evaluation Cost Estimate")
    print("-" * 50)

    estimate = await estimate_evaluation_cost(
        output="Paris is the capital of France and is known for the Eiffel Tower.",
        reference="The capital of France is Paris, famous for its iconic Eiffel Tower.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"Model: {estimate.model}")
    print(f"Estimated total cost: ${estimate.total_cost:.6f}")
    print(f"Estimated tokens:")
    print(f"  Input:  {estimate.input_tokens:,}")
    print(f"  Output: {estimate.output_tokens:,}")
    print(f"  Total:  {estimate.total_tokens:,}")

    print("\nCost by evaluator:")
    for name, data in estimate.by_evaluator.items():
        print(f"  {name}:")
        print(f"    Input tokens:  {data['input_tokens']:,}")
        print(f"    Output tokens: {data['output_tokens']:,}")
        print(f"    Cost: ${data['cost']:.6f}")

    # Example 2: Multiple evaluators cost estimate
    print("\n\n2. Multiple Evaluators Cost Estimate")
    print("-" * 50)

    multi_estimate = await estimate_evaluation_cost(
        output="The quick brown fox jumps over the lazy dog.",
        reference="A fast brown fox leaps above a sleepy canine.",
        criteria="Evaluate semantic similarity and writing quality",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Evaluators: {list(multi_estimate.by_evaluator.keys())}")
    print(f"Total estimated cost: ${multi_estimate.total_cost:.6f}")
    print(f"Total estimated tokens: {multi_estimate.total_tokens:,}")

    # Example 3: Batch cost estimation
    print("\n\n3. Batch Cost Estimation")
    print("-" * 50)

    items = [
        {"output": "Paris is the capital of France", "reference": "France's capital is Paris"},
        {"output": "Tokyo is the capital of Japan", "reference": "Japan's capital is Tokyo"},
        {"output": "Berlin is the capital of Germany", "reference": "Germany's capital is Berlin"},
        {"output": "Rome is the capital of Italy", "reference": "Italy's capital is Rome"},
        {"output": "Madrid is the capital of Spain", "reference": "Spain's capital is Madrid"},
    ]

    batch_estimate = await estimate_batch_cost(
        items=items,
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"Batch size: {batch_estimate.item_count} items")
    print(f"Total estimated cost: ${batch_estimate.total_cost:.6f}")
    print(f"Per-item average cost: ${batch_estimate.per_item_cost:.6f}")
    print(f"Total estimated tokens: {batch_estimate.total_tokens:,}")
    print(f"Per-item average tokens: {batch_estimate.per_item_tokens:,}")

    # Scale up estimation
    print("\nScaling projection:")
    for scale in [100, 1000, 10000]:
        projected_cost = batch_estimate.per_item_cost * scale
        print(f"  {scale:,} items: ${projected_cost:.4f}")

    # Example 4: Dry-run mode with prompt preview
    print("\n\n4. Dry-Run Mode (Preview Prompts)")
    print("-" * 50)

    preview = await get_prompt_preview(
        output="Machine learning is a subset of artificial intelligence.",
        reference="ML is part of AI that enables computers to learn from data.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"Dry run: {preview.dry_run}")
    print(f"Model: {preview.model}")
    print(f"Evaluators: {preview.evaluators}")
    print(f"Estimated cost: ${preview.estimated_cost:.6f}")
    print(f"Estimated tokens: {preview.estimated_tokens:,}")

    print("\nValidation status:")
    for check, passed in preview.validation.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    print("\nPrompt preview (semantic evaluator):")
    print("  System prompt (first 200 chars):")
    system = preview.prompts["semantic"]["system"]
    print(f"    {system[:200]}...")
    print("  User prompt (first 300 chars):")
    user = preview.prompts["semantic"]["user"]
    print(f"    {user[:300]}...")

    # Example 5: Using dry_run parameter with evaluate()
    print("\n\n5. evaluate() with dry_run=True")
    print("-" * 50)

    dry_result = await evaluate(
        output="Neural networks are inspired by biological neurons.",
        reference="Neural nets mimic brain neuron structure.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
        dry_run=True,  # No API calls made
    )

    print(f"Result type: {type(dry_result).__name__}")
    print(f"Estimated cost: ${dry_result.estimated_cost:.6f}")
    print(f"Estimated tokens: {dry_result.estimated_tokens:,}")
    print("No API calls were made!")

    # Example 6: Validation with dry-run
    print("\n\n6. Input Validation with Dry-Run")
    print("-" * 50)

    # Test with empty output
    invalid_preview = await get_prompt_preview(
        output="",  # Invalid: empty output
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print("Testing with empty output:")
    print(f"  output_valid: {invalid_preview.validation['output_valid']}")
    print(f"  evaluators_valid: {invalid_preview.validation['evaluators_valid']}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("\nKey Features Demonstrated:")
    print("  - Single evaluation cost estimation")
    print("  - Multiple evaluators cost breakdown")
    print("  - Batch cost estimation with scaling projections")
    print("  - Prompt preview for debugging")
    print("  - Input validation without API calls")

    print("\nUse Cases:")
    print("  - Budget large evaluation jobs before running")
    print("  - Debug prompts without spending money")
    print("  - Validate inputs before submitting")
    print("  - Compare costs across different models")

    print("\nNext Steps:")
    print("  - See basic_evaluation.py for actual evaluations")
    print("  - See batch_evaluation_example.py for batch processing")
    print("  - See cost_comparison.py for model cost comparisons")


if __name__ == "__main__":
    asyncio.run(main())
