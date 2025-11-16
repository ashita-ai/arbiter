"""Manual Batch Evaluation Example - Processing Multiple Evaluations Efficiently

This example demonstrates how to batch process multiple evaluations using
asyncio.gather for parallel execution. This is useful before Phase 4's
batch_evaluate() API is available.

**Key Features Shown:**
- Parallel evaluation with asyncio.gather
- Progress tracking
- Error handling for batch operations
- Cost aggregation across batch
- Performance comparison (sequential vs parallel)

**Why This Matters:**
- **Performance**: Parallel execution is 3-5x faster than sequential
- **Cost Efficiency**: Process more evaluations in less time
- **Production Ready**: Handle failures gracefully in batch operations

Run with:
    python examples/batch_manual.py
"""

from dotenv import load_dotenv

import asyncio
import os
import time
from typing import List

from arbiter import evaluate
from arbiter.core import LLMManager
from arbiter.core.models import EvaluationResult


async def evaluate_single(
    output: str,
    reference: str,
    model: str = "gpt-4o-mini",
) -> EvaluationResult:
    """Evaluate a single output-reference pair."""
    return await evaluate(
        output=output,
        reference=reference,
        evaluators=["semantic"],
        model=model,
    )


async def batch_evaluate_sequential(
    outputs: List[str],
    references: List[str],
    model: str = "gpt-4o-mini",
) -> List[EvaluationResult]:
    """Evaluate multiple outputs sequentially (slow but simple)."""
    results = []
    for output, reference in zip(outputs, references):
        result = await evaluate_single(output, reference, model)
        results.append(result)
    return results


async def batch_evaluate_parallel(
    outputs: List[str],
    references: List[str],
    model: str = "gpt-4o-mini",
    max_concurrent: int = 10,
) -> List[EvaluationResult]:
    """Evaluate multiple outputs in parallel using asyncio.gather.

    Args:
        outputs: List of outputs to evaluate
        references: List of reference texts (must match outputs length)
        model: Model to use for evaluation
        max_concurrent: Maximum concurrent evaluations (rate limit protection)

    Returns:
        List of EvaluationResult objects in same order as inputs
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_limit(output: str, reference: str) -> EvaluationResult:
        async with semaphore:
            return await evaluate_single(output, reference, model)

    # Create all tasks
    tasks = [
        evaluate_with_limit(output, reference)
        for output, reference in zip(outputs, references)
    ]

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"⚠️  Evaluation {i+1} failed: {result}")
            # Create a failed result placeholder
            # In production, you might want to handle this differently
            processed_results.append(None)
        else:
            processed_results.append(result)

    return processed_results


async def batch_evaluate_with_progress(
    outputs: List[str],
    references: List[str],
    model: str = "gpt-4o-mini",
    callback=None,
) -> List[EvaluationResult]:
    """Evaluate multiple outputs with progress tracking.

    Args:
        outputs: List of outputs to evaluate
        references: List of reference texts
        model: Model to use
        callback: Optional callback function(completed, total, result)

    Returns:
        List of EvaluationResult objects
    """
    total = len(outputs)
    results = []

    # Process in batches for progress tracking
    batch_size = 5
    for i in range(0, total, batch_size):
        batch_outputs = outputs[i:i+batch_size]
        batch_references = references[i:i+batch_size]

        # Process batch in parallel
        batch_results = await asyncio.gather(
            *[evaluate_single(o, r, model) for o, r in zip(batch_outputs, batch_references)],
            return_exceptions=True
        )

        # Process results
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"⚠️  Evaluation {i+j+1} failed: {result}")
                results.append(None)
            else:
                results.append(result)
                if callback:
                    callback(i+j+1, total, result)

    return results


async def print_batch_summary(results: List[EvaluationResult], method: str):
    """Print summary statistics for batch evaluation."""
    valid_results = [r for r in results if r is not None]
    failed_count = len(results) - len(valid_results)

    if not valid_results:
        print(f"\n❌ {method}: All evaluations failed")
        return

    total_tokens = sum(r.total_tokens for r in valid_results)
    total_time = sum(r.processing_time for r in valid_results)

    # Calculate costs asynchronously
    costs = await asyncio.gather(*[r.total_llm_cost() for r in valid_results])
    total_cost = sum(costs)

    avg_score = sum(r.overall_score for r in valid_results) / len(valid_results)

    print(f"\n📊 {method} Summary:")
    print(f"   Evaluations: {len(valid_results)}/{len(results)} successful")
    print(f"   Failed: {failed_count}")
    print(f"   Total Tokens: {total_tokens:,}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Score: {avg_score:.3f}")
    print(f"   Total Cost: ${total_cost:.6f}")
    print(f"   Cost per Evaluation: ${total_cost/len(valid_results):.6f}")


async def main():
    """Run batch evaluation examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔄 Arbiter - Manual Batch Evaluation Example")
    print("=" * 70)
    print("\nThis example shows how to batch process multiple evaluations")
    print("using asyncio.gather for parallel execution.")
    print("=" * 70)

    # Prepare test data
    test_pairs = [
        ("Paris is the capital of France", "The capital of France is Paris"),
        ("Python is a programming language", "Python is used for software development"),
        ("The weather is nice today", "It's a beautiful day outside"),
        ("Machine learning is powerful", "ML enables intelligent systems"),
        ("Open source software is great", "OSS benefits the community"),
    ]

    outputs = [pair[0] for pair in test_pairs]
    references = [pair[1] for pair in test_pairs]

    print(f"\n📝 Test Data: {len(test_pairs)} evaluation pairs")
    for i, (output, reference) in enumerate(test_pairs, 1):
        print(f"   {i}. Output: {output[:40]}...")
        print(f"      Reference: {reference[:40]}...")

    # Example 1: Sequential Evaluation (Baseline)
    print("\n\n📊 Example 1: Sequential Evaluation (Baseline)")
    print("-" * 70)
    print("Processing evaluations one at a time...")

    start_time = time.time()
    sequential_results = await batch_evaluate_sequential(outputs, references)
    sequential_time = time.time() - start_time

    await print_batch_summary(sequential_results, "Sequential")
    print(f"   ⏱️  Total Wall Time: {sequential_time:.2f}s")

    # Example 2: Parallel Evaluation (Fast)
    print("\n\n📊 Example 2: Parallel Evaluation (Fast)")
    print("-" * 70)
    print("Processing evaluations in parallel with asyncio.gather...")

    start_time = time.time()
    parallel_results = await batch_evaluate_parallel(outputs, references, max_concurrent=5)
    parallel_time = time.time() - start_time

    await print_batch_summary(parallel_results, "Parallel")
    print(f"   ⏱️  Total Wall Time: {parallel_time:.2f}s")

    # Performance comparison
    if sequential_time > 0:
        speedup = sequential_time / parallel_time
        print(f"\n⚡ Performance Comparison:")
        print(f"   Sequential: {sequential_time:.2f}s")
        print(f"   Parallel:   {parallel_time:.2f}s")
        print(f"   Speedup:    {speedup:.2f}x faster")

    # Example 3: Progress Tracking
    print("\n\n📊 Example 3: Batch Evaluation with Progress Tracking")
    print("-" * 70)

    def progress_callback(completed: int, total: int, result: EvaluationResult):
        """Callback function for progress updates."""
        percentage = (completed / total) * 100
        print(f"   Progress: {completed}/{total} ({percentage:.1f}%) - Score: {result.overall_score:.3f}")

    start_time = time.time()
    progress_results = await batch_evaluate_with_progress(
        outputs,
        references,
        callback=progress_callback
    )
    progress_time = time.time() - start_time

    await print_batch_summary(progress_results, "With Progress Tracking")
    print(f"   ⏱️  Total Wall Time: {progress_time:.2f}s")

    # Example 4: Error Handling in Batch
    print("\n\n📊 Example 4: Error Handling in Batch Operations")
    print("-" * 70)

    # Mix valid and potentially problematic inputs
    mixed_outputs = outputs + ["", "Very long output " * 100]  # Empty and very long
    mixed_references = references + ["Valid reference", "Another reference"]

    print(f"Processing {len(mixed_outputs)} evaluations (including edge cases)...")

    error_results = await batch_evaluate_parallel(
        mixed_outputs,
        mixed_references,
        max_concurrent=3
    )

    successful = [r for r in error_results if r is not None]
    failed = [i for i, r in enumerate(error_results) if r is None]

    print(f"\n✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    if failed:
        print(f"   Failed indices: {failed}")

    # Example 5: Cost Analysis
    print("\n\n📊 Example 5: Batch Cost Analysis")
    print("-" * 70)

    if parallel_results:
        valid_results = [r for r in parallel_results if r is not None]

        total_tokens = sum(r.total_tokens for r in valid_results)

        # Calculate costs asynchronously
        costs = await asyncio.gather(*[r.total_llm_cost() for r in valid_results])
        total_cost = sum(costs)

        print(f"Batch Evaluation Cost Breakdown:")
        print(f"   Total Evaluations: {len(valid_results)}")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Total Cost: ${total_cost:.6f}")
        print(f"   Average Cost per Evaluation: ${total_cost/len(valid_results):.6f}")
        print(f"   Average Tokens per Evaluation: {total_tokens//len(valid_results):,}")

        # Cost projection
        print(f"\n💡 Cost Projections:")
        for count in [100, 1000, 10000]:
            projected_cost = (total_cost / len(valid_results)) * count
            print(f"   {count:,} evaluations: ${projected_cost:.2f}")

    # Summary
    print("\n\n" + "=" * 70)
    print("✅ All Examples Complete!")
    print("=" * 70)

    print("\n🎯 Key Takeaways:")
    print("   • Parallel execution is 3-5x faster than sequential")
    print("   • asyncio.gather enables efficient batch processing")
    print("   • Progress tracking helps monitor long-running batches")
    print("   • Error handling ensures robustness in production")
    print("   • Cost analysis helps with budget planning")

    print("\n💡 Production Tips:")
    print("   • Use semaphores to limit concurrent requests (rate limiting)")
    print("   • Implement retry logic for transient failures")
    print("   • Track progress for user feedback")
    print("   • Monitor costs and token usage")
    print("   • Handle errors gracefully (don't fail entire batch)")

    print("\n📚 Learn More:")
    print("   • Phase 4 will add batch_evaluate() API (coming soon)")
    print("   • See: examples/multiple_evaluators.py for multi-evaluator usage")
    print("   • See: examples/interaction_tracking_example.py for cost tracking")


if __name__ == "__main__":
    asyncio.run(main())

