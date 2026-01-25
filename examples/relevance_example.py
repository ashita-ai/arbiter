"""Relevance Evaluation - Query-Output Alignment Assessment

This example demonstrates how to evaluate whether LLM outputs are relevant
to given queries, identifying addressed points, missing information, and
off-topic content.

Key Features:
- Query-output alignment validation
- Missing information detection
- Off-topic content identification
- Addressed vs missing point categorization
- Direct evaluator usage for fine-grained control

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/relevance_example.py
"""

import asyncio
import os

from dotenv import load_dotenv

from arbiter_ai import RelevanceEvaluator, evaluate
from arbiter_ai.core import LLMManager


async def main():
    """Run relevance evaluation examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("Arbiter - Relevance Evaluation Example")
    print("=" * 60)

    # Example 1: Fully relevant response
    print("\nExample 1: Fully Relevant Response")
    print("-" * 60)

    query_1 = "What is Python and who created it?"

    response_1 = """
    Python is a high-level, interpreted programming language known for its clear
    syntax and readability. It was created by Guido van Rossum and first released
    in 1991. Python supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """

    result1 = await evaluate(
        output=response_1,
        reference=query_1,
        evaluators=["relevance"],
        model="gpt-4o-mini",
    )

    print(f"Query: {query_1}")
    print(f"Response: {response_1[:100]}...")
    print("\nResults:")
    print(f"  Relevance Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result1.passed else 'No'}")

    for score in result1.scores:
        print(f"\n  {score.name}:")
        print(f"    Value: {score.value:.3f}")
        if score.confidence:
            print(f"    Confidence: {score.confidence:.3f}")
        if score.metadata.get("addressed_points"):
            print("    Addressed Points:")
            for point in score.metadata["addressed_points"]:
                print(f"      - {point}")

    # Example 2: Partially relevant response with missing information
    print("\n\nExample 2: Partial Response (Missing Information)")
    print("-" * 60)

    query_2 = """
    Explain the three main types of machine learning: supervised learning,
    unsupervised learning, and reinforcement learning. Provide an example for each.
    """

    response_2 = """
    Machine learning has three main types. Supervised learning uses labeled data
    to train models, like email spam detection. Unsupervised learning finds patterns
    in unlabeled data, such as customer segmentation.
    """

    result2 = await evaluate(
        output=response_2,
        reference=query_2,
        evaluators=["relevance"],
        model="gpt-4o-mini",
    )

    print(f"Query: {query_2[:80]}...")
    print(f"Response: {response_2[:100]}...")
    print("\nResults:")
    print(f"  Relevance Score: {result2.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result2.passed else 'No'}")

    for score in result2.scores:
        if score.metadata.get("missing_points"):
            print("\n  Missing Points:")
            for point in score.metadata["missing_points"]:
                print(f"    - {point}")
        if score.metadata.get("addressed_points"):
            print("\n  Addressed Points:")
            for point in score.metadata["addressed_points"]:
                print(f"    - {point}")

    # Example 3: Response with off-topic content
    print("\n\nExample 3: Response with Off-Topic Content")
    print("-" * 60)

    query_3 = "What are the health benefits of regular exercise?"

    response_3 = """
    Regular exercise offers numerous health benefits including improved
    cardiovascular health, stronger muscles, and better mental health. It can
    reduce the risk of chronic diseases like diabetes and heart disease.

    Speaking of health, the healthcare industry has seen significant technological
    advances in recent years. AI is being used for medical diagnosis and drug
    discovery. Telemedicine has also grown significantly since 2020.

    Exercise should be done at least 150 minutes per week according to WHO guidelines.
    """

    result3 = await evaluate(
        output=response_3,
        reference=query_3,
        evaluators=["relevance"],
        model="gpt-4o-mini",
    )

    print(f"Query: {query_3}")
    print(f"Response: {response_3[:100]}...")
    print("\nResults:")
    print(f"  Relevance Score: {result3.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result3.passed else 'No'}")

    for score in result3.scores:
        if score.metadata.get("irrelevant_content"):
            print("\n  Off-Topic Content Detected:")
            for content in score.metadata["irrelevant_content"]:
                print(f"    - {content}")
        if score.metadata.get("addressed_count") is not None:
            addressed = score.metadata.get("addressed_count", 0)
            missing = score.metadata.get("missing_count", 0)
            irrelevant = score.metadata.get("irrelevant_count", 0)
            print(f"\n  Content Breakdown:")
            print(f"    Addressed Points: {addressed}")
            print(f"    Missing Points: {missing}")
            print(f"    Irrelevant Content: {irrelevant}")

    # Example 4: Completely off-topic response
    print("\n\nExample 4: Completely Off-Topic Response")
    print("-" * 60)

    query_4 = "How do I reset my password on your platform?"

    response_4 = """
    Our company was founded in 2015 and has grown to serve over 10,000 customers
    worldwide. We offer enterprise solutions and have won several industry awards.
    Our headquarters is located in San Francisco with offices in London and Tokyo.
    """

    result4 = await evaluate(
        output=response_4,
        reference=query_4,
        evaluators=["relevance"],
        model="gpt-4o-mini",
    )

    print(f"Query: {query_4}")
    print(f"Response: {response_4[:80]}...")
    print("\nResults:")
    print(f"  Relevance Score: {result4.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result4.passed else 'No'}")

    for score in result4.scores:
        print(f"\n  Explanation: {score.explanation[:150]}...")

    # Example 5: Direct evaluator usage with criteria
    print("\n\nExample 5: Direct Evaluator Usage with Custom Criteria")
    print("-" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = RelevanceEvaluator(client)

    technical_query = """
    What is the time complexity of quicksort, and when might you choose
    mergesort instead?
    """

    technical_response = """
    Quicksort has an average time complexity of O(n log n) and worst case O(n^2).
    Mergesort consistently runs in O(n log n) time. You might choose mergesort
    when you need guaranteed performance or stable sorting.
    """

    score = await evaluator.evaluate(
        output=technical_response,
        reference=technical_query,
        criteria="technical accuracy and completeness of the comparison",
    )

    print(f"Query: {technical_query[:60]}...")
    print(f"Response: {technical_response[:80]}...")
    print(f"\nRelevance Score: {score.value:.3f}")
    print(f"Confidence: {score.confidence:.3f}")

    if score.metadata.get("addressed_points"):
        print("\nAddressed Points:")
        for point in score.metadata["addressed_points"]:
            print(f"  - {point}")

    # Access interactions directly
    print("\nEvaluator Interactions:")
    interactions = evaluator.get_interactions()
    total_tokens = sum(i.tokens_used for i in interactions)
    print(f"  Total Calls: {len(interactions)}")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    # Session summary
    print("\n" + "=" * 60)
    print("Examples Complete!")

    # Calculate total session cost
    cost1 = await result1.total_llm_cost()
    cost2 = await result2.total_llm_cost()
    cost3 = await result3.total_llm_cost()
    cost4 = await result4.total_llm_cost()
    total_cost = cost1 + cost2 + cost3 + cost4
    total_tokens_all = (
        result1.total_tokens
        + result2.total_tokens
        + result3.total_tokens
        + result4.total_tokens
    )

    print("\nTotal Session Cost:")
    print("  Total Evaluations: 5")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens_all:,}")

    print("\nKey Features Demonstrated:")
    print("  - Query-output alignment validation")
    print("  - Missing information detection")
    print("  - Off-topic content identification")
    print("  - Point categorization (addressed/missing/irrelevant)")
    print("  - Custom criteria evaluation")
    print("  - Direct evaluator usage for fine-grained control")

    print("\nRelated Examples:")
    print("  - See groundedness_example.py for source attribution")
    print("  - See custom_criteria_example.py for domain-specific evaluation")
    print("  - See semantic_example.py for semantic similarity")


if __name__ == "__main__":
    asyncio.run(main())
