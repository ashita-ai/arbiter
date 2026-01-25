"""Groundedness Evaluation - RAG Source Attribution Validation

This example demonstrates how to validate that RAG (Retrieval-Augmented Generation)
outputs are properly grounded in their source documents, helping detect hallucinations
and ensuring citation accuracy.

Key Features:
- Source attribution validation
- Hallucination detection
- Citation mapping
- Grounded vs ungrounded statement categorization
- Direct evaluator usage for fine-grained control

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/groundedness_example.py
"""

import asyncio
import os

from dotenv import load_dotenv

from arbiter_ai import GroundednessEvaluator, evaluate
from arbiter_ai.core import LLMManager


async def main():
    """Run groundedness evaluation examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("Arbiter - Groundedness Evaluation Example")
    print("=" * 60)

    # Example 1: Well-grounded RAG output
    print("\nExample 1: Well-Grounded RAG Output")
    print("-" * 60)

    source_docs_1 = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.
    It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair.
    The tower is 330 meters (1,083 feet) tall and was the world's tallest man-made
    structure until 1930. Gustave Eiffel's company designed and built the tower.
    """

    rag_output_1 = """
    The Eiffel Tower was built between 1887 and 1889 for the World's Fair. It stands
    at 330 meters tall and was designed by Gustave Eiffel's company. The tower is
    located on the Champ de Mars in Paris.
    """

    result1 = await evaluate(
        output=rag_output_1,
        reference=source_docs_1,
        evaluators=["groundedness"],
        model="gpt-4o-mini",
    )

    print(f"RAG Output: {rag_output_1[:100]}...")
    print("\nResults:")
    print(f"  Groundedness Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result1.passed else 'No'}")

    for score in result1.scores:
        print(f"\n  {score.name}:")
        print(f"    Value: {score.value:.3f}")
        if score.confidence:
            print(f"    Confidence: {score.confidence:.3f}")
        if score.metadata.get("grounded_statements"):
            print("    Grounded Statements:")
            for stmt in score.metadata["grounded_statements"][:3]:
                print(f"      - {stmt}")
        if score.metadata.get("ungrounded_statements"):
            print("    Ungrounded Statements:")
            for stmt in score.metadata["ungrounded_statements"]:
                print(f"      - {stmt}")

    # Example 2: RAG output with hallucinations
    print("\n\nExample 2: RAG Output with Hallucinations")
    print("-" * 60)

    source_docs_2 = """
    Python was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability and supports multiple programming paradigms.
    Python 3.0, released in 2008, was a major revision that is not fully
    backward-compatible with Python 2.
    """

    rag_output_2 = """
    Python was created by Guido van Rossum in 1991. It is known for its readability
    and supports multiple programming paradigms. Python 3.0 was released in 2008 and
    included breaking changes from Python 2. Python was originally designed as a
    replacement for ABC programming language at Google.
    """

    result2 = await evaluate(
        output=rag_output_2,
        reference=source_docs_2,
        evaluators=["groundedness"],
        model="gpt-4o-mini",
    )

    print(f"RAG Output: {rag_output_2[:100]}...")
    print("\nResults:")
    print(f"  Groundedness Score: {result2.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result2.passed else 'No'}")

    for score in result2.scores:
        if score.metadata.get("ungrounded_statements"):
            print("\n  Detected Hallucinations:")
            for stmt in score.metadata["ungrounded_statements"]:
                print(f"    - {stmt}")

    # Example 3: Multi-document source validation
    print("\n\nExample 3: Multi-Document Source Validation")
    print("-" * 60)

    multi_source_docs = """
    SOURCE 1: Company financials
    Acme Corp reported revenue of $1.2 billion in Q3 2024, up 15% year-over-year.
    The company has 5,000 employees globally.

    SOURCE 2: Industry report
    The software industry grew 12% in 2024. Acme Corp is ranked #3 in their sector.
    Customer satisfaction scores averaged 4.5/5 across the industry.

    SOURCE 3: Press release
    Acme Corp announced a new product line launching in January 2025.
    The CEO stated the company plans to expand into European markets.
    """

    analyst_summary = """
    Acme Corp achieved $1.2 billion in Q3 2024 revenue, representing 15% growth.
    With 5,000 employees, they are the #3 player in the software sector. The company
    plans to expand into European markets with a new product line in January 2025.
    Industry growth was 12% overall, and Acme maintains a 4.8/5 customer satisfaction
    rating, above the industry average of 4.5/5.
    """

    result3 = await evaluate(
        output=analyst_summary,
        reference=multi_source_docs,
        evaluators=["groundedness"],
        model="gpt-4o-mini",
    )

    print(f"Analyst Summary: {analyst_summary[:100]}...")
    print("\nResults:")
    print(f"  Groundedness Score: {result3.overall_score:.3f}")
    print(f"  Passed: {'Yes' if result3.passed else 'No'}")

    for score in result3.scores:
        if score.metadata.get("grounded_count") is not None:
            grounded = score.metadata.get("grounded_count", 0)
            ungrounded = score.metadata.get("ungrounded_count", 0)
            total = grounded + ungrounded
            print(f"\n  Statement Breakdown:")
            print(f"    Grounded: {grounded}/{total}")
            print(f"    Ungrounded: {ungrounded}/{total}")
        if score.metadata.get("ungrounded_statements"):
            print("\n  Issues Found:")
            for stmt in score.metadata["ungrounded_statements"]:
                print(f"    - {stmt}")

    # Example 4: Direct evaluator usage with criteria
    print("\n\nExample 4: Direct Evaluator Usage with Criteria Focus")
    print("-" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = GroundednessEvaluator(client)

    medical_sources = """
    Clinical Study Results:
    The medication showed 78% efficacy in reducing symptoms over 12 weeks.
    Side effects occurred in 15% of patients, most commonly mild headaches.
    The study enrolled 500 participants aged 18-65.
    No serious adverse events were reported.
    """

    patient_summary = """
    Studies show the medication is about 78% effective at reducing symptoms.
    About 15% of patients experienced side effects, mainly headaches.
    The medication is safe with no serious side effects reported in clinical trials.
    """

    score = await evaluator.evaluate(
        output=patient_summary,
        reference=medical_sources,
        criteria="medical claims and safety statements",
    )

    print(f"Patient Summary: {patient_summary[:80]}...")
    print(f"\nGroundedness Score: {score.value:.3f}")
    print(f"Confidence: {score.confidence:.3f}")

    if score.metadata.get("citations"):
        print("\nCitation Mapping:")
        for stmt, source in list(score.metadata["citations"].items())[:3]:
            print(f"  '{stmt[:50]}...'")
            print(f"    -> '{source[:50]}...'")

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
    total_cost = cost1 + cost2 + cost3
    total_tokens_all = result1.total_tokens + result2.total_tokens + result3.total_tokens

    print("\nTotal Session Cost:")
    print("  Total Evaluations: 4")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens_all:,}")

    print("\nKey Features Demonstrated:")
    print("  - RAG output validation against source documents")
    print("  - Hallucination detection (statements not in sources)")
    print("  - Multi-document source attribution")
    print("  - Citation mapping (statement -> source text)")
    print("  - Criteria-focused groundedness evaluation")
    print("  - Direct evaluator usage for fine-grained control")

    print("\nRelated Examples:")
    print("  - See relevance_example.py for query-output alignment")
    print("  - See factuality_example.py for claim verification")
    print("  - See custom_criteria_example.py for domain-specific evaluation")


if __name__ == "__main__":
    asyncio.run(main())
