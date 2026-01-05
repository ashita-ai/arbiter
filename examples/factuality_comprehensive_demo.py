"""Comprehensive demonstration of Arbiter's factuality evaluation capabilities.

This example shows the complete factuality evaluation system:
- FactualityEvaluator with LLM-based hallucination detection
- SearchVerifier with Tavily API for web search validation
- KnowledgeBaseVerifier with Wikipedia for established facts
- CitationVerifier for RAG source attribution checking

The combined approach achieves higher accuracy than any single method alone.
"""

import asyncio
import os

from dotenv import load_dotenv

from arbiter_ai import FactualityEvaluator, LLMManager
from arbiter_ai.verifiers import (
    CitationVerifier,
    KnowledgeBaseVerifier,
    SearchVerifier,
)

# Load environment variables
load_dotenv()


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


async def demo_llm_only():
    """Demo 1: LLM-only factuality evaluation (baseline)."""
    print_section("Demo 1: LLM-Only Factuality Evaluation (Baseline)")

    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = FactualityEvaluator(llm_client=client)

    # Test with mixed facts
    output = """The Eiffel Tower in Paris was completed in 1889 for the World's Fair.
    It stands approximately 300 meters tall and was the world's tallest structure
    for 41 years. The tower weighs about 10,000 tons and was designed by Gustave Eiffel."""

    reference = """The Eiffel Tower was constructed from 1887 to 1889 as the entrance
    to the 1889 World's Fair. It stands 300 meters (984 feet) tall."""

    score = await evaluator.evaluate(output=output, reference=reference)

    print(f"\nOutput: {output[:100]}...")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"\nFactual Claims ({len(score.metadata['factual_claims'])}):")
    for claim in score.metadata["factual_claims"][:3]:
        print(f"  ✓ {claim}")
    print(f"\nNon-Factual Claims ({len(score.metadata['non_factual_claims'])}):")
    for claim in score.metadata["non_factual_claims"]:
        print(f"  ✗ {claim}")


async def demo_wikipedia_verifier():
    """Demo 2: Wikipedia-based fact verification."""
    print_section("Demo 2: Wikipedia Knowledge Base Verification")

    client = await LLMManager.get_client(model="gpt-4o-mini")
    wikipedia_verifier = KnowledgeBaseVerifier(max_results=3)
    evaluator = FactualityEvaluator(llm_client=client, verifiers=[wikipedia_verifier])

    output = "Paris is the capital of France and is located on the Seine River."

    score = await evaluator.evaluate(output=output)

    print(f"\nClaim: {output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"  - LLM Score: {score.metadata.get('llm_score', 0):.2f}")
    print(f"  - Combined with Wikipedia: {score.value:.2f}")
    print(f"\nVerification Used: {score.metadata.get('verification_used')}")
    print(f"Sources: {score.metadata.get('verification_sources', [])}")


async def demo_tavily_search():
    """Demo 3: Tavily web search verification."""
    print_section("Demo 3: Tavily Web Search Verification")

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("\n⚠️  TAVILY_API_KEY not found in environment")
        print("   Set in .env to run this demo")
        return

    try:
        client = await LLMManager.get_client(model="gpt-4o-mini")
        search_verifier = SearchVerifier(api_key=tavily_api_key, max_results=5)
        evaluator = FactualityEvaluator(llm_client=client, verifiers=[search_verifier])

        output = "Python 3.13 was released in October 2024 with significant performance improvements."

        score = await evaluator.evaluate(output=output)

        print(f"\nClaim: {output}")
        print(f"\nFactuality Score: {score.value:.2f}")
        print(f"  - LLM Score: {score.metadata.get('llm_score', 0):.2f}")
        print(f"  - Combined with Tavily: {score.value:.2f}")
        print("\nVerification Details:")
        print(f"  - Checks performed: {score.metadata.get('verification_count', 0)}")
        print(f"  - Sources: {score.metadata.get('verification_sources', [])}")

    except ImportError:
        print("\n⚠️  tavily-python not installed")
        print("   Run: pip install arbiter-ai[verifiers]")


async def demo_citation_checker():
    """Demo 4: Citation verification for RAG systems."""
    print_section("Demo 4: Citation Verification for RAG Systems")

    client = await LLMManager.get_client(model="gpt-4o-mini")
    citation_verifier = CitationVerifier(min_similarity=0.7)
    evaluator = FactualityEvaluator(llm_client=client, verifiers=[citation_verifier])

    # Simulated RAG output
    rag_output = (
        "The Eiffel Tower was designed by Gustave Eiffel and completed in 1889."
    )

    source_documents = """The Eiffel Tower is a wrought-iron lattice tower on the
    Champ de Mars in Paris, France. It was designed by engineer Gustave Eiffel
    and constructed from 1887 to 1889 as the entrance to the 1889 World's Fair."""

    score = await evaluator.evaluate(output=rag_output, reference=source_documents)

    print(f"\nRAG Output: {rag_output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"  - LLM Score: {score.metadata.get('llm_score', 0):.2f}")
    print(f"  - With Source Verification: {score.value:.2f}")
    print(f"\nGrounded in sources: {score.metadata.get('verification_used')}")


async def demo_all_verifiers():
    """Demo 5: Using all verifiers together (recommended approach)."""
    print_section("Demo 5: All Verifiers Combined (Maximum Accuracy)")

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Build verifier list
    verifiers = [
        CitationVerifier(),
        KnowledgeBaseVerifier(max_results=3),
    ]

    if tavily_api_key:
        try:
            verifiers.append(SearchVerifier(api_key=tavily_api_key, max_results=5))
            print("\n✓ Using all 3 verifiers: Citation + Wikipedia + Tavily")
        except ImportError:
            print("\n⚠️  Tavily not available, using Citation + Wikipedia only")
    else:
        print("\n⚠️  TAVILY_API_KEY not set, using Citation + Wikipedia only")

    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = FactualityEvaluator(llm_client=client, verifiers=verifiers)

    # Test with a complex claim that needs multiple verification sources
    output = """The International Space Station orbits Earth at an altitude of
    approximately 400 kilometers and completes an orbit every 90 minutes."""

    score = await evaluator.evaluate(output=output)

    print(f"\nClaim: {output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"  - LLM-only Score: {score.metadata.get('llm_score', 0):.2f}")
    print(f"  - Final (with verifiers): {score.value:.2f}")
    print("\nVerification Summary:")
    print(f"  - Verifiers used: {len(score.metadata.get('verification_sources', []))}")
    print(f"  - Verification checks: {score.metadata.get('verification_count', 0)}")
    print(f"  - Sources: {score.metadata.get('verification_sources', [])}")

    print("\n💡 Multi-verifier approach:")
    print("  - CitationVerifier: Checks source attribution")
    print("  - KnowledgeBaseVerifier: Validates against Wikipedia")
    print("  - SearchVerifier: Cross-checks with current web information")
    print("  - Combined scoring reduces hallucination false negatives")


async def demo_cost_tracking():
    """Demo 6: Cost tracking for factuality evaluation."""
    print_section("Demo 6: Cost Tracking and Performance")

    client = await LLMManager.get_client(model="gpt-4o-mini")

    # LLM-only (cheapest but less accurate)
    evaluator_llm = FactualityEvaluator(llm_client=client)

    # With all free verifiers
    evaluator_verified = FactualityEvaluator(
        llm_client=client,
        verifiers=[CitationVerifier(), KnowledgeBaseVerifier()],
    )

    test_claim = "Water boils at 100 degrees Celsius at sea level."

    print("\nComparing approaches...")

    # LLM-only
    score_llm = await evaluator_llm.evaluate(output=test_claim)
    interactions_llm = evaluator_llm.get_interactions()

    # With verification
    score_verified = await evaluator_verified.evaluate(output=test_claim)
    interactions_verified = evaluator_verified.get_interactions()

    print("\n1. LLM-Only Approach:")
    print(f"   Score: {score_llm.value:.2f}")
    print(f"   LLM Calls: {len(interactions_llm)}")
    print(f"   Tokens: {interactions_llm[0].total_tokens if interactions_llm else 0}")

    print("\n2. With Verification (Citation + Wikipedia):")
    print(f"   Score: {score_verified.value:.2f}")
    print(f"   LLM Calls: {len(interactions_verified)}")
    print(
        f"   Tokens: {interactions_verified[0].total_tokens if interactions_verified else 0}"
    )
    print(
        f"   External checks: {score_verified.metadata.get('verification_count', 0)} (FREE)"
    )

    print("\n💰 Cost Analysis:")
    print("   - LLM calls: Same cost for both")
    print("   - Wikipedia API: FREE")
    print("   - CitationVerifier: FREE (local computation)")
    print("   - Tavily API: ~$0.001 per search (5 searches ≈ $0.005)")
    print("   - Improved accuracy with minimal cost increase")


async def main():
    """Run all factuality evaluation demos."""
    print("\n" + "=" * 70)
    print(" Arbiter Factuality Evaluation - Comprehensive Demo")
    print("=" * 70)
    print("\nThis demo showcases Arbiter's multi-layered factuality evaluation:")
    print("  1. LLM-based claim extraction and verification")
    print("  2. Wikipedia knowledge base validation")
    print("  3. Tavily web search verification")
    print("  4. Source attribution checking (RAG systems)")
    print("  5. Combined scoring for maximum accuracy")

    await demo_llm_only()
    await demo_wikipedia_verifier()
    await demo_tavily_search()
    await demo_citation_checker()
    await demo_all_verifiers()
    await demo_cost_tracking()

    print("\n" + "=" * 70)
    print(" Summary: Why Arbiter's Factuality Evaluation is Awesome")
    print("=" * 70)
    print("\n1. Multi-Layered Verification:")
    print("   - LLM provides initial assessment and claim extraction")
    print("   - External verifiers catch what LLMs miss")
    print("   - Combined scoring: LLM 50% + verifiers 50%")
    print("\n2. Free External Verification:")
    print("   - Wikipedia API (no cost, no key required)")
    print("   - Citation checking (local computation)")
    print("   - Only Tavily requires API key (optional)")
    print("\n3. Automatic Cost Tracking:")
    print("   - Every LLM call tracked with token counts")
    print("   - Real dollar costs calculated")
    print("   - Compare accuracy vs cost trade-offs")
    print("\n4. Production-Ready:")
    print("   - Type-safe with strict mypy compliance")
    print("   - Comprehensive error handling")
    print("   - Async for high performance")
    print("   - Observable (every interaction logged)")
    print("\n5. Extensible:")
    print("   - Easy to add custom verifiers")
    print("   - Plugin architecture")
    print("   - Works with any LLM provider")


if __name__ == "__main__":
    asyncio.run(main())
