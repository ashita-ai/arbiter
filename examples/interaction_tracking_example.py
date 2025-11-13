"""LLM Interaction Tracking Example - Showcasing Arbiter's Unique Observability

This example demonstrates Arbiter's automatic LLM interaction tracking feature,
which provides complete transparency into how evaluations are performed.

**Key Features Shown:**
- Automatic tracking (no manual instrumentation needed)
- Complete audit trail (every LLM call recorded)
- Cost analysis (token usage and cost calculation)
- Debugging capabilities (inspect prompts and responses)
- Performance monitoring (latency tracking)
- Multi-evaluator transparency (see interactions from each evaluator)

**Why This Matters:**
- **Cost Visibility**: Know exactly what each evaluation costs
- **Debugging**: See the exact prompts and responses used
- **Compliance**: Complete audit trail for regulatory requirements
- **Optimization**: Identify slow or expensive operations

Run with:
    python examples/interaction_tracking_example.py
"""

from dotenv import load_dotenv

import asyncio
import os
from datetime import datetime
from typing import Dict, List

from arbiter import evaluate
from arbiter.core import LLMManager
from arbiter.core.models import LLMInteraction


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def print_interaction_details(interaction: LLMInteraction, index: int):
    """Print detailed information about an LLM interaction."""
    print(f"\n  📞 Interaction #{index}:")
    print(f"     Purpose: {interaction.purpose}")
    print(f"     Model: {interaction.model}")
    print(f"     Latency: {interaction.latency:.3f}s")
    print(f"     Tokens: {interaction.tokens_used:,}")
    print(f"     Timestamp: {format_timestamp(interaction.timestamp)}")

    # Show metadata if present
    if interaction.metadata:
        print(f"     Metadata:")
        for key, value in interaction.metadata.items():
            if key != "system_prompt":  # Don't show full system prompt
                print(f"       • {key}: {value}")

    # Show prompt preview (first 100 chars)
    prompt_preview = interaction.prompt[:100] + "..." if len(interaction.prompt) > 100 else interaction.prompt
    print(f"     Prompt Preview: {prompt_preview}")

    # Show response preview (first 100 chars)
    response_preview = interaction.response[:100] + "..." if len(interaction.response) > 100 else interaction.response
    print(f"     Response Preview: {response_preview}")


def analyze_interactions(interactions: List[LLMInteraction]) -> Dict:
    """Analyze interactions and return summary statistics."""
    if not interactions:
        return {}

    total_tokens = sum(i.tokens_used for i in interactions)
    total_latency = sum(i.latency for i in interactions)
    avg_latency = total_latency / len(interactions) if interactions else 0

    # Group by purpose
    by_purpose: Dict[str, List[LLMInteraction]] = {}
    for interaction in interactions:
        purpose = interaction.purpose
        if purpose not in by_purpose:
            by_purpose[purpose] = []
        by_purpose[purpose].append(interaction)

    # Group by model
    by_model: Dict[str, List[LLMInteraction]] = {}
    for interaction in interactions:
        model = interaction.model
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(interaction)

    return {
        "total_interactions": len(interactions),
        "total_tokens": total_tokens,
        "total_latency": total_latency,
        "avg_latency": avg_latency,
        "by_purpose": by_purpose,
        "by_model": by_model,
    }


async def main():
    """Run interaction tracking examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔬 Arbiter - LLM Interaction Tracking Example")
    print("=" * 70)
    print("\nThis example showcases Arbiter's automatic interaction tracking,")
    print("which provides complete transparency into evaluation operations.")
    print("=" * 70)

    # Example 1: Single Evaluator - Basic Tracking
    print("\n\n📊 Example 1: Single Evaluator - Basic Tracking")
    print("-" * 70)

    result1 = await evaluate(
        output="Paris is the capital of France and a major European city.",
        reference="The capital of France is Paris, which is located in Europe.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")
    print(f"   Score: {result1.overall_score:.3f}")
    print(f"   Processing Time: {result1.processing_time:.3f}s")

    print(f"\n🔍 Interaction Tracking (Automatic - No Manual Instrumentation):")
    print(f"   Total LLM Calls: {len(result1.interactions)}")

    # Show each interaction
    for i, interaction in enumerate(result1.interactions, 1):
        print_interaction_details(interaction, i)

    # Cost analysis
    cost_per_1k = 0.15 / 1000  # GPT-4o-mini pricing ($0.15 per 1M tokens)
    total_cost = result1.total_llm_cost(cost_per_1k_tokens=cost_per_1k)
    print(f"\n💰 Cost Analysis:")
    print(f"   Total Tokens: {result1.total_tokens:,}")
    print(f"   Estimated Cost: ${total_cost:.6f}")
    print(f"   Cost per Evaluation: ${total_cost:.6f}")

    # Example 2: Multiple Evaluators - See Interactions from Each
    print("\n\n📊 Example 2: Multiple Evaluators - Complete Transparency")
    print("-" * 70)

    result2 = await evaluate(
        output="Our new product is revolutionary and will change everything!",
        criteria="Professional tone, factual accuracy, no hyperbole",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Multi-Evaluator Evaluation Complete!")
    print(f"   Score: {result2.overall_score:.3f}")
    print(f"   Evaluators Used: {', '.join(result2.evaluator_names)}")
    print(f"   Processing Time: {result2.processing_time:.3f}s")

    # Analyze interactions
    analysis = analyze_interactions(result2.interactions)

    print(f"\n🔍 Interaction Analysis:")
    print(f"   Total LLM Calls: {analysis['total_interactions']}")
    print(f"   Total Tokens: {analysis['total_tokens']:,}")
    print(f"   Total Latency: {analysis['total_latency']:.3f}s")
    print(f"   Average Latency: {analysis['avg_latency']:.3f}s")

    # Show interactions grouped by purpose
    print(f"\n📋 Interactions by Purpose:")
    for purpose, interactions in analysis['by_purpose'].items():
        total_tokens = sum(i.tokens_used for i in interactions)
        total_latency = sum(i.latency for i in interactions)
        print(f"   • {purpose}:")
        print(f"     - Calls: {len(interactions)}")
        print(f"     - Tokens: {total_tokens:,}")
        print(f"     - Latency: {total_latency:.3f}s")

    # Show all interactions
    print(f"\n🔬 All Interactions:")
    for i, interaction in enumerate(result2.interactions, 1):
        print_interaction_details(interaction, i)

    # Cost breakdown
    total_cost2 = result2.total_llm_cost(cost_per_1k_tokens=cost_per_1k)
    print(f"\n💰 Cost Breakdown:")
    print(f"   Total Cost: ${total_cost2:.6f}")
    for purpose, interactions in analysis['by_purpose'].items():
        purpose_tokens = sum(i.tokens_used for i in interactions)
        purpose_cost = (purpose_tokens / 1000) * cost_per_1k
        print(f"   • {purpose}: ${purpose_cost:.6f} ({purpose_tokens:,} tokens)")

    # Example 3: Debugging - Inspect Prompts and Responses
    print("\n\n📊 Example 3: Debugging - Inspect Exact Prompts and Responses")
    print("-" * 70)

    result3 = await evaluate(
        output="The quick brown fox jumps over the lazy dog",
        reference="A fast brown fox leaps above a sleepy canine",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")
    print(f"   Score: {result3.overall_score:.3f}")

    # Show full prompt and response for debugging
    if result3.interactions:
        interaction = result3.interactions[0]
        print(f"\n🔍 Full Interaction Details (for debugging):")
        print(f"\n   📝 Full Prompt:")
        print(f"   {'-' * 66}")
        print(f"   {interaction.prompt}")
        print(f"   {'-' * 66}")

        print(f"\n   📤 Full Response:")
        print(f"   {'-' * 66}")
        print(f"   {interaction.response}")
        print(f"   {'-' * 66}")

        print(f"\n   💡 Use Case: Debug why a score is what it is")
        print(f"      - See exactly what prompt was sent to the LLM")
        print(f"      - See exactly what response was received")
        print(f"      - Understand how the score was computed")

    # Example 4: Audit Trail - Complete Transparency
    print("\n\n📊 Example 4: Audit Trail - Complete Transparency")
    print("-" * 70)

    result4 = await evaluate(
        output="Medical advice: Take 2 aspirin and call me in the morning.",
        criteria="Medical accuracy, appropriate disclaimer, professional tone",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")
    print(f"   Score: {result4.overall_score:.3f}")
    print(f"   Timestamp: {format_timestamp(result4.timestamp)}")

    print(f"\n📋 Complete Audit Trail:")
    print(f"   • Evaluation ID: {id(result4)}")
    print(f"   • Timestamp: {format_timestamp(result4.timestamp)}")
    print(f"   • Evaluators: {', '.join(result4.evaluator_names)}")
    print(f"   • Processing Time: {result4.processing_time:.3f}s")
    print(f"   • Total Tokens: {result4.total_tokens:,}")

    print(f"\n   🔍 All LLM Interactions:")
    for i, interaction in enumerate(result4.interactions, 1):
        print(f"   {i}. {interaction.purpose}")
        print(f"      Model: {interaction.model}")
        print(f"      Tokens: {interaction.tokens_used:,}")
        print(f"      Latency: {interaction.latency:.3f}s")
        print(f"      Timestamp: {format_timestamp(interaction.timestamp)}")

    print(f"\n   💡 Use Case: Compliance and Regulatory Requirements")
    print(f"      - Complete audit trail of all LLM usage")
    print(f"      - Timestamped records for every call")
    print(f"      - Full transparency for audits")

    # Example 5: Performance Monitoring - Identify Bottlenecks
    print("\n\n📊 Example 5: Performance Monitoring - Identify Bottlenecks")
    print("-" * 70)

    result5 = await evaluate(
        output="This is a longer output that requires more processing time to evaluate properly. "
               "It contains multiple sentences and complex ideas that need careful analysis.",
        reference="This reference text is also longer and contains multiple concepts that need "
                  "to be compared against the output for semantic similarity.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")

    # Performance analysis
    if result5.interactions:
        interaction = result5.interactions[0]
        tokens_per_second = interaction.tokens_used / interaction.latency if interaction.latency > 0 else 0

        print(f"\n⚡ Performance Metrics:")
        print(f"   Latency: {interaction.latency:.3f}s")
        print(f"   Tokens: {interaction.tokens_used:,}")
        print(f"   Throughput: {tokens_per_second:.1f} tokens/second")
        print(f"   Model: {interaction.model}")

        # Performance recommendations
        print(f"\n   💡 Performance Insights:")
        if interaction.latency > 2.0:
            print(f"      ⚠️  High latency detected ({interaction.latency:.3f}s)")
            print(f"         Consider: Using a faster model (gpt-4o-mini)")
        if interaction.tokens_used > 1000:
            print(f"      ⚠️  High token usage ({interaction.tokens_used:,} tokens)")
            print(f"         Consider: Shorter prompts or caching")
        if interaction.latency < 0.5:
            print(f"      ✅ Low latency - good performance!")

    # Summary
    print("\n\n" + "=" * 70)
    print("✅ All Examples Complete!")
    print("=" * 70)

    print("\n🎯 Key Takeaways:")
    print("   • Automatic tracking - no manual instrumentation needed")
    print("   • Complete transparency - see every LLM call")
    print("   • Cost visibility - know exactly what evaluations cost")
    print("   • Debugging support - inspect prompts and responses")
    print("   • Audit trails - full compliance support")
    print("   • Performance monitoring - identify bottlenecks")

    print("\n💡 Real-World Use Cases:")
    print("   • Cost Optimization: Track token usage and optimize prompts")
    print("   • Debugging: See why scores are what they are")
    print("   • Compliance: Complete audit trail for regulations")
    print("   • Performance: Identify slow operations")
    print("   • Transparency: Show stakeholders exactly how evaluations work")

    print("\n📚 Learn More:")
    print("   • See other examples: examples/basic_evaluation.py")
    print("   • Multi-evaluator usage: examples/multiple_evaluators.py")
    print("   • Error handling: examples/error_handling_example.py")


if __name__ == "__main__":
    asyncio.run(main())

