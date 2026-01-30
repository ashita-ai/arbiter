"""Example usage of InstructionFollowingEvaluator.

This example demonstrates how to use the InstructionFollowingEvaluator to
validate that LLM outputs adhere to specified instructions or constraints.

The evaluator is critical for:
- Agent output validation
- Structured output compliance
- Format requirement checking
- Content constraint verification
- Pipeline instruction adherence

Run with: python examples/instruction_following_example.py

Requires: OPENAI_API_KEY environment variable
"""

import asyncio
import os
from typing import Optional

# Check for API key before importing arbiter_ai
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable is required")
    print("Set it with: export OPENAI_API_KEY=sk-...")
    exit(1)

from arbiter_ai import InstructionFollowingEvaluator, LLMManager, evaluate


async def example_json_format_validation():
    """Validate JSON format compliance."""
    print("\n" + "=" * 60)
    print("Example 1: JSON Format Validation")
    print("=" * 60)

    # Output that follows JSON instructions
    good_output = '{"name": "Alice", "age": 30, "city": "New York"}'

    # Output that violates JSON instructions
    bad_output = "Name: Alice, Age: 30, City: New York"

    instructions = (
        "Respond in valid JSON format. "
        "Include 'name', 'age', and 'city' fields. "
        "Do not include any text outside the JSON object."
    )

    # Using the high-level API
    print("\nEvaluating good output (valid JSON):")
    result = await evaluate(
        output=good_output,
        criteria=instructions,
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"  Score: {result.overall_score:.2f}")
    print(f"  Cost: ${await result.total_llm_cost():.6f}")

    print("\nEvaluating bad output (not JSON):")
    result = await evaluate(
        output=bad_output,
        criteria=instructions,
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"  Score: {result.overall_score:.2f}")
    print(f"  Cost: ${await result.total_llm_cost():.6f}")


async def example_multiple_constraints():
    """Validate multiple content constraints."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Content Constraints")
    print("=" * 60)

    output = """Here is a brief summary:

- The project uses Python for backend
- React powers the frontend
- PostgreSQL handles data storage

In conclusion, this is a modern full-stack application."""

    instructions = """
    1. Keep response under 100 words
    2. Use bullet points for the main content
    3. Include a conclusion section
    4. Do not use first person pronouns
    """

    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = InstructionFollowingEvaluator(llm_client=client)

    score = await evaluator.evaluate(output=output, criteria=instructions)

    print(f"\nOutput:\n{output}")
    print(f"\nInstructions:\n{instructions}")
    print(f"\nScore: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"Violation Severity: {score.metadata['violation_severity']}")
    print(f"\nInstructions Followed: {score.metadata['instructions_followed']}")
    print(f"Instructions Violated: {score.metadata['instructions_violated']}")
    print(f"Instructions Partially Met: {score.metadata['instructions_partially_met']}")


async def example_agent_output_validation():
    """Validate agent tool use output."""
    print("\n" + "=" * 60)
    print("Example 3: Agent Tool Use Validation")
    print("=" * 60)

    # Simulated agent output from a customer service bot
    agent_output = """I understand you're having trouble with your order.

Let me look that up for you.

Based on your order #12345, I can see it was shipped on January 15th
and should arrive by January 20th.

Is there anything else I can help you with today?"""

    agent_instructions = """
    As a customer service agent, you must:
    1. Acknowledge the customer's concern
    2. Reference the specific order number when available
    3. Provide concrete next steps or information
    4. End with an offer to help further
    5. Maintain a professional and empathetic tone
    6. Never make promises about refunds without authorization
    """

    result = await evaluate(
        output=agent_output,
        criteria=agent_instructions,
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )

    print(f"\nAgent Output:\n{agent_output}")
    print(f"\nCompliance Score: {result.overall_score:.2f}")

    # Access detailed metrics
    for score in result.scores:
        print(f"\nViolation Severity: {score.metadata.get('violation_severity')}")
        print(f"Followed ({score.metadata.get('followed_count', 0)}):")
        for instruction in score.metadata.get("instructions_followed", []):
            print(f"  - {instruction}")


async def example_structured_output_compliance():
    """Validate structured output requirements."""
    print("\n" + "=" * 60)
    print("Example 4: Structured Output Compliance")
    print("=" * 60)

    # API response that should follow specific structure
    api_response = """{
    "status": "success",
    "data": {
        "user_id": "usr_123",
        "created_at": "2024-01-15T10:30:00Z",
        "profile": {
            "name": "John Doe",
            "email": "john@example.com"
        }
    },
    "metadata": {
        "request_id": "req_abc123",
        "processing_time_ms": 45
    }
}"""

    structure_requirements = """
    API response must:
    1. Be valid JSON
    2. Include a 'status' field with value 'success' or 'error'
    3. Include a 'data' object with the response payload
    4. Include a 'metadata' object with 'request_id' and 'processing_time_ms'
    5. Use snake_case for all field names
    6. Include ISO 8601 timestamps for any date fields
    """

    result = await evaluate(
        output=api_response,
        criteria=structure_requirements,
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )

    print(f"\nAPI Response Structure Score: {result.overall_score:.2f}")
    print(f"Evaluation Cost: ${await result.total_llm_cost():.6f}")


async def example_with_reference_context():
    """Validate output against instructions with reference context."""
    print("\n" + "=" * 60)
    print("Example 5: Instructions with Reference Context")
    print("=" * 60)

    # Original document (reference)
    reference = """
    Quarterly Financial Report - Q4 2024

    Revenue: $10.5 million (up 15% YoY)
    Operating Expenses: $7.2 million
    Net Income: $2.1 million
    Employee Count: 245

    Key Achievements:
    - Launched 3 new product lines
    - Expanded to 5 new markets
    - Achieved ISO 27001 certification
    """

    # Summary output to validate
    output = """
    Q4 2024 saw strong performance with revenue reaching $10.5M,
    a 15% increase year-over-year. The company expanded operations
    and launched new products while maintaining profitability.
    """

    instructions = """
    Summarize the financial report following these rules:
    1. Include the exact revenue figure
    2. Mention the year-over-year growth percentage
    3. Keep the summary under 50 words
    4. Do not include specific employee counts
    5. Focus on high-level achievements
    """

    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = InstructionFollowingEvaluator(llm_client=client)

    score = await evaluator.evaluate(
        output=output,
        reference=reference,
        criteria=instructions,
    )

    print(f"\nReference Document: [Financial Report]")
    print(f"\nSummary Output:\n{output}")
    print(f"\nInstruction Adherence Score: {score.value:.2f}")
    print(f"Violation Severity: {score.metadata['violation_severity']}")


async def example_comparison_with_semantic():
    """Compare instruction following with semantic evaluation."""
    print("\n" + "=" * 60)
    print("Example 6: Combining with Semantic Evaluation")
    print("=" * 60)

    output = "The capital of France is Paris, a beautiful city known for the Eiffel Tower."
    reference = "Paris is the capital of France."
    instructions = "Answer in one sentence. Do not include opinions or adjectives."

    # Run both evaluators
    result = await evaluate(
        output=output,
        reference=reference,
        criteria=instructions,
        evaluators=["semantic", "instruction_following"],
        model="gpt-4o-mini",
    )

    print(f"\nOutput: {output}")
    print(f"Reference: {reference}")
    print(f"Instructions: {instructions}")
    print(f"\nOverall Score: {result.overall_score:.2f}")

    for score in result.scores:
        print(f"\n{score.name}:")
        print(f"  Score: {score.value:.2f}")
        print(f"  Confidence: {score.confidence:.2f}")


async def main():
    """Run all examples."""
    print("InstructionFollowingEvaluator Examples")
    print("=" * 60)

    await example_json_format_validation()
    await example_multiple_constraints()
    await example_agent_output_validation()
    await example_structured_output_compliance()
    await example_with_reference_context()
    await example_comparison_with_semantic()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
