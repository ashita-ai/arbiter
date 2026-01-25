"""Example: Instruction Following Evaluation.

This example demonstrates how to use the InstructionFollowingEvaluator
to validate that LLM outputs adhere to explicit instructions.

Use cases:
- Agent pipeline validation
- Structured output compliance (JSON, markdown, etc.)
- Format requirement checking
- Length/word count limit verification
- Tone/style requirement validation
"""

import asyncio

from arbiter_ai import InstructionFollowingEvaluator, LLMManager, evaluate


async def basic_instruction_following():
    """Basic example of instruction following evaluation."""
    print("=" * 60)
    print("Basic Instruction Following Example")
    print("=" * 60)

    # Example: Checking if output follows JSON format instructions
    result = await evaluate(
        output='{"name": "Alice", "age": 28, "city": "Seattle"}',
        criteria="Respond in valid JSON format. Include 'name', 'age', and 'city' fields.",
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )

    print(f"\nOutput: {'{\"name\": \"Alice\", \"age\": 28, \"city\": \"Seattle\"}'}")
    print(f"Instructions: Respond in valid JSON format. Include required fields.")
    print(f"\nScore: {result.overall_score:.2f}")

    for score in result.scores:
        print(f"\nEvaluator: {score.name}")
        print(f"  Score: {score.value:.2f}")
        print(f"  Confidence: {score.confidence:.2f}")
        print(f"  Explanation: {score.explanation[:100]}...")
        if "instructions_followed" in score.metadata:
            print(f"  Instructions Followed: {score.metadata['instructions_followed']}")
        if "instructions_violated" in score.metadata:
            print(f"  Instructions Violated: {score.metadata['instructions_violated']}")
        if "violation_severity" in score.metadata:
            print(f"  Violation Severity: {score.metadata['violation_severity']}")


async def direct_evaluator_usage():
    """Example using the evaluator directly for more control."""
    print("\n" + "=" * 60)
    print("Direct Evaluator Usage Example")
    print("=" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = InstructionFollowingEvaluator(llm_client=client)

    # Test case 1: Full compliance
    print("\n--- Test 1: Full Compliance ---")
    score = await evaluator.evaluate(
        output="Here is the summary in exactly 3 bullet points:\n"
        "- Point one about the topic\n"
        "- Point two with details\n"
        "- Point three for conclusion",
        criteria="Provide exactly 3 bullet points. Start with 'Here is the summary'.",
    )

    print(f"Score: {score.value:.2f}")
    print(f"Severity: {score.metadata['violation_severity']}")
    print(f"Followed: {score.metadata['instructions_followed']}")

    # Test case 2: Partial compliance
    print("\n--- Test 2: Partial Compliance ---")
    evaluator.clear_interactions()

    score = await evaluator.evaluate(
        output="The answer is: Paris is the capital of France. "
        "It is known for the Eiffel Tower and its rich history.",
        criteria="Respond in exactly one sentence. Do not mention landmarks.",
    )

    print(f"Score: {score.value:.2f}")
    print(f"Severity: {score.metadata['violation_severity']}")
    print(f"Followed: {score.metadata['instructions_followed']}")
    print(f"Violated: {score.metadata['instructions_violated']}")
    print(f"Partially Met: {score.metadata['instructions_partially_met']}")


async def agent_pipeline_validation():
    """Example: Validating agent pipeline outputs."""
    print("\n" + "=" * 60)
    print("Agent Pipeline Validation Example")
    print("=" * 60)

    # Simulated agent output that should follow specific instructions
    agent_output = """
{
    "action": "search",
    "query": "weather in Seattle",
    "confidence": 0.95
}
"""

    agent_instructions = """
You must respond with a JSON object containing:
1. An 'action' field with one of: search, calculate, respond
2. A 'query' field with the processed query
3. A 'confidence' field between 0 and 1
Do not include any text outside the JSON object.
"""

    result = await evaluate(
        output=agent_output.strip(),
        criteria=agent_instructions,
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )

    print(f"\nAgent Output:\n{agent_output}")
    print(f"Score: {result.overall_score:.2f}")

    for score in result.scores:
        print(f"\nSeverity: {score.metadata.get('violation_severity', 'N/A')}")
        print(f"Instructions Followed ({score.metadata.get('instructions_followed_count', 0)}):")
        for instr in score.metadata.get("instructions_followed", []):
            print(f"  - {instr}")


async def format_compliance_checking():
    """Example: Checking format compliance for structured outputs."""
    print("\n" + "=" * 60)
    print("Format Compliance Checking Example")
    print("=" * 60)

    test_cases = [
        {
            "name": "Valid Markdown",
            "output": "# Title\n\nThis is a paragraph.\n\n## Section\n\n- Item 1\n- Item 2",
            "criteria": "Use Markdown format. Include a title with #. Include at least one section with ##.",
        },
        {
            "name": "Invalid Format",
            "output": "Title\n\nThis is plain text without any formatting.",
            "criteria": "Use Markdown format. Include a title with #. Include at least one section with ##.",
        },
    ]

    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        result = await evaluate(
            output=test["output"],
            criteria=test["criteria"],
            evaluators=["instruction_following"],
            model="gpt-4o-mini",
        )

        score = result.scores[0]
        print(f"Output Preview: {test['output'][:50]}...")
        print(f"Score: {score.value:.2f}")
        print(f"Severity: {score.metadata.get('violation_severity', 'N/A')}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Instruction Following Evaluator Examples")
    print("=" * 60)

    await basic_instruction_following()
    await direct_evaluator_usage()
    await agent_pipeline_validation()
    await format_compliance_checking()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
