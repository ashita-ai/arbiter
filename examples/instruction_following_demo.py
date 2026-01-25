"""Demo: InstructionFollowingEvaluator with real LLM calls."""

import asyncio

from arbiter_ai import InstructionFollowingEvaluator, LLMManager, evaluate


async def main():
    print("=" * 70)
    print("InstructionFollowingEvaluator Demo - Real LLM Calls")
    print("=" * 70)

    # Test 1: Full compliance - JSON format with required fields
    print("\n--- Test 1: Full Compliance (JSON with required fields) ---")
    result1 = await evaluate(
        output='{"name": "Alice", "age": 28, "city": "Seattle"}',
        criteria="Respond in valid JSON format. Include 'name', 'age', and 'city' fields.",
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"Output: {{'name': 'Alice', 'age': 28, 'city': 'Seattle'}}")
    print(f"Score: {result1.overall_score:.2f}")
    print(f"Severity: {result1.scores[0].metadata.get('violation_severity')}")
    print(f"Followed: {result1.scores[0].metadata.get('instructions_followed')}")

    # Test 2: Partial compliance - format correct but missing field
    print("\n--- Test 2: Partial Compliance (missing required field) ---")
    result2 = await evaluate(
        output='{"name": "Bob", "age": 35}',
        criteria="Respond in valid JSON format. Include 'name', 'age', and 'city' fields.",
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"Output: {{'name': 'Bob', 'age': 35}}")
    print(f"Score: {result2.overall_score:.2f}")
    print(f"Severity: {result2.scores[0].metadata.get('violation_severity')}")
    print(f"Violated: {result2.scores[0].metadata.get('instructions_violated')}")

    # Test 3: Major violation - wrong format entirely
    print("\n--- Test 3: Major Violation (plain text instead of JSON) ---")
    result3 = await evaluate(
        output="My name is Charlie and I am 42 years old from Portland.",
        criteria="Respond in valid JSON format. Include 'name', 'age', and 'city' fields.",
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"Output: 'My name is Charlie and I am 42 years old from Portland.'")
    print(f"Score: {result3.overall_score:.2f}")
    print(f"Severity: {result3.scores[0].metadata.get('violation_severity')}")
    print(f"Violated: {result3.scores[0].metadata.get('instructions_violated')}")

    # Test 4: Bullet point format compliance
    print("\n--- Test 4: Bullet Point Format ---")
    result4 = await evaluate(
        output="Here is the summary:\n- First key point\n- Second key point\n- Third key point",
        criteria="Provide exactly 3 bullet points. Start with 'Here is the summary:'",
        evaluators=["instruction_following"],
        model="gpt-4o-mini",
    )
    print(f"Output: 'Here is the summary:\\n- First key point\\n- Second...'")
    print(f"Score: {result4.overall_score:.2f}")
    print(f"Severity: {result4.scores[0].metadata.get('violation_severity')}")

    # Test 5: Direct evaluator usage with detailed output
    print("\n--- Test 5: Direct Evaluator Usage ---")
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = InstructionFollowingEvaluator(llm_client=client)

    score = await evaluator.evaluate(
        output="The answer is 42. This is a single sentence response.",
        criteria="Respond in exactly one sentence. Do not use numbers.",
    )
    print(f"Output: 'The answer is 42. This is a single sentence response.'")
    print(f"Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"Severity: {score.metadata.get('violation_severity')}")
    print(f"Followed: {score.metadata.get('instructions_followed')}")
    print(f"Violated: {score.metadata.get('instructions_violated')}")
    print(f"Explanation: {score.explanation[:150]}...")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
