"""Example of evaluating LangChain outputs with Arbiter.

This example shows how to:
1. Evaluate a LangChain chain's output for correctness.
2. Track evaluation costs and LLM interactions.

Requirements:
    pip install langchain langchain-openai

Environment:
    OPENAI_API_KEY must be set in .env or environment

Run with:
    python examples/evaluating_langchain_outputs.py
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Check for LangChain dependencies before importing
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from arbiter_ai import evaluate

# Load environment variables from .env
load_dotenv()


async def evaluate_chain_output():
    """Evaluate a LangChain chain's output quality."""

    # Create a simple LangChain chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template(
        "Answer this question concisely: {question}"
    )
    chain = prompt | llm | StrOutputParser()

    # Run the chain
    question = "What is the capital of France?"
    response = await chain.ainvoke({"question": question})

    print(f"Question: {question}")
    print(f"Response: {response}")

    # Evaluate with Arbiter
    result = await evaluate(
        output=response,
        reference="Paris is the capital of France.",
        evaluators=["semantic", "factuality"],
        model="gpt-4o-mini",
    )

    print(f"\nEvaluation:")
    print(f"  Score: {result.overall_score:.2f}")
    print(f"  Passed: {result.passed}")
    print(f"  Cost: ${await result.total_llm_cost():.6f}")
    print(f"  LLM Calls: {len(result.interactions)}")

    return result


async def main():
    # Check for LangChain dependencies
    if not LANGCHAIN_AVAILABLE:
        print("=" * 60)
        print("LangChain dependencies not installed")
        print("=" * 60)
        print("\nThis example requires LangChain. Install with:")
        print("  pip install langchain langchain-openai")
        print("\nOr with uv:")
        print("  uv pip install langchain langchain-openai")
        sys.exit(0)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        print("Please set it in .env file or environment.")
        return

    print("=" * 60)
    print("Evaluating LangChain Outputs with Arbiter")
    print("=" * 60)

    print("\n1. Evaluating a simple chain's output:")
    print("-" * 40)
    await evaluate_chain_output()


if __name__ == "__main__":
    asyncio.run(main())
