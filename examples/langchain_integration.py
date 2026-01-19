"""LangChain integration example for Arbiter.

This example shows how to:
1. Evaluate LangChain chain outputs
2. Evaluate RAG responses for groundedness
3. Use Arbiter's cost tracking with LangChain pipelines

Requirements:
    pip install langchain langchain-openai

Environment:
    OPENAI_API_KEY must be set in .env or environment

Run with:
    python examples/langchain_integration.py
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


async def evaluate_rag_chain():
    """Evaluate a RAG chain with groundedness checking."""

    # Simulated RAG response with sources
    sources = [
        "Paris is the capital and largest city of France.",
        "The Eiffel Tower is located in Paris.",
    ]
    rag_response = "Paris is the capital of France and home to the Eiffel Tower."

    # Evaluate groundedness (is response supported by sources?)
    result = await evaluate(
        output=rag_response,
        reference="\n".join(sources),
        evaluators=["groundedness"],
        model="gpt-4o-mini",
    )

    print(f"RAG Response: {rag_response}")
    print(f"Sources: {sources}")
    print(f"\nGroundedness Evaluation:")
    print(f"  Score: {result.overall_score:.2f}")
    print(f"  Passed: {result.passed}")
    print(f"  Cost: ${await result.total_llm_cost():.6f}")

    return result


async def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        print("Please set it in .env file or environment.")
        return

    print("=" * 60)
    print("LangChain + Arbiter Integration Example")
    print("=" * 60)

    print("\n1. Evaluating Chain Output:")
    print("-" * 40)
    await evaluate_chain_output()

    print("\n" + "=" * 60)
    print("\n2. Evaluating RAG Groundedness:")
    print("-" * 40)
    await evaluate_rag_chain()


if __name__ == "__main__":
    asyncio.run(main())
