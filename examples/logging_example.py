#!/usr/bin/env python3
"""Example: Structured logging with Arbiter.

This example demonstrates how to enable and configure structured logging
in Arbiter to help debug evaluations and understand what's happening
under the hood.

## Log Levels:
- DEBUG: Token counts, prompt lengths, timing details
- INFO: Evaluation start/complete, batch progress
- WARNING: Retries, fallbacks, issues (default)
- ERROR: Failed evaluations, API errors

## Usage:
    # Enable debug logging
    python examples/logging_example.py

    # Or in your code:
    import logging
    from arbiter import configure_logging
    configure_logging(level=logging.DEBUG)
"""

import asyncio
import logging

from arbiter_ai import configure_logging, evaluate


async def main() -> None:
    """Run evaluation with structured logging enabled."""
    # Method 1: Use the convenience function
    # This sets up logging for all Arbiter components
    configure_logging(level=logging.DEBUG)

    print("=" * 60)
    print("Running evaluation with DEBUG logging enabled...")
    print("=" * 60)
    print()

    # Run an evaluation - you'll see debug logs in stderr
    try:
        result = await evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )
        print(f"\nEvaluation result: score={result.overall_score:.2f}")
    except Exception as e:
        print(f"\nNote: Evaluation failed (expected if no API key): {e}")
        print("The logging output above shows what WOULD be logged during evaluation.")

    print()
    print("=" * 60)
    print("Method 2: Configure specific component loggers")
    print("=" * 60)
    print()

    # Method 2: Configure specific component loggers directly
    # This gives you fine-grained control

    # Only show LLM calls (DEBUG) but keep API at INFO level
    logging.getLogger("arbiter.llm").setLevel(logging.DEBUG)
    logging.getLogger("arbiter.api").setLevel(logging.INFO)
    logging.getLogger("arbiter.evaluators").setLevel(logging.WARNING)

    print("Now running with mixed log levels:")
    print("  - arbiter.llm: DEBUG (shows all LLM calls)")
    print("  - arbiter.api: INFO (shows evaluation start/complete)")
    print("  - arbiter.evaluators: WARNING (quiet)")
    print()

    try:
        result = await evaluate(
            output="Tokyo is the capital of Japan",
            reference="Japan's capital is Tokyo",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )
        print(f"\nEvaluation result: score={result.overall_score:.2f}")
    except Exception as e:
        print(f"\nNote: Evaluation failed (expected if no API key): {e}")

    print()
    print("=" * 60)
    print("Method 3: Log to a file")
    print("=" * 60)
    print()

    # Method 3: Log to a file
    file_handler = logging.FileHandler("arbiter_debug.log")
    configure_logging(
        level=logging.DEBUG,
        handler=file_handler,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("Logging to arbiter_debug.log with timestamps...")

    try:
        result = await evaluate(
            output="Berlin is the capital of Germany",
            reference="Germany's capital is Berlin",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )
        print(f"Evaluation result: score={result.overall_score:.2f}")
        print("Check arbiter_debug.log for detailed logs.")
    except Exception as e:
        print(f"Note: Evaluation failed (expected if no API key): {e}")
        print("Check arbiter_debug.log for what would be logged.")


if __name__ == "__main__":
    asyncio.run(main())
