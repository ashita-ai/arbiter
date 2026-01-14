"""Caching Middleware - Avoiding Duplicate API Calls

This example demonstrates using the enhanced CachingMiddleware for
cost savings through caching evaluation results.

Key Features:
- Memory backend (default) with LRU eviction
- Redis backend for distributed caching
- TTL-based cache expiration
- SHA256 deterministic cache keys
- Cost savings tracking

Requirements:
    export OPENAI_API_KEY=your_key_here
    # Optional for Redis backend:
    export REDIS_URL=redis://localhost:6379

Run with:
    python examples/caching_example.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from arbiter_ai import evaluate
from arbiter_ai.core import MiddlewarePipeline
from arbiter_ai.core.middleware import CachingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def memory_backend_example() -> None:
    """Demonstrate memory backend caching."""
    print("\n1. Memory Backend Example")
    print("-" * 60)

    # Create cache with 1 hour TTL and max 100 entries
    cache = CachingMiddleware(
        backend="memory",
        max_size=100,
        ttl=3600,  # 1 hour
    )
    pipeline = MiddlewarePipeline([cache])

    # Same evaluation multiple times
    output = "Paris is the capital of France"
    reference = "The capital of France is Paris"

    print("\nRunning same evaluation 3 times...")

    for i in range(3):
        result = await evaluate(
            output=output,
            reference=reference,
            evaluators=["semantic"],
            middleware=pipeline,
            model="gpt-4o-mini",
        )
        stats = cache.get_stats()
        print(f"  Run {i + 1}: Score={result.overall_score:.3f}, "
              f"Hits={stats['hits']}, Misses={stats['misses']}")

    # Final statistics
    final_stats = cache.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Cache Hits: {final_stats['hits']}")
    print(f"  Cache Misses: {final_stats['misses']}")
    print(f"  Hit Rate: {final_stats['hit_rate']:.1%}")
    print(f"  Estimated Savings: ${final_stats['estimated_savings_usd']:.6f}")
    print(f"  Cache Size: {final_stats['size']}/{final_stats['max_size']}")


async def different_inputs_example() -> None:
    """Show cache behavior with different inputs."""
    print("\n2. Cache Key Differentiation")
    print("-" * 60)

    cache = CachingMiddleware(max_size=10)
    pipeline = MiddlewarePipeline([cache])

    # Different outputs should have different cache keys
    outputs = [
        "Paris is the capital of France",
        "London is the capital of England",
        "Berlin is the capital of Germany",
    ]
    reference = "European capital city"

    print("\nRunning evaluations with different outputs...")

    for output in outputs:
        result = await evaluate(
            output=output,
            reference=reference,
            evaluators=["semantic"],
            middleware=pipeline,
            model="gpt-4o-mini",
        )
        print(f"  '{output[:30]}...' -> Score={result.overall_score:.3f}")

    stats = cache.get_stats()
    print(f"\nAll different outputs = 3 misses, 0 hits:")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

    # Now repeat one evaluation - should hit cache
    print("\nRepeating first evaluation...")
    result = await evaluate(
        output=outputs[0],
        reference=reference,
        evaluators=["semantic"],
        middleware=pipeline,
        model="gpt-4o-mini",
    )
    stats = cache.get_stats()
    print(f"  Cache Hit! Hits: {stats['hits']}, Misses: {stats['misses']}")


async def lru_eviction_example() -> None:
    """Demonstrate LRU eviction behavior."""
    print("\n3. LRU Eviction Behavior")
    print("-" * 60)

    # Very small cache to demonstrate eviction
    cache = CachingMiddleware(max_size=2)
    pipeline = MiddlewarePipeline([cache])

    reference = "Test reference"

    print("\nCache size = 2, adding 3 different entries...")

    # Add 3 entries - first should be evicted
    for i in range(3):
        await evaluate(
            output=f"Output {i + 1}",
            reference=reference,
            evaluators=["semantic"],
            middleware=pipeline,
            model="gpt-4o-mini",
        )
        print(f"  Added entry {i + 1}, Cache size: {cache.get_stats()['size']}")

    # First entry should be evicted, second and third should remain
    print("\nVerifying eviction (re-checking entries)...")

    # Check entry 1 (should be miss - evicted)
    await evaluate(
        output="Output 1",
        reference=reference,
        evaluators=["semantic"],
        middleware=pipeline,
        model="gpt-4o-mini",
    )
    print(f"  Output 1: {'MISS (evicted)' if cache.misses == 4 else 'HIT'}")

    # Check entry 3 (should be hit - still in cache)
    prev_hits = cache.hits
    await evaluate(
        output="Output 3",
        reference=reference,
        evaluators=["semantic"],
        middleware=pipeline,
        model="gpt-4o-mini",
    )
    print(f"  Output 3: {'HIT' if cache.hits > prev_hits else 'MISS'}")


async def redis_backend_example() -> None:
    """Demonstrate Redis backend (if available)."""
    print("\n4. Redis Backend Example")
    print("-" * 60)

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        print("  Skipped - REDIS_URL environment variable not set")
        print("  To test Redis backend, set: export REDIS_URL=redis://localhost:6379")
        return

    try:
        # Create Redis-backed cache
        cache = CachingMiddleware(
            backend="redis",
            redis_url=redis_url,
            ttl=300,  # 5 minutes
            redis_key_prefix="arbiter:example:",
        )

        # Use as context manager for proper cleanup
        async with cache:
            pipeline = MiddlewarePipeline([cache])

            print(f"\nConnected to Redis at {redis_url}")

            # Run evaluations
            result = await evaluate(
                output="Redis-cached evaluation",
                reference="Test reference",
                evaluators=["semantic"],
                middleware=pipeline,
                model="gpt-4o-mini",
            )
            print(f"  First call (miss): Score={result.overall_score:.3f}")

            result = await evaluate(
                output="Redis-cached evaluation",
                reference="Test reference",
                evaluators=["semantic"],
                middleware=pipeline,
                model="gpt-4o-mini",
            )
            print(f"  Second call (hit): Score={result.overall_score:.3f}")

            stats = cache.get_stats()
            print(f"\n  Hits: {stats['hits']}, Misses: {stats['misses']}")
            print(f"  Redis caching works! Results persist across processes.")

    except ImportError:
        print("  Skipped - redis package not installed")
        print("  To test Redis backend, run: pip install redis")
    except Exception as e:
        print(f"  Error connecting to Redis: {e}")


async def main() -> None:
    """Run all caching examples."""
    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("Arbiter - Caching Middleware Examples")
    print("=" * 60)

    await memory_backend_example()
    await different_inputs_example()
    await lru_eviction_example()
    await redis_backend_example()

    print("\n" + "=" * 60)
    print("Done! Caching helps reduce API costs during development.")


if __name__ == "__main__":
    asyncio.run(main())
