# Arbiter

**Streaming LLM evaluation framework with semantic comparison and composable metrics**

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/yourusername/arbiter)

## What is Arbiter?

Arbiter provides composable evaluation primitives for assessing LLM outputs through research-backed metrics. Instead of simple string matching, Arbiter uses semantic comparison, factuality checking, and consistency verification to provide meaningful quality scores.

**Core Value**: Evaluate LLM outputs at scale with semantic understanding and complete observability.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arbiter
cd arbiter

# Install with uv (recommended)
uv pip install -e .

# Or with standard pip
pip install -e .
```

## Quick Start

```python
from arbiter import evaluate, SemanticEvaluator

# Simple evaluation
evaluator = SemanticEvaluator(model="gpt-4o")

score = await evaluator.score(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    criteria="factuality"
)

print(f"Score: {score.value}")  # High score for semantic match
```

## Key Features

- **🎯 Semantic Comparison**: Milvus-backed vector similarity for deep comparison
- **📊 Multiple Metrics**: Factuality, consistency, semantic similarity, custom criteria
- **🔌 Composable**: Build custom evaluation pipelines with middleware
- **⚡ Streaming Ready**: Optional ByteWax integration for streaming data
- **👁️ Complete Observability**: Full audit trails and performance metrics
- **🧩 Extensible**: Plugin system for custom evaluators and storage

## Core Concepts

### Evaluators

Evaluators assess LLM outputs against criteria:

```python
# Semantic similarity
semantic_score = await evaluator.score(output, reference, "semantic_similarity")

# Factuality checking
factuality_score = await evaluator.score(output, reference, "factuality")

# Custom criteria
custom_score = await evaluator.score(output, reference, "matches company tone")
```

### Batch Evaluation

Process multiple outputs efficiently:

```python
outputs = ["Output 1", "Output 2", "Output 3"]
references = ["Reference 1", "Reference 2", "Reference 3"]

scores = await evaluator.batch_score(outputs, references)
```

### Streaming (Optional)

Integrate with streaming pipelines:

```python
from arbiter.streaming import ByteWaxAdapter

async for batch in kafka_source:
    results = await evaluator.batch_score(batch)
    await sink.send(results)
```

## Architecture

Built on proven patterns from Sifaka with focus on evaluation:

- **PydanticAI**: Structured LLM interactions
- **Milvus**: Vector storage for semantic comparison
- **Middleware**: Logging, metrics, caching, rate limiting
- **Plugin System**: Extensible evaluators and storage backends

## Development

```bash
git clone https://github.com/yourusername/arbiter
cd arbiter
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Project setup and structure
- [ ] Core evaluation engine
- [ ] PydanticAI evaluator implementation
- [ ] Milvus integration for semantic comparison
- [ ] Middleware system (from Sifaka)
- [ ] Storage backends
- [ ] Batch operations
- [ ] ByteWax streaming adapter
- [ ] Example evaluators (factuality, consistency)
- [ ] Documentation and examples

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with inspiration from [Sifaka](https://sifaka.ai) and leveraging patterns for production-grade AI systems.
