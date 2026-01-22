<div align="center">
  <h1>Arbiter</h1>

  <p><strong>The only LLM evaluation framework that shows you exactly what your evaluations cost</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/ashita-ai/arbiter"><img src="https://img.shields.io/badge/version-0.2.0-blue" alt="Version"></a>
    <a href="https://ai.pydantic.dev"><img src="https://img.shields.io/badge/PydanticAI-native-purple" alt="PydanticAI"></a>
  </p>
</div>

---

## Why Arbiter?

Most evaluation frameworks tell you if your outputs are good. **Arbiter tells you that AND exactly what it cost.** Every evaluation automatically tracks tokens, latency, and real dollar costs across any provider.

```python
from arbiter_ai import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Cost: ${await result.total_llm_cost():.6f}")
print(f"Calls: {len(result.interactions)}")
```

**What makes Arbiter different:**

- **Automatic cost tracking** using LiteLLM's bundled pricing database
- **PydanticAI native** with type-safe structured outputs
- **Pure library** with no platform signup or server to run
- **Complete observability** with every LLM interaction tracked

## Installation

```bash
pip install arbiter-ai
```

**Optional features:**
```bash
pip install arbiter-ai[cli]        # Command-line interface
pip install arbiter-ai[scale]      # Fast FAISS semantic backend
pip install arbiter-ai[storage]    # PostgreSQL + Redis persistence
pip install arbiter-ai[verifiers]  # Web search fact verification
```

## Quick Start

Set your API key:
```bash
export OPENAI_API_KEY=sk-...
```

Run an evaluation:
```python
from arbiter_ai import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Cost: ${await result.total_llm_cost():.6f}")
```

## Evaluators

### Semantic Similarity

```python
result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)
```

### Custom Criteria (no reference needed)

```python
result = await evaluate(
    output="Medical advice about diabetes management",
    criteria="Medical accuracy, HIPAA compliance, appropriate tone",
    evaluators=["custom_criteria"],
    model="gpt-4o-mini"
)

print(f"Criteria met: {result.scores[0].metadata['criteria_met']}")
```

### Pairwise Comparison (A/B testing)

```python
from arbiter_ai import compare

comparison = await compare(
    output_a="GPT-4 response",
    output_b="Claude response",
    criteria="accuracy, clarity, completeness",
    model="gpt-4o-mini"
)

print(f"Winner: {comparison.winner}")  # output_a, output_b, or tie
print(f"Confidence: {comparison.confidence:.2f}")
```

### Factuality, Groundedness, Relevance

```python
# Hallucination detection
result = await evaluate(output=text, evaluators=["factuality"])

# RAG source attribution
result = await evaluate(output=rag_response, evaluators=["groundedness"])

# Query-output alignment
result = await evaluate(output=response, reference=query, evaluators=["relevance"])
```

### Multiple Evaluators

```python
result = await evaluate(
    output="Your LLM output",
    reference="Expected output",
    criteria="Accuracy, clarity, completeness",
    evaluators=["semantic", "custom_criteria", "factuality"],
    model="gpt-4o-mini"
)

for score in result.scores:
    print(f"{score.name}: {score.value:.2f}")
```

## Batch Evaluation

```python
from arbiter_ai import batch_evaluate

items = [
    {"output": "Paris is capital of France", "reference": "Paris is France's capital"},
    {"output": "Tokyo is capital of Japan", "reference": "Tokyo is Japan's capital"},
]

result = await batch_evaluate(
    items=items,
    evaluators=["semantic"],
    model="gpt-4o-mini",
    max_concurrency=5
)

print(f"Success: {result.successful_items}/{result.total_items}")
print(f"Total cost: ${await result.total_llm_cost():.4f}")
```

## Command-Line Interface

```bash
pip install arbiter-ai[cli]

# Single evaluation
arbiter evaluate --output "Paris is the capital" --reference "Paris is France's capital"

# Batch evaluation from file
arbiter batch --file inputs.jsonl --evaluators semantic --output results.json

# Compare two outputs
arbiter compare --output-a "Response A" --output-b "Response B"

# List evaluators
arbiter list-evaluators

# Check costs
arbiter cost --model gpt-4o-mini --input-tokens 1000 --output-tokens 500
```

## Provider Support

Arbiter works with any model via PydanticAI:
- OpenAI (GPT-4, GPT-4o, GPT-4o-mini)
- Anthropic (Claude)
- Google (Gemini)
- Groq
- Mistral
- Cohere

Set the appropriate API key as an environment variable.

## Examples

```bash
# Run examples
python examples/basic_evaluation.py
python examples/custom_criteria_example.py
python examples/pairwise_comparison_example.py
python examples/batch_evaluation_example.py
python examples/observability_example.py
```

## Development

```bash
git clone https://github.com/ashita-ai/arbiter.git
cd arbiter
uv sync --all-extras
make test
```

## License

MIT License
