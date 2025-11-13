# evaluate()

Main evaluation function for assessing LLM outputs.

```python
from arbiter import evaluate

result = await evaluate(
    output="Your LLM output",
    reference="Expected output",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)
```

## Parameters

- `output` (str): The LLM output to evaluate
- `reference` (Optional[str]): Reference text for comparison
- `criteria` (Optional[str]): Evaluation criteria (for custom_criteria evaluator)
- `evaluators` (Optional[List[str]]): List of evaluator names (default: ["semantic"])
- `llm_client` (Optional[LLMClient]): Pre-configured LLM client
- `model` (str): Model name (default: "gpt-4o")
- `provider` (Provider): Provider to use (default: Provider.OPENAI)
- `threshold` (float): Score threshold for pass/fail (default: 0.7)
- `middleware` (Optional[MiddlewarePipeline]): Middleware pipeline

## Returns

[`EvaluationResult`](models/evaluation_result.md) with scores, metrics, and complete interaction tracking.

## Examples

See [Examples](../examples/index.md) for more usage examples.

