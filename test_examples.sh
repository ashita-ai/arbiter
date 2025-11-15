#!/bin/bash

source .venv/bin/activate

examples=(
    "examples/basic_evaluation.py"
    "examples/custom_criteria_example.py"
    "examples/pairwise_comparison_example.py"
    "examples/multiple_evaluators.py"
    "examples/middleware_usage.py"
    "examples/error_handling_example.py"
    "examples/interaction_tracking_example.py"
    "examples/evaluator_registry_example.py"
    "examples/provider_switching.py"
    "examples/circuit_breaker_example.py"
    "examples/pairwise_with_middleware.py"
    "examples/advanced_config.py"
    "examples/batch_manual.py"
    "examples/rag_evaluation.py"
)

passed=0
failed=0
failed_examples=()

for example in "${examples[@]}"; do
    echo "Testing $example..."
    if python "$example" > /dev/null 2>&1; then
        echo "✓ PASSED: $example"
        ((passed++))
    else
        echo "✗ FAILED: $example"
        ((failed++))
        failed_examples+=("$example")
    fi
done

echo ""
echo "=========================="
echo "Results: $passed passed, $failed failed"
echo "=========================="

if [ $failed -gt 0 ]; then
    echo "Failed examples:"
    for example in "${failed_examples[@]}"; do
        echo "  - $example"
    done
    exit 1
fi
