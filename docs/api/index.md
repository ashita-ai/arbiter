# API Reference

Complete API documentation for Arbiter.

## Main API

- [`evaluate()`](evaluate.md) - Main evaluation function
- [`compare()`](compare.md) - Pairwise comparison function

## Evaluators

- [Evaluators Overview](evaluators.md)
- [`SemanticEvaluator`](evaluators/semantic.md) - Semantic similarity evaluation
- [`CustomCriteriaEvaluator`](evaluators/custom_criteria.md) - Domain-specific criteria
- [`PairwiseComparisonEvaluator`](evaluators/pairwise.md) - A/B testing and comparison

## Core Models

- [Models Overview](models.md)
- [`EvaluationResult`](models/evaluation_result.md) - Evaluation results
- [`ComparisonResult`](models/comparison_result.md) - Comparison results
- [`Score`](models/score.md) - Individual score
- [`LLMInteraction`](models/llm_interaction.md) - LLM call tracking

## Core Infrastructure

- [`LLMClient`](core/llm_client.md) - Provider-agnostic LLM client
- [`Registry`](core/registry.md) - Evaluator registry system
- [`Middleware`](core/middleware.md) - Middleware pipeline
- [`Retry`](core/retry.md) - Retry configuration

