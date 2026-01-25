# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Filtering methods for `BatchEvaluationResult`: `filter()`, `slice()`, `get_failed_items()`, `get_results_with_indices()`
- InstructionFollowingEvaluator for agent/pipeline validation
- Environment variable configuration support (ARBITER_DEFAULT_MODEL, ARBITER_DEFAULT_THRESHOLD, etc.)
- Custom exception hierarchy with specific error types (RateLimitError, AuthenticationError, etc.)
- Query performance monitoring for storage backends
- EditorConfig for consistent code formatting

### Changed
- Middleware module split into directory structure for better organization

### Fixed
- Database credentials now sanitized in error messages

## [0.2.0] - 2025-12-20

### Added
- CLI tool for command-line evaluations (`arbiter eval`, `arbiter compare`, `arbiter batch`)
- JSON/CSV export methods for `BatchEvaluationResult` (`to_json()`, `to_csv()`, `to_dataframe()`)
- LangChain integration example
- Model class usability improvements: `__repr__`, `summary()`, `to_dict()` methods
- Structured logging integration with logfire
- Timeout parameter for individual evaluations
- Evaluation caching middleware with pluggable backends (memory, Redis)
- Cost estimation and dry-run mode (`evaluate(..., dry_run=True)`)
- FastAPI integration example
- Retry logic with exponential backoff, max delay, and jitter
- Input validation with helpful error messages
- Pre-commit hooks for automated linting and formatting
- Score comparison methods (`is_better_than()`, `is_worse_than()`, etc.)
- LiteLLM bundled pricing database for cost calculation
- Automated coverage reporting with Codecov
- Batch statistics: `average_score()`, `pass_rate()`, `score_distribution()`
- Weighted scoring option for `evaluate()`
- Result merging: `BatchEvaluationResult.merge()`
- Progress callback for `batch_evaluate()`
- Per-evaluator threshold support

### Changed
- Switched pricing from llm-prices.com to LiteLLM bundled database
- All providers now route through PydanticAI for consistency
- Performance optimizations for batch processing

### Fixed
- Model ID normalization for pricing lookups
- Test warnings for unawaited coroutines and pytest collection

## [0.1.2] - 2025-11-27

### Added
- Parallel evaluator execution for improved performance
- External verification plugins support (SearchVerifier, CitationVerifier, KnowledgeBaseVerifier)
- Comprehensive CONTRIBUTING.md and issue templates

### Changed
- Strict mypy compliance for improved type checking

## [0.1.1] - 2025-11-25

### Added
- Initial public release
- Core evaluators: SemanticEvaluator, FactualityEvaluator, CustomCriteriaEvaluator, PairwiseComparisonEvaluator, GroundednessEvaluator, RelevanceEvaluator
- Provider-agnostic LLM support (OpenAI, Anthropic, Google, Groq, Mistral, Cohere)
- Batch evaluation with `batch_evaluate()`
- Complete LLM interaction tracking for observability
- Middleware pipeline for cross-cutting concerns
- Storage backends: PostgreSQL, Redis
- Cost tracking and calculation
- Circuit breaker pattern for fault tolerance
- Connection pooling for LLM clients
- Registry system for custom evaluators
- 95% test coverage

[Unreleased]: https://github.com/ashita-ai/arbiter/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ashita-ai/arbiter/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/ashita-ai/arbiter/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/ashita-ai/arbiter/releases/tag/v0.1.1
