---
name: arbiter-agent
description: Production-grade LLM evaluation framework developer
last_updated: 2025-01-22
---

# Arbiter Agent Guide

**Purpose**: Quick reference for working on Arbiter

**Arbiter**: Production-grade LLM evaluation framework (v0.1.0-alpha)
**Stack**: Python 3.10+, PydanticAI, provider-agnostic (OpenAI/Anthropic/Google/Groq)
**Coverage**: 95% test coverage, strict mypy, comprehensive examples

---

## Boundaries

### Always Do (No Permission Needed)

- Run tests: `make test`, `pytest tests/`, `pytest -v`
- Format code: `make format` (runs black)
- Lint code: `make lint` (runs ruff)
- Type check: `make type-check` (runs mypy in strict mode)
- Add unit tests for new evaluators in `tests/unit/`
- Update docstrings when changing function signatures
- Add examples to `examples/` for new user-facing features
- Export new evaluators in `__init__.py` files
- Run `make all` before committing (format + lint + type-check + test)

### Ask First

- Add new evaluators to `arbiter/evaluators/`
- Modify core API in `arbiter/api.py` (evaluate, compare functions)
- Change template method pattern in `BasePydanticEvaluator`
- Add/update dependencies in `pyproject.toml`
- Change public API examples in `README.md`
- Modify middleware pipeline in `arbiter/core/middleware.py`
- Change LLM client abstraction in `arbiter/core/llm_client.py`
- Add new storage backends in `arbiter/storage/`
- Modify interaction tracking in `arbiter/core/monitoring.py`

### Never Touch

**Security (CRITICAL)**:
- NEVER EVER COMMIT CREDENTIALS TO GITHUB
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is non-negotiable with serious security consequences

**Other Prohibitions**:
- `.env` files or API keys (use environment variables)
- Production deployment configurations
- Git history manipulation (no force push, interactive rebase on shared branches)
- User's `~/.claude/` configuration files
- Any files outside the `arbiter/` repository
- Test files to make them pass (fix the code, not the tests)
- Type checking configuration to reduce strictness
- Coverage thresholds (must maintain >80%)

---

## Communication Preferences

Don't flatter me. I know what [AI sycophancy](https://www.seangoedecke.com/ai-sycophancy/) is and I don't want your praise. Be concise and direct. Don't use emdashes ever.

---

## Directory Structure

```
arbiter/
├── arbiter/
│   ├── api.py              # Public API (evaluate, compare)
│   ├── core/               # Infrastructure (llm_client, middleware, monitoring, registry)
│   ├── evaluators/         # Semantic, CustomCriteria, Pairwise, Factuality, Groundedness, Relevance
│   ├── storage/            # Storage backends (Phase 4)
│   └── tools/              # Utilities
├── examples/               # 15+ comprehensive examples
├── tests/                  # Unit + integration tests
└── pyproject.toml          # Dependencies and config
```

---

## Critical Rules

### 1. Template Method Pattern

All evaluators extend `BasePydanticEvaluator` and implement 4 methods:

```python
class MyEvaluator(BasePydanticEvaluator):
    @property
    def name(self) -> str:
        return "my_evaluator"

    def _get_system_prompt(self) -> str:
        return "You are an expert evaluator..."

    def _get_user_prompt(self, output: str, reference: Optional[str], criteria: Optional[str]) -> str:
        return f"Evaluate '{output}' against: {criteria}"

    def _get_response_type(self) -> Type[BaseModel]:
        return MyEvaluatorResponse  # Pydantic model

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = cast(MyEvaluatorResponse, response)
        return Score(name=self.name, value=resp.score, confidence=resp.confidence, explanation=resp.explanation)
```

### 2. Provider-Agnostic Design

Must work with ANY LLM provider (OpenAI, Anthropic, Google, Groq, Mistral, Cohere).

```python
# GOOD
client = await LLMManager.get_client(provider="anthropic", model="claude-3-5-sonnet")

# BAD
from openai import OpenAI
client = OpenAI()  # Hardcoded to OpenAI
```

### 3. Type Safety (Strict Mypy)

All functions require type hints, no `Any` without justification.

### 4. No Placeholders/TODOs

Production-grade code only. Complete implementations or nothing.

### 5. Complete Features Only

If you start, you finish:
- Implementation complete
- Tests (>80% coverage)
- Docstrings
- Example code
- Exported in `__init__.py`

### 6. PydanticAI for Structured Outputs

All evaluators use PydanticAI for type-safe LLM responses.

---

## Development Workflow

### Before Starting

1. Check `git status` and `git branch`
2. Create feature branch: `git checkout -b feature/my-feature`

### During Development

1. Follow template method pattern
2. Write tests as you code (not after)
3. Run `make test` frequently

### Before Committing

```bash
make test        # Tests pass
make type-check  # Mypy clean
make lint        # Ruff clean
make format      # Black formatted
```

### After Completing

1. Add example to `examples/` if user-facing
2. Update README.md if API changed

---

## Common Tasks

### Add New Evaluator

```bash
# 1. Create evaluator file
touch arbiter/evaluators/my_evaluator.py

# 2. Implement template methods (see Critical Rules #1)

# 3. Export in arbiter/evaluators/__init__.py
from .my_evaluator import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 4. Export in arbiter/__init__.py
from .evaluators import MyEvaluator
__all__ = [..., "MyEvaluator"]

# 5. Write tests
touch tests/unit/test_my_evaluator.py

# 6. Add example
touch examples/my_evaluator_example.py
```

### Run Tests

```bash
make test              # Run all tests with coverage (requires >80% coverage to pass)
pytest tests/unit/     # Run unit tests only (fast, mocked dependencies)
pytest -v              # Run all tests with verbose output (shows test names and results)
make test-cov          # Generate detailed coverage report with missing lines
pytest tests/unit/test_semantic.py -v  # Run specific test file with verbose output
pytest -k "test_evaluate" -v  # Run tests matching pattern "test_evaluate"
```

---

## Code Quality Standards

### Docstrings

```python
async def evaluate(output: str, reference: Optional[str] = None) -> EvaluationResult:
    """Evaluate LLM output against reference or criteria.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison

    Returns:
        EvaluationResult with scores, metrics, and interactions

    Raises:
        ValidationError: If output is empty
        EvaluatorError: If evaluation fails

    Example:
        >>> result = await evaluate(output="Paris", reference="Paris is the capital of France")
        >>> print(result.overall_score)
        0.92
    """
```

### Formatting

- **black**: Line length 88
- **ruff**: Follow pyproject.toml config
- **mypy**: Strict mode (all functions typed)

---

## Quick Reference

### Key Files

- **evaluators/semantic.py**: Reference evaluator implementation
- **pyproject.toml**: Dependencies and config
- **README.md**: User documentation with examples
- **examples/**: 15+ comprehensive examples

### Key Patterns

- **Evaluators**: Template method pattern (4 required methods)
- **Middleware**: Pre/post processing pipeline
- **LLM Client**: Provider-agnostic abstraction
- **Interaction Tracking**: Automatic LLM call logging

### Make Targets

```bash
make test          # Run pytest with coverage (requires >80%, shows missing lines)
make type-check    # Run mypy in strict mode (all functions must have type hints)
make lint          # Run ruff linter (checks code style and potential bugs)
make format        # Run black formatter (line length 88, modifies files in place)
make all           # Run all checks in order: format → lint → type-check → test
```

---

## Questions?

Check evaluators/semantic.py (reference implementation) or README.md (user docs)
