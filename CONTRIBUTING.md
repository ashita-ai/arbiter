# Contributing to Arbiter

Thank you for your interest in contributing to Arbiter!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arbiter
cd arbiter
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass:
```bash
make test
make lint
make type-check
```

3. Commit your changes:
```bash
git add .
git commit -m "feat: description of your changes"
```

4. Push and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public APIs
- Maintain test coverage above 80%
- Use `black` for code formatting
- Use `ruff` for linting
- Use `mypy` for type checking (strict mode)

## Testing

- Write unit tests for new features
- Write integration tests for complex workflows
- Run tests with: `make test`
- Check coverage with: `make test-cov`

## Commit Messages

Follow the Conventional Commits specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

## Questions?

Open an issue or reach out to the maintainers.
