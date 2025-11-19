.PHONY: help install dev-install test test-cov lint format type-check clean docs run-example

help:
	@echo "Available commands:"
	@echo "  install      - Install package (use 'uv sync' instead for development)"
	@echo "  dev-install  - Install package with dev dependencies (use 'uv sync' instead)"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and ruff"
	@echo "  type-check   - Run mypy type checking"
	@echo "  clean        - Remove build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  run-example  - Run an example (usage: make run-example EXAMPLE=basic_evaluation.py)"

install:
	uv sync

dev-install:
	uv sync --all-extras

test:
	uv run pytest tests/

test-cov:
	uv run pytest --cov=arbiter --cov-report=html --cov-report=term-missing tests/

lint:
	uv run ruff check arbiter tests
	uv run black --check arbiter tests

format:
	uv run black arbiter tests
	uv run ruff check --fix arbiter tests

type-check:
	uv run mypy arbiter

run-example:
	uv run python examples/$(EXAMPLE)

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	mkdocs build
