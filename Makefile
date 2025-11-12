.PHONY: help install dev-install test test-cov lint format type-check clean docs

help:
	@echo "Available commands:"
	@echo "  install      - Install package"
	@echo "  dev-install  - Install package with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and ruff"
	@echo "  type-check   - Run mypy type checking"
	@echo "  clean        - Remove build artifacts"
	@echo "  docs         - Build documentation"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/

test-cov:
	pytest --cov=arbiter --cov-report=html --cov-report=term-missing tests/

lint:
	ruff check arbiter tests
	black --check arbiter tests

format:
	black arbiter tests
	ruff check --fix arbiter tests

type-check:
	mypy arbiter

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	mkdocs build
