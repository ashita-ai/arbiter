# Contributing to Arbiter

Thank you for your interest in contributing to Arbiter!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ashita-ai/arbiter.git
cd arbiter

# Install with development dependencies
uv sync --all-extras

# Or with pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Development Workflow

### Before Starting

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the project conventions

3. Run quality checks:
   ```bash
   make all  # Runs format, lint, type-check, and tests
   ```

### Running Tests

```bash
make test          # Run all tests with coverage
make test-cov      # Generate detailed coverage report
pytest tests/unit/ # Run unit tests only
pytest -v          # Run all tests with verbose output
```

### Code Quality

```bash
make format        # Format code with black
make lint          # Check code with ruff
make type-check    # Type check with mypy
```

## Release Process

Arbiter uses semantic versioning (MAJOR.MINOR.PATCH) with alpha/beta pre-releases.

### Version Strategy

- `0.1.0a0` - Alpha releases (current)
- `0.1.0b0` - Beta releases
- `0.1.0` - Stable releases
- `0.1.1` - Patch releases (bug fixes)
- `0.2.0` - Minor releases (new features, backward compatible)
- `1.0.0` - Major releases (breaking changes)

### Publishing a Release

**Prerequisites:**
1. Maintainer access to the GitHub repository
2. PyPI account configured with trusted publishing (see Setup below)

**Steps:**

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.0"  # Remove 'a0' for stable release
   ```

2. **Run quality checks**:
   ```bash
   make all  # Ensure all tests pass
   ```

3. **Commit and push**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.0"
   git push origin main
   ```

4. **Create and push tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. **Create GitHub release**:
   - Go to https://github.com/ashita-ai/arbiter/releases/new
   - Select tag: `v0.1.0`
   - Release title: `v0.1.0`
   - Add release notes describing changes
   - Click "Publish release"

6. **Automated publishing**:
   - GitHub Actions will automatically build and publish to PyPI
   - Monitor progress at https://github.com/ashita-ai/arbiter/actions
   - Package will be available at https://pypi.org/project/arbiter-ai/

### Setting Up Trusted Publishing (Maintainers Only)

PyPI trusted publishing eliminates the need for API tokens:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **Owner**: `evanvolgas`
   - **Repository**: `arbiter`
   - **Workflow**: `publish-to-pypi.yml`
   - **Environment**: Leave blank
3. Save the configuration

### Manual Publishing (Emergency Only)

If automated publishing fails:

```bash
# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

## Testing a Release

### Test on TestPyPI (Recommended Before Production)

1. **Build package**:
   ```bash
   python -m build
   ```

2. **Upload to TestPyPI**:
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ arbiter
   ```

### Verify Installation

```bash
# Create clean environment
python -m venv test-env
source test-env/bin/activate

# Install from PyPI
pip install arbiter-ai

# Test import
python -c "from arbiter import evaluate; print('Success!')"

# Run example
python examples/basic_evaluation.py
```

## Pull Request Guidelines

1. **One feature per PR** - Keep changes focused and reviewable
2. **Tests required** - Add tests for new features (maintain >80% coverage)
3. **Documentation** - Update docstrings and examples
4. **Quality checks** - Run `make all` before submitting
5. **Descriptive commits** - Use clear commit messages

## Project Structure

```
arbiter/
├── arbiter/            # Main package
│   ├── api.py         # Public API (evaluate, compare)
│   ├── core/          # Infrastructure
│   ├── evaluators/    # Evaluator implementations
│   ├── storage/       # Storage backends
│   └── tools/         # Utilities
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── examples/          # Usage examples
├── pyproject.toml     # Project metadata
└── README.md          # User documentation
```

## Questions?

- Open an issue: https://github.com/ashita-ai/arbiter/issues
- Discussion: https://github.com/ashita-ai/arbiter/discussions
