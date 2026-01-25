"""Unit tests for environment variable configuration."""

import os
from unittest.mock import patch

import pytest

from arbiter_ai.core.config import (
    get_default_max_concurrency,
    get_default_model,
    get_default_threshold,
    get_default_timeout,
)
from arbiter_ai.core.exceptions import ConfigurationError


class TestGetDefaultModel:
    """Tests for get_default_model()."""

    def test_returns_fallback_when_not_set(self):
        """Test default fallback value when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if set
            os.environ.pop("ARBITER_DEFAULT_MODEL", None)
            assert get_default_model() == "gpt-4o-mini"

    def test_returns_env_value_when_set(self):
        """Test that env var value is returned."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_MODEL": "claude-3-5-sonnet"}):
            assert get_default_model() == "claude-3-5-sonnet"

    def test_returns_empty_string_if_set_empty(self):
        """Test that empty string is returned if set to empty."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_MODEL": ""}):
            assert get_default_model() == ""


class TestGetDefaultThreshold:
    """Tests for get_default_threshold()."""

    def test_returns_fallback_when_not_set(self):
        """Test default fallback value when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARBITER_DEFAULT_THRESHOLD", None)
            assert get_default_threshold() == 0.7

    def test_returns_env_value_when_set(self):
        """Test that env var value is returned as float."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "0.85"}):
            assert get_default_threshold() == 0.85

    def test_accepts_zero(self):
        """Test that 0.0 is accepted."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "0.0"}):
            assert get_default_threshold() == 0.0

    def test_accepts_one(self):
        """Test that 1.0 is accepted."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "1.0"}):
            assert get_default_threshold() == 1.0

    def test_raises_on_invalid_string(self):
        """Test that invalid string raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "invalid"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_threshold()
            assert "ARBITER_DEFAULT_THRESHOLD" in str(exc_info.value)
            assert "invalid" in str(exc_info.value)

    def test_raises_on_negative_value(self):
        """Test that negative value raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "-0.5"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_threshold()
            assert "between 0.0 and 1.0" in str(exc_info.value)

    def test_raises_on_value_over_one(self):
        """Test that value > 1.0 raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_DEFAULT_THRESHOLD": "1.5"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_threshold()
            assert "between 0.0 and 1.0" in str(exc_info.value)


class TestGetDefaultTimeout:
    """Tests for get_default_timeout()."""

    def test_returns_none_when_not_set(self):
        """Test that None is returned when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARBITER_TIMEOUT", None)
            assert get_default_timeout() is None

    def test_returns_env_value_when_set(self):
        """Test that env var value is returned as float."""
        with patch.dict(os.environ, {"ARBITER_TIMEOUT": "30"}):
            assert get_default_timeout() == 30.0

    def test_accepts_float_values(self):
        """Test that float values are accepted."""
        with patch.dict(os.environ, {"ARBITER_TIMEOUT": "30.5"}):
            assert get_default_timeout() == 30.5

    def test_raises_on_invalid_string(self):
        """Test that invalid string raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_TIMEOUT": "invalid"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_timeout()
            assert "ARBITER_TIMEOUT" in str(exc_info.value)

    def test_raises_on_zero(self):
        """Test that zero raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_TIMEOUT": "0"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_timeout()
            assert "must be positive" in str(exc_info.value)

    def test_raises_on_negative_value(self):
        """Test that negative value raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_TIMEOUT": "-10"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_timeout()
            assert "must be positive" in str(exc_info.value)


class TestGetDefaultMaxConcurrency:
    """Tests for get_default_max_concurrency()."""

    def test_returns_fallback_when_not_set(self):
        """Test default fallback value when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ARBITER_MAX_CONCURRENCY", None)
            assert get_default_max_concurrency() == 10

    def test_returns_env_value_when_set(self):
        """Test that env var value is returned as int."""
        with patch.dict(os.environ, {"ARBITER_MAX_CONCURRENCY": "20"}):
            assert get_default_max_concurrency() == 20

    def test_raises_on_invalid_string(self):
        """Test that invalid string raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_MAX_CONCURRENCY": "invalid"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_max_concurrency()
            assert "ARBITER_MAX_CONCURRENCY" in str(exc_info.value)

    def test_raises_on_float_value(self):
        """Test that float value raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_MAX_CONCURRENCY": "10.5"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_max_concurrency()
            assert "ARBITER_MAX_CONCURRENCY" in str(exc_info.value)

    def test_raises_on_zero(self):
        """Test that zero raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_MAX_CONCURRENCY": "0"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_max_concurrency()
            assert "must be positive" in str(exc_info.value)

    def test_raises_on_negative_value(self):
        """Test that negative value raises ConfigurationError."""
        with patch.dict(os.environ, {"ARBITER_MAX_CONCURRENCY": "-5"}):
            with pytest.raises(ConfigurationError) as exc_info:
                get_default_max_concurrency()
            assert "must be positive" in str(exc_info.value)


class TestConfigExports:
    """Tests for module exports."""

    def test_all_functions_exported_from_core(self):
        """Test that config functions are exported from core."""
        from arbiter_ai.core import (
            get_default_max_concurrency,
            get_default_model,
            get_default_threshold,
            get_default_timeout,
        )

        assert get_default_model is not None
        assert get_default_threshold is not None
        assert get_default_timeout is not None
        assert get_default_max_concurrency is not None

    def test_all_functions_exported_from_package(self):
        """Test that config functions are exported from main package."""
        from arbiter_ai import (
            get_default_max_concurrency,
            get_default_model,
            get_default_threshold,
            get_default_timeout,
        )

        assert get_default_model is not None
        assert get_default_threshold is not None
        assert get_default_timeout is not None
        assert get_default_max_concurrency is not None
