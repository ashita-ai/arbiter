"""Tests for storage backend base classes."""

import pytest

from arbiter_ai.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
    StorageError,
    sanitize_url,
)


def test_storage_exceptions_inheritance():
    """Test that storage exceptions inherit correctly."""
    assert issubclass(ConnectionError, StorageError)
    assert issubclass(SaveError, StorageError)
    assert issubclass(RetrievalError, StorageError)
    assert issubclass(StorageError, Exception)


def test_storage_backend_is_abstract():
    """Test that StorageBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StorageBackend()  # type: ignore


@pytest.mark.asyncio
async def test_storage_backend_context_manager():
    """Test that concrete implementations support async context manager."""

    class TestStorage(StorageBackend):
        def __init__(self):
            self.connected = False
            self.closed = False

        async def connect(self):
            self.connected = True

        async def close(self):
            self.closed = True

        async def save_result(self, result, metadata=None):
            return "test_id"

        async def save_batch_result(self, result, metadata=None):
            return "test_batch_id"

        async def get_result(self, result_id):
            return None

        async def get_batch_result(self, batch_id):
            return None

    storage = TestStorage()
    assert not storage.connected
    assert not storage.closed

    async with storage:
        assert storage.connected
        assert not storage.closed

    assert storage.connected
    assert storage.closed


class TestSanitizeUrl:
    """Test suite for sanitize_url function."""

    def test_sanitize_postgresql_url(self):
        """Test sanitizing PostgreSQL connection URL."""
        url = "postgresql://myuser:mysecretpassword@localhost:5432/mydb"
        sanitized = sanitize_url(url)
        assert sanitized == "postgresql://***:***@localhost:5432/mydb"
        assert "myuser" not in sanitized
        assert "mysecretpassword" not in sanitized

    def test_sanitize_redis_url_with_password_only(self):
        """Test sanitizing Redis URL with password only (no username)."""
        url = "redis://:secretpass@redis.example.com:6379/0"
        sanitized = sanitize_url(url)
        assert sanitized == "redis://***:***@redis.example.com:6379/0"
        assert "secretpass" not in sanitized

    def test_sanitize_url_with_special_characters(self):
        """Test sanitizing URL with special characters in password."""
        url = "postgresql://user:p@ss!word#123@localhost/db"
        sanitized = sanitize_url(url)
        # Should sanitize up to the @ before the host
        assert "p@ss!word#123" not in sanitized
        assert "localhost/db" in sanitized

    def test_sanitize_url_preserves_host_and_path(self):
        """Test that host, port, and path are preserved."""
        url = "postgresql://admin:secret@db.example.com:5432/production"
        sanitized = sanitize_url(url)
        assert "db.example.com:5432/production" in sanitized
        assert sanitized.startswith("postgresql://")

    def test_sanitize_url_without_credentials(self):
        """Test URL without credentials is unchanged."""
        url = "postgresql://localhost:5432/mydb"
        sanitized = sanitize_url(url)
        assert sanitized == url

    def test_sanitize_url_with_query_params(self):
        """Test URL with query parameters."""
        url = "postgresql://user:pass@localhost/db?sslmode=require"
        sanitized = sanitize_url(url)
        assert sanitized == "postgresql://***:***@localhost/db?sslmode=require"
        assert "user" not in sanitized
        assert "pass" not in sanitized.replace("***:***", "")

    def test_sanitize_mysql_url(self):
        """Test sanitizing MySQL connection URL."""
        url = "mysql://root:rootpass@mysql.local:3306/appdb"
        sanitized = sanitize_url(url)
        assert sanitized == "mysql://***:***@mysql.local:3306/appdb"

    def test_sanitize_empty_password(self):
        """Test URL with empty password."""
        url = "postgresql://user:@localhost/db"
        sanitized = sanitize_url(url)
        assert sanitized == "postgresql://***:***@localhost/db"
