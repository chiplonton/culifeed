"""
PyTest Configuration and Fixtures
=================================

Shared fixtures and configuration for CuliFeed tests.

Performance Optimizations:
- Session-scoped database fixtures for 4x speedup
- In-memory database option for unit tests
- Named test database for easier debugging
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before any imports
os.environ["CULIFEED_TELEGRAM__BOT_TOKEN"] = (
    "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11_test"
)
os.environ["CULIFEED_AI__GEMINI_API_KEY"] = "test-gemini-key-for-foundation-testing"
os.environ["CULIFEED_AI__GROQ_API_KEY"] = "test-groq-key-for-foundation-testing"
os.environ["CULIFEED_DEBUG"] = "true"


# ============================================================================
# Database Fixtures (Optimized for Performance)
# ============================================================================


@pytest.fixture(scope="session")
def session_test_db():
    """Session-scoped test database (created once for all tests).

    Provides significant performance improvement by creating the database
    schema only once instead of per-test. Safe because tests use clean_db
    fixture to clear data between tests.

    Database name: culifeed_test.db (easier to inspect/debug)
    """
    from culifeed.database.schema import DatabaseSchema

    # Use named database in temp directory for easier debugging
    test_dir = Path(tempfile.gettempdir()) / "culifeed_tests"
    test_dir.mkdir(exist_ok=True)
    db_path = test_dir / "culifeed_test.db"

    # Remove existing test database
    if db_path.exists():
        db_path.unlink()

    # Create schema once
    schema = DatabaseSchema(str(db_path))
    schema.create_tables()

    yield str(db_path)

    # Cleanup after all tests complete
    try:
        db_path.unlink()
    except FileNotFoundError:
        pass


@pytest.fixture(scope="session")
def memory_db():
    """In-memory database for fast unit tests.

    Use this fixture in unit tests that don't need persistent storage.
    Provides 3-5x speedup over file-based database.

    Usage:
        def test_fast_operation(memory_db):
            conn = DatabaseConnection(memory_db)
            # Test runs much faster
    """
    from culifeed.database.schema import DatabaseSchema

    db_path = ":memory:"
    schema = DatabaseSchema(db_path)
    schema.create_tables()

    yield db_path
    # No cleanup needed for in-memory database


@pytest.fixture
def clean_db(session_test_db):
    """Clean database fixture (clears data between tests).

    Provides test isolation by clearing all data while keeping schema.
    Use this instead of test_database for better performance.

    Returns:
        str: Path to clean database ready for testing
    """
    from culifeed.database.connection import DatabaseConnection

    conn = DatabaseConnection(session_test_db, pool_size=2)

    with conn.get_connection() as db:
        # Clear all data but keep schema (order matters for foreign keys)
        db.execute("DELETE FROM processing_results")
        db.execute("DELETE FROM articles")
        db.execute("DELETE FROM topics")
        db.execute("DELETE FROM feeds")
        db.execute("DELETE FROM user_subscriptions")
        db.execute("DELETE FROM channels")
        db.commit()

    conn.close_all_connections()

    yield session_test_db


@pytest.fixture
def temp_db():
    """Create a temporary database for testing (legacy fixture).

    Deprecated: Use clean_db or memory_db instead for better performance.
    Only use this if test explicitly needs isolated temp file.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def test_database(clean_db):
    """Alias for clean_db to maintain backward compatibility.

    Note: Now uses session-scoped database with cleanup for performance.
    """
    return clean_db


@pytest.fixture
def db_connection(test_database):
    """Create a database connection manager for testing."""
    from culifeed.database.connection import DatabaseConnection

    connection = DatabaseConnection(test_database, pool_size=2)
    yield connection

    # Cleanup connections
    connection.close_all_connections()


@pytest.fixture
def sample_articles():
    """Generate sample articles for testing."""
    from culifeed.database.models import Article

    return [
        Article(
            title="Python Programming Tutorial",
            url="https://example.com/python-tutorial",
            content="Learn Python programming from scratch.",
            source_feed="https://example.com/feed.xml",
        ),
        Article(
            title="JavaScript Framework Guide",
            url="https://example.com/js-guide",
            content="Comprehensive guide to JavaScript frameworks.",
            source_feed="https://example.com/feed.xml",
        ),
        Article(
            title="Data Science with Python",
            url="https://example.com/data-science",
            content="Introduction to data science using Python libraries.",
            source_feed="https://example.com/feed.xml",
        ),
    ]


@pytest.fixture
def sample_topics():
    """Generate sample topics for testing."""
    from culifeed.database.models import Topic

    return [
        Topic(
            chat_id="-1001234567890",
            name="Programming",
            keywords=["python", "javascript", "programming", "coding"],
            exclude_keywords=["beginner"],
            confidence_threshold=0.8,
        ),
        Topic(
            chat_id="-1001234567890",
            name="Data Science",
            keywords=["data science", "machine learning", "analytics"],
            exclude_keywords=[],
            confidence_threshold=0.7,
        ),
    ]


@pytest.fixture
def sample_feeds():
    """Generate sample feeds for testing."""
    from culifeed.database.models import Feed

    return [
        Feed(
            chat_id="-1001234567890",
            url="https://example.com/feed.xml",
            title="Tech News Feed",
            description="Latest technology news and updates",
        ),
        Feed(
            chat_id="-1001234567890",
            url="https://example.com/programming.xml",
            title="Programming Feed",
            description="Programming tutorials and tips",
        ),
    ]


@pytest.fixture
def sample_channels():
    """Generate sample channels for testing."""
    from culifeed.database.models import Channel, ChatType

    return [
        Channel(
            chat_id="-1001234567890",
            chat_title="Tech Discussion Group",
            chat_type=ChatType.SUPERGROUP,
        ),
        Channel(
            chat_id="-1001111222333",
            chat_title="Programming Channel",
            chat_type=ChatType.CHANNEL,
        ),
    ]
