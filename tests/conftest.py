"""
PyTest Configuration and Fixtures
=================================

Shared fixtures and configuration for CuliFeed tests.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before any imports
os.environ['CULIFEED_TELEGRAM__BOT_TOKEN'] = '123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11_test'
os.environ['CULIFEED_AI__GEMINI_API_KEY'] = 'test-gemini-key-for-foundation-testing'
os.environ['CULIFEED_AI__GROQ_API_KEY'] = 'test-groq-key-for-foundation-testing'
os.environ['CULIFEED_DEBUG'] = 'true'


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def test_database():
    """Create a test database with schema."""
    from culifeed.database.schema import DatabaseSchema
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create schema
    schema = DatabaseSchema(db_path)
    schema.create_tables()
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def db_connection(test_database):
    """Create a database connection manager for testing."""
    from culifeed.database.connection import DatabaseConnection
    
    connection = DatabaseConnection(test_database, pool_size=2)
    yield connection
    
    # Cleanup connections
    connection.close_all()


@pytest.fixture
def sample_articles():
    """Generate sample articles for testing."""
    from culifeed.database.models import Article
    
    return [
        Article(
            title="Python Programming Tutorial",
            url="https://example.com/python-tutorial",
            content="Learn Python programming from scratch.",
            source_feed="https://example.com/feed.xml"
        ),
        Article(
            title="JavaScript Framework Guide",
            url="https://example.com/js-guide",
            content="Comprehensive guide to JavaScript frameworks.",
            source_feed="https://example.com/feed.xml"
        ),
        Article(
            title="Data Science with Python",
            url="https://example.com/data-science",
            content="Introduction to data science using Python libraries.",
            source_feed="https://example.com/feed.xml"
        )
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
            confidence_threshold=0.8
        ),
        Topic(
            chat_id="-1001234567890",
            name="Data Science",
            keywords=["data science", "machine learning", "analytics"],
            exclude_keywords=[],
            confidence_threshold=0.7
        )
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
            description="Latest technology news and updates"
        ),
        Feed(
            chat_id="-1001234567890",
            url="https://example.com/programming.xml",
            title="Programming Feed",
            description="Programming tutorials and tips"
        )
    ]


@pytest.fixture
def sample_channels():
    """Generate sample channels for testing."""
    from culifeed.database.models import Channel, ChatType
    
    return [
        Channel(
            chat_id="-1001234567890",
            chat_title="Tech Discussion Group",
            chat_type=ChatType.SUPERGROUP
        ),
        Channel(
            chat_id="-1001111222333",
            chat_title="Programming Channel",
            chat_type=ChatType.CHANNEL
        )
    ]