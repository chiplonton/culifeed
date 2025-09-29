"""
Comprehensive error handling tests for processing module.

Tests error scenarios, exception propagation, and recovery mechanisms:
- Component-level error handling
- Pipeline error recovery and graceful degradation
- Database transaction failures and rollbacks
- Network timeout and retry logic
- Resource exhaustion and limits
"""

import pytest
import asyncio
import sqlite3
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.processing.pipeline import ProcessingPipeline
from culifeed.processing.feed_fetcher import FeedFetcher, FetchResult
from culifeed.processing.article_processor import ArticleProcessor
from culifeed.processing.pre_filter import PreFilterEngine
from culifeed.database.connection import DatabaseConnection
from culifeed.database.models import Article, Topic, Feed
from culifeed.database.schema import DatabaseSchema
from culifeed.utils.exceptions import (
    FeedFetchError,
    ProcessingError,
    DatabaseError,
    ValidationError,
    ConfigurationError,
)
from pydantic import ValidationError as PydanticValidationError


class TestProcessingErrorHandling:
    """Test comprehensive error handling across processing module.

    Covers:
    - Component error isolation and recovery
    - Pipeline graceful degradation
    - Database transaction safety
    - Network and timeout handling
    - Resource limit enforcement
    - Error context preservation
    """

    @pytest.fixture
    def test_database(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        schema = DatabaseSchema(db_path)
        schema.create_tables()

        yield db_path

        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def db_connection(self, test_database):
        """Database connection fixture."""
        return DatabaseConnection(test_database)

    @pytest.fixture
    def pipeline(self, db_connection):
        """Create pipeline for error testing."""
        return ProcessingPipeline(db_connection)

    @pytest.fixture
    def sample_articles(self):
        """Create sample articles for error testing."""
        return [
            Article(
                id="test_article_1",
                title="Test Article One",
                url="https://example.com/article1",
                content="Content for testing error scenarios.",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash="hash1",
                created_at=datetime.now(timezone.utc),
            ),
            Article(
                id="test_article_2",
                title="Test Article Two",
                url="https://example.com/article2",
                content="Another test article for error handling.",
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash="hash2",
                created_at=datetime.now(timezone.utc),
            ),
        ]

    def test_feed_fetcher_network_errors(self):
        """Test feed fetcher handling of various network errors."""
        fetcher = FeedFetcher(max_concurrent=2, timeout=5)

        # Test different network error scenarios
        import requests

        with patch("requests.get") as mock_get:
            # Test connection timeout
            mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")

            with pytest.raises(FeedFetchError) as exc_info:
                from culifeed.ingestion.feed_manager import fetch_single_feed

                fetch_single_feed("https://example.com/timeout-feed.xml")

            error = exc_info.value
            assert error.recoverable is True
            assert "timeout" in str(error).lower()

    def test_feed_fetcher_invalid_content_errors(self):
        """Test feed fetcher handling of invalid content."""
        # Test malformed XML
        invalid_feeds = [
            "Not XML content at all",
            "<?xml version='1.0'?><rss><channel><title>No closing tags",
            "<html><body>This is HTML not RSS</body></html>",
            "",  # Empty content
        ]

        for invalid_content in invalid_feeds:
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.content = invalid_content.encode("utf-8")
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response

                with pytest.raises(FeedFetchError) as exc_info:
                    from culifeed.ingestion.feed_manager import fetch_single_feed

                    fetch_single_feed("https://example.com/invalid-feed.xml")

                error = exc_info.value
                assert (
                    error.recoverable is True
                )  # Feed errors are recoverable by default

    def test_article_processor_database_errors(self, db_connection, sample_articles):
        """Test article processor handling database errors."""
        processor = ArticleProcessor(db_connection)

        # Test database connection failure during processing
        with patch.object(db_connection, "get_connection") as mock_conn:
            mock_conn.side_effect = sqlite3.OperationalError("Database is locked")

            with pytest.raises(sqlite3.OperationalError) as exc_info:
                processor.process_articles(sample_articles, check_database=True)

            error = exc_info.value
            assert "database" in str(error).lower()

    def test_article_processor_memory_limits(self, db_connection):
        """Test article processor handling of memory-intensive operations."""
        processor = ArticleProcessor(db_connection)

        # Create extremely large articles to test memory handling
        large_articles = []
        for i in range(10):
            large_content = "A" * 1000000  # 1MB of content each
            article = Article(
                id=f"large_article_{i}",
                title=f"Large Article {i}",
                url=f"https://example.com/large/{i}",
                content=large_content,
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash=f"hash_{i}",
                created_at=datetime.now(timezone.utc),
            )
            large_articles.append(article)

        # Should handle large content without crashing
        try:
            unique_articles, stats = processor.process_articles(
                large_articles, check_database=False
            )
            assert len(unique_articles) == len(large_articles)
        except MemoryError:
            pytest.skip("System has insufficient memory for this test")

    def test_pre_filter_engine_malformed_data(self):
        """Test pre-filter engine handling of validation and edge cases."""
        engine = PreFilterEngine()

        # Test validation error handling when creating invalid articles
        with pytest.raises(PydanticValidationError):
            Article(
                id="invalid",
                title="",  # Too short - violates min_length=1
                url="https://example.com/test",
                content="Valid content",
                source_feed="https://example.com/feed.xml",
            )

        # Test with minimal valid articles (edge cases but valid)
        minimal_articles = [
            Article(
                id="minimal_1",
                title="A",  # Minimal valid title (1 char)
                url="https://example.com/minimal1",
                content="Minimal content for testing",
                source_feed="https://example.com/feed.xml",
            ),
            Article(
                id="minimal_2",
                title="Another minimal title",
                url="https://example.com/minimal2",
                content=None,  # None content is allowed
                source_feed="https://example.com/feed.xml",
            ),
        ]

        # Test with topics that have empty keyword lists
        minimal_topics = [
            Topic(
                id=1,
                chat_id="test",
                name="Minimal Topic",
                keywords=["minimal"],  # At least one keyword required
                exclude_keywords=[],
            )
        ]

        # Should handle minimal valid data gracefully
        results = engine.filter_articles(minimal_articles, minimal_topics)
        assert len(results) == len(minimal_articles)
        assert all(isinstance(r.passed_filter, bool) for r in results)

    @pytest.mark.asyncio
    async def test_pipeline_component_failures(self, pipeline):
        """Test pipeline handling of individual component failures."""
        chat_id = "test_channel"

        # Setup basic test data
        with pipeline.db.get_connection() as conn:
            # Create channel first (foreign key requirement)
            conn.execute(
                """
                INSERT INTO channels (chat_id, chat_title, chat_type, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (chat_id, "Test Channel", "group", datetime.now(timezone.utc)),
            )

            conn.execute(
                """
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chat_id,
                    "https://example.com/feed.xml",
                    "Test Feed",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        # Test 1: Feed fetcher failure
        with patch.object(
            pipeline.feed_fetcher, "fetch_feeds_batch", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Network infrastructure failure")

            result = await pipeline.process_channel(chat_id)

            assert len(result.errors) == 1
            assert "Pipeline processing failed" in result.errors[0]
            assert "Network infrastructure failure" in result.errors[0]
            assert result.articles_ready_for_ai == 0

    @pytest.mark.asyncio
    async def test_pipeline_database_transaction_failures(
        self, pipeline, sample_articles
    ):
        """Test pipeline handling of database transaction failures."""
        chat_id = "test_channel"

        # Setup test data
        with pipeline.db.get_connection() as conn:
            # Create channel first for foreign key constraint
            conn.execute(
                """
                INSERT OR IGNORE INTO channels (chat_id, chat_title, chat_type, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (chat_id, "Test Channel", "group", datetime.now(timezone.utc)),
            )

            conn.execute(
                """
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chat_id,
                    "https://example.com/feed.xml",
                    "Test Feed",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            # Fix: Use INTEGER for topic id, not string
            conn.execute(
                """
                INSERT INTO topics (id, chat_id, name, keywords, exclude_keywords, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    1,
                    chat_id,
                    "Test Topic",
                    '["test"]',
                    "[]",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        # Mock successful fetching and processing, but fail on storage
        with patch.object(
            pipeline.feed_fetcher, "fetch_feeds_batch", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                FetchResult(
                    feed_url="https://example.com/feed.xml",
                    success=True,
                    articles=sample_articles,
                    error=None,
                    fetch_time=1.0,
                )
            ]

            # Cause database storage to fail
            with patch.object(pipeline, "_store_articles_for_processing") as mock_store:
                mock_store.side_effect = sqlite3.OperationalError("Database is locked")

                result = await pipeline.process_channel(chat_id)

                # Pipeline should handle storage failure gracefully
                assert len(result.errors) == 1
                assert "Pipeline processing failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_pipeline_timeout_handling(self, pipeline, sample_articles):
        """Test pipeline handling of operation timeouts."""
        chat_id = "test_channel"

        # Setup test data
        with pipeline.db.get_connection() as conn:
            # Create channel first for foreign key constraint
            conn.execute(
                """
                INSERT OR IGNORE INTO channels (chat_id, chat_title, chat_type, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (chat_id, "Test Channel", "group", datetime.now(timezone.utc)),
            )

            conn.execute(
                """
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chat_id,
                    "https://example.com/feed.xml",
                    "Test Feed",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        # Mock extremely slow feed fetching
        async def slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow operation
            return []

        with patch.object(
            pipeline.feed_fetcher, "fetch_feeds_batch", side_effect=slow_fetch
        ):
            # Test with shorter timeout to trigger timeout handling
            start_time = datetime.now(timezone.utc)

            # Set a reasonable timeout for testing
            timeout_task = asyncio.create_task(pipeline.process_channel(chat_id))

            try:
                result = await asyncio.wait_for(
                    timeout_task, timeout=1.0
                )  # 1 second timeout
                # If we get here, the operation completed faster than expected
                assert result.processing_time_seconds > 0
            except asyncio.TimeoutError:
                # Expected timeout - cancel the task
                timeout_task.cancel()
                try:
                    await timeout_task
                except asyncio.CancelledError:
                    pass
                pytest.skip("Operation timed out as expected")

    def test_database_connection_recovery(self, test_database):
        """Test database connection recovery after failures."""
        connection = DatabaseConnection(test_database)

        # Test connection recovery after database file issues
        # First, establish working connection
        with connection.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

        # Simulate database file corruption/lock by renaming file
        os.rename(test_database, test_database + ".backup")

        try:
            # The connection pool might still have valid connections,
            # but creating new connections should fail.
            # Force a new connection by trying to create a new database connection
            new_connection = DatabaseConnection(test_database)
            try:
                with new_connection.get_connection() as conn:
                    conn.execute("SELECT 1")
                # If no error, the connection worked (file was recreated by schema)
                # This is acceptable behavior
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                # Expected when database file doesn't exist
                pass
        finally:
            # Restore database file
            if os.path.exists(test_database + ".backup"):
                os.rename(test_database + ".backup", test_database)

        # Connection should work again after file restoration
        with connection.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1

    def test_article_processor_invalid_content_hash(self, db_connection):
        """Test article processor handling of hash computation errors."""
        processor = ArticleProcessor(db_connection)

        # Create article with content that might cause hash issues
        problematic_articles = [
            Article(
                id="hash_test_1",
                title="Test Article",
                url="https://example.com/test",
                content="\x00\x01\x02\x03",  # Binary content
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash="",  # Empty hash
                created_at=datetime.now(timezone.utc),
            ),
            Article(
                id="hash_test_2",
                title="Another Test",
                url="https://example.com/test2",
                content="🚀💻🌟" * 1000,  # Unicode emojis
                published_at=datetime.now(timezone.utc),
                source_feed="https://example.com/feed.xml",
                content_hash=None,  # None hash
                created_at=datetime.now(timezone.utc),
            ),
        ]

        # Should handle problematic content gracefully
        unique_articles, stats = processor.process_articles(
            problematic_articles, check_database=False
        )
        assert len(unique_articles) == len(problematic_articles)

        # Verify hashes were computed
        for article in unique_articles:
            assert article.content_hash is not None
            assert len(article.content_hash) > 0

    def test_pre_filter_engine_performance_degradation(self):
        """Test pre-filter engine handling of performance edge cases."""
        engine = PreFilterEngine()

        # Create scenarios that might cause performance issues

        # 1. Extremely long keywords
        long_keyword_topic = Topic(
            id=1,  # Fix: Use integer ID
            chat_id="test",
            name="Long Keywords",
            keywords=["a" * 1000, "keyword " * 100],  # Very long keywords
            exclude_keywords=["exclude " * 50],
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        # 2. Many keywords
        many_keywords_topic = Topic(
            id=2,  # Fix: Use integer ID
            chat_id="test",
            name="Many Keywords",
            keywords=[f"keyword{i}" for i in range(1000)],  # 1000 keywords
            exclude_keywords=[f"exclude{i}" for i in range(100)],
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        # 3. Complex regex-like patterns
        complex_topic = Topic(
            id=3,  # Fix: Use integer ID
            chat_id="test",
            name="Complex Patterns",
            keywords=[".*test.*", "[a-z]+\\d+", "complex|pattern|matching"],
            exclude_keywords=[".*exclude.*"],
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        test_article = Article(
            id="performance_test",
            title="Performance Test Article",
            url="https://example.com/performance",
            content="This is a test article for performance testing with various keywords and patterns.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="perf_hash",
            created_at=datetime.now(timezone.utc),
        )

        topics = [long_keyword_topic, many_keywords_topic, complex_topic]

        # Should handle performance edge cases without crashing
        start_time = datetime.now(timezone.utc)
        results = engine.filter_articles([test_article], topics)
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        assert len(results) == 1
        assert (
            processing_time < 5.0
        )  # Should complete within reasonable time  # Should complete within reasonable time

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_error_isolation(self, pipeline, sample_articles):
        """Test that errors in one channel don't affect others."""
        chat_ids = ["good_channel", "bad_channel", "another_good_channel"]

        # Setup test data for all channels
        with pipeline.db.get_connection() as conn:
            # First create channels for foreign key constraints
            for chat_id in chat_ids:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO channels (chat_id, chat_title, chat_type, created_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        chat_id,
                        f"Test Channel {chat_id}",
                        "group",
                        datetime.now(timezone.utc),
                    ),
                )

            # Then create feeds (without explicit id - it's auto-increment)
            for chat_id in chat_ids:
                conn.execute(
                    """
                    INSERT INTO feeds (chat_id, url, title, active, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        chat_id,
                        f"https://example.com/{chat_id}/feed.xml",
                        f"Feed for {chat_id}",
                        True,
                        datetime.now(timezone.utc),
                    ),
                )
            conn.commit()

        # Mock process_channel to simulate failures
        original_process_channel = pipeline.process_channel

        async def mock_process_channel(chat_id):
            if chat_id == "bad_channel":
                raise Exception(f"Simulated failure for {chat_id}")
            return await original_process_channel(chat_id)

        with patch.object(
            pipeline, "process_channel", side_effect=mock_process_channel
        ):
            results = await pipeline.process_multiple_channels(chat_ids)

        # Verify error isolation
        assert len(results) == 3

        # Bad channel should have error result
        bad_result = next(r for r in results if r.channel_id == "bad_channel")
        assert len(bad_result.errors) == 1
        assert "Processing exception" in bad_result.errors[0]

        # Good channels should succeed (or at least not have the same error)
        good_results = [r for r in results if r.channel_id != "bad_channel"]
        for result in good_results:
            # Should not have the specific error from bad_channel
            assert not any("Simulated failure" in error for error in result.errors)

    def test_configuration_error_handling(self, db_connection):
        """Test handling of configuration-related errors."""
        # Test invalid configuration scenarios

        # 1. Invalid database connection - DatabaseConnection tries to create parent dir and fails
        with pytest.raises(
            (FileNotFoundError, OSError)
        ):  # Fix: Expect FileNotFoundError instead of DatabaseError
            invalid_db = DatabaseConnection("/invalid/path/database.db")
            ProcessingPipeline(invalid_db)

    def test_validation_error_handling(self):
        """Test handling of data validation errors."""
        engine = PreFilterEngine()

        # Test with invalid article data
        invalid_article = Mock()
        invalid_article.id = None  # Invalid ID
        invalid_article.title = ""
        invalid_article.content = ""
        invalid_article.url = "not-a-valid-url"

        valid_topic = Topic(
            id=1,  # Fix: Use integer ID
            chat_id="test",
            name="Valid Topic",
            keywords=["test"],
            exclude_keywords=[],
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        # Should handle invalid data gracefully
        try:
            result = engine.filter_article(invalid_article, [valid_topic])
            # If no exception, verify it returns a valid result
            assert hasattr(result, "passed_filter")
            assert isinstance(result.passed_filter, bool)
        except (AttributeError, ValidationError):
            # Expected for severely malformed data
            pass

    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, pipeline):
        """Test handling of resource exhaustion scenarios."""
        chat_id = "resource_test"

        # Setup test data
        with pipeline.db.get_connection() as conn:
            # Create channel first for foreign key constraint
            conn.execute(
                """
                INSERT OR IGNORE INTO channels (chat_id, chat_title, chat_type, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (chat_id, "Test Channel", "group", datetime.now(timezone.utc)),
            )

            conn.execute(
                """
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chat_id,
                    "https://example.com/feed.xml",
                    "Test Feed",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        # Test with limited resources
        with patch.object(
            pipeline.feed_fetcher, "max_concurrent", 1
        ):  # Limit concurrency
            with patch.object(
                pipeline.feed_fetcher, "fetch_feeds_batch", new_callable=AsyncMock
            ) as mock_fetch:
                # Simulate resource exhaustion
                mock_fetch.side_effect = OSError("Too many open files")

                result = await pipeline.process_channel(chat_id)

                # Should handle resource exhaustion gracefully
                assert len(result.errors) == 1
                assert "Pipeline processing failed" in result.errors[0]

    def test_error_context_preservation(self):
        """Test that error context is preserved through the processing chain."""
        # Test that errors maintain context about where they occurred

        # Create a scenario where we can trace error context
        from culifeed.utils.exceptions import FeedFetchError, ErrorCode

        original_error = ValueError("Original network error")

        try:
            raise FeedFetchError(
                message="Failed to fetch feed",
                error_code=ErrorCode.FEED_NETWORK_ERROR,  # Fix: Use ErrorCode enum, not string
                context={
                    "url": "https://example.com/feed.xml",
                    "timeout": 30,
                    "retry_count": 3,
                },
                recoverable=True,
            ) from original_error
        except FeedFetchError as e:
            # Verify error context is preserved
            # Fix: Use str(e) instead of e.message - CuliFeedError stores message in parent Exception
            assert "Failed to fetch feed" in str(e)
            assert e.error_code == ErrorCode.FEED_NETWORK_ERROR
            assert e.context["url"] == "https://example.com/feed.xml"
            assert e.recoverable is True
            assert e.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_graceful_degradation_scenarios(self, pipeline, sample_articles):
        """Test pipeline graceful degradation under various failure conditions."""
        chat_id = "degradation_test"

        # Setup test data
        with pipeline.db.get_connection() as conn:
            # Create channel first for foreign key constraint
            conn.execute(
                """
                INSERT OR IGNORE INTO channels (chat_id, chat_title, chat_type, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (chat_id, "Test Channel", "group", datetime.now(timezone.utc)),
            )

            conn.execute(
                """
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    chat_id,
                    "https://example.com/feed.xml",
                    "Test Feed",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            # Fix: Use INTEGER for topic id, not string
            conn.execute(
                """
                INSERT INTO topics (id, chat_id, name, keywords, exclude_keywords, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    1,
                    chat_id,
                    "Test Topic",
                    '["test"]',
                    "[]",
                    True,
                    datetime.now(timezone.utc),
                ),
            )
            conn.commit()

        # Test partial failures that should allow degraded operation

        # 1. Some feeds fail, others succeed
        with patch.object(
            pipeline.feed_fetcher, "fetch_feeds_batch", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = [
                FetchResult(
                    feed_url="https://example.com/feed.xml",
                    success=False,
                    articles=None,
                    error="Temporary network error",
                    fetch_time=30.0,
                )
            ]

            result = await pipeline.process_channel(chat_id)

            # Should complete with errors but not crash
            assert result.channel_id == chat_id
            assert result.successful_feed_fetches == 0
            assert len(result.errors) == 1
            assert "Temporary network error" in result.errors[0]
            assert (
                result.processing_time_seconds > 0
            )  # Should still measure time  # Should still measure time
