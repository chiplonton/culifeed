#!/usr/bin/env python3
"""
Test Manual Processing Service
==============================

Tests for the ManualProcessingService that handles manual feed processing operations.
This covers the fixes made for article storage and processing.
"""

import asyncio
import pytest
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from culifeed.services.manual_processing_service import ManualProcessingService, FeedFetchSummary, BatchProcessingSummary
from culifeed.database.connection import DatabaseConnection
from culifeed.database.schema import DatabaseSchema
from culifeed.database.models import Feed
from culifeed.ingestion.feed_manager import ParsedArticle


class TestManualProcessingService:
    """Test suite for ManualProcessingService."""

    @pytest.fixture
    def test_database(self):
        """Create a temporary test database."""
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
    def db_connection(self, test_database):
        """Create database connection."""
        return DatabaseConnection(test_database)

    @pytest.fixture
    def service(self, db_connection):
        """Create ManualProcessingService instance."""
        return ManualProcessingService(db_connection)

    @pytest.fixture
    def sample_parsed_articles(self):
        """Create sample ParsedArticle objects."""
        return [
            ParsedArticle(
                title="Test Article 1",
                link="https://example.com/article1",
                summary="This is a test article about EC2",
                content="Detailed content about AWS EC2 instances and spot pricing",
                published=datetime.now(timezone.utc),
                guid="article1"
            ),
            ParsedArticle(
                title="Test Article 2",
                link="https://example.com/article2",
                summary="Another test article about reserved instances",
                content="Content about AWS reserved instances and savings",
                published=datetime.now(timezone.utc),
                guid="article2"
            )
        ]

    def test_store_articles_success(self, service, sample_parsed_articles, test_database):
        """Test that _store_articles correctly stores ParsedArticle objects."""
        feed_url = "https://example.com/feed.xml"

        # Store articles
        service._store_articles(sample_parsed_articles, feed_url)

        # Verify articles were stored
        with sqlite3.connect(test_database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles WHERE source_feed = ?", (feed_url,))
            count = cursor.fetchone()[0]

            assert count == 2, f"Expected 2 articles, got {count}"

            # Check article details
            cursor.execute("SELECT title, url, content, source_feed FROM articles WHERE source_feed = ?", (feed_url,))
            articles = cursor.fetchall()

            titles = [article[0] for article in articles]
            assert "Test Article 1" in titles
            assert "Test Article 2" in titles

            # Verify source_feed is set correctly
            for article in articles:
                assert article[3] == feed_url

    def test_store_articles_empty_list(self, service):
        """Test that _store_articles handles empty article list gracefully."""
        # Should not raise an exception
        service._store_articles([], "https://example.com/feed.xml")

    def test_store_articles_content_handling(self, service, test_database):
        """Test that _store_articles handles different content scenarios."""
        articles = [
            ParsedArticle(
                title="No Content Article",
                link="https://example.com/no-content",
                summary="Summary only",
                content=None,  # No content
                published=datetime.now(timezone.utc),
                guid="no-content"
            ),
            ParsedArticle(
                title="Empty Content Article",
                link="https://example.com/empty-content",
                summary="Summary only",
                content="",  # Empty content
                published=datetime.now(timezone.utc),
                guid="empty-content"
            )
        ]

        feed_url = "https://example.com/test-feed.xml"
        service._store_articles(articles, feed_url)

        # Verify articles were stored with fallback to summary
        with sqlite3.connect(test_database) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title, content FROM articles WHERE source_feed = ?", (feed_url,))
            stored_articles = cursor.fetchall()

            assert len(stored_articles) == 2

            # Article with no content should use summary
            no_content_article = next(a for a in stored_articles if a[0] == "No Content Article")
            assert no_content_article[1] == "Summary only"

            # Article with empty content should use summary
            empty_content_article = next(a for a in stored_articles if a[0] == "Empty Content Article")
            assert empty_content_article[1] == "Summary only"

    @pytest.mark.asyncio
    async def test_process_feeds_for_chat_with_storage(self, service, test_database):
        """Test that process_feeds_for_chat actually stores articles (the main bug fix)."""
        chat_id = "test_chat_123"

        # Setup test feed in database
        with sqlite3.connect(test_database) as conn:
            cursor = conn.cursor()

            # Create test channel
            cursor.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, "Test Channel", "group", True, datetime.now(timezone.utc), datetime.now(timezone.utc)))

            # Create test feed
            cursor.execute("""
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (chat_id, "https://example.com/feed.xml", "Test Feed", True, datetime.now(timezone.utc)))

            conn.commit()

        # Mock the feed_manager.fetch_feed to return sample data
        with patch.object(service.feed_manager, 'fetch_feed') as mock_fetch:
            # Mock feed metadata
            mock_metadata = Mock()
            mock_metadata.title = "Test Feed"
            mock_metadata.description = "Test feed description"

            # Mock articles
            mock_articles = [
                ParsedArticle(
                    title="Mock Article 1",
                    link="https://example.com/article1",
                    summary="Summary 1",
                    content="Content 1",
                    published=datetime.now(timezone.utc),
                    guid="mock1"
                ),
                ParsedArticle(
                    title="Mock Article 2",
                    link="https://example.com/article2",
                    summary="Summary 2",
                    content="Content 2",
                    published=datetime.now(timezone.utc),
                    guid="mock2"
                )
            ]

            mock_fetch.return_value = (mock_metadata, mock_articles)

            # Process feeds
            result = await service.process_feeds_for_chat(chat_id)

            # Verify processing results
            assert result.total_feeds == 1
            assert result.successful_feeds == 1
            assert result.failed_feeds == 0
            assert result.total_articles == 2

            # Most importantly: verify articles were actually stored in database
            with sqlite3.connect(test_database) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM articles WHERE source_feed = ?", ("https://example.com/feed.xml",))
                stored_count = cursor.fetchone()[0]

                assert stored_count == 2, f"Expected 2 articles stored, got {stored_count}"

                # Verify article content
                cursor.execute("SELECT title FROM articles WHERE source_feed = ?", ("https://example.com/feed.xml",))
                titles = [row[0] for row in cursor.fetchall()]
                assert "Mock Article 1" in titles
                assert "Mock Article 2" in titles

    @pytest.mark.asyncio
    async def test_fetch_single_feed_success(self, service):
        """Test fetch_single_feed returns correct summary."""
        with patch.object(service.feed_manager, 'fetch_feed') as mock_fetch:
            mock_metadata = Mock()
            mock_metadata.title = "Test Feed"
            mock_metadata.description = "Test description"

            mock_articles = [
                ParsedArticle(
                    title="Sample Article",
                    link="https://example.com/article",
                    summary="Sample summary",
                    content="Sample content",
                    published=datetime.now(timezone.utc),
                    guid="sample"
                )
            ]

            mock_fetch.return_value = (mock_metadata, mock_articles)

            result = await service.fetch_single_feed("https://example.com/feed.xml")

            assert result.success is True
            assert result.title == "Test Feed"
            assert result.article_count == 1
            assert len(result.sample_articles) == 1
            assert result.sample_articles[0]['title'] == "Sample Article"

    @pytest.mark.asyncio
    async def test_fetch_single_feed_failure(self, service):
        """Test fetch_single_feed handles failures correctly."""
        with patch.object(service.feed_manager, 'fetch_feed') as mock_fetch:
            mock_fetch.return_value = (None, [])  # Simulate failure

            result = await service.fetch_single_feed("https://invalid-feed.com/feed.xml")

            assert result.success is False
            assert result.error_message == "Failed to fetch or parse RSS feed"
            assert result.article_count == 0

    @pytest.mark.asyncio
    async def test_process_feeds_no_feeds(self, service, test_database):
        """Test process_feeds_for_chat when no feeds exist."""
        chat_id = "empty_chat"

        result = await service.process_feeds_for_chat(chat_id)

        assert result.total_feeds == 0
        assert result.successful_feeds == 0
        assert result.failed_feeds == 0
        assert result.total_articles == 0


if __name__ == "__main__":
    pytest.main([__file__])