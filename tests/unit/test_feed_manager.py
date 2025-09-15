"""
Unit Tests for RSS Feed Manager
===============================

Tests for RSS feed parsing, content extraction, and error handling.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Tuple

import requests
import aiohttp

from culifeed.ingestion.feed_manager import (
    FeedManager, 
    ParsedArticle, 
    FeedMetadata,
    fetch_single_feed,
    fetch_feeds_batch
)
from culifeed.ingestion.content_cleaner import ContentCleaner
from culifeed.utils.exceptions import FeedFetchError, ContentValidationError, FeedError
from culifeed.config.settings import get_settings


# Sample RSS feed data for testing
SAMPLE_RSS_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test RSS Feed</title>
        <link>http://example.com</link>
        <description>Test feed for unit testing</description>
        <language>en-us</language>
        <pubDate>Sat, 07 Sep 2024 00:00:01 GMT</pubDate>
        <category>Technology</category>
        <generator>CuliFeed Test</generator>
        <item>
            <title>Test Article 1</title>
            <link>http://example.com/article1</link>
            <description>This is a test article summary with &lt;strong&gt;HTML&lt;/strong&gt;</description>
            <pubDate>Thu, 05 Sep 2024 12:00:00 GMT</pubDate>
            <guid>article-1-guid</guid>
            <author>test@example.com</author>
            <category>Tech</category>
        </item>
        <item>
            <title>Test Article 2</title>
            <link>http://example.com/article2</link>
            <description>Another test article with some content</description>
            <pubDate>Wed, 04 Sep 2024 15:30:00 GMT</pubDate>
            <guid>article-2-guid</guid>
        </item>
    </channel>
</rss>'''

SAMPLE_ATOM_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
    <title>Test Atom Feed</title>
    <link href="http://example.com"/>
    <id>http://example.com/feed</id>
    <updated>2024-09-07T00:00:01Z</updated>
    <subtitle>Test Atom feed for unit testing</subtitle>
    <category term="Technology"/>
    
    <entry>
        <title>Atom Test Article</title>
        <link href="http://example.com/atom-article"/>
        <id>http://example.com/atom-article</id>
        <updated>2024-09-05T12:00:00Z</updated>
        <published>2024-09-05T12:00:00Z</published>
        <summary>This is an Atom article summary</summary>
        <content type="html">&lt;p&gt;Full content with &lt;em&gt;formatting&lt;/em&gt;&lt;/p&gt;</content>
        <author>
            <name>Atom Author</name>
            <email>atom@example.com</email>
        </author>
        <category term="Science"/>
    </entry>
</feed>'''

MALFORMED_RSS_FEED = '''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Malformed Feed</title>
        <link>http://example.com</link>
        <description>Feed with malformed XML
        <item>
            <title>Broken Article</title>
            <link>http://example.com/broken
            <description>Missing closing tags
        </item>
    </channel>
</rss>'''


class TestFeedManager:
    """Test cases for FeedManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.feed_manager = FeedManager()
        self.test_url = "http://example.com/rss.xml"
        self.content_cleaner = ContentCleaner()
    
    def test_initialization(self):
        """Test FeedManager initialization."""
        assert self.feed_manager is not None
        assert self.feed_manager.settings is not None
        assert self.feed_manager.logger is not None
        assert self.feed_manager.session is not None
        assert isinstance(self.feed_manager._feed_errors, dict)
        assert isinstance(self.feed_manager._last_fetch_times, dict)
    
    @patch('requests.Session.get')
    def test_fetch_feed_success_rss(self, mock_get):
        """Test successful RSS feed fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.content = SAMPLE_RSS_FEED.encode('utf-8')
        mock_response.headers = {'content-type': 'application/rss+xml'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Fetch feed
        feed_metadata, articles = self.feed_manager.fetch_feed(self.test_url)
        
        # Verify feed metadata
        assert isinstance(feed_metadata, FeedMetadata)
        assert feed_metadata.title == "Test RSS Feed"
        assert feed_metadata.link == "http://example.com"
        assert feed_metadata.description == "Test feed for unit testing"
        assert feed_metadata.language == "en-us"
        assert len(feed_metadata.categories) == 1
        assert "Technology" in feed_metadata.categories
        
        # Verify articles
        assert len(articles) == 2
        
        # Check first article
        article1 = articles[0]
        assert isinstance(article1, ParsedArticle)
        assert article1.title == "Test Article 1"
        assert article1.link == "http://example.com/article1"
        assert "test article summary" in article1.summary.lower()
        assert article1.guid == "article-1-guid"
        assert article1.author == "test@example.com"
        assert "Tech" in article1.categories
        assert isinstance(article1.published, datetime)
        
        # Check second article
        article2 = articles[1] 
        assert article2.title == "Test Article 2"
        assert article2.link == "http://example.com/article2"
        assert "another test article" in article2.summary.lower()
        assert article2.guid == "article-2-guid"
    
    @patch('requests.Session.get')
    def test_fetch_feed_success_atom(self, mock_get):
        """Test successful Atom feed fetching."""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.content = SAMPLE_ATOM_FEED.encode('utf-8')
        mock_response.headers = {'content-type': 'application/atom+xml'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Fetch feed
        feed_metadata, articles = self.feed_manager.fetch_feed(self.test_url)
        
        # Verify feed metadata
        assert feed_metadata.title == "Test Atom Feed"
        assert feed_metadata.link == "http://example.com"
        assert feed_metadata.description == "Test Atom feed for unit testing"
        
        # Verify articles
        assert len(articles) == 1
        article = articles[0]
        assert article.title == "Atom Test Article"
        assert article.link == "http://example.com/atom-article"
        assert article.summary == "This is an Atom article summary"
        assert article.content == "<p>Full content with <em>formatting</em></p>"
        assert article.author == "Atom Author (atom@example.com)"
        assert "Science" in article.categories
    
    @patch('requests.Session.get')
    def test_fetch_feed_malformed_xml(self, mock_get):
        """Test handling of malformed XML feeds."""
        # Mock response with malformed XML
        mock_response = Mock()
        mock_response.content = MALFORMED_RSS_FEED.encode('utf-8')
        mock_response.headers = {'content-type': 'application/rss+xml'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Should still parse but with bozo flag
        feed_metadata, articles = self.feed_manager.fetch_feed(self.test_url)
        
        # Verify it still extracted some data despite malformed XML
        assert feed_metadata.title == "Malformed Feed"
        # May have partial article data depending on parser behavior
    
    def test_fetch_feed_invalid_url(self):
        """Test handling of invalid URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com/feed.xml",
            "javascript:alert('xss')",
            "file:///etc/passwd"
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises(FeedFetchError):
                self.feed_manager.fetch_feed(invalid_url)
    
    @patch('requests.Session.get')
    def test_fetch_feed_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        # Mock HTTP error
        mock_get.side_effect = requests.exceptions.HTTPError("404 Not Found")
        
        with pytest.raises(FeedFetchError):
            self.feed_manager.fetch_feed(self.test_url)
        
        # Check that error was recorded
        assert self.test_url in self.feed_manager._feed_errors
        assert self.feed_manager._feed_errors[self.test_url] == 1
    
    @patch('requests.Session.get')
    def test_fetch_feed_timeout(self, mock_get):
        """Test handling of request timeouts."""
        # Mock timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with pytest.raises(FeedFetchError):
            self.feed_manager.fetch_feed(self.test_url)
    
    @patch('requests.Session.get')
    def test_fetch_feed_connection_error(self, mock_get):
        """Test handling of connection errors."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(FeedFetchError):
            self.feed_manager.fetch_feed(self.test_url)
    
    def test_feed_error_tracking(self):
        """Test feed error counting and disabling."""
        # Simulate multiple errors for a feed
        for i in range(5):
            self.feed_manager._record_feed_error(self.test_url)
        
        assert self.feed_manager._feed_errors[self.test_url] == 5
        
        # If max_feed_errors is 10, feed shouldn't be disabled yet
        assert not self.feed_manager._should_skip_feed(self.test_url)
        
        # Add more errors to exceed limit
        for i in range(10):
            self.feed_manager._record_feed_error(self.test_url)
        
        # Now feed should be disabled
        assert self.feed_manager._should_skip_feed(self.test_url)
    
    def test_feed_error_reset(self):
        """Test resetting feed error counts."""
        # Add some errors
        for i in range(3):
            self.feed_manager._record_feed_error(self.test_url)
        
        assert self.feed_manager._feed_errors[self.test_url] == 3
        
        # Record successful fetch
        self.feed_manager._record_successful_fetch(self.test_url)
        
        # Error count should be reset
        assert self.test_url not in self.feed_manager._feed_errors
        assert self.test_url in self.feed_manager._last_fetch_times
    
    def test_get_feed_health_status(self):
        """Test feed health status reporting."""
        # Add some errors and successful fetches
        self.feed_manager._record_feed_error("http://bad-feed.com/rss")
        self.feed_manager._record_successful_fetch("http://good-feed.com/rss")
        
        status = self.feed_manager.get_feed_health_status()
        
        assert "http://bad-feed.com/rss" in status
        assert "http://good-feed.com/rss" in status
        
        bad_status = status["http://bad-feed.com/rss"]
        assert bad_status["error_count"] == 1
        assert bad_status["last_fetch"] is None
        assert not bad_status["is_disabled"]
        
        good_status = status["http://good-feed.com/rss"]
        assert good_status["error_count"] == 0
        assert good_status["last_fetch"] is not None
        assert not good_status["is_disabled"]
    
    @pytest.mark.asyncio
    async def test_fetch_feed_async(self):
        """Test async feed fetching."""
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.read = AsyncMock(return_value=SAMPLE_RSS_FEED.encode('utf-8'))
        mock_response.headers = {'content-type': 'application/rss+xml'}
        mock_response.raise_for_status = AsyncMock()
        
        # Mock session with proper async context manager
        mock_session = AsyncMock()
        # Use MagicMock to create a proper context manager
        from unittest.mock import MagicMock
        context_manager = MagicMock()
        context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        context_manager.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=context_manager)
        
        # Fetch feed async
        feed_metadata, articles = await self.feed_manager.fetch_feed_async(
            self.test_url, mock_session
        )
        
        # Verify results
        assert feed_metadata.title == "Test RSS Feed"
        assert len(articles) == 2
    
    @pytest.mark.asyncio
    async def test_fetch_multiple_feeds(self):
        """Test concurrent feed fetching."""
        feed_urls = [
            "http://example1.com/rss",
            "http://example2.com/rss",
            "http://example3.com/rss"
        ]
        
        # Mock the async fetch method
        async def mock_fetch_with_error_handling(feed_url, session):
            if "example1" in feed_url:
                # Return successful result for first feed
                feed_metadata = FeedMetadata(
                    title=f"Feed {feed_url}",
                    link=feed_url,
                    description="Test feed"
                )
                articles = [
                    ParsedArticle(
                        title=f"Article from {feed_url}",
                        link=f"{feed_url}/article1",
                        summary="Test article"
                    )
                ]
                return (feed_metadata, articles)
            else:
                # Return None for other feeds (simulating errors)
                return None
        
        # Patch the internal method
        with patch.object(
            self.feed_manager,
            '_fetch_with_error_handling',
            side_effect=mock_fetch_with_error_handling
        ):
            results = await self.feed_manager.fetch_multiple_feeds(feed_urls)
        
        # Should have results for only one successful feed
        assert len(results) == 1
        assert "http://example1.com/rss" in results
    
    def test_extract_feed_metadata_edge_cases(self):
        """Test feed metadata extraction with edge cases."""
        # Test with minimal feed data
        minimal_feed_data = Mock()
        minimal_feed_data.title = ""
        minimal_feed_data.link = ""
        minimal_feed_data.description = ""
        
        metadata = self.feed_manager._extract_feed_metadata(
            minimal_feed_data, "http://test.com/feed"
        )
        
        # Should use fallback values
        assert metadata.title == "Unknown Feed"
        assert metadata.link == "http://test.com/feed"
        assert metadata.description == ""
    
    def test_extract_article_edge_cases(self):
        """Test article extraction with edge cases."""
        # Test with minimal entry data
        minimal_entry = Mock()
        minimal_entry.title = ""
        minimal_entry.link = ""
        minimal_entry.summary = ""
        
        # Should raise ContentValidationError for empty title
        with pytest.raises(ContentValidationError):
            self.feed_manager._extract_article(minimal_entry, "http://test.com")
    
    def test_content_length_validation(self):
        """Test content length validation and truncation."""
        # Mock entry with very long content
        long_entry = Mock()
        long_entry.title = "Test Article"
        long_entry.link = "http://example.com/article"
        long_entry.summary = "x" * 10000  # Very long summary
        
        # Mock settings for testing
        with patch.object(self.feed_manager.settings.processing, 'max_content_length', 1000):
            article = self.feed_manager._extract_article(long_entry, "http://test.com")
            
            # Summary should be truncated
            assert len(article.summary) <= 1003  # 1000 + "..."


class TestParsedArticle:
    """Test cases for ParsedArticle dataclass."""
    
    def test_article_creation(self):
        """Test creating ParsedArticle instances."""
        article = ParsedArticle(
            title="Test Article",
            link="http://example.com/article",
            summary="Test summary"
        )
        
        assert article.title == "Test Article"
        assert article.link == "http://example.com/article"
        assert article.summary == "Test summary"
        assert article.categories == []  # Default value
        assert article.enclosures == []  # Default value
        assert article.content is None
    
    def test_article_with_optional_fields(self):
        """Test article creation with optional fields."""
        published = datetime.now(timezone.utc)
        
        article = ParsedArticle(
            title="Full Article",
            link="http://example.com/full",
            summary="Summary",
            content="Full content",
            published=published,
            author="Test Author",
            categories=["Tech", "News"],
            guid="unique-id",
            enclosures=[{"url": "http://example.com/media.mp3", "type": "audio/mpeg"}]
        )
        
        assert article.content == "Full content"
        assert article.published == published
        assert article.author == "Test Author"
        assert len(article.categories) == 2
        assert article.guid == "unique-id"
        assert len(article.enclosures) == 1


class TestFeedMetadata:
    """Test cases for FeedMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating FeedMetadata instances."""
        metadata = FeedMetadata(
            title="Test Feed",
            link="http://example.com",
            description="Test description"
        )
        
        assert metadata.title == "Test Feed"
        assert metadata.link == "http://example.com"
        assert metadata.description == "Test description"
        assert metadata.categories == []  # Default value
    
    def test_metadata_with_optional_fields(self):
        """Test metadata creation with optional fields."""
        updated = datetime.now(timezone.utc)
        
        metadata = FeedMetadata(
            title="Full Feed",
            link="http://example.com",
            description="Description",
            language="en-us",
            updated=updated,
            generator="Test Generator",
            categories=["Tech", "News"],
            image={"url": "http://example.com/logo.png"}
        )
        
        assert metadata.language == "en-us"
        assert metadata.updated == updated
        assert metadata.generator == "Test Generator"
        assert len(metadata.categories) == 2
        assert metadata.image["url"] == "http://example.com/logo.png"


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    @patch('culifeed.ingestion.feed_manager.FeedManager.fetch_feed')
    def test_fetch_single_feed(self, mock_fetch):
        """Test fetch_single_feed convenience function."""
        # Mock return value
        mock_metadata = FeedMetadata("Test", "http://test.com", "Description")
        mock_articles = [ParsedArticle("Title", "http://test.com/1", "Summary")]
        mock_fetch.return_value = (mock_metadata, mock_articles)
        
        # Call function
        metadata, articles = fetch_single_feed("http://test.com/feed")
        
        # Verify
        assert metadata.title == "Test"
        assert len(articles) == 1
        mock_fetch.assert_called_once_with("http://test.com/feed")
    
    @pytest.mark.asyncio
    @patch('culifeed.ingestion.feed_manager.FeedManager.fetch_multiple_feeds')
    async def test_fetch_feeds_batch(self, mock_fetch):
        """Test fetch_feeds_batch convenience function."""
        # Mock return value
        mock_results = {
            "http://test1.com": (
                FeedMetadata("Test1", "http://test1.com", "Desc1"),
                [ParsedArticle("Title1", "http://test1.com/1", "Summary1")]
            )
        }
        mock_fetch.return_value = mock_results
        
        # Call function
        results = await fetch_feeds_batch(["http://test1.com", "http://test2.com"])
        
        # Verify
        assert len(results) == 1
        assert "http://test1.com" in results
        mock_fetch.assert_called_once()


class TestIntegration:
    """Integration tests combining feed manager and content cleaner."""
    
    def setup_method(self):
        """Set up test environment."""
        self.feed_manager = FeedManager()
        self.content_cleaner = ContentCleaner()
    
    @patch('requests.Session.get')
    def test_feed_with_html_content(self, mock_get):
        """Test processing feed with HTML content."""
        html_content_feed = '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>HTML Content Feed</title>
                <link>http://example.com</link>
                <description>Feed with HTML content</description>
                <item>
                    <title>Article with HTML</title>
                    <link>http://example.com/html-article</link>
                    <description><![CDATA[
                        <p>This article has <strong>HTML formatting</strong> and 
                        <a href="http://example.com/link">links</a>.</p>
                        <script>alert('xss')</script>
                        <p>More content here.</p>
                    ]]></description>
                </item>
            </channel>
        </rss>'''
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.content = html_content_feed.encode('utf-8')
        mock_response.headers = {'content-type': 'application/rss+xml'}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Fetch feed
        feed_metadata, articles = self.feed_manager.fetch_feed("http://example.com/feed")
        
        # Process article content with cleaner
        article = articles[0]
        cleaned_summary = self.content_cleaner.clean_html_content(
            article.summary, "http://example.com"
        )
        
        # Verify HTML was cleaned
        assert "script" not in cleaned_summary.lower()
        assert "alert" not in cleaned_summary.lower()
        assert "HTML formatting" in cleaned_summary
        assert "links" in cleaned_summary
        assert "More content here" in cleaned_summary


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])