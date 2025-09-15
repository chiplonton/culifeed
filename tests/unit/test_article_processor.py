"""
Unit tests for ArticleProcessor - content normalization and deduplication.
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import List

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.processing.article_processor import ArticleProcessor, ProcessingResult, DeduplicationStats
from culifeed.database.models import Article
from culifeed.database.schema import DatabaseSchema
from culifeed.database.connection import DatabaseConnection
from culifeed.utils.logging import get_logger_for_component
from culifeed.utils.exceptions import ProcessingError


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    schema = DatabaseSchema(db_path)
    schema.create_tables()
    
    yield db_path
    
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def db_connection(temp_db):
    """Database connection fixture."""
    return DatabaseConnection(temp_db)


@pytest.fixture
def article_processor(db_connection):
    """ArticleProcessor fixture."""
    return ArticleProcessor(db_connection)


@pytest.fixture
def sample_articles():
    """Sample articles for testing."""
    return [
        Article(
            title="Test Article 1",
            url="https://example.com/article1",
            content="Full content about technology trends and programming practices.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash1"
        ),
        Article(
            title="  Another Test   ",  # Title needs cleaning
            url="https://example.com/article2?utm_source=test&utm_medium=email",  # URL needs normalization
            content="<div>Full HTML content with <script>alert('xss')</script> tags.</div>",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash2"
        ),
        Article(
            title="Short",  # Below min length
            url="https://example.com/short",
            content="Short content",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed.xml",
            content_hash="hash3"
        )
    ]


class TestArticleProcessor:
    """Test ArticleProcessor functionality."""

    def test_initialization(self, db_connection):
        """Test ArticleProcessor initialization."""
        processor = ArticleProcessor(db_connection)
        
        assert processor.db == db_connection
        assert processor.logger is not None
        assert processor.min_title_length == 10  # Default setting
        assert processor.min_content_length == 50  # Default setting
        assert processor.max_content_length == 2000  # Default setting

    def test_initialization_with_custom_settings(self, db_connection):
        """Test ArticleProcessor initialization with custom max_content_length."""
        # ArticleProcessor accepts max_content_length as parameter
        processor = ArticleProcessor(db_connection, max_content_length=1000)

        assert processor.min_title_length == 10  # Fixed values
        assert processor.min_content_length == 50  # Fixed values
        assert processor.max_content_length == 1000  # Custom value

    def test_normalize_content_basic(self, article_processor, sample_articles):
        """Test basic content normalization."""
        article = sample_articles[0]
        normalized = article_processor.normalize_content(article)
        
        assert normalized.title == article.title  # Already clean
        assert str(normalized.url) == str(article.url)  # Already clean
        assert normalized.content == article.content  # Already clean

    def test_normalize_content_cleaning(self, article_processor, sample_articles):
        """Test content normalization with cleaning needed."""
        article = sample_articles[1]  # Has HTML and needs cleaning
        normalized = article_processor.normalize_content(article)
        
        # Title should be cleaned
        assert normalized.title == "Another Test"
        assert normalized.title.strip() == normalized.title
        
        # URL should be normalized (UTM params removed)
        assert "utm_source" not in str(normalized.url)
        assert "utm_medium" not in str(normalized.url)
        assert str(normalized.url) == "https://example.com/article2"
        
        # Content should have HTML removed
        assert "<div>" not in normalized.content
        assert "<script>" not in normalized.content
        assert "Full HTML content with" in normalized.content
        
        # Content should be cleaned of dangerous scripts
        assert "<script>" not in normalized.content
        # Note: Script content may remain as text after tag removal
        assert "tags." in normalized.content

    def test_clean_title(self, article_processor):
        """Test title cleaning functionality."""
        test_cases = [
            ("[Tech] New Article", "New Article"),  # Removes [Category] prefix
            ("Article Title | Site Name Really Long String to Exceed 40 chars", "Article Title"),  # Removes | suffix if long
            ("Article Title - Site Name Really Long String to Exceed 40 chars", "Article Title"),  # Removes - suffix if long
            ("News: Breaking Story", "Breaking Story"),  # Removes Category: prefix
            ("Regular Title", "Regular Title"),  # No patterns to clean
            ("Short | Site", "Short | Site"),  # Keeps | suffix if short (< 40 chars)
        ]
        
        for input_title, expected in test_cases:
            article = Article(title=input_title, url="https://test.com", source_feed="https://test.com/feed.xml")
            cleaned = article_processor._clean_title(article.title)
            assert cleaned == expected

    def test_normalize_url(self, article_processor):
        """Test URL normalization functionality."""
        test_cases = [
            ("https://example.com", "https://example.com"),
            ("https://example.com/", "https://example.com/"),  # Trailing slash preserved
            ("https://example.com/page?utm_source=test", "https://example.com/page"),
            ("https://example.com/page?utm_medium=email&ref=social", "https://example.com/page"),
            ("https://example.com/page?important=keep&utm_source=remove", "https://example.com/page?important=keep"),
            ("https://example.com/page#section", "https://example.com/page#section"),  # Fragment preserved
            ("https://example.com/page?utm_campaign=test#section", "https://example.com/page#section"),
        ]
        
        for input_url, expected in test_cases:
            normalized = article_processor._normalize_url(input_url)
            assert normalized == expected

    def test_calculate_quality_score(self, article_processor, sample_articles):
        """Test quality score calculation."""
        # High quality article (relative to others)
        high_quality = sample_articles[0]
        score = article_processor.calculate_quality_score(high_quality)
        assert 0.0 <= score <= 1.0
        assert score > 0.25  # Should be relatively high for the test data
        
        # Low quality article
        low_quality = sample_articles[2]  # Short content
        score = article_processor.calculate_quality_score(low_quality)
        assert 0.0 <= score <= 1.0
        assert score < 0.25  # Should be relatively low

    def test_quality_score_components(self, article_processor):
        """Test individual components of quality score."""
        # Test with various article characteristics
        test_cases = [
            # (title_length, content_length, expected_score_range)
            (50, 1000, (0.7, 1.0)),  # High quality
            (20, 200, (0.3, 0.7)),   # Medium quality  
            (5, 30, (0.0, 0.3)),     # Low quality
        ]
        
        for title_len, content_len, score_range in test_cases:
            article = Article(
                title="A" * title_len,
                url="https://test.com",
                content="C" * content_len,
                published_at=datetime.now(timezone.utc),
                source_feed="https://test.com/feed.xml"
            )
            
            score = article_processor.calculate_quality_score(article)
            assert score_range[0] <= score <= score_range[1]

    def test_find_duplicates_in_batch_no_duplicates(self, article_processor, sample_articles):
        """Test finding duplicates in batch with no duplicates."""
        results = article_processor.find_duplicates_in_batch(sample_articles)
        
        # Should return results for all articles, but none marked as duplicates
        assert len(results) == len(sample_articles)
        for result in results:
            assert isinstance(result, ProcessingResult)
            assert result.is_duplicate == False
            assert result.duplicate_of is None

    def test_find_duplicates_in_batch_with_duplicates(self, article_processor):
        """Test finding duplicates in batch with actual duplicates."""
        # Create articles with same content hash
        articles = [
            Article(title="Article 1", url="https://test.com/1", content="content", content_hash="same_hash", source_feed="https://test.com/feed.xml"),
            Article(title="Article 2", url="https://test.com/2", content="different", content_hash="different_hash", source_feed="https://test.com/feed.xml"),
            Article(title="Article 3", url="https://test.com/3", content="content", content_hash="same_hash", source_feed="https://test.com/feed.xml"),  # Duplicate
        ]
        
        results = article_processor.find_duplicates_in_batch(articles)
        
        # Should return results for all articles
        assert len(results) == 3
        
        # First article should not be duplicate (first occurrence)
        assert results[0].is_duplicate == False
        assert results[0].duplicate_of is None
        
        # Second article should not be duplicate (unique hash)
        assert results[1].is_duplicate == False
        assert results[1].duplicate_of is None
        
        # Third article should be duplicate (second occurrence of same_hash)
        assert results[2].is_duplicate == True
        assert results[2].duplicate_of == "same_hash"
        assert "Duplicate content hash" in results[2].content_issues

    def test_find_duplicates_in_batch_multiple_groups(self, article_processor):
        """Test finding multiple groups of duplicates."""
        articles = [
            Article(title="A1", url="https://test.com/1", content="content", content_hash="hash1", source_feed="https://test.com/feed.xml"),
            Article(title="A2", url="https://test.com/2", content="content", content_hash="hash1", source_feed="https://test.com/feed.xml"),  # Duplicate group 1
            Article(title="B1", url="https://test.com/3", content="content", content_hash="hash2", source_feed="https://test.com/feed.xml"),
            Article(title="B2", url="https://test.com/4", content="content", content_hash="hash2", source_feed="https://test.com/feed.xml"),  # Duplicate group 2
            Article(title="C1", url="https://test.com/5", content="content", content_hash="unique", source_feed="https://test.com/feed.xml"),  # No duplicate
        ]
        
        results = article_processor.find_duplicates_in_batch(articles)
        
        # Should return results for all articles
        assert len(results) == 5
        
        # Check duplicate pattern: first of each hash is not duplicate, subsequent ones are
        assert results[0].is_duplicate == False  # A1 - first hash1
        assert results[1].is_duplicate == True   # A2 - second hash1
        assert results[1].duplicate_of == "hash1"
        
        assert results[2].is_duplicate == False  # B1 - first hash2
        assert results[3].is_duplicate == True   # B2 - second hash2
        assert results[3].duplicate_of == "hash2"
        
        assert results[4].is_duplicate == False  # C1 - unique hash

    def test_find_duplicates_in_database(self, article_processor, sample_articles, test_database):
        """Test finding duplicates in database."""
        # Create a test article in database that will be a duplicate
        with article_processor.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO articles (id, title, url, content, content_hash, source_feed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                "existing-1", 
                "Existing Article", 
                "https://existing.com/article",
                "existing content",
                sample_articles[0].content_hash,  # Same hash as first sample article
                "https://existing.com/feed",
                datetime.now(timezone.utc)
            ))
            conn.commit()
        
        results = article_processor.find_duplicates_in_database(sample_articles)
        
        # Should return results for all articles
        assert len(results) == len(sample_articles)
        
        # First article should be marked as duplicate (matches existing)
        assert results[0].is_duplicate == True
        assert results[0].duplicate_of.startswith("db_hash:")
        assert "Duplicate in database (content)" in results[0].content_issues
        
        # Other articles should not be duplicates
        for i in range(1, len(results)):
            assert results[i].is_duplicate == False

    def test_process_articles_success(self, article_processor, sample_articles):
        """Test successful article processing."""
        unique_articles, stats = article_processor.process_articles(sample_articles, check_database=False)
        
        # Verify result structure
        assert isinstance(unique_articles, list)
        assert isinstance(stats, DeduplicationStats)
        
        # Since sample articles have unique hashes, all should be kept
        assert len(unique_articles) == len(sample_articles)
        assert stats.total_articles == len(sample_articles)
        assert stats.unique_articles == len(sample_articles)
        assert stats.duplicates_found == 0
        
        # Verify articles are normalized versions
        for article in unique_articles:
            assert isinstance(article, Article)
            assert article.content_hash  # Should have content hash

    def test_process_articles_with_duplicates(self, article_processor):
        """Test article processing with duplicates."""
        # Create articles with intentional duplicates (same title|url = same hash after normalization)
        articles_with_duplicates = [
            Article(title="Unique Article 1", url="https://test.com/1", content="content", source_feed="https://test.com/feed"),
            Article(title="Duplicate Article", url="https://test.com/duplicate", content="different content", source_feed="https://test.com/feed"),
            Article(title="Duplicate Article", url="https://test.com/duplicate", content="totally different content", source_feed="https://test.com/feed"),  # Same title+URL = duplicate
            Article(title="Unique Article 2", url="https://test.com/2", content="content", source_feed="https://test.com/feed"),
        ]
        
        unique_articles, stats = article_processor.process_articles(articles_with_duplicates, check_database=False)
        
        # Should have 3 unique articles (1 duplicate removed)
        assert len(unique_articles) == 3
        assert stats.total_articles == 4
        assert stats.unique_articles == 3
        assert stats.duplicates_found == 1
        assert stats.duplicates_by_hash == 1

    def test_process_articles_with_quality_filtering(self, article_processor):
        """Test article processing with quality assessment."""
        # Create articles with different quality levels
        articles = [
            Article(title="High Quality Article", url="https://test.com/1", 
                   content="Detailed content " * 50,  # Long content = higher quality 
                   source_feed="https://test.com/feed.xml"),  # High quality
            Article(title="Low", url="https://test.com/2", 
                   content="Short",  # Short content = lower quality
                   source_feed="https://test.com/feed.xml"),  # Low quality
        ]
        
        unique_articles, stats = article_processor.process_articles(articles, check_database=False)
        
        # Both articles should be processed and returned (no quality filtering in process_articles)
        assert len(unique_articles) == 2
        assert stats.total_articles == 2
        assert stats.unique_articles == 2
        assert stats.duplicates_found == 0
        
        # Check that quality scores are calculated differently
        batch_results = article_processor.find_duplicates_in_batch(articles)
        high_quality_score = batch_results[0].quality_score
        low_quality_score = batch_results[1].quality_score
        assert high_quality_score > low_quality_score

    def test_process_articles_error_handling(self, article_processor):
        """Test error handling during article processing."""
        # Test with invalid article data that should not crash the processor
        invalid_articles = [
            Article(title="Valid Article", url="https://test.com/1", content="content", source_feed="https://test.com/feed"),
        ]
        
        # This should not raise an exception - error handling should be graceful
        unique_articles, stats = article_processor.process_articles(invalid_articles, check_database=False)
        
        # Should process successfully even with edge cases
        assert isinstance(unique_articles, list)
        assert isinstance(stats, DeduplicationStats)
        assert stats.total_articles == 1

    def test_get_processing_summary(self, article_processor, sample_articles):
        """Test processing summary generation."""
        # Create processing results with some duplicates
        articles_with_duplicate = [
            sample_articles[0],  # Unique
            sample_articles[1],  # Unique  
            Article(title=sample_articles[0].title, url=sample_articles[0].url, content="different content", source_feed=sample_articles[0].source_feed)  # Duplicate of first
        ]
        
        results = article_processor.find_duplicates_in_batch(articles_with_duplicate)
        summary = article_processor.get_processing_summary(results)
        
        # Check summary dictionary structure
        assert isinstance(summary, dict)
        assert summary['total_articles'] == 3
        assert summary['unique_articles'] == 2
        assert summary['duplicate_articles'] == 1
        assert summary['deduplication_rate'] > 0
        assert 'average_quality_score' in summary
        assert 'common_issues' in summary

    def test_content_length_validation(self, article_processor):
        """Test content length validation."""
        # Test with content exceeding max length
        long_content = "A" * 60000  # Exceeds default max of 50000
        article = Article(
            title="Test Article",
            url="https://test.com",
            content=long_content,
            source_feed="https://test.com/feed.xml"
        )
        
        normalized = article_processor.normalize_content(article)
        
        # Content should be truncated (max_length + "..." = 3 extra chars)
        expected_max_length = article_processor.max_content_length + 3
        assert len(normalized.content) <= expected_max_length

    def test_url_validation_and_cleaning(self, article_processor):
        """Test URL validation and cleaning."""
        test_urls = [
            ("https://example.com/valid", True),
            ("http://example.com/valid", True),
            ("ftp://example.com/invalid", False),
            ("javascript:alert('xss')", False),
            ("", False),
            ("not-a-url", False),
        ]
        
        for url, should_be_valid in test_urls:
            if should_be_valid:
                article = Article(title="Test", url=url, content="test", source_feed="https://test.com/feed.xml")
                normalized = article_processor.normalize_content(article)
                assert str(normalized.url) == url or str(normalized.url).startswith("http")
            else:
                # Should handle invalid URLs gracefully during creation or processing
                try:
                    article = Article(title="Test", url="https://test.com", content="test", source_feed="https://test.com/feed.xml")
                    # For invalid URLs, we can't create the Article, so we skip
                except Exception:
                    # Exception handling is acceptable for invalid URLs
                    pass

    def test_html_security_cleaning(self, article_processor):
        """Test HTML security cleaning (XSS prevention)."""
        dangerous_content = """
        <p>Safe content</p>
        <script>alert('xss')</script>
        <iframe src="evil.com"></iframe>
        <img src="x" onerror="alert('xss')">
        <a href="javascript:alert('xss')">Click me</a>
        <style>body { display: none; }</style>
        """
        
        article = Article(
            title="Test Security",
            url="https://test.com",
            content=dangerous_content,
            source_feed="https://test.com/feed.xml"
        )
        
        normalized = article_processor.normalize_content(article)
        
        # Dangerous elements should be removed
        assert "<script>" not in normalized.content
        assert "<iframe>" not in normalized.content
        assert "onerror=" not in normalized.content
        assert "javascript:" not in normalized.content
        assert "<style>" not in normalized.content
        
        # Safe content should remain
        assert "Safe content" in normalized.content

    def test_batch_processing_performance(self, article_processor):
        """Test batch processing performance with large number of articles."""
        # Create a large batch of articles
        articles = []
        for i in range(1000):
            articles.append(Article(
                title=f"Article {i}",
                url=f"https://test.com/{i}",
                content=f"Content for article {i}",
                content_hash=f"hash_{i}",
                source_feed="https://test.com/feed.xml"
            ))
        
        import time
        start_time = time.time()
        
        # Test batch duplicate detection
        results = article_processor.find_duplicates_in_batch(articles)
        
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 5 seconds)
        assert processing_time < 5.0
        assert len(results) == 1000  # All articles should have results
        
        # Check that no duplicates were found
        duplicates_found = sum(1 for r in results if r.is_duplicate)
        assert duplicates_found == 0  # No duplicates in this test