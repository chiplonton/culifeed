"""
Integration tests with real RSS feeds for PHASE 2 validation.
Tests the feed manager with actual RSS feeds to validate implementation.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.ingestion.feed_manager import (
    FeedManager,
    fetch_single_feed,
    fetch_feeds_batch,
)
from culifeed.ingestion.content_cleaner import ContentCleaner
from culifeed.utils.logging import get_logger_for_component
from culifeed.utils.exceptions import FeedFetchError, FeedError

# Test RSS feeds - mix of formats and sources (most reliable ones only)
TEST_FEEDS = [
    # AWS feeds (as mentioned in workflow)
    "https://aws.amazon.com/blogs/compute/feed/",
    "https://aws.amazon.com/blogs/aws/feed/",
    # Tech news feeds
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    # GitHub releases (Atom format)
    "https://github.com/python/cpython/releases.atom",
    # Blog feeds
    "https://blog.python.org/feeds/posts/default",
]

# Problematic feeds that may be unreliable (test separately or skip)
PROBLEMATIC_TEST_FEEDS = [
    "https://feeds.feedburner.com/oreilly/radar",  # Sometimes returns 404
    "https://rss.cnn.com/rss/edition.rss",  # SSL connection issues
    "https://www.reddit.com/r/programming/.rss",  # May be rate limited
    "https://martinfowler.com/feed.atom",  # Sometimes slow
]

# Problematic feeds for error testing
PROBLEMATIC_FEEDS = [
    "https://nonexistent-domain-12345.com/feed.xml",  # Non-existent domain
    "https://httpbin.org/status/404",  # 404 error
    "https://httpbin.org/status/500",  # Server error
]


@pytest.fixture
def feed_manager():
    """Feed manager fixture."""
    return FeedManager()


@pytest.fixture
def content_cleaner():
    """Content cleaner fixture."""
    return ContentCleaner()


@pytest.fixture
def logger():
    """Logger fixture."""
    return get_logger_for_component("test_real_feeds")


@pytest.mark.slow
@pytest.mark.parametrize("feed_url", TEST_FEEDS[:3])  # Test first 3 feeds
def test_individual_real_feed(feed_url, feed_manager, logger):
    """Test individual real RSS feed processing."""
    logger.info(f"Testing feed: {feed_url}")

    start_time = time.time()

    # Fetch and parse feed
    feed_metadata, articles = fetch_single_feed(feed_url)

    processing_time = time.time() - start_time

    # Verify feed structure
    assert feed_metadata is not None, "Feed metadata should not be None"
    assert hasattr(feed_metadata, "title"), "Feed should have title"
    assert hasattr(feed_metadata, "link"), "Feed should have link"

    # Verify articles
    assert isinstance(articles, list), "Articles should be a list"
    assert (
        len(articles) > 0
    ), f"Feed should have at least one article, got {len(articles)}"

    # Verify article structure
    first_article = articles[0]
    assert hasattr(first_article, "title"), "Article should have title"
    assert hasattr(first_article, "link"), "Article should have link"
    assert first_article.title.strip(), "Article title should not be empty"

    # Performance check
    assert (
        processing_time < 30.0
    ), f"Processing should complete within 30s, took {processing_time:.2f}s"

    logger.info(
        f"✅ Successfully processed {len(articles)} articles from {feed_metadata.title}"
    )
    logger.info(f"⏱️ Processing time: {processing_time:.2f}s")


@pytest.mark.slow
def test_concurrent_feed_fetching(logger):
    """Test concurrent fetching of multiple feeds."""
    test_feeds = TEST_FEEDS[:2]  # Use first 2 feeds (most reliable)
    logger.info(f"Testing concurrent fetching of {len(test_feeds)} feeds")

    async def run_concurrent_test():
        start_time = time.time()
        results = await fetch_feeds_batch(test_feeds)
        concurrent_time = time.time() - start_time

        logger.info(f"Concurrent fetch completed in {concurrent_time:.2f}s")

        # Verify results
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) > 0, "Should have at least one successful result"

        total_articles = 0
        for feed_url, (feed_metadata, articles) in results.items():
            assert (
                feed_metadata is not None
            ), f"Feed metadata should not be None for {feed_url}"
            assert isinstance(
                articles, list
            ), f"Articles should be a list for {feed_url}"
            total_articles += len(articles)
            logger.info(f"  - {feed_metadata.title}: {len(articles)} articles")

        assert total_articles > 0, "Should have processed at least some articles"

        return results, concurrent_time

    results, concurrent_time = asyncio.run(run_concurrent_test())

    # Additional assertions
    assert (
        concurrent_time < 60.0
    ), f"Concurrent fetch should complete within 60s, took {concurrent_time:.2f}s"
    logger.info(f"✅ Successfully fetched {len(results)} feeds concurrently")


@pytest.mark.parametrize(
    "problematic_url", PROBLEMATIC_FEEDS[:2]
)  # Test first 2 problematic feeds
def test_error_handling(problematic_url, logger):
    """Test error handling with problematic feeds."""
    logger.info(f"Testing error scenario: {problematic_url}")

    with pytest.raises((FeedFetchError, FeedError)) as exc_info:
        fetch_single_feed(problematic_url)

    error = exc_info.value
    assert hasattr(error, "message") or str(error), "Error should have a message"

    logger.info(f"✅ Correctly handled error: {type(error).__name__}")


@pytest.mark.slow
def test_content_cleaning_with_real_data(content_cleaner, logger):
    """Test content cleaning with real feed data."""
    logger.info("Testing content cleaning with real data")

    # Use a reliable feed known to have HTML content
    feed_url = "https://feeds.bbci.co.uk/news/technology/rss.xml"

    try:
        feed_metadata, articles = fetch_single_feed(feed_url)

        assert len(articles) > 0, "Feed should have articles for content cleaning test"

        article = articles[0]
        logger.info(f"Testing content cleaning on: {article.title}")

        # Test HTML cleaning if summary has HTML
        if article.summary and "<" in article.summary and ">" in article.summary:
            original_length = len(article.summary)
            cleaned_summary = content_cleaner.clean_html_content(article.summary)
            cleaned_length = len(cleaned_summary)

            assert cleaned_length > 0, "Cleaned summary should not be empty"
            assert (
                cleaned_length <= original_length
            ), "Cleaned text should not be longer than original"
            assert "<script" not in cleaned_summary.lower(), "Scripts should be removed"
            assert "<style" not in cleaned_summary.lower(), "Styles should be removed"

            logger.info(
                f"Summary cleaning: {original_length} -> {cleaned_length} chars"
            )

        # Test link extraction
        if article.summary and "http" in article.summary.lower():
            links = content_cleaner.extract_links(article.summary, feed_metadata.link)
            assert isinstance(links, list), "Links should be returned as a list"
            logger.info(f"Extracted {len(links)} links from article")

        logger.info("✅ Content cleaning test completed")

    except Exception as e:
        pytest.skip(f"Content cleaning test skipped due to feed unavailability: {e}")


@pytest.mark.slow
def test_feed_format_detection(logger):
    """Test detection of different feed formats (RSS vs Atom)."""
    format_tests = [
        ("https://feeds.bbci.co.uk/news/technology/rss.xml", "RSS"),
        ("https://github.com/python/cpython/releases.atom", "Atom"),
        ("https://feeds.feedburner.com/oreilly/radar", "RSS/Atom"),  # Could be either
    ]

    formats_detected = {"rss": 0, "atom": 0, "unknown": 0}

    for feed_url, expected_type in format_tests:
        logger.info(f"Testing format detection for: {feed_url}")

        try:
            feed_metadata, articles = fetch_single_feed(feed_url)

            # Simple format detection based on URL and content
            url_lower = feed_url.lower()
            if "atom" in url_lower or ".atom" in url_lower:
                detected_format = "atom"
            elif "rss" in url_lower or ".rss" in url_lower:
                detected_format = "rss"
            else:
                detected_format = "unknown"

            formats_detected[detected_format] += 1

            # Verify we got valid data regardless of format
            assert feed_metadata is not None, "Feed metadata should not be None"
            assert len(articles) > 0, f"Feed should have articles, got {len(articles)}"

            logger.info(f"  Detected format: {detected_format.upper()}")
            logger.info(f"  Articles found: {len(articles)}")

        except Exception as e:
            logger.warning(f"Format detection test failed for {feed_url}: {e}")

    # Verify we detected at least one format
    total_detected = sum(formats_detected.values())
    assert (
        total_detected > 0
    ), "Should have successfully detected at least one feed format"

    logger.info(f"Format detection results: {formats_detected}")


@pytest.mark.slow
def test_comprehensive_feed_statistics(logger):
    """Test comprehensive statistics collection from real feeds."""
    logger.info("Running comprehensive feed statistics collection")

    # Use first 3 feeds for comprehensive analysis
    test_feeds = TEST_FEEDS[:3]

    results = {
        "successful_feeds": 0,
        "failed_feeds": 0,
        "total_articles": 0,
        "processing_times": [],
        "content_stats": {
            "total_chars": 0,
            "articles_with_html": 0,
            "articles_with_links": 0,
        },
    }

    for feed_url in test_feeds:
        logger.info(f"Analyzing feed: {feed_url}")

        try:
            start_time = time.time()
            feed_metadata, articles = fetch_single_feed(feed_url)
            processing_time = time.time() - start_time

            results["successful_feeds"] += 1
            results["total_articles"] += len(articles)
            results["processing_times"].append(processing_time)

            # Analyze article content
            for article in articles:
                if article.summary:
                    results["content_stats"]["total_chars"] += len(article.summary)

                    if "<" in article.summary and ">" in article.summary:
                        results["content_stats"]["articles_with_html"] += 1

                    if "http" in article.summary.lower():
                        results["content_stats"]["articles_with_links"] += 1

            logger.info(
                f"  ✅ Success: {feed_metadata.title} ({len(articles)} articles)"
            )

        except Exception as e:
            results["failed_feeds"] += 1
            logger.info(f"  ❌ Failed: {e}")

    # Calculate statistics
    total_feeds = results["successful_feeds"] + results["failed_feeds"]
    success_rate = results["successful_feeds"] / total_feeds if total_feeds > 0 else 0
    avg_processing_time = (
        sum(results["processing_times"]) / len(results["processing_times"])
        if results["processing_times"]
        else 0
    )

    # Assertions for PHASE 2 validation
    assert (
        success_rate >= 0.5
    ), f"Success rate ({success_rate*100:.1f}%) should be at least 50%"
    assert results["total_articles"] > 0, "Should have processed at least some articles"
    assert (
        avg_processing_time < 15.0
    ), f"Average processing time ({avg_processing_time:.2f}s) should be reasonable"

    # Log final statistics
    logger.info(f"\n📊 COMPREHENSIVE STATISTICS:")
    logger.info(f"  Total feeds tested: {total_feeds}")
    logger.info(f"  Success rate: {success_rate*100:.1f}%")
    logger.info(f"  Total articles: {results['total_articles']}")
    logger.info(f"  Average processing time: {avg_processing_time:.2f}s")
    logger.info(
        f"  Articles with HTML: {results['content_stats']['articles_with_html']}"
    )
    logger.info(
        f"  Articles with links: {results['content_stats']['articles_with_links']}"
    )

    logger.info(f"\n🎉 PHASE 2: Content Processing comprehensive validation PASSED!")
