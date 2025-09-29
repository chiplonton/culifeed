"""
RSS Feed Fetcher
===============

High-performance RSS feed fetching with concurrent processing,
error handling, and content validation.
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, AsyncGenerator, Any
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
import ssl
import certifi
from contextlib import asynccontextmanager

from ..database.models import Article, Feed
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import FeedFetchError, ErrorCode, ContentValidationError
from ..utils.validators import URLValidator, ContentValidator


@dataclass
class FetchResult:
    """Result of feed fetch operation."""

    feed_url: str
    success: bool
    articles: List[Article] = None
    error: Optional[str] = None
    fetch_time: Optional[datetime] = None
    article_count: int = 0

    def __post_init__(self):
        if self.articles:
            self.article_count = len(self.articles)
        if not self.fetch_time:
            self.fetch_time = datetime.now(timezone.utc)


class FeedFetcher:
    """Concurrent RSS feed fetcher with error handling and validation."""

    def __init__(self, max_concurrent: int = None, timeout: int = None):
        """Initialize feed fetcher.

        Args:
            max_concurrent: Maximum concurrent feed fetches (default from config)
            timeout: Request timeout in seconds (default from config)
        """
        settings = get_settings()
        self.max_concurrent = max_concurrent or settings.processing.parallel_feeds
        self.timeout = timeout or settings.limits.request_timeout
        self.logger = get_logger_for_component("feed_fetcher")

        # SSL context for secure requests
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    @asynccontextmanager
    async def get_session(self):
        """Get configured aiohttp session."""
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=self.max_concurrent * 2,
            limit_per_host=5,
            enable_cleanup_closed=True,
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        headers = {
            "User-Agent": "CuliFeed/1.0 (+https://github.com/culifeed/culifeed)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Encoding": "gzip, deflate",
        }

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:
            yield session

    async def fetch_feed(
        self, feed_url: str, session: aiohttp.ClientSession
    ) -> FetchResult:
        """Fetch and parse a single RSS feed.

        Args:
            feed_url: URL of the RSS feed
            session: aiohttp session for requests

        Returns:
            FetchResult with articles or error information
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Validate URL
            validated_url = URLValidator.validate_feed_url(feed_url)

            self.logger.debug(f"Fetching feed: {validated_url}")

            async with session.get(validated_url) as response:
                if response.status != 200:
                    error_msg = f"HTTP {response.status}: {response.reason}"
                    self.logger.warning(
                        f"Feed fetch failed for {validated_url}: {error_msg}"
                    )
                    return FetchResult(
                        feed_url=feed_url,
                        success=False,
                        error=error_msg,
                        fetch_time=start_time,
                    )

                # Get content
                content = await response.text()

                # Parse with feedparser
                feed_data = feedparser.parse(content)

                # Check for feed parsing errors
                if hasattr(feed_data, "bozo") and feed_data.bozo:
                    if hasattr(feed_data, "bozo_exception"):
                        error_msg = f"Feed parse error: {feed_data.bozo_exception}"
                    else:
                        error_msg = "Feed parse error: Invalid XML structure"

                    # Still try to process if we have entries
                    if not hasattr(feed_data, "entries") or not feed_data.entries:
                        self.logger.warning(
                            f"Feed parse failed for {validated_url}: {error_msg}"
                        )
                        return FetchResult(
                            feed_url=feed_url,
                            success=False,
                            error=error_msg,
                            fetch_time=start_time,
                        )
                    else:
                        self.logger.info(
                            f"Feed has parse warnings but contains entries: {validated_url}"
                        )

                # Convert entries to Article models
                articles = self._parse_entries(feed_data, validated_url)

                self.logger.info(
                    f"Fetched {len(articles)} articles from {validated_url} "
                    f"in {(datetime.now(timezone.utc) - start_time).total_seconds():.2f}s"
                )

                return FetchResult(
                    feed_url=feed_url,
                    success=True,
                    articles=articles,
                    fetch_time=start_time,
                )

        except asyncio.TimeoutError:
            error_msg = f"Request timeout after {self.timeout}s"
            self.logger.warning(f"Feed fetch timeout for {feed_url}: {error_msg}")
            return FetchResult(
                feed_url=feed_url, success=False, error=error_msg, fetch_time=start_time
            )

        except Exception as e:
            error_msg = f"Fetch error: {str(e)}"
            self.logger.error(
                f"Feed fetch failed for {feed_url}: {error_msg}", exc_info=True
            )
            return FetchResult(
                feed_url=feed_url, success=False, error=error_msg, fetch_time=start_time
            )

    def _parse_entries(self, feed_data: Any, feed_url: str) -> List[Article]:
        """Parse feed entries into Article models.

        Args:
            feed_data: Parsed feedparser data
            feed_url: Source feed URL

        Returns:
            List of Article models
        """
        articles = []
        
        # Define cutoff date for old articles (7 days ago)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=7)

        for entry in feed_data.entries:
            try:
                # Parse publication date first to filter out old articles early
                published_at = self._parse_date(entry)
                
                # Skip articles that are too old
                if published_at and published_at < cutoff_date:
                    self.logger.debug(
                        f"Skipping old article from {feed_url}: "
                        f"published {published_at.strftime('%Y-%m-%d')} "
                        f"(older than {cutoff_date.strftime('%Y-%m-%d')})"
                    )
                    continue
                
                # Extract article data
                title = ContentValidator.validate_article_title(
                    getattr(entry, "title", "Untitled")
                )

                # Get article URL
                article_url = getattr(entry, "link", "")
                if not article_url:
                    self.logger.warning(
                        f"Entry missing URL in feed {feed_url}, skipping"
                    )
                    continue

                # Validate and normalize URL
                try:
                    article_url = URLValidator.validate_article_url(article_url)
                except Exception as e:
                    self.logger.warning(
                        f"Invalid article URL '{article_url}' in feed {feed_url}: {e}"
                    )
                    continue

                # Extract content
                content = self._extract_content(entry)

                # Create Article model
                article = Article(
                    title=title,
                    url=article_url,
                    content=content,
                    published_at=published_at,
                    source_feed=feed_url,
                )

                articles.append(article)

            except Exception as e:
                self.logger.warning(
                    f"Failed to parse entry in feed {feed_url}: {e}",
                    extra={"entry_title": getattr(entry, "title", "Unknown")},
                )
                continue

        return articles

    def _extract_content(self, entry: Any) -> Optional[str]:
        """Extract content from feed entry.

        Args:
            entry: Feed entry object

        Returns:
            Extracted and cleaned content or None
        """
        # Try different content fields in order of preference
        content_fields = [
            "content",  # Atom content
            "description",  # RSS description
            "summary",  # Atom summary
        ]

        for field in content_fields:
            if hasattr(entry, field):
                raw_content = getattr(entry, field)

                # Handle list format (Atom content)
                if isinstance(raw_content, list) and raw_content:
                    raw_content = raw_content[0]

                # Extract value if it's a dict
                if isinstance(raw_content, dict):
                    raw_content = raw_content.get("value", "")

                # Clean and validate content
                if raw_content and isinstance(raw_content, str):
                    try:
                        cleaned_content = ContentValidator.validate_article_content(
                            raw_content
                        )
                        if cleaned_content:
                            return cleaned_content
                    except ContentValidationError:
                        continue

        return None

    def _parse_date(self, entry: Any) -> Optional[datetime]:
        """Parse publication date from entry.

        Args:
            entry: Feed entry object

        Returns:
            Parsed datetime in UTC or None
        """
        # Try different date fields
        date_fields = [
            "published_parsed",
            "updated_parsed",
            "created_parsed",
        ]

        for field in date_fields:
            if hasattr(entry, field):
                date_tuple = getattr(entry, field)
                if date_tuple:
                    try:
                        # Convert time.struct_time to datetime
                        import time

                        timestamp = time.mktime(date_tuple)
                        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    except (ValueError, OverflowError):
                        continue

        return None

    async def fetch_feeds(
        self, feed_urls: List[str]
    ) -> AsyncGenerator[FetchResult, None]:
        """Fetch multiple RSS feeds concurrently.

        Args:
            feed_urls: List of RSS feed URLs to fetch

        Yields:
            FetchResult objects as feeds are processed
        """
        if not feed_urls:
            return

        self.logger.info(f"Starting concurrent fetch of {len(feed_urls)} feeds")

        async with self.get_session() as session:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def fetch_with_semaphore(url: str) -> FetchResult:
                async with semaphore:
                    return await self.fetch_feed(url, session)

            # Create tasks for all feeds
            tasks = [fetch_with_semaphore(url) for url in feed_urls]

            # Process results as they complete
            for completed_task in asyncio.as_completed(tasks):
                result = await completed_task
                yield result

    async def fetch_feeds_batch(self, feed_urls: List[str]) -> List[FetchResult]:
        """Fetch multiple feeds and return all results.

        Args:
            feed_urls: List of RSS feed URLs to fetch

        Returns:
            List of FetchResult objects
        """
        results = []
        async for result in self.fetch_feeds(feed_urls):
            results.append(result)

        # Log summary
        successful = sum(1 for r in results if r.success)
        total_articles = sum(r.article_count for r in results if r.success)

        self.logger.info(
            f"Feed fetch complete: {successful}/{len(results)} feeds successful, "
            f"{total_articles} total articles"
        )

        return results
