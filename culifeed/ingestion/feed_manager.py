"""
RSS Feed Manager
===============

Handles RSS feed parsing, validation, and content extraction using feedparser.

This module provides robust RSS feed processing with:
- Support for RSS, Atom, and RDF feeds
- Comprehensive error handling and recovery
- Content normalization and validation
- Feed health monitoring
- Parallel processing capabilities
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

import feedparser
import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from culifeed.config.settings import get_settings
from culifeed.utils.logging import get_logger_for_component
from culifeed.utils.exceptions import (
    FeedError,
    FeedFetchError,
    ContentValidationError,
    handle_exception,
)
from culifeed.utils.validators import validate_url, validate_content_length


@dataclass
class ParsedArticle:
    """Represents a parsed RSS article with normalized fields."""

    title: str
    link: str
    summary: str
    content: Optional[str] = None
    published: Optional[datetime] = None
    updated: Optional[datetime] = None
    author: Optional[str] = None
    categories: List[str] = None
    guid: Optional[str] = None
    enclosures: List[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.categories is None:
            self.categories = []
        if self.enclosures is None:
            self.enclosures = []


@dataclass
class FeedMetadata:
    """Represents RSS feed metadata."""

    title: str
    link: str
    description: str
    language: Optional[str] = None
    updated: Optional[datetime] = None
    generator: Optional[str] = None
    categories: List[str] = None
    image: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.categories is None:
            self.categories = []


class FeedManager:
    """
    RSS Feed Manager with robust parsing and error handling.

    Features:
    - Multi-format support (RSS 2.0, RSS 1.0, Atom 1.0)
    - Automatic retry logic with exponential backoff
    - Content sanitization and validation
    - Feed health monitoring
    - Async and sync processing modes
    """

    def __init__(self):
        """Initialize feed manager with configuration."""
        self.settings = get_settings()
        self.logger = get_logger_for_component("feed_manager")

        # Configure requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set user agent and timeout
        self.session.headers.update(
            {
                "User-Agent": f"{self.settings.app_name}/{self.settings.version} (+{self.settings.user.timezone})",
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            }
        )

        # Track feed health
        self._feed_errors: Dict[str, int] = {}
        self._last_fetch_times: Dict[str, datetime] = {}

    def fetch_feed(self, feed_url: str) -> Tuple[FeedMetadata, List[ParsedArticle]]:
        """
        Fetch and parse RSS feed from URL.

        Args:
            feed_url: RSS feed URL to fetch

        Returns:
            Tuple of (feed_metadata, list_of_articles)

        Raises:
            FeedFetchError: If feed cannot be fetched or is invalid
            ContentValidationError: If content fails validation
        """
        try:
            # Validate URL
            if not validate_url(feed_url):
                raise FeedFetchError(f"Invalid feed URL: {feed_url}")

            # Check if feed has too many recent errors
            if self._should_skip_feed(feed_url):
                raise FeedFetchError(
                    f"Feed temporarily disabled due to errors: {feed_url}"
                )

            # Fetch feed content
            self.logger.info(f"Fetching RSS feed: {feed_url}")
            start_time = time.time()

            response = self.session.get(
                feed_url, timeout=self.settings.limits.request_timeout, stream=False
            )
            response.raise_for_status()

            fetch_time = time.time() - start_time
            self.logger.debug(
                f"Feed fetched in {fetch_time:.2f}s, size: {len(response.content)} bytes"
            )

            # Parse feed content
            parsed_feed = feedparser.parse(
                response.content, response_headers=dict(response.headers)
            )

            # Check for parsing errors (bozo detection)
            if parsed_feed.bozo:
                self.logger.warning(
                    f"Feed parsing warning for {feed_url}: {parsed_feed.bozo_exception}"
                )
                # Continue processing - many feeds have minor formatting issues

            # Validate feed structure
            if not hasattr(parsed_feed, "feed") or not hasattr(parsed_feed, "entries"):
                raise FeedFetchError(f"Invalid feed structure: {feed_url}")

            # Extract feed metadata
            feed_metadata = self._extract_feed_metadata(parsed_feed.feed, feed_url)

            # Extract articles
            articles = []
            for entry in parsed_feed.entries:
                try:
                    article = self._extract_article(entry, feed_url)
                    articles.append(article)
                except Exception as e:
                    self.logger.warning(f"Failed to parse article in {feed_url}: {e}")
                    continue

            # Update success tracking
            self._record_successful_fetch(feed_url)

            self.logger.info(
                f"Successfully parsed {len(articles)} articles from {feed_url}"
            )
            return feed_metadata, articles

        except requests.RequestException as e:
            error_msg = f"Failed to fetch feed {feed_url}: {str(e)}"
            self.logger.error(error_msg)
            self._record_feed_error(feed_url)
            raise FeedFetchError(error_msg) from e
        except FeedFetchError:
            # Re-raise FeedFetchError without modification
            raise
        except Exception as e:
            error_msg = f"Unexpected error processing feed {feed_url}: {str(e)}"
            self.logger.error(error_msg)
            self._record_feed_error(feed_url)
            raise FeedError(error_msg) from e

    async def fetch_feed_async(
        self, feed_url: str, session: aiohttp.ClientSession
    ) -> Tuple[FeedMetadata, List[ParsedArticle]]:
        """
        Async version of fetch_feed for concurrent processing.

        Args:
            feed_url: RSS feed URL to fetch
            session: aiohttp session for making requests

        Returns:
            Tuple of (feed_metadata, list_of_articles)
        """
        try:
            # Validate URL
            if not validate_url(feed_url):
                raise FeedFetchError(f"Invalid feed URL: {feed_url}")

            # Check feed error status
            if self._should_skip_feed(feed_url):
                raise FeedFetchError(f"Feed temporarily disabled: {feed_url}")

            self.logger.info(f"Fetching RSS feed (async): {feed_url}")
            start_time = time.time()

            # Fetch with timeout
            timeout = aiohttp.ClientTimeout(total=self.settings.limits.request_timeout)
            async with session.get(feed_url, timeout=timeout) as response:
                response.raise_for_status()
                content = await response.read()

                fetch_time = time.time() - start_time
                self.logger.debug(
                    f"Feed fetched (async) in {fetch_time:.2f}s, size: {len(content)} bytes"
                )

                # Parse feed content
                parsed_feed = feedparser.parse(
                    content, response_headers=dict(response.headers)
                )

                # Check for parsing errors
                if parsed_feed.bozo:
                    self.logger.warning(
                        f"Feed parsing warning for {feed_url}: {parsed_feed.bozo_exception}"
                    )

                # Validate and extract
                if not hasattr(parsed_feed, "feed") or not hasattr(
                    parsed_feed, "entries"
                ):
                    raise FeedFetchError(f"Invalid feed structure: {feed_url}")

                feed_metadata = self._extract_feed_metadata(parsed_feed.feed, feed_url)

                articles = []
                for entry in parsed_feed.entries:
                    try:
                        article = self._extract_article(entry, feed_url)
                        articles.append(article)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to parse article in {feed_url}: {e}"
                        )
                        continue

                self._record_successful_fetch(feed_url)
                self.logger.info(
                    f"Successfully parsed {len(articles)} articles from {feed_url} (async)"
                )

                return feed_metadata, articles

        except aiohttp.ClientError as e:
            error_msg = f"Failed to fetch feed {feed_url}: {str(e)}"
            self.logger.error(error_msg)
            self._record_feed_error(feed_url)
            raise FeedFetchError(error_msg) from e
        except FeedFetchError:
            # Re-raise FeedFetchError without modification
            raise
        except Exception as e:
            error_msg = f"Unexpected error processing feed {feed_url}: {str(e)}"
            self.logger.error(error_msg)
            self._record_feed_error(feed_url)
            raise FeedError(error_msg) from e

    async def fetch_multiple_feeds(
        self, feed_urls: List[str]
    ) -> Dict[str, Tuple[FeedMetadata, List[ParsedArticle]]]:
        """
        Fetch multiple RSS feeds concurrently.

        Args:
            feed_urls: List of RSS feed URLs

        Returns:
            Dictionary mapping feed URLs to (metadata, articles) tuples
        """
        results = {}
        failed_feeds = []

        # Create connector with limits
        connector = aiohttp.TCPConnector(
            limit=self.settings.processing.parallel_feeds,
            limit_per_host=2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(total=self.settings.limits.request_timeout)
        headers = {
            "User-Agent": f"{self.settings.app_name}/{self.settings.version}",
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
        }

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:

            # Create tasks for all feeds
            tasks = []
            for feed_url in feed_urls:
                task = asyncio.create_task(
                    self._fetch_with_error_handling(feed_url, session),
                    name=f"fetch_{urlparse(feed_url).netloc}",
                )
                tasks.append((feed_url, task))

            # Wait for all tasks to complete
            self.logger.info(f"Fetching {len(feed_urls)} RSS feeds concurrently")

            for feed_url, task in tasks:
                try:
                    result = await task
                    if result:
                        results[feed_url] = result
                except Exception as e:
                    self.logger.error(f"Failed to fetch feed {feed_url}: {e}")
                    failed_feeds.append(feed_url)

        self.logger.info(f"Successfully fetched {len(results)}/{len(feed_urls)} feeds")
        if failed_feeds:
            self.logger.warning(f"Failed feeds: {failed_feeds}")

        return results

    async def _fetch_with_error_handling(
        self, feed_url: str, session: aiohttp.ClientSession
    ) -> Optional[Tuple[FeedMetadata, List[ParsedArticle]]]:
        """Helper method to fetch feed with comprehensive error handling."""
        try:
            return await self.fetch_feed_async(feed_url, session)
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {e}")
            return None

    def _extract_feed_metadata(self, feed_data: Any, feed_url: str) -> FeedMetadata:
        """Extract and normalize feed metadata from parsed feed."""
        try:
            # Required fields with fallbacks
            title = getattr(feed_data, "title", "Unknown Feed")
            link = getattr(feed_data, "link", feed_url)
            description = getattr(feed_data, "description", "") or getattr(
                feed_data, "subtitle", ""
            )

            # Optional fields
            language = getattr(feed_data, "language", None)
            generator = getattr(feed_data, "generator", None)

            # Handle dates
            updated = None
            if hasattr(feed_data, "updated_parsed") and feed_data.updated_parsed:
                try:
                    updated = datetime(
                        *feed_data.updated_parsed[:6], tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid updated date in feed: {feed_url}")

            # Extract categories
            categories = []
            if hasattr(feed_data, "categories"):
                categories = [
                    cat.get("term", "") if isinstance(cat, dict) else str(cat)
                    for cat in feed_data.categories
                ]
            elif hasattr(feed_data, "tags"):
                categories = [
                    tag.get("term", "") if isinstance(tag, dict) else str(tag)
                    for tag in feed_data.tags
                ]

            # Extract image
            image = None
            if hasattr(feed_data, "image") and isinstance(feed_data.image, dict):
                image = {
                    "url": feed_data.image.get("href", ""),
                    "title": feed_data.image.get("title", ""),
                    "link": feed_data.image.get("link", ""),
                    "width": feed_data.image.get("width"),
                    "height": feed_data.image.get("height"),
                }

            return FeedMetadata(
                title=title.strip(),
                link=link,
                description=description.strip(),
                language=language,
                updated=updated,
                generator=generator,
                categories=[cat.strip() for cat in categories if cat.strip()],
                image=image,
            )

        except Exception as e:
            self.logger.warning(f"Error extracting feed metadata from {feed_url}: {e}")
            return FeedMetadata(title="Unknown Feed", link=feed_url, description="")

    def _extract_article(self, entry_data: Any, feed_url: str) -> ParsedArticle:
        """Extract and normalize article data from feed entry."""
        # Required fields
        title = getattr(entry_data, "title", "Untitled")
        link = getattr(entry_data, "link", "")

        # Summary/description (try multiple fields)
        summary = ""
        for field in ["summary", "description", "subtitle"]:
            if hasattr(entry_data, field):
                summary = getattr(entry_data, field, "")
                break

        # Content (full text)
        content = None
        if hasattr(entry_data, "content") and entry_data.content:
            # content is usually a list of dictionaries
            if isinstance(entry_data.content, list) and entry_data.content:
                content = entry_data.content[0].get("value", "")
        elif hasattr(entry_data, "description"):
            # Sometimes full content is in description
            desc = getattr(entry_data, "description", "")
            if len(desc) > len(summary):
                content = desc

        # Author
        author = None
        if hasattr(entry_data, "author"):
            author = entry_data.author
        elif hasattr(entry_data, "author_detail") and isinstance(
            entry_data.author_detail, dict
        ):
            name = entry_data.author_detail.get("name", "")
            email = entry_data.author_detail.get("email", "")
            author = f"{name} ({email})" if name and email else name or email

        # Dates
        published = None
        updated = None

        if hasattr(entry_data, "published_parsed") and entry_data.published_parsed:
            try:
                published = datetime(
                    *entry_data.published_parsed[:6], tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                pass

        if hasattr(entry_data, "updated_parsed") and entry_data.updated_parsed:
            try:
                updated = datetime(*entry_data.updated_parsed[:6], tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        # Categories/tags
        categories = []
        if hasattr(entry_data, "tags") and entry_data.tags:
            try:
                # Ensure tags is iterable
                tags_iter = (
                    entry_data.tags if hasattr(entry_data.tags, "__iter__") else []
                )
                categories = [
                    tag.get("term", "") if isinstance(tag, dict) else str(tag)
                    for tag in tags_iter
                ]
            except (TypeError, AttributeError):
                # Handle case where tags is not iterable or accessible
                categories = []
        elif hasattr(entry_data, "category"):
            categories = [entry_data.category]

        # GUID
        guid = getattr(entry_data, "id", None) or getattr(entry_data, "guid", None)

        # Enclosures (media)
        enclosures = []
        if hasattr(entry_data, "enclosures") and entry_data.enclosures:
            try:
                # Ensure enclosures is iterable
                enclosures_iter = (
                    entry_data.enclosures
                    if hasattr(entry_data.enclosures, "__iter__")
                    else []
                )
                for enc in enclosures_iter:
                    if isinstance(enc, dict):
                        enclosures.append(
                            {
                                "url": enc.get("href", ""),
                                "type": enc.get("type", ""),
                                "length": enc.get("length", 0),
                            }
                        )
            except (TypeError, AttributeError):
                # Handle case where enclosures is not iterable or accessible
                enclosures = []

        # Validate required fields
        if not title.strip():
            raise ContentValidationError("Article title is empty")
        if not link.strip():
            raise ContentValidationError("Article link is empty")

        # Validate content length
        if summary and not validate_content_length(
            summary, self.settings.processing.max_content_length
        ):
            self.logger.warning(
                f"Article summary too long, truncating: {len(summary)} chars"
            )
            summary = summary[: self.settings.processing.max_content_length] + "..."

        if content and not validate_content_length(
            content, self.settings.processing.max_content_length * 2
        ):
            self.logger.warning(
                f"Article content too long, truncating: {len(content)} chars"
            )
            content = content[: self.settings.processing.max_content_length * 2] + "..."

        return ParsedArticle(
            title=title.strip(),
            link=link.strip(),
            summary=summary.strip(),
            content=content.strip() if content else None,
            published=published,
            updated=updated,
            author=author.strip() if author else None,
            categories=[cat.strip() for cat in categories if cat.strip()],
            guid=guid,
            enclosures=enclosures,
        )

    def _should_skip_feed(self, feed_url: str) -> bool:
        """Check if feed should be skipped due to too many errors."""
        error_count = self._feed_errors.get(feed_url, 0)
        max_errors = self.settings.limits.max_feed_errors

        if error_count >= max_errors:
            self.logger.warning(f"Feed {feed_url} disabled due to {error_count} errors")
            return True

        return False

    def _record_feed_error(self, feed_url: str) -> None:
        """Record an error for a feed URL."""
        current_errors = self._feed_errors.get(feed_url, 0)
        self._feed_errors[feed_url] = current_errors + 1

        if current_errors + 1 >= self.settings.limits.max_feed_errors:
            self.logger.error(
                f"Feed {feed_url} reached maximum error limit ({self.settings.limits.max_feed_errors})"
            )

    def _record_successful_fetch(self, feed_url: str) -> None:
        """Record successful fetch and reset error count."""
        if feed_url in self._feed_errors:
            del self._feed_errors[feed_url]

        self._last_fetch_times[feed_url] = datetime.now(timezone.utc)

    def get_feed_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all tracked feeds."""
        status = {}

        for feed_url in set(
            list(self._feed_errors.keys()) + list(self._last_fetch_times.keys())
        ):
            status[feed_url] = {
                "error_count": self._feed_errors.get(feed_url, 0),
                "last_fetch": self._last_fetch_times.get(feed_url),
                "is_disabled": self._should_skip_feed(feed_url),
            }

        return status

    def reset_feed_errors(self, feed_url: str) -> None:
        """Reset error count for a specific feed (manual recovery)."""
        if feed_url in self._feed_errors:
            del self._feed_errors[feed_url]
        self.logger.info(f"Reset error count for feed: {feed_url}")


# Convenience functions for quick access
def fetch_single_feed(feed_url: str) -> Tuple[FeedMetadata, List[ParsedArticle]]:
    """Quick function to fetch a single RSS feed."""
    manager = FeedManager()
    return manager.fetch_feed(feed_url)


async def fetch_feeds_batch(
    feed_urls: List[str],
) -> Dict[str, Tuple[FeedMetadata, List[ParsedArticle]]]:
    """Quick function to fetch multiple RSS feeds concurrently."""
    manager = FeedManager()
    return await manager.fetch_multiple_feeds(feed_urls)
