"""
Manual Processing Service
========================

Shared service for manual feed processing operations used by both CLI and Telegram bot.
Provides consistent functionality and eliminates code duplication.

Features:
- Single feed fetching and analysis
- Batch feed processing with async support
- Pipeline testing with comprehensive results
- Consistent result formatting
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..ingestion.feed_manager import FeedManager
from ..processing.feed_fetcher import FeedFetcher, FetchResult
from ..storage.feed_repository import FeedRepository
from ..storage.topic_repository import TopicRepository
from ..database.connection import DatabaseConnection
from ..utils.logging import get_logger_for_component
from ..utils.validators import URLValidator, ValidationError


@dataclass
class FeedFetchSummary:
    """Summary of a single feed fetch operation."""
    url: str
    success: bool
    title: Optional[str] = None
    description: Optional[str] = None
    article_count: int = 0
    error_message: Optional[str] = None
    sample_articles: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.sample_articles is None:
            self.sample_articles = []


@dataclass
class BatchProcessingSummary:
    """Summary of batch feed processing operation."""
    total_feeds: int
    successful_feeds: int
    failed_feeds: int
    total_articles: int
    feed_results: List[Dict[str, Any]]
    processing_time_seconds: float


@dataclass
class PipelineTestSummary:
    """Summary of pipeline testing operation."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[Dict[str, Any]]
    chat_id: str


class ManualProcessingService:
    """
    Shared service for manual feed processing operations.

    Used by both CLI and Telegram bot to provide consistent functionality.
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize the manual processing service.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.feed_manager = FeedManager()
        self.feed_repository = FeedRepository(db_connection)
        self.topic_repository = TopicRepository(db_connection)
        self.feed_fetcher = FeedFetcher(max_concurrent=3, timeout=30)
        self.logger = get_logger_for_component('manual_processing')

    async def fetch_single_feed(self, url: str) -> FeedFetchSummary:
        """Fetch and analyze a single RSS feed.

        Args:
            url: RSS feed URL to fetch

        Returns:
            FeedFetchSummary with results
        """
        try:
            # Validate URL
            URLValidator.validate_feed_url(url)

            self.logger.info(f"Fetching single feed: {url}")

            # Fetch the feed
            feed_metadata, articles = self.feed_manager.fetch_feed(url)

            if not feed_metadata or not articles:
                return FeedFetchSummary(
                    url=url,
                    success=False,
                    error_message="Failed to fetch or parse RSS feed"
                )

            # Extract sample articles
            sample_articles = []
            for article in articles[:3]:  # First 3 articles
                sample_articles.append({
                    'title': article.title or "Untitled",
                    'published': article.published.isoformat() if article.published else None,
                    'link': article.link,
                    'content_preview': (article.content[:150] + "...") if article.content and len(article.content) > 150 else article.content
                })

            return FeedFetchSummary(
                url=url,
                success=True,
                title=feed_metadata.title,
                description=feed_metadata.description,
                article_count=len(articles),
                sample_articles=sample_articles
            )

        except ValidationError as e:
            return FeedFetchSummary(
                url=url,
                success=False,
                error_message=f"Invalid URL: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f"Error fetching feed {url}: {e}")
            return FeedFetchSummary(
                url=url,
                success=False,
                error_message=str(e)
            )

    async def process_feeds_for_chat(self, chat_id: str) -> BatchProcessingSummary:
        """Process all active feeds for a specific chat.

        Args:
            chat_id: Chat ID to process feeds for

        Returns:
            BatchProcessingSummary with results
        """
        start_time = datetime.now()

        try:
            self.logger.info(f"Processing feeds for chat: {chat_id}")

            # Get active feeds for the chat
            feeds = self.feed_repository.get_feeds_for_chat(chat_id, active_only=True)

            if not feeds:
                return BatchProcessingSummary(
                    total_feeds=0,
                    successful_feeds=0,
                    failed_feeds=0,
                    total_articles=0,
                    feed_results=[],
                    processing_time_seconds=0.0
                )

            # Prepare URLs for batch processing
            feed_urls = [str(feed.url) for feed in feeds]

            # Process feeds and store articles
            successful = 0
            failed = 0
            total_articles = 0
            feed_results = []

            for i, feed in enumerate(feeds):
                feed_url = str(feed.url)
                feed_title = feed.title or "Unknown Feed"

                try:
                    # Fetch feed and parse articles
                    feed_metadata, articles = self.feed_manager.fetch_feed(feed_url)

                    if feed_metadata and articles:
                        # Store articles in database
                        self._store_articles(articles, feed_url)

                        successful += 1
                        article_count = len(articles)
                        total_articles += article_count

                        self.logger.info(f"Processed {feed_title}: {article_count} articles stored")

                        feed_results.append({
                            'title': feed_title,
                            'url': feed_url,
                            'success': True,
                            'article_count': article_count,
                            'error': None
                        })
                    else:
                        failed += 1
                        feed_results.append({
                            'title': feed_title,
                            'url': feed_url,
                            'success': False,
                            'article_count': 0,
                            'error': "Failed to fetch or parse feed"
                        })

                except Exception as e:
                    failed += 1
                    error_msg = str(e)
                    self.logger.error(f"Error processing feed {feed_url}: {error_msg}")
                    feed_results.append({
                        'title': feed_title,
                        'url': feed_url,
                        'success': False,
                        'article_count': 0,
                        'error': error_msg
                    })

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return BatchProcessingSummary(
                total_feeds=len(feeds),
                successful_feeds=successful,
                failed_feeds=failed,
                total_articles=total_articles,
                feed_results=feed_results,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            self.logger.error(f"Error processing feeds for chat {chat_id}: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return BatchProcessingSummary(
                total_feeds=0,
                successful_feeds=0,
                failed_feeds=1,
                total_articles=0,
                feed_results=[{
                    'title': 'Processing Error',
                    'url': 'N/A',
                    'success': False,
                    'article_count': 0,
                    'error': str(e)
                }],
                processing_time_seconds=processing_time
            )

    async def process_default_test_feeds(self) -> BatchProcessingSummary:
        """Process default test feeds for general testing.

        Returns:
            BatchProcessingSummary with results
        """
        test_feeds = [
            "https://aws.amazon.com/blogs/compute/feed/",
            "https://blog.cloudflare.com/rss/",
            "https://kubernetes.io/feed.xml"
        ]

        start_time = datetime.now()

        try:
            self.logger.info(f"Processing {len(test_feeds)} default test feeds")

            # Process feeds in batch
            results = await self.feed_fetcher.fetch_feeds_batch(test_feeds)

            # Analyze results
            successful = 0
            failed = 0
            total_articles = 0
            feed_results = []

            for i, result in enumerate(results):
                feed_url = test_feeds[i] if i < len(test_feeds) else "Unknown URL"

                if result and result.success:
                    successful += 1
                    total_articles += result.article_count
                    feed_results.append({
                        'title': f"Test Feed {i+1}",
                        'url': feed_url,
                        'success': True,
                        'article_count': result.article_count,
                        'error': None
                    })
                else:
                    failed += 1
                    error_msg = result.error if result else "No result returned"
                    feed_results.append({
                        'title': f"Test Feed {i+1}",
                        'url': feed_url,
                        'success': False,
                        'article_count': 0,
                        'error': error_msg
                    })

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return BatchProcessingSummary(
                total_feeds=len(test_feeds),
                successful_feeds=successful,
                failed_feeds=failed,
                total_articles=total_articles,
                feed_results=feed_results,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            self.logger.error(f"Error processing default test feeds: {e}")
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            return BatchProcessingSummary(
                total_feeds=len(test_feeds),
                successful_feeds=0,
                failed_feeds=len(test_feeds),
                total_articles=0,
                feed_results=[{
                    'title': 'Processing Error',
                    'url': 'N/A',
                    'success': False,
                    'article_count': 0,
                    'error': str(e)
                }],
                processing_time_seconds=processing_time
            )

    async def run_pipeline_tests(self, chat_id: str = "test_chat") -> PipelineTestSummary:
        """Run comprehensive pipeline tests.

        Args:
            chat_id: Chat ID for testing

        Returns:
            PipelineTestSummary with results
        """
        try:
            self.logger.info(f"Running pipeline tests for chat: {chat_id}")

            # Import test framework
            try:
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                sys.path.insert(0, str(project_root))

                from tests.integration.test_feed_processing import FeedProcessingTester
                tester = FeedProcessingTester()
            except ImportError:
                return PipelineTestSummary(
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    test_results=[{
                        'name': 'Test Framework Import',
                        'success': False,
                        'details': 'Test framework not available'
                    }],
                    chat_id=chat_id
                )

            # Test feeds
            test_feeds = [
                "https://aws.amazon.com/blogs/compute/feed/",
                "https://blog.cloudflare.com/rss/"
            ]

            test_results = []
            passed_tests = 0
            total_tests = 0

            # Test 1: Single feed processing
            total_tests += 1
            try:
                if tester.test_single_feed(test_feeds[0], chat_id):
                    passed_tests += 1
                    test_results.append({
                        'name': 'Single Feed Processing',
                        'success': True,
                        'details': f'Successfully processed {test_feeds[0]}'
                    })
                else:
                    test_results.append({
                        'name': 'Single Feed Processing',
                        'success': False,
                        'details': 'Feed processing failed'
                    })
            except Exception as e:
                test_results.append({
                    'name': 'Single Feed Processing',
                    'success': False,
                    'details': f'Exception: {str(e)}'
                })

            # Test 2: Async batch processing
            total_tests += 1
            try:
                if await tester.test_feed_fetcher(test_feeds):
                    passed_tests += 1
                    test_results.append({
                        'name': 'Async Batch Processing',
                        'success': True,
                        'details': f'Successfully processed {len(test_feeds)} feeds'
                    })
                else:
                    test_results.append({
                        'name': 'Async Batch Processing',
                        'success': False,
                        'details': 'Batch processing failed'
                    })
            except Exception as e:
                test_results.append({
                    'name': 'Async Batch Processing',
                    'success': False,
                    'details': f'Exception: {str(e)}'
                })

            # Test 3: Database operations
            total_tests += 1
            try:
                if tester.test_database_operations(chat_id):
                    passed_tests += 1
                    test_results.append({
                        'name': 'Database Operations',
                        'success': True,
                        'details': 'CRUD operations successful'
                    })
                else:
                    test_results.append({
                        'name': 'Database Operations',
                        'success': False,
                        'details': 'Database operations failed'
                    })
            except Exception as e:
                test_results.append({
                    'name': 'Database Operations',
                    'success': False,
                    'details': f'Exception: {str(e)}'
                })

            return PipelineTestSummary(
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=total_tests - passed_tests,
                test_results=test_results,
                chat_id=chat_id
            )

        except Exception as e:
            self.logger.error(f"Error running pipeline tests: {e}")
            return PipelineTestSummary(
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                test_results=[{
                    'name': 'Pipeline Test Setup',
                    'success': False,
                    'details': f'Setup error: {str(e)}'
                }],
                chat_id=chat_id
            )

    def _store_articles(self, articles: List, feed_url: str) -> None:
        """Store articles in database.

        Args:
            articles: List of ParsedArticle objects to store
            feed_url: Source feed URL
        """
        if not articles:
            return

        try:
            import hashlib
            import uuid
            from datetime import datetime, timezone

            with self.db.get_connection() as conn:
                for article in articles:
                    # Generate unique ID and content hash
                    article_id = str(uuid.uuid4())
                    content_text = article.content or article.summary or ""
                    content_hash = hashlib.md5(content_text.encode()).hexdigest()

                    # Insert or replace article
                    conn.execute("""
                        INSERT OR REPLACE INTO articles
                        (id, title, url, content, published_at, source_feed, content_hash, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        article_id,
                        article.title,
                        article.link,
                        content_text,
                        article.published,
                        feed_url,
                        content_hash,
                        datetime.now(timezone.utc)
                    ))
                conn.commit()
                self.logger.info(f"Stored {len(articles)} articles from {feed_url}")

        except Exception as e:
            self.logger.error(f"Error storing articles from {feed_url}: {e}")
            raise