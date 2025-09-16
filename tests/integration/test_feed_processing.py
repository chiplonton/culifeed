#!/usr/bin/env python3
"""
Feed Processing Test Suite
=========================

Comprehensive testing for RSS feed fetching, processing, and AI analysis pipeline.
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from culifeed.config.settings import get_settings
from culifeed.database.connection import get_db_manager
from culifeed.storage.feed_repository import FeedRepository
from culifeed.storage.topic_repository import TopicRepository
from culifeed.storage.article_repository import ArticleRepository
from culifeed.ingestion.feed_manager import FeedManager
from culifeed.processing.feed_fetcher import FeedFetcher
from culifeed.database.models import Feed, Topic
from culifeed.utils.logging import setup_logger


class FeedProcessingTester:
    """Test suite for feed processing pipeline."""

    def __init__(self):
        """Initialize the tester."""
        self.settings = get_settings()
        self.db = get_db_manager(self.settings.database.path)

        # Setup logging first
        setup_logger(
            name="feed_processing_test",
            level="DEBUG",
            console=True
        )
        self.logger = logging.getLogger("feed_processing_test")

        # Initialize repositories
        self.feed_repo = FeedRepository(self.db)
        self.topic_repo = TopicRepository(self.db)
        self.article_repo = ArticleRepository(self.db)
        
        # Create test channel if it doesn't exist
        self._ensure_test_channel("test_chat")

        # Initialize processing components
        self.feed_manager = FeedManager()
        self.feed_fetcher = FeedFetcher(max_concurrent=3, timeout=30)

    def _ensure_test_channel(self, chat_id: str):
        """Ensure test channel exists in database."""
        try:
            with self.db.get_connection() as conn:
                # Check if channel exists
                existing = conn.execute(
                    "SELECT * FROM channels WHERE chat_id = ?",
                    (chat_id,)
                ).fetchone()
                
                if not existing:
                    # Create test channel
                    conn.execute("""
                        INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chat_id,
                        "Test Channel",
                        "group",
                        True,
                        datetime.now(timezone.utc),
                        datetime.now(timezone.utc)
                    ))
                    conn.commit()
                    self.logger.info(f"Created test channel: {chat_id}")
        except Exception as e:
            self.logger.error(f"Failed to create test channel: {e}")

    def print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")

    def print_step(self, step: str):
        """Print a step description."""
        print(f"\n📋 {step}")

    def print_success(self, message: str):
        """Print a success message."""
        print(f"✅ {message}")

    def print_error(self, message: str):
        """Print an error message."""
        print(f"❌ {message}")

    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"⚠️ {message}")

    def test_single_feed(self, feed_url: str, chat_id: str = "test_chat") -> bool:
        """Test fetching and processing a single RSS feed.

        Args:
            feed_url: RSS feed URL to test
            chat_id: Chat ID for testing

        Returns:
            True if successful, False otherwise
        """
        self.print_header(f"Testing Single Feed: {feed_url}")

        try:
            self.print_step("1. Fetching RSS feed content...")

            # Test with FeedManager (RSS parsing)
            feed_metadata, articles = self.feed_manager.fetch_feed(feed_url)

            if not feed_metadata or not articles:
                self.print_error("Feed fetch returned empty results")
                return False

            self.print_success(f"Feed fetched successfully!")
            print(f"   📊 Articles found: {len(articles)}")
            print(f"   📰 Feed title: {feed_metadata.title or 'Unknown'}")
            print(f"   🔗 Feed description: {(feed_metadata.description or 'None')[:100]}...")

            # Show sample articles
            self.print_step("2. Sample articles from feed:")
            for i, article in enumerate(articles[:3], 1):
                print(f"   {i}. {article.title}")
                print(f"      📅 {article.published or 'No date'}")
                print(f"      🔗 {article.link}")
                print(f"      📝 {article.content[:100]}..." if article.content else "      📝 No content")
                print()

            return True

        except Exception as e:
            self.print_error(f"Feed test failed: {e}")
            return False

    async def test_feed_fetcher(self, feed_urls: List[str]) -> bool:
        """Test the async feed fetcher with multiple feeds.

        Args:
            feed_urls: List of RSS feed URLs

        Returns:
            True if successful, False otherwise
        """
        self.print_header("Testing Async Feed Fetcher")

        try:
            self.print_step(f"Fetching {len(feed_urls)} feeds concurrently...")

            results = await self.feed_fetcher.fetch_feeds_batch(feed_urls)

            print(f"📊 Results: {len(results)}/{len(feed_urls)} feeds processed")

            successful = 0
            failed = 0

            for i, result in enumerate(results):
                feed_url = feed_urls[i] if i < len(feed_urls) else "Unknown"

                if result and result.success:
                    successful += 1
                    self.print_success(f"Feed {i+1}: {result.article_count} articles")
                    print(f"   🔗 {feed_url}")
                else:
                    failed += 1
                    error_msg = result.error if result else "No result"
                    self.print_error(f"Feed {i+1}: {error_msg}")
                    print(f"   🔗 {feed_url}")

            print(f"\n📈 Summary: {successful} successful, {failed} failed")
            return successful > 0

        except Exception as e:
            self.print_error(f"Feed fetcher test failed: {e}")
            return False

    def test_database_operations(self, chat_id: str = "test_chat") -> bool:
        """Test database operations for feeds and topics.

        Args:
            chat_id: Chat ID for testing

        Returns:
            True if successful, False otherwise
        """
        self.print_header("Testing Database Operations")

        try:
            # Ensure test channel exists first (for foreign key constraint)
            self._ensure_test_channel(chat_id)

            # Test feed repository
            self.print_step("1. Testing feed repository...")

            # Create test feed with unique URL to avoid conflicts
            test_feed = Feed(
                chat_id=chat_id,
                url="https://feeds.example.com/test-feed.xml",
                title="Test Feed for Database Operations",
                description="Test feed for database operations testing",
                active=True,
                error_count=0,
                last_fetched_at=datetime.now(timezone.utc)
            )

            feed_id = self.feed_repo.create_feed(test_feed)
            if feed_id:
                self.print_success(f"Feed created with ID: {feed_id}")
            else:
                self.print_error("Failed to create feed")
                return False

            # Test feed retrieval
            retrieved_feed = self.feed_repo.get_feed_by_id(feed_id)
            if retrieved_feed:
                self.print_success(f"Feed retrieved: {retrieved_feed.title}")
            else:
                self.print_error("Failed to retrieve feed")
                return False

            # Test topic repository
            self.print_step("2. Testing topic repository...")

            test_topic = Topic(
                chat_id=chat_id,
                name="AWS Lambda",
                keywords=["lambda", "serverless", "aws functions"],
                confidence_threshold=0.8,
                active=True
            )

            topic_id = self.topic_repo.create_topic(test_topic)
            if topic_id:
                self.print_success(f"Topic created with ID: {topic_id}")
            else:
                self.print_error("Failed to create topic")
                return False

            # Clean up test data
            self.feed_repo.delete_feed(feed_id)
            self.topic_repo.delete_topic(topic_id)
            self.print_success("Test data cleaned up")

            return True

        except Exception as e:
            self.print_error(f"Database test failed: {e}")
            return False

    def test_full_pipeline(self, chat_id: str = "test_chat") -> bool:
        """Test the full processing pipeline end-to-end.

        Args:
            chat_id: Chat ID for testing

        Returns:
            True if successful, False otherwise
        """
        self.print_header("Testing Full Processing Pipeline")

        try:
            # Setup test data
            self.print_step("1. Setting up test data...")

            # Create test topic
            test_topic = Topic(
                chat_id=chat_id,
                name="AWS Updates",
                keywords=["aws", "amazon", "cloud", "lambda", "ec2"],
                confidence_threshold=0.7,
                active=True
            )

            topic_id = self.topic_repo.create_topic(test_topic)
            if not topic_id:
                self.print_error("Failed to create test topic")
                return False

            # Create test feed with unique URL to avoid conflicts
            test_feed = Feed(
                chat_id=chat_id,
                url="https://feeds.example.com/pipeline-test-feed.xml",
                title="Pipeline Test Feed",
                active=True,
                error_count=0
            )

            feed_id = self.feed_repo.create_feed(test_feed)
            if not feed_id:
                self.print_error("Failed to create test feed")
                return False

            self.print_success("Test data created")

            # Fetch feed content
            self.print_step("2. Fetching feed content...")

            feeds = self.feed_repo.get_feeds_for_chat(chat_id)
            if not feeds:
                self.print_error("No feeds found")
                return False

            feed = feeds[0]
            feed_metadata, articles = self.feed_manager.fetch_feed(str(feed.url))

            if not feed_metadata or not articles:
                self.print_error("Failed to fetch feed content")
                return False

            self.print_success(f"Fetched {len(articles)} articles")

            # Process articles (basic)
            self.print_step("3. Processing articles...")

            topics = self.topic_repo.get_topics_for_chat(chat_id)
            if not topics:
                self.print_error("No topics found")
                return False

            topic = topics[0]
            matched_articles = []

            for article in articles[:5]:  # Process first 5 articles
                # Simple keyword matching (without AI)
                content_text = f"{article.title} {article.content or ''}".lower()
                keyword_matches = sum(1 for keyword in topic.keywords if keyword.lower() in content_text)

                if keyword_matches > 0:
                    matched_articles.append({
                        'article': article,
                        'matches': keyword_matches,
                        'topic': topic.name
                    })

            self.print_success(f"Found {len(matched_articles)} relevant articles")

            # Display results
            self.print_step("4. Results summary...")

            for i, match in enumerate(matched_articles, 1):
                article = match['article']
                print(f"   {i}. {article.title}")
                print(f"      🎯 Topic: {match['topic']}")
                print(f"      📊 Keyword matches: {match['matches']}")
                print(f"      📅 Published: {article.published or 'No date'}")
                print()

            # Cleanup
            self.feed_repo.delete_feed(feed_id)
            self.topic_repo.delete_topic(topic_id)

            return True

        except Exception as e:
            self.print_error(f"Full pipeline test failed: {e}")
            return False

    def show_feed_status(self, chat_id: Optional[str] = None) -> bool:
        """Show status of all feeds in the database.

        Args:
            chat_id: Optional chat ID to filter by

        Returns:
            True if successful, False otherwise
        """
        self.print_header("Feed Status Report")

        try:
            if chat_id:
                feeds = self.feed_repo.get_feeds_for_chat(chat_id, active_only=False)
                print(f"📊 Feeds for chat {chat_id}: {len(feeds)}")
            else:
                feeds = self.feed_repo.get_all_active_feeds()
                print(f"📊 Total active feeds: {len(feeds)}")

            if not feeds:
                self.print_warning("No feeds found in database")
                return True

            for i, feed in enumerate(feeds, 1):
                status = "🟢" if feed.active and feed.error_count == 0 else "🟡" if feed.error_count < 5 else "🔴"

                print(f"\n{status} {i}. {feed.title or 'Untitled Feed'}")
                print(f"   🔗 {feed.url}")
                print(f"   💬 Chat: {feed.chat_id}")
                print(f"   📅 Created: {feed.created_at}")
                print(f"   🔄 Last fetch: {feed.last_fetched_at or 'Never'}")
                print(f"   ✅ Last success: {feed.last_success_at or 'Never'}")
                print(f"   ❌ Errors: {feed.error_count}")
                print(f"   🟢 Active: {feed.active}")

            return True

        except Exception as e:
            self.print_error(f"Feed status check failed: {e}")
            return False


async def main():
    """Main test runner."""
    print("🔍 CuliFeed Feed Processing Test Suite")
    print("=" * 60)

    tester = FeedProcessingTester()

    # Test feeds to use
    test_feeds = [
        "https://aws.amazon.com/blogs/compute/feed/",
        "https://blog.cloudflare.com/rss/",
        "https://kubernetes.io/feed.xml",
    ]

    # Run tests
    tests_passed = 0
    total_tests = 0

    # 1. Test single feed
    total_tests += 1
    if tester.test_single_feed(test_feeds[0]):
        tests_passed += 1

    # 2. Test async feed fetcher
    total_tests += 1
    if await tester.test_feed_fetcher(test_feeds):
        tests_passed += 1

    # 3. Test database operations
    total_tests += 1
    if tester.test_database_operations():
        tests_passed += 1

    # 4. Test full pipeline
    total_tests += 1
    if tester.test_full_pipeline():
        tests_passed += 1

    # 5. Show feed status
    total_tests += 1
    if tester.show_feed_status():
        tests_passed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    if tests_passed == total_tests:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())