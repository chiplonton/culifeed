"""
Feed Management Utilities
========================

Feed validation, health monitoring, error tracking, and automatic cleanup
for RSS feed sources.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

from ..database.models import Feed, Article
from ..database.connection import DatabaseConnection
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import FeedManagementError, ErrorCode
from ..utils.validators import URLValidator
from .feed_fetcher import FeedFetcher, FetchResult


@dataclass
class FeedHealthStatus:
    """Feed health and status information."""
    feed: Feed
    is_healthy: bool
    should_disable: bool
    consecutive_errors: int
    last_success_age_hours: Optional[float]
    status_message: str
    
    @property
    def health_score(self) -> float:
        """Calculate health score (0.0 to 1.0)."""
        if self.consecutive_errors == 0:
            return 1.0
        
        # Exponential decay based on error count
        return max(0.0, 1.0 - (self.consecutive_errors / 10.0))


@dataclass
class FeedStats:
    """Feed statistics and metrics."""
    total_feeds: int
    active_feeds: int
    healthy_feeds: int
    error_feeds: int
    disabled_feeds: int
    avg_articles_per_feed: float
    total_articles_fetched: int
    
    @property
    def health_percentage(self) -> float:
        """Percentage of feeds that are healthy."""
        if self.active_feeds == 0:
            return 0.0
        return (self.healthy_feeds / self.active_feeds) * 100


class FeedManager:
    """Feed lifecycle management and health monitoring."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """Initialize feed manager.
        
        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.logger = get_logger_for_component("feed_manager")
        self.settings = get_settings()
        self.fetcher = FeedFetcher()
    
    async def add_feed(self, chat_id: str, feed_url: str, title: str = None, description: str = None) -> Feed:
        """Add a new RSS feed for a channel.
        
        Args:
            chat_id: Channel chat ID
            feed_url: RSS feed URL
            title: Optional feed title
            description: Optional feed description
            
        Returns:
            Created Feed model
            
        Raises:
            FeedManagementError: If feed cannot be added
        """
        try:
            # Validate URL
            validated_url = URLValidator.validate_feed_url(feed_url)
            
            # Check if feed already exists for this channel
            with self.db.get_connection() as conn:
                existing = conn.execute(
                    "SELECT id FROM feeds WHERE chat_id = ? AND url = ?",
                    (chat_id, str(validated_url))
                ).fetchone()
                
                if existing:
                    raise FeedManagementError(
                        f"Feed already exists for channel {chat_id}",
                        error_code=ErrorCode.DUPLICATE_RESOURCE
                    )
            
            # Test feed accessibility
            test_result = await self._test_feed_access(validated_url)
            if not test_result.success:
                raise FeedManagementError(
                    f"Feed is not accessible: {test_result.error}",
                    error_code=ErrorCode.EXTERNAL_SERVICE_ERROR
                )
            
            # Extract metadata from test fetch
            if not title and test_result.articles:
                # Try to extract feed title from first fetch
                title = self._extract_feed_title(test_result)
            
            # Create feed record
            feed = Feed(
                chat_id=chat_id,
                url=validated_url,
                title=title or "Untitled Feed",
                description=description,
                last_fetched_at=test_result.fetch_time,
                last_success_at=test_result.fetch_time if test_result.success else None,
                error_count=0 if test_result.success else 1,
                active=True
            )
            
            # Save to database
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO feeds (chat_id, url, title, description, last_fetched_at, 
                                     last_success_at, error_count, active, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feed.chat_id, str(feed.url), feed.title, feed.description,
                        feed.last_fetched_at, feed.last_success_at, feed.error_count,
                        feed.active, feed.created_at
                    )
                )
                feed.id = cursor.lastrowid
                conn.commit()
            
            self.logger.info(
                f"Added feed for channel {chat_id}: {feed.title} ({validated_url})",
                extra={'feed_id': feed.id, 'chat_id': chat_id}
            )
            
            return feed
            
        except Exception as e:
            if isinstance(e, FeedManagementError):
                raise
            raise FeedManagementError(
                f"Failed to add feed: {e}",
                error_code=ErrorCode.DATABASE_ERROR
            )
    
    async def _test_feed_access(self, feed_url: str) -> FetchResult:
        """Test if feed is accessible and parseable.
        
        Args:
            feed_url: Feed URL to test
            
        Returns:
            FetchResult with test results
        """
        try:
            async with self.fetcher.get_session() as session:
                result = await self.fetcher.fetch_feed(feed_url, session)
                return result
        except Exception as e:
            return FetchResult(
                feed_url=feed_url,
                success=False,
                error=f"Test access failed: {e}"
            )
    
    def _extract_feed_title(self, fetch_result: FetchResult) -> Optional[str]:
        """Extract feed title from fetch result.
        
        Args:
            fetch_result: Feed fetch result with articles
            
        Returns:
            Extracted feed title or None
        """
        # This is a simplified extraction - in practice you'd parse the feed metadata
        # For now, we'll use the domain name as a fallback
        try:
            from urllib.parse import urlparse
            parsed = urlparse(fetch_result.feed_url)
            return f"RSS Feed ({parsed.netloc})"
        except:
            return None
    
    def get_feeds_for_channel(self, chat_id: str, active_only: bool = True) -> List[Feed]:
        """Get all feeds for a channel.
        
        Args:
            chat_id: Channel chat ID
            active_only: Only return active feeds
            
        Returns:
            List of Feed models
        """
        with self.db.get_connection() as conn:
            query = "SELECT * FROM feeds WHERE chat_id = ?"
            params = [chat_id]
            
            if active_only:
                query += " AND active = ?"
                params.append(True)
            
            query += " ORDER BY created_at DESC"
            
            rows = conn.execute(query, params).fetchall()
            
            feeds = []
            for row in rows:
                feed_data = dict(row)
                feeds.append(Feed(**feed_data))
            
            return feeds
    
    def update_feed_status(self, feed_id: int, fetch_result: FetchResult) -> None:
        """Update feed status after fetch attempt.
        
        Args:
            feed_id: Feed ID to update
            fetch_result: Result of fetch operation
        """
        with self.db.get_connection() as conn:
            if fetch_result.success:
                # Reset error count on success
                conn.execute(
                    """
                    UPDATE feeds 
                    SET last_fetched_at = ?, last_success_at = ?, error_count = 0
                    WHERE id = ?
                    """,
                    (fetch_result.fetch_time, fetch_result.fetch_time, feed_id)
                )
            else:
                # Increment error count on failure
                conn.execute(
                    """
                    UPDATE feeds 
                    SET last_fetched_at = ?, error_count = error_count + 1
                    WHERE id = ?
                    """,
                    (fetch_result.fetch_time, feed_id)
                )
                
                # Check if feed should be auto-disabled
                feed_row = conn.execute("SELECT error_count FROM feeds WHERE id = ?", (feed_id,)).fetchone()
                if feed_row and feed_row['error_count'] >= self.settings.limits.max_feed_errors:
                    self._disable_feed(conn, feed_id, "Too many consecutive errors")
            
            conn.commit()
    
    def _disable_feed(self, conn, feed_id: int, reason: str) -> None:
        """Disable a feed due to errors.
        
        Args:
            conn: Database connection
            feed_id: Feed ID to disable
            reason: Reason for disabling
        """
        conn.execute(
            "UPDATE feeds SET active = ? WHERE id = ?",
            (False, feed_id)
        )
        
        self.logger.warning(
            f"Disabled feed {feed_id}: {reason}",
            extra={'feed_id': feed_id, 'disable_reason': reason}
        )
    
    def get_feed_health_status(self, feed_id: int) -> Optional[FeedHealthStatus]:
        """Get health status for a specific feed.
        
        Args:
            feed_id: Feed ID to check
            
        Returns:
            FeedHealthStatus or None if feed not found
        """
        with self.db.get_connection() as conn:
            row = conn.execute("SELECT * FROM feeds WHERE id = ?", (feed_id,)).fetchone()
            if not row:
                return None
            
            feed = Feed(**dict(row))
            
            # Calculate age of last success
            last_success_age_hours = None
            if feed.last_success_at:
                age_delta = datetime.now(timezone.utc) - feed.last_success_at
                last_success_age_hours = age_delta.total_seconds() / 3600
            
            # Determine health status
            is_healthy = feed.is_healthy()
            should_disable = feed.should_disable()
            
            # Generate status message
            if not feed.active:
                status_message = "Feed is disabled"
            elif should_disable:
                status_message = f"Feed failing: {feed.error_count} consecutive errors"
            elif not is_healthy:
                status_message = f"Feed unstable: {feed.error_count} recent errors"
            elif last_success_age_hours and last_success_age_hours > 48:
                status_message = f"No updates in {last_success_age_hours:.1f} hours"
            else:
                status_message = "Feed is healthy"
            
            return FeedHealthStatus(
                feed=feed,
                is_healthy=is_healthy,
                should_disable=should_disable,
                consecutive_errors=feed.error_count,
                last_success_age_hours=last_success_age_hours,
                status_message=status_message
            )
    
    def get_all_feed_health(self, chat_id: str = None) -> List[FeedHealthStatus]:
        """Get health status for all feeds.
        
        Args:
            chat_id: Optional channel ID to filter feeds
            
        Returns:
            List of FeedHealthStatus objects
        """
        with self.db.get_connection() as conn:
            if chat_id:
                rows = conn.execute("SELECT * FROM feeds WHERE chat_id = ?", (chat_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM feeds").fetchall()
            
            health_statuses = []
            for row in rows:
                feed = Feed(**dict(row))
                
                # Calculate health status (reuse logic from single feed method)
                status = self.get_feed_health_status(feed.id)
                if status:
                    health_statuses.append(status)
            
            return health_statuses
    
    def cleanup_old_articles(self, days_to_keep: int = None) -> int:
        """Clean up old articles from database.
        
        Args:
            days_to_keep: Days of articles to keep (default from config)
            
        Returns:
            Number of articles cleaned up
        """
        if days_to_keep is None:
            days_to_keep = self.settings.database.cleanup_days
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        
        with self.db.get_connection() as conn:
            # First delete processing results
            cursor = conn.execute(
                "DELETE FROM processing_results WHERE processed_at < ?",
                (cutoff_date,)
            )
            results_deleted = cursor.rowcount
            
            # Then delete old articles
            cursor = conn.execute(
                "DELETE FROM articles WHERE created_at < ?",
                (cutoff_date,)
            )
            articles_deleted = cursor.rowcount
            
            conn.commit()
        
        total_deleted = articles_deleted + results_deleted
        
        if total_deleted > 0:
            self.logger.info(
                f"Cleanup complete: deleted {articles_deleted} articles and "
                f"{results_deleted} processing results older than {days_to_keep} days"
            )
        
        return total_deleted
    
    def get_feed_statistics(self, chat_id: str = None) -> FeedStats:
        """Get feed statistics and metrics.
        
        Args:
            chat_id: Optional channel ID to filter statistics
            
        Returns:
            FeedStats with comprehensive metrics
        """
        with self.db.get_connection() as conn:
            # Base query conditions
            if chat_id:
                where_clause = "WHERE chat_id = ?"
                params = [chat_id]
            else:
                where_clause = ""
                params = []
            
            # Get feed counts
            feed_stats = conn.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN active THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN active AND error_count < 5 THEN 1 ELSE 0 END) as healthy,
                    SUM(CASE WHEN active AND error_count >= 5 THEN 1 ELSE 0 END) as error,
                    SUM(CASE WHEN NOT active THEN 1 ELSE 0 END) as disabled
                FROM feeds
                {where_clause}
            """, params).fetchone()
            
            # Get article counts
            if chat_id:
                article_count = conn.execute("""
                    SELECT COUNT(*) as total
                    FROM articles a 
                    JOIN feeds f ON a.source_feed = f.url 
                    WHERE f.chat_id = ?
                """, (chat_id,)).fetchone()['total']
            else:
                article_count = conn.execute("SELECT COUNT(*) as total FROM articles").fetchone()['total']
            
            # Calculate average articles per feed
            active_feeds = feed_stats['active']
            avg_articles = article_count / active_feeds if active_feeds > 0 else 0.0
            
            return FeedStats(
                total_feeds=feed_stats['total'],
                active_feeds=active_feeds,
                healthy_feeds=feed_stats['healthy'],
                error_feeds=feed_stats['error'],
                disabled_feeds=feed_stats['disabled'],
                avg_articles_per_feed=avg_articles,
                total_articles_fetched=article_count
            )