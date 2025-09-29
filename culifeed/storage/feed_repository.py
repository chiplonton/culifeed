"""
Feed Repository
===============

Repository pattern implementation for RSS feed data management.
Provides database abstraction layer for feed CRUD operations.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import json
import logging

from ..database.connection import DatabaseConnection
from ..database.models import Feed
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import DatabaseError, ErrorCode


class FeedRepository:
    """Repository for managing RSS feed data in the database."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize feed repository.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.logger = get_logger_for_component("feed_repository")
        self.settings = get_settings()

    def create_feed(self, feed: Feed) -> Optional[int]:
        """Create a new feed in the database.

        Args:
            feed: Feed object to create

        Returns:
            Feed ID if successful, None otherwise

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO feeds (
                        chat_id, url, title, description, 
                        last_fetched_at, last_success_at, error_count, active, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        feed.chat_id,
                        str(feed.url),  # Convert AnyHttpUrl to string
                        feed.title,
                        feed.description,
                        feed.last_fetched_at,
                        feed.last_success_at,
                        feed.error_count,
                        feed.active,
                        feed.created_at or datetime.now(timezone.utc),
                    ),
                )

                feed_id = cursor.lastrowid
                conn.commit()

                self.logger.info(
                    f"Created feed {feed_id} for chat {feed.chat_id}: {feed.url}"
                )
                return feed_id

        except Exception as e:
            self.logger.error(f"Failed to create feed: {e}")
            raise DatabaseError(
                f"Failed to create feed: {e}", error_code=ErrorCode.DATABASE_ERROR
            )

    def get_feed_by_id(self, feed_id: int) -> Optional[Feed]:
        """Get feed by ID.

        Args:
            feed_id: Feed ID

        Returns:
            Feed object if found, None otherwise
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM feeds WHERE id = ?", (feed_id,)
                ).fetchone()

                return self._row_to_feed(row) if row else None

        except Exception as e:
            self.logger.error(f"Failed to get feed {feed_id}: {e}")
            return None

    def get_feed_by_url(self, chat_id: str, url: str) -> Optional[Feed]:
        """Get feed by chat ID and URL.

        Args:
            chat_id: Chat ID
            url: Feed URL

        Returns:
            Feed object if found, None otherwise
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM feeds WHERE chat_id = ? AND url = ?",
                    (chat_id, str(url)),  # Ensure URL is string
                ).fetchone()

                return self._row_to_feed(row) if row else None

        except Exception as e:
            self.logger.error(
                f"Failed to get feed by URL {url} for chat {chat_id}: {e}"
            )
            return None

    def get_feeds_for_chat(self, chat_id: str, active_only: bool = True) -> List[Feed]:
        """Get all feeds for a chat.

        Args:
            chat_id: Chat ID
            active_only: If True, only return active feeds

        Returns:
            List of Feed objects
        """
        try:
            with self.db.get_connection() as conn:
                if active_only:
                    rows = conn.execute(
                        "SELECT * FROM feeds WHERE chat_id = ? AND active = ? ORDER BY created_at",
                        (chat_id, True),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM feeds WHERE chat_id = ? ORDER BY created_at",
                        (chat_id,),
                    ).fetchall()

                return [self._row_to_feed(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get feeds for chat {chat_id}: {e}")
            return []

    def get_all_active_feeds(self) -> List[Feed]:
        """Get all active feeds across all chats.

        Returns:
            List of active Feed objects
        """
        try:
            with self.db.get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM feeds WHERE active = ? ORDER BY chat_id, created_at",
                    (True,),
                ).fetchall()

                return [self._row_to_feed(row) for row in rows]

        except Exception as e:
            self.logger.error(f"Failed to get all active feeds: {e}")
            return []

    def update_feed(self, feed_id: int, **kwargs) -> bool:
        """Update feed fields.

        Args:
            feed_id: Feed ID
            **kwargs: Fields to update

        Returns:
            True if successful, False otherwise
        """
        if not kwargs:
            return True

        try:
            # Build dynamic update query
            fields = []
            values = []

            for field, value in kwargs.items():
                if field in [
                    "title",
                    "description",
                    "last_fetched_at",
                    "last_success_at",
                    "error_count",
                    "active",
                ]:
                    fields.append(f"{field} = ?")
                    values.append(value)

            if not fields:
                self.logger.warning(f"No valid fields to update for feed {feed_id}")
                return False

            values.append(feed_id)
            query = f"UPDATE feeds SET {', '.join(fields)} WHERE id = ?"

            with self.db.get_connection() as conn:
                cursor = conn.execute(query, values)
                conn.commit()

                if cursor.rowcount > 0:
                    self.logger.info(f"Updated feed {feed_id}")
                    return True
                else:
                    self.logger.warning(f"No feed found with ID {feed_id}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to update feed {feed_id}: {e}")
            return False

    def update_feed_success(self, feed_id: int) -> bool:
        """Mark feed as successfully fetched.

        Args:
            feed_id: Feed ID

        Returns:
            True if successful
        """
        return self.update_feed(
            feed_id,
            last_fetched_at=datetime.now(timezone.utc),
            last_success_at=datetime.now(timezone.utc),
            error_count=0,
        )

    def update_feed_error(
        self, feed_id: int, error_count: Optional[int] = None
    ) -> bool:
        """Record feed fetch error.

        Args:
            feed_id: Feed ID
            error_count: New error count (if None, increments current)

        Returns:
            True if successful
        """
        try:
            with self.db.get_connection() as conn:
                if error_count is None:
                    # Increment error count
                    conn.execute(
                        """
                        UPDATE feeds 
                        SET last_fetched_at = ?, error_count = error_count + 1
                        WHERE id = ?
                    """,
                        (datetime.now(timezone.utc), feed_id),
                    )
                else:
                    # Set specific error count
                    conn.execute(
                        """
                        UPDATE feeds 
                        SET last_fetched_at = ?, error_count = ?
                        WHERE id = ?
                    """,
                        (datetime.now(timezone.utc), error_count, feed_id),
                    )

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to update feed error for {feed_id}: {e}")
            return False

    def deactivate_feed(self, feed_id: int) -> bool:
        """Deactivate a feed.

        Args:
            feed_id: Feed ID

        Returns:
            True if successful
        """
        return self.update_feed(feed_id, active=False)

    def activate_feed(self, feed_id: int) -> bool:
        """Activate a feed.

        Args:
            feed_id: Feed ID

        Returns:
            True if successful
        """
        return self.update_feed(feed_id, active=True)

    def delete_feed(self, feed_id: int) -> bool:
        """Delete a feed.

        Args:
            feed_id: Feed ID

        Returns:
            True if successful
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("DELETE FROM feeds WHERE id = ?", (feed_id,))
                conn.commit()

                if cursor.rowcount > 0:
                    self.logger.info(f"Deleted feed {feed_id}")
                    return True
                else:
                    self.logger.warning(f"No feed found with ID {feed_id}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to delete feed {feed_id}: {e}")
            return False

    def delete_feeds_for_chat(self, chat_id: str) -> int:
        """Delete all feeds for a chat.

        Args:
            chat_id: Chat ID

        Returns:
            Number of feeds deleted
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute("DELETE FROM feeds WHERE chat_id = ?", (chat_id,))
                conn.commit()

                self.logger.info(f"Deleted {cursor.rowcount} feeds for chat {chat_id}")
                return cursor.rowcount

        except Exception as e:
            self.logger.error(f"Failed to delete feeds for chat {chat_id}: {e}")
            return 0

    def get_feed_statistics(self, chat_id: Optional[str] = None) -> Dict[str, Any]:
        """Get feed statistics.

        Args:
            chat_id: Optional chat ID to filter by

        Returns:
            Dictionary with statistics
        """
        try:
            with self.db.get_connection() as conn:
                if chat_id:
                    stats = conn.execute(
                        """
                        SELECT 
                            COUNT(*) as total_feeds,
                            COUNT(CASE WHEN active = 1 THEN 1 END) as active_feeds,
                            COUNT(CASE WHEN error_count > 0 THEN 1 END) as feeds_with_errors,
                            AVG(error_count) as avg_error_count,
                            MAX(last_success_at) as last_successful_fetch
                        FROM feeds 
                        WHERE chat_id = ?
                    """,
                        (chat_id,),
                    ).fetchone()
                else:
                    stats = conn.execute(
                        """
                        SELECT 
                            COUNT(*) as total_feeds,
                            COUNT(CASE WHEN active = 1 THEN 1 END) as active_feeds,
                            COUNT(CASE WHEN error_count > 0 THEN 1 END) as feeds_with_errors,
                            AVG(error_count) as avg_error_count,
                            MAX(last_success_at) as last_successful_fetch,
                            COUNT(DISTINCT chat_id) as total_chats
                        FROM feeds
                    """
                    ).fetchone()

                return dict(stats) if stats else {}

        except Exception as e:
            self.logger.error(f"Failed to get feed statistics: {e}")
            return {}

    def _row_to_feed(self, row) -> Feed:
        """Convert database row to Feed object.

        Args:
            row: Database row

        Returns:
            Feed object
        """
        return Feed(
            id=row["id"],
            chat_id=row["chat_id"],
            url=row["url"],
            title=row["title"],
            description=row["description"],
            last_fetched_at=row["last_fetched_at"],
            last_success_at=row["last_success_at"],
            error_count=row["error_count"],
            active=bool(row["active"]),
            created_at=row["created_at"],
        )
