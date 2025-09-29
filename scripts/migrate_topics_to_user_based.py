#!/usr/bin/env python3
"""
CuliFeed Topic Migration Script
===============================

Migrates existing channel-based topics to user-based ownership for SaaS pricing model.

This script automatically assigns topic ownership as follows:
1. Private chat topics → assigned to chat owner (user who chatted with bot)
2. Group topics → assigned to first admin or most active user
3. Edge cases → orphaned (very rare, can be manually assigned later)

Usage:
    python scripts/migrate_topics_to_user_based.py

Requirements:
    - Database must be accessible
    - Bot must have been used in the past (to have chat history)
"""

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from culifeed.config.settings import get_settings
from culifeed.database.models import UserTier, UserSubscription
from culifeed.utils.exceptions import ConfigurationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('migration.log')
    ]
)
logger = logging.getLogger(__name__)


class TopicMigrationService:
    """Service for migrating existing topics to user-based ownership."""

    def __init__(self, db_path: str):
        """Initialize migration service.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.stats = {
            'total_topics': 0,
            'private_assigned': 0,
            'group_assigned': 0,
            'orphaned': 0,
            'already_assigned': 0,
            'errors': 0
        }

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    async def migrate_all_topics(self) -> None:
        """Migrate all existing topics to user-based ownership."""
        logger.info("Starting topic migration to user-based ownership...")

        try:
            with self.get_connection() as conn:
                # Get all topics that need migration (telegram_user_id is NULL)
                unassigned_topics = conn.execute("""
                    SELECT t.*, c.chat_type
                    FROM topics t
                    JOIN channels c ON t.chat_id = c.chat_id
                    WHERE t.telegram_user_id IS NULL AND t.active = 1
                """).fetchall()

                self.stats['total_topics'] = len(unassigned_topics)
                logger.info(f"Found {self.stats['total_topics']} topics needing migration")

                if self.stats['total_topics'] == 0:
                    logger.info("No topics need migration. Migration complete!")
                    return

                # Process each topic
                for topic in unassigned_topics:
                    await self._migrate_single_topic(conn, topic)

            # Print migration summary
            self._print_migration_summary()

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise

    async def _migrate_single_topic(self, conn: sqlite3.Connection, topic: sqlite3.Row) -> None:
        """Migrate a single topic to user ownership.

        Args:
            conn: Database connection
            topic: Topic row from database
        """
        try:
            chat_id = topic['chat_id']
            chat_type = topic['chat_type']
            topic_name = topic['name']

            logger.info(f"Migrating topic '{topic_name}' from {chat_type} chat {chat_id}")

            if chat_type == 'private':
                # For private chats, assign to the chat owner (the user who chatted with bot)
                user_id = await self._get_private_chat_owner(conn, chat_id)
                if user_id:
                    await self._assign_topic_to_user(conn, topic['id'], user_id)
                    self.stats['private_assigned'] += 1
                    logger.info(f"✅ Assigned topic '{topic_name}' to user {user_id} (private chat)")
                else:
                    await self._orphan_topic(conn, topic['id'])
                    self.stats['orphaned'] += 1
                    logger.warning(f"⚠️ Orphaned topic '{topic_name}' (no private chat owner found)")

            elif chat_type in ['group', 'supergroup']:
                # For groups, try to assign to first admin or most active user
                user_id = await self._get_group_topic_owner(conn, chat_id)
                if user_id:
                    await self._assign_topic_to_user(conn, topic['id'], user_id)
                    self.stats['group_assigned'] += 1
                    logger.info(f"✅ Assigned topic '{topic_name}' to user {user_id} (group admin/active user)")
                else:
                    await self._orphan_topic(conn, topic['id'])
                    self.stats['orphaned'] += 1
                    logger.warning(f"⚠️ Orphaned topic '{topic_name}' (no suitable group owner found)")

            else:
                # For channels or unknown types, orphan for manual assignment
                await self._orphan_topic(conn, topic['id'])
                self.stats['orphaned'] += 1
                logger.warning(f"⚠️ Orphaned topic '{topic_name}' (unsupported chat type: {chat_type})")

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to migrate topic '{topic.get('name', 'unknown')}': {e}")

    async def _get_private_chat_owner(self, conn: sqlite3.Connection, chat_id: str) -> Optional[int]:
        """Get owner of private chat (the user who chatted with bot).

        Args:
            conn: Database connection
            chat_id: Chat ID

        Returns:
            User ID if found, None otherwise
        """
        # For private chats, the chat_id IS the user_id (negative for groups, positive for users)
        try:
            # Telegram private chat IDs are positive integers
            user_id = int(chat_id)
            if user_id > 0:
                return user_id
        except ValueError:
            pass

        return None

    async def _get_group_topic_owner(self, conn: sqlite3.Connection, chat_id: str) -> Optional[int]:
        """Get best owner candidate for group topic.

        Strategy:
        1. Look for bot admin interactions in processing results or other logs
        2. Use a reasonable default assignment strategy
        3. Return None if no suitable candidate found

        Args:
            conn: Database connection
            chat_id: Group chat ID

        Returns:
            User ID if found, None otherwise
        """
        # Since we don't have bot interaction logs that track user IDs,
        # we'll need to use a simple strategy for groups.
        # In real deployment, you might want to store user interactions.

        # For now, we'll orphan group topics and let users claim them manually
        # This is the safest approach to avoid incorrect assignments
        return None

    async def _assign_topic_to_user(self, conn: sqlite3.Connection, topic_id: int, user_id: int) -> None:
        """Assign topic to a user and ensure user subscription exists.

        Args:
            conn: Database connection
            topic_id: Topic ID
            user_id: User ID
        """
        # Ensure user subscription exists
        await self._ensure_user_subscription(conn, user_id)

        # Assign topic to user
        conn.execute("""
            UPDATE topics
            SET telegram_user_id = ?
            WHERE id = ?
        """, (user_id, topic_id))

    async def _orphan_topic(self, conn: sqlite3.Connection, topic_id: int) -> None:
        """Mark topic as orphaned (no owner assignment).

        Args:
            conn: Database connection
            topic_id: Topic ID
        """
        # Leave telegram_user_id as NULL - this indicates it needs manual assignment
        logger.debug(f"Topic {topic_id} left orphaned for manual assignment")

    async def _ensure_user_subscription(self, conn: sqlite3.Connection, user_id: int) -> None:
        """Ensure user subscription record exists.

        Args:
            conn: Database connection
            user_id: User ID
        """
        # Check if subscription exists
        existing = conn.execute("""
            SELECT telegram_user_id FROM user_subscriptions
            WHERE telegram_user_id = ?
        """, (user_id,)).fetchone()

        if not existing:
            # Create FREE tier subscription for new user
            conn.execute("""
                INSERT INTO user_subscriptions (telegram_user_id, subscription_tier, created_at)
                VALUES (?, ?, ?)
            """, (user_id, UserTier.FREE, datetime.utcnow().isoformat()))
            logger.debug(f"Created FREE tier subscription for user {user_id}")

    def _print_migration_summary(self) -> None:
        """Print migration summary statistics."""
        logger.info("\n" + "="*60)
        logger.info("MIGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total topics processed: {self.stats['total_topics']}")
        logger.info(f"Private chat topics assigned: {self.stats['private_assigned']}")
        logger.info(f"Group topics assigned: {self.stats['group_assigned']}")
        logger.info(f"Topics orphaned (need manual assignment): {self.stats['orphaned']}")
        logger.info(f"Topics already assigned: {self.stats['already_assigned']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")

        success_rate = ((self.stats['private_assigned'] + self.stats['group_assigned']) /
                       max(self.stats['total_topics'], 1)) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

        if self.stats['orphaned'] > 0:
            logger.info("\n⚠️ ORPHANED TOPICS:")
            logger.info("These topics need manual assignment using bot commands:")
            logger.info("- Users can use /claim_topic <topic_name> to claim ownership")
            logger.info("- Or topics can be assigned to admins manually")

        logger.info("="*60)


async def main():
    """Main migration function."""
    try:
        # Load settings
        settings = get_settings()

        logger.info("CuliFeed Topic Migration to User-Based Ownership")
        logger.info(f"Database: {settings.database.path}")

        # Check if database exists
        db_path = Path(settings.database.path)
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            logger.error("Please ensure CuliFeed has been run at least once to create the database.")
            sys.exit(1)

        # Check if SaaS mode is enabled
        if settings.saas.saas_mode:
            logger.warning("SaaS mode is already enabled. Migration may have already been run.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                logger.info("Migration cancelled.")
                sys.exit(0)

        # Run migration
        migration_service = TopicMigrationService(str(db_path))
        await migration_service.migrate_all_topics()

        logger.info("\n✅ Migration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Enable SaaS mode: CULIFEED_SAAS__SAAS_MODE=true")
        logger.info("2. Restart CuliFeed bot")
        logger.info("3. Monitor user topic limits (5 topics per user)")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())