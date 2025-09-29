"""
User Subscription Service
========================

Service for managing user subscriptions, limits, and SaaS billing logic.
"""

from typing import Optional, Tuple
import sqlite3
from datetime import datetime

from ..database.models import UserSubscription, UserTier
from ..config.settings import get_settings
from ..utils.exceptions import ValidationError, ErrorCode


class UserSubscriptionService:
    """Service for managing user subscriptions and billing limits."""

    def __init__(self, db_connection):
        """Initialize with database connection."""
        self.db = db_connection

    async def get_user_subscription(self, telegram_user_id: int) -> UserSubscription:
        """Get user subscription, creating one if it doesn't exist.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            UserSubscription object
        """
        with self.db.get_connection() as conn:
            # Try to get existing subscription
            row = conn.execute(
                """
                SELECT telegram_user_id, subscription_tier, created_at
                FROM user_subscriptions
                WHERE telegram_user_id = ?
            """,
                (telegram_user_id,),
            ).fetchone()

            if row:
                return UserSubscription(
                    telegram_user_id=row["telegram_user_id"],
                    subscription_tier=UserTier(row["subscription_tier"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            else:
                # Create new FREE tier subscription
                created_at = datetime.utcnow()
                conn.execute(
                    """
                    INSERT INTO user_subscriptions (telegram_user_id, subscription_tier, created_at)
                    VALUES (?, ?, ?)
                """,
                    (telegram_user_id, UserTier.FREE, created_at.isoformat()),
                )
                conn.commit()

                return UserSubscription(
                    telegram_user_id=telegram_user_id,
                    subscription_tier=UserTier.FREE,
                    created_at=created_at,
                )

    async def count_user_topics(self, telegram_user_id: int) -> int:
        """Count active topics owned by user across all chats.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            Number of active topics owned by user
        """
        with self.db.get_connection() as conn:
            result = conn.execute(
                """
                SELECT COUNT(*) as topic_count
                FROM topics
                WHERE telegram_user_id = ? AND active = 1
            """,
                (telegram_user_id,),
            ).fetchone()

            return result["topic_count"] if result else 0

    async def can_add_topic(self, telegram_user_id: int) -> Tuple[bool, str]:
        """Check if user can add another topic based on their subscription.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            Tuple of (can_add: bool, reason: str)
        """
        settings = get_settings()

        # If SaaS mode is disabled, allow unlimited topics
        if not settings.saas.saas_mode:
            return True, "Self-hosted unlimited"

        # Get user subscription and current topic count
        user_subscription = await self.get_user_subscription(telegram_user_id)
        current_count = await self.count_user_topics(telegram_user_id)

        # Check limits based on subscription tier
        if user_subscription.subscription_tier == UserTier.PRO:
            return True, "Pro user unlimited"

        # FREE tier limit check
        if current_count < settings.saas.free_tier_topic_limit_per_user:
            return (
                True,
                f"Within free tier limit ({current_count}/{settings.saas.free_tier_topic_limit_per_user})",
            )
        else:
            return (
                False,
                f"Free tier limit reached ({current_count}/{settings.saas.free_tier_topic_limit_per_user} topics). "
                f"Upgrade to Pro for unlimited topics! Use /upgrade to learn more.",
            )

    async def get_user_topic_summary(self, telegram_user_id: int) -> dict:
        """Get user's topic usage summary across all chats.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            Dictionary with topic usage information
        """
        settings = get_settings()
        user_subscription = await self.get_user_subscription(telegram_user_id)
        current_count = await self.count_user_topics(telegram_user_id)

        # Get topics grouped by chat
        with self.db.get_connection() as conn:
            topics_by_chat = conn.execute(
                """
                SELECT t.chat_id, c.chat_title, c.chat_type, COUNT(*) as topic_count,
                       GROUP_CONCAT(t.name, ', ') as topic_names
                FROM topics t
                JOIN channels c ON t.chat_id = c.chat_id
                WHERE t.telegram_user_id = ? AND t.active = 1
                GROUP BY t.chat_id, c.chat_title, c.chat_type
                ORDER BY topic_count DESC
            """,
                (telegram_user_id,),
            ).fetchall()

        return {
            "user_id": telegram_user_id,
            "subscription_tier": user_subscription.subscription_tier,
            "total_topics": current_count,
            "limit": (
                settings.saas.free_tier_topic_limit_per_user
                if user_subscription.subscription_tier == UserTier.FREE
                else "Unlimited"
            ),
            "can_add_more": user_subscription.subscription_tier == UserTier.PRO
            or current_count < settings.saas.free_tier_topic_limit_per_user,
            "topics_by_chat": [
                {
                    "chat_id": row["chat_id"],
                    "chat_title": row["chat_title"],
                    "chat_type": row["chat_type"],
                    "topic_count": row["topic_count"],
                    "topic_names": row["topic_names"],
                }
                for row in topics_by_chat
            ],
            "saas_mode_enabled": settings.saas.saas_mode,
        }

    async def upgrade_user_to_pro(self, telegram_user_id: int) -> bool:
        """Upgrade user to PRO tier (for future payment integration).

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            True if successful
        """
        # Ensure user subscription exists first
        await self.get_user_subscription(telegram_user_id)

        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE user_subscriptions
                SET subscription_tier = ?
                WHERE telegram_user_id = ?
            """,
                (UserTier.PRO, telegram_user_id),
            )
            conn.commit()

        return True

    async def get_user_topics_across_chats(self, telegram_user_id: int) -> list:
        """Get all topics owned by user across all chats.

        Args:
            telegram_user_id: Telegram user ID

        Returns:
            List of topic dictionaries with chat information
        """
        with self.db.get_connection() as conn:
            topics = conn.execute(
                """
                SELECT t.*, c.chat_title, c.chat_type
                FROM topics t
                JOIN channels c ON t.chat_id = c.chat_id
                WHERE t.telegram_user_id = ? AND t.active = 1
                ORDER BY c.chat_title, t.name
            """,
                (telegram_user_id,),
            ).fetchall()

            return [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "keywords": row["keywords"],
                    "chat_id": row["chat_id"],
                    "chat_title": row["chat_title"],
                    "chat_type": row["chat_type"],
                    "created_at": row["created_at"],
                }
                for row in topics
            ]
