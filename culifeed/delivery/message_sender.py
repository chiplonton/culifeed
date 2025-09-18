"""
Message Delivery System
======================

Handles delivery of curated content to Telegram channels.
Formats articles into digestible messages and manages delivery scheduling.

Features:
- Content formatting for Telegram messages
- Message delivery with error handling
- Delivery statistics and tracking
- Integration with bot service
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from telegram import Bot
from telegram.error import TelegramError, BadRequest, Forbidden, TimedOut
from telegram.constants import ParseMode

from ..database.connection import DatabaseConnection
from ..database.models import Article, Topic, ProcessingResult
from ..storage.article_repository import ArticleRepository
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import DeliveryError, ErrorCode
from .digest_formatter import DigestFormatter, DigestFormat

@dataclass
class DeliveryResult:
    """Result of message delivery operation."""
    chat_id: str
    success: bool
    messages_sent: int
    articles_delivered: int
    error: Optional[str] = None
    delivery_time: Optional[datetime] = None

    def __post_init__(self):
        if not self.delivery_time:
            self.delivery_time = datetime.now(timezone.utc)

class MessageSender:
    """Handles delivery of curated content to Telegram channels."""

    def __init__(self, bot: Bot, db_connection: DatabaseConnection):
        """Initialize message sender.

        Args:
            bot: Telegram bot instance
            db_connection: Database connection manager
        """
        self.bot = bot
        self.db = db_connection
        self.article_repo = ArticleRepository(db_connection)
        self.settings = get_settings()
        self.logger = get_logger_for_component('message_sender')

        # Use DigestFormatter for all formatting needs
        self.formatter = DigestFormatter()

        # Message formatting settings
        self.max_message_length = 4096  # Telegram limit
        self.max_articles_per_message = 5

    async def deliver_daily_digest(self, chat_id: str, limit_per_topic: int = 5) -> DeliveryResult:
        """Deliver daily digest of curated articles to a channel.

        Args:
            chat_id: Channel to deliver to
            limit_per_topic: Maximum articles per topic

        Returns:
            DeliveryResult with delivery statistics
        """
        try:
            self.logger.info(f"Starting daily digest delivery for channel {chat_id}")

            # Get articles ready for delivery (from Phase 2 processing)
            articles_by_topic = await self._get_articles_for_delivery(chat_id, limit_per_topic)

            if not articles_by_topic:
                self.logger.info(f"No articles ready for delivery to channel {chat_id}")
                return DeliveryResult(
                    chat_id=chat_id,
                    success=True,
                    messages_sent=0,
                    articles_delivered=0,
                    error="No articles ready for delivery"
                )

            # Format messages using DigestFormatter
            formatted_messages = self.formatter.format_daily_digest(
                articles_by_topic,
                DigestFormat.DETAILED
            )

            # Send all formatted messages
            messages_sent = 0
            total_articles = sum(len(articles) for articles in articles_by_topic.values())

            for message in formatted_messages:
                success = await self._send_message(chat_id, message)
                if success:
                    messages_sent += 1

                # Small delay between messages to avoid rate limiting
                await asyncio.sleep(0.5)

            # Update delivery status in database
            await self._mark_articles_delivered(chat_id, articles_by_topic)

            self.logger.info(
                f"Digest delivery complete for channel {chat_id}: "
                f"{messages_sent} messages, {total_articles} articles"
            )

            return DeliveryResult(
                chat_id=chat_id,
                success=True,
                messages_sent=messages_sent,
                articles_delivered=total_articles
            )

        except Exception as e:
            self.logger.error(f"Failed to deliver digest to channel {chat_id}: {e}")
            return DeliveryResult(
                chat_id=chat_id,
                success=False,
                messages_sent=0,
                articles_delivered=0,
                error=str(e)
            )

    async def deliver_preview(self, chat_id: str, limit: int = 3) -> DeliveryResult:
        """Deliver a preview of latest content to a channel.

        Args:
            chat_id: Channel to deliver to
            limit: Maximum articles to preview

        Returns:
            DeliveryResult with delivery statistics
        """
        try:
            # Get latest articles for preview
            articles_by_topic = await self._get_articles_for_delivery(chat_id, limit)

            if not articles_by_topic:
                preview_msg = (
                    "📰 *Content Preview*\n\n"
                    "No curated articles available yet.\n\n"
                    "💡 Make sure you have:\n"
                    "• Topics configured (`/topics`)\n"
                    "• RSS feeds added (`/feeds`)\n"
                    "• Recent content processing"
                )
            else:
                # Format preview message
                preview_msg = "📰 *Content Preview*\n\n"
                total_articles = sum(len(articles) for articles in articles_by_topic.values())

                for topic_name, articles in articles_by_topic.items():
                    preview_msg += f"🎯 *{topic_name}* ({len(articles)} articles)\n"

                    for article in articles[:2]:  # Show max 2 per topic in preview
                        title = self.formatter._truncate_text(article.title, 60)
                        preview_msg += f"• {title}\n"

                    if len(articles) > 2:
                        preview_msg += f"• ... and {len(articles) - 2} more\n"

                    preview_msg += "\n"

                preview_msg += f"*Total: {total_articles} articles ready*\n\n"
                preview_msg += "🚀 Daily digest will be delivered automatically!"

            success = await self._send_message(chat_id, preview_msg)

            return DeliveryResult(
                chat_id=chat_id,
                success=success,
                messages_sent=1 if success else 0,
                articles_delivered=0  # Preview doesn't count as delivery
            )

        except Exception as e:
            self.logger.error(f"Failed to deliver preview to channel {chat_id}: {e}")
            return DeliveryResult(
                chat_id=chat_id,
                success=False,
                messages_sent=0,
                articles_delivered=0,
                error=str(e)
            )

    async def _get_articles_for_delivery(self, chat_id: str, limit_per_topic: int) -> Dict[str, List[Article]]:
        """Get articles ready for delivery, organized by topic.

        Args:
            chat_id: Channel chat ID
            limit_per_topic: Maximum articles per topic

        Returns:
            Dictionary mapping topic names to article lists
        """
        try:
            # For Phase 4 implementation without AI processing,
            # we'll get articles that passed pre-filtering
            with self.db.get_connection() as conn:
                # Get topics for this channel
                topic_rows = conn.execute("""
                    SELECT * FROM topics
                    WHERE chat_id = ? AND active = ?
                    ORDER BY created_at
                """, (chat_id, True)).fetchall()

                if not topic_rows:
                    return {}

                articles_by_topic = {}

                for topic_row in topic_rows:
                    topic_data = dict(topic_row)
                    topic_name = topic_data['name']

                    # Get AI-processed articles from feeds for this channel
                    # Prioritize articles with AI analysis, fallback to recent articles
                    article_rows = conn.execute("""
                        SELECT a.* FROM articles a
                        JOIN feeds f ON a.source_feed = f.url
                        WHERE f.chat_id = ? AND f.active = ?
                        AND datetime(a.created_at) >= datetime('now', '-1 days')
                        ORDER BY 
                            CASE WHEN a.ai_relevance_score IS NOT NULL THEN 0 ELSE 1 END,
                            a.ai_relevance_score DESC,
                            a.created_at DESC
                        LIMIT ?
                    """, (chat_id, True, limit_per_topic)).fetchall()

                    if article_rows:
                        articles = []
                        for row in article_rows:
                            article_data = dict(row)
                            article = Article(**article_data)
                            articles.append(article)

                        if articles:
                            articles_by_topic[topic_name] = articles

                return articles_by_topic

        except Exception as e:
            self.logger.error(f"Error getting articles for delivery: {e}")
            return {}

    async def _send_message(self, chat_id: str, message: str, retries: int = 3) -> bool:
        """Send a message to a chat with retry logic.

        Args:
            chat_id: Chat ID to send to
            message: Message text
            retries: Number of retry attempts

        Returns:
            True if message sent successfully, False otherwise
        """
        for attempt in range(retries):
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
                return True

            except BadRequest as e:
                # Invalid chat or message format
                self.logger.warning(f"Bad request sending to {chat_id}: {e}")
                return False

            except Forbidden as e:
                # Bot was blocked or kicked
                self.logger.warning(f"Forbidden to send to {chat_id}: {e}")
                return False

            except TimedOut as e:
                # Timeout, retry
                self.logger.warning(f"Timeout sending to {chat_id} (attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

            except TelegramError as e:
                # Other Telegram errors
                self.logger.error(f"Telegram error sending to {chat_id}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                continue

            except Exception as e:
                # Unexpected errors
                self.logger.error(f"Unexpected error sending to {chat_id}: {e}")
                return False

        return False

    async def _mark_articles_delivered(self, chat_id: str, articles_by_topic: Dict[str, List[Article]]) -> None:
        """Mark articles as delivered in the database.

        Args:
            chat_id: Channel chat ID
            articles_by_topic: Articles that were delivered
        """
        try:
            with self.db.get_connection() as conn:
                for topic_name, articles in articles_by_topic.items():
                    for article in articles:
                        # For Phase 4, we'll just log delivery
                        # In Phase 3, this would update processing_results table
                        self.logger.debug(f"Delivered article {article.id} to {chat_id}")

                # Update channel's last delivery time
                conn.execute("""
                    UPDATE channels
                    SET last_delivery_at = ?
                    WHERE chat_id = ?
                """, (datetime.now(timezone.utc), chat_id))
                conn.commit()

        except Exception as e:
            self.logger.error(f"Error marking articles as delivered: {e}")

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    async def send_custom_message(self, chat_id: str, message: str) -> bool:
        """Send a custom message to a chat.

        Args:
            chat_id: Chat ID to send to
            message: Message text

        Returns:
            True if successful, False otherwise
        """
        return await self._send_message(chat_id, message)

    async def test_delivery(self, chat_id: str) -> DeliveryResult:
        """Test message delivery to a channel.

        Args:
            chat_id: Channel to test

        Returns:
            DeliveryResult with test results
        """
        test_message = (
            "🧪 *CuliFeed Delivery Test*\n\n"
            "This is a test message to verify delivery functionality.\n\n"
            f"✅ If you can see this, delivery is working!\n"
            f"⏰ Test time: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}"
        )

        success = await self._send_message(chat_id, test_message)

        return DeliveryResult(
            chat_id=chat_id,
            success=success,
            messages_sent=1 if success else 0,
            articles_delivered=0,
            error=None if success else "Test message failed"
        )

    def get_delivery_statistics(self, chat_id: str, days: int = 7) -> Dict[str, Any]:
        """Get delivery statistics for a channel.

        Args:
            chat_id: Channel chat ID
            days: Number of days to look back

        Returns:
            Dictionary with delivery statistics
        """
        try:
            with self.db.get_connection() as conn:
                # Get channel info
                channel_row = conn.execute(
                    "SELECT * FROM channels WHERE chat_id = ?",
                    (chat_id,)
                ).fetchone()

                if not channel_row:
                    return {}

                channel = dict(channel_row)

                # Count articles from recent days (placeholder for Phase 3)
                article_count = conn.execute("""
                    SELECT COUNT(*) FROM articles a
                    JOIN feeds f ON a.source_feed = f.url
                    WHERE f.chat_id = ? AND f.active = ?
                    AND datetime(a.created_at) >= datetime('now', '-{} days')
                """.format(days), (chat_id, True)).fetchone()[0]

                return {
                    'channel_id': chat_id,
                    'channel_title': channel['chat_title'],
                    'last_delivery': channel['last_delivery_at'],
                    'articles_available': article_count,
                    'delivery_period_days': days
                }

        except Exception as e:
            self.logger.error(f"Error getting delivery statistics: {e}")
            return {}

# Convenience function for creating message sender
def create_message_sender(bot: Bot, db_connection: DatabaseConnection) -> MessageSender:
    """Create a MessageSender instance.

    Args:
        bot: Telegram bot instance
        db_connection: Database connection

    Returns:
        Configured MessageSender instance
    """
    return MessageSender(bot, db_connection)