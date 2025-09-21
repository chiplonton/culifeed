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
        self.formatter = DigestFormatter(self.settings)



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

            # Format messages for delivery
            formatted_messages = self._format_transparent_digest(articles_by_topic)

            # Send all formatted messages
            messages_sent = 0
            total_articles = sum(len(articles) for articles in articles_by_topic.values())

            for message in formatted_messages:
                success = await self._send_message(chat_id, message)
                if success:
                    messages_sent += 1

                # Small delay between messages to avoid rate limiting
                await asyncio.sleep(self.settings.delivery_quality.message_delay_seconds)

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
            with self.db.get_connection() as conn:
                # Get articles that were successfully processed and matched to topics
                # This fixes the bug where all articles were delivered to all topics
                article_rows = conn.execute("""
                    SELECT a.*, pr.topic_name, pr.ai_relevance_score, pr.confidence_score
                    FROM processing_results pr
                    JOIN articles a ON pr.article_id = a.id
                    JOIN topics t ON pr.chat_id = t.chat_id AND pr.topic_name = t.name
                    WHERE pr.chat_id = ? 
                    AND pr.delivered = 0
                    AND pr.confidence_score >= t.confidence_threshold
                    AND datetime(pr.processed_at) >= datetime('now', '-1 days')
                    ORDER BY pr.topic_name, pr.ai_relevance_score DESC, a.published_at DESC
                """, (chat_id,)).fetchall()

                if not article_rows:
                    # Fallback: if no processing_results, get recent articles (old behavior)
                    # This ensures the system still works during transition period
                    self.logger.warning(f"No processing results found for {chat_id}, falling back to all recent articles")
                    return await self._get_articles_fallback(chat_id, limit_per_topic)

                # Group articles by topic
                articles_by_topic = {}
                topic_article_count = {}
                
                for row in article_rows:
                    row_dict = dict(row)
                    topic_name = row_dict['topic_name']
                    
                    # Limit articles per topic
                    if topic_article_count.get(topic_name, 0) >= limit_per_topic:
                        continue
                    
                    # Create article object
                    article_data = {k: v for k, v in row_dict.items() if k != 'topic_name'}
                    article = Article(**article_data)
                    
                    # Add to topic group
                    if topic_name not in articles_by_topic:
                        articles_by_topic[topic_name] = []
                    
                    articles_by_topic[topic_name].append(article)
                    topic_article_count[topic_name] = topic_article_count.get(topic_name, 0) + 1

                self.logger.info(f"Retrieved {sum(len(articles) for articles in articles_by_topic.values())} articles across {len(articles_by_topic)} topics for delivery")
                return articles_by_topic

        except Exception as e:
            self.logger.error(f"Error getting articles for delivery: {e}")
            return {}
    
    async def _get_articles_fallback(self, chat_id: str, limit_per_topic: int) -> Dict[str, List[Article]]:
        """Fallback method for getting articles when processing_results is empty.
        
        IMPORTANT: This method should return empty results to prevent delivering
        all articles to all topics. The system should wait for proper AI processing.
        """
        self.logger.warning(f"Fallback triggered for {chat_id} - processing_results table is empty")
        self.logger.warning("Returning empty results to prevent incorrect article delivery")
        self.logger.warning("Please ensure AI processing is working correctly")
        
        # Return empty results instead of all articles to all topics
        # This prevents the original bug where unrelated articles get delivered
        return {}

    def _format_transparent_digest(self, articles_by_topic: Dict[str, List[Article]]) -> List[str]:
        """Format daily digest for delivery using DigestFormatter.

        Args:
            articles_by_topic: Dictionary mapping topic names to article lists

        Returns:
            List of formatted message strings
        """
        # Use the DigestFormatter to handle all formatting
        return self.formatter.format_daily_digest(articles_by_topic)

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
                delivered_count = 0
                for topic_name, articles in articles_by_topic.items():
                    for article in articles:
                        # Mark as delivered in processing_results table
                        cursor = conn.execute("""
                            UPDATE processing_results 
                            SET delivered = 1, delivery_error = NULL
                            WHERE article_id = ? AND chat_id = ? AND topic_name = ?
                        """, (article.id, chat_id, topic_name))
                        
                        if cursor.rowcount > 0:
                            delivered_count += 1
                            self.logger.debug(f"Marked article {article.id} as delivered for topic '{topic_name}'")
                        else:
                            self.logger.warning(f"Could not mark article {article.id} as delivered - no processing result found")

                # Update channel's last delivery time
                conn.execute("""
                    UPDATE channels
                    SET last_delivery_at = ?
                    WHERE chat_id = ?
                """, (datetime.now(timezone.utc), chat_id))
                conn.commit()
                
                self.logger.info(f"Marked {delivered_count} articles as delivered for channel {chat_id}")

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