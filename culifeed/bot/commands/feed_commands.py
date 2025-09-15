"""
Feed Management Commands
=======================

Telegram bot commands for managing RSS feeds in CuliFeed channels.
Handles feed addition, removal, testing, and listing.

Commands:
- /feeds - List all RSS feeds for the channel
- /addfeed - Add a new RSS feed
- /removefeed - Remove an existing RSS feed
- /testfeed - Test RSS feed connectivity and content
"""

import asyncio
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from telegram import Update
from telegram.ext import ContextTypes

from ...database.connection import DatabaseConnection
from ...database.models import Feed
from ...processing.feed_fetcher import FeedFetcher, FetchResult
from ...ingestion.feed_manager import FeedManager
from ...utils.logging import get_logger_for_component
from ...utils.validators import URLValidator, ValidationError
from ...utils.exceptions import TelegramError, FeedError, ErrorCode


class FeedCommandHandler:
    """Handler for feed-related bot commands."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize feed command handler.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.feed_manager = FeedManager()
        self.feed_fetcher = FeedFetcher(max_concurrent=1, timeout=15)  # Conservative for bot usage
        self.logger = get_logger_for_component('feed_commands')

    async def handle_list_feeds(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /feeds command - list all feeds for the channel.

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)

            # Get all feeds for this channel
            feeds = self.feed_manager.get_feeds_for_channel(chat_id, active_only=True)

            if not feeds:
                message = (
                    "📡 *No RSS feeds configured*\n\n"
                    "Add your first feed with:\n"
                    "`/addfeed https://aws.amazon.com/blogs/compute/feed/`\n\n"
                    "💡 RSS feeds provide the content I'll curate for your topics!"
                )
            else:
                message = "📡 *Your RSS Feeds:*\n\n"
                for i, feed in enumerate(feeds, 1):
                    # Format last fetch info
                    if feed.last_success_at:
                        last_fetch = feed.last_success_at.strftime("%m/%d %H:%M")
                        status_emoji = "🟢" if feed.error_count == 0 else "🟡"
                    else:
                        last_fetch = "Never"
                        status_emoji = "🔴" if feed.error_count > 0 else "⚪"

                    # Truncate long URLs for display
                    display_url = str(feed.url)
                    if len(display_url) > 50:
                        display_url = display_url[:47] + "..."

                    message += (
                        f"{status_emoji} *{i}. {feed.title or 'Untitled Feed'}*\n"
                        f"URL: `{display_url}`\n"
                        f"Last fetch: {last_fetch}\n"
                    )

                    if feed.error_count > 0:
                        message += f"⚠️ Errors: {feed.error_count}\n"

                    message += "\n"

                message += f"*Total: {len(feeds)} feeds*\n\n"

                # Add health summary
                healthy_feeds = sum(1 for f in feeds if f.is_healthy())
                if healthy_feeds == len(feeds):
                    message += "✅ All feeds are healthy!"
                else:
                    unhealthy = len(feeds) - healthy_feeds
                    message += f"⚠️ {unhealthy} feed(s) need attention"

                message += "\n\n💡 Use `/testfeed <url>` to check feed status."

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await self._handle_error(update, "list feeds", e)

    async def handle_add_feed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /addfeed command - add a new RSS feed.

        Format: /addfeed <rss_url>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)
            args = context.args

            if not args:
                await self._send_add_feed_help(update)
                return

            feed_url = " ".join(args).strip()

            # Validate URL
            try:
                validated_url = URLValidator.validate_feed_url(feed_url)
            except ValidationError as e:
                await update.message.reply_text(
                    f"❌ *Invalid RSS feed URL:*\n{e.message}\n\n"
                    f"Please provide a valid HTTP/HTTPS URL.",
                    parse_mode='Markdown'
                )
                return

            # Check if feed already exists
            existing_feed = self.feed_manager.get_feed_by_url(chat_id, validated_url)
            if existing_feed:
                status = "active" if existing_feed.active else "inactive"
                await update.message.reply_text(
                    f"ℹ️ This feed is already configured ({status}).\n\n"
                    f"*Title:* {existing_feed.title or 'Untitled'}\n"
                    f"*URL:* `{validated_url}`",
                    parse_mode='Markdown'
                )
                return

            # Send "testing feed" message
            test_message = await update.message.reply_text(
                f"🔍 *Testing RSS feed...*\n`{validated_url}`\n\nThis may take a few seconds...",
                parse_mode='Markdown'
            )

            # Test the feed
            try:
                fetch_results = await self.feed_fetcher.fetch_feeds_batch([validated_url])
                result = fetch_results[0] if fetch_results else None

                if not result or not result.success:
                    error_msg = result.error if result else "Unknown error"
                    await test_message.edit_text(
                        f"❌ *Feed test failed:*\n`{validated_url}`\n\n"
                        f"*Error:* {error_msg}\n\n"
                        f"Please check the URL and try again.",
                        parse_mode='Markdown'
                    )
                    return

                # Feed is working, create it
                feed = Feed(
                    chat_id=chat_id,
                    url=validated_url,
                    title=self._extract_feed_title(result),
                    description=None,
                    last_fetched_at=datetime.now(timezone.utc),
                    last_success_at=datetime.now(timezone.utc),
                    error_count=0,
                    active=True
                )

                # Save to database
                feed_id = self.feed_manager.add_feed(feed)

                if feed_id:
                    await test_message.edit_text(
                        f"✅ *RSS feed added successfully!*\n\n"
                        f"*Title:* {feed.title or 'Untitled Feed'}\n"
                        f"*URL:* `{validated_url}`\n"
                        f"*Articles found:* {result.article_count}\n\n"
                        f"🎯 I'll now monitor this feed for content matching your topics!\n\n"
                        f"💡 Make sure you have topics configured with `/topics`.",
                        parse_mode='Markdown'
                    )
                    self.logger.info(f"Added feed '{validated_url}' for channel {chat_id}")
                else:
                    await test_message.edit_text(
                        "❌ Failed to save feed to database. Please try again.",
                        parse_mode='Markdown'
                    )

            except asyncio.TimeoutError:
                await test_message.edit_text(
                    f"⏰ *Feed test timed out:*\n`{validated_url}`\n\n"
                    f"The feed might be slow or temporarily unavailable. Try again later.",
                    parse_mode='Markdown'
                )
            except Exception as test_error:
                await test_message.edit_text(
                    f"❌ *Feed test error:*\n`{validated_url}`\n\n"
                    f"*Error:* {str(test_error)}\n\n"
                    f"Please check the URL and try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "add feed", e)

    async def handle_remove_feed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /removefeed command - remove an existing RSS feed.

        Format: /removefeed <rss_url>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)
            args = context.args

            if not args:
                await update.message.reply_text(
                    "❌ *Missing RSS feed URL*\n\n"
                    "Usage: `/removefeed <rss_url>`\n\n"
                    "Use `/feeds` to see all your feeds.",
                    parse_mode='Markdown'
                )
                return

            feed_url = " ".join(args).strip()

            # Validate URL format (basic validation)
            try:
                validated_url = URLValidator.validate_feed_url(feed_url)
            except ValidationError:
                # Try to find feed by partial URL match
                feeds = self.feed_manager.get_feeds_for_channel(chat_id, active_only=True)
                matching_feeds = [f for f in feeds if feed_url in str(f.url)]

                if len(matching_feeds) == 1:
                    validated_url = str(matching_feeds[0].url)
                elif len(matching_feeds) > 1:
                    await update.message.reply_text(
                        f"❌ Multiple feeds match '{feed_url}'. Please provide the complete URL.",
                        parse_mode='Markdown'
                    )
                    return
                else:
                    await update.message.reply_text(
                        f"❌ Invalid URL format. Please provide a valid RSS feed URL.",
                        parse_mode='Markdown'
                    )
                    return

            # Find the feed
            feed = self.feed_manager.get_feed_by_url(chat_id, validated_url)
            if not feed:
                await update.message.reply_text(
                    f"❌ RSS feed not found: `{validated_url}`\n\n"
                    f"Use `/feeds` to see all your feeds.",
                    parse_mode='Markdown'
                )
                return

            # Remove the feed
            success = self.feed_manager.remove_feed(feed.id)

            if success:
                await update.message.reply_text(
                    f"✅ *RSS feed removed successfully!*\n\n"
                    f"*Title:* {feed.title or 'Untitled Feed'}\n"
                    f"*URL:* `{validated_url}`",
                    parse_mode='Markdown'
                )
                self.logger.info(f"Removed feed '{validated_url}' from channel {chat_id}")
            else:
                await update.message.reply_text(
                    "❌ Failed to remove feed. Please try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "remove feed", e)

    async def handle_test_feed(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /testfeed command - test RSS feed connectivity.

        Format: /testfeed <rss_url>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            args = context.args

            if not args:
                await update.message.reply_text(
                    "❓ *Test an RSS feed:*\n\n"
                    "Usage: `/testfeed <rss_url>`\n\n"
                    "*Example:*\n"
                    "`/testfeed https://aws.amazon.com/blogs/compute/feed/`\n\n"
                    "This will check if the feed is working and show you what content is available.",
                    parse_mode='Markdown'
                )
                return

            feed_url = " ".join(args).strip()

            # Validate URL
            try:
                validated_url = URLValidator.validate_feed_url(feed_url)
            except ValidationError as e:
                await update.message.reply_text(
                    f"❌ *Invalid RSS feed URL:*\n{e.message}",
                    parse_mode='Markdown'
                )
                return

            # Send testing message
            test_message = await update.message.reply_text(
                f"🧪 *Testing RSS feed...*\n`{validated_url}`\n\nPlease wait...",
                parse_mode='Markdown'
            )

            # Test the feed
            try:
                fetch_results = await self.feed_fetcher.fetch_feeds_batch([validated_url])
                result = fetch_results[0] if fetch_results else None

                if not result:
                    await test_message.edit_text(
                        f"❌ *Feed test failed:*\n`{validated_url}`\n\n"
                        f"*Error:* No result returned\n\n"
                        f"The feed might be temporarily unavailable.",
                        parse_mode='Markdown'
                    )
                    return

                if not result.success:
                    await test_message.edit_text(
                        f"❌ *Feed test failed:*\n`{validated_url}`\n\n"
                        f"*Error:* {result.error}\n\n"
                        f"Please check the URL and try again.",
                        parse_mode='Markdown'
                    )
                    return

                # Success - show feed information
                feed_info = (
                    f"✅ *RSS feed is working!*\n\n"
                    f"*URL:* `{validated_url}`\n"
                    f"*Articles found:* {result.article_count}\n"
                    f"*Test time:* {result.fetch_time.strftime('%H:%M:%S')}\n\n"
                )

                if result.articles:
                    feed_info += "*📰 Recent articles:*\n"
                    for i, article in enumerate(result.articles[:3], 1):
                        title = article.title[:60] + "..." if len(article.title) > 60 else article.title
                        feed_info += f"{i}. {title}\n"

                    if len(result.articles) > 3:
                        feed_info += f"... and {len(result.articles) - 3} more\n"

                feed_info += f"\n💡 Use `/addfeed {validated_url}` to add this feed!"

                await test_message.edit_text(feed_info, parse_mode='Markdown')

            except asyncio.TimeoutError:
                await test_message.edit_text(
                    f"⏰ *Feed test timed out:*\n`{validated_url}`\n\n"
                    f"The feed is taking too long to respond. It might be slow or temporarily unavailable.",
                    parse_mode='Markdown'
                )
            except Exception as test_error:
                await test_message.edit_text(
                    f"❌ *Feed test error:*\n`{validated_url}`\n\n"
                    f"*Error:* {str(test_error)}\n\n"
                    f"Please check the URL and try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "test feed", e)

    def _extract_feed_title(self, fetch_result: FetchResult) -> Optional[str]:
        """Extract feed title from fetch result.

        Args:
            fetch_result: Result from feed fetching

        Returns:
            Feed title or None if not available
        """
        # This is a simplified implementation
        # In a full implementation, you'd parse the feed metadata
        if fetch_result.articles:
            # Try to infer title from the source feed URL or articles
            return "RSS Feed"  # Placeholder
        return None

    async def _send_add_feed_help(self, update: Update) -> None:
        """Send help message for /addfeed command."""
        help_message = (
            "❓ *How to add an RSS feed:*\n\n"
            "*Format:* `/addfeed <rss_url>`\n\n"
            "*Examples:*\n"
            "• `/addfeed https://aws.amazon.com/blogs/compute/feed/`\n"
            "• `/addfeed https://blog.docker.com/feed/`\n"
            "• `/addfeed https://kubernetes.io/feed.xml`\n\n"
            "*Tips:*\n"
            "• Make sure the URL is a valid RSS/Atom feed\n"
            "• I'll test the feed before adding it\n"
            "• Configure topics first with `/addtopic` for better curation\n\n"
            "*Need feed URLs?* Many blogs have `/feed/`, `/rss/`, or `/feed.xml` endpoints."
        )
        await update.message.reply_text(help_message, parse_mode='Markdown')

    async def _handle_error(self, update: Update, operation: str, error: Exception) -> None:
        """Handle errors in feed operations.

        Args:
            update: Telegram update object
            operation: Operation that failed
            error: Exception that occurred
        """
        self.logger.error(f"Error in {operation}: {error}")

        try:
            error_message = (
                f"❌ *Error in {operation}*\n\n"
                f"Please try again or use `/help` for usage instructions."
            )
            await update.message.reply_text(error_message, parse_mode='Markdown')
        except Exception as e:
            self.logger.error(f"Failed to send error message: {e}")

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    def get_feed_statistics(self, chat_id: str) -> Dict[str, Any]:
        """Get feed statistics for a channel.

        Args:
            chat_id: Channel chat ID

        Returns:
            Dictionary with feed statistics
        """
        try:
            feeds = self.feed_manager.get_feeds_for_channel(chat_id, active_only=True)

            healthy_count = sum(1 for feed in feeds if feed.is_healthy())
            error_count = sum(1 for feed in feeds if feed.error_count > 0)

            return {
                'total_feeds': len(feeds),
                'healthy_feeds': healthy_count,
                'feeds_with_errors': error_count,
                'average_error_count': sum(f.error_count for f in feeds) / len(feeds) if feeds else 0,
                'feeds': [
                    {
                        'url': str(feed.url),
                        'title': feed.title,
                        'healthy': feed.is_healthy(),
                        'error_count': feed.error_count,
                        'last_success': feed.last_success_at
                    }
                    for feed in feeds
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting feed statistics: {e}")
            return {}

    async def validate_feed_setup(self, chat_id: str) -> Dict[str, Any]:
        """Validate feed setup for a channel.

        Args:
            chat_id: Channel chat ID

        Returns:
            Validation results dictionary
        """
        try:
            feeds = self.feed_manager.get_feeds_for_channel(chat_id, active_only=True)

            issues = []
            warnings = []

            if not feeds:
                issues.append("No RSS feeds configured")
            else:
                # Check for feeds with errors
                error_feeds = [f for f in feeds if f.error_count > 0]
                if error_feeds:
                    warnings.append(f"{len(error_feeds)} feed(s) have errors")

                # Check for feeds that should be disabled
                disabled_feeds = [f for f in feeds if f.should_disable()]
                if disabled_feeds:
                    issues.append(f"{len(disabled_feeds)} feed(s) should be disabled due to errors")

            return {
                'valid': len(issues) == 0,
                'feed_count': len(feeds),
                'issues': issues,
                'warnings': warnings
            }

        except Exception as e:
            self.logger.error(f"Error validating feed setup: {e}")
            return {'valid': False, 'issues': ['Validation error occurred']}