"""
CuliFeed Telegram Bot Service
============================

Main Telegram bot service with auto-registration, command handling,
and integration with the CuliFeed content processing pipeline.

Features:
- Automatic channel registration when bot is added to groups
- Multi-channel support with isolated data
- Topic and feed management commands
- Content delivery and status monitoring
- Integration with Phase 1 & 2 components
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from telegram import Update, Bot, BotCommand, ChatMember
from telegram.ext import (
    Application,
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
    ChatMemberHandler,
)
from telegram.error import TelegramError, BadRequest, Forbidden

from .auto_registration import AutoRegistrationHandler
from .commands.topic_commands import TopicCommandHandler
from .commands.feed_commands import FeedCommandHandler
from ..config.settings import get_settings
from ..database.connection import get_db_manager
from ..database.models import Channel, ChatType
from ..storage.topic_repository import TopicRepository
from ..storage.article_repository import ArticleRepository
from ..delivery.message_sender import MessageSender
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import TelegramError as CuliFeedTelegramError, ErrorCode


class TelegramBotService:
    """
    Main Telegram bot service for CuliFeed.

    Handles:
    - Auto-registration of channels when bot is added
    - Command routing and processing
    - Message delivery to channels
    - Integration with content processing pipeline
    """

    def __init__(self):
        """Initialize the Telegram bot service."""
        self.settings = get_settings()
        self.logger = get_logger_for_component("telegram_bot")
        self.db = get_db_manager(self.settings.database.path)

        # Initialize repositories
        self.topic_repo = TopicRepository(self.db)
        self.article_repo = ArticleRepository(self.db)

        # Initialize command handlers
        self.auto_registration = AutoRegistrationHandler(self.db)
        self.topic_commands = TopicCommandHandler(self.db)
        self.feed_commands = FeedCommandHandler(self.db)

        # Message delivery system
        self.message_sender: Optional[MessageSender] = None

        # Bot application
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None

        # Command registry
        self._commands: List[BotCommand] = []

    async def initialize(self) -> None:
        """Initialize the bot application and handlers."""
        try:
            # Create bot application
            self.application = (
                ApplicationBuilder().token(self.settings.telegram.bot_token).build()
            )

            self.bot = self.application.bot

            # Initialize message sender with bot instance
            self.message_sender = MessageSender(self.bot, self.db)

            # Register command handlers
            await self._register_handlers()

            # Set bot commands menu
            await self._setup_bot_commands()

            self.logger.info("Telegram bot initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            raise CuliFeedTelegramError(
                f"Bot initialization failed: {e}",
                error_code=ErrorCode.TELEGRAM_API_ERROR,
            )

    async def _register_handlers(self) -> None:
        """Register all command and message handlers."""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("help", self._handle_help))
        self.application.add_handler(CommandHandler("status", self._handle_status))

        # Topic management commands
        self.application.add_handler(
            CommandHandler("topics", self.topic_commands.handle_list_topics)
        )
        self.application.add_handler(
            CommandHandler("addtopic", self.topic_commands.handle_add_topic)
        )
        self.application.add_handler(
            CommandHandler("removetopic", self.topic_commands.handle_remove_topic)
        )
        self.application.add_handler(
            CommandHandler("edittopic", self.topic_commands.handle_edit_topic)
        )

        # User account and subscription commands (SaaS features)
        self.application.add_handler(
            CommandHandler("account", self.topic_commands.handle_account)
        )
        self.application.add_handler(
            CommandHandler("topic_usage", self.topic_commands.handle_topic_usage)
        )
        self.application.add_handler(
            CommandHandler("pro_info", self.topic_commands.handle_pro_info)
        )

        # Feed management commands
        self.application.add_handler(
            CommandHandler("feeds", self.feed_commands.handle_list_feeds)
        )
        self.application.add_handler(
            CommandHandler("addfeed", self.feed_commands.handle_add_feed)
        )
        self.application.add_handler(
            CommandHandler("removefeed", self.feed_commands.handle_remove_feed)
        )
        self.application.add_handler(
            CommandHandler("testfeed", self.feed_commands.handle_test_feed)
        )

        # Delivery commands
        self.application.add_handler(CommandHandler("preview", self._handle_preview))
        self.application.add_handler(CommandHandler("settings", self._handle_settings))

        # Auto-registration handler
        self.application.add_handler(
            ChatMemberHandler(
                self.auto_registration.handle_chat_member_update,
                ChatMemberHandler.MY_CHAT_MEMBER,
            )
        )

        # Unknown command handler (must be last)
        self.application.add_handler(
            MessageHandler(filters.COMMAND, self._handle_unknown_command)
        )

        self.logger.info("All command handlers registered")

    async def _setup_bot_commands(self) -> None:
        """Set up the bot commands menu for better UX."""
        commands = [
            BotCommand("start", "Initialize bot for this channel"),
            BotCommand("help", "Show available commands and usage"),
            BotCommand("status", "Show channel status and statistics"),
            BotCommand("topics", "List all topics for this channel"),
            BotCommand("addtopic", "Add a new topic with keywords"),
            BotCommand("removetopic", "Remove a topic"),
            BotCommand("edittopic", "Edit existing topic"),
            BotCommand("feeds", "List all RSS feeds for this channel"),
            BotCommand("addfeed", "Add a new RSS feed"),
            BotCommand("removefeed", "Remove RSS feed"),
            BotCommand("testfeed", "Test feed connectivity"),
            BotCommand("account", "Manage your account and subscription"),
            BotCommand("topic_usage", "Show topic usage and limits"),
            BotCommand("pro_info", "Learn about Pro tier benefits"),
            BotCommand("preview", "Preview latest content"),
            BotCommand("settings", "Show channel settings"),
        ]

        try:
            await self.bot.set_my_commands(commands)
            self._commands = commands
            self.logger.info(f"Set {len(commands)} bot commands")
        except Exception as e:
            self.logger.warning(f"Failed to set bot commands: {e}")

    # ================================================================
    # AUTO-REGISTRATION HANDLERS (now handled by AutoRegistrationHandler)
    # ================================================================

    # ================================================================
    # BASIC COMMAND HANDLERS
    # ================================================================

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        try:
            chat_id = str(update.effective_chat.id)
            chat_title = update.effective_chat.title or "this chat"

            # Ensure channel is registered
            await self._ensure_channel_registered(update.effective_chat)

            start_msg = (
                f"🤖 *CuliFeed Bot Started*\n\n"
                f"I'm ready to help you curate content for *{chat_title}*!\n\n"
                f"*Quick Setup:*\n"
                f"• `/addtopic` - Define topics you care about\n"
                f"• `/addfeed` - Add RSS feeds to monitor\n"
                f"• `/status` - Check your current setup\n\n"
                f"Type `/help` for detailed instructions."
            )

            await update.message.reply_text(start_msg, parse_mode="Markdown")

        except Exception as e:
            await self._handle_command_error(update, context, "start", e)

    async def _handle_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command."""
        try:
            help_msg = (
                "🤖 *CuliFeed Bot Commands*\n\n"
                "*📋 Basic Commands:*\n"
                "• `/start` - Initialize bot for this channel\n"
                "• `/help` - Show this help message\n"
                "• `/status` - Channel status and statistics\n\n"
                "*🎯 Topic Management:*\n"
                "• `/topics` - List topics in this chat (with global limits)\n"
                "• `/addtopic <name> <keywords>` - Add topic\n"
                "• `/removetopic <name>` - Remove topic\n"
                "• `/edittopic <name>` - Edit existing topic\n\n"
                "*📡 Feed Management:*\n"
                "• `/feeds` - List all RSS feeds\n"
                "• `/addfeed <url>` - Add RSS feed\n"
                "• `/removefeed <url>` - Remove feed\n"
                "• `/testfeed <url>` - Test feed connectivity\n\n"
                "*💎 Account & Subscription:*\n"
                "• `/account` - Manage account and subscription\n"
                "• `/topic_usage` - Show topic usage and limits\n"
                "• `/pro_info` - Learn about Pro tier benefits\n\n"
                "*⚙️ Content & Settings:*\n"
                "• `/preview` - Preview latest content\n"
                "• `/settings` - Show channel settings\n\n"
                "*Example Usage:*\n"
                "`/addtopic AI machine learning, artificial intelligence, ML`\n"
                "`/addfeed https://aws.amazon.com/blogs/compute/feed/`"
            )

            await update.message.reply_text(help_msg, parse_mode="Markdown")

        except Exception as e:
            await self._handle_command_error(update, context, "help", e)

    async def _handle_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command."""
        try:
            chat_id = str(update.effective_chat.id)

            # Get channel statistics
            with self.db.get_connection() as conn:
                # Count topics
                topic_count = conn.execute(
                    "SELECT COUNT(*) FROM topics WHERE chat_id = ? AND active = ?",
                    (chat_id, True),
                ).fetchone()[0]

                # Count feeds
                feed_count = conn.execute(
                    "SELECT COUNT(*) FROM feeds WHERE chat_id = ? AND active = ?",
                    (chat_id, True),
                ).fetchone()[0]

                # Count articles processed
                article_count = conn.execute(
                    "SELECT COUNT(*) FROM articles WHERE source_feed IN ("
                    "SELECT url FROM feeds WHERE chat_id = ? AND active = ?"
                    ")",
                    (chat_id, True),
                ).fetchone()[0]

                # Get channel info
                channel_row = conn.execute(
                    "SELECT * FROM channels WHERE chat_id = ?", (chat_id,)
                ).fetchone()

            if not channel_row:
                await update.message.reply_text(
                    "❌ Channel not registered. Use `/start` to initialize."
                )
                return

            channel = dict(channel_row)
            registered_date = (
                channel["registered_at"][:10] if channel["registered_at"] else "Unknown"
            )

            status_msg = (
                f"📊 *Channel Status*\n\n"
                f"*📋 Configuration:*\n"
                f"• Topics: {topic_count}\n"
                f"• RSS Feeds: {feed_count}\n"
                f"• Articles Processed: {article_count}\n\n"
                f"*ℹ️ Info:*\n"
                f"• Channel: {channel['chat_title']}\n"
                f"• Type: {channel['chat_type']}\n"
                f"• Registered: {registered_date}\n"
                f"• Status: {'🟢 Active' if channel['active'] else '🔴 Inactive'}\n\n"
            )

            if topic_count == 0:
                status_msg += (
                    "⚠️ *No topics configured.* Use `/addtopic` to get started.\n"
                )
            if feed_count == 0:
                status_msg += (
                    "⚠️ *No feeds configured.* Use `/addfeed` to add RSS sources.\n"
                )

            if topic_count > 0 and feed_count > 0:
                status_msg += "✅ *Setup complete!* Ready for content curation."

            await update.message.reply_text(status_msg, parse_mode="Markdown")

        except Exception as e:
            await self._handle_command_error(update, context, "status", e)

    async def _handle_preview(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /preview command."""
        try:
            chat_id = str(update.effective_chat.id)

            # Ensure channel is registered
            await self._ensure_channel_registered(update.effective_chat)

            # Use message sender to deliver preview
            if self.message_sender:
                result = await self.message_sender.deliver_preview(chat_id, limit=3)
                if not result.success:
                    await update.message.reply_text(
                        f"❌ Failed to generate preview: {result.error or 'Unknown error'}",
                        parse_mode="Markdown",
                    )
            else:
                await update.message.reply_text(
                    "❌ Message delivery system not initialized", parse_mode="Markdown"
                )

        except Exception as e:
            await self._handle_command_error(update, context, "preview", e)

    async def _handle_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /settings command."""
        await update.message.reply_text(
            "🚧 Settings management coming soon!\n"
            "This will show and allow editing channel settings."
        )

    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle unknown commands."""
        try:
            unknown_msg = (
                "❓ *Unknown command*\n\n" "Type `/help` to see all available commands."
            )
            await update.message.reply_text(unknown_msg, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Error handling unknown command: {e}")

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    async def _ensure_channel_registered(self, chat) -> None:
        """Ensure a channel is registered in the database."""
        chat_id = str(chat.id)

        with self.db.get_connection() as conn:
            existing = conn.execute(
                "SELECT * FROM channels WHERE chat_id = ?", (chat_id,)
            ).fetchone()

            if not existing:
                # Auto-register the channel
                channel = Channel(
                    chat_id=chat_id,
                    chat_title=chat.title or chat.first_name or "Unknown",
                    chat_type=(
                        ChatType.PRIVATE if chat.type == "private" else ChatType.GROUP
                    ),
                    active=True,
                    registered_at=datetime.now(timezone.utc),
                )

                conn.execute(
                    """
                    INSERT INTO channels
                    (chat_id, chat_title, chat_type, active, registered_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        channel.chat_id,
                        channel.chat_title,
                        channel.chat_type.value,
                        channel.active,
                        channel.registered_at,
                        channel.registered_at,
                    ),
                )
                conn.commit()

                self.logger.info(f"Auto-registered channel: {chat_id}")

    async def _handle_command_error(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        command: str,
        error: Exception,
    ) -> None:
        """Handle errors in command processing."""
        self.logger.error(f"Error in /{command} command: {error}")

        try:
            error_msg = (
                f"❌ *Error processing /{command} command*\n\n"
                f"Please try again or contact support if the issue persists."
            )
            await update.message.reply_text(error_msg, parse_mode="Markdown")
        except Exception as e:
            self.logger.error(f"Failed to send error message: {e}")

    # ================================================================
    # BOT LIFECYCLE METHODS
    # ================================================================

    async def start_polling(self) -> None:
        """Start the bot using polling mode."""
        if not self.application:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        self.logger.info("Starting Telegram bot polling...")
        # Let run_polling handle everything - it manages its own event loop
        await self.application.run_polling(
            allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
        )

    async def stop(self) -> None:
        """Stop the bot service."""
        if self.application:
            try:
                await self.application.stop()
                self.logger.info("Telegram bot stopped")
            except Exception as e:
                self.logger.error(f"Error stopping bot: {e}")

    def run(self) -> None:
        """Run the bot using the recommended python-telegram-bot pattern."""
        # Create application
        self.application = (
            ApplicationBuilder().token(self.settings.telegram.bot_token).build()
        )

        self.bot = self.application.bot

        # Initialize message sender with bot instance
        self.message_sender = MessageSender(self.bot, self.db)

        # Register handlers synchronously
        self._register_handlers_sync()

        # Set up bot commands menu
        self._setup_bot_commands_sync()

        # Start polling - this manages the entire lifecycle
        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES, drop_pending_updates=True
        )

    def _register_handlers_sync(self) -> None:
        """Register all command and message handlers synchronously."""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._handle_start))
        self.application.add_handler(CommandHandler("help", self._handle_help))
        self.application.add_handler(CommandHandler("status", self._handle_status))

        # Topic management commands
        self.application.add_handler(
            CommandHandler("topics", self.topic_commands.handle_list_topics)
        )
        self.application.add_handler(
            CommandHandler("addtopic", self.topic_commands.handle_add_topic)
        )
        self.application.add_handler(
            CommandHandler("removetopic", self.topic_commands.handle_remove_topic)
        )
        self.application.add_handler(
            CommandHandler("edittopic", self.topic_commands.handle_edit_topic)
        )

        # User account and subscription commands (SaaS features)
        self.application.add_handler(
            CommandHandler("account", self.topic_commands.handle_account)
        )
        self.application.add_handler(
            CommandHandler("topic_usage", self.topic_commands.handle_topic_usage)
        )
        self.application.add_handler(
            CommandHandler("pro_info", self.topic_commands.handle_pro_info)
        )

        # Feed management commands
        self.application.add_handler(
            CommandHandler("feeds", self.feed_commands.handle_list_feeds)
        )
        self.application.add_handler(
            CommandHandler("addfeed", self.feed_commands.handle_add_feed)
        )
        self.application.add_handler(
            CommandHandler("removefeed", self.feed_commands.handle_remove_feed)
        )
        self.application.add_handler(
            CommandHandler("testfeed", self.feed_commands.handle_test_feed)
        )

        # Delivery commands
        self.application.add_handler(CommandHandler("preview", self._handle_preview))
        self.application.add_handler(CommandHandler("settings", self._handle_settings))

        # Auto-registration handler
        self.application.add_handler(
            ChatMemberHandler(
                self.auto_registration.handle_chat_member_update,
                ChatMemberHandler.MY_CHAT_MEMBER,
            )
        )

        # Unknown command handler (must be last)
        self.application.add_handler(
            MessageHandler(filters.COMMAND, self._handle_unknown_command)
        )

        self.logger.info("All command handlers registered")

    def _setup_bot_commands_sync(self) -> None:
        """Set up the bot commands menu synchronously."""
        commands = [
            BotCommand("start", "Initialize bot for this channel"),
            BotCommand("help", "Show available commands and usage"),
            BotCommand("status", "Show channel status and statistics"),
            BotCommand("topics", "List all topics for this channel"),
            BotCommand("addtopic", "Add a new topic with keywords"),
            BotCommand("removetopic", "Remove a topic"),
            BotCommand("edittopic", "Edit existing topic"),
            BotCommand("feeds", "List all RSS feeds for this channel"),
            BotCommand("addfeed", "Add a new RSS feed"),
            BotCommand("removefeed", "Remove RSS feed"),
            BotCommand("testfeed", "Test feed connectivity"),
            BotCommand("account", "Manage your account and subscription"),
            BotCommand("topic_usage", "Show topic usage and limits"),
            BotCommand("pro_info", "Learn about Pro tier benefits"),
            BotCommand("preview", "Preview latest content"),
            BotCommand("settings", "Show channel settings"),
        ]

        try:
            # Use the application's initialization hook to set commands
            async def setup_commands(application):
                await self.bot.set_my_commands(commands)
                self._commands = commands
                self.logger.info(f"Set {len(commands)} bot commands")

            # Add to post_init hook instead of running immediately
            self.application.post_init = setup_commands

        except Exception as e:
            self.logger.warning(f"Failed to set bot commands: {e}")

    async def send_message(
        self, chat_id: str, message: str, parse_mode: str = "Markdown"
    ) -> bool:
        """Send a message to a specific chat.

        Args:
            chat_id: Chat ID to send message to
            message: Message text to send
            parse_mode: Message formatting mode

        Returns:
            True if message sent successfully, False otherwise
        """
        if not self.bot:
            self.logger.error("Bot not initialized")
            return False

        try:
            await self.bot.send_message(
                chat_id=chat_id, text=message, parse_mode=parse_mode
            )
            return True

        except (BadRequest, Forbidden) as e:
            self.logger.warning(f"Failed to send message to {chat_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error sending message to {chat_id}: {e}")
            return False

    async def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a chat.

        Args:
            chat_id: Chat ID to get info for

        Returns:
            Chat information dictionary or None if error
        """
        if not self.bot:
            return None

        try:
            chat = await self.bot.get_chat(chat_id)
            return {
                "id": chat.id,
                "title": chat.title,
                "type": chat.type,
                "member_count": (
                    await self.bot.get_chat_member_count(chat_id)
                    if chat.type != "private"
                    else 1
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed to get chat info for {chat_id}: {e}")
            return None


# Convenience function for external usage
async def create_bot_service() -> TelegramBotService:
    """Create and initialize a TelegramBotService instance."""
    service = TelegramBotService()
    await service.initialize()
    return service
