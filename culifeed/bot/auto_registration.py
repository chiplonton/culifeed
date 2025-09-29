"""
Auto-Registration Module
=======================

Handles automatic registration and deregistration of channels when the
CuliFeed bot is added to or removed from Telegram chats.

Features:
- Automatic channel registration with metadata
- Channel deactivation when bot is removed
- Welcome message delivery
- Database integration with channel management
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from telegram import Update, Chat, ChatMember
from telegram.ext import ContextTypes

from ..database.connection import DatabaseConnection
from ..database.models import Channel, ChatType
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import TelegramError, ErrorCode


class AutoRegistrationHandler:
    """
    Handles automatic registration and deregistration of channels
    when the bot is added to or removed from chats.
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize auto-registration handler.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.logger = get_logger_for_component("auto_registration")

    async def handle_chat_member_update(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle bot being added to or removed from chats.

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_member_update = update.my_chat_member

            if not chat_member_update:
                return

            chat = chat_member_update.chat
            old_status = chat_member_update.old_chat_member.status
            new_status = chat_member_update.new_chat_member.status

            # Bot was added to a chat
            if old_status == ChatMember.LEFT and new_status in [
                ChatMember.MEMBER,
                ChatMember.ADMINISTRATOR,
            ]:
                await self._register_channel(chat, context)

            # Bot was removed from a chat
            elif (
                old_status in [ChatMember.MEMBER, ChatMember.ADMINISTRATOR]
                and new_status == ChatMember.LEFT
            ):
                await self._unregister_channel(chat)

            # Bot status changed (e.g., promoted to admin)
            elif old_status != new_status and new_status in [
                ChatMember.MEMBER,
                ChatMember.ADMINISTRATOR,
            ]:
                await self._update_channel_status(chat, new_status)

        except Exception as e:
            self.logger.error(f"Error handling chat member update: {e}")

    async def _register_channel(
        self, chat: Chat, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Register a new channel when bot is added.

        Args:
            chat: Telegram chat object
            context: Bot context for sending messages
        """
        try:
            chat_id = str(chat.id)
            chat_title = self._get_chat_title(chat)
            chat_type = self._get_chat_type(chat)

            # Check if channel already exists
            existing_channel = self._get_existing_channel(chat_id)

            if existing_channel:
                # Reactivate existing channel
                await self._reactivate_channel(chat_id, chat_title)
                welcome_message = self._get_welcome_back_message(chat_title)
            else:
                # Create new channel
                await self._create_new_channel(chat_id, chat_title, chat_type)
                welcome_message = self._get_welcome_message(chat_title)

            # Send welcome message
            await self._send_welcome_message(context, chat_id, welcome_message)

            self.logger.info(
                f"Successfully registered channel: {chat_title} ({chat_id})"
            )

        except Exception as e:
            self.logger.error(f"Failed to register channel {chat.id}: {e}")
            # Try to send error message to chat
            try:
                await context.bot.send_message(
                    chat_id=chat.id,
                    text="❌ Error during registration. Please try /start command.",
                )
            except:
                pass  # Ignore if we can't send error message

    async def _unregister_channel(self, chat: Chat) -> None:
        """Handle bot being removed from a channel.

        Args:
            chat: Telegram chat object
        """
        try:
            chat_id = str(chat.id)
            chat_title = self._get_chat_title(chat)

            # Deactivate channel (preserve data for potential re-addition)
            with self.db.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE channels
                    SET active = ?, last_delivery_at = ?
                    WHERE chat_id = ?
                """,
                    (False, datetime.now(timezone.utc), chat_id),
                )
                conn.commit()

            self.logger.info(f"Deactivated channel: {chat_title} ({chat_id})")

        except Exception as e:
            self.logger.error(f"Failed to unregister channel {chat.id}: {e}")

    async def _update_channel_status(self, chat: Chat, new_status: str) -> None:
        """Update channel status when bot permissions change.

        Args:
            chat: Telegram chat object
            new_status: New chat member status
        """
        try:
            chat_id = str(chat.id)

            # Log the status change
            self.logger.info(f"Bot status changed in {chat_id}: {new_status}")

            # Update channel metadata if needed
            # (Currently just logging, can be extended later)

        except Exception as e:
            self.logger.error(f"Failed to update channel status for {chat.id}: {e}")

    def _get_chat_title(self, chat: Chat) -> str:
        """Get appropriate title for a chat.

        Args:
            chat: Telegram chat object

        Returns:
            Chat title or fallback name
        """
        if chat.title:
            return chat.title
        elif chat.first_name:
            full_name = chat.first_name
            if chat.last_name:
                full_name += f" {chat.last_name}"
            return full_name
        elif chat.username:
            return f"@{chat.username}"
        else:
            return f"Chat {chat.id}"

    def _get_chat_type(self, chat: Chat) -> ChatType:
        """Map Telegram chat type to internal ChatType enum.

        Args:
            chat: Telegram chat object

        Returns:
            Internal ChatType enum value
        """
        type_mapping = {
            "private": ChatType.PRIVATE,
            "group": ChatType.GROUP,
            "supergroup": ChatType.SUPERGROUP,
            "channel": ChatType.CHANNEL,
        }
        return type_mapping.get(chat.type, ChatType.GROUP)

    def _get_existing_channel(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Check if channel already exists in database.

        Args:
            chat_id: Chat ID to check

        Returns:
            Channel data dictionary or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM channels WHERE chat_id = ?", (chat_id,)
                ).fetchone()

                return dict(row) if row else None

        except Exception as e:
            self.logger.error(f"Error checking existing channel {chat_id}: {e}")
            return None

    async def _create_new_channel(
        self, chat_id: str, chat_title: str, chat_type: ChatType
    ) -> None:
        """Create a new channel record in database.

        Args:
            chat_id: Chat ID
            chat_title: Chat title
            chat_type: Chat type
        """
        now = datetime.now(timezone.utc)

        with self.db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO channels
                (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (chat_id, chat_title, chat_type.value, True, now, now),
            )
            conn.commit()

    async def _reactivate_channel(self, chat_id: str, chat_title: str) -> None:
        """Reactivate an existing channel.

        Args:
            chat_id: Chat ID
            chat_title: Updated chat title
        """
        now = datetime.now(timezone.utc)

        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE channels
                SET active = ?, chat_title = ?, registered_at = ?
                WHERE chat_id = ?
            """,
                (True, chat_title, now, chat_id),
            )
            conn.commit()

    def _get_welcome_message(self, chat_title: str) -> str:
        """Generate welcome message for new channels.

        Args:
            chat_title: Name of the chat

        Returns:
            Formatted welcome message
        """
        return f"""🎉 *Welcome to CuliFeed!*

I've been added to *{chat_title}* and am ready to deliver curated RSS content.

*🚀 Quick Setup:*
1. `/addtopic` - Define topics you're interested in
2. `/addfeed` - Add RSS feeds to monitor
3. `/status` - Check your configuration

*📚 Getting Started:*
• Type `/help` for all available commands
• Use `/status` to see your current setup
• Configure topics first, then add RSS feeds

I'll automatically process feeds and deliver relevant content based on your topics. Let's get started! 🎯"""

    def _get_welcome_back_message(self, chat_title: str) -> str:
        """Generate welcome back message for returning channels.

        Args:
            chat_title: Name of the chat

        Returns:
            Formatted welcome back message
        """
        return f"""👋 *Welcome back to CuliFeed!*

I've been re-added to *{chat_title}*. Your previous topics and feeds have been preserved.

*🔄 Status:*
• Your channel has been reactivated
• Previous configuration restored
• Ready to resume content delivery

Type `/status` to check your current setup or `/help` for available commands."""

    async def _send_welcome_message(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: str, message: str
    ) -> None:
        """Send welcome message to a chat.

        Args:
            context: Bot context
            chat_id: Chat ID to send message to
            message: Message text to send
        """
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
        except Exception as e:
            self.logger.error(f"Failed to send welcome message to {chat_id}: {e}")

    # ================================================================
    # MANUAL REGISTRATION METHODS
    # ================================================================

    async def manually_register_channel(
        self, chat_id: str, chat_title: str, chat_type: str = "group"
    ) -> bool:
        """Manually register a channel (for testing or admin purposes).

        Args:
            chat_id: Chat ID to register
            chat_title: Chat title
            chat_type: Chat type (default: group)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert string chat_type to enum
            type_mapping = {
                "private": ChatType.PRIVATE,
                "group": ChatType.GROUP,
                "supergroup": ChatType.SUPERGROUP,
                "channel": ChatType.CHANNEL,
            }
            enum_chat_type = type_mapping.get(chat_type.lower(), ChatType.GROUP)

            await self._create_new_channel(chat_id, chat_title, enum_chat_type)
            self.logger.info(f"Manually registered channel: {chat_title} ({chat_id})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to manually register channel {chat_id}: {e}")
            return False

    def get_channel_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered channels.

        Returns:
            Dictionary with channel statistics
        """
        try:
            with self.db.get_connection() as conn:
                # Total channels
                total = conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0]

                # Active channels
                active = conn.execute(
                    "SELECT COUNT(*) FROM channels WHERE active = ?", (True,)
                ).fetchone()[0]

                # Channels by type
                type_stats = {}
                for chat_type in ChatType:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM channels WHERE chat_type = ? AND active = ?",
                        (chat_type.value, True),
                    ).fetchone()[0]
                    type_stats[chat_type.value] = count

                return {
                    "total_channels": total,
                    "active_channels": active,
                    "inactive_channels": total - active,
                    "by_type": type_stats,
                }

        except Exception as e:
            self.logger.error(f"Error getting channel statistics: {e}")
            return {}
