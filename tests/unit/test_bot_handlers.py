"""
Unit Tests for Bot Handlers
===========================

Tests for the main Telegram bot service handlers including auto-registration,
basic commands, and bot lifecycle management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from telegram import Update, Chat, User, ChatMember, Bot
from telegram.ext import ContextTypes

from culifeed.bot.telegram_bot import TelegramBotService
from culifeed.bot.auto_registration import AutoRegistrationHandler
from culifeed.database.models import Channel, ChatType
from culifeed.utils.exceptions import TelegramError


class TestTelegramBotService:
    """Test cases for the main TelegramBotService."""

    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = Mock()
        mock_conn = Mock()

        # Mock the context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=mock_conn)
        context_manager.__exit__ = Mock(return_value=None)
        db.get_connection.return_value = context_manager

        return db

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        settings = Mock()
        settings.telegram.bot_token = "test_token"
        settings.database.path = "test.db"
        settings.logging.level = "INFO"
        return settings

    @pytest.fixture
    def bot_service(self, mock_db, mock_settings):
        """Create bot service with mocked dependencies."""
        with patch(
            "culifeed.bot.telegram_bot.get_settings", return_value=mock_settings
        ), patch("culifeed.bot.telegram_bot.get_db_manager", return_value=mock_db):
            service = TelegramBotService()
            return service

    @pytest.mark.asyncio
    async def test_bot_initialization(self, bot_service, mock_settings):
        """Test bot service initialization."""
        with patch("culifeed.bot.telegram_bot.ApplicationBuilder") as mock_builder:
            mock_app = Mock()
            mock_builder.return_value.token.return_value.build.return_value = mock_app
            mock_app.bot = Mock()

            await bot_service.initialize()

            assert bot_service.application == mock_app
            assert bot_service.bot == mock_app.bot
            mock_builder.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_command_handler(self, bot_service):
        """Test /start command handler."""
        # Mock update and context
        update = Mock()
        update.effective_chat = Mock()
        update.effective_chat.id = 12345
        update.effective_chat.title = "Test Group"
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        context = Mock()

        # Mock channel registration
        with patch.object(
            bot_service, "_ensure_channel_registered", new_callable=AsyncMock
        ):
            await bot_service._handle_start(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        assert "CuliFeed Bot Started" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_help_command_handler(self, bot_service):
        """Test /help command handler."""
        update = Mock()
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        context = Mock()

        await bot_service._handle_help(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        assert "CuliFeed Bot Commands" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_status_command_handler(self, bot_service, mock_db):
        """Test /status command handler."""
        # Setup database mock responses
        mock_cursor = Mock()
        mock_cursor.fetchone.side_effect = [
            [5],  # topic count
            [3],  # feed count
            [25],  # article count
            {  # channel info
                "chat_title": "Test Group",
                "chat_type": "group",
                "registered_at": "2024-01-01",
                "active": True,
            },
        ]

        # Get the mock connection from the context manager
        mock_conn = mock_db.get_connection.return_value.__enter__.return_value
        mock_conn.execute.return_value = mock_cursor

        update = Mock()
        update.effective_chat = Mock()
        update.effective_chat.id = 12345
        update.message = Mock()
        update.message.reply_text = AsyncMock()

        context = Mock()

        await bot_service._handle_status(update, context)

        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args
        assert "Channel Status" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_send_message_success(self, bot_service):
        """Test successful message sending."""
        mock_bot = Mock()
        mock_bot.send_message = AsyncMock()
        bot_service.bot = mock_bot

        result = await bot_service.send_message("12345", "Test message")

        assert result is True
        mock_bot.send_message.assert_called_once_with(
            chat_id="12345", text="Test message", parse_mode="Markdown"
        )

    @pytest.mark.asyncio
    async def test_send_message_failure(self, bot_service):
        """Test message sending failure."""
        from telegram.error import BadRequest

        mock_bot = Mock()
        mock_bot.send_message = AsyncMock(side_effect=BadRequest("Chat not found"))
        bot_service.bot = mock_bot

        result = await bot_service.send_message("12345", "Test message")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_chat_info_success(self, bot_service):
        """Test successful chat info retrieval."""
        mock_chat = Mock()
        mock_chat.id = 12345
        mock_chat.title = "Test Group"
        mock_chat.type = "group"

        mock_bot = Mock()
        mock_bot.get_chat = AsyncMock(return_value=mock_chat)
        mock_bot.get_chat_member_count = AsyncMock(return_value=42)
        bot_service.bot = mock_bot

        result = await bot_service.get_chat_info("12345")

        assert result is not None
        assert result["id"] == 12345
        assert result["title"] == "Test Group"
        assert result["type"] == "group"
        assert result["member_count"] == 42

    @pytest.mark.asyncio
    async def test_ensure_channel_registered(self, bot_service, mock_db):
        """Test channel auto-registration."""
        # Get the mock connection from the context manager
        mock_conn = mock_db.get_connection.return_value.__enter__.return_value
        mock_conn.execute.return_value.fetchone.return_value = (
            None  # No existing channel
        )

        chat = Mock()
        chat.id = 12345
        chat.title = "Test Group"
        chat.type = "group"

        await bot_service._ensure_channel_registered(chat)

        # Verify channel creation SQL was called
        assert mock_conn.execute.call_count >= 1
        assert mock_conn.commit.called


class TestAutoRegistrationHandler:
    """Test cases for the AutoRegistrationHandler."""

    @pytest.fixture
    def mock_db(self):
        """Mock database connection."""
        db = Mock()
        return db

    @pytest.fixture
    def handler(self, mock_db):
        """Create auto-registration handler."""
        return AutoRegistrationHandler(mock_db)

    @pytest.mark.asyncio
    async def test_handle_bot_added_to_group(self, handler):
        """Test bot being added to a group."""
        # Create mock update for bot being added
        update = Mock()
        chat_member_update = Mock()
        chat_member_update.chat = Mock()
        chat_member_update.chat.id = 12345
        chat_member_update.chat.title = "Test Group"
        chat_member_update.chat.type = "group"

        old_member = Mock()
        old_member.status = ChatMember.LEFT
        new_member = Mock()
        new_member.status = ChatMember.MEMBER

        chat_member_update.old_chat_member = old_member
        chat_member_update.new_chat_member = new_member
        update.my_chat_member = chat_member_update

        context = Mock()
        context.bot = Mock()
        context.bot.send_message = AsyncMock()

        with patch.object(
            handler, "_create_new_channel", new_callable=AsyncMock
        ) as mock_create, patch.object(
            handler, "_get_existing_channel", return_value=None
        ):

            await handler.handle_chat_member_update(update, context)

            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_bot_removed_from_group(self, handler):
        """Test bot being removed from a group."""
        update = Mock()
        chat_member_update = Mock()
        chat_member_update.chat = Mock()
        chat_member_update.chat.id = 12345

        old_member = Mock()
        old_member.status = ChatMember.MEMBER
        new_member = Mock()
        new_member.status = ChatMember.LEFT

        chat_member_update.old_chat_member = old_member
        chat_member_update.new_chat_member = new_member
        update.my_chat_member = chat_member_update

        context = Mock()

        with patch.object(
            handler, "_unregister_channel", new_callable=AsyncMock
        ) as mock_unreg:
            await handler.handle_chat_member_update(update, context)

            mock_unreg.assert_called_once()

    def test_get_chat_type_mapping(self, handler):
        """Test chat type mapping."""
        chat = Mock()

        # Test different chat types
        chat.type = "private"
        assert handler._get_chat_type(chat) == ChatType.PRIVATE

        chat.type = "group"
        assert handler._get_chat_type(chat) == ChatType.GROUP

        chat.type = "supergroup"
        assert handler._get_chat_type(chat) == ChatType.SUPERGROUP

        chat.type = "channel"
        assert handler._get_chat_type(chat) == ChatType.CHANNEL

    def test_get_chat_title(self, handler):
        """Test chat title extraction."""
        chat = Mock()

        # Test with title
        chat.title = "Test Group"
        chat.first_name = None
        chat.last_name = None
        assert handler._get_chat_title(chat) == "Test Group"

        # Test with first_name (private chat)
        chat.title = None
        chat.first_name = "John"
        chat.last_name = "Doe"
        assert handler._get_chat_title(chat) == "John Doe"

        # Test with first_name only
        chat.title = None
        chat.first_name = "John"
        chat.last_name = None
        assert handler._get_chat_title(chat) == "John"

        # Test with username fallback
        chat.title = None
        chat.first_name = None
        chat.last_name = None
        chat.username = "testuser"
        assert handler._get_chat_title(chat) == "@testuser"

        # Test final fallback to chat ID
        chat.title = None
        chat.first_name = None
        chat.last_name = None
        chat.username = None
        chat.id = 12345
        assert handler._get_chat_title(chat) == "Chat 12345"
