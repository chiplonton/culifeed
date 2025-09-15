"""
Minimal Bot Commands Tests
=========================

Focused tests for Phase 4 bot command functionality validation.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from culifeed.bot.commands.topic_commands import TopicCommandHandler
from culifeed.bot.commands.feed_commands import FeedCommandHandler
from culifeed.database.models import Topic


class TestTopicCommandsBasic:
    """Basic tests for topic commands - Phase 4 validation."""

    @pytest.fixture
    def handler(self):
        """Create TopicCommandHandler instance."""
        mock_db = Mock()
        return TopicCommandHandler(mock_db)

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = Mock()
        update.effective_chat.id = 12345
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        context = Mock()
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_handle_list_topics_empty(self, handler, mock_update, mock_context):
        """Test listing topics when none exist."""
        handler.topic_repo.get_topics_for_channel = Mock(return_value=[])

        await handler.handle_list_topics(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "No topics configured" in call_args

    @pytest.mark.asyncio
    async def test_handle_add_topic_no_args(self, handler, mock_update, mock_context):
        """Test topic addition with no arguments shows help."""
        mock_context.args = []

        await handler.handle_add_topic(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "How to add a topic" in call_args

    @pytest.mark.asyncio
    async def test_handle_remove_topic_no_args(self, handler, mock_update, mock_context):
        """Test topic removal with no arguments shows help."""
        mock_context.args = []

        await handler.handle_remove_topic(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Missing topic name" in call_args

    @pytest.mark.asyncio
    async def test_handle_edit_topic_no_args(self, handler, mock_update, mock_context):
        """Test topic editing with no arguments shows help."""
        mock_context.args = []

        await handler.handle_edit_topic(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "How to edit a topic" in call_args

    @pytest.mark.asyncio
    async def test_get_topic_statistics(self, handler):
        """Test topic statistics functionality."""
        chat_id = "12345"
        handler.topic_repo.get_topics_for_channel = Mock(return_value=[])

        stats = handler.get_topic_statistics(chat_id)

        assert isinstance(stats, dict)
        assert 'total_topics' in stats


class TestFeedCommandsBasic:
    """Basic tests for feed commands - Phase 4 validation."""

    @pytest.fixture
    def handler(self):
        """Create FeedCommandHandler instance."""
        mock_db = Mock()
        return FeedCommandHandler(mock_db)

    @pytest.fixture
    def mock_update(self):
        """Create mock Telegram update."""
        update = Mock()
        update.effective_chat.id = 12345
        update.message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self):
        """Create mock context."""
        context = Mock()
        context.args = []
        return context

    @pytest.mark.asyncio
    async def test_handle_list_feeds_empty(self, handler, mock_update, mock_context):
        """Test listing feeds when none exist."""
        handler.feed_manager.get_feeds_for_channel = Mock(return_value=[])

        await handler.handle_list_feeds(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "No RSS feeds configured" in call_args

    @pytest.mark.asyncio
    async def test_handle_add_feed_no_args(self, handler, mock_update, mock_context):
        """Test feed addition with no arguments shows help."""
        mock_context.args = []

        await handler.handle_add_feed(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "How to add an RSS feed" in call_args

    @pytest.mark.asyncio
    async def test_handle_remove_feed_no_args(self, handler, mock_update, mock_context):
        """Test feed removal with no arguments shows help."""
        mock_context.args = []

        await handler.handle_remove_feed(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Missing RSS feed URL" in call_args

    @pytest.mark.asyncio
    async def test_handle_test_feed_no_args(self, handler, mock_update, mock_context):
        """Test feed testing with no arguments shows help."""
        mock_context.args = []

        await handler.handle_test_feed(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Test an RSS feed" in call_args


class TestBotCommandsIntegration:
    """Integration tests for bot commands - Phase 4 validation."""

    def test_topic_handler_initialization(self):
        """Test TopicCommandHandler can be initialized."""
        mock_db = Mock()
        handler = TopicCommandHandler(mock_db)

        assert handler is not None
        assert handler.db == mock_db
        assert hasattr(handler, 'topic_repo')
        assert hasattr(handler, 'logger')

    def test_feed_handler_initialization(self):
        """Test FeedCommandHandler can be initialized."""
        mock_db = Mock()
        handler = FeedCommandHandler(mock_db)

        assert handler is not None
        assert handler.db == mock_db
        assert hasattr(handler, 'feed_manager')
        assert hasattr(handler, 'logger')

    def test_command_handlers_have_required_methods(self):
        """Test that command handlers have all required methods."""
        mock_db = Mock()

        # Topic handler methods
        topic_handler = TopicCommandHandler(mock_db)
        required_topic_methods = [
            'handle_list_topics',
            'handle_add_topic',
            'handle_remove_topic',
            'handle_edit_topic'
        ]

        for method_name in required_topic_methods:
            assert hasattr(topic_handler, method_name), f"TopicCommandHandler missing {method_name}"
            assert callable(getattr(topic_handler, method_name))

        # Feed handler methods
        feed_handler = FeedCommandHandler(mock_db)
        required_feed_methods = [
            'handle_list_feeds',
            'handle_add_feed',
            'handle_remove_feed',
            'handle_test_feed'
        ]

        for method_name in required_feed_methods:
            assert hasattr(feed_handler, method_name), f"FeedCommandHandler missing {method_name}"
            assert callable(getattr(feed_handler, method_name))