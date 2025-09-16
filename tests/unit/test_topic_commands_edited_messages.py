#!/usr/bin/env python3
"""
Test Topic Commands - Edited Message Handling
=============================================

Tests for topic command handlers to ensure they work correctly with both
regular messages and edited messages (the fix for NoneType errors).
"""

import pytest
import tempfile
import os
import sqlite3
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone

from culifeed.bot.commands.topic_commands import TopicCommandHandler
from culifeed.database.connection import DatabaseConnection
from culifeed.database.schema import DatabaseSchema


class TestTopicCommandsEditedMessages:
    """Test suite for topic command edited message handling."""

    @pytest.fixture
    def test_database(self):
        """Create a temporary test database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Create schema
        schema = DatabaseSchema(db_path)
        schema.create_tables()

        yield db_path

        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def db_connection(self, test_database):
        """Create database connection."""
        return DatabaseConnection(test_database)

    @pytest.fixture
    def handler(self, db_connection):
        """Create TopicCommandHandler instance."""
        return TopicCommandHandler(db_connection)

    @pytest.fixture
    def setup_test_data(self, test_database):
        """Setup test channel and topic data."""
        chat_id = "test_chat_123"

        with sqlite3.connect(test_database) as conn:
            cursor = conn.cursor()

            # Create test channel
            cursor.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, "Test Channel", "group", True, datetime.now(timezone.utc), datetime.now(timezone.utc)))

            # Create test topic
            cursor.execute("""
                INSERT INTO topics (chat_id, name, keywords, confidence_threshold, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, "test_topic", '["test", "example"]', 0.8, True, datetime.now(timezone.utc)))

            conn.commit()

        return chat_id

    def create_regular_message_update(self, chat_id: str):
        """Create mock update object for regular message."""
        update = Mock()
        update.effective_chat.id = chat_id
        update.message = AsyncMock()  # Regular message exists
        update.effective_message = update.message  # Same as message for regular messages
        return update

    def create_edited_message_update(self, chat_id: str):
        """Create mock update object for edited message."""
        update = Mock()
        update.effective_chat.id = chat_id
        update.message = None  # No message for edited messages
        update.effective_message = AsyncMock()  # But effective_message exists
        return update

    @pytest.mark.asyncio
    async def test_remove_topic_regular_message(self, handler, setup_test_data):
        """Test remove topic command with regular message."""
        chat_id = setup_test_data
        update = self.create_regular_message_update(chat_id)
        context = Mock()
        context.args = ["test_topic"]

        # Should work without error
        await handler.handle_remove_topic(update, context)

        # Verify response was sent
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "removed successfully" in call_args

    @pytest.mark.asyncio
    async def test_remove_topic_edited_message(self, handler, setup_test_data):
        """Test remove topic command with edited message (the bug fix)."""
        chat_id = setup_test_data
        update = self.create_edited_message_update(chat_id)
        context = Mock()
        context.args = ["test_topic"]

        # Should work without error (this was failing before the fix)
        await handler.handle_remove_topic(update, context)

        # Verify response was sent using effective_message
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "removed successfully" in call_args

    @pytest.mark.asyncio
    async def test_remove_topic_nonexistent_regular_message(self, handler, setup_test_data):
        """Test remove topic command for nonexistent topic with regular message."""
        chat_id = setup_test_data
        update = self.create_regular_message_update(chat_id)
        context = Mock()
        context.args = ["nonexistent_topic"]

        await handler.handle_remove_topic(update, context)

        # Verify error response was sent
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "not found" in call_args

    @pytest.mark.asyncio
    async def test_remove_topic_nonexistent_edited_message(self, handler, setup_test_data):
        """Test remove topic command for nonexistent topic with edited message."""
        chat_id = setup_test_data
        update = self.create_edited_message_update(chat_id)
        context = Mock()
        context.args = ["nonexistent_topic"]

        await handler.handle_remove_topic(update, context)

        # Verify error response was sent using effective_message
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "not found" in call_args

    @pytest.mark.asyncio
    async def test_remove_topic_no_args_edited_message(self, handler, setup_test_data):
        """Test remove topic command with no arguments and edited message."""
        chat_id = setup_test_data
        update = self.create_edited_message_update(chat_id)
        context = Mock()
        context.args = []

        await handler.handle_remove_topic(update, context)

        # Verify usage help was sent
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "Missing topic name" in call_args
        assert "Usage:" in call_args

    @pytest.mark.asyncio
    async def test_error_handler_with_edited_message(self, handler, setup_test_data):
        """Test error handler works with edited messages."""
        chat_id = setup_test_data
        update = self.create_edited_message_update(chat_id)

        # Create a mock exception
        test_error = Exception("Test error")

        # Call the error handler directly
        await handler._handle_error(update, "test operation", test_error)

        # Verify error response was sent using effective_message
        assert update.effective_message.reply_text.called
        call_args = update.effective_message.reply_text.call_args[0][0]
        assert "Error in test operation" in call_args

    @pytest.mark.asyncio
    async def test_error_handler_with_no_effective_message(self, handler):
        """Test error handler when effective_message is also None."""
        update = Mock()
        update.effective_message = None

        test_error = Exception("Test error")

        # Should not raise an exception, just log warning
        await handler._handle_error(update, "test operation", test_error)

        # No assertion needed - the test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_multiple_message_types_consistency(self, handler, setup_test_data):
        """Test that both regular and edited messages produce same results."""
        chat_id = setup_test_data

        # Test with regular message
        regular_update = self.create_regular_message_update(chat_id)
        regular_context = Mock()
        regular_context.args = ["nonexistent_topic"]

        await handler.handle_remove_topic(regular_update, regular_context)
        regular_response = regular_update.effective_message.reply_text.call_args[0][0]

        # Test with edited message
        edited_update = self.create_edited_message_update(chat_id)
        edited_context = Mock()
        edited_context.args = ["nonexistent_topic"]

        await handler.handle_remove_topic(edited_update, edited_context)
        edited_response = edited_update.effective_message.reply_text.call_args[0][0]

        # Both should produce the same response content
        assert regular_response == edited_response


if __name__ == "__main__":
    pytest.main([__file__])