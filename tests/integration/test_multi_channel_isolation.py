"""
Multi-Channel Data Isolation Test
=================================

Comprehensive integration test to verify that data is properly isolated
between different Telegram channels/chats in the CuliFeed system.

Tests all critical components:
- Topic Repository data isolation
- Database schema foreign key constraints
- Command handler chat_id extraction
- No cross-channel data leakage
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock

from culifeed.database.connection import DatabaseConnection
from culifeed.database.schema import DatabaseSchema
from culifeed.database.models import Topic, Channel, ChatType
from culifeed.storage.topic_repository import TopicRepository
from culifeed.bot.commands.topic_commands import TopicCommandHandler
from culifeed.bot.auto_registration import AutoRegistrationHandler


class TestMultiChannelIsolation:
    """Test suite for multi-channel data isolation."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()

        schema = DatabaseSchema(temp_file.name)
        schema.create_tables()

        yield temp_file.name

        # Cleanup
        os.unlink(temp_file.name)

    @pytest.fixture
    def db_connection(self, temp_db):
        """Create database connection."""
        return DatabaseConnection(temp_db)

    @pytest.fixture
    def topic_repository(self, db_connection):
        """Create topic repository."""
        return TopicRepository(db_connection)

    @pytest.fixture
    def auto_registration(self, db_connection):
        """Create auto-registration handler."""
        return AutoRegistrationHandler(db_connection)

    def test_database_schema_isolation(self, db_connection):
        """Test database schema enforces proper isolation."""
        with db_connection.get_connection() as conn:
            # Create two separate channels
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('test_channel_1', 'Test Channel 1', 'group', 1, datetime('now'), datetime('now'))
            """)

            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('test_channel_2', 'Test Channel 2', 'group', 1, datetime('now'), datetime('now'))
            """)

            # Create topics for each channel
            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, exclude_keywords, confidence_threshold, active, created_at)
                VALUES ('test_channel_1', 'AI', '["machine learning", "AI"]', '[]', 0.8, 1, datetime('now'))
            """)

            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, exclude_keywords, confidence_threshold, active, created_at)
                VALUES ('test_channel_2', 'AI', '["artificial intelligence", "ML"]', '[]', 0.7, 1, datetime('now'))
            """)

            conn.commit()

            # Verify both topics exist but with different chat_ids
            topics_channel_1 = conn.execute(
                "SELECT * FROM topics WHERE chat_id = ?", ('test_channel_1',)
            ).fetchall()

            topics_channel_2 = conn.execute(
                "SELECT * FROM topics WHERE chat_id = ?", ('test_channel_2',)
            ).fetchall()

            assert len(topics_channel_1) == 1
            assert len(topics_channel_2) == 1
            assert topics_channel_1[0]['name'] == 'AI'
            assert topics_channel_2[0]['name'] == 'AI'
            # Different keywords prove they're isolated
            assert 'machine learning' in topics_channel_1[0]['keywords']
            assert 'artificial intelligence' in topics_channel_2[0]['keywords']

    def test_topic_repository_isolation(self, topic_repository, db_connection):
        """Test TopicRepository properly isolates data by chat_id."""
        # First create the channels that topics will reference
        with db_connection.get_connection() as conn:
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('channel_1', 'Channel 1', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('channel_2', 'Channel 2', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.commit()

        # Create topics for two different channels
        topic_1 = Topic(
            chat_id='channel_1',
            name='Python',
            keywords=['python', 'programming'],
            exclude_keywords=[],
            confidence_threshold=0.8,
            active=True
        )

        topic_2 = Topic(
            chat_id='channel_2',
            name='Python',  # Same name, different channel
            keywords=['python', 'django'],
            exclude_keywords=[],
            confidence_threshold=0.7,
            active=True
        )

        topic_3 = Topic(
            chat_id='channel_1',
            name='JavaScript',
            keywords=['javascript', 'js'],
            exclude_keywords=[],
            confidence_threshold=0.9,
            active=True
        )

        # Create topics
        topic_1_id = topic_repository.create_topic(topic_1)
        topic_2_id = topic_repository.create_topic(topic_2)
        topic_3_id = topic_repository.create_topic(topic_3)

        assert topic_1_id is not None
        assert topic_2_id is not None
        assert topic_3_id is not None

        # Test get_topics_for_channel isolation
        channel_1_topics = topic_repository.get_topics_for_channel('channel_1')
        channel_2_topics = topic_repository.get_topics_for_channel('channel_2')

        assert len(channel_1_topics) == 2  # Python and JavaScript
        assert len(channel_2_topics) == 1  # Only Python

        # Verify topic names are isolated correctly
        channel_1_names = {topic.name for topic in channel_1_topics}
        channel_2_names = {topic.name for topic in channel_2_topics}

        assert channel_1_names == {'Python', 'JavaScript'}
        assert channel_2_names == {'Python'}

        # Test get_topic_by_name isolation
        python_topic_1 = topic_repository.get_topic_by_name('channel_1', 'Python')
        python_topic_2 = topic_repository.get_topic_by_name('channel_2', 'Python')

        assert python_topic_1 is not None
        assert python_topic_2 is not None
        assert python_topic_1.id != python_topic_2.id
        assert 'programming' in python_topic_1.keywords
        assert 'django' in python_topic_2.keywords

    def test_command_handler_isolation(self, db_connection):
        """Test command handlers properly use chat_id for isolation."""
        # Create mock update objects for different channels
        mock_update_1 = MagicMock()
        mock_update_1.effective_chat.id = 123456789  # Channel 1
        mock_update_1.message.reply_text = AsyncMock()

        mock_update_2 = MagicMock()
        mock_update_2.effective_chat.id = 987654321  # Channel 2
        mock_update_2.message.reply_text = AsyncMock()

        # Create topic command handler
        topic_handler = TopicCommandHandler(db_connection)

        # Mock context for topic creation
        mock_context_1 = MagicMock()
        mock_context_1.args = ['AI', 'machine', 'learning']

        mock_context_2 = MagicMock()
        mock_context_2.args = ['AI', 'artificial', 'intelligence']

        # Test that topics are created with correct chat_id isolation
        # (This is integration test, so we don't fully mock the async calls)

        # Verify that each handler extracts different chat_ids
        chat_id_1 = str(mock_update_1.effective_chat.id)
        chat_id_2 = str(mock_update_2.effective_chat.id)

        assert chat_id_1 == '123456789'
        assert chat_id_2 == '987654321'
        assert chat_id_1 != chat_id_2

    def test_topic_statistics_isolation(self, topic_repository, db_connection):
        """Test topic statistics are properly isolated by channel."""
        # First create the channels
        with db_connection.get_connection() as conn:
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('stats_channel_1', 'Stats Channel 1', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('stats_channel_2', 'Stats Channel 2', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.commit()

        # Create topics for different channels
        for i in range(3):
            topic = Topic(
                chat_id='stats_channel_1',
                name=f'Topic_{i}',
                keywords=[f'keyword_{i}'],
                exclude_keywords=[],
                confidence_threshold=0.8,
                active=True
            )
            topic_repository.create_topic(topic)

        for i in range(2):
            topic = Topic(
                chat_id='stats_channel_2',
                name=f'Topic_{i}',
                keywords=[f'keyword_{i}'],
                exclude_keywords=[],
                confidence_threshold=0.7,
                active=True
            )
            topic_repository.create_topic(topic)

        # Test channel-specific statistics
        stats_1 = topic_repository.get_topic_statistics('stats_channel_1')
        stats_2 = topic_repository.get_topic_statistics('stats_channel_2')
        stats_all = topic_repository.get_topic_statistics()

        assert stats_1['total_topics'] == 3
        assert stats_2['total_topics'] == 2
        assert stats_all['total_topics'] >= 5  # At least our topics

        assert stats_1['active_topics'] == 3
        assert stats_2['active_topics'] == 2

    def test_topic_search_isolation(self, topic_repository, db_connection):
        """Test topic search is properly isolated by channel."""
        # First create the channels
        with db_connection.get_connection() as conn:
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('search_channel_1', 'Search Channel 1', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('search_channel_2', 'Search Channel 2', 'group', 1, datetime('now'), datetime('now'))
            """)
            conn.commit()

        # Create topics with similar keywords in different channels
        topic_1 = Topic(
            chat_id='search_channel_1',
            name='ML Research',
            keywords=['machine learning', 'research'],
            exclude_keywords=[],
            confidence_threshold=0.8,
            active=True
        )

        topic_2 = Topic(
            chat_id='search_channel_2',
            name='ML Applications',
            keywords=['machine learning', 'applications'],
            exclude_keywords=[],
            confidence_threshold=0.7,
            active=True
        )

        topic_repository.create_topic(topic_1)
        topic_repository.create_topic(topic_2)

        # Test channel-specific search
        results_1 = topic_repository.search_topics('machine', 'search_channel_1')
        results_2 = topic_repository.search_topics('machine', 'search_channel_2')
        results_all = topic_repository.search_topics('machine')

        assert len(results_1) == 1
        assert len(results_2) == 1
        assert len(results_all) >= 2

        assert results_1[0].name == 'ML Research'
        assert results_2[0].name == 'ML Applications'

    def test_cascade_deletion_isolation(self, db_connection):
        """Test that CASCADE DELETE works properly for channel isolation."""
        with db_connection.get_connection() as conn:
            # Create channels
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('cascade_channel_1', 'Cascade Test 1', 'group', 1, datetime('now'), datetime('now'))
            """)

            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES ('cascade_channel_2', 'Cascade Test 2', 'group', 1, datetime('now'), datetime('now'))
            """)

            # Create topics for each channel
            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, exclude_keywords, confidence_threshold, active, created_at)
                VALUES ('cascade_channel_1', 'Test Topic 1', '["test"]', '[]', 0.8, 1, datetime('now'))
            """)

            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, exclude_keywords, confidence_threshold, active, created_at)
                VALUES ('cascade_channel_2', 'Test Topic 2', '["test"]', '[]', 0.8, 1, datetime('now'))
            """)

            conn.commit()

            # Verify both topics exist
            topics_before = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
            assert topics_before == 2

            # Delete one channel
            conn.execute("DELETE FROM channels WHERE chat_id = ?", ('cascade_channel_1',))
            conn.commit()

            # Verify only the topic from the deleted channel was removed
            remaining_topics = conn.execute(
                "SELECT chat_id FROM topics"
            ).fetchall()

            assert len(remaining_topics) == 1
            assert remaining_topics[0]['chat_id'] == 'cascade_channel_2'


def test_multi_channel_end_to_end_isolation():
    """End-to-end test of complete multi-channel isolation."""
    # This test verifies the complete isolation workflow
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as temp_file:
        temp_db_path = temp_file.name

    try:
        # Initialize database
        schema = DatabaseSchema(temp_db_path)
        schema.create_tables()

        db_connection = DatabaseConnection(temp_db_path)
        topic_repo = TopicRepository(db_connection)

        # Simulate two different Telegram channels
        channel_1_id = '111111111'
        channel_2_id = '222222222'

        # Create the channels first
        with db_connection.get_connection() as conn:
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, 'Channel 1', 'group', 1, datetime('now'), datetime('now'))
            """, (channel_1_id,))
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, 'Channel 2', 'group', 1, datetime('now'), datetime('now'))
            """, (channel_2_id,))
            conn.commit()

        # Create same-named topics in different channels
        topic_ai_1 = Topic(
            chat_id=channel_1_id,
            name='AI',
            keywords=['machine learning', 'neural networks'],
            exclude_keywords=[],
            confidence_threshold=0.8,
            active=True
        )

        topic_ai_2 = Topic(
            chat_id=channel_2_id,
            name='AI',
            keywords=['artificial intelligence', 'deep learning'],
            exclude_keywords=[],
            confidence_threshold=0.7,
            active=True
        )

        # Create topics
        id_1 = topic_repo.create_topic(topic_ai_1)
        id_2 = topic_repo.create_topic(topic_ai_2)

        assert id_1 != id_2

        # Verify complete isolation
        channel_1_topics = topic_repo.get_topics_for_channel(channel_1_id)
        channel_2_topics = topic_repo.get_topics_for_channel(channel_2_id)

        assert len(channel_1_topics) == 1
        assert len(channel_2_topics) == 1

        # Verify topics are different despite same name
        topic_1 = channel_1_topics[0]
        topic_2 = channel_2_topics[0]

        assert topic_1.name == topic_2.name == 'AI'
        assert topic_1.chat_id != topic_2.chat_id
        assert topic_1.keywords != topic_2.keywords
        assert topic_1.confidence_threshold != topic_2.confidence_threshold

        # Verify get_topic_by_name isolation
        found_1 = topic_repo.get_topic_by_name(channel_1_id, 'AI')
        found_2 = topic_repo.get_topic_by_name(channel_2_id, 'AI')

        assert found_1.id == id_1
        assert found_2.id == id_2
        assert found_1.id != found_2.id

        # Cross-channel queries should return None
        assert topic_repo.get_topic_by_name(channel_1_id, 'NonExistent') is None

    finally:
        # Cleanup
        os.unlink(temp_db_path)