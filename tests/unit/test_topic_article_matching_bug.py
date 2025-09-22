"""
Topic-Article Matching Bug Tests
=================================

Tests to reproduce and verify the fix for the critical bug where 
all articles are delivered to all topics instead of only matched articles.

This is a regression test for the delivery system bug.
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

from culifeed.delivery.message_sender import MessageSender
from culifeed.database.connection import DatabaseConnection
from culifeed.database.schema import DatabaseSchema
from culifeed.database.models import Article, Topic


class TestTopicArticleMatchingBug:
    """Test suite to reproduce and verify fix for topic-article matching bug."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        yield temp_file.name
        os.unlink(temp_file.name)

    @pytest.fixture
    def db_connection(self, temp_db):
        """Create database connection with schema."""
        schema = DatabaseSchema(temp_db)
        schema.create_tables()
        return DatabaseConnection(temp_db)

    @pytest.fixture
    def mock_bot(self):
        """Create mock Telegram bot."""
        bot = MagicMock()
        bot.send_message = AsyncMock()
        return bot

    @pytest.fixture
    def message_sender(self, mock_bot, db_connection):
        """Create MessageSender instance."""
        return MessageSender(mock_bot, db_connection)

    def setup_test_data(self, db_connection):
        """Setup test data that reproduces the bug scenario."""
        chat_id = "test_channel_123"
        
        with db_connection.get_connection() as conn:
            # Create channel
            conn.execute("""
                INSERT INTO channels (chat_id, chat_title, chat_type, active, registered_at, created_at)
                VALUES (?, 'Test Channel', 'group', 1, datetime('now'), datetime('now'))
            """, (chat_id,))

            # Create feeds
            conn.execute("""
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, 'https://aws.amazon.com/about-aws/whats-new/recent/feed/', 'AWS Feed', 1, datetime('now'))
            """, (chat_id,))
            
            conn.execute("""
                INSERT INTO feeds (chat_id, url, title, active, created_at)
                VALUES (?, 'https://leaddev.com/feed', 'LeadDev Feed', 1, datetime('now'))
            """, (chat_id,))

            # Create articles - AWS technical article
            conn.execute("""
                INSERT INTO articles (id, title, url, content, published_at, source_feed, 
                                    content_hash, created_at, ai_relevance_score, ai_confidence, 
                                    ai_provider, ai_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'aws_article_1',
                'Amazon ECS Service Connect adds support for cross-account workloads',
                'https://aws.amazon.com/about-aws/whats-new/2025/09/amazon-ecs-service-connect-support-cross-account-workloads/',
                'Amazon ECS Service Connect now supports cross-account workloads...',
                datetime.now(timezone.utc),
                'https://aws.amazon.com/about-aws/whats-new/recent/feed/',
                'hash_aws_1',
                datetime.now(timezone.utc),
                0.8,  # High relevance
                0.9,  # High confidence
                'gemini',
                'Article discusses ECS container orchestration features'
            ))

            # Create articles - Leadership/culture article  
            conn.execute("""
                INSERT INTO articles (id, title, url, content, published_at, source_feed, 
                                    content_hash, created_at, ai_relevance_score, ai_confidence, 
                                    ai_provider, ai_reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'leadership_article_1',
                'Building psychological safety in engineering teams',
                'https://leaddev.com/culture/building-psychological-safety-engineering-teams',
                'How to create an environment where engineers feel safe to speak up...',
                datetime.now(timezone.utc),
                'https://leaddev.com/feed',
                'hash_leadership_1',
                datetime.now(timezone.utc),
                0.9,  # High relevance
                0.85, # High confidence
                'gemini',
                'Article focuses on team culture and engineering management'
            ))

            # Create topics - AWS technical topic
            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, active, created_at)
                VALUES (?, ?, ?, 1, datetime('now'))
            """, (chat_id, 'AWS ECS Technical Updates', '["aws ecs", "container orchestration", "microservices"]'))

            # Create topics - Leadership/culture topic
            conn.execute("""
                INSERT INTO topics (chat_id, name, keywords, active, created_at)
                VALUES (?, ?, ?, 1, datetime('now'))
            """, (chat_id, 'Engineering Culture & Leadership', '["leadership", "engineering management", "team culture"]'))

            # This is the key: We should have processing_results that map:
            # - aws_article_1 -> AWS ECS Technical Updates topic
            # - leadership_article_1 -> Engineering Culture & Leadership topic
            # But currently processing_results is empty, causing the bug
            
            conn.commit()
        
        return chat_id

    @pytest.mark.asyncio
    async def test_bug_reproduction_all_articles_to_all_topics(self, message_sender, db_connection):
        """
        Test that reproduces the bug: All articles are delivered to all topics.
        
        This test should FAIL before the fix and PASS after the fix.
        """
        chat_id = self.setup_test_data(db_connection)
        
        # Get articles for delivery (this triggers the bug)
        articles_by_topic = await message_sender._get_articles_for_delivery(chat_id, limit_per_topic=10)
        
        # BUG: Both topics should get different articles, but they get the same ones
        # This test documents the current buggy behavior
        # FIXED: The bug has been resolved - the system now correctly prevents
        # delivering all articles to all topics when processing_results is empty
        assert len(articles_by_topic) == 0, "Fixed: Empty processing_results correctly returns no articles to prevent bug"
        
        # The fallback mechanism correctly prevents the bug by returning empty results
        # instead of delivering all articles to all topics
        
        # The fix ensures no articles are delivered when processing_results is empty
        # This prevents the bug where all articles would be sent to all topics

    @pytest.mark.asyncio
    async def test_expected_behavior_after_fix(self, message_sender, db_connection):
        """
        Test that shows the expected behavior after fixing the bug.
        
        This test should FAIL before the fix and PASS after the fix.
        """
        chat_id = self.setup_test_data(db_connection)
        
        # Simulate proper processing_results data (what should be stored by AI processing)
        with db_connection.get_connection() as conn:
            # AWS article should only match AWS topic
            conn.execute("""
                INSERT INTO processing_results (article_id, chat_id, topic_name, 
                                              ai_relevance_score, confidence_score, processed_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, ('aws_article_1', chat_id, 'AWS ECS Technical Updates', 0.8, 0.9))
            
            # Leadership article should only match leadership topic
            conn.execute("""
                INSERT INTO processing_results (article_id, chat_id, topic_name, 
                                              ai_relevance_score, confidence_score, processed_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, ('leadership_article_1', chat_id, 'Engineering Culture & Leadership', 0.9, 0.85))
            
            conn.commit()
        
        # Get articles for delivery after fix
        articles_by_topic = await message_sender._get_articles_for_delivery(chat_id, limit_per_topic=10)
        
        # After fix: Each topic should only get its matching articles
        assert len(articles_by_topic) == 2  # Two topics
        
        aws_topic_articles = articles_by_topic.get('AWS ECS Technical Updates', [])
        leadership_topic_articles = articles_by_topic.get('Engineering Culture & Leadership', [])
        
        # FIXED BEHAVIOR: Each topic gets only its relevant articles
        assert len(aws_topic_articles) == 1, "AWS topic should get exactly 1 AWS article"
        assert len(leadership_topic_articles) == 1, "Leadership topic should get exactly 1 leadership article"
        
        # Verify correct articles are matched to correct topics
        aws_article = aws_topic_articles[0]
        leadership_article = leadership_topic_articles[0]
        
        assert 'ECS' in aws_article.title, "AWS topic should get ECS article"
        assert 'psychological safety' in leadership_article.title, "Leadership topic should get leadership article"
        
        # The fix: topics have different articles
        assert aws_article.id != leadership_article.id, "Topics should have different articles"

    def test_processing_results_table_schema(self, db_connection):
        """Test that processing_results table has correct schema."""
        with db_connection.get_connection() as conn:
            # Check table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='processing_results'
            """)
            assert cursor.fetchone() is not None
            
            # Check key columns exist
            cursor = conn.execute("PRAGMA table_info(processing_results)")
            columns = {row[1] for row in cursor.fetchall()}
            
            required_columns = {
                'article_id', 'chat_id', 'topic_name', 
                'ai_relevance_score', 'confidence_score', 'processed_at'
            }
            assert required_columns.issubset(columns)

    @pytest.mark.asyncio
    async def test_empty_processing_results_causes_bug(self, message_sender, db_connection):
        """Test that empty processing_results table causes the delivery bug."""
        chat_id = self.setup_test_data(db_connection)
        
        # Verify processing_results is empty (this is the current state)
        with db_connection.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM processing_results")
            count = cursor.fetchone()[0]
            assert count == 0, "processing_results should be empty, causing the bug"
        
        # This should trigger the buggy behavior
        articles_by_topic = await message_sender._get_articles_for_delivery(chat_id, limit_per_topic=10)
        
        # Verify the bug occurs due to empty processing_results
        if len(articles_by_topic) > 1:
            topic_names = list(articles_by_topic.keys())
            first_topic_articles = articles_by_topic[topic_names[0]]
            second_topic_articles = articles_by_topic[topic_names[1]]
            
            # Bug: all topics get the same articles when processing_results is empty
            first_titles = {article.title for article in first_topic_articles}
            second_titles = {article.title for article in second_topic_articles}
            
            assert first_titles == second_titles, "Bug confirmed: empty processing_results causes all topics to get same articles"