"""
Tests for Repository Components
===============================

Test suite for ArticleRepository and TopicRepository with comprehensive
coverage of CRUD operations and error handling.
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from typing import List

from culifeed.database.connection import DatabaseConnection
from culifeed.database.schema import DatabaseSchema
from culifeed.database.models import Article, Topic, Channel, ChatType
from culifeed.storage.article_repository import ArticleRepository
from culifeed.storage.topic_repository import TopicRepository
from culifeed.utils.exceptions import DatabaseError


class TestArticleRepository:
    """Test suite for ArticleRepository."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_file.name
        temp_file.close()
        
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
    def article_repo(self, temp_db):
        """Create ArticleRepository with test database."""
        db_connection = DatabaseConnection(temp_db, pool_size=2)
        repo = ArticleRepository(db_connection)
        yield repo
        db_connection.close_all_connections()
    
    @pytest.fixture
    def sample_articles(self):
        """Create sample articles for testing."""
        return [
            Article(
                title="Test Article 1",
                url="https://example.com/article1",
                content="This is test content for article 1",
                source_feed="https://example.com/feed1.xml"
            ),
            Article(
                title="Test Article 2", 
                url="https://example.com/article2",
                content="This is test content for article 2",
                source_feed="https://example.com/feed2.xml"
            )
        ]
    
    def test_create_article(self, article_repo, sample_articles):
        """Test article creation."""
        article = sample_articles[0]
        article_id = article_repo.create_article(article)
        
        assert article_id == article.id
        
        # Verify article was created
        retrieved = article_repo.get_article(article_id)
        assert retrieved is not None
        assert retrieved.title == article.title
        assert str(retrieved.url) == str(article.url)
        assert retrieved.content == article.content
    
    def test_create_articles_batch(self, article_repo, sample_articles):
        """Test batch article creation."""
        created_count = article_repo.create_articles_batch(sample_articles)
        
        assert created_count == len(sample_articles)
        
        # Verify all articles were created
        for article in sample_articles:
            retrieved = article_repo.get_article(article.id)
            assert retrieved is not None
            assert retrieved.title == article.title
    
    def test_create_articles_batch_empty(self, article_repo):
        """Test batch creation with empty list."""
        created_count = article_repo.create_articles_batch([])
        assert created_count == 0
    
    def test_get_article_not_found(self, article_repo):
        """Test getting non-existent article."""
        result = article_repo.get_article("nonexistent-id")
        assert result is None
    
    def test_get_articles_by_feed(self, article_repo, sample_articles):
        """Test getting articles by feed."""
        # Create articles
        article_repo.create_articles_batch(sample_articles)
        
        # Test getting articles from specific feed
        feed_url = sample_articles[0].source_feed
        articles = article_repo.get_articles_by_feed(feed_url)
        
        assert len(articles) == 1
        assert articles[0].source_feed == feed_url
    
    def test_get_recent_articles(self, article_repo, sample_articles):
        """Test getting recent articles."""
        # Create articles
        article_repo.create_articles_batch(sample_articles)
        
        # Get recent articles
        recent = article_repo.get_recent_articles(hours=1)
        assert len(recent) == len(sample_articles)
        
        # Test with very short time window
        old_recent = article_repo.get_recent_articles(hours=0)
        assert len(old_recent) == 0
    
    def test_find_by_content_hash(self, article_repo, sample_articles):
        """Test finding article by content hash."""
        article = sample_articles[0]
        article_repo.create_article(article)
        
        # Find by content hash
        found = article_repo.find_by_content_hash(article.content_hash)
        assert found is not None
        assert found.id == article.id
        
        # Test with non-existent hash
        not_found = article_repo.find_by_content_hash("nonexistent-hash")
        assert not_found is None
    
    def test_check_duplicates(self, article_repo, sample_articles):
        """Test duplicate checking."""
        # Create one article
        article_repo.create_article(sample_articles[0])
        
        # Check duplicates with both articles (one existing, one new)
        unique = article_repo.check_duplicates(sample_articles)
        
        # Should only return the non-duplicate article
        assert len(unique) == 1
        assert unique[0].id == sample_articles[1].id
    
    def test_update_article(self, article_repo, sample_articles):
        """Test article updates."""
        article = sample_articles[0]
        article_repo.create_article(article)
        
        # Update article
        updates = {'title': 'Updated Title', 'content': 'Updated content'}
        success = article_repo.update_article(article.id, updates)
        assert success
        
        # Verify updates
        updated = article_repo.get_article(article.id)
        assert updated.title == 'Updated Title'
        assert updated.content == 'Updated content'
    
    def test_update_article_empty(self, article_repo, sample_articles):
        """Test update with empty changes."""
        article = sample_articles[0]
        article_repo.create_article(article)
        
        success = article_repo.update_article(article.id, {})
        assert success
    
    def test_delete_article(self, article_repo, sample_articles):
        """Test article deletion."""
        article = sample_articles[0]
        article_repo.create_article(article)
        
        # Delete article
        success = article_repo.delete_article(article.id)
        assert success
        
        # Verify deletion
        deleted = article_repo.get_article(article.id)
        assert deleted is None
    
    def test_delete_nonexistent_article(self, article_repo):
        """Test deleting non-existent article."""
        success = article_repo.delete_article("nonexistent-id")
        assert not success
    
    def test_delete_old_articles(self, article_repo, sample_articles):
        """Test deleting old articles."""
        # Create articles
        article_repo.create_articles_batch(sample_articles)
        
        # Delete articles older than 0 days (should delete all)
        deleted_count = article_repo.delete_old_articles(days_to_keep=0)
        assert deleted_count == len(sample_articles)
        
        # Verify articles were deleted
        count = article_repo.get_article_count()
        assert count == 0
    
    def test_get_article_count(self, article_repo, sample_articles):
        """Test article count."""
        assert article_repo.get_article_count() == 0
        
        article_repo.create_articles_batch(sample_articles)
        assert article_repo.get_article_count() == len(sample_articles)
    
    def test_get_feed_article_stats(self, article_repo, sample_articles):
        """Test feed statistics."""
        article_repo.create_articles_batch(sample_articles)
        
        stats = article_repo.get_feed_article_stats()
        assert len(stats) == 2  # Two different feeds
        assert stats[sample_articles[0].source_feed] == 1
        assert stats[sample_articles[1].source_feed] == 1


class TestTopicRepository:
    """Test suite for TopicRepository."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_file.name
        temp_file.close()
        
        # Create schema
        schema = DatabaseSchema(db_path)
        schema.create_tables()
        
        # Create test channel (required for foreign key)
        db_conn = DatabaseConnection(db_path, pool_size=2)
        with db_conn.get_connection() as conn:
            conn.execute('''
                INSERT INTO channels (chat_id, chat_title, chat_type, registered_at, active, created_at)
                VALUES (?, ?, ?, datetime('now'), ?, datetime('now'))
            ''', ('-1001234567890', 'Test Channel', 'supergroup', True))
            conn.commit()
        db_conn.close_all_connections()
        
        yield db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def topic_repo(self, temp_db):
        """Create TopicRepository with test database."""
        db_connection = DatabaseConnection(temp_db, pool_size=2)
        repo = TopicRepository(db_connection)
        yield repo
        db_connection.close_all_connections()
    
    @pytest.fixture
    def sample_topics(self):
        """Create sample topics for testing."""
        return [
            Topic(
                chat_id='-1001234567890',
                name='Technology',
                keywords=['tech', 'innovation', 'gadgets'],
                exclude_keywords=['spam', 'ads'],
                confidence_threshold=0.8
            ),
            Topic(
                chat_id='-1001234567890',
                name='Programming',
                keywords=['code', 'programming', 'software'],
                exclude_keywords=['beginner'],
                confidence_threshold=0.7
            )
        ]
    
    def test_create_topic(self, topic_repo, sample_topics):
        """Test topic creation."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        assert isinstance(topic_id, int)
        assert topic_id > 0
        
        # Verify topic was created
        retrieved = topic_repo.get_topic(topic_id)
        assert retrieved is not None
        assert retrieved.name == topic.name
        assert retrieved.keywords == topic.keywords
        assert retrieved.exclude_keywords == topic.exclude_keywords
    
    def test_get_topic_not_found(self, topic_repo):
        """Test getting non-existent topic."""
        result = topic_repo.get_topic(999)
        assert result is None
    
    def test_get_topic_by_name(self, topic_repo, sample_topics):
        """Test getting topic by name."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        found = topic_repo.get_topic_by_name(topic.chat_id, topic.name)
        assert found is not None
        assert found.id == topic_id
        assert found.name == topic.name
    
    def test_get_topic_by_name_not_found(self, topic_repo):
        """Test getting non-existent topic by name."""
        result = topic_repo.get_topic_by_name('-1001234567890', 'Nonexistent')
        assert result is None
    
    def test_get_topics_for_chat(self, topic_repo, sample_topics):
        """Test getting all topics for a chat."""
        # Create topics
        for topic in sample_topics:
            topic_repo.create_topic(topic)
        
        # Get topics for chat
        chat_topics = topic_repo.get_topics_for_chat('-1001234567890')
        assert len(chat_topics) == len(sample_topics)
        
        # Test with non-existent chat
        empty_topics = topic_repo.get_topics_for_chat('-1001111111111')
        assert len(empty_topics) == 0
    
    def test_get_topics_for_chat_active_only(self, topic_repo, sample_topics):
        """Test getting only active topics."""
        # Create topics
        for topic in sample_topics:
            topic_id = topic_repo.create_topic(topic)
            if topic.name == 'Programming':
                # Deactivate one topic
                topic_repo.deactivate_topic(topic_id)
        
        # Get only active topics
        active_topics = topic_repo.get_topics_for_chat('-1001234567890', active_only=True)
        assert len(active_topics) == 1
        assert active_topics[0].name == 'Technology'
        
        # Get all topics
        all_topics = topic_repo.get_topics_for_chat('-1001234567890', active_only=False)
        assert len(all_topics) == 2
    
    def test_get_all_active_topics(self, topic_repo, sample_topics):
        """Test getting all active topics across chats."""
        for topic in sample_topics:
            topic_repo.create_topic(topic)
        
        active_topics = topic_repo.get_all_active_topics()
        assert len(active_topics) == len(sample_topics)
    
    def test_update_topic(self, topic_repo, sample_topics):
        """Test topic updates."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        # Update topic
        updates = {
            'name': 'Updated Technology',
            'keywords': ['updated', 'keywords'],
            'confidence_threshold': 0.9
        }
        success = topic_repo.update_topic(topic_id, updates)
        assert success
        
        # Verify updates
        updated = topic_repo.get_topic(topic_id)
        assert updated.name == 'Updated Technology'
        assert set(updated.keywords) == {'updated', 'keywords'}  # Order doesn't matter for keywords
        assert updated.confidence_threshold == 0.9
    
    def test_update_topic_empty(self, topic_repo, sample_topics):
        """Test update with empty changes."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        success = topic_repo.update_topic(topic_id, {})
        assert success
    
    def test_update_last_match(self, topic_repo, sample_topics):
        """Test updating last match timestamp."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        success = topic_repo.update_last_match(topic_id)
        assert success
        
        # Verify timestamp was updated
        updated = topic_repo.get_topic(topic_id)
        assert updated.last_match_at is not None
    
    def test_activate_deactivate_topic(self, topic_repo, sample_topics):
        """Test topic activation/deactivation."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        # Deactivate topic
        success = topic_repo.deactivate_topic(topic_id)
        assert success
        
        deactivated = topic_repo.get_topic(topic_id)
        assert not deactivated.active
        
        # Reactivate topic
        success = topic_repo.activate_topic(topic_id)
        assert success
        
        activated = topic_repo.get_topic(topic_id)
        assert activated.active
    
    def test_delete_topic(self, topic_repo, sample_topics):
        """Test topic deletion."""
        topic = sample_topics[0]
        topic_id = topic_repo.create_topic(topic)
        
        # Delete topic
        success = topic_repo.delete_topic(topic_id)
        assert success
        
        # Verify deletion
        deleted = topic_repo.get_topic(topic_id)
        assert deleted is None
    
    def test_delete_nonexistent_topic(self, topic_repo):
        """Test deleting non-existent topic."""
        success = topic_repo.delete_topic(999)
        assert not success
    
    def test_delete_topics_for_chat(self, topic_repo, sample_topics):
        """Test deleting all topics for a chat."""
        for topic in sample_topics:
            topic_repo.create_topic(topic)
        
        deleted_count = topic_repo.delete_topics_for_chat('-1001234567890')
        assert deleted_count == len(sample_topics)
        
        # Verify deletion
        remaining = topic_repo.get_topics_for_chat('-1001234567890')
        assert len(remaining) == 0
    
    def test_search_topics(self, topic_repo, sample_topics):
        """Test topic search functionality."""
        for topic in sample_topics:
            topic_repo.create_topic(topic)
        
        # Search by name
        results = topic_repo.search_topics('tech', '-1001234567890')
        assert len(results) == 1
        assert results[0].name == 'Technology'
        
        # Search by keyword
        results = topic_repo.search_topics('code', '-1001234567890')
        assert len(results) == 1
        assert results[0].name == 'Programming'
        
        # Search with no results
        results = topic_repo.search_topics('nonexistent', '-1001234567890')
        assert len(results) == 0
    
    def test_get_topic_statistics(self, topic_repo, sample_topics):
        """Test topic statistics."""
        for topic in sample_topics:
            topic_repo.create_topic(topic)
        
        stats = topic_repo.get_topic_statistics('-1001234567890')
        
        assert stats['total_topics'] == 2
        assert stats['active_topics'] == 2
        assert stats['inactive_topics'] == 0
        assert stats['avg_confidence_threshold'] == 0.75  # (0.8 + 0.7) / 2
        assert stats['avg_keywords_per_topic'] == 3.0  # Both have 3 keywords
        assert stats['max_keywords_per_topic'] == 3
        assert stats['topics_with_recent_matches'] == 0
    
    def test_topic_json_parsing(self, topic_repo):
        """Test JSON parsing of keywords."""
        topic = Topic(
            chat_id='-1001234567890',
            name='Test Topic',
            keywords=['test', 'json', 'parsing'],
            exclude_keywords=['exclude', 'test']
        )
        
        topic_id = topic_repo.create_topic(topic)
        retrieved = topic_repo.get_topic(topic_id)
        
        # Verify keywords were stored and retrieved correctly
        assert isinstance(retrieved.keywords, list)
        assert isinstance(retrieved.exclude_keywords, list)
        # Sort lists before comparison since JSON parsing may change order
        assert sorted(retrieved.keywords) == sorted(topic.keywords)
        assert sorted(retrieved.exclude_keywords) == sorted(topic.exclude_keywords)


class TestRepositoryIntegration:
    """Integration tests for repository components."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        db_path = temp_file.name
        temp_file.close()
        
        # Create schema
        schema = DatabaseSchema(db_path)
        schema.create_tables()
        
        # Create test channel
        db_conn = DatabaseConnection(db_path, pool_size=2)
        with db_conn.get_connection() as conn:
            conn.execute('''
                INSERT INTO channels (chat_id, chat_title, chat_type, registered_at, active, created_at)
                VALUES (?, ?, ?, datetime('now'), ?, datetime('now'))
            ''', ('-1001234567890', 'Test Channel', 'supergroup', True))
            conn.commit()
        db_conn.close_all_connections()
        
        yield db_path
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def repos(self, temp_db):
        """Create both repositories with shared database."""
        db_connection = DatabaseConnection(temp_db, pool_size=2)
        article_repo = ArticleRepository(db_connection)
        topic_repo = TopicRepository(db_connection)
        yield article_repo, topic_repo
        db_connection.close_all_connections()
    
    def test_repository_integration(self, repos):
        """Test that both repositories work together."""
        article_repo, topic_repo = repos
        
        # Create a topic
        topic = Topic(
            chat_id='-1001234567890',
            name='Integration Test',
            keywords=['integration', 'test'],
            confidence_threshold=0.8
        )
        topic_id = topic_repo.create_topic(topic)
        
        # Create an article
        article = Article(
            title="Integration Test Article",
            url="https://example.com/integration",
            content="This is an integration test article",
            source_feed="https://example.com/feed.xml"
        )
        article_id = article_repo.create_article(article)
        
        # Verify both were created
        assert topic_repo.get_topic(topic_id) is not None
        assert article_repo.get_article(article_id) is not None
        
        # Test statistics
        topic_stats = topic_repo.get_topic_statistics('-1001234567890')
        assert topic_stats['total_topics'] == 1
        
        article_count = article_repo.get_article_count()
        assert article_count == 1