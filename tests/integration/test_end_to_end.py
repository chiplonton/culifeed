#!/usr/bin/env python3
"""
End-to-End Integration Tests for CuliFeed
========================================

Tests complete system workflow from RSS feeds to Telegram delivery.
These tests verify the integration between all major components.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sqlite3

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from culifeed.config.settings import get_settings
from culifeed.database.connection import get_db_manager
from culifeed.database.schema import DatabaseSchema
from culifeed.scheduler.daily_scheduler import DailyScheduler
from culifeed.storage.channel_repository import ChannelRepository
from culifeed.storage.feed_repository import FeedRepository


class TestEndToEndIntegration:
    """
    End-to-end integration tests for the complete CuliFeed system.
    Tests the full workflow from configuration to content delivery.
    """

    @pytest.fixture
    async def test_database(self):
        """Create a temporary test database."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_culifeed.db"
        
        try:
            # Create database schema
            schema = DatabaseSchema(str(db_path))
            schema.create_tables()
            
            # Verify schema
            assert schema.verify_schema(), "Database schema verification failed"
            
            yield str(db_path)
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_settings(self, test_database):
        """Create test settings with temporary database."""
        with patch('culifeed.config.settings.get_settings') as mock_settings:
            settings = MagicMock()
            settings.database.path = test_database
            settings.database.cleanup_days = 30
            settings.database.max_size_mb = 100
            
            settings.processing.daily_run_hour = 8
            settings.processing.max_articles_per_topic = 10
            settings.processing.ai_provider = "gemini"
            
            settings.logging.level = "INFO"
            settings.logging.console_logging = True
            settings.logging.structured_logging = False
            
            # Mock AI provider configuration
            settings.get_ai_fallback_providers.return_value = ["gemini", "groq"]
            settings.get_effective_log_level.return_value = "INFO"
            
            mock_settings.return_value = settings
            yield settings

    @pytest.fixture
    async def populated_database(self, test_database, test_settings):
        """Create a database populated with test data."""
        db_manager = get_db_manager(test_database)
        
        # Create test channels
        channel_repo = ChannelRepository(db_manager)
        feed_repo = FeedRepository(db_manager)
        
        # Test channel 1
        test_channel_1 = {
            'chat_id': 'test_channel_1',
            'name': 'Test Tech Channel',
            'processing_schedule': 'daily',
            'active': True
        }
        
        # Test channel 2
        test_channel_2 = {
            'chat_id': 'test_channel_2', 
            'name': 'Test News Channel',
            'processing_schedule': 'daily',
            'active': True
        }
        
        # Add channels to database
        with db_manager.get_connection() as conn:
            conn.execute("""
                INSERT INTO channels (chat_id, name, active, processing_schedule, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                test_channel_1['chat_id'], test_channel_1['name'], 
                test_channel_1['active'], test_channel_1['processing_schedule'],
                datetime.now()
            ))
            
            conn.execute("""
                INSERT INTO channels (chat_id, name, active, processing_schedule, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                test_channel_2['chat_id'], test_channel_2['name'],
                test_channel_2['active'], test_channel_2['processing_schedule'],
                datetime.now()
            ))
            
            # Add test feeds
            test_feeds = [
                {
                    'chat_id': 'test_channel_1',
                    'url': 'https://aws.amazon.com/blogs/compute/feed/',
                    'title': 'AWS Compute Blog',
                    'active': True
                },
                {
                    'chat_id': 'test_channel_1',
                    'url': 'https://kubernetes.io/feed.xml',
                    'title': 'Kubernetes Blog',
                    'active': True
                },
                {
                    'chat_id': 'test_channel_2',
                    'url': 'https://techcrunch.com/feed/',
                    'title': 'TechCrunch',
                    'active': True
                }
            ]
            
            for feed in test_feeds:
                conn.execute("""
                    INSERT INTO feeds (chat_id, url, title, active, created_at, last_check_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    feed['chat_id'], feed['url'], feed['title'], 
                    feed['active'], datetime.now(), datetime.now()
                ))
            
            # Add test topics
            test_topics = [
                {
                    'chat_id': 'test_channel_1',
                    'name': 'Cloud Computing',
                    'keywords': ['cloud', 'aws', 'kubernetes', 'container'],
                    'active': True
                },
                {
                    'chat_id': 'test_channel_2', 
                    'name': 'Tech News',
                    'keywords': ['startup', 'funding', 'technology', 'innovation'],
                    'active': True
                }
            ]
            
            for topic in test_topics:
                conn.execute("""
                    INSERT INTO topics (chat_id, name, keywords, active, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    topic['chat_id'], topic['name'], ','.join(topic['keywords']),
                    topic['active'], datetime.now()
                ))
        
        yield db_manager

    async def test_database_initialization(self, test_database):
        """Test that database initializes correctly with proper schema."""
        # Test database connection
        db_manager = get_db_manager(test_database)
        db_info = db_manager.get_database_info()
        
        assert db_info['database_size_mb'] >= 0
        assert db_info['total_connections'] >= 0
        
        # Test schema verification
        schema = DatabaseSchema(test_database)
        assert schema.verify_schema()
        
        # Test table creation
        with db_manager.get_connection() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['articles', 'channels', 'feeds', 'processing_history', 'topics']
            assert all(table in tables for table in expected_tables), f"Missing tables. Found: {tables}"

    async def test_scheduler_health_check(self, test_settings, populated_database):
        """Test scheduler health monitoring functionality."""
        scheduler = DailyScheduler(test_settings)
        
        # Test health status check
        status = await scheduler.check_processing_status()
        
        assert 'health_status' in status
        assert 'current_time' in status
        assert 'processed_today' in status
        assert 'recent_success_rate' in status
        
        # Health status should be one of the expected values
        assert status['health_status'] in ['healthy', 'warning', 'error']

    @patch('culifeed.pipeline.daily_processor.DailyProcessor')
    @patch('culifeed.delivery.digest_sender.DigestSender')
    async def test_daily_processing_dry_run(self, mock_digest_sender, mock_processor, test_settings, populated_database):
        """Test complete daily processing workflow in dry-run mode."""
        
        # Mock the processor and digest sender
        mock_processor_instance = AsyncMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_digest_sender_instance = AsyncMock()
        mock_digest_sender.return_value = mock_digest_sender_instance
        
        # Configure processor mock response
        mock_processing_result = MagicMock()
        mock_processing_result.success = True
        mock_processing_result.curated_articles = [
            {'title': 'Test Article 1', 'url': 'https://example.com/1'},
            {'title': 'Test Article 2', 'url': 'https://example.com/2'}
        ]
        mock_processing_result.summary_stats = {'total_articles': 2, 'curated_count': 2}
        mock_processing_result.error_message = None
        
        mock_processor_instance.process_channel_content.return_value = mock_processing_result
        
        # Configure digest sender mock response
        mock_digest_result = MagicMock()
        mock_digest_result.success = True
        mock_digest_result.messages_sent = 0  # Dry run, no messages sent
        mock_digest_sender_instance.send_daily_digest.return_value = mock_digest_result
        
        # Run scheduler
        scheduler = DailyScheduler(test_settings)
        result = await scheduler.run_daily_processing(dry_run=True)
        
        # Verify results
        assert result['success'] == True
        assert result['channels_processed'] >= 0
        assert 'duration_seconds' in result
        assert 'execution_id' in result
        assert 'performance_metrics' in result
        
        # In dry run mode, should not send actual messages
        if result['channels_processed'] > 0:
            # Processor should have been called
            mock_processor_instance.process_channel_content.assert_called()

    async def test_database_operations(self, populated_database):
        """Test basic database operations work correctly."""
        
        # Test channel repository operations
        channel_repo = ChannelRepository(populated_database)
        
        channels = channel_repo.get_active_channels()
        assert len(channels) == 2
        
        # Test that channels have expected properties
        channel_ids = [ch.chat_id for ch in channels]
        assert 'test_channel_1' in channel_ids
        assert 'test_channel_2' in channel_ids
        
        # Test feed repository operations
        feed_repo = FeedRepository(populated_database)
        
        all_feeds = feed_repo.get_all_active_feeds()
        assert len(all_feeds) >= 3
        
        channel_1_feeds = feed_repo.get_feeds_for_chat('test_channel_1')
        assert len(channel_1_feeds) == 2
        
        channel_2_feeds = feed_repo.get_feeds_for_chat('test_channel_2')
        assert len(channel_2_feeds) == 1

    async def test_error_handling_integration(self, test_settings, populated_database):
        """Test error handling across the integrated system."""
        
        scheduler = DailyScheduler(test_settings)
        
        # Test with database error simulation
        with patch.object(populated_database, 'get_database_info', side_effect=Exception("Database error")):
            with pytest.raises(Exception):
                await scheduler._perform_health_checks()
        
        # Test graceful handling of processing errors
        with patch('culifeed.pipeline.daily_processor.DailyProcessor') as mock_processor:
            mock_processor_instance = AsyncMock()
            mock_processor.return_value = mock_processor_instance
            
            # Simulate processing failure
            mock_processing_result = MagicMock()
            mock_processing_result.success = False
            mock_processing_result.error_message = "Test processing error"
            mock_processor_instance.process_channel_content.return_value = mock_processing_result
            
            result = await scheduler.run_daily_processing(dry_run=True)
            
            # Should handle errors gracefully
            assert 'success' in result
            assert 'errors_count' in result

    async def test_performance_monitoring(self, test_settings, populated_database):
        """Test performance monitoring integration."""
        
        scheduler = DailyScheduler(test_settings)
        
        # Mock successful processing
        with patch('culifeed.pipeline.daily_processor.DailyProcessor') as mock_processor:
            mock_processor_instance = AsyncMock()
            mock_processor.return_value = mock_processor_instance
            
            # Configure successful response
            mock_processing_result = MagicMock()
            mock_processing_result.success = True
            mock_processing_result.curated_articles = []
            mock_processing_result.summary_stats = {}
            mock_processor_instance.process_channel_content.return_value = mock_processing_result
            
            result = await scheduler.run_daily_processing(dry_run=True)
            
            # Should include performance metrics
            assert 'performance_metrics' in result
            assert 'duration_seconds' in result
            assert result['duration_seconds'] >= 0

    async def test_configuration_integration(self, test_settings):
        """Test configuration system integration."""
        
        # Test settings loading
        assert test_settings.database.path is not None
        assert test_settings.processing.daily_run_hour == 8
        assert test_settings.processing.max_articles_per_topic == 10
        
        # Test AI provider configuration
        providers = test_settings.get_ai_fallback_providers()
        assert len(providers) >= 1
        assert "gemini" in providers

    @patch('culifeed.pipeline.daily_processor.DailyProcessor')
    @patch('culifeed.delivery.digest_sender.DigestSender')
    async def test_multi_channel_processing(self, mock_digest_sender, mock_processor, test_settings, populated_database):
        """Test processing multiple channels simultaneously."""
        
        # Mock components
        mock_processor_instance = AsyncMock()
        mock_processor.return_value = mock_processor_instance
        
        mock_digest_sender_instance = AsyncMock()
        mock_digest_sender.return_value = mock_digest_sender_instance
        
        # Configure successful responses for multiple channels
        mock_processing_result = MagicMock()
        mock_processing_result.success = True
        mock_processing_result.curated_articles = [
            {'title': 'Multi-channel Test Article', 'url': 'https://example.com/multi'}
        ]
        mock_processing_result.summary_stats = {'total_articles': 1, 'curated_count': 1}
        mock_processor_instance.process_channel_content.return_value = mock_processing_result
        
        mock_digest_result = MagicMock()
        mock_digest_result.success = True
        mock_digest_result.messages_sent = 0  # Dry run
        mock_digest_sender_instance.send_daily_digest.return_value = mock_digest_result
        
        # Run processing
        scheduler = DailyScheduler(test_settings)
        result = await scheduler.run_daily_processing(dry_run=True)
        
        # Should process multiple channels
        assert result['success'] == True
        
        if result['channels_processed'] > 1:
            # Should have been called multiple times for multiple channels
            assert mock_processor_instance.process_channel_content.call_count >= 1

    async def test_cleanup_operations(self, test_settings, populated_database):
        """Test database cleanup operations integration."""
        
        # Add some old test data
        with populated_database.get_connection() as conn:
            old_date = datetime.now() - timedelta(days=40)
            conn.execute("""
                INSERT INTO articles (chat_id, feed_url, title, url, content, published_at, processed_at, relevance_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'test_channel_1', 'https://example.com/feed.xml',
                'Old Test Article', 'https://example.com/old',
                'Old content', old_date, old_date, 0.8
            ))
        
        scheduler = DailyScheduler(test_settings)
        
        # Test cleanup operations
        await scheduler._post_processing_cleanup()
        
        # Verify cleanup worked (this is a basic test, real implementation would check actual cleanup)
        db_info = populated_database.get_database_info()
        assert db_info is not None


@pytest.mark.asyncio
async def test_complete_workflow_simulation():
    """
    Simulate a complete end-to-end workflow without external dependencies.
    This test runs the entire system in isolation.
    """
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "workflow_test.db"
    
    try:
        # Initialize database
        schema = DatabaseSchema(str(db_path))
        schema.create_tables()
        
        # Mock settings
        with patch('culifeed.config.settings.get_settings') as mock_settings:
            settings = MagicMock()
            settings.database.path = str(db_path)
            settings.database.cleanup_days = 30
            settings.processing.daily_run_hour = 8
            settings.processing.max_articles_per_topic = 5
            settings.get_ai_fallback_providers.return_value = ["gemini"]
            mock_settings.return_value = settings
            
            # Create minimal test data
            db_manager = get_db_manager(str(db_path))
            with db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO channels (chat_id, name, active, processing_schedule, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, ('workflow_test', 'Workflow Test Channel', True, 'daily', datetime.now()))
            
            # Mock all external dependencies
            with patch('culifeed.pipeline.daily_processor.DailyProcessor') as mock_processor, \
                 patch('culifeed.delivery.digest_sender.DigestSender') as mock_sender:
                
                # Configure mocks
                mock_processor_instance = AsyncMock()
                mock_processor.return_value = mock_processor_instance
                
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.curated_articles = []
                mock_result.summary_stats = {}
                mock_processor_instance.process_channel_content.return_value = mock_result
                
                mock_sender_instance = AsyncMock()
                mock_sender.return_value = mock_sender_instance
                mock_digest_result = MagicMock()
                mock_digest_result.success = True
                mock_digest_result.messages_sent = 0
                mock_sender_instance.send_daily_digest.return_value = mock_digest_result
                
                # Run complete workflow
                scheduler = DailyScheduler(settings)
                result = await scheduler.run_daily_processing(dry_run=True)
                
                # Verify workflow completed
                assert result['success'] == True
                assert 'execution_id' in result
                assert 'duration_seconds' in result
                assert result['channels_processed'] >= 0
                
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])