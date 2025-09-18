#!/usr/bin/env python3
"""
Unit Tests for AI Keyword Generation Feature
=============================================

Tests the integrated AI keyword generation functionality in TopicCommandHandler
using AIManager pattern with provider fallback.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from culifeed.ai.ai_manager import AIManager
from culifeed.ai.providers.base import AIResult
from culifeed.bot.commands.topic_commands import TopicCommandHandler
from culifeed.database.connection import DatabaseConnection
from culifeed.storage.topic_repository import TopicRepository


class TestAIKeywordGeneration:
    """Test AI keyword generation functionality."""

    @pytest.fixture
    def mock_db_connection(self):
        """Create a mock database connection."""
        return MagicMock(spec=DatabaseConnection)

    @pytest.fixture
    def mock_topic_repo(self):
        """Create a mock topic repository."""
        return MagicMock(spec=TopicRepository)

    @pytest.fixture
    def mock_ai_manager(self):
        """Create a mock AI manager."""
        return MagicMock(spec=AIManager)

    @pytest.fixture
    def topic_handler(self, mock_db_connection, mock_topic_repo, mock_ai_manager):
        """Create a topic command handler with mocked dependencies."""
        handler = TopicCommandHandler(mock_db_connection)
        handler.topic_repo = mock_topic_repo
        handler.ai_manager = mock_ai_manager
        return handler

    @pytest.mark.asyncio
    async def test_generate_keywords_with_ai_success(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test successful AI keyword generation."""
        # Mock existing topics
        mock_topic_repo.get_topics_for_channel.return_value = []
        
        # Mock AI manager success response
        mock_result = AIResult(
            success=True,
            relevance_score=1.0,
            confidence=0.9,
            content=['machine learning', 'artificial intelligence', 'deep learning', 'neural networks']
        )
        mock_ai_manager.generate_keywords.return_value = mock_result

        # Test keyword generation
        keywords = await topic_handler._generate_keywords_with_ai("Machine Learning", "test_chat_123")

        # Verify results
        assert isinstance(keywords, list)
        assert len(keywords) == 4
        assert 'machine learning' in keywords
        assert 'artificial intelligence' in keywords
        
        # Verify AI manager was called correctly
        mock_ai_manager.generate_keywords.assert_called_once_with(
            "Machine Learning", 
            "", 
            max_keywords=7
        )

    @pytest.mark.asyncio
    async def test_generate_keywords_with_context(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test AI keyword generation with existing topic context."""
        # Mock existing topics with proper name attributes
        mock_topic1 = MagicMock()
        mock_topic1.name = "Python Programming"
        mock_topic2 = MagicMock()
        mock_topic2.name = "Web Development"
        mock_topics = [mock_topic1, mock_topic2]
        mock_topic_repo.get_topics_for_channel.return_value = mock_topics
        
        # Mock AI manager success response
        mock_result = AIResult(
            success=True,
            relevance_score=1.0,
            confidence=0.9,
            content=['react', 'javascript', 'frontend', 'components', 'hooks']
        )
        mock_ai_manager.generate_keywords.return_value = mock_result

        # Test keyword generation with context
        keywords = await topic_handler._generate_keywords_with_ai("React Development", "test_chat_123")

        # Verify results
        assert isinstance(keywords, list)
        assert len(keywords) == 5
        
        # Verify context was included
        mock_ai_manager.generate_keywords.assert_called_once_with(
            "React Development",
            " User interests: Python Programming, Web Development.",
            max_keywords=7
        )

    @pytest.mark.asyncio
    async def test_generate_keywords_ai_failure_fallback(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test fallback when AI keyword generation fails."""
        # Mock existing topics
        mock_topic_repo.get_topics_for_channel.return_value = []
        
        # Mock AI manager failure response
        mock_result = AIResult(
            success=False,
            relevance_score=0.0,
            confidence=0.0,
            error_message="API rate limit exceeded"
        )
        mock_ai_manager.generate_keywords.return_value = mock_result

        # Test keyword generation fallback
        keywords = await topic_handler._generate_keywords_with_ai("Cloud Computing", "test_chat_123")

        # Verify fallback results
        assert isinstance(keywords, list)
        assert len(keywords) == 2
        assert keywords[0] == "cloud computing"
        assert keywords[1] == "cloud computing technology"

    @pytest.mark.asyncio
    async def test_generate_keywords_exception_fallback(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test fallback when AI keyword generation raises exception."""
        # Mock existing topics
        mock_topic_repo.get_topics_for_channel.return_value = []
        
        # Mock AI manager exception
        mock_ai_manager.generate_keywords.side_effect = Exception("Network error")

        # Test keyword generation exception handling
        keywords = await topic_handler._generate_keywords_with_ai("DevOps", "test_chat_123")

        # Verify fallback results
        assert isinstance(keywords, list)
        assert len(keywords) == 2
        assert keywords[0] == "devops"
        assert keywords[1] == "devops technology"

    @pytest.mark.asyncio
    async def test_generate_keywords_empty_content_fallback(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test fallback when AI returns empty content."""
        # Mock existing topics  
        mock_topic_repo.get_topics_for_channel.return_value = []
        
        # Mock AI manager with empty content
        mock_result = AIResult(
            success=True,
            relevance_score=1.0,
            confidence=0.9,
            content=[]  # Empty keywords
        )
        mock_ai_manager.generate_keywords.return_value = mock_result

        # Test keyword generation with empty content
        keywords = await topic_handler._generate_keywords_with_ai("Blockchain", "test_chat_123")

        # Verify fallback to default fallback keywords
        assert isinstance(keywords, list)
        assert len(keywords) == 2  # Falls back to [topic.lower(), f"{topic.lower()} technology"]
        assert keywords[0] == "blockchain"
        assert keywords[1] == "blockchain technology"

    @pytest.mark.asyncio
    async def test_generate_keywords_limits_to_max(self, topic_handler, mock_ai_manager, mock_topic_repo):
        """Test that keyword generation respects max limit."""
        # Mock existing topics
        mock_topic_repo.get_topics_for_channel.return_value = []
        
        # Mock AI manager with more than 7 keywords
        mock_result = AIResult(
            success=True,
            relevance_score=1.0,
            confidence=0.9,
            content=[f'keyword{i}' for i in range(10)]  # 10 keywords
        )
        mock_ai_manager.generate_keywords.return_value = mock_result

        # Test keyword generation with limit
        keywords = await topic_handler._generate_keywords_with_ai("Test Topic", "test_chat_123")

        # Verify limit is enforced
        assert isinstance(keywords, list)
        assert len(keywords) <= 7  # Should be limited to 7

    def test_parse_add_topic_args_ai_generation(self, topic_handler):
        """Test parsing arguments for AI keyword generation."""
        # Test single topic name (no commas) - should trigger AI generation
        result = topic_handler._parse_add_topic_args(["Machine", "Learning"])
        assert result is not None
        topic_name, keywords = result
        assert topic_name == "Machine Learning"
        assert keywords is None  # None indicates AI generation

    def test_parse_add_topic_args_manual_keywords(self, topic_handler):
        """Test parsing arguments for manual keyword specification."""
        # Test with commas - should trigger manual keywords
        result = topic_handler._parse_add_topic_args(["Cloud,", "AWS,", "Azure,", "GCP"])
        assert result is not None
        topic_name, keywords = result
        assert topic_name == "Cloud"
        assert keywords == ["AWS", "Azure", "GCP"]

    def test_parse_add_topic_args_single_word_ai(self, topic_handler):
        """Test parsing single word topic for AI generation."""
        # Test single word - should trigger AI generation
        result = topic_handler._parse_add_topic_args(["Python"])
        assert result is not None
        topic_name, keywords = result
        assert topic_name == "Python"
        assert keywords is None  # None indicates AI generation