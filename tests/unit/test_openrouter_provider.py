"""
OpenRouter Provider Tests
========================

Unit tests for OpenRouter AI provider integration with FREE models.
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from culifeed.ai.providers.openrouter_provider import OpenRouterProvider
from culifeed.ai.providers.base import AIResult, AIProviderType
from culifeed.database.models import Article, Topic
from culifeed.utils.exceptions import ErrorCode


class TestOpenRouterProvider:
    """Test suite for OpenRouter provider."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client for OpenRouter."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        mock_client.close = AsyncMock()
        return mock_client

    @pytest.fixture
    def provider(self, mock_openai_client):
        """Create OpenRouter provider with mocked client."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI', return_value=mock_openai_client):
            provider = OpenRouterProvider(api_key="test-key")
            provider.client = mock_openai_client
            return provider

    @pytest.fixture
    def sample_article(self):
        """Create sample article for testing."""
        return Article(
            id="test-article-1",
            title="AWS ECS Service Connect adds cross-account support",
            url="https://aws.amazon.com/about-aws/whats-new/ecs-service-connect/",
            content="Amazon ECS Service Connect now supports cross-account workloads...",
            published_at=datetime.now(timezone.utc),
            source_feed="https://aws.amazon.com/feed/",
            content_hash="test-hash",
            created_at=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_topic(self):
        """Create sample topic for testing."""
        return Topic(
            chat_id="test-channel",
            name="AWS ECS Technical Updates",
            keywords=["aws ecs", "container orchestration", "microservices"],
            confidence_threshold=0.7,
            active=True
        )

    def test_provider_initialization(self):
        """Test OpenRouter provider initialization."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            provider = OpenRouterProvider(api_key="test-key")

            assert provider.provider_type == AIProviderType.OPENROUTER
            assert provider.model_name == "meta-llama/llama-3.2-3b-instruct:free"  # Default free model
            assert len(provider.available_models) >= 2  # Should have multiple free models

            # Verify all models are free models (end with :free)
            for model in provider.available_models:
                assert model.endswith(":free"), f"Model {model} is not a free model"

    def test_rate_limits_free_plan(self):
        """Test that rate limits are correctly set for free plan."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            provider = OpenRouterProvider(api_key="test-key")
            rate_limits = provider.get_rate_limits()

            assert rate_limits.requests_per_minute == 20  # Free model limit
            assert rate_limits.requests_per_day == 50     # Free plan daily limit

    @pytest.mark.asyncio
    async def test_analyze_relevance_success(self, provider, mock_openai_client, sample_article, sample_topic):
        """Test successful relevance analysis."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "relevance_score": 0.85,
            "confidence": 0.9,
            "reasoning": "Article discusses AWS ECS cross-account features, matching topic keywords"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await provider.analyze_relevance(sample_article, sample_topic)

        assert result.success is True
        assert result.relevance_score == 0.85
        assert result.confidence == 0.9
        assert "AWS ECS" in result.reasoning
        assert result.provider == "openrouter"
        assert result.model_used == "meta-llama/llama-3.2-3b-instruct:free"

    @pytest.mark.asyncio
    async def test_analyze_relevance_fallback_to_second_model(self, provider, mock_openai_client, sample_article, sample_topic):
        """Test fallback to second model when first fails."""
        # First call fails, second succeeds
        mock_openai_client.chat.completions.create.side_effect = [
            Exception("First model failed"),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"relevance_score": 0.7, "confidence": 0.8, "reasoning": "Fallback success"}'))])
        ]

        result = await provider.analyze_relevance(sample_article, sample_topic)

        assert result.success is True
        assert result.relevance_score == 0.7
        assert mock_openai_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_summary_success(self, provider, mock_openai_client, sample_article):
        """Test successful summary generation."""
        # Mock successful API response - using same pattern as analyze_relevance
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This article discusses AWS ECS Service Connect's new cross-account support feature."
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = await provider.generate_summary(sample_article)

        assert result.success is True
        assert "AWS ECS Service Connect" in result.content
        assert result.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, provider, mock_openai_client, sample_article, sample_topic):
        """Test rate limit handling."""
        # Simulate rate limit exhaustion
        provider._request_count_minute = 25  # Exceed 20 req/min limit

        result = await provider.analyze_relevance(sample_article, sample_topic)

        assert result.success is False
        assert "Rate limit exceeded" in result.error_message

    @pytest.mark.asyncio
    async def test_connection_test(self, provider, mock_openai_client):
        """Test connection testing functionality."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_openai_client.chat.completions.create.return_value = mock_response

        is_connected = await provider.test_connection()

        assert is_connected is True

    @pytest.mark.asyncio
    async def test_connection_test_failure(self, provider, mock_openai_client):
        """Test connection test failure handling."""
        mock_openai_client.chat.completions.create.side_effect = Exception("Connection failed")

        is_connected = await provider.test_connection()

        assert is_connected is False

    def test_parse_relevance_response_malformed_json(self, provider):
        """Test handling of malformed JSON response."""
        malformed_response = "This is not JSON at all"

        result = provider._parse_relevance_response(malformed_response)

        # Should return fallback values
        assert result["relevance_score"] == 0.5
        assert result["confidence"] == 0.3
        assert "fallback" in result["reasoning"].lower()

    def test_free_models_configuration(self):
        """Test that recommended models are all free variants."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            provider = OpenRouterProvider(api_key="test-key")

            # All recommended models should end with :free
            for model in provider.RECOMMENDED_MODELS:
                assert model.endswith(":free"), f"Model {model} is not a free variant"

            # Should have multiple free models for fallback
            assert len(provider.RECOMMENDED_MODELS) >= 2  # At least 2 models for fallback

    @pytest.mark.asyncio
    async def test_cleanup(self, provider, mock_openai_client):
        """Test proper cleanup of resources."""
        await provider.close()

        mock_openai_client.close.assert_called_once()


class TestOpenRouterIntegration:
    """Integration tests for OpenRouter with AI Manager."""

    @pytest.mark.skipif(not os.getenv('OPENROUTER_API_KEY'), reason="OpenRouter API key not available")
    @pytest.mark.asyncio
    async def test_real_openrouter_connection(self):
        """Test real connection to OpenRouter (requires API key)."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        provider = OpenRouterProvider(api_key=api_key)

        try:
            is_connected = await provider.test_connection()
            assert is_connected is True
        finally:
            await provider.close()

    def test_ai_manager_integration(self):
        """Test that OpenRouter integrates properly with AI Manager."""
        from culifeed.ai.ai_manager import AIManager
        from culifeed.config.settings import CuliFeedSettings

        # Mock settings with OpenRouter key
        settings = CuliFeedSettings()
        settings.ai.openrouter_api_key = "test-key"

        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            ai_manager = AIManager(settings=settings)

            # Should have OpenRouter provider available
            assert AIProviderType.OPENROUTER in ai_manager.providers
            assert AIProviderType.OPENROUTER in ai_manager.provider_health

            # Provider should be marked as healthy
            health = ai_manager.provider_health[AIProviderType.OPENROUTER]
            assert health.available is True
            assert health.is_healthy is True


# Additional tests for specific free model behaviors
class TestFreeModelLimitations:
    """Test specific behaviors related to free model limitations."""

    def test_free_model_naming_convention(self):
        """Verify all free models follow :free naming convention."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            provider = OpenRouterProvider(api_key="test-key")

            for model in provider.RECOMMENDED_MODELS:
                assert ":free" in model, f"Model {model} doesn't follow free naming convention"

    def test_rate_limit_configuration_matches_free_tier(self):
        """Verify rate limits match OpenRouter free tier specifications."""
        with patch('culifeed.ai.providers.openrouter_provider.openai.AsyncOpenAI'):
            provider = OpenRouterProvider(api_key="test-key")
            limits = provider.get_rate_limits()

            # OpenRouter free tier: 20 req/min for :free models, 50 req/day total
            assert limits.requests_per_minute == 20
            assert limits.requests_per_day == 50