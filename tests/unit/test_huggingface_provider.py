#!/usr/bin/env python3
"""
HuggingFace Provider Unit Tests
==============================

Comprehensive unit tests for HuggingFace provider functionality including:
- Provider initialization and configuration
- Model switching and validation
- API response handling (mocked)
- Error handling and rate limiting
- Confirmed working models testing
"""

import sys
import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import aiohttp
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from culifeed.database.models import Article, Topic
from culifeed.ai.providers.base import AIResult, AIError, RateLimitInfo, AIProviderType
from culifeed.utils.exceptions import ErrorCode


class TestHuggingFaceProviderBasics:
    """Test basic HuggingFace provider functionality."""

    def test_provider_imports(self):
        """Test that HuggingFace provider imports correctly."""
        try:
            from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider
            assert HuggingFaceProvider is not None
            print("✅ HuggingFace provider imports successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import HuggingFace provider: {e}")

    def test_provider_initialization_success(self):
        """Test successful provider initialization."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        # Mock aiohttp availability
        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            provider = HuggingFaceProvider(
                api_key="test_api_key_12345",
                model_name="facebook/bart-large-cnn"
            )

            assert provider.api_key == "test_api_key_12345"
            assert provider.model_name == "facebook/bart-large-cnn"
            assert provider.provider_type == AIProviderType.HUGGINGFACE
            assert len(provider.available_models) == 5  # Confirmed working models
            print("✅ Provider initialization successful")

    def test_provider_initialization_no_aiohttp(self):
        """Test provider initialization fails without aiohttp."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', False):
            with pytest.raises(AIError) as exc_info:
                HuggingFaceProvider(api_key="test_key")

            assert exc_info.value.error_code == ErrorCode.AI_PROVIDER_UNAVAILABLE
            assert "aiohttp library not installed" in str(exc_info.value)
            print("✅ Provider correctly fails without aiohttp")

    def test_provider_initialization_no_api_key(self):
        """Test provider initialization fails without API key."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            with pytest.raises(AIError) as exc_info:
                HuggingFaceProvider(api_key="")

            assert exc_info.value.error_code == ErrorCode.AI_INVALID_CREDENTIALS
            assert "API token is required" in str(exc_info.value)
            print("✅ Provider correctly fails without API key")

    def test_confirmed_working_models(self):
        """Test that provider uses confirmed working models."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        expected_models = [
            "facebook/bart-large-cnn",                            # Summarization (primary)
            "facebook/bart-large",                                # Text generation
            "sshleifer/distilbart-cnn-12-6",                      # Fast summarization
            "google/pegasus-xsum",                                # Alternative summarization
            "cardiffnlp/twitter-roberta-base-sentiment-latest"    # Sentiment analysis
        ]

        # Test static method
        static_models = HuggingFaceProvider.get_available_models()
        assert static_models == expected_models

        # Test instance models
        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            provider = HuggingFaceProvider(api_key="test_key")
            assert provider.available_models == expected_models
            print("✅ Confirmed working models correctly configured")


class TestHuggingFaceProviderModels:
    """Test model management functionality."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    def test_model_switching(self, provider):
        """Test model switching functionality."""
        original_model = provider.model_name

        # Test valid model switch
        provider.set_model("facebook/bart-large")
        assert provider.model_name == "facebook/bart-large"

        # Test invalid model switch (should log warning but not crash)
        provider.set_model("invalid/model")
        assert provider.model_name == "facebook/bart-large"  # Should remain unchanged

        print("✅ Model switching works correctly")

    def test_default_model_selection(self, provider):
        """Test default model selection."""
        # Should default to first recommended model
        assert provider.model_name == "facebook/bart-large-cnn"
        print("✅ Default model selection correct")

    def test_custom_model_in_available_list(self):
        """Test custom model gets added to available models."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            provider = HuggingFaceProvider(
                api_key="test_key",
                model_name="custom/model"
            )

            assert "custom/model" in provider.available_models
            assert provider.available_models[0] == "custom/model"  # Should be first
            print("✅ Custom model correctly added to available models")


class TestHuggingFaceProviderRateLimiting:
    """Test rate limiting functionality."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    def test_rate_limit_info(self, provider):
        """Test rate limit information."""
        rate_info = provider.get_rate_limits()

        assert isinstance(rate_info, RateLimitInfo)
        assert rate_info.requests_per_minute == 100
        assert rate_info.requests_per_day == 24000
        assert rate_info.tokens_per_minute is None
        assert rate_info.tokens_per_day is None
        print("✅ Rate limit info correct")

    def test_rate_limit_checking(self, provider):
        """Test rate limit checking logic."""
        # Initially should be able to make requests
        assert provider._can_make_request() is True

        # Simulate hitting rate limit
        provider._request_count_minute = 100  # At limit
        assert provider._can_make_request() is False

        # Simulate time passing (reset)
        provider._minute_start = time.time() - 61  # Over a minute ago
        assert provider._can_make_request() is True
        print("✅ Rate limit checking works correctly")

    def test_rate_limit_update(self, provider):
        """Test rate limit update on request."""
        initial_count = provider._request_count_minute
        initial_time = provider._last_request_time

        provider._update_rate_limit()

        assert provider._request_count_minute == initial_count + 1
        assert provider._last_request_time > initial_time
        print("✅ Rate limit update works correctly")


class TestHuggingFaceProviderAPIHandling:
    """Test API request handling with mocked responses."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_successful_api_response_list_format(self, provider):
        """Test successful API response in list format (BART models)."""
        mock_response = [{"summary_text": "Test summary text"}]

        with patch.object(provider, '_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=mock_response)

            mock_session.post.return_value.__aenter__.return_value = mock_resp

            result = await provider._make_inference_request("facebook/bart-large-cnn", "test prompt")
            assert result == mock_response
            print("✅ List format API response handled correctly")

    @pytest.mark.asyncio
    async def test_model_loading_error(self, provider):
        """Test model loading error (503 status)."""
        with patch.object(provider, '_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 503
            mock_resp.json = AsyncMock(return_value={"estimated_time": 45})

            mock_session.post.return_value.__aenter__.return_value = mock_resp

            with pytest.raises(AIError) as exc_info:
                await provider._make_inference_request("facebook/bart-large-cnn", "test prompt")

            assert "Model loading, wait 45s" in str(exc_info.value)
            assert exc_info.value.retryable is True
            print("✅ Model loading error handled correctly")

    @pytest.mark.asyncio
    async def test_api_error_404(self, provider):
        """Test API 404 error."""
        with patch.object(provider, '_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 404
            mock_resp.text = AsyncMock(return_value="Not Found")

            mock_session.post.return_value.__aenter__.return_value = mock_resp

            with pytest.raises(AIError) as exc_info:
                await provider._make_inference_request("invalid/model", "test prompt")

            assert "HuggingFace API error 404" in str(exc_info.value)
            print("✅ 404 error handled correctly")

    @pytest.mark.asyncio
    async def test_empty_api_response(self, provider):
        """Test handling of empty API response."""
        with patch.object(provider, '_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value=[])  # Empty list

            mock_session.post.return_value.__aenter__.return_value = mock_resp

            result = await provider._make_inference_request("facebook/bart-large-cnn", "test prompt")
            assert result == []
            print("✅ Empty response handled correctly")

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, provider):
        """Test handling of malformed JSON response."""
        with patch.object(provider, '_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(side_effect=json.JSONDecodeError("test", "test", 0))

            mock_session.post.return_value.__aenter__.return_value = mock_resp

            with pytest.raises(json.JSONDecodeError):
                await provider._make_inference_request("facebook/bart-large-cnn", "test prompt")
            print("✅ Malformed JSON handled correctly")

    @pytest.mark.asyncio
    async def test_network_timeout_error(self, provider):
        """Test handling of network timeout."""
        with patch.object(provider, '_session') as mock_session:
            mock_session.post.side_effect = asyncio.TimeoutError("Request timeout")

            with pytest.raises(asyncio.TimeoutError):
                await provider._make_inference_request("facebook/bart-large-cnn", "test prompt")
            print("✅ Network timeout handled correctly")


class TestHuggingFaceProviderAnalysis:
    """Test analysis functionality with mocked API responses."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    @pytest.fixture
    def test_data(self):
        """Create test article and topic."""
        article = Article(
            title="Machine Learning Advances in AI",
            url="https://example.com/ml-article",
            content="Recent developments in machine learning and artificial intelligence have revolutionized the tech industry.",
            source_feed="https://example.com/feed"
        )

        topic = Topic(
            chat_id="test_chat",
            name="Artificial Intelligence",
            keywords=["artificial intelligence", "machine learning", "neural networks", "deep learning"]
        )

        return article, topic

    @pytest.mark.asyncio
    async def test_relevance_analysis_success(self, provider, test_data):
        """Test successful relevance analysis."""
        article, topic = test_data

        # Mock API response (BART summarization response)
        mock_api_response = [{"summary_text": "Machine learning and artificial intelligence advances in technology"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is True
            assert result.relevance_score == 0.7  # Should match keywords
            assert result.confidence == 0.8
            assert "HuggingFace analysis" in result.reasoning
            print("✅ Relevance analysis successful")

    @pytest.mark.asyncio
    async def test_relevance_analysis_no_keyword_match(self, provider, test_data):
        """Test relevance analysis with no keyword matches."""
        article, topic = test_data

        # Mock API response without matching keywords (completely different domain)
        mock_api_response = [{"summary_text": "Sports news about baseball and football games today"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is True
            assert result.relevance_score == 0.3  # No keyword matches
            assert result.confidence == 0.8
            print("✅ Relevance analysis without matches handled correctly")

    @pytest.mark.asyncio
    async def test_summary_generation_success(self, provider, test_data):
        """Test successful summary generation."""
        article, topic = test_data

        # Mock API response (BART summarization)
        mock_api_response = [{"summary_text": "AI and machine learning are transforming technology"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.generate_summary(article)

            assert result.success is True
            assert result.summary == "AI and machine learning are transforming technology"
            assert result.confidence == 0.9
            print("✅ Summary generation successful")

    @pytest.mark.asyncio
    async def test_keyword_generation(self, provider):
        """Test keyword generation (uses simple fallback)."""
        result = await provider.generate_keywords("Machine Learning", max_keywords=5)

        assert result.success is True
        assert len(result.content) <= 5
        assert "machine learning" in result.content
        assert result.confidence == 0.7
        print("✅ Keyword generation successful")

    @pytest.mark.asyncio
    async def test_empty_response_parsing(self, provider, test_data):
        """Test analysis with empty API response."""
        article, topic = test_data

        # Mock empty response
        mock_api_response = []

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is True
            assert result.relevance_score == 0.3  # No content to match
            print("✅ Empty response analysis handled correctly")

    @pytest.mark.asyncio
    async def test_dict_response_parsing(self, provider, test_data):
        """Test analysis with dict response format."""
        article, topic = test_data

        # Mock dict response (some models return this format)
        mock_api_response = {"generated_text": "artificial intelligence and machine learning research"}

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is True
            assert result.relevance_score == 0.7  # Should match keywords
            print("✅ Dict response analysis handled correctly")

    @pytest.mark.asyncio
    async def test_string_response_parsing(self, provider, test_data):
        """Test analysis with string response format."""
        article, topic = test_data

        # Mock string response
        mock_api_response = "Research in artificial intelligence and neural networks"

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is True
            assert result.relevance_score == 0.7  # Should match keywords
            print("✅ String response analysis handled correctly")


class TestHuggingFaceProviderFallback:
    """Test provider fallback behavior."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    @pytest.fixture
    def test_data(self):
        """Create test article and topic."""
        article = Article(
            title="Test Article",
            url="https://example.com/test",
            content="Test content",
            source_feed="https://example.com/feed"
        )

        topic = Topic(
            chat_id="test_chat",
            name="Test Topic",
            keywords=["test"]
        )

        return article, topic

    @pytest.mark.asyncio
    async def test_model_fallback_all_fail(self, provider, test_data):
        """Test behavior when all models fail."""
        article, topic = test_data

        # Mock all API calls to fail
        with patch.object(provider, '_make_inference_request', side_effect=Exception("API Error")):
            result = await provider.analyze_relevance(article, topic)

            assert result.success is False
            assert "All HuggingFace models failed" in result.error_message
            print("✅ All models failing handled correctly")

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, provider, test_data):
        """Test behavior when rate limit is exceeded."""
        article, topic = test_data

        # Mock rate limit exceeded
        provider._request_count_minute = 100  # At limit

        result = await provider.analyze_relevance(article, topic)

        assert result.success is False
        assert "Rate limit exceeded" in result.error_message
        print("✅ Rate limit exceeded handled correctly")


class TestHuggingFaceProviderModelSpecific:
    """Test model-specific functionality."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    @pytest.fixture
    def test_data(self):
        """Create test article and topic."""
        article = Article(
            title="AI Research Paper",
            url="https://example.com/ai-paper",
            content="This paper discusses advances in neural networks and deep learning architectures.",
            source_feed="https://example.com/feed"
        )

        topic = Topic(
            chat_id="test_chat",
            name="AI Research",
            keywords=["neural networks", "deep learning", "AI"]
        )

        return article, topic

    @pytest.mark.asyncio
    async def test_analyze_with_specific_model(self, provider, test_data):
        """Test analysis with specific model."""
        article, topic = test_data

        mock_api_response = [{"summary_text": "Neural networks and deep learning research"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.analyze_relevance_with_model(
                article, topic, "facebook/bart-large"
            )

            assert result.success is True
            assert result.relevance_score == 0.7  # Should match keywords
            print("✅ Model-specific analysis successful")

    @pytest.mark.asyncio
    async def test_generate_summary_with_specific_model(self, provider, test_data):
        """Test summary generation with specific model."""
        article, topic = test_data

        mock_api_response = [{"summary_text": "AI research on neural networks"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.generate_summary_with_model(
                article, "sshleifer/distilbart-cnn-12-6"
            )

            assert result.success is True
            assert result.summary == "AI research on neural networks"
            print("✅ Model-specific summary generation successful")


class TestHuggingFaceProviderConnection:
    """Test connection functionality."""

    @pytest.fixture
    def provider(self):
        """Create a test provider instance."""
        from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            return HuggingFaceProvider(api_key="test_key")

    @pytest.mark.asyncio
    async def test_connection_success(self, provider):
        """Test successful connection test."""
        mock_api_response = [{"generated_text": "Hello response"}]

        with patch.object(provider, '_make_inference_request', return_value=mock_api_response):
            result = await provider.test_connection()

            assert result is True
            print("✅ Connection test successful")

    @pytest.mark.asyncio
    async def test_connection_failure(self, provider):
        """Test connection test failure."""
        with patch.object(provider, '_make_inference_request', side_effect=Exception("Connection error")):
            result = await provider.test_connection()

            assert result is False
            print("✅ Connection test failure handled correctly")

    @pytest.mark.asyncio
    async def test_session_cleanup(self, provider):
        """Test session cleanup."""
        # Create mock session
        mock_session = AsyncMock()
        provider._session = mock_session

        await provider.close()

        mock_session.close.assert_called_once()
        assert provider._session is None
        print("✅ Session cleanup successful")


def test_huggingface_provider_string_representation():
    """Test string representations of provider."""
    from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

    with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
        provider = HuggingFaceProvider(api_key="test_key")

        # Test __str__
        str_repr = str(provider)
        assert "HuggingFaceProvider" in str_repr
        assert provider.model_name in str_repr

        # Test __repr__
        repr_str = repr(provider)
        assert "HuggingFaceProvider" in repr_str
        assert provider.model_name in repr_str

        print("✅ String representations work correctly")


if __name__ == "__main__":
    # Run basic tests without pytest
    print("🧪 HuggingFace Provider Unit Tests")
    print("=" * 50)

    # Test basic functionality
    test_basics = TestHuggingFaceProviderBasics()
    test_basics.test_provider_imports()
    test_basics.test_provider_initialization_success()
    test_basics.test_confirmed_working_models()

    # Test string representation
    test_huggingface_provider_string_representation()

    print("\n🎉 Basic HuggingFace provider tests completed!")
    print("💡 Run with pytest for full async test coverage:")
    print("   pytest tests/unit/test_huggingface_provider.py -v")