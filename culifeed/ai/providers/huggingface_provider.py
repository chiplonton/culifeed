"""
HuggingFace AI Provider Implementation
=====================================

HuggingFace Inference API provider for accessing open-source models
with comprehensive error handling and rate limiting for free tier.
"""

import asyncio
import time
import json
from typing import Optional, List, Dict, Any
import logging

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .base import AIProvider, AIResult, AIError, RateLimitInfo, AIProviderType
from ...database.models import Article, Topic
from ...utils.exceptions import ErrorCode
from ...utils.logging import get_logger_for_component


class HuggingFaceProvider(AIProvider):
    """HuggingFace Inference API provider with free tier optimizations."""
    
    # HuggingFace FREE tier rate limits (generous compared to others)
    DEFAULT_RATE_LIMITS = RateLimitInfo(
        requests_per_minute=100,     # Conservative estimate for free tier
        requests_per_day=24000,      # 24K daily requests (much better than OpenRouter)
        tokens_per_minute=None,      # No explicit token limits
        tokens_per_day=None
    )
    
    # FREE models available on HuggingFace Inference API (confirmed working)
    RECOMMENDED_MODELS = [
        "facebook/bart-large-cnn",                            # Summarization (primary)
        "facebook/bart-large",                                # Text generation
        "sshleifer/distilbart-cnn-12-6",                      # Fast summarization
        "google/pegasus-xsum",                                # Alternative summarization
        "cardiffnlp/twitter-roberta-base-sentiment-latest"    # Sentiment analysis
    ]
    
    BASE_URL = "https://api-inference.huggingface.co/models"
    
    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """Initialize HuggingFace provider.
        
        Args:
            api_key: HuggingFace API token
            model_name: Specific model to use (default: use recommended models)
            
        Raises:
            AIError: If required libraries not available or invalid configuration
        """
        if not AIOHTTP_AVAILABLE:
            raise AIError(
                "aiohttp library not installed. Run: pip install aiohttp",
                provider="huggingface",
                error_code=ErrorCode.AI_PROVIDER_UNAVAILABLE
            )
        
        if not api_key:
            raise AIError(
                "HuggingFace API token is required",
                provider="huggingface",
                error_code=ErrorCode.AI_INVALID_CREDENTIALS
            )
        
        # Use first recommended model if none specified
        default_model = model_name or self.RECOMMENDED_MODELS[0]
        super().__init__(api_key, default_model, AIProviderType.HUGGINGFACE)
        
        # Available models for fallback
        self.available_models = self.RECOMMENDED_MODELS.copy()
        if model_name and model_name not in self.available_models:
            self.available_models.insert(0, model_name)
        
        # Set up logging and rate limiting
        self.logger = get_logger_for_component("huggingface_provider")
        self._rate_limit_info = self.DEFAULT_RATE_LIMITS
        self._last_request_time = 0.0
        self._request_count_minute = 0
        self._minute_start = time.time()
        
        # HTTP session for connection reuse
        self._session: Optional[aiohttp.ClientSession] = None
        
        self.logger.info(f"HuggingFace provider initialized with models: {self.available_models}")
    
    async def analyze_relevance(self, article: Article, topic: Topic) -> AIResult:
        """Analyze article relevance using HuggingFace models with fallback.
        
        Args:
            article: Article to analyze
            topic: Topic to match against
            
        Returns:
            AIResult with relevance score and reasoning
        """
        # Rate limiting check
        if not self._can_make_request():
            return self._create_error_result("Rate limit exceeded")
        
        # Try each available model
        last_error = None
        for model_name in self.available_models:
            try:
                return await self._analyze_with_model(article, topic, model_name)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        return self._create_error_result("All HuggingFace models failed")
    
    async def _analyze_with_model(self, article: Article, topic: Topic, model_name: str) -> AIResult:
        """Analyze relevance using specific model."""
        self._update_rate_limit()
        
        # For BART models, use summarization to understand content
        if 'bart' in model_name.lower():
            prompt = f"Summarize and analyze relevance to '{topic.name}': {article.title}. {article.content[:1000]}"
        else:
            prompt = f"Analyze relevance to '{topic.name}': {article.title}"
        
        try:
            response_data = await self._make_inference_request(model_name, prompt)
            
            # Extract text from HuggingFace response (can be list or dict)
            if isinstance(response_data, list) and len(response_data) > 0:
                # Handle list response from summarization models
                if isinstance(response_data[0], dict):
                    response_text = response_data[0].get('summary_text', '') or response_data[0].get('generated_text', '') or str(response_data[0])
                else:
                    response_text = str(response_data[0])
            elif isinstance(response_data, dict):
                response_text = response_data.get('summary_text', '') or response_data.get('generated_text', '') or str(response_data)
            else:
                response_text = str(response_data)
            
            # Simple heuristic analysis for HuggingFace models
            relevance_score = 0.7 if any(keyword.lower() in response_text.lower() 
                                       for keyword in topic.keywords) else 0.3
            
            return self._create_success_result(
                relevance_score=relevance_score,
                confidence=0.8,
                reasoning=f"HuggingFace analysis with {model_name}: {response_text[:200]}"
            )
        
        except Exception as e:
            raise AIError(f"HuggingFace inference failed: {e}", provider="huggingface")
    
    async def generate_summary(self, article: Article, max_sentences: int = 3) -> AIResult:
        """Generate article summary using HuggingFace models.
        
        Args:
            article: Article to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            AIResult with generated summary
        """
        if not self._can_make_request():
            return self._create_error_result("Rate limit exceeded")
        
        # Try each available model, prioritizing summarization models
        for model_name in self.available_models:
            if 'bart' in model_name.lower() or 'pegasus' in model_name.lower():
                try:
                    return await self._generate_summary_with_model(article.content, model_name)
                except Exception as e:
                    self.logger.warning(f"Summary failed with {model_name}: {e}")
                    continue
        
        return self._create_error_result("All HuggingFace summarization models failed")
    
    async def _generate_summary_with_model(self, content: str, model_name: str) -> AIResult:
        """Generate summary using specific model."""
        self._update_rate_limit()
        
        # Truncate content if too long
        max_content_length = 1000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        try:
            summary = await self._make_inference_request(model_name, content)
            
            # Extract summary text from response
            if isinstance(summary, list) and len(summary) > 0:
                if isinstance(summary[0], dict) and 'summary_text' in summary[0]:
                    summary = summary[0]['summary_text']
                else:
                    summary = str(summary[0])
            elif isinstance(summary, str):
                summary = summary
            else:
                summary = str(summary)
            
            return self._create_success_result(
                relevance_score=1.0,
                confidence=0.9,
                summary=summary
            )
        
        except Exception as e:
            raise AIError(f"HuggingFace summary failed: {e}", provider="huggingface")
    
    async def generate_keywords(self, topic_name: str, context: str = "", max_keywords: int = 7) -> AIResult:
        """Generate keywords for a topic using HuggingFace models."""
        # Simple fallback for HuggingFace (not all models support keyword generation)
        keywords = [topic_name.lower(), f"{topic_name.lower()} technology", f"{topic_name.lower()} development"]
        return self._create_success_result(
            relevance_score=1.0,
            confidence=0.7,
            content=keywords[:max_keywords]
        )
    

    
    async def _make_inference_request(self, model_name: str, prompt: str) -> str:
        """Make inference request to HuggingFace API."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}/{model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": prompt}
        
        async with self._session.post(url, headers=headers, json=payload, timeout=30) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 503:
                error_data = await response.json()
                estimated_time = error_data.get("estimated_time", 60)
                raise AIError(f"Model loading, wait {estimated_time}s", provider="huggingface", retryable=True)
            else:
                error_text = await response.text()
                raise AIError(f"HuggingFace API error {response.status}: {error_text}", provider="huggingface")
    

    
    def _can_make_request(self) -> bool:
        """Check if we can make another request based on rate limits."""
        current_time = time.time()
        if current_time - self._minute_start >= 60:
            self._request_count_minute = 0
            self._minute_start = current_time
        return self._request_count_minute < self._rate_limit_info.requests_per_minute
    
    def _update_rate_limit(self) -> None:
        """Update rate limit tracking after making a request."""
        self._last_request_time = time.time()
        self._request_count_minute += 1
    

    
    def get_rate_limits(self) -> RateLimitInfo:
        """Get current rate limit information."""
        return self._rate_limit_info
    
    async def test_connection(self) -> bool:
        """Test HuggingFace connection with a simple request."""
        try:
            response = await self._make_inference_request(
                self.available_models[0],
                "Hello, this is a test."
            )
            return bool(response)
        except Exception as e:
            self.logger.error(f"HuggingFace connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
    
    def set_model(self, model_name: str) -> None:
        """Switch to a different model for this provider instance."""
        if model_name in self.available_models:
            self.model_name = model_name
            self.logger.info(f"Switched HuggingFace model to: {model_name}")
        else:
            self.logger.warning(f"Unknown HuggingFace model: {model_name}")
    
    async def analyze_relevance_with_model(self, article: Article, topic: Topic, model_name: str) -> AIResult:
        """Analyze relevance with a specific model."""
        original_model = self.model_name
        self.set_model(model_name)
        try:
            return await self.analyze_relevance(article, topic)
        finally:
            self.model_name = original_model
    
    async def generate_summary_with_model(self, article: Article, model_name: str, max_sentences: int = 3) -> AIResult:
        """Generate summary with a specific model."""
        original_model = self.model_name
        self.set_model(model_name)
        try:
            return await self.generate_summary(article, max_sentences)
        finally:
            self.model_name = original_model
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available HuggingFace models."""
        return [
            "facebook/bart-large-cnn",                            # Summarization (primary)
            "facebook/bart-large",                                # Text generation
            "sshleifer/distilbart-cnn-12-6",                      # Fast summarization
            "google/pegasus-xsum",                                # Alternative summarization
            "cardiffnlp/twitter-roberta-base-sentiment-latest"    # Sentiment analysis
        ]
    
    def __str__(self) -> str:
        """String representation of provider."""
        return f"HuggingFaceProvider(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HuggingFaceProvider(model={self.model_name}, rate_limit={self._request_count_minute}/{self.DEFAULT_RATE_LIMITS.requests_per_minute})"