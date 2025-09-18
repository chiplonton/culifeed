"""
AI Manager - Multi-Provider Orchestration
=========================================

Manages multiple AI providers with intelligent fallback, load balancing,
and cost optimization for article relevance analysis and summarization.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from .providers.base import AIProvider, AIResult, AIError, AIProviderType, RateLimitInfo
from .providers.groq_provider import GroqProvider
from ..database.models import Article, Topic
from ..config.settings import get_settings, AIProvider as ConfigAIProvider
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import ErrorCode


@dataclass
class ProviderHealth:
    """Health status of an AI provider."""
    provider_type: AIProviderType
    available: bool
    last_success: Optional[float] = None
    last_error: Optional[float] = None
    error_count: int = 0
    consecutive_errors: int = 0
    rate_limited: bool = False
    rate_limit_reset: Optional[float] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if provider is considered healthy."""
        if not self.available:
            return False
        if self.rate_limited and self.rate_limit_reset and time.time() < self.rate_limit_reset:
            return False
        return self.consecutive_errors < 3
    
    def record_success(self):
        """Record successful request."""
        self.last_success = time.time()
        self.consecutive_errors = 0
        self.rate_limited = False
    
    def record_error(self, is_rate_limit: bool = False):
        """Record failed request."""
        self.last_error = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        if is_rate_limit:
            self.rate_limited = True
            self.rate_limit_reset = time.time() + 300  # 5 minute cooldown


class FallbackStrategy(str, Enum):
    """Fallback strategies when primary provider fails."""
    NEXT_AVAILABLE = "next_available"  # Try next healthy provider
    KEYWORDS_ONLY = "keywords_only"    # Fall back to keyword matching
    FAIL_FAST = "fail_fast"            # Don't try alternatives


class AIManager:
    """Multi-provider AI manager with intelligent fallback."""
    
    def __init__(self, primary_provider: Optional[ConfigAIProvider] = None):
        """Initialize AI manager.
        
        Args:
            primary_provider: Primary provider to use (default from settings)
        """
        self.settings = get_settings()
        self.logger = get_logger_for_component("ai_manager")
        
        # Provider management
        self.providers: Dict[AIProviderType, AIProvider] = {}
        self.provider_health: Dict[AIProviderType, ProviderHealth] = {}
        self.primary_provider = primary_provider or self.settings.processing.ai_provider
        
        # Initialize available providers
        self._initialize_providers()
        
        # Fallback configuration
        self.fallback_strategy = FallbackStrategy.NEXT_AVAILABLE
        self.enable_keyword_fallback = self.settings.limits.fallback_to_keywords
        
        self.logger.info(
            f"AI Manager initialized with primary: {self.primary_provider}, "
            f"available providers: {list(self.providers.keys())}"
        )
    
    def _initialize_providers(self) -> None:
        """Initialize all available AI providers."""
        # Initialize Groq if API key available
        if self.settings.ai.groq_api_key:
            try:
                groq_provider = GroqProvider(
                    api_key=self.settings.ai.groq_api_key,
                    model_name=self.settings.ai.groq_model
                )
                self.providers[AIProviderType.GROQ] = groq_provider
                self.provider_health[AIProviderType.GROQ] = ProviderHealth(
                    provider_type=AIProviderType.GROQ,
                    available=True
                )
                self.logger.info("Groq provider initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Groq provider: {e}")
        
        # TODO: Initialize Gemini provider when implemented
        # if self.settings.ai.gemini_api_key:
        #     try:
        #         gemini_provider = GeminiProvider(...)
        #         self.providers[AIProviderType.GEMINI] = gemini_provider
        #         ...
        
        # TODO: Initialize OpenAI provider when implemented
        # if self.settings.ai.openai_api_key:
        #     try:
        #         openai_provider = OpenAIProvider(...)
        #         self.providers[AIProviderType.OPENAI] = openai_provider
        #         ...
        
        if not self.providers:
            self.logger.error("No AI providers available! Check API keys in configuration.")
    
    async def analyze_relevance(self, article: Article, topic: Topic, 
                              fallback_strategy: FallbackStrategy = None) -> AIResult:
        """Analyze article relevance with automatic fallback.
        
        Args:
            article: Article to analyze
            topic: Topic to match against
            fallback_strategy: Override default fallback strategy
            
        Returns:
            AIResult with relevance analysis
        """
        strategy = fallback_strategy or self.fallback_strategy
        
        # Get provider priority order
        provider_order = self._get_provider_priority_order()
        
        last_error = None
        
        for provider_type in provider_order:
            provider = self.providers.get(provider_type)
            health = self.provider_health.get(provider_type)
            
            if not provider or not health or not health.is_healthy:
                continue
            
            try:
                self.logger.debug(f"Analyzing relevance with {provider_type.value}")
                
                result = await provider.analyze_relevance(article, topic)
                
                if result.success:
                    health.record_success()
                    self.logger.debug(
                        f"Relevance analysis successful: {provider_type.value} "
                        f"score={result.relevance_score:.3f}"
                    )
                    return result
                else:
                    health.record_error()
                    last_error = AIError(
                        result.error_message or "Unknown analysis error",
                        provider=provider_type.value
                    )
                    
            except AIError as e:
                health.record_error(e.rate_limited)
                last_error = e
                self.logger.warning(f"AI provider {provider_type.value} failed: {e.user_message}")
                
                if strategy == FallbackStrategy.FAIL_FAST:
                    raise e
            
            except Exception as e:
                health.record_error()
                last_error = AIError(
                    f"Unexpected error: {e}",
                    provider=provider_type.value,
                    error_code=ErrorCode.AI_PROCESSING_ERROR
                )
                self.logger.error(f"Unexpected error with {provider_type.value}: {e}")
        
        # All providers failed - try keyword fallback if enabled
        if strategy == FallbackStrategy.NEXT_AVAILABLE and self.enable_keyword_fallback:
            self.logger.info("All AI providers failed, falling back to keyword matching")
            return self._keyword_fallback_analysis(article, topic)
        
        # No fallback or fallback disabled
        error_msg = f"All AI providers failed. Last error: {last_error.user_message if last_error else 'Unknown'}"
        self.logger.error(error_msg)
        
        return AIResult(
            success=False,
            relevance_score=0.0,
            confidence=0.0,
            error_message=error_msg
        )
    
    async def generate_summary(self, article: Article, max_sentences: int = 3) -> AIResult:
        """Generate article summary with automatic fallback.
        
        Args:
            article: Article to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            AIResult with generated summary
        """
        # Get provider priority order
        provider_order = self._get_provider_priority_order()
        
        for provider_type in provider_order:
            provider = self.providers.get(provider_type)
            health = self.provider_health.get(provider_type)
            
            if not provider or not health or not health.is_healthy:
                continue
            
            try:
                self.logger.debug(f"Generating summary with {provider_type.value}")
                
                result = await provider.generate_summary(article, max_sentences)
                
                if result.success:
                    health.record_success()
                    self.logger.debug(f"Summary generation successful: {provider_type.value}")
                    return result
                else:
                    health.record_error()
                    
            except AIError as e:
                health.record_error(e.rate_limited)
                self.logger.warning(f"Summary generation failed with {provider_type.value}: {e.user_message}")
            
            except Exception as e:
                health.record_error()
                self.logger.error(f"Unexpected summary error with {provider_type.value}: {e}")
        
        # All providers failed - create simple fallback summary
        self.logger.warning("All providers failed for summarization, creating fallback summary")
        return self._create_fallback_summary(article, max_sentences)
    
    def _get_provider_priority_order(self) -> List[AIProviderType]:
        """Get provider priority order based on health and configuration.
        
        Returns:
            List of provider types in priority order
        """
        available_providers = []
        
        # Start with primary provider if healthy
        primary_type = self._config_to_provider_type(self.primary_provider)
        if primary_type and primary_type in self.providers:
            health = self.provider_health.get(primary_type)
            if health and health.is_healthy:
                available_providers.append(primary_type)
        
        # Add other healthy providers
        for provider_type, health in self.provider_health.items():
            if provider_type not in available_providers and health.is_healthy:
                available_providers.append(provider_type)
        
        # Add unhealthy providers as last resort (if not rate limited)
        for provider_type, health in self.provider_health.items():
            if provider_type not in available_providers and not health.rate_limited:
                available_providers.append(provider_type)
        
        return available_providers
    
    def _config_to_provider_type(self, config_provider: ConfigAIProvider) -> Optional[AIProviderType]:
        """Convert configuration provider to provider type.
        
        Args:
            config_provider: Provider from configuration
            
        Returns:
            Corresponding AIProviderType or None
        """
        mapping = {
            ConfigAIProvider.GROQ: AIProviderType.GROQ,
            ConfigAIProvider.GEMINI: AIProviderType.GEMINI,
            ConfigAIProvider.OPENAI: AIProviderType.OPENAI
        }
        return mapping.get(config_provider)
    
    def _keyword_fallback_analysis(self, article: Article, topic: Topic) -> AIResult:
        """Fallback relevance analysis using keyword matching.
        
        Args:
            article: Article to analyze
            topic: Topic to match against
            
        Returns:
            AIResult with keyword-based analysis
        """
        if not topic.keywords:
            return AIResult(
                success=False,
                relevance_score=0.0,
                confidence=0.0,
                error_message="No keywords available for fallback analysis"
            )
        
        # Simple keyword matching
        article_text = f"{article.title} {article.content}".lower()
        matched_keywords = []
        keyword_matches = 0
        
        for keyword in topic.keywords:
            if keyword.lower() in article_text:
                matched_keywords.append(keyword)
                keyword_matches += 1
        
        # Check exclude keywords
        excluded = False
        if topic.exclude_keywords:
            for exclude_keyword in topic.exclude_keywords:
                if exclude_keyword.lower() in article_text:
                    excluded = True
                    break
        
        # Calculate simple relevance score
        if excluded:
            relevance_score = max(0.0, (keyword_matches / len(topic.keywords)) * 0.3)  # Penalize excluded
        else:
            relevance_score = min(0.8, (keyword_matches / len(topic.keywords)) * 0.7)  # Cap at 0.8 for keyword-only
        
        confidence = min(0.6, keyword_matches / len(topic.keywords))  # Lower confidence for keyword-only
        
        return AIResult(
            success=True,
            relevance_score=relevance_score,
            confidence=confidence,
            matched_keywords=matched_keywords,
            reasoning=f"Keyword-based analysis: {keyword_matches}/{len(topic.keywords)} keywords matched",
            provider="keyword_fallback"
        )
    
    def _create_fallback_summary(self, article: Article, max_sentences: int) -> AIResult:
        """Create simple fallback summary from article content.
        
        Args:
            article: Article to summarize
            max_sentences: Maximum sentences in summary
            
        Returns:
            AIResult with simple summary
        """
        # Simple extractive summarization - take first few sentences
        content = article.content or ""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            summary = article.title or "No content available for summary"
        else:
            # Take first N sentences up to max_sentences
            selected_sentences = sentences[:max_sentences]
            summary = '. '.join(selected_sentences)
            if not summary.endswith('.'):
                summary += '.'
        
        return AIResult(
            success=True,
            relevance_score=1.0,
            confidence=0.3,  # Low confidence for fallback summary
            summary=summary,
            provider="fallback_summary"
        )
    
    async def test_all_providers(self) -> Dict[AIProviderType, bool]:
        """Test connection for all configured providers.
        
        Returns:
            Dictionary mapping provider types to connection status
        """
        results = {}
        
        for provider_type, provider in self.providers.items():
            try:
                self.logger.info(f"Testing {provider_type.value} connection...")
                success = await provider.test_connection()
                results[provider_type] = success
                
                if success:
                    self.provider_health[provider_type].record_success()
                    self.logger.info(f"{provider_type.value} connection test passed")
                else:
                    self.provider_health[provider_type].record_error()
                    self.logger.warning(f"{provider_type.value} connection test failed")
                    
            except Exception as e:
                results[provider_type] = False
                self.provider_health[provider_type].record_error()
                self.logger.error(f"{provider_type.value} connection test error: {e}")
        
        return results
    
    def get_provider_status(self) -> Dict[str, Dict]:
        """Get detailed status of all providers.
        
        Returns:
            Dictionary with provider status information
        """
        status = {}
        
        for provider_type, health in self.provider_health.items():
            provider = self.providers.get(provider_type)
            rate_limits = provider.get_rate_limits() if provider else None
            
            status[provider_type.value] = {
                'available': health.available,
                'healthy': health.is_healthy,
                'error_count': health.error_count,
                'consecutive_errors': health.consecutive_errors,
                'rate_limited': health.rate_limited,
                'last_success': health.last_success,
                'last_error': health.last_error,
                'rate_limits': {
                    'requests_per_minute': rate_limits.requests_per_minute if rate_limits else None,
                    'current_usage': rate_limits.current_usage if rate_limits else None,
                } if rate_limits else None
            }
        
        return status
    
    def reset_provider_health(self, provider_type: AIProviderType) -> None:
        """Reset health status for a specific provider.
        
        Args:
            provider_type: Provider to reset
        """
        if provider_type in self.provider_health:
            health = self.provider_health[provider_type]
            health.consecutive_errors = 0
            health.rate_limited = False
            health.rate_limit_reset = None
            self.logger.info(f"Reset health status for {provider_type.value}")
    
    async def shutdown(self) -> None:
        """Cleanup resources and close provider connections."""
        self.logger.info("Shutting down AI Manager...")
        
        # Close async clients if needed
        for provider in self.providers.values():
            if hasattr(provider, 'async_client') and hasattr(provider.async_client, 'close'):
                try:
                    await provider.async_client.aclose()
                except Exception as e:
                    self.logger.warning(f"Error closing provider client: {e}")
        
        self.logger.info("AI Manager shutdown complete")
    
    def __str__(self) -> str:
        """String representation."""
        return f"AIManager(primary={self.primary_provider.value}, providers={len(self.providers)})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        healthy_count = sum(1 for h in self.provider_health.values() if h.is_healthy)
        return (f"AIManager(primary={self.primary_provider.value}, "
                f"providers={len(self.providers)}, healthy={healthy_count})")