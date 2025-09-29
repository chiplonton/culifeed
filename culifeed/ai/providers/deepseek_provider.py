"""
DeepSeek AI Provider Implementation
===================================

DeepSeek provider for advanced reasoning and chat models with comprehensive error handling,
rate limiting, and fallback support for article relevance analysis.
"""

import asyncio
import time
from typing import Optional, List
import logging

try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None
    AsyncOpenAI = None

from .base import AIProvider, AIResult, AIError, RateLimitInfo, AIProviderType
from ...database.models import Article, Topic
from ...utils.exceptions import ErrorCode
from ...utils.logging import get_logger_for_component


class DeepSeekProvider(AIProvider):
    """DeepSeek AI provider with async support and comprehensive error handling."""

    # DeepSeek rate limits (estimated based on their API documentation)
    DEFAULT_RATE_LIMITS = RateLimitInfo(
        requests_per_minute=60,
        requests_per_day=10000,
        tokens_per_minute=100000,
        tokens_per_day=None,  # No daily token limit mentioned
    )

    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        """Initialize DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model_name: Model to use (default: deepseek-chat)

        Raises:
            AIError: If OpenAI library not available or invalid configuration
        """
        if not OPENAI_AVAILABLE:
            raise AIError(
                "OpenAI library not installed. Run: pip install openai",
                provider="deepseek",
                error_code=ErrorCode.AI_PROVIDER_UNAVAILABLE,
            )

        if not api_key:
            raise AIError(
                "DeepSeek API key is required",
                provider="deepseek",
                error_code=ErrorCode.AI_INVALID_CREDENTIALS,
            )

        # Use custom AIProviderType for DeepSeek
        super().__init__(api_key, model_name, AIProviderType.DEEPSEEK)

        # Initialize clients with DeepSeek base URL
        self.client = openai.OpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.deepseek.com"
        )

        # Set up logging and rate limiting
        self.logger = get_logger_for_component("deepseek_provider")
        self._rate_limit_info = self.DEFAULT_RATE_LIMITS
        self._last_request_time = 0.0
        self._request_count_minute = 0
        self._minute_start = time.time()

        self.logger.info(f"DeepSeek provider initialized with model: {model_name}")

    async def analyze_relevance(self, article: Article, topic: Topic) -> AIResult:
        """Analyze article relevance using DeepSeek.

        Args:
            article: Article to analyze
            topic: Topic to match against

        Returns:
            AIResult with relevance analysis
        """
        if not self.can_make_request():
            return self._create_error_result("Rate limit exceeded")

        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_relevance_prompt(article, topic)

            # Make API request
            self.logger.debug(
                f"Analyzing relevance for article: {article.title[:50]}..."
            )

            response = await self._make_chat_completion(prompt)

            # Parse response
            relevance_score, confidence, matched_keywords, reasoning = (
                self._parse_relevance_response(response.choices[0].message.content)
            )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update usage tracking
            tokens_used = (
                getattr(response.usage, "total_tokens", None)
                if hasattr(response, "usage")
                else None
            )
            self.update_rate_limit_usage(tokens_used or 0)

            self.logger.debug(
                f"Relevance analysis complete: score={relevance_score:.3f}, "
                f"confidence={confidence:.3f}, time={processing_time_ms}ms"
            )

            return self._create_success_result(
                relevance_score=relevance_score,
                confidence=confidence,
                reasoning=reasoning,
                matched_keywords=matched_keywords,
                tokens_used=tokens_used,
                processing_time_ms=processing_time_ms,
            )

        except openai.RateLimitError as e:
            self.logger.warning(f"DeepSeek rate limit exceeded: {e}")
            self._handle_rate_limit_error(e)
            return self._create_error_result("Rate limit exceeded")

        except openai.APIConnectionError as e:
            self.logger.error(f"DeepSeek connection error: {e}")
            raise AIError(
                f"Connection to DeepSeek failed: {e}",
                provider="deepseek",
                error_code=ErrorCode.AI_CONNECTION_ERROR,
                retryable=True,
            )

        except openai.APIStatusError as e:
            self.logger.error(f"DeepSeek API error: {e.status_code} - {e.message}")

            if e.status_code == 401:
                raise AIError(
                    "Invalid DeepSeek API key",
                    provider="deepseek",
                    error_code=ErrorCode.AI_INVALID_CREDENTIALS,
                )
            elif e.status_code == 429:
                self._handle_rate_limit_error(e)
                return self._create_error_result("Rate limit exceeded")
            else:
                raise AIError(
                    f"DeepSeek API error: {e.status_code} - {e.message}",
                    provider="deepseek",
                    error_code=ErrorCode.AI_API_ERROR,
                    retryable=e.status_code >= 500,
                )

        except Exception as e:
            self.logger.error(f"Unexpected DeepSeek error: {e}", exc_info=True)
            raise AIError(
                f"Unexpected error during relevance analysis: {e}",
                provider="deepseek",
                error_code=ErrorCode.AI_PROCESSING_ERROR,
            )

    async def generate_summary(
        self, article: Article, max_sentences: int = 3
    ) -> AIResult:
        """Generate article summary using DeepSeek.

        Args:
            article: Article to summarize
            max_sentences: Maximum sentences in summary

        Returns:
            AIResult with generated summary
        """
        if not self.can_make_request():
            return self._create_error_result("Rate limit exceeded")

        start_time = time.time()

        try:
            # Build prompt
            prompt = self._build_summary_prompt(article, max_sentences)

            # Make API request
            self.logger.debug(
                f"Generating summary for article: {article.title[:50]}..."
            )

            response = await self._make_chat_completion(prompt)
            summary = response.choices[0].message.content.strip()

            # Clean up summary format
            if summary.startswith("SUMMARY:"):
                summary = summary.replace("SUMMARY:", "").strip()

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update usage tracking
            tokens_used = (
                getattr(response.usage, "total_tokens", None)
                if hasattr(response, "usage")
                else None
            )
            self.update_rate_limit_usage(tokens_used or 0)

            self.logger.debug(
                f"Summary generated: {len(summary)} chars, time={processing_time_ms}ms"
            )

            return self._create_success_result(
                relevance_score=1.0,  # Summary always succeeds if we get here
                confidence=0.9,  # High confidence for summarization
                summary=summary,
                tokens_used=tokens_used,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            # Handle similar errors as in analyze_relevance
            self.logger.error(f"Summary generation error: {e}", exc_info=True)
            return self._create_error_result(f"Summary generation failed: {e}")

    async def generate_keywords(
        self, topic_name: str, context: str = "", max_keywords: int = 7
    ) -> AIResult:
        """Generate keywords for a topic using DeepSeek.

        Args:
            topic_name: Topic name to generate keywords for
            context: Additional context (e.g., existing user topics)
            max_keywords: Maximum number of keywords to generate

        Returns:
            AIResult with generated keywords
        """
        if not self.can_make_request():
            return self._create_error_result("Rate limit exceeded")

        start_time = time.time()

        try:
            # Build prompt for keyword generation
            prompt = f"Generate {max_keywords} relevant keywords for '{topic_name}'.{context} Return comma-separated keywords only."

            # Make API request
            self.logger.debug(f"Generating keywords for topic: {topic_name}")

            response = await self._make_chat_completion(prompt, max_tokens=150)
            response_text = response.choices[0].message.content.strip()

            # Parse keywords
            keywords = [
                k.strip().strip("\"'") for k in response_text.split(",") if k.strip()
            ]
            keywords = keywords[:max_keywords]  # Ensure max limit

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Update usage tracking
            tokens_used = (
                getattr(response.usage, "total_tokens", None)
                if hasattr(response, "usage")
                else None
            )
            self.update_rate_limit_usage(tokens_used or 0)

            self.logger.debug(
                f"Keywords generated: {len(keywords)} keywords, time={processing_time_ms}ms"
            )

            return self._create_success_result(
                relevance_score=1.0,  # Keywords always succeed if we get here
                confidence=0.8,  # High confidence for keyword generation
                content=keywords,  # Store keywords in content field
                tokens_used=tokens_used,
                processing_time_ms=processing_time_ms,
            )

        except Exception as e:
            # Handle similar errors as in other methods
            self.logger.error(f"Keyword generation error: {e}", exc_info=True)
            return self._create_error_result(f"Keyword generation failed: {e}")

    async def test_connection(self) -> bool:
        """Test DeepSeek API connection and authentication.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing DeepSeek connection...")

            # Simple test request
            response = await self._make_chat_completion(
                "Respond with exactly: 'Connection test successful'", max_tokens=10
            )

            success = "successful" in response.choices[0].message.content.lower()

            if success:
                self.logger.info("DeepSeek connection test successful")
            else:
                self.logger.warning(
                    "DeepSeek connection test failed - unexpected response"
                )

            return success

        except Exception as e:
            self.logger.error(f"DeepSeek connection test failed: {e}")
            return False

    def get_rate_limits(self) -> RateLimitInfo:
        """Get current rate limit information.

        Returns:
            RateLimitInfo with current usage and limits
        """
        # Update minute-based tracking
        current_time = time.time()
        if current_time - self._minute_start >= 60:
            self._request_count_minute = 0
            self._minute_start = current_time

        # Update rate limit info
        self._rate_limit_info.current_usage = self._request_count_minute

        return self._rate_limit_info

    def can_make_request(self) -> bool:
        """Check if we can make another request within rate limits.

        Returns:
            True if request can be made, False if rate limited
        """
        current_time = time.time()

        # Reset minute counter if needed
        if current_time - self._minute_start >= 60:
            self._request_count_minute = 0
            self._minute_start = current_time

        # Check per-minute rate limit
        if self._request_count_minute >= self.DEFAULT_RATE_LIMITS.requests_per_minute:
            return False

        # Check minimum time between requests (basic throttling)
        if current_time - self._last_request_time < 0.1:  # 100ms minimum
            return False

        return True

    def update_rate_limit_usage(self, tokens_used: int = 0) -> None:
        """Update rate limit usage tracking.

        Args:
            tokens_used: Number of tokens consumed
        """
        self._request_count_minute += 1
        self._last_request_time = time.time()

        if self._rate_limit_info:
            self._rate_limit_info.current_usage = self._request_count_minute

    def set_model(self, model_name: str) -> None:
        """Switch to a different model for this provider instance.

        Args:
            model_name: New model name to use
        """
        if model_name in self.get_available_models():
            self.model_name = model_name
            self.logger.info(f"Switched DeepSeek model to: {model_name}")
        else:
            self.logger.warning(
                f"Unknown DeepSeek model: {model_name}, keeping current: {self.model_name}"
            )

    async def analyze_relevance_with_model(
        self, article: Article, topic: Topic, model_name: str
    ) -> AIResult:
        """Analyze relevance with a specific model.

        Args:
            article: Article to analyze
            topic: Topic to match against
            model_name: Specific model to use for this request

        Returns:
            AIResult with relevance analysis
        """
        # Temporarily switch model
        original_model = self.model_name
        self.set_model(model_name)

        try:
            result = await self.analyze_relevance(article, topic)
            return result
        finally:
            # Restore original model
            self.model_name = original_model

    async def generate_summary_with_model(
        self, article: Article, model_name: str, max_sentences: int = 3
    ) -> AIResult:
        """Generate summary with a specific model.

        Args:
            article: Article to summarize
            model_name: Specific model to use for this request
            max_sentences: Maximum sentences in summary

        Returns:
            AIResult with generated summary
        """
        # Temporarily switch model
        original_model = self.model_name
        self.set_model(model_name)

        try:
            result = await self.generate_summary(article, max_sentences)
            return result
        finally:
            # Restore original model
            self.model_name = original_model

    async def _make_chat_completion(self, prompt: str, max_tokens: int = 500) -> any:
        """Make chat completion request to DeepSeek.

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens in response

        Returns:
            DeepSeek completion response
        """
        # Add small delay for rate limiting
        await asyncio.sleep(0.1)

        response = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that analyzes content accurately and provides structured responses.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.1,  # Low temperature for consistent results
            top_p=0.9,
        )

        return response

    def _handle_rate_limit_error(self, error) -> None:
        """Handle rate limit error and update internal tracking.

        Args:
            error: Rate limit error from DeepSeek API
        """
        self.logger.warning(f"Rate limit hit: {error}")

        # Set reset time (estimate 1 minute)
        if self._rate_limit_info:
            self._rate_limit_info.reset_time = time.time() + 60
            self._rate_limit_info.current_usage = (
                self._rate_limit_info.requests_per_minute
            )

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available DeepSeek models.

        Returns:
            List of model names
        """
        return [
            "deepseek-chat",  # Main chat model
            "deepseek-reasoner",  # Reasoning model with CoT
            "deepseek-coder",  # Code-focused model
        ]

    def __str__(self) -> str:
        """String representation of provider."""
        return f"DeepSeekProvider(model={self.model_name})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"DeepSeekProvider(model={self.model_name}, rate_limit={self._request_count_minute}/{self.DEFAULT_RATE_LIMITS.requests_per_minute})"
