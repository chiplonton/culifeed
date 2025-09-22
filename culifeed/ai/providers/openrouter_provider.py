"""
OpenRouter AI Provider Implementation
====================================

OpenRouter provider for accessing 400+ models through unified API
with intelligent model selection and comprehensive error handling.
"""

import asyncio
import time
import json
from typing import Optional, List, Dict, Any
import logging

try:
    import aiohttp
    import openai
    AIOHTTP_AVAILABLE = True
    OPENAI_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    OPENAI_AVAILABLE = False
    aiohttp = None
    openai = None

from .base import AIProvider, AIResult, AIError, RateLimitInfo, AIProviderType
from ...database.models import Article, Topic
from ...utils.exceptions import ErrorCode
from ...utils.logging import get_logger_for_component


class OpenRouterProvider(AIProvider):
    """OpenRouter AI provider with multi-model support and unified API access."""

    # OpenRouter FREE PLAN rate limits
    DEFAULT_RATE_LIMITS = RateLimitInfo(
        requests_per_minute=20,  # Free model variants limit: 20 req/min
        requests_per_day=50,     # Free plan daily limit (no credits purchased)
        tokens_per_minute=None,  # Varies by model
        tokens_per_day=None
    )

    # FREE models for OpenRouter (ending with :free)
    RECOMMENDED_MODELS = [
        "meta-llama/llama-3.2-3b-instruct:free",    # Free Llama model
        "mistralai/mistral-7b-instruct:free",       # Free Mistral model
        # Removed invalid model: "huggingface/meta-llama/llama-3.2-1b-instruct:free"
        # Removed models with frequent upstream issues: "google/gemma-2-9b-it:free"
    ]

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key
            model_name: Specific model to use (default: use recommended models)

        Raises:
            AIError: If required libraries not available or invalid configuration
        """
        if not AIOHTTP_AVAILABLE:
            raise AIError(
                "aiohttp library not installed. Run: pip install aiohttp",
                provider="openrouter",
                error_code=ErrorCode.AI_PROVIDER_UNAVAILABLE
            )

        if not OPENAI_AVAILABLE:
            raise AIError(
                "openai library not installed. Run: pip install openai",
                provider="openrouter",
                error_code=ErrorCode.AI_PROVIDER_UNAVAILABLE
            )

        if not api_key:
            raise AIError(
                "OpenRouter API key is required",
                provider="openrouter",
                error_code=ErrorCode.AI_INVALID_CREDENTIALS
            )

        # Use first recommended model if none specified
        default_model = model_name or self.RECOMMENDED_MODELS[0]
        super().__init__(api_key, default_model, AIProviderType.OPENROUTER)

        # Initialize OpenAI client configured for OpenRouter
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=self.BASE_URL
        )

        # Available models for fallback
        self.available_models = self.RECOMMENDED_MODELS.copy()
        if model_name and model_name not in self.available_models:
            self.available_models.insert(0, model_name)

        # Set up logging and rate limiting
        self.logger = get_logger_for_component("openrouter_provider")
        self._rate_limit_info = self.DEFAULT_RATE_LIMITS
        self._last_request_time = 0.0
        self._request_count_minute = 0
        self._minute_start = time.time()

        self.logger.info(f"OpenRouter provider initialized with models: {self.available_models}")

    async def analyze_relevance(self, article: Article, topic: Topic) -> AIResult:
        """Analyze article relevance using OpenRouter models with fallback.

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

        # All models failed
        error_msg = f"All OpenRouter models failed. Last error: {last_error}"
        self.logger.error(error_msg)
        return AIResult(
            success=False,
            relevance_score=0.0,
            confidence=0.0,
            error_message=error_msg
        )

    async def _analyze_with_model(self, article: Article, topic: Topic, model_name: str) -> AIResult:
        """Analyze relevance using specific model."""
        self._update_rate_limit()

        # Build analysis prompt
        prompt = self._build_relevance_prompt(article, topic)

        try:
            # Make API request
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing article relevance to topics. Respond with JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30
            )

            # Parse response
            content = response.choices[0].message.content.strip()
            result = self._parse_relevance_response(content)

            self.logger.debug(f"OpenRouter analysis successful with {model_name}")
            return self._create_success_result(
                relevance_score=result["relevance_score"],
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )

        except openai.RateLimitError as e:
            self._handle_rate_limit_error()
            raise AIError(
                f"OpenRouter rate limit: {e}",
                provider="openrouter",
                error_code=ErrorCode.AI_RATE_LIMIT
            )
        except openai.APIError as e:
            raise AIError(
                f"OpenRouter API error: {e}",
                provider="openrouter",
                error_code=ErrorCode.AI_API_ERROR
            )
        except Exception as e:
            raise AIError(
                f"OpenRouter request failed: {e}",
                provider="openrouter",
                error_code=ErrorCode.AI_API_ERROR
            )

    async def generate_summary(self, article: Article, max_sentences: int = 3) -> AIResult:
        """Generate article summary using OpenRouter models.

        Args:
            article: Article to summarize
            max_sentences: Maximum sentences in summary

        Returns:
            AIResult with generated summary
        """
        if not self._can_make_request():
            return self._create_error_result("Rate limit exceeded")

        # Try each available model
        for model_name in self.available_models:
            try:
                return await self._generate_summary_with_model(article.content, model_name)
            except Exception as e:
                self.logger.warning(f"Summary generation failed with {model_name}: {e}")
                continue

        return self._create_error_result("All OpenRouter models failed for summary generation")

    async def _generate_summary_with_model(self, content: str, model_name: str) -> AIResult:
        """Generate summary using specific model."""
        self._update_rate_limit()

        # Truncate content if too long
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."

        prompt = f"""Summarize this article in 2-3 sentences, focusing on key insights and main points:

{content}

Provide a concise, informative summary that captures the essence of the article."""

        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, informative article summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200,
                timeout=30
            )

            summary = response.choices[0].message.content.strip()

            return self._create_success_result(
                relevance_score=1.0,  # Summary always succeeds if we get here
                confidence=0.9,       # High confidence for summarization
                summary=summary,
                content=summary,      # Also set content for general access
                tokens_used=getattr(response.usage, 'total_tokens', None) if hasattr(response, 'usage') else None
            )

        except Exception as e:
            raise AIError(
                f"OpenRouter summary generation failed: {e}",
                provider="openrouter",
                error_code=ErrorCode.AI_API_ERROR
            )

    async def generate_keywords(self, topic_name: str, context: str = "", max_keywords: int = 7) -> AIResult:
        """Generate keywords for a topic using OpenRouter models.

        Args:
            topic_name: Name of the topic to generate keywords for
            context: Additional context (not used to prevent contamination)
            max_keywords: Maximum number of keywords to generate

        Returns:
            AIResult with generated keywords
        """
        if not self._can_make_request():
            return self._create_error_result("Rate limit exceeded")

        # Try each available model
        for model_name in self.available_models:
            try:
                return await self._generate_keywords_with_model(topic_name, max_keywords, model_name)
            except Exception as e:
                self.logger.warning(f"Keyword generation failed with {model_name}: {e}")
                continue

        return self._create_error_result("All OpenRouter models failed for keyword generation")

    async def _generate_keywords_with_model(self, topic_name: str, max_keywords: int, model_name: str) -> AIResult:
        """Generate keywords using specific model."""
        self._update_rate_limit()

        prompt = f"""Generate {max_keywords} relevant keywords for the topic "{topic_name}".

Requirements:
- Return ONLY keywords, one per line
- No explanations or additional text
- Keywords should be specific and relevant
- Include both broad and specific terms
- Focus on technology, business, and industry terms

Topic: {topic_name}

Keywords:"""

        try:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert at generating relevant keywords for content topics. Respond with only keywords, one per line."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                timeout=30
            )

            content = response.choices[0].message.content.strip()

            # Parse keywords from response
            keywords = []
            for line in content.split('\n'):
                keyword = line.strip().strip('-').strip('•').strip()
                if keyword and len(keyword) > 1:
                    keywords.append(keyword)

            # Limit to requested number
            keywords = keywords[:max_keywords]

            # Ensure we have at least the topic name as fallback
            if not keywords:
                keywords = [topic_name.lower()]

            return self._create_success_result(
                relevance_score=1.0,  # Keywords always succeed if we get here
                confidence=0.9,       # High confidence for keyword generation
                content=keywords
            )

        except Exception as e:
            raise AIError(
                f"OpenRouter keyword generation failed: {e}",
                provider="openrouter",
                error_code=ErrorCode.AI_API_ERROR
            )

    def _build_relevance_prompt(self, article: Article, topic: Topic) -> str:
        """Build prompt for relevance analysis."""
        keywords_str = ", ".join(topic.keywords)

        return f"""Analyze if this article is relevant to the topic "{topic.name}".

Topic Keywords: {keywords_str}
Confidence Threshold: {topic.confidence_threshold}

Article:
Title: {article.title}
Content: {article.content[:2000]}

Respond with JSON only:
{{
    "relevance_score": <float 0.0-1.0>,
    "confidence": <float 0.0-1.0>,
    "reasoning": "<brief explanation of relevance>"
}}"""

    def _parse_relevance_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from relevance analysis."""
        try:
            # Try to find JSON in response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")

            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            # Validate required fields
            if not all(key in result for key in ["relevance_score", "confidence", "reasoning"]):
                raise ValueError("Missing required fields in response")

            # Ensure scores are in valid range
            result["relevance_score"] = max(0.0, min(1.0, float(result["relevance_score"])))
            result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))

            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to parse OpenRouter response: {e}")
            # Return neutral fallback values when response is unparseable
            return {
                "relevance_score": 0.5,  # Neutral score when AI response fails
                "confidence": 0.3,       # Low confidence for fallback
                "reasoning": "Fallback response due to unparseable AI output"
            }

    def _can_make_request(self) -> bool:
        """Check if we can make another request based on rate limits."""
        current_time = time.time()

        # Reset minute counter if needed
        if current_time - self._minute_start >= 60:
            self._request_count_minute = 0
            self._minute_start = current_time

        return self._request_count_minute < self._rate_limit_info.requests_per_minute

    def _update_rate_limit(self) -> None:
        """Update rate limit tracking after making a request."""
        self._last_request_time = time.time()
        self._request_count_minute += 1

    def _handle_rate_limit_error(self) -> None:
        """Handle rate limit error from API."""
        self._rate_limit_info.reset_time = time.time() + 60  # 1 minute cooldown
        self.logger.warning("OpenRouter rate limit hit, applying cooldown")

    def get_rate_limits(self) -> RateLimitInfo:
        """Get current rate limit information."""
        return self._rate_limit_info

    async def test_connection(self) -> bool:
        """Test OpenRouter connection with a simple request."""
        try:
            response = await self.client.chat.completions.create(
                model=self.available_models[0],
                messages=[{"role": "user", "content": "Hello, respond with 'OK' if you can hear me."}],
                max_tokens=10,
                timeout=10
            )

            return bool(response.choices and response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"OpenRouter connection test failed: {e}")
            return False

    async def close(self) -> None:
        """Close client connections."""
        try:
            await self.client.close()
        except Exception as e:
            self.logger.warning(f"Error closing OpenRouter client: {e}")