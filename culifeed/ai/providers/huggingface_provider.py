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
    
    # FREE models available on HuggingFace Inference API
    RECOMMENDED_MODELS = [
        "microsoft/DialoGPT-medium",           # Conversational AI
        "google/flan-t5-large",                # Text-to-text generation
        "meta-llama/Llama-2-7b-chat-hf",      # Chat model (if available)
        "mistralai/Mistral-7B-Instruct-v0.1", # Instruction following
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
        
        # All models failed
        error_msg = f"All HuggingFace models failed. Last error: {last_error}"
        self.logger.error(error_msg)
        return AIResult(
            success=False,
            relevance_score=0.0,
            confidence=0.0,
            error_message=error_msg,
            provider="huggingface",
            error_code=ErrorCode.AI_API_ERROR
        )
    
    async def _analyze_with_model(self, article: Article, topic: Topic, model_name: str) -> AIResult:
        """Analyze relevance using specific model."""
        self._update_rate_limit()
        
        # Build analysis prompt
        prompt = self._build_relevance_prompt(article, topic)
        
        try:
            # Make API request
            response_text = await self._make_inference_request(model_name, prompt)
            
            # Parse response - HuggingFace models may return different formats
            result = self._parse_relevance_response(response_text)
            
            self.logger.debug(f"HuggingFace analysis successful with {model_name}")
            return self._create_success_result(
                relevance_score=result["relevance_score"],
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                matched_keywords=result.get("matched_keywords", [])
            )
        
        except Exception as e:
            raise AIError(
                f"HuggingFace inference failed: {e}",
                provider="huggingface",
                error_code=ErrorCode.AI_API_ERROR
            )
    
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
        
        # Try each available model
        for model_name in self.available_models:
            try:
                return await self._generate_summary_with_model(article.content, model_name)
            except Exception as e:
                self.logger.warning(f"Summary generation failed with {model_name}: {e}")
                continue
        
        return self._create_error_result("All HuggingFace models failed for summary generation")
    
    async def _generate_summary_with_model(self, content: str, model_name: str) -> AIResult:
        """Generate summary using specific model."""
        self._update_rate_limit()
        
        # Truncate content if too long
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        prompt = f"""Summarize this article in 2-3 sentences, focusing on key insights and main points:

{content}

Provide a concise, informative summary that captures the essence of the article."""
        
        try:
            summary = await self._make_inference_request(model_name, prompt)
            
            # Clean up the summary
            summary = summary.strip()
            if summary.startswith("Summary:"):
                summary = summary.replace("Summary:", "").strip()
            
            return self._create_success_result(
                relevance_score=1.0,  # Summary always succeeds if we get here
                confidence=0.9,       # High confidence for summarization
                summary=summary
            )
        
        except Exception as e:
            raise AIError(
                f"HuggingFace summary generation failed: {e}",
                provider="huggingface",
                error_code=ErrorCode.AI_API_ERROR
            )
    
    async def generate_keywords(self, topic_name: str, context: str = "", max_keywords: int = 7) -> AIResult:
        """Generate keywords for a topic using HuggingFace models.
        
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
        
        return self._create_error_result("All HuggingFace models failed for keyword generation")
    
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
            response = await self._make_inference_request(model_name, prompt)
            
            # Parse keywords from response
            keywords = []
            for line in response.split('\n'):
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
                f"HuggingFace keyword generation failed: {e}",
                provider="huggingface",
                error_code=ErrorCode.AI_API_ERROR
            )
    
    async def _make_inference_request(self, model_name: str, prompt: str, max_length: int = 500) -> str:
        """Make inference request to HuggingFace API.
        
        Args:
            model_name: Model to use for inference
            prompt: Input prompt
            max_length: Maximum length of generated text
            
        Returns:
            Generated text response
        """
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}/{model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Different payload structures for different model types
        if "t5" in model_name.lower():
            # T5 models expect text-to-text format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": max_length,
                    "temperature": 0.1,
                    "do_sample": True
                }
            }
        else:
            # Most other models expect text generation format
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": 0.1,
                    "return_full_text": False
                }
            }
        
        try:
            async with self._session.post(url, headers=headers, json=payload, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Parse response based on model type
                    if isinstance(result, list) and len(result) > 0:
                        if "generated_text" in result[0]:
                            return result[0]["generated_text"]
                        elif "translation_text" in result[0]:
                            return result[0]["translation_text"]
                        else:
                            return str(result[0])
                    elif isinstance(result, dict):
                        return result.get("generated_text", str(result))
                    else:
                        return str(result)
                
                elif response.status == 503:
                    # Model is loading
                    error_data = await response.json()
                    estimated_time = error_data.get("estimated_time", 60)
                    raise AIError(
                        f"Model {model_name} is loading, estimated time: {estimated_time}s",
                        provider="huggingface",
                        error_code=ErrorCode.AI_API_ERROR,
                        retryable=True
                    )
                
                elif response.status == 429:
                    # Rate limited
                    self._handle_rate_limit_error()
                    raise AIError(
                        "HuggingFace rate limit exceeded",
                        provider="huggingface",
                        error_code=ErrorCode.AI_RATE_LIMIT,
                        rate_limited=True
                    )
                
                else:
                    error_text = await response.text()
                    raise AIError(
                        f"HuggingFace API error {response.status}: {error_text}",
                        provider="huggingface",
                        error_code=ErrorCode.AI_API_ERROR
                    )
        
        except aiohttp.ClientError as e:
            raise AIError(
                f"HuggingFace connection error: {e}",
                provider="huggingface",
                error_code=ErrorCode.AI_CONNECTION_ERROR,
                retryable=True
            )
    
    def _parse_relevance_response(self, content: str) -> Dict[str, Any]:
        """Parse relevance response from HuggingFace model.
        
        Args:
            content: Raw response content
            
        Returns:
            Dictionary with parsed relevance data
        """
        try:
            # Try to parse structured response first
            lines = content.strip().split('\n')
            result = {
                "relevance_score": 0.0,
                "confidence": 0.0,
                "reasoning": "No reasoning provided",
                "matched_keywords": []
            }
            
            for line in lines:
                line = line.strip()
                if "relevance" in line.lower() and any(char.isdigit() for char in line):
                    # Extract relevance score
                    import re
                    score_match = re.search(r'(\d+\.?\d*)', line)
                    if score_match:
                        score = float(score_match.group(1))
                        # Normalize to 0-1 range if needed
                        if score > 1.0:
                            score = score / 10.0 if score <= 10.0 else score / 100.0
                        result["relevance_score"] = min(1.0, max(0.0, score))
                
                elif "confidence" in line.lower() and any(char.isdigit() for char in line):
                    # Extract confidence score
                    import re
                    conf_match = re.search(r'(\d+\.?\d*)', line)
                    if conf_match:
                        conf = float(conf_match.group(1))
                        if conf > 1.0:
                            conf = conf / 10.0 if conf <= 10.0 else conf / 100.0
                        result["confidence"] = min(1.0, max(0.0, conf))
            
            # If no structured scores found, use heuristic analysis
            if result["relevance_score"] == 0.0:
                content_lower = content.lower()
                if any(word in content_lower for word in ["relevant", "matches", "related", "applies"]):
                    result["relevance_score"] = 0.7
                    result["confidence"] = 0.6
                elif any(word in content_lower for word in ["irrelevant", "unrelated", "different"]):
                    result["relevance_score"] = 0.2
                    result["confidence"] = 0.6
                else:
                    result["relevance_score"] = 0.5
                    result["confidence"] = 0.4
            
            result["reasoning"] = content[:200] + "..." if len(content) > 200 else content
            return result
        
        except Exception as e:
            self.logger.warning(f"Failed to parse HuggingFace response: {e}")
            # Return conservative fallback
            return {
                "relevance_score": 0.3,  # Conservative score when parsing fails
                "confidence": 0.3,
                "reasoning": "Unable to parse model response, using conservative score",
                "matched_keywords": []
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
        self.logger.warning("HuggingFace rate limit hit, applying cooldown")
    
    def get_rate_limits(self) -> RateLimitInfo:
        """Get current rate limit information."""
        return self._rate_limit_info
    
    async def test_connection(self) -> bool:
        """Test HuggingFace connection with a simple request."""
        try:
            response = await self._make_inference_request(
                self.available_models[0],
                "Hello, respond with 'OK' if you can hear me.",
                max_length=10
            )
            
            return bool(response and len(response.strip()) > 0)
        
        except Exception as e:
            self.logger.error(f"HuggingFace connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None