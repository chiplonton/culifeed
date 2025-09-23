"""
Smart Keyword Analyzer with Confidence Scoring
==============================================

Core component for intelligent article relevance assessment with confidence scoring.
Reduces AI processing costs by identifying obviously relevant/irrelevant articles.

Features:
- Multi-factor confidence scoring (keyword matching, content quality, URL patterns)
- High/low confidence routing decisions
- Basic content similarity detection
- Configurable thresholds for routing decisions
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone

from ..database.models import Article, Topic
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component


@dataclass
class ConfidenceResult:
    """Result of smart keyword analysis with confidence scoring."""
    relevance_score: float
    confidence_level: float  # 0.0-1.0, how confident we are in the relevance_score
    matched_keywords: List[str]
    content_quality_score: float
    url_quality_score: float
    routing_decision: str  # "high_confidence", "low_confidence", "uncertain"
    reasoning: List[str]  # Human-readable explanation of scoring factors


class SmartKeywordAnalyzer:
    """Enhanced keyword analyzer with multi-factor confidence scoring."""
    
    def __init__(self):
        """Initialize smart keyword analyzer."""
        self.settings = get_settings()
        self.logger = get_logger_for_component('smart_analyzer')
        
        # Simple content similarity cache (hash -> relevance_score)
        self._similarity_cache: Dict[str, float] = {}
        
        # URL quality patterns
        self._high_quality_domains = {
            'github.com', 'stackoverflow.com', 'medium.com', 'dev.to',
            'docs.aws.amazon.com', 'cloud.google.com', 'azure.microsoft.com'
        }
        
        self._low_quality_patterns = [
            r'clickbait|listicle|sponsored|advertisement',
            r'\d+\s+reasons?\s+why',
            r'you\s+won\'?t\s+believe',
            r'shocking|amazing|incredible'
        ]
        
        # Content quality indicators
        self._quality_indicators = {
            'high': [
                r'documentation|tutorial|guide|best\s+practices',
                r'implementation|example|code\s+sample',
                r'official|announcement|release\s+notes',
                r'technical\s+analysis|deep\s+dive|comprehensive'
            ],
            'low': [
                r'breaking|urgent|must\s+read|trending',
                r'top\s+\d+|list\s+of|ranked',
                r'opinion|rant|controversial',
                r'quick\s+tip|life\s+hack|simple\s+trick'
            ]
        }

    def analyze_article_confidence(self, article: Article, topic: Topic) -> ConfidenceResult:
        """
        Analyze article relevance with confidence scoring.
        
        Args:
            article: Article to analyze
            topic: Topic to match against
            
        Returns:
            ConfidenceResult with routing decision
        """
        reasoning = []
        
        # 1. Keyword matching analysis
        keyword_score, matched_keywords = self._analyze_keyword_matching(article, topic)
        reasoning.append(f"Keyword matching: {keyword_score:.2f} (matched: {', '.join(matched_keywords) if matched_keywords else 'none'})")
        
        # 2. Content quality assessment
        content_quality = self._assess_content_quality(article)
        reasoning.append(f"Content quality: {content_quality:.2f}")
        
        # 3. URL quality assessment
        url_quality = self._assess_url_quality(article)
        reasoning.append(f"URL quality: {url_quality:.2f}")
        
        # 4. Content similarity check
        similarity_bonus = self._check_content_similarity(article)
        if similarity_bonus > 0:
            reasoning.append(f"Similar content bonus: +{similarity_bonus:.2f}")
        
        # 5. Calculate composite relevance score
        relevance_score = self._calculate_composite_score(
            keyword_score, content_quality, url_quality, similarity_bonus
        )
        
        # 6. Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            keyword_score, matched_keywords, content_quality, url_quality
        )
        
        # 7. Make routing decision
        routing_decision = self._make_routing_decision(relevance_score, confidence_level)
        
        reasoning.append(f"Final: score={relevance_score:.2f}, confidence={confidence_level:.2f}, routing={routing_decision}")
        
        return ConfidenceResult(
            relevance_score=relevance_score,
            confidence_level=confidence_level,
            matched_keywords=matched_keywords,
            content_quality_score=content_quality,
            url_quality_score=url_quality,
            routing_decision=routing_decision,
            reasoning=reasoning
        )

    def _analyze_keyword_matching(self, article: Article, topic: Topic) -> Tuple[float, List[str]]:
        """Analyze keyword matching with weighted scoring."""
        import json
        
        # Handle different keyword formats
        try:
            if isinstance(topic.keywords, list):
                # Already a Python list
                keywords = [k.strip().lower() for k in topic.keywords if k.strip()]
            elif isinstance(topic.keywords, str):
                if topic.keywords.startswith('[') and topic.keywords.endswith(']'):
                    # JSON array format: ["aws", "amazon", ...]
                    keywords = [k.strip().lower() for k in json.loads(topic.keywords) if k.strip()]
                else:
                    # Comma-separated format: aws, amazon, ...
                    keywords = [k.strip().lower() for k in topic.keywords.split(',') if k.strip()]
            else:
                self.logger.warning(f"Unexpected keywords type for topic {topic.name}: {type(topic.keywords)}")
                return 0.0, []
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse keywords for topic {topic.name}: {e}")
            return 0.0, []
        
        if not keywords:
            return 0.0, []
        
        # Combine title and content for analysis (Article model uses 'summary', not 'description')
        text_content = f"{article.title} {article.summary or ''} {article.content or ''}".lower()
        
        matched_keywords = []
        total_score = 0.0
        
        # IMPROVED: Get generic patterns from configuration instead of hard-coding
        generic_patterns = set()
        
        if self.settings.smart_processing.generic_patterns_enabled:
            # Flatten all categorized patterns into a single set
            for category, patterns in self.settings.smart_processing.generic_patterns.items():
                generic_patterns.update(patterns)
            
            self.logger.debug(f"Loaded {len(generic_patterns)} generic patterns from {len(self.settings.smart_processing.generic_patterns)} categories")
        else:
            self.logger.debug("Generic pattern classification disabled in settings")
        
        topic_specific_keywords = set()
        
        # FIXED: Classify keywords based on semantic meaning, not just word count
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if kw_lower not in generic_patterns:
                # Only truly domain-specific keywords count as topic-specific
                topic_specific_keywords.add(kw_lower)
            
        self.logger.debug(f"Keyword classification - Generic patterns: {len(generic_patterns)}, Topic-specific: {topic_specific_keywords}")
        
        # Track if we match any topic-specific keywords
        matched_topic_specific = False
        
        for keyword in keywords:
            if not keyword:
                continue
                
            # Determine keyword specificity weight
            is_generic = keyword.lower() in generic_patterns
            is_topic_specific = keyword.lower() in topic_specific_keywords
            specificity_multiplier = 0.3 if is_generic else 1.0  # Generic keywords get 30% weight
                
            # Multi-word keyword handling (these are usually more specific)
            if ' ' in keyword:
                # Exact phrase matching (higher weight for specific multi-word terms)
                if keyword in text_content:
                    matched_keywords.append(keyword)
                    if is_topic_specific:
                        matched_topic_specific = True
                    phrase_score = self.settings.filtering.exact_phrase_weight * 1.5  # Boost multi-word matches
                    total_score += phrase_score * specificity_multiplier
                else:
                    # Partial word matching for multi-word keywords
                    words = keyword.split()
                    word_matches = sum(1 for word in words if word in text_content)
                    if word_matches > 0:
                        partial_score = (word_matches / len(words)) * self.settings.filtering.partial_word_weight
                        total_score += partial_score * specificity_multiplier
                        if word_matches == len(words):
                            matched_keywords.append(keyword)
                            if is_topic_specific:
                                matched_topic_specific = True
            else:
                # Single word matching with TF cap
                word_count = text_content.count(keyword)
                if word_count > 0:
                    matched_keywords.append(keyword)
                    if is_topic_specific:
                        matched_topic_specific = True
                    # Apply TF cap to prevent keyword stuffing inflation
                    tf_score = min(word_count * 0.1, self.settings.filtering.single_word_tf_cap)
                    total_score += tf_score * specificity_multiplier
        
        # Apply keyword match bonus for multiple matches
        if len(matched_keywords) > 1:
            total_score *= (1 + self.settings.filtering.keyword_match_bonus)
        
        # CRITICAL IMPROVEMENTS for semantic accuracy:
        
        # DEBUG: Log that semantic improvements are active
        self.logger.info(f"🔧 SEMANTIC ANALYZER v2.0 ACTIVE - Processing: {article.title[:30]}...")
        
        # 1. If no topic-specific keywords matched, severely penalize the score
        if not matched_topic_specific and matched_keywords:
            self.logger.info(f"❌ No topic-specific keywords matched, only generic: {matched_keywords}")
            total_score *= 0.1  # Reduce to 10% if only generic keywords matched
        
        # 2. SEMANTIC CONTEXT CHECK: Require multiple related keywords for high confidence
        # Single ambiguous keyword matches should not get high confidence
        ambiguous_keywords = {'coding', 'programming', 'software engineers', 'algorithm', 'development', 'aws', 'amazon'}
        
        if len(matched_keywords) == 1 and matched_keywords[0].lower() in ambiguous_keywords:
            self.logger.debug(f"Single ambiguous keyword '{matched_keywords[0]}' - reducing confidence")
            total_score *= 0.4  # Reduce score for single ambiguous matches
        
        # 3. TOPIC COHERENCE CHECK: For topics with specific themes, require theme coherence
        # Check if matched keywords make sense together for the topic
        topic_name_lower = topic.name.lower()
        
        if 'tiktok' in topic_name_lower:
            # TikTok topic should have TikTok-specific or social media context
            tiktok_specific = ['tiktok', 'social media', 'app development', 'mobile app']
            has_tiktok_context = any(kw.lower() in tiktok_specific for kw in matched_keywords)
            if not has_tiktok_context and matched_keywords:
                self.logger.debug(f"TikTok topic without TikTok context, reducing score")
                total_score *= 0.3
        
        elif 'ecs' in topic_name_lower or 'eks' in topic_name_lower:
            # ECS/EKS topic should have container/kubernetes context
            container_specific = ['ecs', 'eks', 'kubernetes', 'container', 'containerization', 'aws ecs', 'aws eks', 'amazon ecs', 'amazon eks']
            has_container_context = any(kw.lower() in container_specific for kw in matched_keywords)
            if not has_container_context and matched_keywords:
                self.logger.debug(f"ECS/EKS topic without container context, reducing score")
                total_score *= 0.3
        
        elif 'engineering culture' in topic_name_lower or 'management' in topic_name_lower:
            # Management topic should have leadership/culture context
            management_specific = ['leadership', 'management', 'culture', 'team', 'workplace', 'emotional intelligence']
            has_management_context = any(kw.lower() in management_specific for kw in matched_keywords)
            if not has_management_context and matched_keywords:
                self.logger.debug(f"Management topic without leadership context, reducing score")
                total_score *= 0.3
        
        # Normalize score to 0-1 range
        normalized_score = min(total_score, 1.0)
        
        return normalized_score, matched_keywords

    def _assess_content_quality(self, article: Article) -> float:
        """Assess content quality indicators."""
        text_content = f"{article.title} {article.summary or ''} {article.content or ''}".lower()
        
        quality_score = 0.5  # Neutral baseline
        
        # Check for high-quality indicators
        for pattern in self._quality_indicators['high']:
            if re.search(pattern, text_content, re.IGNORECASE):
                quality_score += 0.15
        
        # Check for low-quality indicators (penalty)
        for pattern in self._quality_indicators['low']:
            if re.search(pattern, text_content, re.IGNORECASE):
                quality_score -= 0.1
        
        # Content length factor (moderate length preferred)
        content_length = len(article.content or '')
        if 500 <= content_length <= 3000:
            quality_score += 0.1
        elif content_length < 100:
            quality_score -= 0.2
        
        return max(0.0, min(1.0, quality_score))

    def _assess_url_quality(self, article: Article) -> float:
        """Assess URL quality indicators."""
        if not article.url:
            return 0.5
        
        url_lower = str(article.url).lower()
        quality_score = 0.5  # Neutral baseline
        
        # High-quality domains
        for domain in self._high_quality_domains:
            if domain in url_lower:
                quality_score += 0.3
                break
        
        # Low-quality patterns (penalty)
        for pattern in self._low_quality_patterns:
            if re.search(pattern, url_lower, re.IGNORECASE):
                quality_score -= 0.2
                break
        
        # URL structure quality
        if '/docs/' in url_lower or '/documentation/' in url_lower:
            quality_score += 0.2
        elif '/blog/' in url_lower or '/news/' in url_lower:
            quality_score += 0.1
        elif '/ad/' in url_lower or '/promo/' in url_lower:
            quality_score -= 0.3
        
        return max(0.0, min(1.0, quality_score))

    def _check_content_similarity(self, article: Article) -> float:
        """Check for similar content using simple hash-based similarity."""
        if not article.content:
            return 0.0
        
        # Create content hash for similarity detection
        content_normalized = re.sub(r'\s+', ' ', article.content.lower().strip())
        content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
        
        # Check cache for similar content
        if content_hash in self._similarity_cache:
            cached_score = self._similarity_cache[content_hash]
            # Apply small bonus for consistent scoring
            return min(0.1, cached_score * 0.1)
        
        # Store in cache for future similarity checks
        # For now, return 0 since we don't have historical data
        self._similarity_cache[content_hash] = 0.5
        return 0.0

    def _calculate_composite_score(self, keyword_score: float, content_quality: float, 
                                 url_quality: float, similarity_bonus: float) -> float:
        """Calculate composite relevance score with weighted factors."""
        # Weights from filtering settings
        title_weight = self.settings.filtering.title_quality_weight
        content_weight = self.settings.filtering.content_quality_weight
        url_weight = self.settings.filtering.url_quality_weight
        
        # Keyword score is primary factor
        composite_score = (
            keyword_score * 0.6 +  # Primary relevance indicator
            content_quality * content_weight +
            url_quality * url_weight +
            similarity_bonus
        )
        
        return max(0.0, min(1.0, composite_score))

    def _calculate_confidence_level(self, keyword_score: float, matched_keywords: List[str],
                                  content_quality: float, url_quality: float) -> float:
        """Calculate confidence level in the relevance assessment."""
        confidence_factors = []
        
        # Keyword confidence
        if keyword_score > 0.8 and len(matched_keywords) >= 2:
            confidence_factors.append(0.9)  # High keyword confidence
        elif keyword_score > 0.5 and len(matched_keywords) >= 1:
            confidence_factors.append(0.7)  # Moderate keyword confidence
        elif keyword_score > 0.2:
            confidence_factors.append(0.4)  # Low keyword confidence
        else:
            confidence_factors.append(0.1)  # Very low keyword confidence
        
        # Content quality confidence
        if content_quality > 0.7:
            confidence_factors.append(0.8)
        elif content_quality > 0.4:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # URL quality confidence
        if url_quality > 0.7:
            confidence_factors.append(0.7)
        elif url_quality > 0.4:
            confidence_factors.append(0.5)
        else:
            confidence_factors.append(0.2)
        
        # Average confidence with keyword dominance
        keyword_weight = 0.6
        other_weight = 0.4 / (len(confidence_factors) - 1) if len(confidence_factors) > 1 else 0.4
        
        weighted_confidence = (
            confidence_factors[0] * keyword_weight +
            sum(confidence_factors[1:]) * other_weight
        )
        
        return max(0.1, min(1.0, weighted_confidence))

    def _make_routing_decision(self, relevance_score: float, confidence_level: float) -> str:
        """Make routing decision based on score and confidence."""
        # High confidence routing
        if confidence_level >= 0.8:
            if relevance_score >= 0.7:
                return "high_confidence"  # Skip AI, definitely relevant
            elif relevance_score <= 0.3:
                return "low_confidence"   # Skip AI, definitely irrelevant
        
        # Medium confidence routing
        elif confidence_level >= 0.6:
            if relevance_score >= 0.8:
                return "high_confidence"  # Skip AI, very likely relevant
            elif relevance_score <= 0.2:
                return "low_confidence"   # Skip AI, very likely irrelevant
        
        # IMPROVED: Block semantically penalized articles with low scores
        # These are articles that matched keywords but failed semantic coherence checks
        elif confidence_level <= 0.3 and relevance_score <= 0.35:
            self.logger.debug(f"Blocking semantically penalized article: score={relevance_score:.3f}, confidence={confidence_level:.3f}")
            return "definitely_irrelevant"  # Block articles that are semantically incoherent
        
        # Default to uncertain (needs AI processing)
        return "uncertain"

    def clear_similarity_cache(self):
        """Clear the similarity cache (useful for testing)."""
        self._similarity_cache.clear()
        self.logger.debug("Similarity cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get similarity cache statistics."""
        return {
            'cache_size': len(self._similarity_cache),
            'total_entries': len(self._similarity_cache)
        }