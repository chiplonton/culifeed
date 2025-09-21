"""
Pre-Filtering Engine
===================

High-performance pre-filtering system using keyword matching, content analysis,
and relevance scoring to reduce AI processing costs.
"""

import re
import math
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timezone

from ..database.models import Article, Topic
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component
from ..utils.exceptions import ProcessingError, ErrorCode


@dataclass
class FilterResult:
    """Result of pre-filtering operation."""
    article: Article
    matched_topics: List[str]
    relevance_scores: Dict[str, float]
    passed_filter: bool
    filter_reason: Optional[str] = None
    
    @property
    def best_match_topic(self) -> Optional[str]:
        """Get the topic with highest relevance score."""
        if not self.relevance_scores:
            return None
        return max(self.relevance_scores.keys(), key=lambda k: self.relevance_scores[k])
    
    @property
    def best_match_score(self) -> float:
        """Get the highest relevance score."""
        if not self.relevance_scores:
            return 0.0
        return max(self.relevance_scores.values())


class PreFilterEngine:
    """Keyword-based pre-filtering to reduce AI processing costs."""
    
    def __init__(self, settings: Optional['CuliFeedSettings'] = None):
        """Initialize pre-filter engine.
        
        Args:
            settings: CuliFeed settings with configurable thresholds
        """
        # Import here to avoid circular imports
        if settings is None:
            from ..config.settings import get_settings
            settings = get_settings()
        
        self.settings = settings
        self.min_relevance_threshold = settings.filtering.min_relevance_threshold
        self.logger = get_logger_for_component("pre_filter")
        
        # Configure filtering thresholds from settings
        self.thresholds = {
            'exact_phrase_weight': settings.filtering.exact_phrase_weight,
            'partial_word_weight': settings.filtering.partial_word_weight,
            'single_word_tf_cap': settings.filtering.single_word_tf_cap,
            'keyword_match_bonus': settings.filtering.keyword_match_bonus,
        }
        
        # Compile common stop words for better keyword matching
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how',
            'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so'
        }
        
        self.logger.info("Pre-filter engine initialized with configurable thresholds", 
                        extra={'min_relevance_threshold': self.min_relevance_threshold, 
                               'thresholds': self.thresholds})
    
    def _extract_text_features(self, article: Article) -> Dict[str, any]:
        """Extract text features from article for analysis.
        
        Args:
            article: Article to analyze
            
        Returns:
            Dictionary of text features
        """
        # Combine title and content for analysis
        full_text = (article.title or '') + ' ' + (article.content or '')
        full_text = full_text.lower()
        
        # Clean text: remove HTML tags, special chars, normalize whitespace
        clean_text = re.sub(r'<[^>]+>', ' ', full_text)  # Remove HTML tags
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)  # Remove special chars
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
        
        # Extract words (excluding stop words)
        words = [word for word in clean_text.split() if len(word) > 2 and word not in self.stop_words]
        
        # Create word frequency counter
        word_counts = Counter(words)
        
        # Calculate TF (term frequency) scores
        total_words = len(words)
        tf_scores = {word: count / total_words for word, count in word_counts.items()}
        
        return {
            'clean_text': clean_text,
            'words': words,
            'word_counts': word_counts,
            'tf_scores': tf_scores,
            'total_words': total_words
        }
    
    def _calculate_keyword_relevance(self, text_features: Dict[str, any], keywords: List[str]) -> float:
        """Calculate relevance score for a set of keywords.
        
        Args:
            text_features: Extracted text features from article
            keywords: List of keywords to match against
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        if not keywords or not text_features['words']:
            return 0.0
        
        clean_text = text_features['clean_text']
        word_counts = text_features['word_counts']
        tf_scores = text_features['tf_scores']
        
        total_score = 0.0
        matched_keywords = 0
        
        for keyword in keywords:
            keyword = keyword.lower().strip()
            if not keyword:
                continue
            
            keyword_score = 0.0
            
            # Exact phrase matching (highest weight)
            if keyword in clean_text:
                # Count phrase occurrences
                phrase_count = clean_text.count(keyword)
                phrase_score = min(phrase_count * self.thresholds['exact_phrase_weight'], 1.0)  # Cap at 1.0
                keyword_score = max(keyword_score, phrase_score)
                matched_keywords += 1
            
            # Individual word matching within keyword phrase
            keyword_words = [w for w in keyword.split() if len(w) > 2]
            if len(keyword_words) > 1:
                # Multi-word keyword: check for individual word matches
                word_matches = sum(1 for word in keyword_words if word in word_counts)
                if word_matches > 0:
                    # Partial match score based on word coverage
                    word_coverage = word_matches / len(keyword_words)
                    word_score = word_coverage * self.thresholds['partial_word_weight']  # Lower weight than exact phrase
                    keyword_score = max(keyword_score, word_score)
                    if word_matches == len(keyword_words):
                        matched_keywords += 1
            
            elif keyword_words:
                # Single word keyword: use TF score
                word = keyword_words[0]
                if word in tf_scores:
                    word_score = min(tf_scores[word] * 10, self.thresholds['single_word_tf_cap'])  # Scale TF and cap
                    keyword_score = max(keyword_score, word_score)
                    matched_keywords += 1
            
            total_score += keyword_score
        
        if not matched_keywords:
            return 0.0
        
        # Normalize by number of keywords and apply bonus for multiple matches
        base_score = total_score / len(keywords)
        match_bonus = min(matched_keywords / len(keywords), 1.0) * self.thresholds['keyword_match_bonus']
        
        final_score = min(base_score + match_bonus, 1.0)
        return final_score
    
    def _check_exclusion_keywords(self, text_features: Dict[str, any], exclude_keywords: List[str]) -> bool:
        """Check if article should be excluded based on exclusion keywords.
        
        Args:
            text_features: Extracted text features from article
            exclude_keywords: List of keywords that should exclude the article
            
        Returns:
            True if article should be excluded, False otherwise
        """
        if not exclude_keywords:
            return False
        
        clean_text = text_features['clean_text']
        
        for keyword in exclude_keywords:
            keyword = keyword.lower().strip()
            if keyword and keyword in clean_text:
                return True
        
        return False
    
    def filter_article(self, article: Article, topics: List[Topic]) -> FilterResult:
        """Filter a single article against topics.
        
        Args:
            article: Article to filter
            topics: List of topics to match against
            
        Returns:
            FilterResult with relevance scores and filter decision
        """
        if not topics:
            return FilterResult(
                article=article,
                matched_topics=[],
                relevance_scores={},
                passed_filter=False,
                filter_reason="No topics configured"
            )
        
        # Extract text features once
        text_features = self._extract_text_features(article)
        
        relevance_scores = {}
        matched_topics = []
        exclusion_reasons = []
        
        for topic in topics:
            # Skip inactive topics
            if not topic.active:
                continue
            
            # Check exclusion keywords first
            if self._check_exclusion_keywords(text_features, topic.exclude_keywords):
                exclusion_reasons.append(f"Excluded by topic '{topic.name}'")
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_keyword_relevance(text_features, topic.keywords)
            
            if relevance_score > 0:
                relevance_scores[topic.name] = relevance_score
                
                # Check if score meets minimum threshold
                if relevance_score >= self.min_relevance_threshold:
                    matched_topics.append(topic.name)
        
        # Determine if article passes filter
        passed_filter = len(matched_topics) > 0
        filter_reason = None
        
        if not passed_filter:
            if exclusion_reasons:
                filter_reason = exclusion_reasons[0]
            elif not relevance_scores:
                filter_reason = "No keyword matches found"
            else:
                best_score = max(relevance_scores.values())
                filter_reason = f"Best score {best_score:.3f} below threshold {self.min_relevance_threshold}"
        
        return FilterResult(
            article=article,
            matched_topics=matched_topics,
            relevance_scores=relevance_scores,
            passed_filter=passed_filter,
            filter_reason=filter_reason
        )
    
    def filter_articles(self, articles: List[Article], topics: List[Topic]) -> List[FilterResult]:
        """Filter multiple articles against topics.
        
        Args:
            articles: List of articles to filter
            topics: List of topics to match against
            
        Returns:
            List of FilterResult objects
        """
        if not articles:
            return []
        
        self.logger.info(f"Pre-filtering {len(articles)} articles against {len(topics)} topics")
        
        results = []
        for article in articles:
            result = self.filter_article(article, topics)
            results.append(result)
        
        # Log filtering summary
        passed_count = sum(1 for r in results if r.passed_filter)
        reduction_percent = ((len(articles) - passed_count) / len(articles)) * 100 if articles else 0
        
        self.logger.info(
            f"Pre-filtering complete: {passed_count}/{len(articles)} articles passed "
            f"({reduction_percent:.1f}% reduction)"
        )
        
        # Log topic match distribution
        topic_matches = {}
        for result in results:
            for topic in result.matched_topics:
                topic_matches[topic] = topic_matches.get(topic, 0) + 1
        
        if topic_matches:
            self.logger.info(
                f"Topic matches: {dict(sorted(topic_matches.items(), key=lambda x: x[1], reverse=True))}"
            )
        
        return results
    
    def get_filtered_articles(self, articles: List[Article], topics: List[Topic]) -> List[Tuple[Article, str, float]]:
        """Get articles that pass pre-filtering with their best topic match.
        
        Args:
            articles: List of articles to filter  
            topics: List of topics to match against
            
        Returns:
            List of (article, topic_name, relevance_score) tuples for articles that passed
        """
        results = self.filter_articles(articles, topics)
        
        filtered_articles = []
        for result in results:
            if result.passed_filter:
                filtered_articles.append((
                    result.article,
                    result.best_match_topic,
                    result.best_match_score
                ))
        
        return filtered_articles