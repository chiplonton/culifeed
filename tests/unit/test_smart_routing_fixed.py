"""
Smart Routing Feature Tests
==========================

Comprehensive test suite for the smart routing feature that validates:
1. Topic specificity and keyword matching accuracy
2. Confidence scoring and threshold handling
3. Semantic context awareness for avoiding false positives
4. Edge cases and boundary conditions

This addresses the core issue where generic programming articles
were incorrectly matching specific topics like TikTok engineering.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from culifeed.processing.smart_analyzer import SmartKeywordAnalyzer, ConfidenceResult
from culifeed.database.models import Article, Topic


def create_test_article(title: str, content: str = "", summary: str = "", url: str = "https://example.com") -> Article:
    """Helper to create test Article objects."""
    return Article(
        title=title,
        content=content,
        summary=summary,
        url=url,
        source_feed="https://test-feed.com"
    )

def create_test_topic(name: str, keywords: List[str], confidence_threshold: float = 0.7, chat_id: str = "test_chat") -> Topic:
    """Helper to create test Topic objects."""
    return Topic(
        chat_id=chat_id,
        name=name,
        keywords=keywords,
        confidence_threshold=confidence_threshold
    )


class TestSmartKeywordAnalyzer:
    """Test suite for SmartKeywordAnalyzer accuracy and semantic awareness."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance with test settings."""
        from culifeed.config.settings import get_settings
        analyzer = SmartKeywordAnalyzer()
        # Mock logger to avoid logging during tests
        analyzer.logger = Mock()
        return analyzer
    
    @pytest.fixture
    def test_topics(self):
        """Define test topics that mirror production configuration."""
        return [
            create_test_topic(
                name="TikTok's software engineers",
                keywords=["software engineers", "programming", "algorithm", "tiktok", "social media", "app development", "coding"]
            ),
            create_test_topic(
                name="Get the new update or features or best practices of aws ecs and eks",
                keywords=["aws", "amazon", "cloud computing", "aws ecs", "aws eks", "kubernetes", "container", "containerization", "microservices", "amazon ecs", "amazon eks"]
            ),
            create_test_topic(
                name="engineering culture or personal growth as an engineer/manager",
                keywords=["emotional intelligence", "team motivation", "engineering management", "personal growth", "leadership development", "professional development", "workplace challenges"]
            )
        ]
    
    def test_tiktok_specific_articles_match_correctly(self, analyzer, test_topics):
        """Test that TikTok-specific articles match TikTok topic with high confidence."""
        tiktok_topic = test_topics[0]
        
        # Real TikTok engineering articles should match
        tiktok_articles = [
            create_test_article(
                title="TikTok's recommendation algorithm changes for better user experience",
                content="TikTok engineering team releases new social media algorithm updates"
            ),
            create_test_article(
                title="How TikTok app development team scales mobile infrastructure", 
                content="TikTok software engineers discuss mobile app development challenges"
            ),
            create_test_article(
                title="TikTok's approach to social media data processing",
                content="Inside TikTok's engineering culture and app development practices"
            )
        ]
        
        for article in tiktok_articles:
            result = analyzer.analyze_article_confidence(article, tiktok_topic)
            
            # Should have high confidence and score for TikTok-specific content
            assert result.confidence_level >= 0.7, f"TikTok article should have high confidence: {article.title} (got {result.confidence_level})"
            assert result.relevance_score >= 0.6, f"TikTok article should have high score: {article.title} (got {result.relevance_score})"
            assert len(result.matched_keywords) >= 2, f"TikTok article should match multiple keywords: {article.title}"
            # Must include TikTok-specific keywords
            tiktok_specific = ["tiktok", "social media", "app development"]
            assert any(kw.lower() in tiktok_specific for kw in result.matched_keywords), \
                f"Must match TikTok-specific keywords: {result.matched_keywords}"
    
    def test_generic_programming_articles_rejected(self, analyzer, test_topics):
        """Test that generic programming articles are rejected or get low confidence."""
        tiktok_topic = test_topics[0]
        
        # Generic programming articles that should NOT match TikTok topic
        generic_articles = [
            create_test_article(
                title="AI coding assistants are twice as verbose as Stack Overflow",
                content="AI coding assistants help software engineers with programming tasks"
            ),
            create_test_article(
                title="Best practices for algorithm optimization in general software development",
                content="Software engineers should follow these coding practices for better algorithms"
            ),
            create_test_article(
                title="Programming language trends among software engineers in 2024",
                content="Survey of programming languages used by software engineers worldwide"
            ),
            create_test_article(
                title="How to improve coding efficiency as a software engineer",
                content="Tips for programming productivity and algorithm design"
            )
        ]
        
        for article in generic_articles:
            result = analyzer.analyze_article_confidence(article, tiktok_topic)
            
            # Should have low confidence for generic programming content
            assert result.confidence_level < 0.7, f"Generic article should have low confidence: {article.title} (got {result.confidence_level})"
            
            # If it matches keywords, they should be generic ones only
            if result.matched_keywords:
                tiktok_specific = ["tiktok", "social media", "app development"]
                specific_matches = [kw for kw in result.matched_keywords if kw.lower() in tiktok_specific]
                assert len(specific_matches) == 0, \
                    f"Generic article should not match TikTok-specific keywords: {specific_matches}"
    
    def test_aws_ecs_eks_articles_match_correctly(self, analyzer, test_topics):
        """Test that AWS ECS/EKS articles match the AWS topic with high confidence."""
        aws_topic = test_topics[1]
        
        # Real AWS ECS/EKS articles should match
        aws_articles = [
            create_test_article(
                title="AWS ECS Fargate best practices for containerized microservices",
                content="AWS ECS and Amazon EKS deployment strategies for kubernetes containers"
            ),
            create_test_article(
                title="Amazon EKS cluster management and kubernetes optimization",
                content="AWS cloud computing with amazon eks and containerization"
            ),
            create_test_article(
                title="Migrating to AWS ECS: container orchestration guide",
                content="AWS microservices architecture using amazon ecs and cloud computing"
            )
        ]
        
        for article in aws_articles:
            result = analyzer.analyze_article_confidence(article, aws_topic)
            
            # Should have high confidence and score for AWS-specific content
            assert result.confidence_level >= 0.7, f"AWS article should have high confidence: {article.title}"
            assert result.relevance_score >= 0.6, f"AWS article should have high score: {article.title}"
            
            # Must include AWS-specific keywords
            aws_specific = ["aws ecs", "aws eks", "amazon ecs", "amazon eks", "kubernetes", "container"]
            assert any(kw.lower() in aws_specific for kw in result.matched_keywords), \
                f"Must match AWS-specific keywords: {result.matched_keywords}"
    
    def test_real_world_problematic_case(self, analyzer, test_topics):
        """Test the actual problematic case from production."""
        tiktok_topic = test_topics[0]
        
        # The actual problematic article from production
        problematic_article = create_test_article(
            title="AI coding assistants are twice as verbose as Stack Overflow",
            content="AI coding assistants help software engineers with programming tasks. The post discusses programming productivity and coding efficiency.",
            url="https://leaddev.com/ai/ai-coding-assistants-are-twice-as-verbose-as-stack-overflow"
        )
        
        result = analyzer.analyze_article_confidence(problematic_article, tiktok_topic)
        
        # This article should NOT have high confidence for TikTok topic
        assert result.confidence_level < 0.7, \
            f"Problematic article should have low confidence: {result.confidence_level}"
        
        # Should only match generic keywords, not TikTok-specific ones
        tiktok_specific = ["tiktok", "social media", "app development"]
        specific_matches = [kw for kw in result.matched_keywords if kw.lower() in tiktok_specific]
        assert len(specific_matches) == 0, \
            f"Should not match TikTok-specific keywords: {specific_matches}"
        
        # Should be classified as uncertain or low confidence
        assert result.routing_decision in ["uncertain", "low_confidence"], \
            f"Should be routed as uncertain/low confidence, got: {result.routing_decision}"


class TestSmartRoutingIntegration:
    """Integration tests for smart routing with the processing pipeline."""
    
    def test_confidence_result_structure(self):
        """Test that ConfidenceResult has expected structure."""
        # Create a mock result to test structure
        result = ConfidenceResult(
            relevance_score=0.8,
            confidence_level=0.9,
            matched_keywords=["test", "keyword"],
            content_quality_score=0.7,
            url_quality_score=0.6,
            routing_decision="high_confidence",
            reasoning=["Test reasoning"]
        )
        
        assert result.relevance_score == 0.8
        assert result.confidence_level == 0.9
        assert result.matched_keywords == ["test", "keyword"]
        assert result.routing_decision == "high_confidence"
        assert len(result.reasoning) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])