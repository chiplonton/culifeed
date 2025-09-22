"""
Simple Smart Routing Test
========================

Basic test to validate smart routing behavior without complex mocking.
Tests the real-world problematic case that was happening in production.
"""

import pytest
from unittest.mock import Mock

from culifeed.processing.smart_analyzer import SmartKeywordAnalyzer
from culifeed.database.models import Article, Topic


def create_test_article(title: str, content: str = "", url: str = "https://example.com") -> Article:
    """Helper to create test Article objects."""
    return Article(
        title=title,
        content=content,
        url=url,
        source_feed="https://test-feed.com"
    )

def create_test_topic(name: str, keywords: list, chat_id: str = "test_chat") -> Topic:
    """Helper to create test Topic objects."""
    return Topic(
        chat_id=chat_id,
        name=name,
        keywords=keywords,
        confidence_threshold=0.7
    )


class TestSmartRoutingRealWorld:
    """Test smart routing with real-world scenarios."""
    
    def test_problematic_article_should_be_rejected(self):
        """Test that the actual problematic article gets properly handled."""
        # Create analyzer with real settings
        analyzer = SmartKeywordAnalyzer()
        analyzer.logger = Mock()  # Mock logger to avoid logging during tests
        
        # Create the TikTok topic exactly as it exists in production
        tiktok_topic = create_test_topic(
            name="TikTok's software engineers",
            keywords=["software engineers", "programming", "algorithm", "tiktok", "social media", "app development", "coding"]
        )
        
        # The actual problematic article from production
        problematic_article = create_test_article(
            title="AI coding assistants are twice as verbose as Stack Overflow",
            content="AI coding assistants help software engineers with programming tasks. The post discusses programming productivity and coding efficiency.",
            url="https://leaddev.com/ai/ai-coding-assistants-are-twice-as-verbose-as-stack-overflow"
        )
        
        # Analyze with smart routing
        result = analyzer.analyze_article_confidence(problematic_article, tiktok_topic)
        
        print(f"\nProblematic Article Analysis:")
        print(f"Title: {problematic_article.title}")
        print(f"Topic: {tiktok_topic.name}")
        print(f"Relevance Score: {result.relevance_score:.3f}")
        print(f"Confidence Level: {result.confidence_level:.3f}")
        print(f"Matched Keywords: {result.matched_keywords}")
        print(f"Routing Decision: {result.routing_decision}")
        print(f"Reasoning:")
        for reason in result.reasoning:
            print(f"  - {reason}")
        
        # This article should NOT have high confidence for TikTok topic
        # The semantic improvements should prevent this mismatch
        assert result.confidence_level < 0.8, \
            f"Problematic article should have low confidence: {result.confidence_level}"
        
        # Should only match generic keywords, not TikTok-specific ones
        tiktok_specific = ["tiktok", "social media", "app development"]
        specific_matches = [kw for kw in result.matched_keywords if kw.lower() in tiktok_specific]
        assert len(specific_matches) == 0, \
            f"Should not match TikTok-specific keywords: {specific_matches}"
    
    def test_genuine_tiktok_article_should_match(self):
        """Test that a genuine TikTok article gets high confidence."""
        analyzer = SmartKeywordAnalyzer()
        analyzer.logger = Mock()
        
        tiktok_topic = create_test_topic(
            name="TikTok's software engineers",
            keywords=["software engineers", "programming", "algorithm", "tiktok", "social media", "app development", "coding"]
        )
        
        # A genuine TikTok engineering article
        genuine_article = create_test_article(
            title="TikTok's new recommendation algorithm improves social media engagement",
            content="TikTok engineering team releases major updates to their social media app development platform, focusing on algorithm improvements for better user experience.",
            url="https://engineering.tiktok.com/algorithm-updates"
        )
        
        result = analyzer.analyze_article_confidence(genuine_article, tiktok_topic)
        
        print(f"\nGenuine TikTok Article Analysis:")
        print(f"Title: {genuine_article.title}")
        print(f"Topic: {tiktok_topic.name}")
        print(f"Relevance Score: {result.relevance_score:.3f}")
        print(f"Confidence Level: {result.confidence_level:.3f}")
        print(f"Matched Keywords: {result.matched_keywords}")
        print(f"Routing Decision: {result.routing_decision}")
        
        # This should have high confidence and match TikTok-specific keywords
        tiktok_specific = ["tiktok", "social media", "app development", "algorithm"]
        specific_matches = [kw for kw in result.matched_keywords if kw.lower() in tiktok_specific]
        assert len(specific_matches) >= 2, \
            f"Should match multiple TikTok-specific keywords: {specific_matches}"
        
        # Should have reasonable confidence (though exact threshold may vary)
        assert result.confidence_level > 0.5, \
            f"Genuine TikTok article should have decent confidence: {result.confidence_level}"
    
    def test_aws_ecs_article_specificity(self):
        """Test AWS ECS/EKS topic specificity."""
        analyzer = SmartKeywordAnalyzer()
        analyzer.logger = Mock()
        
        aws_topic = create_test_topic(
            name="Get the new update or features or best practices of aws ecs and eks",
            keywords=["aws", "amazon", "cloud computing", "aws ecs", "aws eks", "kubernetes", "container", "containerization", "microservices", "amazon ecs", "amazon eks"]
        )
        
        # Specific AWS ECS article - should match well
        specific_article = create_test_article(
            title="AWS ECS Fargate best practices for container orchestration",
            content="AWS ECS and Amazon EKS deployment strategies for kubernetes microservices and containerization",
            url="https://aws.amazon.com/blogs/containers/ecs-fargate-best-practices"
        )
        
        # Generic AWS article - should have lower confidence
        generic_article = create_test_article(
            title="AWS S3 bucket optimization for data storage",
            content="Amazon S3 storage optimization with AWS cloud computing features",
            url="https://aws.amazon.com/blogs/storage/s3-optimization"
        )
        
        specific_result = analyzer.analyze_article_confidence(specific_article, aws_topic)
        generic_result = analyzer.analyze_article_confidence(generic_article, aws_topic)
        
        print(f"\nAWS Specificity Test:")
        print(f"Specific ECS Article - Confidence: {specific_result.confidence_level:.3f}, Score: {specific_result.relevance_score:.3f}")
        print(f"Generic AWS Article - Confidence: {generic_result.confidence_level:.3f}, Score: {generic_result.relevance_score:.3f}")
        
        # Specific ECS article should have higher confidence than generic AWS
        assert specific_result.confidence_level >= generic_result.confidence_level, \
            f"ECS-specific article should have higher confidence than generic AWS article"
        
        # Check that ECS article matches container-related keywords
        container_keywords = ["aws ecs", "amazon ecs", "kubernetes", "container", "containerization"]
        ecs_matches = [kw for kw in specific_result.matched_keywords if kw.lower() in container_keywords]
        assert len(ecs_matches) >= 1, \
            f"ECS article should match container-related keywords: {ecs_matches}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print output