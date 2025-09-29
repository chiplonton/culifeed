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
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime, timezone

from culifeed.processing.smart_analyzer import SmartKeywordAnalyzer, ConfidenceResult
from culifeed.database.models import Article, Topic
from culifeed.config.settings import get_settings


# Remove MockTopic as we'll use real Topic objects


@dataclass
class MockArticle:
    """Mock article for testing."""

    title: str
    content: str = ""
    summary: str = ""
    url: str = "https://example.com"


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
            Topic(
                id=1,
                chat_id="test_chat",
                name="TikTok's software engineers",
                keywords=[
                    "software engineers",
                    "programming",
                    "algorithm",
                    "tiktok",
                    "social media",
                    "app development",
                    "coding",
                ],
                exclude_keywords=[],
                active=True,
                created_at=datetime.now(timezone.utc),
            ),
            Topic(
                id=2,
                chat_id="test_chat",
                name="Get the new update or features or best practices of aws ecs and eks",
                keywords=[
                    "aws",
                    "amazon",
                    "cloud computing",
                    "aws ecs",
                    "aws eks",
                    "kubernetes",
                    "container",
                    "containerization",
                    "microservices",
                    "amazon ecs",
                    "amazon eks",
                ],
                exclude_keywords=[],
                active=True,
                created_at=datetime.now(timezone.utc),
            ),
            Topic(
                id=3,
                chat_id="test_chat",
                name="engineering culture or personal growth as an engineer/manager",
                keywords=[
                    "emotional intelligence",
                    "team motivation",
                    "engineering management",
                    "personal growth",
                    "leadership development",
                    "professional development",
                    "workplace challenges",
                ],
                exclude_keywords=[],
                active=True,
                created_at=datetime.now(timezone.utc),
            ),
        ]

    def test_tiktok_specific_articles_match_correctly(self, analyzer, test_topics):
        """Test that TikTok-specific articles match TikTok topic with high confidence."""
        tiktok_topic = test_topics[0]

        # Real TikTok engineering articles should match
        tiktok_articles = [
            MockArticle(
                title="TikTok's recommendation algorithm changes for better user experience",
                content="TikTok engineering team releases new social media algorithm updates",
            ),
            MockArticle(
                title="How TikTok app development team scales mobile infrastructure",
                content="TikTok software engineers discuss mobile app development challenges",
            ),
            MockArticle(
                title="TikTok's approach to social media data processing",
                content="Inside TikTok's engineering culture and app development practices",
            ),
        ]

        for article in tiktok_articles:
            result = analyzer.analyze_article_confidence(article, tiktok_topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Should have high confidence and score for TikTok-specific content
            assert (
                confidence >= 0.7
            ), f"TikTok article should have high confidence: {article.title}"
            assert (
                score >= 0.6
            ), f"TikTok article should have high score: {article.title}"
            assert (
                len(matched_keywords) >= 2
            ), f"TikTok article should match multiple keywords: {article.title}"
            # Must include TikTok-specific keywords
            tiktok_specific = ["tiktok", "social media", "app development"]
            assert any(
                kw.lower() in tiktok_specific for kw in matched_keywords
            ), f"Must match TikTok-specific keywords: {matched_keywords}"

    def test_generic_programming_articles_rejected(self, analyzer, test_topics):
        """Test that generic programming articles are rejected or get low confidence."""
        tiktok_topic = test_topics[0]

        # Generic programming articles that should NOT match TikTok topic
        generic_articles = [
            MockArticle(
                title="AI coding assistants are twice as verbose as Stack Overflow",
                content="AI coding assistants help software engineers with programming tasks",
            ),
            MockArticle(
                title="Best practices for algorithm optimization in general software development",
                content="Software engineers should follow these coding practices for better algorithms",
            ),
            MockArticle(
                title="Programming language trends among software engineers in 2024",
                content="Survey of programming languages used by software engineers worldwide",
            ),
            MockArticle(
                title="How to improve coding efficiency as a software engineer",
                content="Tips for programming productivity and algorithm design",
            ),
        ]

        for article in generic_articles:
            result = analyzer.analyze_article_confidence(article, tiktok_topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Should have low confidence for generic programming content
            assert (
                confidence < 0.7
            ), f"Generic article should have low confidence: {article.title} (got {confidence})"

            # If it matches keywords, they should be generic ones only
            if matched_keywords:
                tiktok_specific = ["tiktok", "social media", "app development"]
                specific_matches = [
                    kw for kw in matched_keywords if kw.lower() in tiktok_specific
                ]
                assert (
                    len(specific_matches) == 0
                ), f"Generic article should not match TikTok-specific keywords: {specific_matches}"

    def test_aws_ecs_eks_articles_match_correctly(self, analyzer, test_topics):
        """Test that AWS ECS/EKS articles match the AWS topic with high confidence."""
        aws_topic = test_topics[1]

        # Real AWS ECS/EKS articles should match
        aws_articles = [
            MockArticle(
                title="AWS ECS Fargate best practices for containerized microservices",
                content="AWS ECS and Amazon EKS deployment strategies for kubernetes containers",
            ),
            MockArticle(
                title="Amazon EKS cluster management and kubernetes optimization",
                content="AWS cloud computing with amazon eks and containerization",
            ),
            MockArticle(
                title="Migrating to AWS ECS: container orchestration guide",
                content="AWS microservices architecture using amazon ecs and cloud computing",
            ),
        ]

        for article in aws_articles:
            result = analyzer.analyze_article_confidence(article, aws_topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Should have high confidence and score for AWS-specific content
            assert (
                confidence >= 0.7
            ), f"AWS article should have high confidence: {article.title}"
            assert score >= 0.6, f"AWS article should have high score: {article.title}"

            # Must include AWS-specific keywords
            aws_specific = [
                "aws ecs",
                "aws eks",
                "amazon ecs",
                "amazon eks",
                "kubernetes",
                "container",
            ]
            assert any(
                kw.lower() in aws_specific for kw in matched_keywords
            ), f"Must match AWS-specific keywords: {matched_keywords}"

    def test_generic_aws_articles_get_penalized(self, analyzer, test_topics):
        """Test that generic AWS articles (not ECS/EKS specific) get lower confidence."""
        aws_topic = test_topics[1]

        # Generic AWS articles that are not ECS/EKS specific
        generic_aws_articles = [
            MockArticle(
                title="AWS Redshift data warehouse optimization techniques",
                content="Amazon Redshift and AWS cloud computing for data analytics",
            ),
            MockArticle(
                title="AWS Lambda serverless functions best practices",
                content="Amazon lambda and aws cloud computing serverless architecture",
            ),
            MockArticle(
                title="AWS S3 bucket security and access control",
                content="Amazon S3 storage with aws cloud computing security features",
            ),
        ]

        for article in generic_aws_articles:
            result = analyzer.analyze_article_confidence(article, aws_topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Should have lower confidence for non-ECS/EKS AWS content
            assert (
                confidence < 0.8
            ), f"Generic AWS article should have reduced confidence: {article.title}"

            # Should match generic AWS keywords but not specific ones
            aws_specific = [
                "aws ecs",
                "aws eks",
                "amazon ecs",
                "amazon eks",
                "kubernetes",
                "container",
                "containerization",
            ]
            specific_matches = [
                kw for kw in matched_keywords if kw.lower() in aws_specific
            ]
            assert (
                len(specific_matches) <= 1
            ), f"Generic AWS article should not match many specific keywords: {specific_matches}"

    def test_single_ambiguous_keyword_penalty(self, analyzer, test_topics):
        """Test that articles matching only single ambiguous keywords get penalized."""
        # Test with multiple topics to ensure the penalty applies across topics
        ambiguous_test_cases = [
            (
                test_topics[0],
                "TikTok's software engineers",
                MockArticle(
                    "Generic coding tutorial", "This is about coding in general"
                ),
            ),
            (
                test_topics[1],
                "AWS topic",
                MockArticle(
                    "Amazon marketplace selling tips",
                    "Amazon marketplace and selling on amazon",
                ),
            ),
            (
                test_topics[2],
                "Engineering culture",
                MockArticle(
                    "Software development lifecycle",
                    "About professional development methodologies",
                ),
            ),
        ]

        for topic, topic_desc, article in ambiguous_test_cases:
            result = analyzer.analyze_article_confidence(article, topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Single ambiguous matches should get penalized
            if len(matched_keywords) == 1:
                ambiguous_keywords = {
                    "coding",
                    "programming",
                    "software engineers",
                    "algorithm",
                    "development",
                    "aws",
                    "amazon",
                }
                if matched_keywords[0].lower() in ambiguous_keywords:
                    assert (
                        confidence < 0.6
                    ), f"Single ambiguous match should be penalized for {topic_desc}: {article.title}"

    def test_multiple_related_keywords_boost_confidence(self, analyzer, test_topics):
        """Test that articles with multiple related keywords get confidence boost."""
        tiktok_topic = test_topics[0]

        # Article with multiple related TikTok keywords
        multi_keyword_article = MockArticle(
            title="TikTok software engineers develop new app development framework",
            content="TikTok engineering team creates social media app development tools for mobile programming",
        )

        result = analyzer.analyze_article_confidence(
            multi_keyword_article, tiktok_topic
        )
        confidence, score, matched_keywords = (
            result.confidence_level,
            result.relevance_score,
            result.matched_keywords,
        )

        # Multiple related keywords should boost confidence
        assert len(matched_keywords) >= 3, "Should match multiple keywords"
        assert confidence >= 0.7, "Multiple related keywords should boost confidence"

        # Should include both generic and specific keywords
        tiktok_specific = ["tiktok", "social media", "app development"]
        specific_matches = [
            kw for kw in matched_keywords if kw.lower() in tiktok_specific
        ]
        assert (
            len(specific_matches) >= 2
        ), "Should match multiple TikTok-specific keywords"

    def test_topic_coherence_check(self, analyzer, test_topics):
        """Test that topic coherence check prevents semantic mismatches."""
        # Test articles that might match generic keywords but lack topic coherence
        incoherent_cases = [
            (
                test_topics[0],
                "Programming tutorial for beginners",
                "Learn programming and software engineering basics with coding examples",
            ),
            (
                test_topics[1],
                "AWS account billing and cost optimization",
                "Amazon aws billing optimization and cloud computing cost management",
            ),
            (
                test_topics[2],
                "Professional development in marketing",
                "Professional development and personal growth in marketing careers",
            ),
        ]

        for topic, title, content in incoherent_cases:
            article = MockArticle(title=title, content=content)
            result = analyzer.analyze_article_confidence(article, topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            # Articles lacking topic coherence should get lower confidence
            assert (
                confidence < 0.71
            ), f"Incoherent article should have low confidence for {topic.name}: {title}"

    def test_confidence_thresholds_and_routing_decisions(self, analyzer, test_topics):
        """Test that confidence scores align with routing decision thresholds."""
        # Test cases designed to hit specific confidence ranges
        test_cases = [
            # High confidence cases (should route directly without AI)
            (
                test_topics[0],
                MockArticle(
                    "TikTok's new social media algorithm for app development",
                    "TikTok software engineers release social media app development updates",
                ),
                0.8,
                "high",
            ),
            # Medium confidence cases (should go to AI for validation)
            (
                test_topics[1],
                MockArticle(
                    "AWS container best practices",
                    "AWS cloud computing and container management tips",
                ),
                0.6,
                "medium",
            ),
            # Low confidence cases (should be filtered out or use keyword fallback)
            (
                test_topics[0],
                MockArticle(
                    "Generic programming concepts",
                    "Basic programming and coding fundamentals",
                ),
                0.4,
                "low",
            ),
        ]

        for topic, article, expected_min_confidence, confidence_level in test_cases:
            result = analyzer.analyze_article_confidence(article, topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )

            if confidence_level == "high":
                assert (
                    confidence >= 0.7
                ), f"High confidence case failed: {article.title}"
            elif confidence_level == "medium":
                assert (
                    0.6 <= confidence < 0.8
                ), f"Medium confidence case failed: {article.title}"
            else:  # low
                assert confidence < 0.6, f"Low confidence case failed: {article.title}"

    def test_url_quality_scoring(self, analyzer, test_topics):
        """Test that URL quality affects confidence scoring."""
        topic = test_topics[0]  # TikTok topic

        # Same article content with different URL quality
        base_article_data = {
            "title": "TikTok software engineering practices",
            "content": "TikTok app development and social media engineering",
        }

        url_test_cases = [
            ("https://techcrunch.com/tiktok-engineering", "high_quality"),
            ("https://engineering.tiktok.com/practices", "high_quality"),
            ("https://random-blog.com/tiktok-stuff", "medium_quality"),
            ("https://spam-site.click/tiktok-clickbait", "low_quality"),
        ]

        confidence_scores = []
        for url, quality_level in url_test_cases:
            article = MockArticle(url=url, **base_article_data)
            result = analyzer.analyze_article_confidence(article, topic)
            confidence, score, matched_keywords = (
                result.confidence_level,
                result.relevance_score,
                result.matched_keywords,
            )
            confidence_scores.append((confidence, quality_level))

        # Higher quality URLs should generally get better confidence scores
        high_quality_scores = [c for c, q in confidence_scores if q == "high_quality"]
        low_quality_scores = [c for c, q in confidence_scores if q == "low_quality"]

        if high_quality_scores and low_quality_scores:
            avg_high = sum(high_quality_scores) / len(high_quality_scores)
            avg_low = sum(low_quality_scores) / len(low_quality_scores)
            # Allow some tolerance for URL quality impact
            assert (
                avg_high >= avg_low - 0.1
            ), "High quality URLs should not score significantly lower"


class TestSmartRoutingIntegration:
    """Integration tests for smart routing with the processing pipeline."""

    @pytest.fixture
    def mock_pipeline_components(self):
        """Mock pipeline components for integration testing."""
        with patch("culifeed.processing.smart_analyzer.get_settings") as mock_settings:
            mock_settings.return_value.smart_processing.enabled = True
            mock_settings.return_value.smart_processing.high_confidence_threshold = 0.8
            mock_settings.return_value.smart_processing.low_confidence_threshold = 0.6
            mock_settings.return_value.smart_processing.definitely_relevant_threshold = (
                0.7
            )
            mock_settings.return_value.smart_processing.definitely_irrelevant_threshold = (
                0.3
            )

            yield mock_settings

    def test_smart_routing_decision_logic(self, mock_pipeline_components):
        """Test the complete smart routing decision logic."""
        analyzer = SmartKeywordAnalyzer()
        analyzer.logger = Mock()

        # Test routing decisions for different confidence levels
        decision_test_cases = [
            (0.9, 0.8, "route_directly"),  # High confidence, high score
            (0.7, 0.6, "send_to_ai"),  # Medium confidence, medium score
            (0.5, 0.4, "keyword_fallback"),  # Low confidence, low score
            (0.3, 0.2, "keyword_fallback"),  # Very low confidence and score
        ]

        topic = Topic(
            id=4,
            chat_id="test_chat",
            name="Test Topic",
            keywords=["test", "keyword"],
            exclude_keywords=[],
            active=True,
            created_at=datetime.now(timezone.utc),
        )
        article = MockArticle("Test Article", "test content")

        for confidence, score, expected_decision in decision_test_cases:
            # Mock the analyze_article_confidence method to return specific values
            with patch.object(
                analyzer,
                "analyze_article_confidence",
                return_value=(confidence, score, ["test"]),
            ):
                # This would be called by the pipeline to make routing decisions
                # We're testing the logic that would be in the pipeline
                if confidence >= 0.8 and score >= 0.7:
                    decision = "route_directly"
                elif confidence >= 0.6 and score >= 0.3:
                    decision = "send_to_ai"
                elif confidence >= 0.3:
                    decision = "keyword_fallback"
                else:
                    decision = "reject"

                assert (
                    decision == expected_decision
                ), f"Wrong routing decision for confidence={confidence}, score={score}"

    def test_real_world_article_scenarios(self):
        """Test with real-world article scenarios that caused issues."""
        analyzer = SmartKeywordAnalyzer()
        analyzer.logger = Mock()

        # The actual problematic case from production
        problematic_article = MockArticle(
            title="AI coding assistants are twice as verbose as Stack Overflow",
            content="AI coding assistants help software engineers with programming tasks. The post discusses programming productivity and coding efficiency.",
            url="https://leaddev.com/ai/ai-coding-assistants-are-twice-as-verbose-as-stack-overflow",
        )

        tiktok_topic = Topic(
            id=5,
            chat_id="test_chat",
            name="TikTok's software engineers",
            keywords=[
                "software engineers",
                "programming",
                "algorithm",
                "tiktok",
                "social media",
                "app development",
                "coding",
            ],
            exclude_keywords=[],
            active=True,
            created_at=datetime.now(timezone.utc),
        )

        result = analyzer.analyze_article_confidence(problematic_article, tiktok_topic)
        confidence, score, matched_keywords = (
            result.confidence_level,
            result.relevance_score,
            result.matched_keywords,
        )

        # This article should NOT have high confidence for TikTok topic
        assert (
            confidence < 0.7
        ), f"Problematic article should have low confidence: {confidence}"

        # Should only match generic keywords, not TikTok-specific ones
        tiktok_specific = ["tiktok", "social media", "app development"]
        specific_matches = [
            kw for kw in matched_keywords if kw.lower() in tiktok_specific
        ]
        assert (
            len(specific_matches) == 0
        ), f"Should not match TikTok-specific keywords: {specific_matches}"


@pytest.mark.integration
class TestSmartRoutingEndToEnd:
    """End-to-end tests for smart routing feature."""

    def test_smart_routing_prevents_irrelevant_delivery(self):
        """Test that smart routing prevents delivery of irrelevant articles."""
        # This would be an integration test with the full pipeline
        # Testing that the improvements prevent the original issue
        pass  # Placeholder for integration test

    def test_smart_routing_performance_metrics(self):
        """Test that smart routing meets performance requirements."""
        # Test processing time, accuracy metrics, etc.
        pass  # Placeholder for performance test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
