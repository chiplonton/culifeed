#!/usr/bin/env python3
"""
Trust Features Test Suite
=========================

Comprehensive tests for Phase 1 trust improvements:
- Quality monitoring system
- Cross-validation between AI and pre-filter scores
- Provider transparency formatting
- Hybrid fallback mechanisms
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from culifeed.monitoring.quality_monitor import QualityMonitor, QualityMetrics, QualityAlert, AlertLevel
from culifeed.monitoring.trust_validator import TrustValidator, ValidationResult, ValidationOutcome
from culifeed.delivery.transparency_formatter import TransparencyFormatter, TransparencyInfo
from culifeed.database.models import Article, Topic
from culifeed.ai.providers.base import AIResult


class TestQualityMonitor(unittest.TestCase):
    """Test quality monitoring system."""
    
    def setUp(self):
        self.monitor = QualityMonitor()
    
    def test_validation_metrics_tracking(self):
        """Test validation attempt tracking."""
        # Record successful validation
        self.monitor.record_validation_attempt(
            ai_score=0.8, prefilter_score=0.75, 
            provider="groq", success=True
        )
        
        # Record failed validation
        self.monitor.record_validation_attempt(
            ai_score=0.3, prefilter_score=0.9, 
            provider="groq", success=False, reason="Large score difference"
        )
        
        metrics = self.monitor.get_current_metrics()
        
        self.assertEqual(metrics.validation_attempts, 2)
        self.assertEqual(metrics.validation_passes, 1)
        self.assertEqual(metrics.validation_failures, 1)
        self.assertEqual(metrics.validation_success_rate, 0.5)
    
    def test_processing_metrics_tracking(self):
        """Test processing attempt tracking."""
        # Successful processing
        self.monitor.record_processing_attempt(
            article_id="art1", provider="groq", success=True,
            processing_time_ms=1500.0, used_fallback=False
        )
        
        # Failed processing with fallback
        self.monitor.record_processing_attempt(
            article_id="art2", provider="groq", success=False,
            processing_time_ms=3000.0, used_fallback=True
        )
        
        metrics = self.monitor.get_current_metrics()
        
        self.assertEqual(metrics.ai_processing_success_rate, 0.5)
        self.assertEqual(metrics.keyword_fallback_rate, 0.5)
        self.assertEqual(metrics.avg_processing_time_ms, 2250.0)
    
    def test_quality_alerts(self):
        """Test quality alert generation."""
        # Trigger score difference alert
        self.monitor.record_validation_attempt(
            ai_score=0.2, prefilter_score=0.8, 
            provider="groq", success=False
        )
        
        alerts = self.monitor.get_recent_alerts()
        self.assertTrue(len(alerts) > 0)
        
        alert = alerts[0]
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertIn("Large score difference", alert.message)
        self.assertEqual(alert.component, "cross_validation")


class TestTrustValidator(unittest.TestCase):
    """Test cross-validation system."""
    
    def setUp(self):
        self.validator = TrustValidator()
        self.article = Article(
            title="Test Article",
            url="https://example.com/test",
            content="AWS Lambda serverless functions are great for cloud computing",
            source_feed="https://example.com/feed.xml"
        )
        self.topic = Topic(
            chat_id="test_chat",
            name="Cloud Computing",
            keywords=["aws", "lambda", "serverless", "cloud"]
        )
    
    def test_validation_pass(self):
        """Test validation passes with similar scores."""
        ai_result = AIResult(
            success=True,
            relevance_score=0.8,
            confidence=0.9,
            provider="groq"
        )
        
        result = self.validator.validate_ai_result(
            ai_result=ai_result,
            prefilter_score=0.75,
            article=self.article,
            topic=self.topic
        )
        
        self.assertEqual(result.outcome, ValidationOutcome.PASS)
        self.assertEqual(result.reason, "Scores are consistent")
        self.assertGreaterEqual(result.adjusted_confidence, 0.8)
    
    def test_validation_warning(self):
        """Test validation warning with moderate difference."""
        ai_result = AIResult(
            success=True,
            relevance_score=0.6,
            confidence=0.8,
            provider="groq"
        )
        
        result = self.validator.validate_ai_result(
            ai_result=ai_result,
            prefilter_score=0.3,
            article=self.article,
            topic=self.topic
        )
        
        self.assertEqual(result.outcome, ValidationOutcome.WARNING)
        self.assertIn("Moderate score difference", result.reason)
        self.assertLess(result.adjusted_confidence, ai_result.confidence)
    
    def test_validation_fail(self):
        """Test validation fails with large difference."""
        ai_result = AIResult(
            success=True,
            relevance_score=0.9,
            confidence=0.9,
            provider="groq"
        )
        
        result = self.validator.validate_ai_result(
            ai_result=ai_result,
            prefilter_score=0.2,
            article=self.article,
            topic=self.topic
        )
        
        self.assertEqual(result.outcome, ValidationOutcome.FAIL)
        self.assertIn("Score difference too large", result.reason)
        self.assertLess(result.adjusted_confidence, 0.5)
    
    def test_provider_quality_adjustment(self):
        """Test provider quality affects confidence."""
        ai_result = AIResult(
            success=True,
            relevance_score=0.8,
            confidence=0.8,
            provider="huggingface"  # Lower quality provider
        )
        
        result = self.validator.validate_ai_result(
            ai_result=ai_result,
            prefilter_score=0.8,
            article=self.article,
            topic=self.topic
        )
        
        # Confidence should be adjusted down for lower quality provider
        self.assertLess(result.adjusted_confidence, ai_result.confidence)


class TestTransparencyFormatter(unittest.TestCase):
    """Test transparency formatting system."""
    
    def setUp(self):
        self.formatter = TransparencyFormatter()
        self.article = Article(
            title="AWS Lambda Performance Tips",
            url="https://example.com/lambda-tips",
            content="Learn how to optimize AWS Lambda functions for better performance",
            source_feed="https://example.com/feed.xml",
            ai_provider="groq",
            ai_confidence=0.85,
            ai_relevance_score=0.8,
            validation_outcome="pass"
        )
    
    def test_transparency_info_extraction(self):
        """Test extracting transparency information."""
        info = self.formatter.get_transparency_info(self.article)
        
        self.assertEqual(info.provider_name, "Groq")
        self.assertEqual(info.confidence, 0.85)
        self.assertEqual(info.validation_status, "pass")
        self.assertEqual(info.processing_method, "AI Analysis (Validated)")
    
    def test_article_formatting_with_transparency(self):
        """Test article formatting with transparency info."""
        formatted = self.formatter.format_article_with_transparency(
            self.article, include_transparency=True
        )
        
        self.assertIn("AWS Lambda Performance Tips", formatted)
        self.assertIn("🚀 Groq", formatted)
        self.assertIn("85%", formatted)
    
    def test_article_formatting_without_transparency(self):
        """Test article formatting without transparency info."""
        formatted = self.formatter.format_article_with_transparency(
            self.article, include_transparency=False
        )
        
        self.assertIn("AWS Lambda Performance Tips", formatted)
        self.assertNotIn("🚀 Groq", formatted)
    
    def test_fallback_transparency(self):
        """Test transparency for fallback processing."""
        fallback_article = Article(
            title="Cloud Computing Basics",
            url="https://example.com/cloud-basics",
            content="Introduction to cloud computing concepts",
            source_feed="https://example.com/feed.xml",
            ai_provider="keyword_backup",
            validation_outcome="fallback"
        )
        
        info = self.formatter.get_transparency_info(fallback_article)
        
        self.assertEqual(info.provider_name, "Keywords")
        self.assertEqual(info.validation_status, "fallback")
        self.assertEqual(info.processing_method, "Keyword Analysis")
        
        formatted = self.formatter.format_article_with_transparency(fallback_article)
        self.assertIn("📊 Keywords", formatted)


class TestTrustIntegration(unittest.TestCase):
    """Test integration between trust components."""
    
    def setUp(self):
        self.quality_monitor = QualityMonitor()
        self.trust_validator = TrustValidator()
        self.transparency_formatter = TransparencyFormatter()
        
        self.article = Article(
            title="Serverless Architecture Guide",
            url="https://example.com/serverless-guide",
            content="Complete guide to serverless architecture with AWS Lambda",
            source_feed="https://example.com/feed.xml"
        )
        
        self.topic = Topic(
            chat_id="test_chat",
            name="Serverless",
            keywords=["serverless", "lambda", "aws"]
        )
    
    def test_end_to_end_validation_flow(self):
        """Test complete validation and monitoring flow."""
        # Simulate AI processing result
        ai_result = AIResult(
            success=True,
            relevance_score=0.85,
            confidence=0.9,
            provider="groq",
            reasoning="Article discusses serverless and AWS Lambda extensively"
        )
        
        prefilter_score = 0.8
        
        # 1. Validate AI result
        validation = self.trust_validator.validate_ai_result(
            ai_result, prefilter_score, self.article, self.topic
        )
        
        # 2. Record validation attempt
        self.quality_monitor.record_validation_attempt(
            ai_score=ai_result.relevance_score,
            prefilter_score=prefilter_score,
            provider=ai_result.provider,
            success=(validation.outcome == ValidationOutcome.PASS)
        )
        
        # 3. Record processing attempt
        self.quality_monitor.record_processing_attempt(
            article_id=self.article.id,
            provider=ai_result.provider,
            success=ai_result.success,
            processing_time_ms=1500.0,
            ai_result=ai_result,
            used_fallback=False
        )
        
        # 4. Update article with validation results
        self.article.ai_provider = ai_result.provider
        self.article.ai_confidence = validation.adjusted_confidence
        self.article.ai_relevance_score = ai_result.relevance_score
        self.article.validation_outcome = validation.outcome.value
        self.article.validation_reason = validation.reason
        self.article.prefilter_score = prefilter_score
        
        # 5. Format with transparency
        formatted = self.transparency_formatter.format_article_with_transparency(
            self.article
        )
        
        # Verify end-to-end flow
        self.assertEqual(validation.outcome, ValidationOutcome.PASS)
        self.assertGreaterEqual(validation.adjusted_confidence, 0.8)
        
        metrics = self.quality_monitor.get_current_metrics()
        self.assertEqual(metrics.validation_success_rate, 1.0)
        self.assertEqual(metrics.ai_processing_success_rate, 1.0)
        
        self.assertIn("Serverless Architecture Guide", formatted)
        self.assertIn("🚀 Groq", formatted)
        self.assertIn("90%", formatted)


def main():
    """Run all trust feature tests."""
    print("🔍 CuliFeed Trust Features Test Suite")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestQualityMonitor,
        TestTrustValidator, 
        TestTransparencyFormatter,
        TestTrustIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All trust feature tests PASSED")
        return True
    else:
        print("❌ Some trust feature tests FAILED")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)