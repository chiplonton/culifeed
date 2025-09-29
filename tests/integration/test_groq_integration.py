#!/usr/bin/env python3
"""
Test Groq AI Integration
========================

Test script to verify Groq API integration with the CuliFeed system.
This script tests provider initialization, connection, relevance analysis, and summarization.
"""

import os
import sys
import asyncio
import pytest
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from culifeed.ai.ai_manager import AIManager
from culifeed.ai.providers.groq_provider import GroqProvider
from culifeed.database.models import Article, Topic
from culifeed.config.settings import get_settings
from culifeed.utils.logging import configure_application_logging

# Test data
SAMPLE_ARTICLE = Article(
    id="test_article_1",
    title="AWS Announces New Serverless Container Service for Edge Computing",
    content="""
    Amazon Web Services announced today a new serverless container service designed for edge computing scenarios. 
    The service, called AWS Lambda Edge Containers, allows developers to deploy containerized applications 
    that automatically scale based on demand at edge locations worldwide.
    
    This new offering combines the benefits of serverless computing with the flexibility of containers,
    enabling developers to run applications closer to users for reduced latency. The service supports
    popular container runtimes and integrates with existing AWS services like CloudWatch and X-Ray.
    
    "Edge computing is becoming increasingly important for delivering low-latency applications," said
    the AWS product manager. "This new service makes it easier for developers to deploy containerized
    applications at scale without managing infrastructure."
    
    The service is available in preview in select regions and will be generally available next quarter.
    Pricing follows a pay-per-request model similar to AWS Lambda, with additional charges for container
    registry storage and data transfer.
    """,
    url="https://example.com/aws-edge-containers",
    published_at=datetime.now(timezone.utc),
    source_feed="AWS News",
    content_hash="sample_hash_123",
    created_at=datetime.now(timezone.utc),
)

SAMPLE_TOPIC = Topic(
    id=1,
    chat_id="test_chat",
    name="Cloud Computing & AWS",
    keywords=[
        "AWS",
        "serverless",
        "cloud computing",
        "containers",
        "edge computing",
        "lambda",
    ],
    exclude_keywords=["pricing", "marketing"],
    active=True,
    created_at=datetime.now(timezone.utc),
)

IRRELEVANT_ARTICLE = Article(
    id="test_article_2",
    title="Local Restaurant Opens New Location Downtown",
    content="""
    Mario's Pizza announced the opening of their third location in downtown Springfield yesterday.
    The new restaurant features an expanded menu with gluten-free options and a full bar.
    
    Owner Mario Gonzalez said he's excited to serve the downtown community and expects to hire
    15 additional staff members over the next month. The restaurant will be open seven days
    a week from 11 AM to 11 PM.
    
    The new location includes outdoor seating for 30 customers and features live music on
    weekend evenings. Grand opening specials will run through the end of the month.
    """,
    url="https://example.com/marios-pizza-opens",
    published_at=datetime.now(timezone.utc),
    source_feed="Local News",
    content_hash="sample_hash_456",
    created_at=datetime.now(timezone.utc),
)


@pytest.mark.asyncio
async def test_groq_connection():
    """Test basic Groq API connection."""
    print("🔌 Testing Groq API Connection...")

    try:
        settings = get_settings()

        if not settings.ai.groq_api_key:
            print("❌ No Groq API key found in configuration")
            print("   Set CULIFEED_AI__GROQ_API_KEY environment variable")
            return False

        provider = GroqProvider(
            api_key=settings.ai.groq_api_key, model_name=settings.ai.groq_model
        )

        success = await provider.test_connection()

        if success:
            print("✅ Groq connection successful")
            return True
        else:
            print("❌ Groq connection failed")
            return False

    except Exception as e:
        print(f"❌ Groq connection error: {e}")
        return False


@pytest.mark.asyncio
async def test_relevance_analysis():
    """Test article relevance analysis."""
    print("\n🤔 Testing Relevance Analysis...")

    try:
        settings = get_settings()
        provider = GroqProvider(
            api_key=settings.ai.groq_api_key, model_name=settings.ai.groq_model
        )

        # Test relevant article
        print("  📰 Testing relevant article (AWS/serverless)...")
        result1 = await provider.analyze_relevance(SAMPLE_ARTICLE, SAMPLE_TOPIC)

        if result1.success:
            print(f"     ✅ Relevance Score: {result1.relevance_score:.3f}")
            print(f"     ✅ Confidence: {result1.confidence:.3f}")
            print(f"     ✅ Matched Keywords: {result1.matched_keywords}")
            print(f"     ✅ Reasoning: {result1.reasoning}")
            print(f"     ⏱️ Processing Time: {result1.processing_time_ms}ms")
        else:
            print(f"     ❌ Analysis failed: {result1.error_message}")
            return False

        # Test irrelevant article
        print("  📰 Testing irrelevant article (restaurant)...")
        result2 = await provider.analyze_relevance(IRRELEVANT_ARTICLE, SAMPLE_TOPIC)

        if result2.success:
            print(f"     ✅ Relevance Score: {result2.relevance_score:.3f}")
            print(f"     ✅ Confidence: {result2.confidence:.3f}")
            print(f"     ✅ Matched Keywords: {result2.matched_keywords}")
            print(f"     ⏱️ Processing Time: {result2.processing_time_ms}ms")
        else:
            print(f"     ❌ Analysis failed: {result2.error_message}")
            return False

        # Verify results make sense
        if result1.relevance_score > result2.relevance_score:
            print("     ✅ Relevance scoring working correctly (AWS > restaurant)")
        else:
            print("     ⚠️ Unexpected relevance scoring (restaurant >= AWS)")

        return True

    except Exception as e:
        print(f"     ❌ Relevance analysis error: {e}")
        return False


@pytest.mark.asyncio
async def test_summarization():
    """Test article summarization."""
    print("\n📝 Testing Summarization...")

    try:
        settings = get_settings()
        provider = GroqProvider(
            api_key=settings.ai.groq_api_key, model_name=settings.ai.groq_model
        )

        print("  📰 Generating summary for AWS article...")
        result = await provider.generate_summary(SAMPLE_ARTICLE, max_sentences=2)

        if result.success and result.summary:
            print(f"     ✅ Summary: {result.summary}")
            print(f"     ✅ Length: {len(result.summary)} characters")
            print(f"     ⏱️ Processing Time: {result.processing_time_ms}ms")

            # Basic quality checks
            sentences = result.summary.count(".") + result.summary.count("!")
            if 1 <= sentences <= 3:
                print("     ✅ Summary length appropriate")
            else:
                print(f"     ⚠️ Summary has {sentences} sentences (expected 1-3)")

            return True
        else:
            print(
                f"     ❌ Summarization failed: {result.error_message if result else 'No result'}"
            )
            return False

    except Exception as e:
        print(f"     ❌ Summarization error: {e}")
        return False


@pytest.mark.asyncio
async def test_ai_manager():
    """Test AI Manager with multiple scenarios."""
    print("\n🎛️ Testing AI Manager...")

    try:
        manager = AIManager()

        # Test provider status
        print("  📊 Checking provider status...")
        status = manager.get_provider_status()
        for provider_name, provider_status in status.items():
            health_icon = "✅" if provider_status["healthy"] else "⚠️"
            print(
                f"     {health_icon} {provider_name}: {'healthy' if provider_status['healthy'] else 'unhealthy'}"
            )

        # Test connection for all providers
        print("  🔌 Testing all provider connections...")
        connection_results = await manager.test_all_providers()
        for provider_type, success in connection_results.items():
            status_icon = "✅" if success else "❌"
            print(
                f"     {status_icon} {provider_type.value}: {'connected' if success else 'failed'}"
            )

        # Test relevance analysis with fallback
        print("  🤔 Testing relevance analysis with fallback...")
        result = await manager.analyze_relevance(SAMPLE_ARTICLE, SAMPLE_TOPIC)

        if result.success:
            print(f"     ✅ Analysis successful with {result.provider}")
            print(f"     ✅ Relevance Score: {result.relevance_score:.3f}")
            print(f"     ✅ Quality Score: {result.quality_score:.3f}")
        else:
            print(f"     ❌ Analysis failed: {result.error_message}")
            return False

        # Test summarization
        print("  📝 Testing summarization with fallback...")
        summary_result = await manager.generate_summary(SAMPLE_ARTICLE)

        if summary_result.success and summary_result.summary:
            print(f"     ✅ Summarization successful with {summary_result.provider}")
            print(f"     ✅ Summary: {summary_result.summary[:100]}...")
        else:
            print(
                f"     ❌ Summarization failed: {summary_result.error_message if summary_result else 'No result'}"
            )

        await manager.shutdown()
        return True

    except Exception as e:
        print(f"     ❌ AI Manager error: {e}")
        return False


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting behavior."""
    print("\n⏱️ Testing Rate Limiting...")

    try:
        settings = get_settings()
        provider = GroqProvider(
            api_key=settings.ai.groq_api_key, model_name=settings.ai.groq_model
        )

        # Check initial rate limits
        rate_info = provider.get_rate_limits()
        print(
            f"  📊 Rate Limits: {rate_info.requests_per_minute}/min, {rate_info.requests_per_day}/day"
        )
        print(f"  📊 Current Usage: {rate_info.current_usage}")

        # Test a few rapid requests
        print("  🚀 Making rapid requests to test throttling...")
        for i in range(3):
            start_time = asyncio.get_event_loop().time()
            result = await provider.analyze_relevance(SAMPLE_ARTICLE, SAMPLE_TOPIC)
            end_time = asyncio.get_event_loop().time()

            if result.success:
                print(f"     ✅ Request {i+1}: {(end_time - start_time)*1000:.1f}ms")
            else:
                print(f"     ⚠️ Request {i+1} failed: {result.error_message}")

        # Check updated usage
        updated_rate_info = provider.get_rate_limits()
        print(f"  📊 Updated Usage: {updated_rate_info.current_usage}")

        return True

    except Exception as e:
        print(f"     ❌ Rate limiting test error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 CuliFeed Groq AI Integration Tests")
    print("=" * 50)

    # Configure logging
    configure_application_logging(
        log_level="INFO", enable_console=True, structured_logging=False
    )

    tests = [
        ("Connection Test", test_groq_connection),
        ("Relevance Analysis", test_relevance_analysis),
        ("Summarization", test_summarization),
        ("AI Manager", test_ai_manager),
        ("Rate Limiting", test_rate_limiting),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            success = await test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

        print()  # Empty line between tests

    # Results summary
    print("=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Groq integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
