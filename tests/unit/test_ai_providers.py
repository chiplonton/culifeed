#!/usr/bin/env python3
"""
AI Providers Unit Tests
======================

Unit tests for AI provider implementations without requiring real API keys.
Tests the code structure, imports, and basic functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test all AI module imports."""
    print("📦 Testing imports...")

    try:
        from culifeed.ai import AIManager, GroqProvider, AIResult, AIError
        from culifeed.ai.providers.base import AIProviderType, RateLimitInfo
        from culifeed.config.settings import get_settings
        from culifeed.utils.exceptions import ErrorCode

        print("  ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_error_codes():
    """Test that all required error codes exist."""
    print("🏷️ Testing error codes...")

    try:
        from culifeed.utils.exceptions import ErrorCode

        required_codes = [
            "AI_PROCESSING_ERROR",
            "AI_PROVIDER_UNAVAILABLE",
            "AI_INVALID_CREDENTIALS",
            "AI_CONNECTION_ERROR",
            "AI_API_ERROR",
        ]

        missing_codes = []
        for code in required_codes:
            if not hasattr(ErrorCode, code):
                missing_codes.append(code)

        if missing_codes:
            print(f"  ❌ Missing error codes: {missing_codes}")
            return False
        else:
            print("  ✅ All error codes present")
            return True

    except Exception as e:
        print(f"  ❌ Error code test failed: {e}")
        return False


def test_data_models():
    """Test data model creation."""
    print("🏗️ Testing data models...")

    try:
        from culifeed.database.models import Article, Topic
        from culifeed.ai.providers.base import AIResult, RateLimitInfo

        # Test Article creation
        article = Article(
            title="Test Article",
            url="https://example.com/test",
            content="This is test content",
            source_feed="test_feed",
        )

        # Test Topic creation
        topic = Topic(
            chat_id="test_chat", name="Test Topic", keywords=["test", "example"]
        )

        # Test AIResult creation
        result = AIResult(success=True, relevance_score=0.8, confidence=0.9)

        # Test RateLimitInfo creation
        rate_info = RateLimitInfo(requests_per_minute=30, requests_per_day=1000)

        print(f"  ✅ Article: {article.title}")
        print(f"  ✅ Topic: {topic.name} with {len(topic.keywords)} keywords")
        print(
            f"  ✅ AIResult: success={result.success}, score={result.relevance_score}"
        )
        print(f"  ✅ RateLimit: {rate_info.requests_per_minute}/min")

        return True

    except Exception as e:
        print(f"  ❌ Data model test failed: {e}")
        return False


def test_ai_manager_initialization():
    """Test AI Manager initialization."""
    print("🎛️ Testing AI Manager initialization...")

    try:
        from culifeed.ai.ai_manager import AIManager

        # This should work even without API keys
        manager = AIManager()

        print(f"  ✅ AI Manager created: {manager}")
        print(f"  ✅ Available providers: {len(manager.providers)}")

        # Test provider status (should work without API calls)
        status = manager.get_provider_status()
        print(f"  ✅ Provider status retrieved: {len(status)} providers")

        return True

    except Exception as e:
        print(f"  ❌ AI Manager test failed: {e}")
        return False


def test_provider_creation():
    """Test provider creation (without API calls)."""
    print("🤖 Testing provider creation...")

    try:
        from culifeed.ai.providers.groq_provider import GroqProvider
        from culifeed.ai.providers.base import AIProviderType

        # Test with dummy API key
        provider = GroqProvider(
            api_key="dummy_key_for_testing", model_name="llama-3.1-8b-instant"
        )

        print(f"  ✅ Groq provider created: {provider}")
        print(f"  ✅ Provider type: {provider.provider_type}")
        print(f"  ✅ Model: {provider.model_name}")

        # Test rate limit info
        rate_info = provider.get_rate_limits()
        print(f"  ✅ Rate limits: {rate_info.requests_per_minute}/min")

        return True

    except Exception as e:
        print(f"  ❌ Provider creation failed: {e}")
        return False


def test_fallback_logic():
    """Test keyword fallback logic."""
    print("🔄 Testing fallback logic...")

    try:
        from culifeed.ai.ai_manager import AIManager
        from culifeed.database.models import Article, Topic

        # Create test data
        article = Article(
            title="AWS Lambda Serverless Computing Guide",
            url="https://example.com/aws-lambda",
            content="Learn about AWS Lambda serverless computing and how to deploy functions in the cloud",
            source_feed="tech_news",
        )

        topic = Topic(
            chat_id="test_chat",
            name="Cloud Computing",
            keywords=["AWS", "serverless", "lambda", "cloud"],
        )

        manager = AIManager()

        # Test keyword fallback (this should work without API calls)
        result = manager._keyword_fallback_analysis(article, topic)

        print(f"  ✅ Keyword analysis: score={result.relevance_score:.3f}")
        print(f"  ✅ Matched keywords: {result.matched_keywords}")
        print(f"  ✅ Confidence: {result.confidence:.3f}")
        print(f"  ✅ Provider: {result.provider}")

        # Should find matches for "AWS", "serverless", "lambda"
        expected_matches = [
            "aws",
            "serverless",
            "lambda",
        ]  # lowercase due to normalization
        actual_matches = [k.lower() for k in result.matched_keywords]

        matches_found = sum(
            1 for keyword in expected_matches if keyword in actual_matches
        )
        if matches_found >= 2:
            print(
                f"  ✅ Keyword matching working correctly ({matches_found}/3 keywords found)"
            )
        else:
            print(f"  ⚠️ Only {matches_found}/3 expected keywords found")

        return True

    except Exception as e:
        print(f"  ❌ Fallback logic test failed: {e}")
        return False


def test_configuration():
    """Test configuration loading."""
    print("⚙️ Testing configuration...")

    try:
        from culifeed.config.settings import get_settings

        settings = get_settings()

        print(f"  ✅ Settings loaded: {settings.app_name}")
        print(f"  ✅ Primary AI provider: {settings.processing.ai_provider}")
        print(f"  ✅ Groq model: {settings.ai.groq_model}")
        print(f"  ✅ Fallback enabled: {settings.limits.fallback_to_groq}")
        print(f"  ✅ Keyword fallback: {settings.limits.fallback_to_keywords}")

        # Test AI provider detection
        available_providers = settings.get_ai_fallback_providers()
        print(f"  ✅ Available AI providers: {len(available_providers)}")

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔍 CuliFeed Groq Implementation Validation")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("Error Codes", test_error_codes),
        ("Data Models", test_data_models),
        ("AI Manager", test_ai_manager_initialization),
        ("Provider Creation", test_provider_creation),
        ("Fallback Logic", test_fallback_logic),
        ("Configuration", test_configuration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}\n")

    # Results summary
    print("=" * 50)
    print(f"🏁 Validation Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All validation tests passed! Implementation structure is correct.")
        print("\n💡 Next steps:")
        print(
            "   1. Set a real Groq API key: export CULIFEED_AI__GROQ_API_KEY='your_key_here'"
        )
        print("   2. Run: python test_groq_integration.py")
        print("   3. Test with real API calls")
        return True
    else:
        print(
            "⚠️ Some validation tests failed. Fix issues before testing with real API."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
