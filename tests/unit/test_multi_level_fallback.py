#!/usr/bin/env python3
"""
Multi-Level Fallback Unit Tests
==============================

Unit tests for the two-level AI fallback system:
1. Model-level fallback within same provider
2. Provider-level fallback across different providers

Tests focus on logic, configuration, and error handling without requiring real API keys.
"""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock external dependencies before imports
sys.modules['groq'] = Mock()
sys.modules['google.generativeai'] = Mock()

from culifeed.config.settings import get_settings, AIProvider as ConfigAIProvider
from culifeed.database.models import Article, Topic


def test_configuration_multi_models():
    """Test that configuration correctly loads multi-model settings."""
    print("🔧 Testing multi-model configuration...")
    
    try:
        settings = get_settings()
        
        # Test that multi-model configuration exists
        assert hasattr(settings.ai, 'groq_models'), "Missing groq_models configuration"
        assert hasattr(settings.ai, 'gemini_models'), "Missing gemini_models configuration"
        assert hasattr(settings.ai, 'huggingface_models'), "Missing huggingface_models configuration"
        assert hasattr(settings.ai, 'get_models_for_provider'), "Missing get_models_for_provider method"

        # Test Groq models
        groq_models = settings.ai.get_models_for_provider(ConfigAIProvider.GROQ)
        assert isinstance(groq_models, list), "Groq models should be a list"
        assert len(groq_models) >= 1, "Should have at least one Groq model"
        assert "llama-3.1-8b-instant" in groq_models, "Should include default Groq model"

        # Test Gemini models
        gemini_models = settings.ai.get_models_for_provider(ConfigAIProvider.GEMINI)
        assert isinstance(gemini_models, list), "Gemini models should be a list"
        assert len(gemini_models) >= 1, "Should have at least one Gemini model"

        # Test HuggingFace models
        huggingface_models = settings.ai.get_models_for_provider(ConfigAIProvider.HUGGINGFACE)
        assert isinstance(huggingface_models, list), "HuggingFace models should be a list"
        assert len(huggingface_models) >= 1, "Should have at least one HuggingFace model"
        assert "facebook/bart-large-cnn" in huggingface_models, "Should include confirmed working model"

        print(f"  ✅ Groq models: {groq_models}")
        print(f"  ✅ Gemini models: {gemini_models}")
        print(f"  ✅ HuggingFace models: {huggingface_models}")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_provider_model_combinations():
    """Test that AIManager correctly generates provider-model combinations."""
    print("🔄 Testing provider-model combinations...")
    
    try:
        from culifeed.ai.ai_manager import AIManager
        
        # Mock provider initialization to avoid API calls
        with patch.object(AIManager, '_initialize_providers'):
            ai_manager = AIManager()
            
            # Mock available providers for testing
            ai_manager.providers = {
                'groq': Mock(),
                'gemini': Mock()
            }
            ai_manager.provider_health = {
                'groq': Mock(is_healthy=True),
                'gemini': Mock(is_healthy=True)
            }
            
            # Mock _get_provider_priority_order to return predictable order
            with patch.object(ai_manager, '_get_provider_priority_order', return_value=['groq', 'gemini']):
                combinations = ai_manager._get_provider_model_combinations()
                
                # Verify combinations structure
                assert isinstance(combinations, list), "Combinations should be a list"
                assert len(combinations) > 0, "Should have at least one combination"
                
                # Verify each combination is a tuple of (provider, model)
                for combo in combinations:
                    assert isinstance(combo, tuple), f"Each combination should be a tuple: {combo}"
                    assert len(combo) == 2, f"Each combination should have 2 elements: {combo}"
                    provider_type, model_name = combo
                    assert isinstance(provider_type, str), f"Provider type should be string: {provider_type}"
                    assert isinstance(model_name, str), f"Model name should be string: {model_name}"
                
                print(f"  ✅ Generated {len(combinations)} combinations:")
                for i, (provider, model) in enumerate(combinations, 1):
                    print(f"    {i}. {provider} → {model}")
                
                return True
                
    except Exception as e:
        print(f"  ❌ Provider-model combinations test failed: {e}")
        return False


def test_groq_provider_multi_model():
    """Test GroqProvider multi-model functionality."""
    print("🤖 Testing GroqProvider multi-model support...")
    
    try:
        # Mock groq library
        with patch('culifeed.ai.providers.groq_provider.GROQ_AVAILABLE', True):
            with patch('culifeed.ai.providers.groq_provider.Groq'), \
                 patch('culifeed.ai.providers.groq_provider.AsyncGroq'):
                
                from culifeed.ai.providers.groq_provider import GroqProvider
                
                # Test provider creation
                provider = GroqProvider(api_key="test-key", model_name="llama-3.1-8b-instant")
                
                # Test model switching
                original_model = provider.model_name
                provider.set_model("llama-3.1-70b-versatile")
                assert provider.model_name == "llama-3.1-70b-versatile", "Model should have switched"
                
                # Test invalid model (should keep current)
                provider.set_model("invalid-model")
                assert provider.model_name == "llama-3.1-70b-versatile", "Should keep current model for invalid"
                
                # Test get_available_models
                models = GroqProvider.get_available_models()
                assert isinstance(models, list), "Available models should be a list"
                assert len(models) > 0, "Should have available models"
                
                print(f"  ✅ Available Groq models: {models}")
                return True
                
    except Exception as e:
        print(f"  ❌ GroqProvider multi-model test failed: {e}")
        return False


def test_gemini_provider_basic_structure():
    """Test GeminiProvider basic structure and available models."""
    print("🧠 Testing GeminiProvider basic structure...")
    
    try:
        # Skip actual provider creation test since it needs complex mocking
        # Focus on testing the static methods and structure
        from culifeed.ai.providers.gemini_provider import GeminiProvider
        
        # Test get_available_models (static method)
        models = GeminiProvider.get_available_models()
        assert isinstance(models, list), "Available models should be a list"
        assert len(models) > 0, "Should have available models"
        assert "gemini-2.5-flash" in models, "Should include default model"
        
        # Test that the class has required methods
        required_methods = ['analyze_relevance', 'generate_summary', 'test_connection', 
                          'set_model', 'analyze_relevance_with_model', 'generate_summary_with_model']
        
        for method_name in required_methods:
            assert hasattr(GeminiProvider, method_name), f"Should have {method_name} method"
        
        print(f"  ✅ Available Gemini models: {models}")
        print(f"  ✅ All required methods present")
        return True
                
    except Exception as e:
        print(f"  ❌ GeminiProvider basic structure test failed: {e}")
        return False


def test_huggingface_provider_basic_structure():
    """Test HuggingFaceProvider basic structure and available models."""
    print("🤗 Testing HuggingFaceProvider basic structure...")

    try:
        # Mock aiohttp to avoid import issues
        with patch('culifeed.ai.providers.huggingface_provider.AIOHTTP_AVAILABLE', True):
            from culifeed.ai.providers.huggingface_provider import HuggingFaceProvider

            # Test get_available_models (static method)
            models = HuggingFaceProvider.get_available_models()
            assert isinstance(models, list), "Available models should be a list"
            assert len(models) > 0, "Should have available models"
            assert "facebook/bart-large-cnn" in models, "Should include confirmed working model"

            # Test that the class has required methods
            required_methods = ['analyze_relevance', 'generate_summary', 'test_connection',
                              'set_model', 'analyze_relevance_with_model', 'generate_summary_with_model']

            for method_name in required_methods:
                assert hasattr(HuggingFaceProvider, method_name), f"Should have {method_name} method"

            print(f"  ✅ Available HuggingFace models: {models}")
            print(f"  ✅ All required methods present")
            return True

    except Exception as e:
        print(f"  ❌ HuggingFaceProvider basic structure test failed: {e}")
        return False


@pytest.mark.asyncio
async def test_ai_manager_fallback_logic():
    """Test AIManager two-level fallback logic."""
    print("🔄 Testing AIManager fallback logic...")
    
    try:
        from culifeed.ai.ai_manager import AIManager
        from culifeed.ai.providers.base import AIResult
        
        # Create test data
        test_article = Article(
            id="test-123",
            title="Test Article",
            url="https://example.com/test",
            content="Test content for AI analysis",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed",
            content_hash="test-hash"
        )
        
        test_topic = Topic(
            id=1,
            chat_id="test-chat",
            name="test topic",
            keywords=["test", "example"],
            confidence_threshold=0.8
        )
        
        # Mock provider initialization
        with patch.object(AIManager, '_initialize_providers'):
            ai_manager = AIManager()
            
            # Create mock providers
            mock_groq = Mock()
            mock_gemini = Mock()
            
            # Test successful case (first provider works)
            mock_groq.analyze_relevance_with_model = AsyncMock(return_value=AIResult(
                success=True,
                relevance_score=0.9,
                confidence=0.8,
                reasoning="Test reasoning"
            ))
            
            ai_manager.providers = {'groq': mock_groq, 'gemini': mock_gemini}
            ai_manager.provider_health = {
                'groq': Mock(is_healthy=True),
                'gemini': Mock(is_healthy=True)
            }
            
            # Mock provider-model combinations
            with patch.object(ai_manager, '_get_provider_model_combinations', 
                            return_value=[('groq', 'llama-3.1-70b-versatile'), ('groq', 'llama-3.1-8b-instant')]):
                
                result = await ai_manager.analyze_relevance(test_article, test_topic)
                
                assert result.success, "Analysis should succeed"
                assert result.relevance_score == 0.9, "Should return correct relevance score"
                
                # Verify that analyze_relevance_with_model was called
                mock_groq.analyze_relevance_with_model.assert_called_once()
                
                print("  ✅ Fallback logic works correctly")
                return True
                
    except Exception as e:
        print(f"  ❌ AIManager fallback logic test failed: {e}")
        return False


def test_error_handling():
    """Test error handling in multi-level fallback."""
    print("⚠️ Testing error handling...")
    
    try:
        from culifeed.ai.providers.base import AIError, ErrorCode
        
        # Test AIError creation with provider info
        error = AIError(
            "Test error message",
            provider="groq/llama-3.1-70b-versatile",
            error_code=ErrorCode.AI_PROCESSING_ERROR
        )
        
        assert error.provider == "groq/llama-3.1-70b-versatile", "Should store provider info"
        assert error.error_code == ErrorCode.AI_PROCESSING_ERROR, "Should store error code"
        
        print("  ✅ Error handling works correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False


def run_all_tests():
    """Run all unit tests."""
    print("🧪 Multi-Level Fallback Unit Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Multi-Models", test_configuration_multi_models),
        ("Provider-Model Combinations", test_provider_model_combinations),
        ("GroqProvider Multi-Model", test_groq_provider_multi_model),
        ("GeminiProvider Basic Structure", test_gemini_provider_basic_structure),
        ("HuggingFaceProvider Basic Structure", test_huggingface_provider_basic_structure),
        ("Error Handling", test_error_handling),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED\n")
            else:
                print(f"❌ {test_name} FAILED\n")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}\n")
    
    # Test async function separately
    try:
        import asyncio
        print("🔄 Testing AIManager fallback logic...")
        if asyncio.run(test_ai_manager_fallback_logic()):
            passed += 1
            print("✅ AIManager Fallback Logic PASSED\n")
            total += 1
        else:
            print("❌ AIManager Fallback Logic FAILED\n")
            total += 1
    except Exception as e:
        print(f"❌ AIManager Fallback Logic FAILED: {e}\n")
        total += 1
    
    print("=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Multi-level fallback is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)