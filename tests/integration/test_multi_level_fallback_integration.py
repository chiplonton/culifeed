#!/usr/bin/env python3
"""
Final Integration Test for Multi-Level Fallback
===============================================

Tests the complete multi-level fallback functionality without external dependencies.
Verifies that our implementation is working end-to-end.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ['CULIFEED_AI__GEMINI_API_KEY'] = 'test-gemini-key'
os.environ['CULIFEED_AI__GROQ_API_KEY'] = 'test-groq-key'


async def test_complete_integration():
    """Test complete multi-level fallback integration."""
    print("🔬 Complete Multi-Level Fallback Integration Test")
    print("=" * 60)
    
    # Test 1: Configuration loading
    print("1️⃣ Testing configuration loading...")
    try:
        from culifeed.config.settings import get_settings, AIProvider as ConfigAIProvider
        
        settings = get_settings()
        assert settings.ai.groq_models == ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant']
        assert settings.ai.gemini_models == ['gemini-2.5-flash', 'gemini-2.5-flash-lite']
        print("   ✅ Multi-model configuration loaded correctly")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False
    
    # Test 2: Provider-model combinations generation
    print("\n2️⃣ Testing provider-model combinations...")
    try:
        from culifeed.ai.ai_manager import AIManager
        
        # Mock provider initialization to avoid API dependencies
        with patch.object(AIManager, '_initialize_providers'):
            ai_manager = AIManager()
            ai_manager.providers = {'groq': Mock(), 'gemini': Mock()}
            ai_manager.provider_health = {
                'groq': Mock(is_healthy=True),
                'gemini': Mock(is_healthy=True)
            }
            
            # Mock provider priority order
            with patch.object(ai_manager, '_get_provider_priority_order', return_value=['groq', 'gemini']):
                combinations = ai_manager._get_provider_model_combinations()
                
                expected_combinations = [
                    ('groq', 'llama-3.1-70b-versatile'),
                    ('groq', 'llama-3.1-8b-instant'),
                    ('gemini', 'gemini-2.5-flash'),
                    ('gemini', 'gemini-2.5-flash-lite')
                ]
                
                assert combinations == expected_combinations
                print(f"   ✅ Generated correct combinations: {len(combinations)} total")
                for i, (provider, model) in enumerate(combinations, 1):
                    print(f"      {i}. {provider} → {model}")
        
    except Exception as e:
        print(f"   ❌ Provider-model combinations test failed: {e}")
        return False
    
    # Test 3: Fallback logic simulation
    print("\n3️⃣ Testing fallback logic simulation...")
    try:
        from culifeed.ai.ai_manager import AIManager
        from culifeed.ai.providers.base import AIResult, AIError
        from culifeed.database.models import Article, Topic
        from datetime import datetime, timezone
        
        # Create test data
        test_article = Article(
            id="test-123",
            title="Test Article about AWS Lambda Functions",
            url="https://example.com/test",
            content="This article discusses serverless computing with AWS Lambda functions and EC2 integration.",
            published_at=datetime.now(timezone.utc),
            source_feed="https://example.com/feed",
            content_hash="test-hash"
        )
        
        test_topic = Topic(
            id=1,
            chat_id="test-chat",
            name="lambda function ec2",
            keywords=["lambda", "aws lambda", "serverless", "ec2"],
            confidence_threshold=0.8
        )
        
        # Mock providers with different failure scenarios
        mock_groq = Mock()
        mock_gemini = Mock()
        
        # Scenario: First model fails, second succeeds
        mock_groq.analyze_relevance_with_model = AsyncMock(side_effect=[
            AIError("Rate limit exceeded", provider="groq"),  # First model fails
            AIResult(success=True, relevance_score=0.85, confidence=0.9, reasoning="Test reasoning")  # Second model succeeds
        ])
        
        with patch.object(AIManager, '_initialize_providers'):
            ai_manager = AIManager()
            ai_manager.providers = {'groq': mock_groq, 'gemini': mock_gemini}
            ai_manager.provider_health = {
                'groq': Mock(is_healthy=True, record_error=Mock(), record_success=Mock()),
                'gemini': Mock(is_healthy=True)
            }
            ai_manager.enable_keyword_fallback = True
            
            # Mock combinations to test specific fallback sequence
            combinations = [
                ('groq', 'llama-3.1-70b-versatile'),    # This will fail
                ('groq', 'llama-3.1-8b-instant')  # This will succeed
            ]
            
            with patch.object(ai_manager, '_get_provider_model_combinations', return_value=combinations):
                result = await ai_manager.analyze_relevance(test_article, test_topic)
                
                assert result.success, "Analysis should succeed on second attempt"
                assert result.relevance_score == 0.85, "Should return correct relevance score"
                
                # Verify that both models were tried
                assert mock_groq.analyze_relevance_with_model.call_count == 2
                
                print("   ✅ Fallback logic works correctly")
                print(f"      📊 Result: score={result.relevance_score}, confidence={result.confidence}")
        
    except Exception as e:
        print(f"   ❌ Fallback logic simulation failed: {e}")
        return False
    
    # Test 4: Provider switching capabilities
    print("\n4️⃣ Testing provider switching capabilities...")
    try:
        from culifeed.ai.providers.groq_provider import GroqProvider
        from culifeed.ai.providers.gemini_provider import GeminiProvider
        
        # Test GroqProvider model switching (static test)
        available_groq_models = GroqProvider.get_available_models()
        assert "llama-3.1-8b-instant" in available_groq_models
        assert "llama-3.1-70b-versatile" in available_groq_models
        print(f"   ✅ GroqProvider supports {len(available_groq_models)} models")
        
        # Test GeminiProvider model switching (static test)
        available_gemini_models = GeminiProvider.get_available_models()
        assert "gemini-2.5-flash" in available_gemini_models
        assert "gemini-2.5-flash-lite" in available_gemini_models
        print(f"   ✅ GeminiProvider supports {len(available_gemini_models)} models")
        
    except Exception as e:
        print(f"   ❌ Provider switching test failed: {e}")
        return False
    
    # Test 5: Backward compatibility
    print("\n5️⃣ Testing backward compatibility...")
    try:
        settings = get_settings()
        
        # Test that legacy single-model fields still work
        assert hasattr(settings.ai, 'groq_model'), "Legacy groq_model field should exist"
        assert hasattr(settings.ai, 'gemini_model'), "Legacy gemini_model field should exist"
        
        # Test that legacy imports still work
        from culifeed.ai import AIManager
        from culifeed.ai.providers import GroqProvider
        
        print("   ✅ Backward compatibility maintained")
        
    except Exception as e:
        print(f"   ❌ Backward compatibility test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("\n✅ Multi-level fallback system is working correctly:")
    print("   • Configuration loading ✓")
    print("   • Provider-model combinations ✓")
    print("   • Two-level fallback logic ✓")
    print("   • Provider switching capabilities ✓")
    print("   • Backward compatibility ✓")
    print("\n🚀 Ready for production deployment!")
    
    return True


async def main():
    """Main test runner."""
    success = await test_complete_integration()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed!")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())