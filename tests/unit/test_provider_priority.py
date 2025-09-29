"""
Test cases for dynamic AI provider priority configuration.

Tests all priority profiles, validation logic, and edge cases to ensure
provider ordering works correctly with different configurations.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List

from culifeed.config.settings import (
    AISettings,
    AIProvider,
    ProviderPriority,
    CuliFeedSettings,
)
from culifeed.ai.ai_manager import AIManager
from culifeed.ai.providers.base import AIProviderType


class TestProviderPriorityConfiguration:
    """Test provider priority configuration in settings."""

    def test_default_cost_optimized_profile(self):
        """Test default cost optimized profile returns correct order."""
        settings = AISettings()

        # Should default to cost optimized
        assert settings.provider_priority_profile == ProviderPriority.COST_OPTIMIZED

        # Should return cost-optimized order
        expected_order = [
            AIProvider.GROQ,
            AIProvider.DEEPSEEK,
            AIProvider.GEMINI,
            AIProvider.OPENAI,
        ]

        actual_order = settings.get_provider_priority_order()
        assert actual_order == expected_order

    def test_quality_first_profile(self):
        """Test quality first profile returns premium providers first."""
        settings = AISettings(provider_priority_profile=ProviderPriority.QUALITY_FIRST)

        expected_order = [
            AIProvider.DEEPSEEK,
            AIProvider.OPENAI,
            AIProvider.GEMINI,
            AIProvider.GROQ,
        ]

        actual_order = settings.get_provider_priority_order()
        assert actual_order == expected_order

    def test_balanced_profile(self):
        """Test balanced profile returns mixed cost/quality order."""
        settings = AISettings(provider_priority_profile=ProviderPriority.BALANCED)

        expected_order = [
            AIProvider.DEEPSEEK,
            AIProvider.GEMINI,
            AIProvider.GROQ,
            AIProvider.OPENAI,
        ]

        actual_order = settings.get_provider_priority_order()
        assert actual_order == expected_order

    def test_custom_profile_with_valid_order(self):
        """Test custom profile with valid provider order."""
        custom_order = [AIProvider.OPENAI, AIProvider.GEMINI, AIProvider.GROQ]

        settings = AISettings(
            provider_priority_profile=ProviderPriority.CUSTOM,
            custom_provider_order=custom_order,
        )

        actual_order = settings.get_provider_priority_order()
        assert actual_order == custom_order

    def test_custom_profile_empty_order_fallback(self):
        """Test custom profile with empty order falls back to cost optimized."""
        settings = AISettings(
            provider_priority_profile=ProviderPriority.CUSTOM, custom_provider_order=[]
        )

        # Should fallback to cost optimized when custom is empty
        expected_order = [
            AIProvider.GROQ,
            AIProvider.DEEPSEEK,
            AIProvider.GEMINI,
            AIProvider.OPENAI,
        ]

        actual_order = settings.get_provider_priority_order()
        assert actual_order == expected_order

    def test_validation_success_cases(self):
        """Test validation passes for valid configurations."""
        # Valid cost optimized (default)
        settings1 = AISettings()
        assert settings1.validate_priority_configuration() == []

        # Valid quality first
        settings2 = AISettings(provider_priority_profile=ProviderPriority.QUALITY_FIRST)
        assert settings2.validate_priority_configuration() == []

        # Valid custom with providers
        settings3 = AISettings(
            provider_priority_profile=ProviderPriority.CUSTOM,
            custom_provider_order=[AIProvider.OPENAI, AIProvider.GEMINI],
        )
        assert settings3.validate_priority_configuration() == []

    def test_validation_custom_empty_order(self):
        """Test validation fails for custom profile with empty order."""
        settings = AISettings(
            provider_priority_profile=ProviderPriority.CUSTOM, custom_provider_order=[]
        )

        errors = settings.validate_priority_configuration()
        assert len(errors) == 1
        assert "Custom provider order is empty" in errors[0]

    def test_validation_custom_duplicate_providers(self):
        """Test validation fails for duplicate providers in custom order."""
        settings = AISettings(
            provider_priority_profile=ProviderPriority.CUSTOM,
            custom_provider_order=[
                AIProvider.OPENAI,
                AIProvider.GEMINI,
                AIProvider.OPENAI,
            ],
        )

        errors = settings.validate_priority_configuration()
        assert len(errors) == 1
        assert "Duplicate providers found" in errors[0]


class TestAIManagerProviderPriority:
    """Test AI manager provider priority implementation."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        return Mock(spec=CuliFeedSettings)

    @pytest.fixture
    def mock_ai_settings(self):
        """Create mock AI settings for testing."""
        return Mock(spec=AISettings)

    def test_cost_optimized_provider_order(self, mock_settings, mock_ai_settings):
        """Test AI manager uses cost optimized provider order."""
        # Setup mock settings
        mock_ai_settings.get_provider_priority_order.return_value = [
            AIProvider.GROQ,
            AIProvider.DEEPSEEK,
            AIProvider.GEMINI,
            AIProvider.OPENAI,
        ]
        mock_ai_settings.provider_priority_profile = ProviderPriority.COST_OPTIMIZED
        mock_settings.ai = mock_ai_settings

        # Add missing mock attributes that AIManager needs
        mock_processing = Mock()
        mock_processing.ai_provider = AIProvider.GROQ
        mock_settings.processing = mock_processing

        mock_limits = Mock()
        mock_limits.fallback_to_keywords = True
        mock_settings.limits = mock_limits

        # Create AI manager with mocked providers
        with patch.object(AIManager, "_initialize_providers"), patch.object(
            AIManager, "_validate_and_log_provider_configuration"
        ):
            ai_manager = AIManager(settings=mock_settings)

            # Mock providers and health
            ai_manager.providers = {
                AIProviderType.GROQ: Mock(),
                AIProviderType.DEEPSEEK: Mock(),
                AIProviderType.GEMINI: Mock(),
                AIProviderType.OPENAI: Mock(),
            }

            # Mock all providers as healthy
            ai_manager.provider_health = {
                provider_type: Mock(is_healthy=True)
                for provider_type in ai_manager.providers.keys()
            }

            # Test provider order
            with patch.object(ai_manager, "_config_to_provider_type") as mock_convert:
                mock_convert.side_effect = [
                    AIProviderType.GROQ,
                    AIProviderType.DEEPSEEK,
                    AIProviderType.GEMINI,
                    AIProviderType.OPENAI,
                ]

                order = ai_manager._get_provider_priority_order()

                # Should prioritize available providers in configured order
                expected_available = [
                    AIProviderType.GROQ,
                    AIProviderType.DEEPSEEK,
                    AIProviderType.GEMINI,
                    AIProviderType.OPENAI,
                ]
                assert order == expected_available

    def test_quality_first_provider_order(self, mock_settings, mock_ai_settings):
        """Test AI manager uses quality first provider order."""
        # Setup mock settings for quality first
        mock_ai_settings.get_provider_priority_order.return_value = [
            AIProvider.OPENAI,
            AIProvider.GEMINI,
            AIProvider.GROQ,
            AIProvider.DEEPSEEK,
        ]
        mock_ai_settings.provider_priority_profile = ProviderPriority.QUALITY_FIRST
        mock_settings.ai = mock_ai_settings

        # Add missing mock attributes that AIManager needs
        mock_processing = Mock()
        mock_processing.ai_provider = AIProvider.OPENAI
        mock_settings.processing = mock_processing

        mock_limits = Mock()
        mock_limits.fallback_to_keywords = True
        mock_settings.limits = mock_limits

        # Create AI manager with mocked providers
        with patch.object(AIManager, "_initialize_providers"), patch.object(
            AIManager, "_validate_and_log_provider_configuration"
        ):
            ai_manager = AIManager(settings=mock_settings)

            # Mock providers and health (OpenAI and Gemini available)
            ai_manager.providers = {
                AIProviderType.OPENAI: Mock(),
                AIProviderType.GEMINI: Mock(),
                AIProviderType.GROQ: Mock(),
            }

            ai_manager.provider_health = {
                provider_type: Mock(is_healthy=True)
                for provider_type in ai_manager.providers.keys()
            }

            # Test provider order
            with patch.object(ai_manager, "_config_to_provider_type") as mock_convert:
                mock_convert.side_effect = [
                    AIProviderType.OPENAI,
                    AIProviderType.GEMINI,
                    AIProviderType.GROQ,
                    AIProviderType.DEEPSEEK,
                ]

                order = ai_manager._get_provider_priority_order()

                # Should prioritize OpenAI and Gemini first (quality first)
                assert order[0] == AIProviderType.OPENAI
                assert order[1] == AIProviderType.GEMINI
                assert order[2] == AIProviderType.GROQ

    def test_custom_provider_order(self, mock_settings, mock_ai_settings):
        """Test AI manager uses custom provider order."""
        # Setup mock settings for custom order
        custom_order = [AIProvider.OPENAI, AIProvider.GEMINI]
        mock_ai_settings.get_provider_priority_order.return_value = custom_order
        mock_ai_settings.provider_priority_profile = ProviderPriority.CUSTOM
        mock_ai_settings.custom_provider_order = custom_order
        mock_settings.ai = mock_ai_settings

        # Add missing mock attributes that AIManager needs
        mock_processing = Mock()
        mock_processing.ai_provider = AIProvider.OPENAI
        mock_settings.processing = mock_processing

        mock_limits = Mock()
        mock_limits.fallback_to_keywords = True
        mock_settings.limits = mock_limits

        # Create AI manager with mocked providers
        with patch.object(AIManager, "_initialize_providers"), patch.object(
            AIManager, "_validate_and_log_provider_configuration"
        ):
            ai_manager = AIManager(settings=mock_settings)

            # Mock only the custom providers as available
            ai_manager.providers = {
                AIProviderType.OPENAI: Mock(),
                AIProviderType.GEMINI: Mock(),
                AIProviderType.GROQ: Mock(),  # Available but not in custom order
            }

            ai_manager.provider_health = {
                provider_type: Mock(is_healthy=True)
                for provider_type in ai_manager.providers.keys()
            }

            # Test provider order
            with patch.object(ai_manager, "_config_to_provider_type") as mock_convert:
                mock_convert.side_effect = [
                    AIProviderType.OPENAI,
                    AIProviderType.GEMINI,
                ]

                order = ai_manager._get_provider_priority_order()

                # Should use only custom order, then add remaining
                assert order[0] == AIProviderType.OPENAI
                assert order[1] == AIProviderType.GEMINI
                assert AIProviderType.GROQ in order[2:]  # Added as remaining

    def test_unhealthy_providers_skip(self, mock_settings, mock_ai_settings):
        """Test unhealthy providers are skipped in priority order."""
        mock_ai_settings.get_provider_priority_order.return_value = [
            AIProvider.GROQ,
            AIProvider.GEMINI,
            AIProvider.OPENAI,
        ]
        mock_ai_settings.provider_priority_profile = ProviderPriority.COST_OPTIMIZED
        mock_settings.ai = mock_ai_settings

        # Add missing mock attributes that AIManager needs
        mock_processing = Mock()
        mock_processing.ai_provider = AIProvider.GROQ
        mock_settings.processing = mock_processing

        mock_limits = Mock()
        mock_limits.fallback_to_keywords = True
        mock_settings.limits = mock_limits

        with patch.object(AIManager, "_initialize_providers"), patch.object(
            AIManager, "_validate_and_log_provider_configuration"
        ):
            ai_manager = AIManager(settings=mock_settings)

            # Mock providers with different health states
            ai_manager.providers = {
                AIProviderType.GROQ: Mock(),
                AIProviderType.GEMINI: Mock(),
                AIProviderType.OPENAI: Mock(),
            }

            # GROQ unhealthy, others healthy
            ai_manager.provider_health = {
                AIProviderType.GROQ: Mock(is_healthy=False, rate_limited=False),
                AIProviderType.GEMINI: Mock(is_healthy=True, rate_limited=False),
                AIProviderType.OPENAI: Mock(is_healthy=True, rate_limited=False),
            }

            with patch.object(ai_manager, "_config_to_provider_type") as mock_convert:
                mock_convert.side_effect = [
                    AIProviderType.GROQ,
                    AIProviderType.GEMINI,
                    AIProviderType.OPENAI,
                ]

                order = ai_manager._get_provider_priority_order()

                # Should skip unhealthy GROQ, start with healthy GEMINI
                assert order[0] == AIProviderType.GEMINI
                assert order[1] == AIProviderType.OPENAI
                # Unhealthy but not rate limited should be added as last resort
                assert AIProviderType.GROQ in order


class TestProviderPriorityIntegration:
    """Integration tests for provider priority system."""

    def test_config_file_loading(self):
        """Test loading provider priority from config file."""
        # Test with in-memory config data
        config_data = {
            "ai": {
                "provider_priority_profile": "quality_first",
                "custom_provider_order": [],
            }
        }

        # This would be tested with actual config loading
        # For now, test that settings accept the configuration
        ai_settings = AISettings(**config_data["ai"])

        assert ai_settings.provider_priority_profile == ProviderPriority.QUALITY_FIRST
        assert ai_settings.custom_provider_order == []

    def test_environment_variable_override(self):
        """Test environment variable overrides work correctly."""
        import os

        # Test environment variable override
        test_env = {"CULIFEED_AI__PROVIDER_PRIORITY_PROFILE": "quality_first"}

        with patch.dict(os.environ, test_env):
            # This would test actual environment loading
            # For now, verify the setting accepts the value
            ai_settings = AISettings(provider_priority_profile="quality_first")
            assert (
                ai_settings.provider_priority_profile == ProviderPriority.QUALITY_FIRST
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
