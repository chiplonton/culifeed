"""
Trust Validation System
======================

Provides cross-validation between AI providers and pre-filtering results
to ensure consistency and detect quality issues in content processing.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..database.models import Article, Topic
from ..utils.logging import get_logger_for_component
from ..ai.providers.base import AIResult


class ValidationOutcome(str, Enum):
    """Possible outcomes of trust validation."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of trust validation process."""
    outcome: ValidationOutcome
    ai_score: float
    prefilter_score: float
    score_difference: float
    confidence_adjustment: float
    reason: str
    provider: str
    metadata: Dict[str, Any]

    @property
    def adjusted_confidence(self) -> float:
        """Get confidence adjusted for validation outcome."""
        base_confidence = self.metadata.get('original_confidence', 0.5)
        return base_confidence * self.confidence_adjustment

    @property
    def is_reliable(self) -> bool:
        """Check if this result can be trusted."""
        return self.outcome in [ValidationOutcome.PASS, ValidationOutcome.WARNING]


class TrustValidator:
    """Cross-validation system for AI processing results."""

    def __init__(self, settings: Optional['CuliFeedSettings'] = None):
        """Initialize trust validator.

        Args:
            settings: CuliFeed settings with configurable thresholds
        """
        self.logger = get_logger_for_component("trust_validator")

        # Import here to avoid circular imports
        if settings is None:
            from ..config.settings import get_settings
            settings = get_settings()
        
        self.settings = settings

        # Use configurable validation thresholds from settings
        self.thresholds = {
            'max_score_difference': settings.trust_validation.max_score_difference,
            'warning_score_difference': settings.trust_validation.warning_score_difference,
            'min_prefilter_for_high_ai': settings.trust_validation.min_prefilter_for_high_ai,
            'max_ai_for_low_prefilter': settings.trust_validation.max_ai_for_low_prefilter,
            'confidence_penalty_factor': settings.trust_validation.confidence_penalty_factor,
            'confidence_failure_factor': settings.trust_validation.confidence_failure_factor,
        }

        # Use configurable provider quality factors from settings
        self.provider_quality_factors = {
            'groq': settings.provider_quality.groq,
            'gemini': settings.provider_quality.gemini,
            'openai': settings.provider_quality.openai,
            'huggingface': settings.provider_quality.huggingface,
            'openrouter': settings.provider_quality.openrouter,
            'keyword_backup': settings.provider_quality.keyword_backup,
            'keyword_fallback': settings.provider_quality.keyword_fallback,
        }

        self.logger.info("Trust validator initialized with configurable thresholds", 
                        extra={'thresholds': self.thresholds, 'provider_quality': self.provider_quality_factors})

    def validate_ai_result(self, ai_result: AIResult, prefilter_score: float,
                          article: Article, topic: Topic) -> ValidationResult:
        """Validate AI result against pre-filter score.

        Args:
            ai_result: Result from AI processing
            prefilter_score: Score from pre-filtering
            article: Article being analyzed
            topic: Topic being matched against

        Returns:
            ValidationResult with validation outcome and adjustments
        """
        if not ai_result.success:
            return ValidationResult(
                outcome=ValidationOutcome.SKIP,
                ai_score=0.0,
                prefilter_score=prefilter_score,
                score_difference=0.0,
                confidence_adjustment=0.0,
                reason="AI processing failed",
                provider=ai_result.provider or "unknown",
                metadata={'original_confidence': 0.0}
            )

        ai_score = ai_result.relevance_score
        score_difference = abs(ai_score - prefilter_score)
        provider = ai_result.provider or "unknown"
        original_confidence = ai_result.confidence

        # Apply provider quality factor
        provider_quality = self.provider_quality_factors.get(provider.lower(), 0.5)

        # Determine validation outcome
        outcome, reason, confidence_adjustment = self._determine_outcome(
            ai_score, prefilter_score, score_difference, provider
        )

        # Apply provider quality factor to confidence adjustment
        final_confidence_adjustment = confidence_adjustment * provider_quality

        result = ValidationResult(
            outcome=outcome,
            ai_score=ai_score,
            prefilter_score=prefilter_score,
            score_difference=score_difference,
            confidence_adjustment=final_confidence_adjustment,
            reason=reason,
            provider=provider,
            metadata={
                'original_confidence': original_confidence,
                'provider_quality_factor': provider_quality,
                'article_id': article.id,
                'topic_name': topic.name,
                'article_title': article.title[:100],  # Truncated for logging
            }
        )

        self.logger.debug(
            f"Validation {outcome.value}: AI={ai_score:.3f}, Prefilter={prefilter_score:.3f}, "
            f"Diff={score_difference:.3f}, Provider={provider}, Confidence={final_confidence_adjustment:.3f}"
        )

        return result

    def _determine_outcome(self, ai_score: float, prefilter_score: float,
                          score_difference: float, provider: str) -> Tuple[ValidationOutcome, str, float]:
        """Determine validation outcome based on score comparison.

        Returns:
            Tuple of (outcome, reason, confidence_adjustment)
        """
        # Check for critical mismatches
        if self._is_critical_mismatch(ai_score, prefilter_score):
            return (
                ValidationOutcome.FAIL,
                f"Critical mismatch: AI score {ai_score:.3f} vs prefilter {prefilter_score:.3f}",
                self.thresholds['confidence_failure_factor']
            )

        # Check for large score differences
        if score_difference > self.thresholds['max_score_difference']:
            return (
                ValidationOutcome.FAIL,
                f"Score difference too large: {score_difference:.3f} > {self.thresholds['max_score_difference']}",
                self.thresholds['confidence_failure_factor']
            )

        # Check for moderate score differences (warning)
        if score_difference > self.thresholds['warning_score_difference']:
            return (
                ValidationOutcome.WARNING,
                f"Moderate score difference: {score_difference:.3f}",
                self.thresholds['confidence_penalty_factor']
            )

        # Check for suspicious high AI scores with low prefilter
        if (ai_score > self.settings.trust_validation.suspicious_high_ai_threshold and 
            prefilter_score < self.thresholds['min_prefilter_for_high_ai']):
            return (
                ValidationOutcome.WARNING,
                f"High AI score ({ai_score:.3f}) with low prefilter score ({prefilter_score:.3f})",
                self.thresholds['confidence_penalty_factor']
            )

        # Check for suspicious low AI scores with high prefilter
        if (prefilter_score > self.settings.trust_validation.suspicious_high_prefilter_threshold and 
            ai_score < self.settings.trust_validation.suspicious_low_ai_threshold):
            return (
                ValidationOutcome.WARNING,
                f"Low AI score ({ai_score:.3f}) with high prefilter score ({prefilter_score:.3f})",
                self.thresholds['confidence_penalty_factor']
            )

        # All checks passed
        return (
            ValidationOutcome.PASS,
            "Scores are consistent",
            1.0  # No confidence adjustment for passing validation
        )

    def _is_critical_mismatch(self, ai_score: float, prefilter_score: float) -> bool:
        """Check for critical mismatches that indicate serious problems.

        Args:
            ai_score: AI relevance score
            prefilter_score: Pre-filter relevance score

        Returns:
            True if this is a critical mismatch
        """
        # Very high AI score with very low prefilter score
        if (ai_score > self.settings.trust_validation.critical_high_ai_threshold and 
            prefilter_score < self.settings.trust_validation.critical_low_prefilter_threshold):
            return True

        # Very low AI score with very high prefilter score
        if (ai_score < self.settings.trust_validation.critical_low_ai_threshold and 
            prefilter_score > self.settings.trust_validation.critical_high_prefilter_threshold):
            return True

        # Impossible scores
        if ai_score > 1.0 or ai_score < 0.0:
            return True

        if prefilter_score > 1.0 or prefilter_score < 0.0:
            return True

        return False

    def validate_batch_consistency(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Validate consistency across a batch of validation results.

        Args:
            validation_results: List of validation results to analyze

        Returns:
            Dictionary with batch consistency analysis
        """
        if not validation_results:
            return {
                'consistent': True,
                'reason': 'No results to validate',
                'recommendations': []
            }

        # Calculate consistency metrics
        pass_rate = len([r for r in validation_results if r.outcome == ValidationOutcome.PASS]) / len(validation_results)
        fail_rate = len([r for r in validation_results if r.outcome == ValidationOutcome.FAIL]) / len(validation_results)
        warning_rate = len([r for r in validation_results if r.outcome == ValidationOutcome.WARNING]) / len(validation_results)

        # Analyze score differences
        score_differences = [r.score_difference for r in validation_results if r.outcome != ValidationOutcome.SKIP]
        avg_score_difference = sum(score_differences) / len(score_differences) if score_differences else 0.0

        # Provider analysis
        provider_outcomes = {}
        for result in validation_results:
            provider = result.provider
            if provider not in provider_outcomes:
                provider_outcomes[provider] = {'pass': 0, 'fail': 0, 'warning': 0, 'total': 0}

            provider_outcomes[provider][result.outcome.value] += 1
            provider_outcomes[provider]['total'] += 1

        # Determine overall consistency
        consistent = (
            pass_rate >= self.settings.trust_validation.min_pass_rate and  # Configurable min pass rate
            fail_rate <= self.settings.trust_validation.max_fail_rate and  # Configurable max fail rate
            avg_score_difference <= self.thresholds['warning_score_difference']
        )

        # Generate recommendations
        recommendations = []
        if pass_rate < self.settings.trust_validation.min_pass_rate:
            recommendations.append("Low validation pass rate - review AI provider configuration")
        if fail_rate > self.settings.trust_validation.max_fail_rate:
            recommendations.append("High validation fail rate - investigate AI result quality")
        if avg_score_difference > self.thresholds['warning_score_difference']:
            recommendations.append("High average score difference - check scoring consistency")

        # Provider-specific recommendations
        for provider, outcomes in provider_outcomes.items():
            if outcomes['total'] >= 5:  # Only analyze providers with sufficient data
                provider_fail_rate = outcomes['fail'] / outcomes['total']
                if provider_fail_rate > self.settings.trust_validation.provider_fail_rate_threshold:
                    recommendations.append(f"Provider '{provider}' has high fail rate ({provider_fail_rate:.1%})")

        return {
            'consistent': consistent,
            'pass_rate': pass_rate,
            'fail_rate': fail_rate,
            'warning_rate': warning_rate,
            'avg_score_difference': avg_score_difference,
            'provider_analysis': provider_outcomes,
            'recommendations': recommendations,
            'total_results': len(validation_results)
        }

    def get_quality_score(self, validation_result: ValidationResult) -> float:
        """Calculate quality score for a validation result.

        Args:
            validation_result: Validation result to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        base_score = 1.0

        # Reduce score based on outcome
        if validation_result.outcome == ValidationOutcome.FAIL:
            base_score *= 0.3
        elif validation_result.outcome == ValidationOutcome.WARNING:
            base_score *= 0.7
        elif validation_result.outcome == ValidationOutcome.SKIP:
            base_score *= 0.1

        # Apply confidence adjustment
        base_score *= validation_result.confidence_adjustment

        # Apply provider quality factor
        provider_quality = self.provider_quality_factors.get(
            validation_result.provider.lower(), 0.5
        )
        base_score *= provider_quality

        return max(0.0, min(1.0, base_score))

    def should_deliver_content(self, validation_result: ValidationResult,
                             min_quality_threshold: float = 0.4) -> bool:
        """Determine if content should be delivered based on validation.

        Args:
            validation_result: Validation result to evaluate
            min_quality_threshold: Minimum quality threshold for delivery

        Returns:
            True if content should be delivered
        """
        if validation_result.outcome == ValidationOutcome.SKIP:
            return False

        quality_score = self.get_quality_score(validation_result)
        return quality_score >= min_quality_threshold

    def get_delivery_category(self, validation_result: ValidationResult) -> str:
        """Get delivery category based on validation quality.

        Args:
            validation_result: Validation result to categorize

        Returns:
            Category string for message formatting
        """
        quality_score = self.get_quality_score(validation_result)

        if quality_score >= 0.8:
            return "high_quality"
        elif quality_score >= 0.6:
            return "medium_quality"
        elif quality_score >= 0.4:
            return "low_quality"
        else:
            return "questionable_quality"