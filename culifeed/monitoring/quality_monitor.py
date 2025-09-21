"""
Quality Monitoring System
========================

Monitors AI processing quality, tracks consistency metrics, and provides alerting
for trust issues in the content processing pipeline.
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..utils.logging import get_logger_for_component


class AlertLevel(str, Enum):
    """Quality alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityAlert:
    """Quality monitoring alert."""
    level: AlertLevel
    message: str
    component: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.component}: {self.message}"


@dataclass
class QualityMetrics:
    """Quality metrics for AI processing."""
    # Cross-validation metrics
    validation_attempts: int = 0
    validation_passes: int = 0
    validation_failures: int = 0
    avg_score_difference: float = 0.0

    # Provider reliability metrics
    provider_success_rate: Dict[str, float] = field(default_factory=dict)
    provider_avg_confidence: Dict[str, float] = field(default_factory=dict)
    provider_consistency: Dict[str, float] = field(default_factory=dict)

    # Processing quality metrics
    ai_processing_success_rate: float = 0.0
    keyword_fallback_rate: float = 0.0
    silent_failure_rate: float = 0.0

    # Performance metrics
    avg_processing_time_ms: float = 0.0
    processing_timeout_rate: float = 0.0

    @property
    def validation_success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.validation_attempts == 0:
            return 0.0
        return self.validation_passes / self.validation_attempts

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score (0.0 to 1.0)."""
        scores = [
            self.validation_success_rate,
            self.ai_processing_success_rate,
            1.0 - self.silent_failure_rate,  # Invert silent failure rate
            1.0 - self.keyword_fallback_rate * 0.5  # Fallback reduces quality but isn't terrible
        ]
        return sum(scores) / len(scores)


class QualityMonitor:
    """Quality monitoring system for AI processing pipeline."""

    def __init__(self, settings: Optional['CuliFeedSettings'] = None):
        """Initialize quality monitor.

        Args:
            settings: CuliFeed settings with configurable thresholds
        """
        self.logger = get_logger_for_component("quality_monitor")

        # Import here to avoid circular imports
        if settings is None:
            from ..config.settings import get_settings
            settings = get_settings()
        
        self.settings = settings

        # Use configurable alert thresholds from settings
        self.alert_thresholds = {
            'validation_success_rate_min': settings.quality_monitoring.validation_success_rate_min,
            'score_difference_max': settings.quality_monitoring.score_difference_max,
            'silent_failure_rate_max': settings.quality_monitoring.silent_failure_rate_max,
            'provider_consistency_min': settings.quality_monitoring.provider_consistency_min,
            'overall_quality_min': settings.quality_monitoring.overall_quality_min,
        }

        # Tracking state
        self._metrics = QualityMetrics()
        self._recent_validations: List[Dict[str, Any]] = []
        self._recent_processing: List[Dict[str, Any]] = []
        self._alerts: List[QualityAlert] = []

        # Time windows for metrics calculation
        self._metrics_window_hours = 24
        self._max_stored_events = 1000

        self.logger.info("Quality monitor initialized with configurable thresholds", 
                        extra={'thresholds': self.alert_thresholds})

    def record_validation_attempt(self, ai_score: float, prefilter_score: float,
                                 provider: str, success: bool, reason: Optional[str] = None) -> None:
        """Record a cross-validation attempt.

        Args:
            ai_score: AI relevance score
            prefilter_score: Pre-filter relevance score
            provider: AI provider used
            success: Whether validation passed
            reason: Reason for failure if applicable
        """
        validation_event = {
            'timestamp': datetime.now(timezone.utc),
            'ai_score': ai_score,
            'prefilter_score': prefilter_score,
            'provider': provider,
            'success': success,
            'score_difference': abs(ai_score - prefilter_score),
            'reason': reason
        }

        self._recent_validations.append(validation_event)
        self._cleanup_old_events()

        # Update metrics
        self._update_validation_metrics()

        # Check for alerts
        if not success:
            self._check_validation_alerts(validation_event)

        self.logger.debug(
            f"Validation {'passed' if success else 'failed'}: "
            f"AI={ai_score:.3f}, Prefilter={prefilter_score:.3f}, "
            f"Provider={provider}, Diff={validation_event['score_difference']:.3f}"
        )

    def record_processing_attempt(self, article_id: str, provider: str,
                                success: bool, processing_time_ms: float,
                                ai_result: Optional[Any] = None,
                                used_fallback: bool = False) -> None:
        """Record an AI processing attempt.

        Args:
            article_id: Article being processed
            provider: AI provider used
            success: Whether processing succeeded
            processing_time_ms: Processing time in milliseconds
            ai_result: AI processing result if available
            used_fallback: Whether keyword fallback was used
        """
        processing_event = {
            'timestamp': datetime.now(timezone.utc),
            'article_id': article_id,
            'provider': provider,
            'success': success,
            'processing_time_ms': processing_time_ms,
            'confidence': ai_result.confidence if ai_result else None,
            'relevance_score': ai_result.relevance_score if ai_result else None,
            'used_fallback': used_fallback
        }

        self._recent_processing.append(processing_event)
        self._cleanup_old_events()

        # Update metrics
        self._update_processing_metrics()

        # Check for alerts
        self._check_processing_alerts(processing_event)

        self.logger.debug(
            f"Processing {'succeeded' if success else 'failed'}: "
            f"Article={article_id}, Provider={provider}, "
            f"Time={processing_time_ms:.1f}ms, Fallback={used_fallback}"
        )

    def get_current_metrics(self) -> QualityMetrics:
        """Get current quality metrics.

        Returns:
            Current QualityMetrics object
        """
        # Refresh metrics before returning
        self._update_validation_metrics()
        self._update_processing_metrics()

        return self._metrics

    def get_recent_alerts(self, hours: int = 24) -> List[QualityAlert]:
        """Get recent quality alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent QualityAlert objects
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [alert for alert in self._alerts if alert.timestamp >= cutoff]

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        self._alerts.clear()
        self.logger.info("Quality monitor alerts cleared")

    def _update_validation_metrics(self) -> None:
        """Update validation-related metrics."""
        if not self._recent_validations:
            return

        recent_window = self._get_recent_events(self._recent_validations)

        self._metrics.validation_attempts = len(recent_window)
        self._metrics.validation_passes = sum(1 for v in recent_window if v['success'])
        self._metrics.validation_failures = self._metrics.validation_attempts - self._metrics.validation_passes

        if recent_window:
            self._metrics.avg_score_difference = statistics.mean(v['score_difference'] for v in recent_window)

    def _update_processing_metrics(self) -> None:
        """Update processing-related metrics."""
        if not self._recent_processing:
            return

        recent_window = self._get_recent_events(self._recent_processing)

        total_attempts = len(recent_window)
        if total_attempts == 0:
            return

        # Processing success metrics
        successful = [p for p in recent_window if p['success']]
        self._metrics.ai_processing_success_rate = len(successful) / total_attempts

        # Fallback usage metrics
        fallback_used = sum(1 for p in recent_window if p['used_fallback'])
        self._metrics.keyword_fallback_rate = fallback_used / total_attempts

        # Silent failure detection (failed processing without fallback)
        silent_failures = sum(1 for p in recent_window if not p['success'] and not p['used_fallback'])
        self._metrics.silent_failure_rate = silent_failures / total_attempts

        # Performance metrics
        if recent_window:
            self._metrics.avg_processing_time_ms = statistics.mean(p['processing_time_ms'] for p in recent_window)

        # Provider-specific metrics
        self._update_provider_metrics(recent_window)

    def _update_provider_metrics(self, recent_processing: List[Dict[str, Any]]) -> None:
        """Update provider-specific metrics."""
        provider_stats = {}

        for event in recent_processing:
            provider = event['provider']
            if provider not in provider_stats:
                provider_stats[provider] = {
                    'attempts': 0,
                    'successes': 0,
                    'confidences': [],
                    'relevance_scores': []
                }

            stats = provider_stats[provider]
            stats['attempts'] += 1

            if event['success']:
                stats['successes'] += 1
                if event['confidence'] is not None:
                    stats['confidences'].append(event['confidence'])
                if event['relevance_score'] is not None:
                    stats['relevance_scores'].append(event['relevance_score'])

        # Calculate provider metrics
        for provider, stats in provider_stats.items():
            # Success rate
            self._metrics.provider_success_rate[provider] = stats['successes'] / stats['attempts']

            # Average confidence
            if stats['confidences']:
                self._metrics.provider_avg_confidence[provider] = statistics.mean(stats['confidences'])

            # Consistency (inverse of standard deviation)
            if len(stats['relevance_scores']) > 1:
                std_dev = statistics.stdev(stats['relevance_scores'])
                self._metrics.provider_consistency[provider] = max(0.0, 1.0 - std_dev)
            elif stats['relevance_scores']:
                self._metrics.provider_consistency[provider] = 1.0  # Single score is consistent

    def _get_recent_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get events within the metrics time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._metrics_window_hours)
        return [event for event in events if event['timestamp'] >= cutoff]

    def _cleanup_old_events(self) -> None:
        """Remove old events to prevent memory growth."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._metrics_window_hours * 2)

        self._recent_validations = [v for v in self._recent_validations if v['timestamp'] >= cutoff]
        self._recent_processing = [p for p in self._recent_processing if p['timestamp'] >= cutoff]
        self._alerts = [a for a in self._alerts if a.timestamp >= cutoff]

        # Also limit by count to prevent memory issues
        if len(self._recent_validations) > self._max_stored_events:
            self._recent_validations = self._recent_validations[-self._max_stored_events:]

        if len(self._recent_processing) > self._max_stored_events:
            self._recent_processing = self._recent_processing[-self._max_stored_events:]

    def _check_validation_alerts(self, validation_event: Dict[str, Any]) -> None:
        """Check for validation-related quality alerts."""
        score_diff = validation_event['score_difference']

        if score_diff > self.alert_thresholds['score_difference_max']:
            alert = QualityAlert(
                level=AlertLevel.WARNING,
                message=f"Large score difference detected: AI={validation_event['ai_score']:.3f}, "
                       f"Prefilter={validation_event['prefilter_score']:.3f} (diff={score_diff:.3f})",
                component="cross_validation",
                metadata={
                    'ai_score': validation_event['ai_score'],
                    'prefilter_score': validation_event['prefilter_score'],
                    'provider': validation_event['provider'],
                    'score_difference': score_diff
                }
            )
            self._alerts.append(alert)
            self.logger.warning(str(alert))

    def _check_processing_alerts(self, processing_event: Dict[str, Any]) -> None:
        """Check for processing-related quality alerts."""
        # Check for overall quality degradation
        current_metrics = self.get_current_metrics()

        if current_metrics.overall_quality_score < self.alert_thresholds['overall_quality_min']:
            alert = QualityAlert(
                level=AlertLevel.ERROR,
                message=f"Overall quality score below threshold: "
                       f"{current_metrics.overall_quality_score:.3f} < "
                       f"{self.alert_thresholds['overall_quality_min']}",
                component="overall_quality",
                metadata={'quality_score': current_metrics.overall_quality_score}
            )
            self._alerts.append(alert)
            self.logger.error(str(alert))

        # Check for high silent failure rate
        if current_metrics.silent_failure_rate > self.alert_thresholds['silent_failure_rate_max']:
            alert = QualityAlert(
                level=AlertLevel.CRITICAL,
                message=f"Silent failure rate too high: "
                       f"{current_metrics.silent_failure_rate:.3f} > "
                       f"{self.alert_thresholds['silent_failure_rate_max']}",
                component="silent_failures",
                metadata={'silent_failure_rate': current_metrics.silent_failure_rate}
            )
            self._alerts.append(alert)
            self.logger.critical(str(alert))

    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report.

        Returns:
            Dictionary containing quality report data
        """
        metrics = self.get_current_metrics()
        recent_alerts = self.get_recent_alerts(hours=24)

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': {
                'validation_success_rate': metrics.validation_success_rate,
                'ai_processing_success_rate': metrics.ai_processing_success_rate,
                'silent_failure_rate': metrics.silent_failure_rate,
                'keyword_fallback_rate': metrics.keyword_fallback_rate,
                'overall_quality_score': metrics.overall_quality_score,
                'avg_score_difference': metrics.avg_score_difference,
                'avg_processing_time_ms': metrics.avg_processing_time_ms
            },
            'provider_metrics': {
                'success_rates': dict(metrics.provider_success_rate),
                'avg_confidence': dict(metrics.provider_avg_confidence),
                'consistency': dict(metrics.provider_consistency)
            },
            'alerts': {
                'total_count': len(recent_alerts),
                'by_level': {
                    level.value: len([a for a in recent_alerts if a.level == level])
                    for level in AlertLevel
                },
                'recent_alerts': [
                    {
                        'level': alert.level.value,
                        'message': alert.message,
                        'component': alert.component,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in recent_alerts[-10:]  # Last 10 alerts
                ]
            },
            'recommendations': self._generate_recommendations(metrics)
        }

    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate recommendations based on current metrics."""
        recommendations = []

        if metrics.validation_success_rate < self.settings.quality_monitoring.high_validation_threshold:
            recommendations.append(
                "Consider reviewing AI provider configuration - low validation success rate"
            )

        if metrics.silent_failure_rate > self.settings.quality_monitoring.low_silent_failure_threshold:
            recommendations.append(
                "Implement better error handling - silent failures detected"
            )

        if metrics.keyword_fallback_rate > self.settings.quality_monitoring.high_fallback_rate_threshold:
            recommendations.append(
                "High fallback usage suggests AI provider reliability issues"
            )

        if metrics.avg_processing_time_ms > 5000:
            recommendations.append(
                "Processing times are high - consider optimization or provider changes"
            )

        if not recommendations:
            recommendations.append("Quality metrics are within acceptable ranges")

        return recommendations