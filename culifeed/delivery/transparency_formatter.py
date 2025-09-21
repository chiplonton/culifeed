"""
Transparency Formatter
=====================

Provides transparent formatting for AI processing results, including provider
information, confidence levels, and quality indicators in delivered messages.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..database.models import Article
from ..utils.logging import get_logger_for_component


class QualityIndicator(str, Enum):
    """Quality indicators for AI processing results."""
    HIGH_QUALITY = "🎯"      # High confidence, validated result
    GOOD_QUALITY = "✅"      # Good confidence, minor validation concerns
    MODERATE_QUALITY = "⚠️"  # Moderate confidence, validation warnings
    LOW_QUALITY = "🔍"       # Low confidence, validation issues
    FALLBACK_QUALITY = "📊"  # Keyword-based fallback analysis


class ProviderIcon(str, Enum):
    """Icons for different AI providers."""
    GROQ = "🚀"
    GEMINI = "🧠"
    OPENAI = "🤖"
    HUGGINGFACE = "🤗"
    OPENROUTER = "🔄"
    KEYWORD_FALLBACK = "📊"
    UNKNOWN = "❓"


@dataclass
class TransparencyInfo:
    """Transparency information for an article analysis."""
    provider_name: str
    provider_icon: str
    confidence: float
    quality_indicator: str
    validation_status: Optional[str] = None
    processing_method: str = "AI Analysis"
    quality_note: Optional[str] = None

    def get_transparency_text(self, include_details: bool = True) -> str:
        """Get formatted transparency text for message display.

        Args:
            include_details: Whether to include detailed information

        Returns:
            Formatted transparency string
        """
        # Basic provider and quality info
        base_text = f"{self.quality_indicator} {self.provider_icon} {self.provider_name}"

        if not include_details:
            return base_text

        # Add confidence if available
        if self.confidence > 0:
            confidence_pct = int(self.confidence * 100)
            base_text += f" ({confidence_pct}%)"

        # Add quality note if available
        if self.quality_note:
            base_text += f" - {self.quality_note}"

        return base_text

    def get_detailed_info(self) -> str:
        """Get detailed transparency information for debugging/admin views."""
        details = [
            f"Provider: {self.provider_name}",
            f"Processing: {self.processing_method}",
            f"Confidence: {self.confidence:.3f}"
        ]

        if self.validation_status:
            details.append(f"Validation: {self.validation_status}")

        if self.quality_note:
            details.append(f"Note: {self.quality_note}")

        return " | ".join(details)


class TransparencyFormatter:
    """Formats transparency information for AI processing results."""

    def __init__(self, settings: Optional['CuliFeedSettings'] = None):
        """Initialize transparency formatter.
        
        Args:
            settings: CuliFeed settings with configurable thresholds
        """
        self.logger = get_logger_for_component("transparency_formatter")

        # Import here to avoid circular imports
        if settings is None:
            from ..config.settings import get_settings
            settings = get_settings()
        
        self.settings = settings

        # Provider mapping
        self.provider_mapping = {
            'groq': ('Groq', ProviderIcon.GROQ),
            'gemini': ('Gemini', ProviderIcon.GEMINI),
            'openai': ('OpenAI', ProviderIcon.OPENAI),
            'huggingface': ('HuggingFace', ProviderIcon.HUGGINGFACE),
            'openrouter': ('OpenRouter', ProviderIcon.OPENROUTER),
            'keyword_fallback': ('Keywords', ProviderIcon.KEYWORD_FALLBACK),
            'keyword_backup': ('Keywords', ProviderIcon.KEYWORD_FALLBACK),
        }

        # Quality thresholds for indicators (now configurable)
        self.quality_thresholds = {
            'high': settings.delivery_quality.high_quality_threshold,
            'good': settings.delivery_quality.good_quality_threshold,
            'moderate': settings.delivery_quality.moderate_quality_threshold,
            'low': settings.delivery_quality.low_quality_threshold
        }

    def get_transparency_info(self, article: Article) -> TransparencyInfo:
        """Get transparency information for an article.

        Args:
            article: Article with AI processing results

        Returns:
            TransparencyInfo object with formatted transparency data
        """
        # Extract provider information
        provider_raw = getattr(article, 'ai_provider', 'unknown') or 'unknown'
        provider_name, provider_icon = self._get_provider_info(provider_raw)

        # Get confidence
        confidence = getattr(article, 'ai_confidence', 0.0) or 0.0

        # Determine quality indicator and note
        quality_indicator, quality_note = self._determine_quality_indicator(
            confidence, provider_raw, article
        )

        # Get validation status
        validation_status = getattr(article, 'validation_outcome', None)

        # Determine processing method
        processing_method = self._get_processing_method(provider_raw, validation_status)

        return TransparencyInfo(
            provider_name=provider_name,
            provider_icon=provider_icon.value,
            confidence=confidence,
            quality_indicator=quality_indicator.value,
            validation_status=validation_status,
            processing_method=processing_method,
            quality_note=quality_note
        )

    def _get_provider_info(self, provider_raw: str) -> tuple[str, ProviderIcon]:
        """Get provider name and icon from raw provider string.

        Args:
            provider_raw: Raw provider string (e.g., "groq/llama-3.1-70b")

        Returns:
            Tuple of (provider_name, provider_icon)
        """
        # Extract base provider name
        provider_base = provider_raw.lower().split('/')[0] if '/' in provider_raw else provider_raw.lower()

        # Handle special cases
        if 'keyword' in provider_base or 'fallback' in provider_base:
            return self.provider_mapping.get('keyword_fallback', ('Keywords', ProviderIcon.KEYWORD_FALLBACK))

        # Look up in mapping
        if provider_base in self.provider_mapping:
            return self.provider_mapping[provider_base]

        # Default for unknown providers
        return ('Unknown', ProviderIcon.UNKNOWN)

    def _determine_quality_indicator(self, confidence: float, provider_raw: str,
                                   article: Article) -> tuple[QualityIndicator, Optional[str]]:
        """Determine quality indicator and note based on confidence and validation.

        Args:
            confidence: AI confidence score
            provider_raw: Raw provider string
            article: Article with processing results

        Returns:
            Tuple of (quality_indicator, quality_note)
        """
        # Check for fallback methods
        if 'keyword' in provider_raw.lower() or 'fallback' in provider_raw.lower():
            return QualityIndicator.FALLBACK_QUALITY, "Keyword analysis"

        # Check validation outcome if available
        validation_outcome = getattr(article, 'validation_outcome', None)
        if validation_outcome == 'fail':
            return QualityIndicator.LOW_QUALITY, "Validation concerns"
        elif validation_outcome == 'warning':
            return QualityIndicator.MODERATE_QUALITY, "Minor validation issues"

        # Determine quality based on confidence
        if confidence >= self.quality_thresholds['high']:
            return QualityIndicator.HIGH_QUALITY, "High confidence"
        elif confidence >= self.quality_thresholds['good']:
            return QualityIndicator.GOOD_QUALITY, "Good confidence"
        elif confidence >= self.quality_thresholds['moderate']:
            return QualityIndicator.MODERATE_QUALITY, "Moderate confidence"
        else:
            return QualityIndicator.LOW_QUALITY, "Low confidence"

    def _get_processing_method(self, provider_raw: str, validation_status: Optional[str]) -> str:
        """Get processing method description.

        Args:
            provider_raw: Raw provider string
            validation_status: Validation outcome

        Returns:
            Processing method description
        """
        if 'keyword' in provider_raw.lower() or 'fallback' in provider_raw.lower():
            return "Keyword Analysis"

        method = "AI Analysis"
        if validation_status:
            method += f" (Validated)"

        return method

    def format_article_with_transparency(self, article: Article, title_format: str = "**{title}**",
                                       include_transparency: bool = True,
                                       include_summary: bool = True,
                                       include_url: bool = True) -> str:
        """Format an article with transparency information.

        Args:
            article: Article to format
            title_format: Format string for article title
            include_transparency: Whether to include transparency info
            include_summary: Whether to include article summary
            include_url: Whether to include article URL

        Returns:
            Formatted article string
        """
        lines = []

        # Article title
        title = title_format.format(title=article.title)
        lines.append(title)

        # Transparency information
        if include_transparency:
            transparency_info = self.get_transparency_info(article)
            transparency_text = transparency_info.get_transparency_text(include_details=True)
            lines.append(f"*{transparency_text}*")

        # Article summary
        if include_summary and hasattr(article, 'summary') and article.summary:
            lines.append(f"{article.summary}")

        # Article URL
        if include_url and article.url:
            lines.append(f"🔗 [Read more]({article.url})")

        return "\n".join(lines)

    def format_topic_section(self, topic_name: str, articles: List[Article],
                           max_articles: int = 5) -> str:
        """Format a topic section with articles and transparency.

        Args:
            topic_name: Name of the topic
            articles: List of articles for this topic
            max_articles: Maximum articles to include

        Returns:
            Formatted topic section
        """
        if not articles:
            return ""

        lines = [f"## 📌 {topic_name}"]

        # Sort articles by AI confidence (highest first)
        sorted_articles = sorted(
            articles[:max_articles],
            key=lambda a: getattr(a, 'ai_confidence', 0.0),
            reverse=True
        )

        for i, article in enumerate(sorted_articles, 1):
            article_text = self.format_article_with_transparency(
                article,
                title_format=f"{i}. **{{title}}**",
                include_transparency=True,
                include_summary=True,
                include_url=True
            )
            lines.append(article_text)
            lines.append("")  # Empty line between articles

        return "\n".join(lines)

    def get_quality_summary(self, articles: List[Article]) -> Dict[str, Any]:
        """Get quality summary for a list of articles.

        Args:
            articles: List of articles to analyze

        Returns:
            Dictionary with quality summary statistics
        """
        if not articles:
            return {
                'total_articles': 0,
                'quality_distribution': {},
                'average_confidence': 0.0,
                'providers_used': {},
                'validation_summary': {}
            }

        # Analyze quality distribution
        quality_counts = {}
        confidence_scores = []
        provider_counts = {}
        validation_counts = {}

        for article in articles:
            # Get transparency info
            transparency_info = self.get_transparency_info(article)

            # Count quality indicators
            quality_key = transparency_info.quality_indicator
            quality_counts[quality_key] = quality_counts.get(quality_key, 0) + 1

            # Collect confidence scores
            if transparency_info.confidence > 0:
                confidence_scores.append(transparency_info.confidence)

            # Count providers
            provider = transparency_info.provider_name
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

            # Count validation outcomes
            validation = transparency_info.validation_status or 'none'
            validation_counts[validation] = validation_counts.get(validation, 0) + 1

        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return {
            'total_articles': len(articles),
            'quality_distribution': quality_counts,
            'average_confidence': avg_confidence,
            'providers_used': provider_counts,
            'validation_summary': validation_counts
        }

    def format_quality_footer(self, articles: List[Article]) -> str:
        """Format quality summary footer for message.

        Args:
            articles: List of articles in the message

        Returns:
            Formatted quality footer
        """
        summary = self.get_quality_summary(articles)

        if summary['total_articles'] == 0:
            return ""

        lines = ["---", "📊 **Quality Summary**"]

        # Quality distribution
        quality_dist = summary['quality_distribution']
        if quality_dist:
            quality_text = []
            for indicator, count in quality_dist.items():
                quality_text.append(f"{indicator} {count}")
            lines.append(f"Quality: {' | '.join(quality_text)}")

        # Average confidence
        if summary['average_confidence'] > 0:
            avg_conf_pct = int(summary['average_confidence'] * 100)
            lines.append(f"Average Confidence: {avg_conf_pct}%")

        # Providers used
        providers = summary['providers_used']
        if len(providers) <= 3:
            provider_text = ', '.join(providers.keys())
            lines.append(f"Analyzed by: {provider_text}")

        return "\n".join(lines)