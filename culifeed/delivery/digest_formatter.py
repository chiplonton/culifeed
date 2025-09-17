"""
Digest Formatter
================

Advanced message formatting for CuliFeed content delivery.
Handles various message formats, templates, and presentation styles.

Features:
- Multiple digest formats (compact, detailed, summary)
- Telegram-optimized message formatting
- Template-based message generation
- Content truncation and optimization
"""

import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from ..database.models import Article, Topic
from ..config.settings import get_settings
from ..utils.logging import get_logger_for_component


class DigestFormat(str, Enum):
    """Available digest formats."""
    COMPACT = "compact"
    DETAILED = "detailed"
    SUMMARY = "summary"
    HEADLINES = "headlines"


class DigestFormatter:
    """Formats curated content into various digest styles."""

    def __init__(self):
        """Initialize digest formatter."""
        self.settings = get_settings()
        self.logger = get_logger_for_component('digest_formatter')

        # Telegram formatting limits
        self.max_message_length = 4096
        self.max_caption_length = 1024

        # Content limits per format
        self.format_limits = {
            DigestFormat.COMPACT: {'articles_per_topic': 3, 'title_length': 120, 'summary_length': 100},
            DigestFormat.DETAILED: {'articles_per_topic': 5, 'title_length': 120, 'summary_length': 200},
            DigestFormat.SUMMARY: {'articles_per_topic': 8, 'title_length': 120, 'summary_length': 0},
            DigestFormat.HEADLINES: {'articles_per_topic': 10, 'title_length': 120, 'summary_length': 0}
        }

    def format_daily_digest(self, articles_by_topic: Dict[str, List[Article]],
                          format_type: DigestFormat = DigestFormat.DETAILED) -> List[str]:
        """Format articles into a daily digest.

        Args:
            articles_by_topic: Articles organized by topic
            format_type: Desired format style

        Returns:
            List of formatted message strings
        """
        if not articles_by_topic:
            return [self._format_empty_digest()]

        messages = []
        limits = self.format_limits[format_type]

        # Add header message
        header = self._format_digest_header(articles_by_topic, format_type)
        messages.append(header)

        # Format each topic
        for topic_name, articles in articles_by_topic.items():
            topic_messages = self._format_topic_section(
                topic_name, articles, format_type, limits
            )
            messages.extend(topic_messages)

        # Add footer if needed
        footer = self._format_digest_footer(articles_by_topic, format_type)
        if footer:
            messages.append(footer)

        return messages

    def format_topic_preview(self, topic_name: str, articles: List[Article],
                           limit: int = 3) -> str:
        """Format a preview of articles for a specific topic.

        Args:
            topic_name: Name of the topic
            articles: List of articles
            limit: Maximum articles to include

        Returns:
            Formatted preview message
        """
        if not articles:
            return f"🎯 *{topic_name}*\n\nNo recent articles found."

        preview_articles = articles[:limit]

        message = f"🎯 *{topic_name}* Preview\n\n"

        for i, article in enumerate(preview_articles, 1):
            title = self._truncate_text(article.title, 70)
            pub_date = ""
            if article.published_at:
                pub_date = f" • {article.published_at.strftime('%m/%d')}"

            message += f"*{i}.* {title}{pub_date}\n"

        if len(articles) > limit:
            message += f"\n... and {len(articles) - limit} more articles"

        return message

    def format_article_summary(self, article: Article, include_content: bool = True) -> str:
        """Format a single article for detailed display.

        Args:
            article: Article to format
            include_content: Whether to include content summary

        Returns:
            Formatted article message
        """
        message = f"📄 *{article.title}*\n\n"

        # Add content/summary if available and requested
        if include_content and article.content:
            content_preview = self._extract_content_preview(article.content)
            if content_preview:
                message += f"{content_preview}\n\n"

        # Add metadata
        if article.published_at:
            pub_date = article.published_at.strftime("%B %d, %Y at %H:%M")
            message += f"📅 *Published:* {pub_date}\n"

        # Add source info
        source_name = self._extract_source_name(article.source_feed)
        message += f"📡 *Source:* {source_name}\n\n"

        # Add read link
        message += f"🔗 [Read Full Article]({article.url})"

        return message

    def _format_digest_header(self, articles_by_topic: Dict[str, List[Article]],
                             format_type: DigestFormat) -> str:
        """Format header for daily digest.

        Args:
            articles_by_topic: Articles by topic
            format_type: Digest format

        Returns:
            Formatted header message
        """
        total_articles = sum(len(articles) for articles in articles_by_topic.values())
        topic_count = len(articles_by_topic)

        today = datetime.now(timezone.utc).strftime("%B %d, %Y")

        # Get simplified date
        today_simple = "Today" if datetime.now(timezone.utc).date() == datetime.now(timezone.utc).date() else today

        # Format varies by type
        if format_type == DigestFormat.COMPACT:
            header = (
                f"🌟 *Your Daily Brief*\n"
                f"📅 {today_simple} • {total_articles} fresh articles\n\n"
            )
        elif format_type == DigestFormat.HEADLINES:
            header = (
                f"📈 *Top Headlines*\n"
                f"📅 {today_simple} • {topic_count} topics\n\n"
            )
        else:  # DETAILED or SUMMARY
            reading_time = max(1, total_articles * 0.5)
            header = (
                f"🌟 *Your Daily Tech Digest*\n"
                f"📅 {today_simple} • {today}\n\n"
                f"🎯 *{total_articles}* curated articles from *{topic_count}* topic{'s' if topic_count != 1 else ''}\n"
                f"⏱️ ~{reading_time:.0f} min read • 📡 Fresh content\n\n"
            )

        # Add topic overview for detailed format
        if format_type == DigestFormat.DETAILED:
            for topic_name, articles in articles_by_topic.items():
                header += f"• *{topic_name}*: {len(articles)} articles\n"
            header += "\n"

        return header.rstrip()

    def _format_topic_section(self, topic_name: str, articles: List[Article],
                             format_type: DigestFormat, limits: Dict[str, int]) -> List[str]:
        """Format articles for a topic section.

        Args:
            topic_name: Name of the topic
            articles: List of articles
            format_type: Digest format
            limits: Format-specific limits

        Returns:
            List of formatted messages for this topic
        """
        messages = []
        article_limit = limits['articles_per_topic']
        selected_articles = articles[:article_limit]

        if format_type == DigestFormat.HEADLINES:
            # Compact headlines format with better bullets
            message = f"🎯 *{topic_name}* ({len(selected_articles)} articles)\n\n"
            for i, article in enumerate(selected_articles, 1):
                title = self._truncate_text(article.title, limits['title_length'])
                message += f"▶️ {title}\n"

            if len(articles) > article_limit:
                message += f"\n💫 ... and {len(articles) - article_limit} more articles\n"

            message += "\n━━━━━━━━━━━━━━━━━━━━\n"
            messages.append(message)

        elif format_type == DigestFormat.SUMMARY:
            # Title-only format
            message = f"🎯 *{topic_name}*\n\n"
            for i, article in enumerate(selected_articles, 1):
                title = self._truncate_text(article.title, limits['title_length'])
                pub_info = ""
                if article.published_at:
                    pub_info = f" • {article.published_at.strftime('%m/%d')}"
                message += f"*{i}.* {title}{pub_info}\n"

            messages.append(message)

        else:  # COMPACT or DETAILED
            current_message = f"🎯 *{topic_name}*\n\n"

            for i, article in enumerate(selected_articles, 1):
                article_text = self._format_article_item(
                    article, i, format_type, limits
                )

                # Check message length
                if len(current_message + article_text) > self.max_message_length:
                    messages.append(current_message.rstrip())
                    current_message = f"🎯 *{topic_name}* (continued)\n\n" + article_text
                else:
                    current_message += article_text

            if current_message.strip():
                messages.append(current_message.rstrip())

        return messages

    def _format_article_item(self, article: Article, index: int,
                           format_type: DigestFormat, limits: Dict[str, int]) -> str:
        """Format a single article item.

        Args:
            article: Article to format
            index: Article index number
            format_type: Digest format
            limits: Format-specific limits

        Returns:
            Formatted article text
        """
        title = self._truncate_text(article.title, limits['title_length'])
        article_text = f"*{index}. {title}*\n\n"

        # Add summary for detailed format with improved icon
        if format_type == DigestFormat.DETAILED and limits['summary_length'] > 0:
            if hasattr(article, 'summary') and article.summary:
                summary = self._truncate_text(article.summary, limits['summary_length'])
                article_text += f"💡 {summary}\n\n"
            elif article.content:
                content_preview = self._extract_content_preview(
                    article.content, limits['summary_length']
                )
                if content_preview:
                    article_text += f"💡 {content_preview}\n\n"

        # Add metadata with improved formatting
        metadata_parts = []

        if article.published_at:
            # Use "Today" for today's articles, otherwise short format
            today = datetime.now(timezone.utc).date()
            article_date = article.published_at.date()
            if article_date == today:
                time_str = article.published_at.strftime("%H:%M")
                metadata_parts.append(f"🕒 Today {time_str}")
            else:
                pub_date = article.published_at.strftime("%m/%d %H:%M")
                metadata_parts.append(f"🕒 {pub_date}")

        # Add source for detailed format
        if format_type == DigestFormat.DETAILED:
            source_name = self._extract_source_name(article.source_feed)
            if source_name:
                metadata_parts.append(f"📡 {source_name}")

        if metadata_parts:
            article_text += " • ".join(metadata_parts) + "\n"

        # Add read link with better CTA
        article_text += f"🔗 [Read Full Article]({article.url})\n\n"

        # Add visual separator for detailed format
        if format_type == DigestFormat.DETAILED:
            article_text += "━━━━━━━━━━━━━━━━━━━━\n\n"

        return article_text

    def _format_digest_footer(self, articles_by_topic: Dict[str, List[Article]],
                             format_type: DigestFormat) -> Optional[str]:
        """Format footer for daily digest.

        Args:
            articles_by_topic: Articles by topic
            format_type: Digest format

        Returns:
            Formatted footer message or None
        """
        if format_type in [DigestFormat.COMPACT, DigestFormat.HEADLINES]:
            return None

        total_articles = sum(len(articles) for articles in articles_by_topic.values())
        estimated_reading_time = max(1, total_articles * 0.5)  # 30 seconds per article

        footer = (
            f"📚 *Enjoyed this digest?*\n"
            f"Forward to colleagues or save ⭐ articles you found useful\n\n"
            f"🤖 Powered by CuliFeed • Daily AI curation\n"
            f"⏱️ Total reading time: ~{estimated_reading_time:.0f} minutes"
        )

        return footer

    def _format_empty_digest(self) -> str:
        """Format message for empty digest.

        Returns:
            Formatted empty digest message
        """
        today = datetime.now(timezone.utc).strftime("%B %d, %Y")

        return (
            f"📰 *CuliFeed Daily Digest*\n"
            f"📅 {today}\n\n"
            f"🤔 No curated articles today.\n\n"
            f"*Possible reasons:*\n"
            f"• No new content in your RSS feeds\n"
            f"• No content matched your topics\n"
            f"• Feed processing still in progress\n\n"
            f"💡 Check your feeds and topics with `/status`"
        )

    def _extract_content_preview(self, content: str, max_length: int = 150) -> str:
        """Extract a preview from article content.

        Args:
            content: Full article content
            max_length: Maximum preview length

        Returns:
            Content preview
        """
        if not content:
            return ""

        # Clean up content
        cleaned = re.sub(r'\s+', ' ', content.strip())

        # Find a good break point
        if len(cleaned) <= max_length:
            return cleaned

        # Try to break at sentence end
        preview = cleaned[:max_length]
        last_period = preview.rfind('.')
        last_exclamation = preview.rfind('!')
        last_question = preview.rfind('?')

        best_break = max(last_period, last_exclamation, last_question)

        if best_break > max_length * 0.7:  # If break point is reasonable
            return cleaned[:best_break + 1]
        else:
            return cleaned[:max_length - 3] + "..."

    def _extract_source_name(self, source_feed: str) -> str:
        """Extract a readable source name from feed URL.

        Args:
            source_feed: Feed URL

        Returns:
            Readable source name
        """
        try:
            # Extract domain name
            import re
            match = re.search(r'https?://([^/]+)', source_feed)
            if match:
                domain = match.group(1)
                # Remove 'www.' prefix
                domain = re.sub(r'^www\.', '', domain)
                # Capitalize first letter
                return domain.split('.')[0].capitalize()
            return "Unknown Source"
        except:
            return "RSS Feed"

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length with ellipsis.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        # For longer max_length, try to break at word boundaries
        if max_length > 10:
            truncate_pos = max_length - 3
            last_space = text.rfind(' ', 0, truncate_pos)

            # Only break at word boundary if it's not too short
            if last_space > max_length // 2:
                return text[:last_space] + "..."

        return text[:max_length - 3] + "..."

    # ================================================================
    # TEMPLATE-BASED FORMATTING
    # ================================================================

    def format_with_template(self, template_name: str, **kwargs) -> str:
        """Format content using a predefined template.

        Args:
            template_name: Name of the template to use
            **kwargs: Template variables

        Returns:
            Formatted message
        """
        templates = {
            'welcome': (
                "🎉 *Welcome to CuliFeed!*\n\n"
                "I'm ready to curate content for *{channel_name}*.\n\n"
                "Get started with `/help` for available commands."
            ),
            'setup_complete': (
                "✅ *Setup Complete!*\n\n"
                "You have {topic_count} topics and {feed_count} feeds configured.\n"
                "I'll start delivering curated content daily!"
            ),
            'error_message': (
                "❌ *{error_type}*\n\n"
                "{error_message}\n\n"
                "Use `/help` if you need assistance."
            )
        }

        template = templates.get(template_name, "{content}")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            return kwargs.get('content', 'Error in message formatting')

    def estimate_reading_time(self, articles: List[Article]) -> int:
        """Estimate reading time for a list of articles.

        Args:
            articles: List of articles

        Returns:
            Estimated reading time in minutes
        """
        total_words = 0
        for article in articles:
            # Estimate words from title and content
            title_words = len(article.title.split()) if article.title else 0
            content_words = len(article.content.split()) if article.content else 100  # Default estimate
            total_words += title_words + min(content_words, 300)  # Cap at 300 words per article

        # Average reading speed: 200 words per minute
        reading_time = max(1, total_words / 200)
        return int(reading_time)


# Convenience function
def create_digest_formatter() -> DigestFormatter:
    """Create a DigestFormatter instance.

    Returns:
        Configured DigestFormatter instance
    """
    return DigestFormatter()