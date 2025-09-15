"""
Topic Management Commands
========================

Telegram bot commands for managing topics in CuliFeed channels.
Handles topic creation, editing, deletion, and listing.

Commands:
- /topics - List all topics for the channel
- /addtopic - Add a new topic with keywords
- /removetopic - Remove an existing topic
- /edittopic - Edit an existing topic
"""

import re
from typing import List, Optional, Dict, Any

from telegram import Update
from telegram.ext import ContextTypes

from ...database.connection import DatabaseConnection
from ...database.models import Topic
from ...storage.topic_repository import TopicRepository
from ...utils.logging import get_logger_for_component
from ...utils.validators import ContentValidator, ValidationError
from ...utils.exceptions import TelegramError, ErrorCode


class TopicCommandHandler:
    """Handler for topic-related bot commands."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize topic command handler.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.topic_repo = TopicRepository(db_connection)
        self.logger = get_logger_for_component('topic_commands')

    async def handle_list_topics(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /topics command - list all topics for the channel.

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)

            # Get all topics for this channel
            topics = self.topic_repo.get_topics_for_channel(chat_id, active_only=True)

            if not topics:
                message = (
                    "📝 *No topics configured*\n\n"
                    "Add your first topic with:\n"
                    "`/addtopic AI machine learning, artificial intelligence, ML`\n\n"
                    "Topics help me understand what content you're interested in!"
                )
            else:
                message = "📝 *Your Topics:*\n\n"
                for i, topic in enumerate(topics, 1):
                    keywords_preview = ", ".join(topic.keywords[:3])
                    if len(topic.keywords) > 3:
                        keywords_preview += f" (+{len(topic.keywords) - 3} more)"

                    message += (
                        f"*{i}. {topic.name}*\n"
                        f"Keywords: {keywords_preview}\n"
                        f"Threshold: {topic.confidence_threshold:.1f}\n\n"
                    )

                message += f"*Total: {len(topics)} topics*\n\n"
                message += "💡 Use `/addtopic` to add more or `/removetopic` to remove."

            await update.message.reply_text(message, parse_mode='Markdown')

        except Exception as e:
            await self._handle_error(update, "list topics", e)

    async def handle_add_topic(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /addtopic command - add a new topic.

        Format: /addtopic <name> <keyword1, keyword2, keyword3>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)
            args = context.args

            if not args:
                await self._send_add_topic_help(update)
                return

            # Parse arguments
            parsed_data = self._parse_add_topic_args(args)
            if not parsed_data:
                await self._send_add_topic_help(update)
                return

            topic_name, keywords = parsed_data

            # Validate topic name
            try:
                validated_name = ContentValidator.validate_topic_name(topic_name)
            except ValidationError as e:
                await update.message.reply_text(
                    f"❌ *Invalid topic name:* {e.message}",
                    parse_mode='Markdown'
                )
                return

            # Validate keywords
            try:
                validated_keywords = ContentValidator.validate_keywords(keywords)
            except ValidationError as e:
                await update.message.reply_text(
                    f"❌ *Invalid keywords:* {e.message}",
                    parse_mode='Markdown'
                )
                return

            # Check if topic already exists
            existing_topic = self.topic_repo.get_topic_by_name(chat_id, validated_name)
            if existing_topic:
                await update.message.reply_text(
                    f"❌ Topic *'{validated_name}'* already exists.\n"
                    f"Use `/edittopic {validated_name}` to modify it.",
                    parse_mode='Markdown'
                )
                return

            # Create new topic
            topic = Topic(
                chat_id=chat_id,
                name=validated_name,
                keywords=validated_keywords,
                exclude_keywords=[],
                confidence_threshold=0.7,  # Default threshold
                active=True
            )

            # Save to database
            topic_id = self.topic_repo.create_topic(topic)

            if topic_id:
                success_message = (
                    f"✅ *Topic '{validated_name}' created successfully!*\n\n"
                    f"*Keywords:* {', '.join(validated_keywords)}\n"
                    f"*Confidence threshold:* {topic.confidence_threshold}\n\n"
                    f"🎯 I'll now look for content matching these keywords!\n\n"
                    f"💡 Add RSS feeds with `/addfeed` to start getting content."
                )
                await update.message.reply_text(success_message, parse_mode='Markdown')

                self.logger.info(f"Created topic '{validated_name}' for channel {chat_id}")
            else:
                await update.message.reply_text(
                    "❌ Failed to create topic. Please try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "add topic", e)

    async def handle_remove_topic(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /removetopic command - remove an existing topic.

        Format: /removetopic <name>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)
            args = context.args

            if not args:
                await update.message.reply_text(
                    "❌ *Missing topic name*\n\n"
                    "Usage: `/removetopic <topic_name>`\n"
                    "Example: `/removetopic AI`\n\n"
                    "Use `/topics` to see all your topics.",
                    parse_mode='Markdown'
                )
                return

            topic_name = " ".join(args).strip()

            # Find the topic
            topic = self.topic_repo.get_topic_by_name(chat_id, topic_name)
            if not topic:
                await update.message.reply_text(
                    f"❌ Topic *'{topic_name}'* not found.\n\n"
                    f"Use `/topics` to see all your topics.",
                    parse_mode='Markdown'
                )
                return

            # Remove the topic
            success = self.topic_repo.delete_topic(topic.id)

            if success:
                await update.message.reply_text(
                    f"✅ Topic *'{topic_name}'* removed successfully!",
                    parse_mode='Markdown'
                )
                self.logger.info(f"Removed topic '{topic_name}' from channel {chat_id}")
            else:
                await update.message.reply_text(
                    "❌ Failed to remove topic. Please try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "remove topic", e)

    async def handle_edit_topic(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /edittopic command - edit an existing topic.

        Format: /edittopic <name> <new_keywords>

        Args:
            update: Telegram update object
            context: Bot context
        """
        try:
            chat_id = str(update.effective_chat.id)
            args = context.args

            if not args:
                await self._send_edit_topic_help(update)
                return

            # Parse arguments (first arg is topic name, rest are keywords)
            if len(args) < 2:
                await self._send_edit_topic_help(update)
                return

            topic_name = args[0]
            keywords_text = " ".join(args[1:])
            keywords = [k.strip() for k in keywords_text.split(",") if k.strip()]

            if not keywords:
                await self._send_edit_topic_help(update)
                return

            # Find the topic
            topic = self.topic_repo.get_topic_by_name(chat_id, topic_name)
            if not topic:
                await update.message.reply_text(
                    f"❌ Topic *'{topic_name}'* not found.\n\n"
                    f"Use `/topics` to see all your topics.",
                    parse_mode='Markdown'
                )
                return

            # Validate new keywords
            try:
                validated_keywords = ContentValidator.validate_keywords(keywords)
            except ValidationError as e:
                await update.message.reply_text(
                    f"❌ *Invalid keywords:* {e.message}",
                    parse_mode='Markdown'
                )
                return

            # Update the topic
            topic.keywords = validated_keywords
            success = self.topic_repo.update_topic_object(topic)

            if success:
                await update.message.reply_text(
                    f"✅ Topic *'{topic_name}'* updated successfully!\n\n"
                    f"*New keywords:* {', '.join(validated_keywords)}",
                    parse_mode='Markdown'
                )
                self.logger.info(f"Updated topic '{topic_name}' for channel {chat_id}")
            else:
                await update.message.reply_text(
                    "❌ Failed to update topic. Please try again.",
                    parse_mode='Markdown'
                )

        except Exception as e:
            await self._handle_error(update, "edit topic", e)

    def _parse_add_topic_args(self, args: List[str]) -> Optional[tuple[str, List[str]]]:
        """Parse arguments for /addtopic command.

        Args:
            args: Command arguments

        Returns:
            Tuple of (topic_name, keywords) or None if invalid
        """
        if len(args) < 2:
            return None

        # Join all args and split by comma to handle various formats
        full_text = " ".join(args)

        # Try to split by comma first
        if "," in full_text:
            # Format: /addtopic AI machine learning, artificial intelligence, ML
            parts = [part.strip() for part in full_text.split(",")]
            if len(parts) >= 2:
                topic_name = parts[0]
                keywords = parts[1:]
                return topic_name, keywords

        # Try space-separated format
        if len(args) >= 2:
            # Format: /addtopic AI "machine learning" "artificial intelligence"
            topic_name = args[0]
            keywords = args[1:]
            return topic_name, keywords

        return None

    async def _send_add_topic_help(self, update: Update) -> None:
        """Send help message for /addtopic command."""
        help_message = (
            "❓ *How to add a topic:*\n\n"
            "*Format:* `/addtopic <name> <keyword1, keyword2, keyword3>`\n\n"
            "*Examples:*\n"
            "• `/addtopic AI machine learning, artificial intelligence, ML`\n"
            "• `/addtopic Cloud AWS, Azure, GCP, cloud computing`\n"
            "• `/addtopic Python python programming, django, flask`\n\n"
            "*Tips:*\n"
            "• Use specific keywords for better matching\n"
            "• Separate keywords with commas\n"
            "• Topic names should be short and descriptive"
        )
        await update.message.reply_text(help_message, parse_mode='Markdown')

    async def _send_edit_topic_help(self, update: Update) -> None:
        """Send help message for /edittopic command."""
        help_message = (
            "❓ *How to edit a topic:*\n\n"
            "*Format:* `/edittopic <topic_name> <new_keywords>`\n\n"
            "*Examples:*\n"
            "• `/edittopic AI machine learning, deep learning, neural networks`\n"
            "• `/edittopic Cloud kubernetes, docker, containers`\n\n"
            "*Note:* This replaces all keywords for the topic.\n"
            "Use `/topics` to see your current topics."
        )
        await update.message.reply_text(help_message, parse_mode='Markdown')

    async def _handle_error(self, update: Update, operation: str, error: Exception) -> None:
        """Handle errors in topic operations.

        Args:
            update: Telegram update object
            operation: Operation that failed
            error: Exception that occurred
        """
        self.logger.error(f"Error in {operation}: {error}")

        try:
            error_message = (
                f"❌ *Error in {operation}*\n\n"
                f"Please try again or use `/help` for usage instructions."
            )
            await update.message.reply_text(error_message, parse_mode='Markdown')
        except Exception as e:
            self.logger.error(f"Failed to send error message: {e}")

    # ================================================================
    # UTILITY METHODS
    # ================================================================

    def get_topic_statistics(self, chat_id: str) -> Dict[str, Any]:
        """Get topic statistics for a channel.

        Args:
            chat_id: Channel chat ID

        Returns:
            Dictionary with topic statistics
        """
        try:
            topics = self.topic_repo.get_topics_for_channel(chat_id, active_only=True)

            total_keywords = sum(len(topic.keywords) for topic in topics)
            avg_keywords = total_keywords / len(topics) if topics else 0

            return {
                'total_topics': len(topics),
                'total_keywords': total_keywords,
                'average_keywords_per_topic': round(avg_keywords, 1),
                'topics': [
                    {
                        'name': topic.name,
                        'keyword_count': len(topic.keywords),
                        'threshold': topic.confidence_threshold
                    }
                    for topic in topics
                ]
            }

        except Exception as e:
            self.logger.error(f"Error getting topic statistics: {e}")
            return {}

    async def validate_topic_setup(self, chat_id: str) -> Dict[str, Any]:
        """Validate topic setup for a channel.

        Args:
            chat_id: Channel chat ID

        Returns:
            Validation results dictionary
        """
        try:
            topics = self.topic_repo.get_topics_for_channel(chat_id, active_only=True)

            issues = []
            warnings = []

            if not topics:
                issues.append("No topics configured")
            else:
                # Check for topics with very few keywords
                for topic in topics:
                    if len(topic.keywords) < 2:
                        warnings.append(f"Topic '{topic.name}' has only {len(topic.keywords)} keyword(s)")

                # Check for very low thresholds
                low_threshold_topics = [t for t in topics if t.confidence_threshold < 0.3]
                if low_threshold_topics:
                    warnings.append(f"{len(low_threshold_topics)} topic(s) have very low confidence thresholds")

            return {
                'valid': len(issues) == 0,
                'topic_count': len(topics),
                'issues': issues,
                'warnings': warnings
            }

        except Exception as e:
            self.logger.error(f"Error validating topic setup: {e}")
            return {'valid': False, 'issues': ['Validation error occurred']}