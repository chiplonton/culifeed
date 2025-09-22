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
from ...ai.ai_manager import AIManager
from ...utils.exceptions import TelegramError, ErrorCode, AIError


class TopicCommandHandler:
    """Handler for topic-related bot commands."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize topic command handler.

        Args:
            db_connection: Database connection manager
        """
        self.db = db_connection
        self.topic_repo = TopicRepository(db_connection)
        self.ai_manager = AIManager()
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
                for topic in topics:
                    # Show all keywords with clear visual separation
                    keywords_display = ", ".join(topic.keywords)

                    message += (
                        f"🎯 **{topic.name}**\n"
                        f"    → {keywords_display}\n\n"
                    )

                message += f"*{len(topics)} topics configured*\n\n"
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

            # Handle keywords - either provided or AI-generated
            if keywords is None:
                # AI keyword generation - validate topic has enough context
                try:
                    ai_validated_name = ContentValidator.validate_topic_name_for_ai_generation(validated_name)
                except ValidationError as e:
                    await update.message.reply_text(
                        f"❌ *AI keyword generation needs more context:*\n\n{str(e).split('] ')[1]}",
                        parse_mode='Markdown'
                    )
                    return

                progress_msg = await update.message.reply_text(f"🤖 Generating keywords for '{ai_validated_name}'...")
                try:
                    validated_keywords = await self._generate_keywords_with_ai(ai_validated_name, chat_id)
                    await progress_msg.edit_text(f"✅ Generated keywords: {', '.join(validated_keywords)}")
                except Exception as e:
                    await progress_msg.edit_text(f"⚠️ Using fallback keywords")
                    validated_keywords = [ai_validated_name.lower()]
            else:
                # Manual keywords provided
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
                confidence_threshold=0.6,  # Default threshold (Phase 1)
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

            # Get the appropriate message object for reply
            message = None
            if update.message:
                message = update.message
            elif update.effective_message:
                message = update.effective_message
            elif update.callback_query and update.callback_query.message:
                message = update.callback_query.message

            if not message:
                self.logger.warning("No message object available for remove topic response")
                return

            if not args:
                await message.reply_text(
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
                await message.reply_text(
                    f"❌ Topic *'{topic_name}'* not found.\n\n"
                    f"Use `/topics` to see all your topics.",
                    parse_mode='Markdown'
                )
                return

            # Remove the topic
            success = self.topic_repo.delete_topic(topic.id)

            if success:
                await message.reply_text(
                    f"✅ Topic *'{topic_name}'* removed successfully!",
                    parse_mode='Markdown'
                )
                self.logger.info(f"Removed topic '{topic_name}' from channel {chat_id}")
            else:
                await message.reply_text(
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

    async def _generate_keywords_with_ai(self, topic_name: str, chat_id: str) -> List[str]:
        """Generate keywords for a topic using AI."""
        try:
            # Remove context contamination to prevent keyword bleeding between unrelated topics
            # Each topic should generate keywords based solely on its own content
            context = ""
            
            # Use AIManager with proper fallback strategy - same as other AI operations
            result = await self.ai_manager.generate_keywords(topic_name, context, max_keywords=7)
            
            if result.success and result.content:
                keywords = result.content if isinstance(result.content, list) else []
                return keywords[:7] if keywords else [topic_name.lower()]
            else:
                # AI failed, use fallback
                self.logger.warning(f"AI keyword generation failed: {result.error_message}")
                raise AIError(result.error_message or "AI generation failed")
            
        except Exception as e:
            self.logger.error(f"AI keyword generation failed: {e}")
            # Simple fallback - same as AIManager fallback but simpler
            return [topic_name.lower(), f"{topic_name.lower()} technology"]

    def _parse_add_topic_args(self, args: List[str]) -> Optional[tuple[str, Optional[List[str]]]]:
        """Parse arguments for /addtopic command.

        Args:
            args: Command arguments

        Returns:
            Tuple of (topic_name, keywords) or None if invalid.
            keywords can be None to indicate AI generation should be used.
        """
        if len(args) < 1:
            return None

        # Join all args to analyze the full input
        full_text = " ".join(args)

        # Check if input contains commas (explicit manual keyword format)
        if "," in full_text:
            # Format: /addtopic AI machine learning, artificial intelligence, ML
            parts = [part.strip() for part in full_text.split(",")]
            if len(parts) >= 2:
                topic_name = parts[0]
                keywords = parts[1:]
                return topic_name, keywords

        # No commas found - treat entire input as topic name for AI generation
        # This handles both single words and multi-word topic names
        topic_name = full_text.strip()
        return topic_name, None  # None indicates AI generation

    async def _send_add_topic_help(self, update: Update) -> None:
        """Send help message for /addtopic command."""
        help_message = (
            "❓ *How to add a topic:*\n\n"
            "*🤖 AI Generation:* `/addtopic <topic_name>`\n"
            "*📝 Manual Keywords:* `/addtopic <topic_name>, <keyword1>, <keyword2>, <keyword3>`\n\n"
            "*Examples:*\n"
            "• `/addtopic Machine Learning` - AI will generate keywords\n"
            "• `/addtopic AWS ECS Performance` - AI will generate keywords\n"
            "• `/addtopic Cloud, AWS, Azure, GCP, kubernetes` - Manual keywords\n"
            "• `/addtopic Python, python programming, django, flask` - Manual keywords\n\n"
            "*Tips:*\n"
            "• No commas = AI generates keywords automatically\n"
            "• With commas = Manual keyword specification\n"
            "• AI considers your existing topics for context"
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
            
            # Try multiple message sources in order of preference
            message = None
            if update.message:
                message = update.message
            elif update.effective_message:
                message = update.effective_message
            elif update.callback_query and update.callback_query.message:
                message = update.callback_query.message
                
            if message:
                await message.reply_text(error_message, parse_mode='Markdown')
            else:
                self.logger.warning("No message object available to send error response")
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