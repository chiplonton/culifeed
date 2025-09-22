"""
CuliFeed Input Validators
========================

Comprehensive input validation utilities for URLs, content, configuration,
and user inputs with proper sanitization and security checks.
"""

import re
import json
from urllib.parse import urlparse, urlunparse
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .exceptions import ValidationError, ErrorCode


class URLValidator:
    """URL validation and sanitization utilities."""
    
    # Allowed schemes for RSS feeds
    ALLOWED_SCHEMES = {'http', 'https'}
    
    # Common RSS/Atom feed patterns
    RSS_PATTERNS = [
        r'\.rss$', r'\.xml$', r'\.atom$',
        r'/rss/?$', r'/feed/?$', r'/feeds/?$',
        r'/atom/?$', r'/rss\.xml$', r'/feed\.xml$'
    ]
    
    @classmethod
    def validate_feed_url(cls, url: str) -> str:
        """Validate and normalize RSS feed URL.
        
        Args:
            url: URL to validate
            
        Returns:
            Normalized URL
            
        Raises:
            ValidationError: If URL is invalid
        """
        if not url or not isinstance(url, str):
            raise ValidationError(
                "URL is required and must be a string",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="url"
            )
        
        url = url.strip()
        
        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(
                f"Invalid URL format: {str(e)}",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="url"
            )
        
        # Validate scheme
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            raise ValidationError(
                f"URL scheme must be {' or '.join(cls.ALLOWED_SCHEMES)}",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="url"
            )
        
        # Validate hostname
        if not parsed.netloc:
            raise ValidationError(
                "URL must include a hostname",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="url"
            )
        
        # Check for suspicious patterns
        if cls._has_suspicious_patterns(url):
            raise ValidationError(
                "URL contains suspicious patterns",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="url"
            )
        
        # Normalize URL
        return urlunparse(parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=parsed.path or '/',
            fragment=''  # Remove fragments
        ))
    
    @classmethod
    def _has_suspicious_patterns(cls, url: str) -> bool:
        """Check for suspicious URL patterns."""
        suspicious_patterns = [
            r'javascript:',
            r'data:',
            r'file:',
            r'ftp:',
            r'localhost',
            r'127\.0\.0\.1',
            r'10\.\d+\.\d+\.\d+',
            r'192\.168\.\d+\.\d+',
        ]
        
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in suspicious_patterns)
    
    @classmethod
    def validate_article_url(cls, url: str) -> str:
        """Validate and normalize article URL.
        
        Args:
            url: Article URL to validate
            
        Returns:
            Normalized URL
            
        Raises:
            ValidationError: If URL is invalid
        """
        # Use same validation as feed URLs but with more lenient patterns
        validated = cls.validate_feed_url(url)
        return validated
    
    @classmethod
    def is_likely_feed_url(cls, url: str) -> bool:
        """Check if URL is likely an RSS/Atom feed."""
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in cls.RSS_PATTERNS)


class ContentValidator:
    """Content validation and sanitization utilities."""
    
    # Maximum lengths for various content types
    MAX_TITLE_LENGTH = 1000
    MAX_CONTENT_LENGTH = 50000
    MAX_SUMMARY_LENGTH = 1000
    MAX_TOPIC_NAME_LENGTH = 200
    MAX_KEYWORD_LENGTH = 100
    
    @classmethod
    def validate_article_title(cls, title: str) -> str:
        """Validate and sanitize article title.
        
        Args:
            title: Article title to validate
            
        Returns:
            Sanitized title
            
        Raises:
            ValidationError: If title is invalid
        """
        if not title or not isinstance(title, str):
            raise ValidationError(
                "Title is required",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="title"
            )
        
        title = title.strip()
        
        if not title:
            raise ValidationError(
                "Title cannot be empty",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="title"
            )
        
        if len(title) > cls.MAX_TITLE_LENGTH:
            raise ValidationError(
                f"Title cannot exceed {cls.MAX_TITLE_LENGTH} characters",
                error_code=ErrorCode.VALIDATION_OUT_OF_RANGE,
                field_name="title"
            )
        
        # Basic sanitization
        return cls._sanitize_text(title)
    
    @classmethod
    def validate_article_content(cls, content: str) -> Optional[str]:
        """Validate and sanitize article content.
        
        Args:
            content: Article content to validate
            
        Returns:
            Sanitized content or None if empty
            
        Raises:
            ValidationError: If content is invalid
        """
        if not content:
            return None
        
        if not isinstance(content, str):
            raise ValidationError(
                "Content must be a string",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="content"
            )
        
        content = content.strip()
        
        if len(content) > cls.MAX_CONTENT_LENGTH:
            # Truncate content with warning
            content = content[:cls.MAX_CONTENT_LENGTH] + "... [truncated]"
        
        return cls._sanitize_text(content)
    
    @classmethod
    def validate_topic_name(cls, name: str) -> str:
        """Validate and sanitize topic name.
        
        Args:
            name: Topic name to validate
            
        Returns:
            Sanitized topic name
            
        Raises:
            ValidationError: If name is invalid
        """
        if not name or not isinstance(name, str):
            raise ValidationError(
                "Topic name is required",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="topic_name"
            )
        
        name = name.strip()
        
        if not name:
            raise ValidationError(
                "Topic name cannot be empty",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="topic_name"
            )
        
        if len(name) > cls.MAX_TOPIC_NAME_LENGTH:
            raise ValidationError(
                f"Topic name cannot exceed {cls.MAX_TOPIC_NAME_LENGTH} characters",
                error_code=ErrorCode.VALIDATION_OUT_OF_RANGE,
                field_name="topic_name"
            )
        
        # Check for invalid characters
        if re.search(r'[<>"\'\\\n\r\t]', name):
            raise ValidationError(
                "Topic name contains invalid characters",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="topic_name"
            )

        return name

    @classmethod
    def validate_topic_name_for_ai_generation(cls, name: str) -> str:
        """Validate topic name specifically for AI keyword generation.

        Requires more context (5-20 words) to generate better keywords.

        Args:
            name: Topic name to validate for AI generation

        Returns:
            Sanitized topic name

        Raises:
            ValidationError: If name doesn't have enough context for AI
        """
        # First do standard validation
        validated_name = cls.validate_topic_name(name)

        # Then check word count for AI context
        words = validated_name.split()
        word_count = len(words)

        if word_count < 5:
            raise ValidationError(
                f"Topic needs at least 5 words for better AI keyword generation (you provided {word_count})\n\n"
                f"💡 Examples:\n"
                f"• TikTok software engineering architecture practices\n"
                f"• Machine learning applications in healthcare systems\n"
                f"• DevOps kubernetes deployment best practices\n"
                f"• JavaScript frontend development frameworks comparison\n\n"
                f"Or use manual keywords: `/addtopic {validated_name}, keyword1, keyword2, keyword3`",
                error_code=ErrorCode.VALIDATION_OUT_OF_RANGE,
                field_name="topic_name"
            )

        if word_count > 20:
            raise ValidationError(
                f"Topic is too long ({word_count} words). Please keep it under 20 words for clarity\n\n"
                f"💡 Try focusing on the specific aspect you're most interested in",
                error_code=ErrorCode.VALIDATION_OUT_OF_RANGE,
                field_name="topic_name"
            )

        return validated_name

    @classmethod
    def validate_keywords(cls, keywords: List[str]) -> List[str]:
        """Validate and sanitize keyword list.
        
        Args:
            keywords: List of keywords to validate
            
        Returns:
            List of sanitized keywords
            
        Raises:
            ValidationError: If keywords are invalid
        """
        if not keywords:
            raise ValidationError(
                "At least one keyword is required",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="keywords"
            )
        
        if not isinstance(keywords, list):
            raise ValidationError(
                "Keywords must be a list",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="keywords"
            )
        
        validated_keywords = []
        
        for keyword in keywords:
            if not isinstance(keyword, str):
                continue  # Skip non-string keywords
            
            keyword = keyword.strip().lower()
            
            if not keyword:
                continue  # Skip empty keywords
            
            if len(keyword) > cls.MAX_KEYWORD_LENGTH:
                keyword = keyword[:cls.MAX_KEYWORD_LENGTH]
            
            # Basic sanitization
            keyword = cls._sanitize_keyword(keyword)
            
            if keyword and keyword not in validated_keywords:
                validated_keywords.append(keyword)
        
        if not validated_keywords:
            raise ValidationError(
                "No valid keywords provided",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="keywords"
            )
        
        return validated_keywords
    
    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Sanitize text content."""
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @classmethod
    def _sanitize_keyword(cls, keyword: str) -> str:
        """Sanitize keyword."""
        # Remove special characters except spaces and hyphens
        keyword = re.sub(r'[^\w\s\-]', '', keyword)
        
        # Normalize whitespace
        keyword = re.sub(r'\s+', ' ', keyword)
        
        return keyword.strip()


class ConfigValidator:
    """Configuration validation utilities."""
    
    @classmethod
    def validate_confidence_threshold(cls, threshold: float) -> float:
        """Validate confidence threshold value.
        
        Args:
            threshold: Confidence threshold to validate
            
        Returns:
            Validated threshold
            
        Raises:
            ValidationError: If threshold is invalid
        """
        if not isinstance(threshold, (int, float)):
            raise ValidationError(
                "Confidence threshold must be a number",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="confidence_threshold"
            )
        
        threshold = float(threshold)
        
        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(
                "Confidence threshold must be between 0.0 and 1.0",
                error_code=ErrorCode.VALIDATION_OUT_OF_RANGE,
                field_name="confidence_threshold"
            )
        
        return threshold
    
    @classmethod
    def validate_chat_id(cls, chat_id: str) -> str:
        """Validate Telegram chat ID.
        
        Args:
            chat_id: Chat ID to validate
            
        Returns:
            Validated chat ID
            
        Raises:
            ValidationError: If chat ID is invalid
        """
        if not chat_id or not isinstance(chat_id, str):
            raise ValidationError(
                "Chat ID is required",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="chat_id"
            )
        
        chat_id = chat_id.strip()
        
        # Telegram chat IDs are integers (may be negative for groups)
        if not re.match(r'^-?\d+$', chat_id):
            raise ValidationError(
                "Invalid chat ID format",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name="chat_id"
            )
        
        return chat_id
    
    @classmethod
    def validate_json_field(cls, field_value: str, field_name: str) -> Dict[str, Any]:
        """Validate JSON field content.
        
        Args:
            field_value: JSON string to validate
            field_name: Name of the field for error messages
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON is invalid
        """
        if not field_value:
            return {}
        
        if not isinstance(field_value, str):
            raise ValidationError(
                f"{field_name} must be a JSON string",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name=field_name
            )
        
        try:
            return json.loads(field_value)
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in {field_name}: {str(e)}",
                error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
                field_name=field_name
            )


class TelegramValidator:
    """Telegram-specific validation utilities."""
    
    MAX_MESSAGE_LENGTH = 4096
    MAX_CAPTION_LENGTH = 1024
    
    @classmethod
    def validate_message_content(cls, content: str) -> str:
        """Validate and truncate Telegram message content.
        
        Args:
            content: Message content to validate
            
        Returns:
            Validated and potentially truncated content
            
        Raises:
            ValidationError: If content is invalid
        """
        if not content or not isinstance(content, str):
            raise ValidationError(
                "Message content is required",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="message_content"
            )
        
        content = content.strip()
        
        if not content:
            raise ValidationError(
                "Message content cannot be empty",
                error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
                field_name="message_content"
            )
        
        # Truncate if too long
        if len(content) > cls.MAX_MESSAGE_LENGTH:
            content = content[:cls.MAX_MESSAGE_LENGTH - 20] + "\n\n... [truncated]"
        
        return content
    
    @classmethod
    def sanitize_markdown(cls, text: str) -> str:
        """Sanitize text for Telegram Markdown formatting.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text safe for Telegram Markdown
        """
        # Escape special Markdown characters
        special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        
        for char in special_chars:
            text = text.replace(char, f'\\{char}')
        
        return text


def validate_file_path(file_path: str, must_exist: bool = False) -> Path:
    """Validate file path.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must already exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If path is invalid
    """
    if not file_path or not isinstance(file_path, str):
        raise ValidationError(
            "File path is required",
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name="file_path"
        )
    
    try:
        path = Path(file_path).resolve()
    except Exception as e:
        raise ValidationError(
            f"Invalid file path: {str(e)}",
            error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
            field_name="file_path"
        )
    
    if must_exist and not path.exists():
        raise ValidationError(
            f"File does not exist: {file_path}",
            error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
            field_name="file_path"
        )
    
    return path


def validate_environment_variable(var_name: str, required: bool = True) -> Optional[str]:
    """Validate environment variable.
    
    Args:
        var_name: Environment variable name
        required: Whether the variable is required
        
    Returns:
        Environment variable value or None if not required and not set
        
    Raises:
        ValidationError: If required variable is missing
    """
    import os
    
    value = os.getenv(var_name)
    
    if required and not value:
        raise ValidationError(
            f"Required environment variable {var_name} is not set",
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name=var_name
        )
    
    return value


# Convenience functions for common validation operations
def validate_url(url: str) -> bool:
    """
    Quick validation function for URLs.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        URLValidator.validate_feed_url(url)
        return True
    except ValidationError:
        return False


def validate_content_length(content: str, max_length: int) -> bool:
    """
    Validate content length.
    
    Args:
        content: Content to validate
        max_length: Maximum allowed length
        
    Returns:
        True if content length is valid, False otherwise
    """
    if not content:
        return True
    
    return len(content) <= max_length


def validate_article_data(title: str, link: str, summary: str) -> None:
    """
    Validate required article data fields.
    
    Args:
        title: Article title
        link: Article URL
        summary: Article summary
        
    Raises:
        ValidationError: If any field is invalid
    """
    if not title or not title.strip():
        raise ValidationError(
            "Article title is required",
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name="title"
        )
    
    if not link or not link.strip():
        raise ValidationError(
            "Article link is required", 
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name="link"
        )
    
    # Validate the URL format
    try:
        URLValidator.validate_article_url(link)
    except ValidationError as e:
        raise ValidationError(
            f"Invalid article URL: {e.message}",
            error_code=e.error_code,
            field_name="link"
        ) from e
    
    # Summary is optional but if provided should have reasonable length
    if summary and len(summary) > 10000:  # 10KB limit for summaries
        raise ValidationError(
            "Article summary is too long",
            error_code=ErrorCode.VALIDATION_INVALID_FORMAT,
            field_name="summary"
        )


def validate_feed_metadata(title: str, link: str, description: str) -> None:
    """
    Validate RSS feed metadata.
    
    Args:
        title: Feed title
        link: Feed URL
        description: Feed description
        
    Raises:
        ValidationError: If any field is invalid
    """
    if not title or not title.strip():
        raise ValidationError(
            "Feed title is required",
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name="title"
        )
    
    if not link or not link.strip():
        raise ValidationError(
            "Feed link is required",
            error_code=ErrorCode.VALIDATION_REQUIRED_FIELD,
            field_name="link"  
        )
    
    # Validate the URL format
    try:
        URLValidator.validate_feed_url(link)
    except ValidationError as e:
        raise ValidationError(
            f"Invalid feed URL: {e.message}",
            error_code=e.error_code,
            field_name="link"
        ) from e
