#!/usr/bin/env python3
"""
CuliFeed Error Management System
===============================

Comprehensive error handling, classification, and recovery coordination.
Provides structured error reporting and automated recovery strategies.
"""

import asyncio
import logging
import traceback
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
import json

from ..utils.logging import get_logger_for_component
from ..utils.exceptions import CuliFeedError, ErrorCode


class ErrorSeverity(Enum):
    """Error severity levels for classification and prioritization."""

    LOW = "low"  # Minor issues, system continues normally
    MEDIUM = "medium"  # Notable issues, some functionality affected
    HIGH = "high"  # Serious issues, major functionality impacted
    CRITICAL = "critical"  # System-breaking issues, requires immediate attention


class ErrorCategory(Enum):
    """Categories for error classification and handling strategy."""

    NETWORK = "network"  # Network connectivity, API calls, timeouts
    DATABASE = "database"  # Database connections, queries, schema issues
    PROCESSING = "processing"  # Content processing, AI analysis failures
    CONFIGURATION = "configuration"  # Settings, environment, validation errors
    RESOURCE = "resource"  # Memory, disk space, system resources
    AUTHENTICATION = "auth"  # API keys, tokens, permissions
    TELEGRAM = "telegram"  # Bot API, message sending, channel issues
    FEED = "feed"  # RSS feed parsing, fetching, validation
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class ErrorContext:
    """Contextual information about an error occurrence."""

    component: str  # Component where error occurred
    operation: str  # Operation being performed
    channel_id: Optional[str] = None
    feed_url: Optional[str] = None
    user_data: Optional[Dict] = None
    system_state: Optional[Dict] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ErrorEvent:
    """Structured representation of an error event."""

    id: str  # Unique error identifier
    timestamp: datetime  # When error occurred
    category: ErrorCategory  # Error classification
    severity: ErrorSeverity  # Impact level
    message: str  # Human-readable error description
    exception_type: str  # Exception class name
    stack_trace: Optional[str]  # Full stack trace
    context: ErrorContext  # Contextual information
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["timestamp"] = self.timestamp.isoformat()
        if self.resolved_at:
            data["resolved_at"] = self.resolved_at.isoformat()
        # Convert enums to values
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorEvent":
        """Create ErrorEvent from dictionary."""
        # Convert ISO strings back to datetime
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        if data.get("resolved_at"):
            data["resolved_at"] = datetime.fromisoformat(data["resolved_at"])
        # Convert enum values back to enums
        data["category"] = ErrorCategory(data["category"])
        data["severity"] = ErrorSeverity(data["severity"])
        # Reconstruct context
        context_data = data.pop("context", {})
        data["context"] = ErrorContext(**context_data)

        return cls(**data)


class ErrorClassifier:
    """Classifies exceptions into categories and severity levels."""

    # Exception type to category mapping
    CATEGORY_MAPPING = {
        # Network-related errors
        "ConnectionError": ErrorCategory.NETWORK,
        "TimeoutError": ErrorCategory.NETWORK,
        "HTTPError": ErrorCategory.NETWORK,
        "URLError": ErrorCategory.NETWORK,
        "ConnectTimeout": ErrorCategory.NETWORK,
        "ReadTimeout": ErrorCategory.NETWORK,
        # Database errors
        "DatabaseError": ErrorCategory.DATABASE,
        "IntegrityError": ErrorCategory.DATABASE,
        "OperationalError": ErrorCategory.DATABASE,
        "ProgrammingError": ErrorCategory.DATABASE,
        "DataError": ErrorCategory.DATABASE,
        # Resource errors
        "MemoryError": ErrorCategory.RESOURCE,
        "OSError": ErrorCategory.RESOURCE,
        "PermissionError": ErrorCategory.RESOURCE,
        "FileNotFoundError": ErrorCategory.RESOURCE,
        # Authentication
        "AuthenticationError": ErrorCategory.AUTHENTICATION,
        "PermissionDenied": ErrorCategory.AUTHENTICATION,
        "Unauthorized": ErrorCategory.AUTHENTICATION,
        # Processing errors
        "ValueError": ErrorCategory.PROCESSING,
        "TypeError": ErrorCategory.PROCESSING,
        "KeyError": ErrorCategory.PROCESSING,
        "AttributeError": ErrorCategory.PROCESSING,
    }

    # Message patterns for category classification
    MESSAGE_PATTERNS = {
        ErrorCategory.NETWORK: [
            "connection",
            "timeout",
            "network",
            "dns",
            "socket",
            "http",
            "ssl",
            "certificate",
            "proxy",
        ],
        ErrorCategory.DATABASE: [
            "database",
            "sql",
            "table",
            "column",
            "constraint",
            "sqlite",
            "postgres",
            "mysql",
        ],
        ErrorCategory.TELEGRAM: [
            "telegram",
            "bot",
            "chat",
            "message",
            "channel",
            "token",
            "webhook",
        ],
        ErrorCategory.FEED: [
            "rss",
            "feed",
            "xml",
            "parse",
            "malformed",
            "entry",
            "article",
        ],
        ErrorCategory.AUTHENTICATION: [
            "api key",
            "token",
            "auth",
            "permission",
            "unauthorized",
            "forbidden",
            "credential",
        ],
        ErrorCategory.CONFIGURATION: [
            "config",
            "setting",
            "environment",
            "variable",
            "missing",
            "invalid",
        ],
    }

    @classmethod
    def classify(
        cls, exception: Exception, context: ErrorContext
    ) -> tuple[ErrorCategory, ErrorSeverity]:
        """
        Classify an exception into category and severity.

        Args:
            exception: The exception to classify
            context: Contextual information about the error

        Returns:
            Tuple of (category, severity)
        """
        # Get exception type name
        exception_type = type(exception).__name__

        # Check direct type mapping
        category = cls.CATEGORY_MAPPING.get(exception_type, ErrorCategory.UNKNOWN)

        # Check message patterns for refinement (especially for generic exceptions)
        # This allows specific context-based errors to override generic type classification
        error_message = str(exception).lower()
        for cat, patterns in cls.MESSAGE_PATTERNS.items():
            if any(pattern in error_message for pattern in patterns):
                # Override category with more specific one if patterns match
                # Allow overriding for generic categories or when message strongly indicates different context
                if category in (ErrorCategory.UNKNOWN, ErrorCategory.PROCESSING) or (
                    category == ErrorCategory.NETWORK and cat == ErrorCategory.DATABASE
                ):
                    category = cat
                    break

        # Determine severity based on category and context
        severity = cls._determine_severity(exception, category, context)

        return category, severity

    @classmethod
    def _determine_severity(
        cls, exception: Exception, category: ErrorCategory, context: ErrorContext
    ) -> ErrorSeverity:
        """Determine error severity based on various factors."""

        # Critical severity conditions
        if isinstance(exception, MemoryError):
            return ErrorSeverity.CRITICAL

        if (
            category == ErrorCategory.DATABASE
            and "connection" in str(exception).lower()
        ):
            return ErrorSeverity.CRITICAL

        if category == ErrorCategory.RESOURCE and isinstance(
            exception, PermissionError
        ):
            return ErrorSeverity.CRITICAL

        # High severity conditions
        if category == ErrorCategory.AUTHENTICATION:
            return ErrorSeverity.HIGH

        if category == ErrorCategory.CONFIGURATION:
            return ErrorSeverity.HIGH

        if "critical" in str(exception).lower() or "fatal" in str(exception).lower():
            return ErrorSeverity.HIGH

        # Medium severity conditions
        if category in [ErrorCategory.PROCESSING, ErrorCategory.TELEGRAM]:
            return ErrorSeverity.MEDIUM

        if category == ErrorCategory.NETWORK:
            # Timeouts are usually less severe than connection failures
            error_msg = str(exception).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                return ErrorSeverity.LOW
            else:
                return ErrorSeverity.MEDIUM

        # Default to low severity
        return ErrorSeverity.LOW


class ErrorHandler:
    """
    Central error handling and recovery coordination system.

    Provides comprehensive error management including:
    - Error classification and severity assessment
    - Structured error logging and storage
    - Recovery strategy coordination
    - Error rate monitoring and alerting
    """

    def __init__(self, db_manager=None):
        self.logger = get_logger_for_component("error_handler")
        self.db_manager = db_manager
        self.classifier = ErrorClassifier()

        # Error tracking
        self.error_events: List[ErrorEvent] = []
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}

        # Recovery handlers
        self.recovery_handlers: Dict[ErrorCategory, List[Callable]] = {}

        # Rate limiting for error reporting
        self.error_rate_limits = {
            ErrorSeverity.CRITICAL: timedelta(minutes=1),  # Report immediately
            ErrorSeverity.HIGH: timedelta(minutes=5),  # Max once per 5 minutes
            ErrorSeverity.MEDIUM: timedelta(minutes=15),  # Max once per 15 minutes
            ErrorSeverity.LOW: timedelta(hours=1),  # Max once per hour
        }

    def handle_error(
        self, exception: Exception, context: ErrorContext, attempt_recovery: bool = True
    ) -> ErrorEvent:
        """
        Process and handle an error occurrence.

        Args:
            exception: The exception that occurred
            context: Contextual information about the error
            attempt_recovery: Whether to attempt automated recovery

        Returns:
            ErrorEvent object representing the handled error
        """

        # Generate unique error ID
        error_id = f"{context.component}_{int(time.time() * 1000)}"

        # Classify the error
        category, severity = self.classifier.classify(exception, context)

        # Create error event
        error_event = ErrorEvent(
            id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            context=context,
        )

        # Log the error with appropriate level
        self._log_error(error_event)

        # Store error event
        self._store_error_event(error_event)

        # Update error tracking
        self._update_error_tracking(error_event)

        # Attempt recovery if enabled and appropriate
        if attempt_recovery and self._should_attempt_recovery(error_event):
            success = self._attempt_recovery(error_event)
            error_event.recovery_attempted = True
            error_event.recovery_successful = success

            if success:
                error_event.resolved_at = datetime.now()
                self.logger.info(f"Error {error_id} recovered successfully")

        # Check if error rate alerting is needed
        self._check_error_rate_alerts(error_event)

        return error_event

    def _log_error(self, error_event: ErrorEvent) -> None:
        """Log error with appropriate level and structured data."""

        log_data = {
            "error_id": error_event.id,
            "category": error_event.category.value,
            "severity": error_event.severity.value,
            "component": error_event.context.component,
            "operation": error_event.context.operation,
        }

        # Add optional context data
        if error_event.context.channel_id:
            log_data["channel_id"] = error_event.context.channel_id
        if error_event.context.feed_url:
            log_data["feed_url"] = error_event.context.feed_url

        # Log with appropriate level
        if error_event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_event.message, extra=log_data)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.logger.error(error_event.message, extra=log_data)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_event.message, extra=log_data)
        else:
            self.logger.info(error_event.message, extra=log_data)

        # Log stack trace for high severity errors
        if error_event.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            if error_event.stack_trace:
                self.logger.debug(
                    f"Stack trace for {error_event.id}:\n{error_event.stack_trace}"
                )

    def _store_error_event(self, error_event: ErrorEvent) -> None:
        """Store error event for analysis and reporting."""

        # Store in memory for immediate access
        self.error_events.append(error_event)

        # Keep only recent errors in memory (last 1000)
        if len(self.error_events) > 1000:
            self.error_events = self.error_events[-1000:]

        # Store in database if available
        if self.db_manager:
            try:
                with self.db_manager.get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO error_events 
                        (id, timestamp, category, severity, message, exception_type, 
                         stack_trace, context_json, recovery_attempted, recovery_successful)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            error_event.id,
                            error_event.timestamp,
                            error_event.category.value,
                            error_event.severity.value,
                            error_event.message,
                            error_event.exception_type,
                            error_event.stack_trace,
                            json.dumps(error_event.context.to_dict()),
                            error_event.recovery_attempted,
                            error_event.recovery_successful,
                        ),
                    )
            except Exception as db_error:
                self.logger.warning(
                    f"Failed to store error event in database: {db_error}"
                )

    def _update_error_tracking(self, error_event: ErrorEvent) -> None:
        """Update error frequency tracking."""

        # Track by category
        category_key = f"category_{error_event.category.value}"
        self.error_counts[category_key] = self.error_counts.get(category_key, 0) + 1
        self.last_error_time[category_key] = error_event.timestamp

        # Track by component
        component_key = f"component_{error_event.context.component}"
        self.error_counts[component_key] = self.error_counts.get(component_key, 0) + 1
        self.last_error_time[component_key] = error_event.timestamp

        # Track by severity
        severity_key = f"severity_{error_event.severity.value}"
        self.error_counts[severity_key] = self.error_counts.get(severity_key, 0) + 1
        self.last_error_time[severity_key] = error_event.timestamp

    def _should_attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Determine if automated recovery should be attempted."""

        # Don't attempt recovery for configuration errors
        if error_event.category == ErrorCategory.CONFIGURATION:
            return False

        # Don't attempt recovery for authentication errors
        if error_event.category == ErrorCategory.AUTHENTICATION:
            return False

        # Limit recovery attempts per error type
        category_key = (
            f"recovery_{error_event.category.value}_{error_event.context.component}"
        )
        recent_recovery_attempts = [
            event
            for event in self.error_events[-50:]  # Check last 50 errors
            if (
                event.category == error_event.category
                and event.context.component == error_event.context.component
                and event.recovery_attempted
                and event.timestamp > datetime.now() - timedelta(minutes=30)
            )
        ]

        # Limit to 3 recovery attempts per 30 minutes per category+component
        if len(recent_recovery_attempts) >= 3:
            return False

        return True

    def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt automated recovery for the error."""

        recovery_handlers = self.recovery_handlers.get(error_event.category, [])

        for handler in recovery_handlers:
            try:
                success = handler(error_event)
                if success:
                    self.logger.info(
                        f"Recovery successful for {error_event.id} using {handler.__name__}"
                    )
                    return True
            except Exception as recovery_error:
                self.logger.warning(
                    f"Recovery handler {handler.__name__} failed: {recovery_error}"
                )

        return False

    def _check_error_rate_alerts(self, error_event: ErrorEvent) -> None:
        """Check if error rate alerts should be triggered."""

        # Check if we should report this error based on rate limiting
        rate_limit = self.error_rate_limits.get(error_event.severity)
        if not rate_limit:
            return

        # Check last error of same severity
        severity_key = f"alert_{error_event.severity.value}"
        last_alert_time = self.last_error_time.get(severity_key)

        if last_alert_time and (datetime.now() - last_alert_time) < rate_limit:
            return  # Too soon to alert again

        # Update last alert time
        self.last_error_time[severity_key] = datetime.now()

        # Generate alert (in a real system, this might send notifications)
        self._generate_alert(error_event)

    def _generate_alert(self, error_event: ErrorEvent) -> None:
        """Generate alert for high-severity or frequent errors."""

        alert_message = f"CuliFeed Alert: {error_event.severity.value.upper()} error in {error_event.context.component}"

        # Log alert
        self.logger.warning(
            f"ALERT: {alert_message}",
            extra={
                "error_id": error_event.id,
                "alert_type": "error_severity",
                "severity": error_event.severity.value,
            },
        )

        # In a production system, you might send notifications here:
        # - Email alerts
        # - Slack notifications
        # - PagerDuty alerts
        # - SMS notifications

    def register_recovery_handler(
        self, category: ErrorCategory, handler: Callable
    ) -> None:
        """Register a recovery handler for a specific error category."""

        if category not in self.recovery_handlers:
            self.recovery_handlers[category] = []

        self.recovery_handlers[category].append(handler)
        self.logger.info(
            f"Registered recovery handler {handler.__name__} for {category.value}"
        )

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the specified time period."""

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_events if e.timestamp > cutoff_time]

        # Count by category
        category_counts = {}
        for error in recent_errors:
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1

        # Count by severity
        severity_counts = {}
        for error in recent_errors:
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Count by component
        component_counts = {}
        for error in recent_errors:
            component = error.context.component
            component_counts[component] = component_counts.get(component, 0) + 1

        # Recovery statistics
        recovery_attempted = len([e for e in recent_errors if e.recovery_attempted])
        recovery_successful = len([e for e in recent_errors if e.recovery_successful])
        recovery_rate = (
            (recovery_successful / recovery_attempted * 100)
            if recovery_attempted > 0
            else 0
        )

        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "by_component": component_counts,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "recovery_rate_percent": round(recovery_rate, 1),
        }

    def get_recent_errors(
        self, count: int = 50, severity: Optional[ErrorSeverity] = None
    ) -> List[ErrorEvent]:
        """Get recent errors, optionally filtered by severity."""

        errors = (
            self.error_events[-count:]
            if not severity
            else [e for e in self.error_events[-count * 2 :] if e.severity == severity]
        )

        return errors[-count:]  # Return most recent up to count

    def clear_resolved_errors(self, older_than_hours: int = 24) -> int:
        """Clear resolved errors older than specified time."""

        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        initial_count = len(self.error_events)
        self.error_events = [
            e
            for e in self.error_events
            if not (e.resolved_at and e.resolved_at < cutoff_time)
        ]

        cleared_count = initial_count - len(self.error_events)

        if cleared_count > 0:
            self.logger.info(
                f"Cleared {cleared_count} resolved errors older than {older_than_hours} hours"
            )

        return cleared_count


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler(db_manager=None) -> ErrorHandler:
    """Get the global error handler instance."""
    global _global_error_handler

    if _global_error_handler is None:
        _global_error_handler = ErrorHandler(db_manager)

    return _global_error_handler


def handle_error(
    exception: Exception, context: ErrorContext, attempt_recovery: bool = True
) -> ErrorEvent:
    """Convenience function to handle errors using the global handler."""
    handler = get_error_handler()
    return handler.handle_error(exception, context, attempt_recovery)
