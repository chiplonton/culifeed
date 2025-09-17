#!/usr/bin/env python3
"""
CuliFeed Retry Logic and Circuit Breaker System
=============================================

Advanced retry mechanisms with exponential backoff, circuit breakers,
and intelligent failure handling for resilient system operation.
"""

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from functools import wraps
import inspect

from ..utils.logging import get_logger_for_component
from .error_handler import ErrorContext, ErrorHandler, ErrorCategory


T = TypeVar('T')


class RetryStrategy(Enum):
    """Different retry strategy types."""
    FIXED_DELAY = "fixed_delay"              # Fixed interval between retries
    EXPONENTIAL_BACKOFF = "exponential"     # Exponentially increasing delays
    LINEAR_BACKOFF = "linear"               # Linearly increasing delays
    JITTERED_EXPONENTIAL = "jittered"       # Exponential with random jitter
    FIBONACCI = "fibonacci"                 # Fibonacci sequence delays


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"                       # Normal operation
    OPEN = "open"                          # Failing, requests rejected
    HALF_OPEN = "half_open"                # Testing if recovery is possible


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3                   # Maximum retry attempts
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0                # Base delay in seconds
    max_delay: float = 60.0                # Maximum delay in seconds
    jitter: bool = True                    # Add randomization to delays
    exponential_base: float = 2.0          # Exponential backoff multiplier
    
    # Conditions for retrying
    retry_on_exceptions: tuple = (ConnectionError, TimeoutError)
    retry_on_status_codes: tuple = (429, 500, 502, 503, 504)
    
    # Circuit breaker settings
    circuit_failure_threshold: int = 5      # Failures before circuit opens
    circuit_recovery_timeout: float = 30.0  # Seconds before trying half-open
    circuit_success_threshold: int = 2       # Successes needed to close circuit


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay: float
    exception: Optional[Exception]
    timestamp: datetime
    success: bool


class RetryStatistics:
    """Statistics tracking for retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.retry_attempts: List[RetryAttempt] = []
        self.success_rate = 0.0
        self.average_attempts = 0.0
        
    def record_attempt(self, attempt: RetryAttempt):
        """Record a retry attempt."""
        self.retry_attempts.append(attempt)
        self.total_attempts += 1
        
        if attempt.success:
            self.total_successes += 1
        else:
            self.total_failures += 1
        
        # Update statistics
        self._update_stats()
        
        # Keep only recent attempts (last 1000)
        if len(self.retry_attempts) > 1000:
            self.retry_attempts = self.retry_attempts[-1000:]
    
    def _update_stats(self):
        """Update computed statistics."""
        if self.total_attempts > 0:
            self.success_rate = (self.total_successes / self.total_attempts) * 100
        
        if self.retry_attempts:
            # Calculate average attempts per operation
            operations = {}
            for attempt in self.retry_attempts:
                key = attempt.timestamp.date()
                if key not in operations:
                    operations[key] = []
                operations[key].append(attempt)
            
            if operations:
                total_ops = len(operations)
                total_attempts = sum(len(attempts) for attempts in operations.values())
                self.average_attempts = total_attempts / total_ops
    
    def get_recent_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for recent time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_attempts = [a for a in self.retry_attempts if a.timestamp > cutoff]
        
        if not recent_attempts:
            return {
                'period_hours': hours,
                'attempts': 0,
                'success_rate': 0.0,
                'average_delay': 0.0
            }
        
        successes = len([a for a in recent_attempts if a.success])
        success_rate = (successes / len(recent_attempts)) * 100
        average_delay = sum(a.delay for a in recent_attempts) / len(recent_attempts)
        
        return {
            'period_hours': hours,
            'attempts': len(recent_attempts),
            'successes': successes,
            'failures': len(recent_attempts) - successes,
            'success_rate': round(success_rate, 1),
            'average_delay': round(average_delay, 2)
        }


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = get_logger_for_component('circuit_breaker')
    
    def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        now = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if (self.last_failure_time and 
                (now - self.last_failure_time).total_seconds() > self.config.circuit_recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self):
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.circuit_success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.circuit_failure_threshold:
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
        
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.logger.warning("Circuit breaker reopened during half-open test")
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current circuit breaker state information."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryManager:
    """Advanced retry manager with multiple strategies and circuit breaking."""
    
    def __init__(self, config: Optional[RetryConfig] = None, error_handler: Optional[ErrorHandler] = None):
        self.config = config or RetryConfig()
        self.error_handler = error_handler
        self.logger = get_logger_for_component('retry_manager')
        
        # Circuit breakers per operation type
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Statistics tracking
        self.statistics = RetryStatistics()
        
        # Delay calculation methods
        self._delay_calculators = {
            RetryStrategy.FIXED_DELAY: self._calculate_fixed_delay,
            RetryStrategy.EXPONENTIAL_BACKOFF: self._calculate_exponential_delay,
            RetryStrategy.LINEAR_BACKOFF: self._calculate_linear_delay,
            RetryStrategy.JITTERED_EXPONENTIAL: self._calculate_jittered_exponential_delay,
            RetryStrategy.FIBONACCI: self._calculate_fibonacci_delay
        }
    
    async def retry_async(self, 
                         func: Callable[..., Any], 
                         *args, 
                         context: Optional[ErrorContext] = None,
                         config: Optional[RetryConfig] = None,
                         circuit_breaker_key: Optional[str] = None,
                         **kwargs) -> Any:
        """
        Retry an async function with configured strategy.
        
        Args:
            func: Async function to retry
            *args: Function arguments
            context: Error context for logging
            config: Override default retry configuration
            circuit_breaker_key: Key for circuit breaker (defaults to function name)
            **kwargs: Function keyword arguments
            
        Returns:
            Function result if successful
            
        Raises:
            Last exception if all retries fail
        """
        retry_config = config or self.config
        cb_key = circuit_breaker_key or func.__name__
        
        # Get or create circuit breaker
        if cb_key not in self.circuit_breakers:
            self.circuit_breakers[cb_key] = CircuitBreaker(retry_config)
        
        circuit_breaker = self.circuit_breakers[cb_key]
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker is OPEN for {cb_key}")
        
        last_exception = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Execute the function
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Record success
                circuit_breaker.record_success()
                
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    delay=0.0,
                    exception=None,
                    timestamp=datetime.now(),
                    success=True
                )
                self.statistics.record_attempt(attempt_record)
                
                if attempt > 1:
                    self.logger.info(f"Retry successful for {func.__name__} on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry_exception(e, retry_config):
                    self.logger.info(f"Not retrying {func.__name__} due to non-retryable exception: {e}")
                    circuit_breaker.record_failure()
                    raise e
                
                # Record failure
                circuit_breaker.record_failure()
                
                # Handle error if handler available
                if self.error_handler and context:
                    error_context = ErrorContext(
                        component=context.component,
                        operation=f"{func.__name__}_retry_attempt_{attempt}",
                        channel_id=context.channel_id,
                        feed_url=context.feed_url,
                        user_data={'attempt': attempt, 'max_attempts': retry_config.max_attempts}
                    )
                    self.error_handler.handle_error(e, error_context, attempt_recovery=False)
                
                # Calculate delay for next attempt
                if attempt < retry_config.max_attempts:
                    delay = self._calculate_delay(attempt, retry_config)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt+1}/{retry_config.max_attempts})"
                    )
                    
                    # Record attempt
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        delay=delay,
                        exception=e,
                        timestamp=datetime.now(),
                        success=False
                    )
                    self.statistics.record_attempt(attempt_record)
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        delay=0.0,
                        exception=e,
                        timestamp=datetime.now(),
                        success=False
                    )
                    self.statistics.record_attempt(attempt_record)
                    
                    self.logger.error(f"All {retry_config.max_attempts} attempts failed for {func.__name__}")
        
        # All attempts failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed with no exception recorded")
    
    def retry_sync(self, 
                   func: Callable[..., Any], 
                   *args, 
                   context: Optional[ErrorContext] = None,
                   config: Optional[RetryConfig] = None,
                   circuit_breaker_key: Optional[str] = None,
                   **kwargs) -> Any:
        """
        Retry a synchronous function with configured strategy.
        Similar to retry_async but for sync functions.
        """
        retry_config = config or self.config
        cb_key = circuit_breaker_key or func.__name__
        
        # Get or create circuit breaker
        if cb_key not in self.circuit_breakers:
            self.circuit_breakers[cb_key] = CircuitBreaker(retry_config)
        
        circuit_breaker = self.circuit_breakers[cb_key]
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker is OPEN for {cb_key}")
        
        last_exception = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                
                # Record success
                circuit_breaker.record_success()
                
                attempt_record = RetryAttempt(
                    attempt_number=attempt,
                    delay=0.0,
                    exception=None,
                    timestamp=datetime.now(),
                    success=True
                )
                self.statistics.record_attempt(attempt_record)
                
                if attempt > 1:
                    self.logger.info(f"Retry successful for {func.__name__} on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry_exception(e, retry_config):
                    self.logger.info(f"Not retrying {func.__name__} due to non-retryable exception: {e}")
                    circuit_breaker.record_failure()
                    raise e
                
                # Record failure
                circuit_breaker.record_failure()
                
                # Handle error if handler available
                if self.error_handler and context:
                    error_context = ErrorContext(
                        component=context.component,
                        operation=f"{func.__name__}_retry_attempt_{attempt}",
                        channel_id=context.channel_id,
                        feed_url=context.feed_url,
                        user_data={'attempt': attempt, 'max_attempts': retry_config.max_attempts}
                    )
                    self.error_handler.handle_error(e, error_context, attempt_recovery=False)
                
                # Calculate delay for next attempt
                if attempt < retry_config.max_attempts:
                    delay = self._calculate_delay(attempt, retry_config)
                    
                    self.logger.warning(
                        f"Attempt {attempt} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s (attempt {attempt+1}/{retry_config.max_attempts})"
                    )
                    
                    # Record attempt
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        delay=delay,
                        exception=e,
                        timestamp=datetime.now(),
                        success=False
                    )
                    self.statistics.record_attempt(attempt_record)
                    
                    # Wait before retry
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    attempt_record = RetryAttempt(
                        attempt_number=attempt,
                        delay=0.0,
                        exception=e,
                        timestamp=datetime.now(),
                        success=False
                    )
                    self.statistics.record_attempt(attempt_record)
                    
                    self.logger.error(f"All {retry_config.max_attempts} attempts failed for {func.__name__}")
        
        # All attempts failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed with no exception recorded")
    
    def _should_retry_exception(self, exception: Exception, config: RetryConfig) -> bool:
        """Determine if an exception should trigger a retry."""
        
        # Check exception type
        for retry_exception in config.retry_on_exceptions:
            if isinstance(exception, retry_exception):
                return True
        
        # Check HTTP status codes if applicable
        if hasattr(exception, 'response') and hasattr(exception.response, 'status_code'):
            if exception.response.status_code in config.retry_on_status_codes:
                return True
        
        # Check for specific error messages
        error_message = str(exception).lower()
        retry_messages = [
            'timeout', 'connection', 'network', 'temporary', 'rate limit',
            'server error', 'service unavailable', 'bad gateway'
        ]
        
        if any(msg in error_message for msg in retry_messages):
            return True
        
        return False
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt based on strategy."""
        
        calculator = self._delay_calculators.get(config.strategy, self._calculate_exponential_delay)
        delay = calculator(attempt, config)
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter if enabled, but not for strategies that have inherent randomization
        # or when we want deterministic delays for testing
        if config.jitter and config.strategy not in (RetryStrategy.JITTERED_EXPONENTIAL, RetryStrategy.FIXED_DELAY, RetryStrategy.LINEAR_BACKOFF):
            # Add ±25% jitter
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.1, delay)  # Minimum 0.1 second delay
    
    def _calculate_fixed_delay(self, attempt: int, config: RetryConfig) -> float:
        """Fixed delay strategy."""
        return config.base_delay
    
    def _calculate_exponential_delay(self, attempt: int, config: RetryConfig) -> float:
        """Exponential backoff strategy."""
        return config.base_delay * (config.exponential_base ** (attempt - 1))
    
    def _calculate_linear_delay(self, attempt: int, config: RetryConfig) -> float:
        """Linear backoff strategy."""
        return config.base_delay * attempt
    
    def _calculate_jittered_exponential_delay(self, attempt: int, config: RetryConfig) -> float:
        """Exponential backoff with full jitter."""
        exponential_delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        return random.uniform(0, exponential_delay)
    
    def _calculate_fibonacci_delay(self, attempt: int, config: RetryConfig) -> float:
        """Fibonacci sequence delay strategy."""
        def fibonacci(n):
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        
        return config.base_delay * fibonacci(attempt)
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            key: breaker.get_state_info() 
            for key, breaker in self.circuit_breakers.items()
        }
    
    def reset_circuit_breaker(self, key: str) -> bool:
        """Manually reset a circuit breaker to closed state."""
        if key in self.circuit_breakers:
            breaker = self.circuit_breakers[key]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
            self.logger.info(f"Circuit breaker {key} manually reset to CLOSED")
            return True
        return False
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive retry statistics."""
        return {
            'total_attempts': self.statistics.total_attempts,
            'total_successes': self.statistics.total_successes,
            'total_failures': self.statistics.total_failures,
            'success_rate': self.statistics.success_rate,
            'average_attempts': self.statistics.average_attempts,
            'recent_24h': self.statistics.get_recent_stats(24),
            'recent_1h': self.statistics.get_recent_stats(1),
            'circuit_breakers': self.get_circuit_breaker_status()
        }


# Decorator functions for easy retry application
def with_retry(config: Optional[RetryConfig] = None,
               context: Optional[ErrorContext] = None,
               circuit_breaker_key: Optional[str] = None):
    """Decorator to add retry functionality to async functions."""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            retry_manager = RetryManager(config)
            return await retry_manager.retry_async(
                func, *args, 
                context=context,
                config=config,
                circuit_breaker_key=circuit_breaker_key,
                **kwargs
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            retry_manager = RetryManager(config)
            return retry_manager.retry_sync(
                func, *args,
                context=context,
                config=config,
                circuit_breaker_key=circuit_breaker_key,
                **kwargs
            )
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global retry manager instance
_global_retry_manager: Optional[RetryManager] = None


def get_retry_manager(config: Optional[RetryConfig] = None, 
                     error_handler: Optional[ErrorHandler] = None) -> RetryManager:
    """Get the global retry manager instance."""
    global _global_retry_manager
    
    if _global_retry_manager is None:
        _global_retry_manager = RetryManager(config, error_handler)
    
    return _global_retry_manager