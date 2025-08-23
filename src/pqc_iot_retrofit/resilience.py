"""Generation 2: Resilience and Retry Mechanisms.

Advanced resilience patterns providing:
- Exponential backoff retry strategies
- Circuit breaker patterns
- Timeout management
- Graceful degradation
- Failure isolation and recovery
"""

import time
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retriable_exceptions: List[Type[Exception]] = None
    non_retriable_exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retriable_exceptions is None:
            # Default retriable exceptions
            self.retriable_exceptions = [
                ConnectionError,
                TimeoutError,
                IOError,
                OSError
            ]
        
        if self.non_retriable_exceptions is None:
            # Exceptions that should not be retried
            self.non_retriable_exceptions = [
                ValueError,
                TypeError,
                AttributeError,
                KeyError
            ]


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2  # For half-open state
    timeout: float = 30.0


class RetryableError(Exception):
    """Exception that can be retried."""
    pass


class NonRetryableError(Exception):
    """Exception that should not be retried."""
    pass


class CircuitOpenError(Exception):
    """Exception thrown when circuit is open."""
    pass


class RetryHandler:
    """Advanced retry mechanism with multiple strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def _is_retriable_exception(self, exception: Exception) -> bool:
        """Check if exception should trigger a retry."""
        
        # Non-retriable exceptions take precedence
        for exc_type in self.config.non_retriable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # Check retriable exceptions
        for exc_type in self.config.retriable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        # Special handling for custom exceptions
        if isinstance(exception, NonRetryableError):
            return False
        
        if isinstance(exception, RetryableError):
            return True
        
        # Default to not retriable for unknown exceptions
        return False
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
            
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
            
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
            
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            # Generate fibonacci sequence
            if attempt <= 2:
                delay = self.config.base_delay
            else:
                fib_a, fib_b = 1, 1
                for _ in range(attempt - 2):
                    fib_a, fib_b = fib_b, fib_a + fib_b
                delay = self.config.base_delay * fib_b
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                logger.debug(f"Attempt {attempt}/{self.config.max_attempts} for {func.__name__}")
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should be retried
                if not self._is_retriable_exception(e):
                    logger.error(f"Non-retriable exception in {func.__name__}: {e}")
                    raise e
                
                # Don't sleep after the last attempt
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt} failed for {func.__name__}: {e}. "
                                 f"Retrying in {delay:.2f}s")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.config.max_attempts} attempts failed for {func.__name__}")
        
        # All attempts exhausted
        raise last_exception


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset."""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
            else:
                self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("Circuit breaker reset to closed state")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        
        with self.lock:
            current_state = self.state
        
        # Check circuit state
        if current_state == CircuitState.OPEN:
            if self._should_attempt_reset():
                with self.lock:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker moved to half-open state")
                current_state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time
        }


class TimeoutManager:
    """Timeout management for operations."""
    
    @staticmethod
    def with_timeout(timeout_seconds: float):
        """Decorator to add timeout to function execution."""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds}s")
                
                # Set timeout handler
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout_seconds))
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Restore old handler and cancel alarm
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            return wrapper
        return decorator


class ResilientExecutor:
    """Combined resilience patterns executor."""
    
    def __init__(self, 
                 retry_config: Optional[RetryConfig] = None,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 timeout_seconds: Optional[float] = None):
        
        self.retry_handler = RetryHandler(retry_config or RetryConfig())
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.timeout_seconds = timeout_seconds
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with all resilience patterns."""
        
        def execute_with_circuit_breaker():
            return self.circuit_breaker.call(func, *args, **kwargs)
        
        def execute_with_timeout():
            if self.timeout_seconds:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {self.timeout_seconds}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout_seconds))
                
                try:
                    return execute_with_circuit_breaker()
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                return execute_with_circuit_breaker()
        
        # Execute with retry logic
        return self.retry_handler.execute(execute_with_timeout)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all resilience components."""
        return {
            'circuit_breaker': self.circuit_breaker.get_state(),
            'retry_config': {
                'max_attempts': self.retry_handler.config.max_attempts,
                'strategy': self.retry_handler.config.strategy.value
            },
            'timeout_seconds': self.timeout_seconds
        }


class GracefulDegradation:
    """Graceful degradation patterns."""
    
    @staticmethod
    def with_fallback(fallback_func: Callable):
        """Decorator to provide fallback function on failure."""
        
        def decorator(main_func: Callable) -> Callable:
            @wraps(main_func)
            def wrapper(*args, **kwargs):
                try:
                    return main_func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Main function {main_func.__name__} failed: {e}. "
                                 f"Using fallback {fallback_func.__name__}")
                    return fallback_func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def with_default(default_value: Any):
        """Decorator to return default value on failure."""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Function {func.__name__} failed: {e}. "
                                 f"Returning default value: {default_value}")
                    return default_value
            
            return wrapper
        return decorator


# Decorators for easy use
def resilient(retry_config: Optional[RetryConfig] = None,
             circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
             timeout_seconds: Optional[float] = None):
    """Decorator to make function resilient with all patterns."""
    
    def decorator(func: Callable) -> Callable:
        executor = ResilientExecutor(retry_config, circuit_breaker_config, timeout_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return executor.execute(func, *args, **kwargs)
        
        # Attach status method to wrapper
        wrapper.get_resilience_status = executor.get_status
        
        return wrapper
    
    return decorator


def retry(max_attempts: int = 3, 
         strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
         base_delay: float = 1.0,
         max_delay: float = 60.0):
    """Simple retry decorator."""
    
    config = RetryConfig(
        max_attempts=max_attempts,
        strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay
    )
    
    def decorator(func: Callable) -> Callable:
        retry_handler = RetryHandler(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.execute(func, *args, **kwargs)
        
        return wrapper
    
    return decorator


def circuit_breaker(failure_threshold: int = 5,
                   recovery_timeout: float = 60.0):
    """Simple circuit breaker decorator."""
    
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    
    breaker = CircuitBreaker(config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach status method
        wrapper.get_circuit_state = breaker.get_state
        
        return wrapper
    
    return decorator


# Health check utilities
class HealthChecker:
    """System health checking utilities."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                duration = time.time() - start_time
                
                results[name] = {
                    'healthy': bool(is_healthy),
                    'duration_ms': round(duration * 1000, 2),
                    'timestamp': time.time()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                overall_healthy = False
        
        results['overall'] = {
            'healthy': overall_healthy,
            'total_checks': len(self.checks),
            'passed_checks': sum(1 for r in results.values() if isinstance(r, dict) and r.get('healthy')),
            'timestamp': time.time()
        }
        
        return results


# Global health checker instance
global_health_checker = HealthChecker()

def health_check(name: str):
    """Decorator to register function as health check."""
    
    def decorator(func: Callable) -> Callable:
        global_health_checker.register_check(name, func)
        return func
    
    return decorator


# Export main components
__all__ = [
    'RetryStrategy', 'CircuitState', 'RetryConfig', 'CircuitBreakerConfig',
    'RetryableError', 'NonRetryableError', 'CircuitOpenError',
    'RetryHandler', 'CircuitBreaker', 'TimeoutManager', 'ResilientExecutor',
    'GracefulDegradation', 'HealthChecker',
    'resilient', 'retry', 'circuit_breaker', 'health_check',
    'global_health_checker'
]