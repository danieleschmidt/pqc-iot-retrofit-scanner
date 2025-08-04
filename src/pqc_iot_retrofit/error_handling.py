"""
Comprehensive error handling and validation for PQC IoT Retrofit Scanner.

This module provides:
- Custom exception classes for different error scenarios
- Input validation utilities
- Error recovery mechanisms
- Circuit breaker patterns for resilience
"""

import functools
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    CRYPTO = "crypto"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    EXTERNAL_DEPENDENCY = "external_dependency"
    INTERNAL_ERROR = "internal_error"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    input_data: Dict[str, Any]
    system_state: Dict[str, Any]
    timestamp: float
    stack_trace: Optional[str] = None


class PQCRetrofitError(Exception):
    """Base exception for PQC Retrofit Scanner."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.INTERNAL_ERROR,
                 context: Optional[ErrorContext] = None,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context
        self.recoverable = recoverable
        self.timestamp = time.time()


class ValidationError(PQCRetrofitError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message, 
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            recoverable=True
        )
        self.field = field
        self.value = value


class FirmwareAnalysisError(PQCRetrofitError):
    """Firmware analysis specific errors."""
    
    def __init__(self, message: str, firmware_path: str = None, address: int = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CRYPTO,
            recoverable=False
        )
        self.firmware_path = firmware_path
        self.address = address


class PQCImplementationError(PQCRetrofitError):
    """PQC implementation generation errors."""
    
    def __init__(self, message: str, algorithm: str = None, target_arch: str = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CRYPTO,
            recoverable=True
        )
        self.algorithm = algorithm
        self.target_arch = target_arch


class BinaryPatchingError(PQCRetrofitError):
    """Binary patching errors."""
    
    def __init__(self, message: str, patch_address: int = None, patch_type: str = None):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.FILESYSTEM,
            recoverable=False
        )
        self.patch_address = patch_address
        self.patch_type = patch_type


class MemoryConstraintError(PQCRetrofitError):
    """Memory constraint violations."""
    
    def __init__(self, message: str, required: int = None, available: int = None,
                 constraint_type: str = "memory"):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            recoverable=True
        )
        self.required = required
        self.available = available
        self.constraint_type = constraint_type


class ExternalDependencyError(PQCRetrofitError):
    """External dependency failures."""
    
    def __init__(self, message: str, dependency: str = None, operation: str = None):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_DEPENDENCY,
            recoverable=True
        )
        self.dependency = dependency
        self.operation = operation


class ConfigurationError(PQCRetrofitError):
    """Configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, config_value: Any = None):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recoverable=True
        )
        self.config_key = config_key
        self.config_value = config_value


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[PQCRetrofitError] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        self.recovery_strategies[ErrorCategory.EXTERNAL_DEPENDENCY] = self._retry_with_backoff
        self.recovery_strategies[ErrorCategory.NETWORK] = self._retry_with_backoff
        self.recovery_strategies[ErrorCategory.MEMORY] = self._reduce_memory_usage
        self.recovery_strategies[ErrorCategory.CONFIGURATION] = self._reload_configuration
    
    def handle_error(self, error: PQCRetrofitError, operation: str = None) -> bool:
        """
        Handle an error with appropriate logging and recovery.
        
        Returns:
            True if error was recovered, False otherwise
        """
        # Log the error
        self._log_error(error, operation)
        
        # Record error in history
        self.error_history.append(error)
        
        # Attempt recovery if error is recoverable
        if error.recoverable and error.category in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error.category]
                return recovery_func(error)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error}: {recovery_error}")
                return False
        
        return False
    
    def _log_error(self, error: PQCRetrofitError, operation: str = None):
        """Log error with appropriate level and context."""
        log_message = f"[{error.category.value.upper()}] {error.message}"
        
        if operation:
            log_message = f"{operation}: {log_message}"
        
        # Add context if available
        if error.context:
            log_message += f" (Operation: {error.context.operation}, Timestamp: {error.context.timestamp})"
        
        # Log with appropriate level
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _retry_with_backoff(self, error: PQCRetrofitError) -> bool:
        """Retry operation with exponential backoff."""
        operation = error.context.operation if error.context else "unknown"
        
        # Get or create circuit breaker
        circuit_breaker = self.circuit_breakers.get(operation)
        if not circuit_breaker:
            circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
            self.circuit_breakers[operation] = circuit_breaker
        
        if circuit_breaker.can_execute():
            self.logger.info(f"Attempting retry for {operation}")
            return True
        else:
            self.logger.warning(f"Circuit breaker open for {operation}, skipping retry")
            return False
    
    def _reduce_memory_usage(self, error: MemoryConstraintError) -> bool:
        """Attempt to reduce memory usage."""
        self.logger.info("Attempting memory optimization...")
        
        # This would implement actual memory reduction strategies
        # For now, just log the attempt
        if hasattr(error, 'required') and hasattr(error, 'available'):
            reduction_needed = error.required - error.available
            self.logger.info(f"Need to reduce memory usage by {reduction_needed} bytes")
        
        return True
    
    def _reload_configuration(self, error: ConfigurationError) -> bool:
        """Attempt to reload configuration."""
        self.logger.info("Attempting configuration reload...")
        
        # This would implement actual config reload
        # For now, just log the attempt
        if hasattr(error, 'config_key'):
            self.logger.info(f"Reloading configuration key: {error.config_key}")
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_history:
            return {"total_errors": 0}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate error rate (errors per hour)
        if self.error_history:
            time_span = time.time() - self.error_history[0].timestamp
            error_rate = len(self.error_history) / max(time_span / 3600, 0.01)  # Avoid division by zero
        else:
            error_rate = 0
        
        return {
            "total_errors": len(self.error_history),
            "error_rate_per_hour": error_rate,
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "recent_errors": [
                {
                    "message": error.message,
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "timestamp": error.timestamp
                }
                for error in self.error_history[-10:]  # Last 10 errors
            ]
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation for resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if operation can be executed."""
        current_time = time.time()
        
        if self.state == "open":
            if current_time - self.last_failure_time >= self.timeout:
                self.state = "half-open"
                return True
            return False
        
        return True
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Validation utilities
class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_firmware_path(path: str) -> bool:
        """Validate firmware file path."""
        if not path:
            raise ValidationError("Firmware path cannot be empty", field="firmware_path", value=path)
        
        if not isinstance(path, str):
            raise ValidationError("Firmware path must be a string", field="firmware_path", value=path)
        
        # Check file extension
        valid_extensions = ['.bin', '.elf', '.hex', '.fw', '.img']
        if not any(path.lower().endswith(ext) for ext in valid_extensions):
            raise ValidationError(
                f"Firmware file must have one of: {valid_extensions}",
                field="firmware_path",
                value=path
            )
        
        return True
    
    @staticmethod
    def validate_architecture(arch: str) -> bool:
        """Validate target architecture."""
        valid_architectures = [
            'cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7',
            'esp32', 'riscv32', 'avr'
        ]
        
        if not arch:
            raise ValidationError("Architecture cannot be empty", field="architecture", value=arch)
        
        if arch not in valid_architectures:
            raise ValidationError(
                f"Architecture must be one of: {valid_architectures}",
                field="architecture",
                value=arch
            )
        
        return True
    
    @staticmethod
    def validate_memory_constraints(constraints: Dict[str, int]) -> bool:
        """Validate memory constraints."""
        if not isinstance(constraints, dict):
            raise ValidationError("Memory constraints must be a dictionary", 
                                field="memory_constraints", value=constraints)
        
        required_keys = ['flash', 'ram']
        for key in required_keys:
            if key not in constraints:
                raise ValidationError(f"Missing required constraint: {key}",
                                    field="memory_constraints", value=constraints)
            
            if not isinstance(constraints[key], int) or constraints[key] <= 0:
                raise ValidationError(f"Constraint {key} must be a positive integer",
                                    field="memory_constraints", value=constraints[key])
        
        # Sanity checks
        if constraints['flash'] < 1024:  # Less than 1KB
            raise ValidationError("Flash constraint too small (minimum 1KB)",
                                field="memory_constraints", value=constraints)
        
        if constraints['ram'] < 512:  # Less than 512 bytes
            raise ValidationError("RAM constraint too small (minimum 512 bytes)",
                                field="memory_constraints", value=constraints)
        
        return True
    
    @staticmethod
    def validate_address(address: Union[int, str]) -> int:
        """Validate and normalize memory address."""
        if isinstance(address, str):
            try:
                # Handle hex strings
                if address.startswith('0x') or address.startswith('0X'):
                    address = int(address, 16)
                else:
                    address = int(address)
            except ValueError:
                raise ValidationError(f"Invalid address format: {address}",
                                    field="address", value=address)
        
        if not isinstance(address, int):
            raise ValidationError("Address must be an integer",
                                field="address", value=address)
        
        if address < 0:
            raise ValidationError("Address cannot be negative",
                                field="address", value=address)
        
        if address > 0xFFFFFFFF:  # 32-bit address limit
            raise ValidationError("Address exceeds 32-bit limit",
                                field="address", value=address)
        
        return address


# Decorator for automatic error handling
def handle_errors(operation_name: str = None, 
                 retry_count: int = 0,
                 fallback_value: Any = None):
    """Decorator for automatic error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except PQCRetrofitError as e:
                    # Add context
                    if not e.context:
                        e.context = ErrorContext(
                            operation=op_name,
                            input_data={"args": str(args), "kwargs": str(kwargs)},
                            system_state={},
                            timestamp=time.time()
                        )
                    
                    # Handle the error
                    recovered = error_handler.handle_error(e, op_name)
                    
                    if not recovered and attempt < retry_count:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    
                    if not recovered and fallback_value is not None:
                        return fallback_value
                    
                    raise
                except Exception as e:
                    # Convert to PQCRetrofitError
                    pqc_error = PQCRetrofitError(
                        message=str(e),
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.INTERNAL_ERROR,
                        context=ErrorContext(
                            operation=op_name,
                            input_data={"args": str(args), "kwargs": str(kwargs)},
                            system_state={},
                            timestamp=time.time(),
                            stack_trace=str(e.__traceback__)
                        ),
                        recoverable=False
                    )
                    
                    error_handler.handle_error(pqc_error, op_name)
                    
                    if fallback_value is not None:
                        return fallback_value
                    
                    raise pqc_error
        
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def get_error_statistics() -> Dict[str, Any]:
    """Get global error statistics."""
    return global_error_handler.get_error_statistics()


def reset_error_history():
    """Reset global error history (useful for testing)."""
    global_error_handler.error_history.clear()
    global_error_handler.circuit_breakers.clear()