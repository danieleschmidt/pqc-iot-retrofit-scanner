#!/usr/bin/env python3
"""
Robust Error Handling System - Generation 2
Comprehensive error handling, logging, and monitoring for PQC scanner
"""

import os
import sys
import json
import logging
import traceback
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import wraps
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Error categories for classification."""
    FIRMWARE_ANALYSIS = "firmware_analysis"
    CRYPTO_DETECTION = "crypto_detection"
    PATCH_GENERATION = "patch_generation"
    FILE_IO = "file_io"
    MEMORY_CONSTRAINT = "memory_constraint"
    VALIDATION = "validation"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass
class ErrorDetails:
    """Detailed error information."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    mitigation: Optional[str] = None
    user_action: Optional[str] = None


class RobustErrorHandler:
    """Comprehensive error handling system."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize error handler with logging."""
        self.errors: List[ErrorDetails] = []
        self.error_counts: Dict[str, int] = {}
        self.setup_logging(log_level)
        self.mitigation_strategies = self._load_mitigation_strategies()
    
    def setup_logging(self, log_level: int):
        """Setup structured logging."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        
        # File handler
        file_handler = logging.FileHandler('logs/pqc_scanner.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Setup logger
        self.logger = logging.getLogger('PQCScanner')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Error handling system initialized")
    
    def _load_mitigation_strategies(self) -> Dict[str, Dict[str, str]]:
        """Load error mitigation strategies."""
        return {
            ErrorCategory.FIRMWARE_ANALYSIS.value: {
                "corrupt_firmware": "Verify file integrity and try alternative parsing methods",
                "unsupported_architecture": "Use generic analysis mode or add architecture support",
                "memory_limit_exceeded": "Process firmware in smaller chunks"
            },
            ErrorCategory.CRYPTO_DETECTION.value: {
                "false_positive": "Refine detection patterns and add validation",
                "pattern_not_found": "Update signature database or use heuristic analysis",
                "ambiguous_algorithm": "Use multiple detection methods for confirmation"
            },
            ErrorCategory.PATCH_GENERATION.value: {
                "insufficient_memory": "Use lightweight PQC variants or hybrid mode",
                "compilation_failed": "Check compiler compatibility and dependencies",
                "patch_too_large": "Enable code compression and optimization"
            },
            ErrorCategory.FILE_IO.value: {
                "permission_denied": "Check file permissions and run with appropriate privileges",
                "file_not_found": "Verify file path and existence",
                "disk_full": "Free up disk space or use temporary storage"
            },
            ErrorCategory.MEMORY_CONSTRAINT.value: {
                "flash_overflow": "Optimize code size or increase flash capacity",
                "ram_overflow": "Use memory-efficient algorithms or external storage",
                "stack_overflow": "Reduce recursion depth and optimize stack usage"
            }
        }
    
    def handle_error(self, error: Exception, severity: ErrorSeverity, 
                    category: ErrorCategory, context: Dict[str, Any] = None) -> ErrorDetails:
        """Handle and log an error with full context."""
        if context is None:
            context = {}
        
        # Generate unique error ID
        error_id = hashlib.md5(
            f"{str(error)}{time.time()}{category.value}".encode()
        ).hexdigest()[:8]
        
        # Get mitigation strategy
        mitigation = self._get_mitigation_strategy(category, str(error))
        user_action = self._get_user_action(severity, category)
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=str(error),
            context=context,
            stack_trace=traceback.format_exc() if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH] else None,
            mitigation=mitigation,
            user_action=user_action
        )
        
        # Store error
        self.errors.append(error_details)
        self.error_counts[category.value] = self.error_counts.get(category.value, 0) + 1
        
        # Log error
        self._log_error(error_details)
        
        return error_details
    
    def _get_mitigation_strategy(self, category: ErrorCategory, error_msg: str) -> str:
        """Get mitigation strategy for error."""
        strategies = self.mitigation_strategies.get(category.value, {})
        
        # Try to match error message to known patterns
        error_lower = error_msg.lower()
        for pattern, strategy in strategies.items():
            if pattern.replace('_', ' ') in error_lower:
                return strategy
        
        # Default mitigation
        return f"Review error details and consult documentation for {category.value} issues"
    
    def _get_user_action(self, severity: ErrorSeverity, category: ErrorCategory) -> str:
        """Get recommended user action."""
        if severity == ErrorSeverity.CRITICAL:
            return "CRITICAL: Stop processing and address immediately"
        elif severity == ErrorSeverity.HIGH:
            return "HIGH: Review error and take corrective action"
        elif severity == ErrorSeverity.MEDIUM:
            return "MEDIUM: Monitor and address when convenient"
        else:
            return "LOW: Log for reference, continue processing"
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level."""
        log_msg = f"[{error_details.error_id}] {error_details.message}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_msg)
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_msg)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)
        
        # Log context if available
        if error_details.context:
            self.logger.debug(f"[{error_details.error_id}] Context: {json.dumps(error_details.context, indent=2)}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        if not self.errors:
            return {"total_errors": 0, "summary": "No errors recorded"}
        
        severity_counts = {}
        category_counts = {}
        
        for error in self.errors:
            # Count by severity
            sev = error.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            
            # Count by category
            cat = error.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        recent_errors = [
            {
                "error_id": e.error_id,
                "severity": e.severity.value,
                "category": e.category.value,
                "message": e.message
            }
            for e in self.errors[-5:]  # Last 5 errors
        ]
        
        return {
            "total_errors": len(self.errors),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "recent_errors": recent_errors,
            "error_rate": len(self.errors) / max(1, time.time() - self.errors[0].timestamp if self.errors else 1)
        }
    
    def export_error_report(self, output_path: str):
        """Export comprehensive error report."""
        report = {
            "report_timestamp": time.time(),
            "summary": self.get_error_summary(),
            "detailed_errors": [asdict(error) for error in self.errors],
            "mitigation_applied": self.mitigation_strategies
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Error report exported to: {output_path}")


def robust_error_decorator(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          category: ErrorCategory = ErrorCategory.SYSTEM,
                          retry_count: int = 0,
                          fallback_value: Any = None):
    """Decorator for robust error handling with retry logic."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = getattr(wrapper, '_error_handler', None)
            if error_handler is None:
                error_handler = RobustErrorHandler()
                wrapper._error_handler = error_handler
            
            last_error = None
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    
                    context = {
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "max_attempts": retry_count + 1,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                    
                    error_details = error_handler.handle_error(e, severity, category, context)
                    
                    if attempt < retry_count:
                        error_handler.logger.info(f"Retrying {func.__name__} (attempt {attempt + 2})")
                        time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    else:
                        error_handler.logger.error(f"All retry attempts failed for {func.__name__}")
            
            # If all retries failed, return fallback value or re-raise
            if fallback_value is not None:
                error_handler.logger.warning(f"Using fallback value for {func.__name__}")
                return fallback_value
            else:
                raise last_error
        
        return wrapper
    return decorator


class InputValidator:
    """Robust input validation system."""
    
    @staticmethod
    def validate_firmware_file(file_path: str) -> bool:
        """Validate firmware file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Firmware file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("Firmware file is empty")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise ValueError(f"Firmware file too large: {file_size / (1024*1024):.1f}MB")
        
        # Check file type (basic heuristic)
        with open(file_path, 'rb') as f:
            header = f.read(16)
            if len(header) < 4:
                raise ValueError("Invalid firmware file: too small")
        
        return True
    
    @staticmethod
    def validate_memory_constraints(flash_size: int, ram_size: int) -> bool:
        """Validate memory constraints."""
        if flash_size <= 0:
            raise ValueError("Flash size must be positive")
        
        if ram_size <= 0:
            raise ValueError("RAM size must be positive")
        
        if flash_size < 32 * 1024:  # 32KB minimum
            raise ValueError("Flash size too small for PQC implementations")
        
        if ram_size < 4 * 1024:  # 4KB minimum
            raise ValueError("RAM size too small for PQC operations")
        
        return True
    
    @staticmethod
    def validate_pqc_algorithm(algorithm: str) -> bool:
        """Validate PQC algorithm selection."""
        valid_algorithms = ['dilithium2', 'dilithium3', 'kyber512', 'kyber768', 'kyber1024']
        
        if algorithm not in valid_algorithms:
            raise ValueError(f"Unsupported PQC algorithm: {algorithm}")
        
        return True


class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'error_rate': 0.0,
            'processing_speed': 0.0
        }
        self.health_status = "healthy"
        self.last_check = time.time()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        try:
            # Simplified health checks (would use psutil in production)
            self.metrics['memory_usage'] = self._check_memory_usage()
            self.metrics['disk_usage'] = self._check_disk_usage()
            self.metrics['error_rate'] = self._check_error_rate()
            
            # Determine health status
            if self.metrics['memory_usage'] > 90 or self.metrics['disk_usage'] > 95:
                self.health_status = "critical"
            elif self.metrics['memory_usage'] > 80 or self.metrics['disk_usage'] > 85:
                self.health_status = "warning"
            else:
                self.health_status = "healthy"
            
            self.last_check = time.time()
            
            return {
                "status": self.health_status,
                "metrics": self.metrics,
                "last_check": self.last_check,
                "recommendations": self._get_health_recommendations()
            }
        
        except Exception as e:
            return {
                "status": "unknown",
                "error": str(e),
                "last_check": self.last_check
            }
    
    def _check_memory_usage(self) -> float:
        """Check memory usage percentage."""
        # Simplified check - would use actual system metrics in production
        return 25.0  # Mock 25% usage
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        # Simplified check
        return 45.0  # Mock 45% usage
    
    def _check_error_rate(self) -> float:
        """Check error rate per minute."""
        # Would calculate from actual error logs
        return 0.5  # Mock 0.5 errors per minute
    
    def _get_health_recommendations(self) -> List[str]:
        """Get health improvement recommendations."""
        recommendations = []
        
        if self.metrics['memory_usage'] > 80:
            recommendations.append("Consider reducing memory usage or increasing available RAM")
        
        if self.metrics['disk_usage'] > 80:
            recommendations.append("Free up disk space or add storage capacity")
        
        if self.metrics['error_rate'] > 1.0:
            recommendations.append("Investigate and resolve recurring errors")
        
        if not recommendations:
            recommendations.append("System health is optimal")
        
        return recommendations


def main():
    """Demo of robust error handling system."""
    print("Robust Error Handling System - Demo")
    print("=" * 50)
    
    # Initialize error handler
    error_handler = RobustErrorHandler()
    
    # Initialize health monitor
    health_monitor = HealthMonitor()
    
    # Demo error handling
    try:
        # Simulate various errors
        
        # File not found error
        try:
            raise FileNotFoundError("Firmware file missing: /path/to/firmware.bin")
        except Exception as e:
            error_handler.handle_error(
                e, ErrorSeverity.HIGH, ErrorCategory.FILE_IO,
                {"file_path": "/path/to/firmware.bin", "operation": "read"}
            )
        
        # Memory constraint error
        try:
            raise ValueError("Insufficient flash memory: 32KB required, 16KB available")
        except Exception as e:
            error_handler.handle_error(
                e, ErrorSeverity.CRITICAL, ErrorCategory.MEMORY_CONSTRAINT,
                {"required_flash": 32768, "available_flash": 16384}
            )
        
        # Crypto detection error
        try:
            raise RuntimeError("Ambiguous cryptographic pattern detected")
        except Exception as e:
            error_handler.handle_error(
                e, ErrorSeverity.MEDIUM, ErrorCategory.CRYPTO_DETECTION,
                {"pattern": "RSA-like", "confidence": 0.65}
            )
        
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # Show error summary
    print("\nError Summary:")
    summary = error_handler.get_error_summary()
    print(json.dumps(summary, indent=2))
    
    # Check system health
    print("\nSystem Health Check:")
    health = health_monitor.check_system_health()
    print(json.dumps(health, indent=2))
    
    # Export error report
    error_handler.export_error_report("logs/error_report.json")
    print("\nError report exported to logs/error_report.json")
    
    # Demo retry decorator
    @robust_error_decorator(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.VALIDATION,
        retry_count=2,
        fallback_value="fallback_result"
    )
    def unreliable_function():
        import random
        if random.random() < 0.7:  # 70% chance of failure
            raise RuntimeError("Random failure for demo")
        return "success"
    
    print("\nTesting retry decorator:")
    result = unreliable_function()
    print(f"Result: {result}")


if __name__ == '__main__':
    main()