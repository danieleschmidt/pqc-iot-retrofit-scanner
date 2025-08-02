"""
Structured logging configuration for PQC IoT Retrofit Scanner.

Provides JSON-structured logging with correlation IDs, security context,
and comprehensive audit trails for compliance and debugging.
"""

import os
import sys
import json
import time
import uuid
import logging
import logging.config
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextvars import ContextVar
from enum import Enum
import traceback
from pathlib import Path


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class SecurityEventType(Enum):
    """Security event types for audit logging."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VULNERABILITY_DETECTED = "vulnerability_detected"
    PATCH_GENERATED = "patch_generated"
    CRYPTO_OPERATION = "crypto_operation"
    ACCESS_DENIED = "access_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_SCAN = "security_scan"


@dataclass
class SecurityContext:
    """Security context for audit logging."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    api_key_id: Optional[str] = None
    permissions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PerformanceContext:
    """Performance context for monitoring."""
    duration_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    io_operations: Optional[int] = None
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# Context variables for request tracing
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default="")
request_id_var: ContextVar[str] = ContextVar('request_id', default="")
security_context_var: ContextVar[SecurityContext] = ContextVar('security_context', default=SecurityContext())


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, service_name: str = "pqc-iot-retrofit-scanner"):
        """Initialize structured formatter.
        
        Args:
            service_name: Name of the service for log identification
        """
        super().__init__()
        self.service_name = service_name
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log entry
        log_entry = {
            'timestamp': time.time(),
            'datetime': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'service': self.service_name,
            'hostname': self.hostname,
            'pid': os.getpid(),
            'thread': record.thread,
            'thread_name': record.threadName,
        }
        
        # Add correlation and request IDs if available
        correlation_id = correlation_id_var.get("")
        if correlation_id:
            log_entry['correlation_id'] = correlation_id
        
        request_id = request_id_var.get("")
        if request_id:
            log_entry['request_id'] = request_id
        
        # Add security context if available
        security_context = security_context_var.get(SecurityContext())
        if security_context.user_id or security_context.session_id:
            log_entry['security'] = security_context.to_dict()
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info'}:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class SecurityAuditLogger:
    """Specialized logger for security events and audit trails."""
    
    def __init__(self, logger_name: str = "pqc.security.audit"):
        """Initialize security audit logger.
        
        Args:
            logger_name: Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
        self.enabled = os.getenv("PQC_AUDIT_LOGGING", "true").lower() == "true"
    
    def log_security_event(self, event_type: SecurityEventType, message: str,
                          details: Optional[Dict[str, Any]] = None,
                          severity: str = "INFO"):
        """Log a security event.
        
        Args:
            event_type: Type of security event
            message: Human-readable message
            details: Additional event details
            severity: Log severity level
        """
        if not self.enabled:
            return
        
        log_data = {
            'event_type': event_type.value,
            'security_event': True,
            'audit': True,
            'details': details or {}
        }
        
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(log_level, message, extra=log_data)
    
    def log_vulnerability_detection(self, algorithm: str, confidence: float,
                                   firmware_path: str, details: Dict[str, Any]):
        """Log vulnerability detection event."""
        self.log_security_event(
            SecurityEventType.VULNERABILITY_DETECTED,
            f"Vulnerability detected: {algorithm} (confidence: {confidence:.2f})",
            {
                'algorithm': algorithm,
                'confidence': confidence,
                'firmware_path': firmware_path,
                **details
            },
            severity="WARNING"
        )
    
    def log_patch_generation(self, algorithm: str, target_device: str,
                            patch_id: str, success: bool):
        """Log patch generation event."""
        self.log_security_event(
            SecurityEventType.PATCH_GENERATED,
            f"Patch {'generated' if success else 'generation failed'}: {algorithm} for {target_device}",
            {
                'algorithm': algorithm,
                'target_device': target_device,
                'patch_id': patch_id,
                'success': success
            },
            severity="INFO" if success else "ERROR"
        )
    
    def log_crypto_operation(self, operation: str, algorithm: str,
                           duration_ms: float, success: bool):
        """Log cryptographic operation."""
        self.log_security_event(
            SecurityEventType.CRYPTO_OPERATION,
            f"Crypto operation: {operation} with {algorithm} ({'success' if success else 'failed'})",
            {
                'operation': operation,
                'algorithm': algorithm,
                'duration_ms': duration_ms,
                'success': success
            },
            severity="DEBUG"
        )
    
    def log_access_attempt(self, resource: str, action: str, allowed: bool,
                          reason: Optional[str] = None):
        """Log access attempt."""
        event_type = SecurityEventType.AUTHORIZATION if allowed else SecurityEventType.ACCESS_DENIED
        self.log_security_event(
            event_type,
            f"Access {'granted' if allowed else 'denied'}: {action} on {resource}",
            {
                'resource': resource,
                'action': action,
                'allowed': allowed,
                'reason': reason
            },
            severity="INFO" if allowed else "WARNING"
        )
    
    def log_suspicious_activity(self, activity_type: str, description: str,
                               risk_score: float, details: Dict[str, Any]):
        """Log suspicious activity."""
        self.log_security_event(
            SecurityEventType.SUSPICIOUS_ACTIVITY,
            f"Suspicious activity detected: {activity_type}",
            {
                'activity_type': activity_type,
                'description': description,
                'risk_score': risk_score,
                **details
            },
            severity="WARNING" if risk_score < 0.7 else "ERROR"
        )


class PerformanceLogger:
    """Logger for performance monitoring and optimization."""
    
    def __init__(self, logger_name: str = "pqc.performance"):
        """Initialize performance logger.
        
        Args:
            logger_name: Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
        self.enabled = os.getenv("PQC_PERFORMANCE_LOGGING", "true").lower() == "true"
    
    def log_operation_performance(self, operation: str, context: PerformanceContext,
                                 threshold_ms: Optional[float] = None):
        """Log operation performance metrics.
        
        Args:
            operation: Name of the operation
            context: Performance context data
            threshold_ms: Performance threshold for warnings
        """
        if not self.enabled:
            return
        
        log_data = {
            'performance_event': True,
            'operation': operation,
            **context.to_dict()
        }
        
        # Determine log level based on performance
        severity = "DEBUG"
        if threshold_ms and context.duration_ms and context.duration_ms > threshold_ms:
            severity = "WARNING"
        
        message = f"Performance: {operation} took {context.duration_ms:.2f}ms"
        if context.memory_mb:
            message += f", used {context.memory_mb:.1f}MB"
        
        log_level = getattr(logging, severity, logging.DEBUG)
        self.logger.log(log_level, message, extra=log_data)
    
    def log_resource_usage(self, resource_type: str, usage_percent: float,
                          threshold_percent: float = 80.0):
        """Log resource usage metrics.
        
        Args:
            resource_type: Type of resource (cpu, memory, disk)
            usage_percent: Current usage percentage
            threshold_percent: Warning threshold
        """
        if not self.enabled:
            return
        
        log_data = {
            'resource_event': True,
            'resource_type': resource_type,
            'usage_percent': usage_percent,
            'threshold_percent': threshold_percent
        }
        
        severity = "WARNING" if usage_percent > threshold_percent else "DEBUG"
        message = f"Resource usage: {resource_type} at {usage_percent:.1f}%"
        
        log_level = getattr(logging, severity, logging.DEBUG)
        self.logger.log(log_level, message, extra=log_data)


def setup_logging(config_path: Optional[str] = None) -> None:
    """Setup structured logging configuration.
    
    Args:
        config_path: Path to logging configuration file
    """
    # Default logging configuration
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'structured': {
                '()': StructuredFormatter,
                'service_name': 'pqc-iot-retrofit-scanner'
            },
            'simple': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'structured' if os.getenv('PQC_LOG_FORMAT', 'structured') == 'structured' else 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.getenv('PQC_LOG_FILE', 'logs/pqc-scanner.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 5,
                'formatter': 'structured'
            },
            'audit_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': os.getenv('PQC_AUDIT_LOG_FILE', 'logs/pqc-audit.log'),
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
                'formatter': 'structured'
            }
        },
        'loggers': {
            'pqc_iot_retrofit': {
                'level': os.getenv('PQC_LOG_LEVEL', 'INFO'),
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'pqc.security.audit': {
                'level': 'INFO',
                'handlers': ['console', 'audit_file'],
                'propagate': False
            },
            'pqc.performance': {
                'level': os.getenv('PQC_PERFORMANCE_LOG_LEVEL', 'DEBUG'),
                'handlers': ['console', 'file'],
                'propagate': False
            }
        },
        'root': {
            'level': 'WARNING',
            'handlers': ['console']
        }
    }
    
    # Create log directories
    log_file = default_config['handlers']['file']['filename']
    audit_log_file = default_config['handlers']['audit_file']['filename']
    
    for log_path in [log_file, audit_log_file]:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Load custom configuration if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    custom_config = json.load(f)
                else:
                    import yaml
                    custom_config = yaml.safe_load(f)
            
            # Merge with default config
            default_config.update(custom_config)
        except Exception as e:
            print(f"Failed to load logging config from {config_path}: {e}")
    
    # Apply configuration
    logging.config.dictConfig(default_config)


def get_correlation_id() -> str:
    """Get or generate correlation ID for request tracing."""
    correlation_id = correlation_id_var.get("")
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for request tracing."""
    correlation_id_var.set(correlation_id)


def get_request_id() -> str:
    """Get or generate request ID."""
    request_id = request_id_var.get("")
    if not request_id:
        request_id = str(uuid.uuid4())
        request_id_var.set(request_id)
    return request_id


def set_request_id(request_id: str) -> None:
    """Set request ID."""
    request_id_var.set(request_id)


def set_security_context(context: SecurityContext) -> None:
    """Set security context for logging."""
    security_context_var.set(context)


def get_security_context() -> SecurityContext:
    """Get current security context."""
    return security_context_var.get(SecurityContext())


# Global logger instances
security_logger = SecurityAuditLogger()
performance_logger = PerformanceLogger()

# Convenience functions
def log_vulnerability_detection(algorithm: str, confidence: float,
                               firmware_path: str, details: Dict[str, Any]):
    """Log vulnerability detection event."""
    security_logger.log_vulnerability_detection(algorithm, confidence, firmware_path, details)


def log_patch_generation(algorithm: str, target_device: str,
                        patch_id: str, success: bool):
    """Log patch generation event."""
    security_logger.log_patch_generation(algorithm, target_device, patch_id, success)


def log_crypto_operation(operation: str, algorithm: str,
                        duration_ms: float, success: bool):
    """Log cryptographic operation."""
    security_logger.log_crypto_operation(operation, algorithm, duration_ms, success)


def log_performance(operation: str, context: PerformanceContext,
                   threshold_ms: Optional[float] = None):
    """Log operation performance."""
    performance_logger.log_operation_performance(operation, context, threshold_ms)


def log_resource_usage(resource_type: str, usage_percent: float,
                      threshold_percent: float = 80.0):
    """Log resource usage."""
    performance_logger.log_resource_usage(resource_type, usage_percent, threshold_percent)


# Initialize logging on module import
setup_logging()