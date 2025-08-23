"""Generation 2: Enhanced Logging and Monitoring System.

Comprehensive logging framework providing:
- Structured logging with correlation IDs
- Security-aware log sanitization
- Performance tracking and metrics
- Audit trail capabilities
- Real-time monitoring hooks
"""

import logging
import json
import time
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
import traceback

# Thread-local storage for correlation tracking
_context = threading.local()


class LogLevel(Enum):
    """Enhanced log levels with security context."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"  # Security-specific events
    AUDIT = "AUDIT"        # Audit trail events


class EventType(Enum):
    """Types of events for structured logging."""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SCAN_START = "scan_start"
    SCAN_COMPLETE = "scan_complete"
    SCAN_ERROR = "scan_error"
    VULNERABILITY_FOUND = "vulnerability_found"
    VALIDATION_ERROR = "validation_error"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_METRIC = "performance_metric"
    USER_ACTION = "user_action"
    FILE_ACCESS = "file_access"
    CONFIGURATION_CHANGE = "configuration_change"


@dataclass
class LogContext:
    """Context information for structured logging."""
    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


@dataclass
class SecurityEvent:
    """Security-specific event information."""
    event_type: str
    severity: str
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    affected_resource: Optional[str] = None
    attack_indicators: List[str] = field(default_factory=list)
    mitigation_applied: bool = False


@dataclass
class PerformanceMetric:
    """Performance measurement data."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    cpu_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


class SecureLogSanitizer:
    """Sanitize log data to prevent information disclosure."""
    
    # Patterns to redact from logs
    SENSITIVE_PATTERNS = {
        'password': r'(?i)(password|pwd|pass)["\s]*[:=]["\s]*([^\s"]+)',
        'api_key': r'(?i)(api_key|apikey|key)["\s]*[:=]["\s]*([^\s"]+)',
        'token': r'(?i)(token|auth)["\s]*[:=]["\s]*([^\s"]+)',
        'secret': r'(?i)(secret)["\s]*[:=]["\s]*([^\s"]+)',
        'private_key': r'-----BEGIN[^-]*PRIVATE KEY-----.*?-----END[^-]*PRIVATE KEY-----',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    }
    
    @classmethod
    def sanitize_message(cls, message: str, redaction_level: str = "partial") -> str:
        """Sanitize log message to remove sensitive information."""
        
        import re
        
        sanitized = message
        
        for pattern_name, pattern in cls.SENSITIVE_PATTERNS.items():
            if redaction_level == "full":
                # Complete redaction
                sanitized = re.sub(pattern, f'[{pattern_name.upper()}_REDACTED]', sanitized)
            else:
                # Partial redaction (show first few characters)
                def redact_match(match):
                    if len(match.groups()) >= 2:
                        value = match.group(2)
                        if len(value) > 4:
                            return match.group(0).replace(value, value[:2] + '*' * (len(value) - 2))
                    return f'[{pattern_name.upper()}_REDACTED]'
                
                sanitized = re.sub(pattern, redact_match, sanitized)
        
        return sanitized
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any], redaction_level: str = "partial") -> Dict[str, Any]:
        """Sanitize dictionary data recursively."""
        
        sanitized = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key indicates sensitive data
            if any(sensitive in key_lower for sensitive in ['password', 'secret', 'key', 'token']):
                if redaction_level == "full":
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = str(value)[:2] + "*" * max(0, len(str(value)) - 2)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value, redaction_level)
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_message(value, redaction_level)
            else:
                sanitized[key] = value
        
        return sanitized


class StructuredLogger:
    """Enhanced structured logging with security and performance tracking."""
    
    def __init__(self, name: str, log_file: Optional[str] = None, 
                 log_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.sanitizer = SecureLogSanitizer()
        self._setup_logging(log_file, log_level)
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetric] = []
        self.metrics_lock = threading.Lock()
        
        # Event handlers
        self.event_handlers: Dict[EventType, List[Callable]] = {}
    
    def _setup_logging(self, log_file: Optional[str], log_level: LogLevel):
        """Setup logging configuration."""
        
        # Create formatter for structured logs
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Set log level
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.SECURITY: logging.ERROR,
            LogLevel.AUDIT: logging.INFO
        }
        
        self.logger.setLevel(level_mapping.get(log_level, logging.INFO))
    
    def get_context(self) -> LogContext:
        """Get current logging context."""
        if not hasattr(_context, 'log_context'):
            _context.log_context = LogContext(
                correlation_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4())
            )
        return _context.log_context
    
    def set_context(self, **kwargs):
        """Set logging context properties."""
        context = self.get_context()
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.metadata[key] = value
    
    def _format_structured_message(self, event_type: EventType, message: str,
                                  context: Optional[Dict] = None,
                                  sanitize: bool = True) -> str:
        """Format structured log message."""
        
        log_context = self.get_context()
        
        structured_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type.value,
            'message': message,
            'context': log_context.to_dict(),
            'additional_context': context or {}
        }
        
        if sanitize:
            structured_data = self.sanitizer.sanitize_dict(structured_data)
        
        try:
            return json.dumps(structured_data, default=str)
        except Exception:
            # Fallback to simple message if JSON serialization fails
            return f"{event_type.value}: {message}"
    
    def log_event(self, level: LogLevel, event_type: EventType, message: str,
                  context: Optional[Dict] = None, sanitize: bool = True):
        """Log structured event."""
        
        formatted_message = self._format_structured_message(
            event_type, message, context, sanitize
        )
        
        # Map custom levels to standard logging levels
        if level == LogLevel.SECURITY:
            self.logger.error(f"SECURITY: {formatted_message}")
        elif level == LogLevel.AUDIT:
            self.logger.info(f"AUDIT: {formatted_message}")
        else:
            getattr(self.logger, level.value.lower())(formatted_message)
        
        # Trigger event handlers
        self._trigger_event_handlers(event_type, {
            'level': level,
            'message': message,
            'context': context
        })
    
    def log_security_event(self, event: SecurityEvent, message: str):
        """Log security-specific event."""
        
        context = {
            'security_event': asdict(event),
            'severity': event.severity,
            'source_ip': event.source_ip,
            'affected_resource': event.affected_resource
        }
        
        self.log_event(LogLevel.SECURITY, EventType.SECURITY_ALERT, message, context)
    
    def log_performance_metric(self, metric: PerformanceMetric):
        """Log performance metric."""
        
        with self.metrics_lock:
            self.performance_metrics.append(metric)
        
        context = {
            'operation': metric.operation_name,
            'duration_ms': round(metric.duration * 1000, 2),
            'success': metric.success,
            'memory_delta': metric.memory_after - metric.memory_before if metric.memory_after and metric.memory_before else None
        }
        
        self.log_event(LogLevel.INFO, EventType.PERFORMANCE_METRIC, 
                      f"Operation {metric.operation_name} completed in {metric.duration:.3f}s",
                      context)
    
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """Add event handler for specific event types."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event_handlers(self, event_type: EventType, event_data: Dict):
        """Trigger registered event handlers."""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Event handler failed for {event_type}: {e}")
    
    def get_performance_summary(self, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        
        with self.metrics_lock:
            metrics = self.performance_metrics.copy()
        
        if operation_filter:
            metrics = [m for m in metrics if operation_filter in m.operation_name]
        
        if not metrics:
            return {'total_operations': 0}
        
        total_operations = len(metrics)
        successful_operations = sum(1 for m in metrics if m.success)
        total_duration = sum(m.duration for m in metrics)
        avg_duration = total_duration / total_operations
        
        durations = [m.duration for m in metrics]
        durations.sort()
        
        summary = {
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations,
            'total_duration_seconds': total_duration,
            'average_duration_seconds': avg_duration,
            'median_duration_seconds': durations[len(durations) // 2] if durations else 0,
            'min_duration_seconds': min(durations) if durations else 0,
            'max_duration_seconds': max(durations) if durations else 0
        }
        
        return summary


def performance_monitor(operation_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from first argument if it's a class instance
            logger = None
            if args and hasattr(args[0], 'logger') and isinstance(args[0].logger, StructuredLogger):
                logger = args[0].logger
            else:
                # Fallback to global logger
                logger = StructuredLogger('performance_monitor')
            
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Memory tracking (if psutil available)
            memory_before = None
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss
            except ImportError:
                pass
            
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                logger.set_context(operation=op_name)
                logger.log_event(LogLevel.DEBUG, EventType.SYSTEM_START, 
                               f"Starting operation: {op_name}")
                
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                success = False
                error_message = str(e)
                logger.log_event(LogLevel.ERROR, EventType.SCAN_ERROR,
                               f"Operation failed: {op_name}", {'error': str(e)})
                raise
                
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                # Memory after
                memory_after = None
                try:
                    import psutil
                    process = psutil.Process()
                    memory_after = process.memory_info().rss
                except ImportError:
                    pass
                
                # Create performance metric
                metric = PerformanceMetric(
                    operation_name=op_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    success=success,
                    error_message=error_message
                )
                
                logger.log_performance_metric(metric)
        
        return wrapper
    return decorator


def audit_trail(action: str, resource: Optional[str] = None):
    """Decorator to create audit trail for sensitive operations."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = StructuredLogger('audit_trail')
            
            context = {
                'action': action,
                'resource': resource,
                'function': f"{func.__module__}.{func.__name__}",
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            logger.log_event(LogLevel.AUDIT, EventType.USER_ACTION,
                           f"Audit: {action} on {resource or 'system'}", context)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class LogAggregator:
    """Aggregate and analyze log data."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
    
    def add_event(self, event_data: Dict[str, Any]):
        """Add event to aggregation."""
        with self.lock:
            self.events.append({
                **event_data,
                'timestamp': time.time()
            })
    
    def get_events_by_type(self, event_type: EventType) -> List[Dict[str, Any]]:
        """Get events filtered by type."""
        with self.lock:
            return [e for e in self.events if e.get('event_type') == event_type.value]
    
    def get_security_events(self) -> List[Dict[str, Any]]:
        """Get all security-related events."""
        return self.get_events_by_type(EventType.SECURITY_ALERT)
    
    def get_error_summary(self, time_window_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Get error summary statistics."""
        
        current_time = time.time()
        cutoff_time = current_time - (time_window_seconds or 3600)  # Default 1 hour
        
        with self.lock:
            recent_events = [e for e in self.events if e['timestamp'] >= cutoff_time]
        
        error_events = [e for e in recent_events 
                       if e.get('level') in ['ERROR', 'CRITICAL', 'SECURITY']]
        
        error_types = {}
        for event in error_events:
            event_type = event.get('event_type', 'unknown')
            error_types[event_type] = error_types.get(event_type, 0) + 1
        
        return {
            'total_events': len(recent_events),
            'error_events': len(error_events),
            'error_rate': len(error_events) / max(1, len(recent_events)),
            'error_types': error_types,
            'time_window_seconds': time_window_seconds or 3600
        }


# Global instances
_global_logger = None
_global_aggregator = LogAggregator()

def get_logger(name: str = "pqc_scanner") -> StructuredLogger:
    """Get global structured logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(name)
        
        # Connect to aggregator
        def aggregate_handler(event_data):
            _global_aggregator.add_event(event_data)
        
        for event_type in EventType:
            _global_logger.add_event_handler(event_type, aggregate_handler)
    
    return _global_logger

def get_log_aggregator() -> LogAggregator:
    """Get global log aggregator."""
    return _global_aggregator


# Export main components
__all__ = [
    'LogLevel', 'EventType', 'LogContext', 'SecurityEvent', 'PerformanceMetric',
    'StructuredLogger', 'SecureLogSanitizer', 'LogAggregator',
    'performance_monitor', 'audit_trail', 'get_logger', 'get_log_aggregator'
]