"""
Prometheus metrics collection for PQC IoT Retrofit Scanner.

Provides comprehensive metrics for monitoring application performance,
security analysis results, and system health.
"""

import os
import time
from typing import Dict, List, Optional, Any, Counter as CounterType
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import prometheus_client, gracefully handle if not available
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, Enum as PrometheusEnum,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def labels(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass
    
    PrometheusEnum = Gauge  # Fallback
    CollectorRegistry = None
    generate_latest = lambda *args: b""
    CONTENT_TYPE_LATEST = "text/plain"


class MetricsCollector:
    """Centralized metrics collection for Prometheus monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.
        
        Args:
            registry: Custom Prometheus registry (uses default if None)
        """
        self.enabled = PROMETHEUS_AVAILABLE and os.getenv("PQC_METRICS_ENABLED", "false").lower() == "true"
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
        if not self.enabled:
            self.logger.warning("Prometheus metrics disabled (prometheus_client not available or PQC_METRICS_ENABLED=false)")
            return
        
        # Application info
        self.app_info = Info(
            'pqc_scanner_info',
            'Information about the PQC IoT Retrofit Scanner',
            registry=registry
        )
        self.app_info.info({
            'version': os.getenv('VERSION', '0.1.0'),
            'build_date': os.getenv('BUILD_DATE', 'unknown'),
            'vcs_ref': os.getenv('VCS_REF', 'unknown')
        })
        
        # Firmware analysis metrics
        self.firmware_analyses_total = Counter(
            'pqc_firmware_analyses_total',
            'Total number of firmware analyses performed',
            ['architecture', 'status'],
            registry=registry
        )
        
        self.firmware_analysis_duration = Histogram(
            'pqc_firmware_analysis_duration_seconds',
            'Time spent analyzing firmware',
            ['architecture'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=registry
        )
        
        self.firmware_size_bytes = Histogram(
            'pqc_firmware_size_bytes',
            'Size of analyzed firmware files',
            ['architecture'],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600],
            registry=registry
        )
        
        # Vulnerability detection metrics
        self.vulnerabilities_detected_total = Counter(
            'pqc_vulnerabilities_detected_total',
            'Total number of vulnerabilities detected',
            ['algorithm', 'severity', 'architecture'],
            registry=registry
        )
        
        self.vulnerability_confidence = Histogram(
            'pqc_vulnerability_confidence',
            'Confidence scores of detected vulnerabilities',
            ['algorithm', 'architecture'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry
        )
        
        # Patch generation metrics
        self.patches_generated_total = Counter(
            'pqc_patches_generated_total',
            'Total number of PQC patches generated',
            ['algorithm', 'target_device', 'status'],
            registry=registry
        )
        
        self.patch_generation_duration = Histogram(
            'pqc_patch_generation_duration_seconds',
            'Time spent generating patches',
            ['algorithm', 'target_device'],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0],
            registry=registry
        )
        
        # Cryptographic operation metrics
        self.crypto_operations_total = Counter(
            'pqc_crypto_operations_total',
            'Total cryptographic operations performed',
            ['operation', 'algorithm', 'status'],
            registry=registry
        )
        
        self.crypto_operation_duration = Histogram(
            'pqc_crypto_operation_duration_seconds',
            'Duration of cryptographic operations',
            ['operation', 'algorithm'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=registry
        )
        
        # System resource metrics
        self.system_cpu_usage = Gauge(
            'pqc_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=registry
        )
        
        self.system_memory_usage = Gauge(
            'pqc_system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],  # available, used, total
            registry=registry
        )
        
        self.process_memory_usage = Gauge(
            'pqc_process_memory_usage_bytes',
            'Process memory usage in bytes',
            ['type'],  # rss, vms
            registry=registry
        )
        
        self.open_file_descriptors = Gauge(
            'pqc_open_file_descriptors',
            'Number of open file descriptors',
            registry=registry
        )
        
        # Error and health metrics
        self.errors_total = Counter(
            'pqc_errors_total',
            'Total number of errors',
            ['component', 'error_type'],
            registry=registry
        )
        
        self.health_check_status = PrometheusEnum(
            'pqc_health_check_status',
            'Health check status',
            ['check_name'],
            states=['healthy', 'warning', 'critical', 'unknown'],
            registry=registry
        )
        
        self.health_check_duration = Histogram(
            'pqc_health_check_duration_seconds',
            'Duration of health checks',
            ['check_name'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=registry
        )
        
        # Cache and performance metrics
        self.cache_operations_total = Counter(
            'pqc_cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],  # hit, miss, set, delete
            registry=registry
        )
        
        self.analysis_queue_size = Gauge(
            'pqc_analysis_queue_size',
            'Number of analyses in queue',
            registry=registry
        )
        
        self.concurrent_analyses = Gauge(
            'pqc_concurrent_analyses',
            'Number of concurrent analyses running',
            registry=registry
        )
        
        self.logger.info("Prometheus metrics collector initialized")
    
    def record_firmware_analysis(self, architecture: str, duration_seconds: float, 
                                firmware_size: int, status: str = "success"):
        """Record firmware analysis metrics."""
        if not self.enabled:
            return
        
        self.firmware_analyses_total.labels(architecture=architecture, status=status).inc()
        self.firmware_analysis_duration.labels(architecture=architecture).observe(duration_seconds)
        self.firmware_size_bytes.labels(architecture=architecture).observe(firmware_size)
    
    def record_vulnerability(self, algorithm: str, severity: str, architecture: str, confidence: float):
        """Record detected vulnerability metrics."""
        if not self.enabled:
            return
        
        self.vulnerabilities_detected_total.labels(
            algorithm=algorithm, 
            severity=severity, 
            architecture=architecture
        ).inc()
        self.vulnerability_confidence.labels(
            algorithm=algorithm, 
            architecture=architecture
        ).observe(confidence)
    
    def record_patch_generation(self, algorithm: str, target_device: str, 
                               duration_seconds: float, status: str = "success"):
        """Record patch generation metrics."""
        if not self.enabled:
            return
        
        self.patches_generated_total.labels(
            algorithm=algorithm, 
            target_device=target_device, 
            status=status
        ).inc()
        self.patch_generation_duration.labels(
            algorithm=algorithm, 
            target_device=target_device
        ).observe(duration_seconds)
    
    def record_crypto_operation(self, operation: str, algorithm: str, 
                               duration_seconds: float, status: str = "success"):
        """Record cryptographic operation metrics."""
        if not self.enabled:
            return
        
        self.crypto_operations_total.labels(
            operation=operation, 
            algorithm=algorithm, 
            status=status
        ).inc()
        self.crypto_operation_duration.labels(
            operation=operation, 
            algorithm=algorithm
        ).observe(duration_seconds)
    
    def update_system_metrics(self, cpu_percent: float, memory_info: Dict[str, int], 
                             process_memory: Dict[str, int], open_fds: int):
        """Update system resource metrics."""
        if not self.enabled:
            return
        
        self.system_cpu_usage.set(cpu_percent)
        
        for memory_type, value in memory_info.items():
            self.system_memory_usage.labels(type=memory_type).set(value)
        
        for memory_type, value in process_memory.items():
            self.process_memory_usage.labels(type=memory_type).set(value)
        
        self.open_file_descriptors.set(open_fds)
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence."""
        if not self.enabled:
            return
        
        self.errors_total.labels(component=component, error_type=error_type).inc()
    
    def update_health_check(self, check_name: str, status: str, duration_seconds: float):
        """Update health check metrics."""
        if not self.enabled:
            return
        
        if hasattr(self.health_check_status.labels(check_name=check_name), 'state'):
            self.health_check_status.labels(check_name=check_name).state(status)
        self.health_check_duration.labels(check_name=check_name).observe(duration_seconds)
    
    def record_cache_operation(self, operation: str, result: str):
        """Record cache operation metrics."""
        if not self.enabled:
            return
        
        self.cache_operations_total.labels(operation=operation, result=result).inc()
    
    def update_queue_metrics(self, queue_size: int, concurrent_analyses: int):
        """Update analysis queue metrics."""
        if not self.enabled:
            return
        
        self.analysis_queue_size.set(queue_size)
        self.concurrent_analyses.set(concurrent_analyses)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        if not self.enabled:
            return b""
        
        return generate_latest(self.registry)


class MetricsServer:
    """HTTP server for exposing Prometheus metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 9090):
        """Initialize metrics server.
        
        Args:
            metrics_collector: Metrics collector instance
            port: Port to serve metrics on
        """
        self.metrics_collector = metrics_collector
        self.port = port
        self.enabled = metrics_collector.enabled and os.getenv("PQC_METRICS_SERVER", "false").lower() == "true"
        self.logger = logging.getLogger(__name__)
        self.server_started = False
    
    def start(self):
        """Start the metrics server."""
        if not self.enabled:
            self.logger.info("Metrics server disabled")
            return
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Cannot start metrics server: prometheus_client not available")
            return
        
        try:
            start_http_server(self.port, registry=self.metrics_collector.registry)
            self.server_started = True
            self.logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    def is_running(self) -> bool:
        """Check if metrics server is running."""
        return self.server_started


class MetricsPusher:
    """Pushes metrics to Prometheus Pushgateway."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 gateway_url: str, job_name: str = "pqc_scanner"):
        """Initialize metrics pusher.
        
        Args:
            metrics_collector: Metrics collector instance
            gateway_url: Pushgateway URL
            job_name: Job name for metrics
        """
        self.metrics_collector = metrics_collector
        self.gateway_url = gateway_url
        self.job_name = job_name
        self.enabled = (metrics_collector.enabled and 
                       os.getenv("PQC_METRICS_PUSH", "false").lower() == "true")
        self.logger = logging.getLogger(__name__)
    
    def push_metrics(self, additional_labels: Optional[Dict[str, str]] = None):
        """Push metrics to Pushgateway.
        
        Args:
            additional_labels: Additional labels to include
        """
        if not self.enabled:
            return
        
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Cannot push metrics: prometheus_client not available")
            return
        
        try:
            labels = additional_labels or {}
            push_to_gateway(
                self.gateway_url,
                job=self.job_name,
                registry=self.metrics_collector.registry,
                grouping_key=labels
            )
            self.logger.debug(f"Metrics pushed to {self.gateway_url}")
        except Exception as e:
            self.logger.error(f"Failed to push metrics: {e}")


# Global metrics collector instance
metrics_collector = MetricsCollector()

# Global metrics server instance
metrics_server = MetricsServer(metrics_collector, port=int(os.getenv("PQC_METRICS_PORT", "9090")))

# Global metrics pusher (if configured)
pushgateway_url = os.getenv("PQC_PUSHGATEWAY_URL")
metrics_pusher = MetricsPusher(metrics_collector, pushgateway_url) if pushgateway_url else None


# Convenience functions
def record_firmware_analysis(architecture: str, duration_seconds: float, 
                           firmware_size: int, status: str = "success"):
    """Convenience function to record firmware analysis metrics."""
    metrics_collector.record_firmware_analysis(architecture, duration_seconds, firmware_size, status)


def record_vulnerability(algorithm: str, severity: str, architecture: str, confidence: float):
    """Convenience function to record vulnerability metrics."""
    metrics_collector.record_vulnerability(algorithm, severity, architecture, confidence)


def record_patch_generation(algorithm: str, target_device: str, 
                          duration_seconds: float, status: str = "success"):
    """Convenience function to record patch generation metrics."""
    metrics_collector.record_patch_generation(algorithm, target_device, duration_seconds, status)


def record_crypto_operation(operation: str, algorithm: str, 
                          duration_seconds: float, status: str = "success"):
    """Convenience function to record crypto operation metrics."""
    metrics_collector.record_crypto_operation(operation, algorithm, duration_seconds, status)


def record_error(component: str, error_type: str):
    """Convenience function to record errors."""
    metrics_collector.record_error(component, error_type)


def start_metrics_server():
    """Start the metrics server if enabled."""
    metrics_server.start()


def push_metrics(additional_labels: Optional[Dict[str, str]] = None):
    """Push metrics to Pushgateway if configured."""
    if metrics_pusher:
        metrics_pusher.push_metrics(additional_labels)


# Decorator for automatic metrics collection
def measure_with_metrics(operation: str, algorithm: str = "unknown"):
    """Decorator to automatically collect metrics for functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                record_error(func.__module__, type(e).__name__)
                raise
            finally:
                duration = time.time() - start_time
                record_crypto_operation(operation, algorithm, duration, status)
        
        return wrapper
    return decorator