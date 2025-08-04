"""
Enhanced monitoring and observability for PQC IoT Retrofit Scanner.

This module extends the existing monitoring capabilities with:
- Advanced performance tracking
- Real-time metrics
- Health checks and alerts
- Integration with external monitoring systems
"""

import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import psutil
import os

from .error_handling import ErrorSeverity, ErrorCategory, global_error_handler


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: float
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, max_history_size: int = 10000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.aggregates: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        
        # Performance counters
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # System metrics
        self.process = psutil.Process()
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, unit: str = "", tags: Dict[str, str] = None):
        """Record a metric value."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
            self._update_aggregates(name, value)
    
    def _update_aggregates(self, name: str, value: float):
        """Update aggregate statistics for a metric."""
        if name not in self.aggregates:
            self.aggregates[name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'last': 0.0
            }
        
        agg = self.aggregates[name]
        agg['count'] += 1
        agg['sum'] += value
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
        agg['last'] = value
        agg['mean'] = agg['sum'] / agg['count']
    
    def get_metric_summary(self, name: str) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.aggregates:
                return None
            
            agg = self.aggregates[name].copy()
            
            # Calculate additional statistics
            recent_values = [m.value for m in list(self.metrics[name])[-100:]]  # Last 100 values
            if recent_values:
                agg['recent_mean'] = sum(recent_values) / len(recent_values)
                
                # Calculate percentiles
                sorted_values = sorted(recent_values)
                n = len(sorted_values)
                if n > 0:
                    agg['p50'] = sorted_values[n // 2]
                    agg['p95'] = sorted_values[int(n * 0.95)] if n > 20 else sorted_values[-1]
                    agg['p99'] = sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1]
            
            return agg
    
    def record_operation_start(self, operation: str) -> str:
        """Record the start of an operation."""
        operation_id = f"{operation}_{time.time()}_{threading.current_thread().ident}"
        self.operation_counts[operation] += 1
        return operation_id
    
    def record_operation_end(self, operation: str, operation_id: str, success: bool = True):
        """Record the end of an operation."""
        # Calculate duration (simplified - in real implementation would track by ID)
        duration = time.time() - float(operation_id.split('_')[1])
        
        with self.lock:
            self.operation_times[operation].append(duration)
            
            # Keep only recent times (last 1000 operations)
            if len(self.operation_times[operation]) > 1000:
                self.operation_times[operation] = self.operation_times[operation][-1000:]
        
        # Record metrics
        self.record_metric(f"{operation}.duration", duration, "seconds", {"success": str(success)})
        self.record_metric(f"{operation}.count", 1, "operations")
        
        if not success:
            self.error_counts[operation] += 1
            self.record_metric(f"{operation}.errors", 1, "errors")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            # CPU and memory usage
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent(interval=0.1)
            system_memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            return {
                'process': {
                    'cpu_percent': cpu_percent,
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms,
                    'memory_percent': self.process.memory_percent(),
                    'num_threads': self.process.num_threads(),
                    'uptime_seconds': time.time() - self.start_time
                },
                'system': {
                    'cpu_percent': system_cpu,
                    'memory_total': system_memory.total,
                    'memory_available': system_memory.available,
                    'memory_percent': system_memory.percent,
                    'disk_total': disk_usage.total,
                    'disk_free': disk_usage.free,
                    'disk_percent': disk_usage.percent
                }
            }
        except Exception as e:
            logging.warning(f"Failed to collect system metrics: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            return {
                'aggregates': dict(self.aggregates),
                'operation_counts': dict(self.operation_counts),
                'error_counts': dict(self.error_counts),
                'system': self.get_system_metrics(),
                'timestamp': time.time()
            }


class HealthMonitor:
    """System health monitoring with checks and alerts."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.health_checks: Dict[str, Callable] = {}
        self.alert_thresholds: Dict[str, Dict] = {}
        self.health_history: deque = deque(maxlen=1000)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_health_check("memory_usage", self._check_memory_usage)
        self.register_health_check("cpu_usage", self._check_cpu_usage)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("error_rate", self._check_error_rate)
        self.register_health_check("operation_latency", self._check_operation_latency)
        
        # Set default thresholds
        self.alert_thresholds = {
            "memory_usage": {"warning": 80, "critical": 90},
            "cpu_usage": {"warning": 80, "critical": 95},
            "disk_space": {"warning": 85, "critical": 95},
            "error_rate": {"warning": 5, "critical": 10},  # Errors per minute
            "operation_latency": {"warning": 30, "critical": 60}  # Seconds
        }
    
    def register_health_check(self, name: str, check_func: Callable):
        """Register a custom health check."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                
                # Record metric
                status_value = 1 if result.status == "healthy" else 0
                self.metrics.record_metric(f"health.{name}", status_value, "status")
                
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    timestamp=time.time()
                )
        
        # Store in history
        overall_status = self._calculate_overall_status(results)
        self.health_history.append({
            'timestamp': time.time(),
            'overall_status': overall_status,
            'checks': {name: result.status for name, result in results.items()}
        })
        
        return results
    
    def _calculate_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Calculate overall system health status."""
        if not results:
            return "unknown"
        
        statuses = [result.status for result in results.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check system memory usage."""
        system_metrics = self.metrics.get_system_metrics()
        
        if not system_metrics or 'system' not in system_metrics:
            return HealthCheckResult(
                name="memory_usage",
                status="unhealthy",
                message="Unable to collect memory metrics",
                timestamp=time.time()
            )
        
        memory_percent = system_metrics['system']['memory_percent']
        thresholds = self.alert_thresholds.get("memory_usage", {})
        
        if memory_percent >= thresholds.get("critical", 90):
            status = "unhealthy"
            message = f"Critical memory usage: {memory_percent:.1f}%"
        elif memory_percent >= thresholds.get("warning", 80):
            status = "degraded"
            message = f"High memory usage: {memory_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Memory usage normal: {memory_percent:.1f}%"
        
        return HealthCheckResult(
            name="memory_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"memory_percent": memory_percent}
        )
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        system_metrics = self.metrics.get_system_metrics()
        
        if not system_metrics or 'system' not in system_metrics:
            return HealthCheckResult(
                name="cpu_usage",
                status="unhealthy",
                message="Unable to collect CPU metrics",
                timestamp=time.time()
            )
        
        cpu_percent = system_metrics['system']['cpu_percent']
        thresholds = self.alert_thresholds.get("cpu_usage", {})
        
        if cpu_percent >= thresholds.get("critical", 95):
            status = "unhealthy"
            message = f"Critical CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent >= thresholds.get("warning", 80):
            status = "degraded"
            message = f"High CPU usage: {cpu_percent:.1f}%"
        else:
            status = "healthy"
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        
        return HealthCheckResult(
            name="cpu_usage",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"cpu_percent": cpu_percent}
        )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space usage."""
        system_metrics = self.metrics.get_system_metrics()
        
        if not system_metrics or 'system' not in system_metrics:
            return HealthCheckResult(
                name="disk_space",
                status="unhealthy",
                message="Unable to collect disk metrics",
                timestamp=time.time()
            )
        
        disk_percent = system_metrics['system']['disk_percent']
        thresholds = self.alert_thresholds.get("disk_space", {})
        
        if disk_percent >= thresholds.get("critical", 95):
            status = "unhealthy"
            message = f"Critical disk usage: {disk_percent:.1f}%"
        elif disk_percent >= thresholds.get("warning", 85):
            status = "degraded"
            message = f"High disk usage: {disk_percent:.1f}%"
        else:
            status = "healthy"
            message = f"Disk usage normal: {disk_percent:.1f}%"
        
        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"disk_percent": disk_percent}
        )
    
    def _check_error_rate(self) -> HealthCheckResult:
        """Check error rate."""
        error_stats = global_error_handler.get_error_statistics()
        error_rate = error_stats.get("error_rate_per_hour", 0)
        
        # Convert to errors per minute
        error_rate_per_minute = error_rate / 60
        
        thresholds = self.alert_thresholds.get("error_rate", {})
        
        if error_rate_per_minute >= thresholds.get("critical", 10):
            status = "unhealthy"
            message = f"Critical error rate: {error_rate_per_minute:.2f} errors/min"
        elif error_rate_per_minute >= thresholds.get("warning", 5):
            status = "degraded"
            message = f"High error rate: {error_rate_per_minute:.2f} errors/min"
        else:
            status = "healthy"
            message = f"Error rate normal: {error_rate_per_minute:.2f} errors/min"
        
        return HealthCheckResult(
            name="error_rate",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"error_rate_per_minute": error_rate_per_minute}
        )
    
    def _check_operation_latency(self) -> HealthCheckResult:
        """Check operation latency."""
        # Get average latency across all operations
        all_metrics = self.metrics.get_all_metrics()
        aggregates = all_metrics.get('aggregates', {})
        
        # Find duration metrics
        duration_metrics = {k: v for k, v in aggregates.items() if k.endswith('.duration')}
        
        if not duration_metrics:
            return HealthCheckResult(
                name="operation_latency",
                status="healthy",
                message="No operation latency data available",
                timestamp=time.time()
            )
        
        # Calculate average latency across all operations
        total_mean = sum(metric['mean'] for metric in duration_metrics.values())
        avg_latency = total_mean / len(duration_metrics)
        
        thresholds = self.alert_thresholds.get("operation_latency", {})
        
        if avg_latency >= thresholds.get("critical", 60):
            status = "unhealthy"
            message = f"Critical operation latency: {avg_latency:.2f}s"
        elif avg_latency >= thresholds.get("warning", 30):
            status = "degraded"
            message = f"High operation latency: {avg_latency:.2f}s"
        else:
            status = "healthy"
            message = f"Operation latency normal: {avg_latency:.2f}s"
        
        return HealthCheckResult(
            name="operation_latency",
            status=status,
            message=message,
            timestamp=time.time(),
            details={"avg_latency_seconds": avg_latency}
        )


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self.alert_handlers: List[Callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.suppression_rules: Dict[str, Dict] = {}
        
        # Register default alert handlers
        self.register_alert_handler(self._log_alert)
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def check_and_send_alerts(self):
        """Check health status and send alerts if needed."""
        health_results = self.health_monitor.run_health_checks()
        
        for name, result in health_results.items():
            if result.status in ["degraded", "unhealthy"]:
                # Check if alert should be suppressed
                if not self._should_suppress_alert(name, result.status):
                    self._send_alert(name, result)
    
    def _should_suppress_alert(self, check_name: str, status: str) -> bool:
        """Check if alert should be suppressed based on rules."""
        if check_name not in self.suppression_rules:
            return False
        
        rules = self.suppression_rules[check_name]
        
        # Time-based suppression
        if 'last_alert_time' in rules:
            min_interval = rules.get('min_interval', 300)  # 5 minutes default
            if time.time() - rules['last_alert_time'] < min_interval:
                return True
        
        return False
    
    def _send_alert(self, check_name: str, result: HealthCheckResult):
        """Send alert through all registered handlers."""
        alert_data = {
            'check_name': check_name,
            'status': result.status,
            'message': result.message,
            'timestamp': result.timestamp,
            'details': result.details
        }
        
        # Record alert
        self.alert_history.append(alert_data)
        
        # Update suppression rules
        if check_name not in self.suppression_rules:
            self.suppression_rules[check_name] = {}
        self.suppression_rules[check_name]['last_alert_time'] = time.time()
        
        # Send through all handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def _log_alert(self, alert_data: Dict[str, Any]):
        """Default alert handler that logs alerts."""
        severity = "ERROR" if alert_data['status'] == "unhealthy" else "WARNING"
        logging.log(
            logging.ERROR if severity == "ERROR" else logging.WARNING,
            f"[ALERT] {alert_data['check_name']}: {alert_data['message']}"
        )


class MonitoringDashboard:
    """Simple monitoring dashboard and reporting."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_monitor: HealthMonitor):
        self.metrics = metrics_collector
        self.health_monitor = health_monitor
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report."""
        health_results = self.health_monitor.run_health_checks()
        all_metrics = self.metrics.get_all_metrics()
        error_stats = global_error_handler.get_error_statistics()
        
        # Calculate uptime
        uptime_seconds = time.time() - self.metrics.start_time
        
        return {
            'timestamp': time.time(),
            'uptime_seconds': uptime_seconds,
            'uptime_human': self._format_duration(uptime_seconds),
            'overall_status': self.health_monitor._calculate_overall_status(health_results),
            'health_checks': {name: asdict(result) for name, result in health_results.items()},
            'system_metrics': all_metrics.get('system', {}),
            'operation_metrics': {
                'total_operations': sum(all_metrics.get('operation_counts', {}).values()),
                'operation_counts': all_metrics.get('operation_counts', {}),
                'error_counts': all_metrics.get('error_counts', {})
            },
            'error_statistics': error_stats,
            'performance_summary': self._get_performance_summary(all_metrics)
        }
    
    def _get_performance_summary(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance summary from metrics."""
        aggregates = all_metrics.get('aggregates', {})
        
        # Find key performance metrics
        key_metrics = {}
        
        for metric_name, data in aggregates.items():
            if '.duration' in metric_name:
                operation_name = metric_name.replace('.duration', '')
                key_metrics[f"{operation_name}_avg_duration"] = data.get('mean', 0)
                key_metrics[f"{operation_name}_max_duration"] = data.get('max', 0)
        
        return key_metrics
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            return f"{seconds/86400:.1f} days"
    
    def export_metrics_to_file(self, filepath: str):
        """Export metrics to file for external monitoring systems."""
        report = self.generate_status_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        all_metrics = self.metrics.get_all_metrics()
        health_results = self.health_monitor.run_health_checks()
        
        lines = []
        
        # System metrics
        system = all_metrics.get('system', {})
        if 'process' in system:
            process = system['process']
            lines.append(f'pqc_process_cpu_percent {process.get("cpu_percent", 0)}')
            lines.append(f'pqc_process_memory_rss {process.get("memory_rss", 0)}')
            lines.append(f'pqc_process_uptime_seconds {process.get("uptime_seconds", 0)}')
        
        # Health check status (1 = healthy, 0 = not healthy)
        for name, result in health_results.items():
            status_value = 1 if result.status == "healthy" else 0
            lines.append(f'pqc_health_check{{check="{name}"}} {status_value}')
        
        # Operation counts
        operation_counts = all_metrics.get('operation_counts', {})
        for operation, count in operation_counts.items():
            lines.append(f'pqc_operation_total{{operation="{operation}"}} {count}')
        
        return '\n'.join(lines)


# Global monitoring instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor(metrics_collector)
alert_manager = AlertManager(health_monitor)
dashboard = MonitoringDashboard(metrics_collector, health_monitor)


def start_monitoring_thread(interval: int = 60):
    """Start background monitoring thread."""
    
    def monitoring_loop():
        while True:
            try:
                # Run health checks and send alerts
                alert_manager.check_and_send_alerts()
                
                # Record system metrics
                system_metrics = metrics_collector.get_system_metrics()
                if system_metrics:
                    for category, metrics in system_metrics.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                if isinstance(value, (int, float)):
                                    metrics_collector.record_metric(
                                        f"system.{category}.{metric_name}",
                                        value,
                                        tags={"category": category}
                                    )
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    thread = threading.Thread(target=monitoring_loop, daemon=True)
    thread.start()
    return thread


# Decorators for automatic performance tracking
def track_performance(operation_name: str = None):
    """Decorator to automatically track operation performance."""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Start tracking
            operation_id = metrics_collector.record_operation_start(op_name)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                duration = time.time() - start_time
                metrics_collector.record_operation_end(op_name, operation_id, success=True)
                metrics_collector.record_metric(f"{op_name}.success", 1, "operations")
                
                return result
                
            except Exception as e:
                # Record failure
                duration = time.time() - start_time
                metrics_collector.record_operation_end(op_name, operation_id, success=False)
                metrics_collector.record_metric(f"{op_name}.failure", 1, "operations")
                
                raise
        
        return wrapper
    return decorator