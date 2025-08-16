#!/usr/bin/env python3
"""
Monitoring and Metrics System - Generation 2
Real-time performance monitoring, metrics collection, and alerting
"""

import os
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from enum import Enum
import logging


class MetricType(Enum):
    """Types of metrics to track."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False


class MetricsCollector:
    """Real-time metrics collection and storage."""
    
    def __init__(self, retention_period: int = 3600):
        """Initialize metrics collector."""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.retention_period = retention_period
        self.start_time = time.time()
        self.lock = threading.RLock()
        
        # Built-in system metrics
        self._start_system_monitoring()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None,
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value."""
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {},
                metric_type=metric_type
            )
            
            self.metrics[name].append(metric)
            
            # Handle different metric types
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.TIMER:
                self.timers[name].append(value)
                # Keep only recent timer values
                if len(self.timers[name]) > 1000:
                    self.timers[name] = self.timers[name][-1000:]
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, tags, MetricType.COUNTER)
    
    def time_operation(self, name: str, tags: Dict[str, str] = None):
        """Context manager for timing operations."""
        return TimerContext(self, name, tags)
    
    def get_metric_summary(self, name: str, duration: int = 300) -> Dict[str, Any]:
        """Get summary statistics for a metric over the specified duration."""
        if name not in self.metrics:
            return {"error": f"Metric '{name}' not found"}
        
        cutoff_time = time.time() - duration
        recent_metrics = [
            m for m in self.metrics[name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No recent data", "duration": duration}
        
        values = [m.value for m in recent_metrics]
        
        return {
            "name": name,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
            "duration": duration,
            "first_timestamp": recent_metrics[0].timestamp,
            "last_timestamp": recent_metrics[-1].timestamp
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get current state of all metrics."""
        with self.lock:
            current_metrics = {}
            
            for name, metric_deque in self.metrics.items():
                if metric_deque:
                    latest = metric_deque[-1]
                    current_metrics[name] = {
                        "value": latest.value,
                        "timestamp": latest.timestamp,
                        "age_seconds": time.time() - latest.timestamp
                    }
            
            # Add counter totals
            for name, total in self.counters.items():
                current_metrics[f"{name}_total"] = {
                    "value": total,
                    "timestamp": time.time(),
                    "age_seconds": 0
                }
            
            return current_metrics
    
    def _start_system_monitoring(self):
        """Start background system monitoring."""
        def monitor_system():
            while True:
                try:
                    # Record uptime
                    uptime = time.time() - self.start_time
                    self.record_metric("system_uptime_seconds", uptime)
                    
                    # Record memory usage (simplified)
                    self.record_metric("system_memory_usage_mb", 128.5)  # Mock value
                    
                    # Record CPU usage (simplified)
                    self.record_metric("system_cpu_usage_percent", 15.3)  # Mock value
                    
                    # Record metrics count
                    total_metrics = sum(len(deque) for deque in self.metrics.values())
                    self.record_metric("metrics_stored_count", total_metrics)
                    
                except Exception as e:
                    logging.error(f"System monitoring error: {e}")
                
                time.sleep(30)  # Monitor every 30 seconds
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Dict[str, str] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_metric(
                f"{self.name}_duration_seconds", 
                duration, 
                self.tags, 
                MetricType.TIMER
            )


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager."""
        self.metrics_collector = metrics_collector
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        
        # Start alert monitoring
        self._start_alert_monitoring()
    
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      condition: str = "gt", severity: AlertSeverity = AlertSeverity.WARNING,
                      duration: int = 60):
        """Add an alert rule."""
        rule = {
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,  # "gt", "lt", "eq"
            "severity": severity,
            "duration": duration,
            "enabled": True
        }
        self.alert_rules.append(rule)
    
    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules against current metrics."""
        with self.lock:
            for rule in self.alert_rules:
                if not rule["enabled"]:
                    continue
                
                metric_name = rule["metric_name"]
                summary = self.metrics_collector.get_metric_summary(
                    metric_name, rule["duration"]
                )
                
                if "error" in summary:
                    continue
                
                current_value = summary["latest"]
                threshold = rule["threshold"]
                condition = rule["condition"]
                
                # Evaluate condition
                should_alert = False
                if condition == "gt" and current_value > threshold:
                    should_alert = True
                elif condition == "lt" and current_value < threshold:
                    should_alert = True
                elif condition == "eq" and abs(current_value - threshold) < 0.001:
                    should_alert = True
                
                alert_id = f"{metric_name}_{condition}_{threshold}"
                
                if should_alert:
                    if alert_id not in self.active_alerts:
                        # Create new alert
                        alert = Alert(
                            id=alert_id,
                            severity=rule["severity"],
                            message=f"{metric_name} {condition} {threshold} (current: {current_value})",
                            timestamp=time.time(),
                            metric_name=metric_name,
                            threshold=threshold,
                            current_value=current_value
                        )
                        
                        self.active_alerts[alert_id] = alert
                        self.alert_history.append(alert)
                        
                        # Send notifications
                        self._send_notifications(alert)
                else:
                    # Resolve alert if it exists
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        alert.resolved = True
                        del self.active_alerts[alert_id]
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Notification callback error: {e}")
    
    def _start_alert_monitoring(self):
        """Start background alert monitoring."""
        def monitor_alerts():
            while True:
                try:
                    self._evaluate_alerts()
                except Exception as e:
                    logging.error(f"Alert monitoring error: {e}")
                
                time.sleep(10)  # Check every 10 seconds
        
        monitor_thread = threading.Thread(target=monitor_alerts, daemon=True)
        monitor_thread.start()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts."""
        with self.lock:
            return {
                "active_alerts": len(self.active_alerts),
                "total_rules": len(self.alert_rules),
                "alerts_by_severity": {
                    severity.value: sum(
                        1 for alert in self.active_alerts.values() 
                        if alert.severity == severity
                    )
                    for severity in AlertSeverity
                },
                "recent_alerts": [
                    asdict(alert) for alert in self.alert_history[-10:]
                ]
            }


class PerformanceProfiler:
    """Performance profiling and optimization insights."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance profiler."""
        self.metrics_collector = metrics_collector
        self.function_timings: Dict[str, List[float]] = defaultdict(list)
        self.optimization_recommendations: List[str] = []
    
    def profile_function(self, func_name: str):
        """Decorator for profiling function performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    
                    # Record timing
                    self.function_timings[func_name].append(duration)
                    self.metrics_collector.record_metric(
                        f"function_{func_name}_duration", duration, 
                        {"success": str(success)}, MetricType.TIMER
                    )
                    
                    # Analyze performance
                    self._analyze_performance(func_name, duration)
                
                return result
            return wrapper
        return decorator
    
    def _analyze_performance(self, func_name: str, duration: float):
        """Analyze function performance and generate recommendations."""
        timings = self.function_timings[func_name]
        
        if len(timings) < 10:
            return  # Need more samples
        
        avg_duration = sum(timings) / len(timings)
        
        # Generate recommendations based on performance patterns
        if duration > avg_duration * 2:
            recommendation = f"Function '{func_name}' took {duration:.3f}s (2x avg), consider optimization"
            if recommendation not in self.optimization_recommendations:
                self.optimization_recommendations.append(recommendation)
        
        if avg_duration > 1.0:
            recommendation = f"Function '{func_name}' average duration {avg_duration:.3f}s is high"
            if recommendation not in self.optimization_recommendations:
                self.optimization_recommendations.append(recommendation)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            "function_performance": {},
            "recommendations": self.optimization_recommendations[-10:],  # Last 10
            "summary": {
                "total_functions_profiled": len(self.function_timings),
                "total_recommendations": len(self.optimization_recommendations)
            }
        }
        
        # Analyze each function
        for func_name, timings in self.function_timings.items():
            if timings:
                report["function_performance"][func_name] = {
                    "call_count": len(timings),
                    "avg_duration": sum(timings) / len(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                    "total_time": sum(timings),
                    "latest_duration": timings[-1]
                }
        
        return report


def console_alert_handler(alert: Alert):
    """Simple console alert handler."""
    severity_emoji = {
        AlertSeverity.CRITICAL: "🚨",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.INFO: "ℹ️"
    }
    
    emoji = severity_emoji.get(alert.severity, "⚠️")
    print(f"{emoji} ALERT [{alert.severity.value.upper()}]: {alert.message}")


def main():
    """Demo of monitoring and metrics system."""
    print("Monitoring and Metrics System - Demo")
    print("=" * 50)
    
    # Initialize components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager(metrics_collector)
    profiler = PerformanceProfiler(metrics_collector)
    
    # Add console alert handler
    alert_manager.add_notification_callback(console_alert_handler)
    
    # Add some alert rules
    alert_manager.add_alert_rule("cpu_usage_percent", 80.0, "gt", AlertSeverity.WARNING)
    alert_manager.add_alert_rule("memory_usage_mb", 1000.0, "gt", AlertSeverity.CRITICAL)
    alert_manager.add_alert_rule("error_rate", 5.0, "gt", AlertSeverity.WARNING)
    
    print("Alert rules configured:")
    print("- CPU usage > 80% (WARNING)")
    print("- Memory usage > 1000MB (CRITICAL)")
    print("- Error rate > 5/min (WARNING)")
    
    # Simulate some metrics
    print("\nSimulating metrics collection...")
    
    # Simulate firmware analysis metrics
    with metrics_collector.time_operation("firmware_analysis"):
        time.sleep(0.1)  # Simulate work
    
    metrics_collector.increment_counter("firmware_files_processed")
    metrics_collector.increment_counter("vulnerabilities_detected", 3)
    metrics_collector.record_metric("analysis_accuracy_percent", 94.5)
    
    # Simulate varying CPU usage
    for i in range(5):
        cpu_usage = 60 + i * 8  # Gradually increase
        metrics_collector.record_metric("cpu_usage_percent", cpu_usage)
        time.sleep(0.5)
    
    # Create a function to profile
    @profiler.profile_function("crypto_detection")
    def simulate_crypto_detection():
        time.sleep(0.05)  # Simulate crypto detection work
        return "RSA-2048 detected"
    
    # Run profiled function multiple times
    for _ in range(5):
        result = simulate_crypto_detection()
        metrics_collector.increment_counter("crypto_patterns_found")
    
    # Wait a bit for monitoring to catch alerts
    print("Waiting for alert monitoring...")
    time.sleep(2)
    
    # Show current metrics
    print("\nCurrent Metrics:")
    current_metrics = metrics_collector.get_all_metrics()
    for name, data in current_metrics.items():
        print(f"  {name}: {data['value']}")
    
    # Show metric summaries
    print("\nMetric Summaries (last 5 minutes):")
    for metric_name in ["cpu_usage_percent", "firmware_analysis_duration_seconds"]:
        summary = metrics_collector.get_metric_summary(metric_name, 300)
        if "error" not in summary:
            print(f"  {metric_name}:")
            print(f"    Count: {summary['count']}")
            print(f"    Avg: {summary['avg']:.3f}")
            print(f"    Min/Max: {summary['min']:.3f}/{summary['max']:.3f}")
    
    # Show alerts
    print("\nAlert Summary:")
    alert_summary = alert_manager.get_alert_summary()
    print(json.dumps(alert_summary, indent=2, default=str))
    
    # Show performance report
    print("\nPerformance Report:")
    perf_report = profiler.get_performance_report()
    print(json.dumps(perf_report, indent=2))
    
    print("\nMonitoring system demo complete!")


if __name__ == '__main__':
    main()