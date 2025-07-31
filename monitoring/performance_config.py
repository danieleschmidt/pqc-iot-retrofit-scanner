"""Performance monitoring configuration for PQC IoT Retrofit Scanner."""

import os
import time
import psutil
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import json
import logging


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    cpu_percent: float
    memory_mb: float
    peak_memory_mb: float
    io_read_bytes: int
    io_write_bytes: int
    error_occurred: bool = False
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)


class PerformanceMonitor:
    """Performance monitoring utility for development and production."""
    
    def __init__(self, 
                 enabled: bool = True,
                 log_to_file: bool = True,
                 metrics_file: str = "performance_metrics.jsonl"):
        """Initialize performance monitor.
        
        Args:
            enabled: Whether monitoring is enabled
            log_to_file: Whether to log metrics to file
            metrics_file: File path for metrics storage
        """
        self.enabled = enabled and os.getenv("PQC_PERFORMANCE_MONITORING", "false").lower() == "true"
        self.log_to_file = log_to_file
        self.metrics_file = metrics_file
        self.logger = logging.getLogger(__name__)
        
        # Create metrics directory if needed
        if self.log_to_file:
            os.makedirs(os.path.dirname(metrics_file) or ".", exist_ok=True)
    
    @contextmanager
    def measure(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for measuring operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            context: Additional context information
            
        Yields:
            PerformanceMetrics object that will be populated
        """
        if not self.enabled:
            yield None
            return
        
        # Initialize metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            end_time=0,
            duration_ms=0,
            cpu_percent=0,
            memory_mb=0,
            peak_memory_mb=0,
            io_read_bytes=0,
            io_write_bytes=0,
            context=context
        )
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_io = process.io_counters()
        peak_memory = initial_memory
        
        try:
            # Measure CPU usage during operation
            cpu_percent_start = process.cpu_percent()
            
            yield metrics
            
            # Update final measurements
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            
            final_memory = process.memory_info().rss / 1024 / 1024
            final_io = process.io_counters()
            
            metrics.memory_mb = final_memory
            metrics.peak_memory_mb = max(peak_memory, final_memory)
            metrics.cpu_percent = process.cpu_percent()
            metrics.io_read_bytes = final_io.read_bytes - initial_io.read_bytes
            metrics.io_write_bytes = final_io.write_bytes - initial_io.write_bytes
            
        except Exception as e:
            metrics.error_occurred = True
            metrics.error_message = str(e)
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            
        finally:
            # Log metrics
            self._log_metrics(metrics)
    
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log metrics to configured outputs."""
        if not self.enabled:
            return
        
        # Log to application logger
        self.logger.info(
            f"Performance: {metrics.operation_name} took {metrics.duration_ms:.2f}ms, "
            f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_mb:.1f}MB"
        )
        
        # Log to file if enabled
        if self.log_to_file:
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(json.dumps(metrics.to_dict()) + "\n")
            except Exception as e:
                self.logger.warning(f"Failed to write metrics to file: {e}")
    
    def measure_function(self, operation_name: Optional[str] = None):
        """Decorator for measuring function performance.
        
        Args:
            operation_name: Name for the operation (defaults to function name)
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args_count": len(args),
                    "kwargs_count": len(kwargs)
                }
                
                with self.measure(name, context) as metrics:
                    result = func(*args, **kwargs)
                    if metrics:
                        metrics.context["result_type"] = type(result).__name__
                    return result
            
            return wrapper
        return decorator
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get current system performance information."""
        if not self.enabled:
            return {}
        
        try:
            process = psutil.Process()
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "process_memory_mb": process.memory_info().rss / 1024 / 1024,
                "process_cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads()
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system info: {e}")
            return {}


# Global performance monitor instance
performance_monitor = PerformanceMonitor(
    enabled=True,
    log_to_file=True,
    metrics_file=os.path.join("monitoring", "performance_metrics.jsonl")
)


# Convenience decorators
def measure_performance(operation_name: Optional[str] = None):
    """Convenience decorator for measuring function performance."""
    return performance_monitor.measure_function(operation_name)


@contextmanager
def measure_operation(operation_name: str, **context):
    """Convenience context manager for measuring operations."""
    with performance_monitor.measure(operation_name, context) as metrics:
        yield metrics


class PerformanceProfiler:
    """Advanced performance profiling for development."""
    
    def __init__(self):
        self.enabled = os.getenv("PQC_PROFILING", "false").lower() == "true"
        self.profiles: Dict[str, List[PerformanceMetrics]] = {}
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations over multiple runs."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)
                
                with performance_monitor.measure(operation_name) as metrics:
                    result = func(*args, **kwargs)
                    
                    if metrics:
                        if operation_name not in self.profiles:
                            self.profiles[operation_name] = []
                        self.profiles[operation_name].append(metrics)
                    
                    return result
            return wrapper
        return decorator
    
    def get_statistics(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for an operation."""
        if operation_name not in self.profiles:
            return None
        
        metrics_list = self.profiles[operation_name]
        durations = [m.duration_ms for m in metrics_list]
        memory_usage = [m.memory_mb for m in metrics_list]
        
        return {
            "count": len(durations),
            "avg_duration_ms": sum(durations) / len(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "max_memory_mb": max(memory_usage),
            "error_rate": sum(1 for m in metrics_list if m.error_occurred) / len(metrics_list)
        }
    
    def export_report(self, output_file: str):
        """Export performance report to file."""
        report = {}
        for operation_name in self.profiles:
            report[operation_name] = self.get_statistics(operation_name)
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)


# Global profiler instance
profiler = PerformanceProfiler()