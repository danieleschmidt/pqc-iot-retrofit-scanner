#!/usr/bin/env python3
"""
Autonomous Reliability Engine - Generation 2 Implementation
Self-healing systems with predictive failure detection and automatic recovery.
"""

import json
import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
import asyncio
import hashlib
import pickle
import sys

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"

class FailureType(Enum):
    """Types of system failures."""
    TRANSIENT = "transient"
    PERSISTENT = "persistent"
    CASCADING = "cascading"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"

@dataclass
class HealthMetric:
    """Individual health metric measurement."""
    name: str
    value: float
    timestamp: datetime
    status: HealthStatus
    threshold_warning: float = 0.8
    threshold_critical: float = 0.95
    
    def evaluate_status(self) -> HealthStatus:
        """Evaluate health status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

@dataclass
class FailureEvent:
    """Represents a system failure event."""
    id: str
    type: FailureType
    component: str
    message: str
    timestamp: datetime
    stack_trace: Optional[str] = None
    recovery_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class CircuitBreaker:
    """Circuit breaker pattern for failure isolation."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, expected_exception: type = Exception):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def __call__(self, func):
        """Decorator to wrap function with circuit breaker."""
        def wrapper(*args, **kwargs):
            if self.state == "open":
                if self._should_attempt_reset():
                    self.state = "half_open"
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

class RetryManager:
    """Advanced retry mechanism with exponential backoff."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_multiplier: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    delay = min(
                        self.base_delay * (self.backoff_multiplier ** attempt),
                        self.max_delay
                    )
                    time.sleep(delay)
                else:
                    break
        
        raise last_exception

class HealthMonitor:
    """Continuous health monitoring with predictive capabilities."""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.thresholds = {}
        self.anomaly_detectors = {}
        self.running = False
        self.monitor_thread = None
        
    def register_metric(self, name: str, threshold_warning: float = 0.8, 
                       threshold_critical: float = 0.95):
        """Register a new metric for monitoring."""
        self.thresholds[name] = {
            'warning': threshold_warning,
            'critical': threshold_critical
        }
        
    def record_metric(self, name: str, value: float):
        """Record a metric value."""
        timestamp = datetime.now(timezone.utc)
        metric = HealthMetric(
            name=name,
            value=value,
            timestamp=timestamp,
            status=HealthStatus.HEALTHY,
            threshold_warning=self.thresholds.get(name, {}).get('warning', 0.8),
            threshold_critical=self.thresholds.get(name, {}).get('critical', 0.95)
        )
        metric.status = metric.evaluate_status()
        self.metrics[name].append(metric)
        
    def get_current_health(self) -> Dict[str, HealthStatus]:
        """Get current health status for all metrics."""
        health = {}
        for name, metric_queue in self.metrics.items():
            if metric_queue:
                latest_metric = metric_queue[-1]
                health[name] = latest_metric.status
            else:
                health[name] = HealthStatus.HEALTHY
        return health
    
    def detect_anomalies(self, name: str) -> List[str]:
        """Detect anomalies in metric patterns."""
        if name not in self.metrics or len(self.metrics[name]) < 10:
            return []
        
        values = [m.value for m in self.metrics[name]]
        recent_values = values[-5:]  # Last 5 values
        historical_avg = sum(values[:-5]) / len(values[:-5]) if len(values) > 5 else 0
        recent_avg = sum(recent_values) / len(recent_values)
        
        anomalies = []
        
        # Detect sudden spikes
        if recent_avg > historical_avg * 1.5:
            anomalies.append(f"Sudden spike detected in {name}: {recent_avg:.2f} vs {historical_avg:.2f}")
        
        # Detect continuous degradation
        if len(recent_values) >= 3 and all(recent_values[i] > recent_values[i-1] for i in range(1, len(recent_values))):
            anomalies.append(f"Continuous degradation detected in {name}")
            
        return anomalies

class AutonomousReliabilityEngine:
    """
    Autonomous Reliability Engine with Self-Healing Capabilities.
    
    Provides comprehensive error handling, monitoring, recovery, and 
    predictive failure prevention for production systems.
    """
    
    def __init__(self, project_root: Path = None):
        """Initialize the Autonomous Reliability Engine."""
        self.project_root = project_root or Path.cwd()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Core components
        self.health_monitor = HealthMonitor()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.failure_events: List[FailureEvent] = []
        self.recovery_strategies: Dict[str, List[Callable]] = defaultdict(list)
        
        # State management
        self.system_health = HealthStatus.HEALTHY
        self.running = False
        self.monitor_thread = None
        
        # Configuration
        self.config = {
            'health_check_interval': 30,  # seconds
            'failure_retention_days': 7,
            'auto_recovery_enabled': True,
            'predictive_analysis_enabled': True
        }
        
        # Setup logging
        self._setup_logging()
        self.logger.info("🛡️ Autonomous Reliability Engine initialized")
        
    def _setup_logging(self):
        """Setup comprehensive logging system."""
        log_file = self.logs_dir / f"reliability_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(component)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Custom logger with component context
        self.logger = logging.getLogger(__name__)
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.component = getattr(record, 'component', 'RELIABILITY')
            return record
            
        logging.setLogRecordFactory(record_factory)
    
    @contextmanager
    def error_boundary(self, component: str, auto_recover: bool = True):
        """Context manager for error boundary with automatic recovery."""
        try:
            yield
        except Exception as e:
            failure_event = self._create_failure_event(component, e)
            self.failure_events.append(failure_event)
            
            self.logger.error(f"💥 Failure in {component}: {str(e)}")
            
            if auto_recover:
                self._attempt_recovery(failure_event)
            
            # Re-raise if not recovered
            if not failure_event.resolved:
                raise e
    
    def _create_failure_event(self, component: str, exception: Exception) -> FailureEvent:
        """Create failure event from exception."""
        failure_id = hashlib.md5(
            f"{component}_{str(exception)}_{time.time()}".encode()
        ).hexdigest()[:8]
        
        failure_type = self._classify_failure(exception)
        
        return FailureEvent(
            id=failure_id,
            type=failure_type,
            component=component,
            message=str(exception),
            timestamp=datetime.now(timezone.utc),
            stack_trace=traceback.format_exc()
        )
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type based on exception."""
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return FailureType.TRANSIENT
        elif isinstance(exception, MemoryError):
            return FailureType.RESOURCE_EXHAUSTION
        elif isinstance(exception, (ImportError, ModuleNotFoundError)):
            return FailureType.DEPENDENCY
        elif isinstance(exception, (ValueError, TypeError)):
            return FailureType.CONFIGURATION
        else:
            return FailureType.PERSISTENT
    
    def _attempt_recovery(self, failure_event: FailureEvent):
        """Attempt automatic recovery from failure."""
        component = failure_event.component
        
        if component in self.recovery_strategies:
            for strategy in self.recovery_strategies[component]:
                try:
                    self.logger.info(f"🔄 Attempting recovery for {component}")
                    strategy(failure_event)
                    failure_event.resolved = True
                    failure_event.resolution_time = datetime.now(timezone.utc)
                    failure_event.recovery_actions.append(f"Applied strategy: {strategy.__name__}")
                    self.logger.info(f"✅ Recovery successful for {component}")
                    break
                except Exception as recovery_error:
                    self.logger.warning(f"❌ Recovery strategy failed: {recovery_error}")
                    continue
    
    def register_circuit_breaker(self, name: str, failure_threshold: int = 5, 
                                recovery_timeout: int = 60) -> CircuitBreaker:
        """Register a new circuit breaker."""
        cb = CircuitBreaker(name, failure_threshold, recovery_timeout)
        self.circuit_breakers[name] = cb
        return cb
    
    def register_recovery_strategy(self, component: str, strategy: Callable):
        """Register recovery strategy for a component."""
        self.recovery_strategies[component].append(strategy)
        self.logger.info(f"📝 Registered recovery strategy for {component}")
    
    def start_monitoring(self):
        """Start autonomous health monitoring."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("🎯 Autonomous monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("⏹️ Monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._analyze_health_trends()
                self._perform_predictive_analysis()
                self._cleanup_old_events()
                
                time.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                self.logger.error(f"💥 Error in monitoring loop: {e}")
                time.sleep(10)  # Fallback delay
    
    def _collect_system_metrics(self):
        """Collect system health metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_usage = psutil.cpu_percent() / 100.0
            self.health_monitor.record_metric("cpu_usage", cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            self.health_monitor.record_metric("memory_usage", memory_usage)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent / 100.0
            self.health_monitor.record_metric("disk_usage", disk_usage)
            
        except ImportError:
            # Fallback metrics without psutil
            import os
            
            # Simple load average (Unix only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()[0] / os.cpu_count()
                self.health_monitor.record_metric("system_load", load_avg)
    
    def _analyze_health_trends(self):
        """Analyze health trends and detect degradation."""
        current_health = self.health_monitor.get_current_health()
        overall_status = HealthStatus.HEALTHY
        
        critical_count = sum(1 for status in current_health.values() 
                           if status == HealthStatus.CRITICAL)
        degraded_count = sum(1 for status in current_health.values() 
                           if status == HealthStatus.DEGRADED)
        
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        
        if overall_status != self.system_health:
            self.logger.warning(f"🏥 System health changed: {self.system_health.value} -> {overall_status.value}")
            self.system_health = overall_status
    
    def _perform_predictive_analysis(self):
        """Perform predictive failure analysis."""
        if not self.config['predictive_analysis_enabled']:
            return
        
        for metric_name in self.health_monitor.metrics:
            anomalies = self.health_monitor.detect_anomalies(metric_name)
            for anomaly in anomalies:
                self.logger.warning(f"🔮 Predictive alert: {anomaly}")
                # Could trigger proactive recovery actions here
    
    def _cleanup_old_events(self):
        """Clean up old failure events."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(
            days=self.config['failure_retention_days']
        )
        
        old_count = len(self.failure_events)
        self.failure_events = [
            event for event in self.failure_events 
            if event.timestamp > cutoff_time
        ]
        
        cleaned = old_count - len(self.failure_events)
        if cleaned > 0:
            self.logger.info(f"🧹 Cleaned up {cleaned} old failure events")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        current_health = self.health_monitor.get_current_health()
        
        # Recent failures (last 24 hours)
        recent_failures = [
            event for event in self.failure_events
            if event.timestamp > datetime.now(timezone.utc) - timedelta(hours=24)
        ]
        
        # Circuit breaker states
        cb_states = {name: cb.state for name, cb in self.circuit_breakers.items()}
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_health": self.system_health.value,
            "metric_health": {name: status.value for name, status in current_health.items()},
            "recent_failures_24h": len(recent_failures),
            "circuit_breaker_states": cb_states,
            "recovery_strategies_registered": {
                component: len(strategies) 
                for component, strategies in self.recovery_strategies.items()
            },
            "monitoring_active": self.running
        }
    
    def export_reliability_metrics(self) -> Dict[str, Any]:
        """Export comprehensive reliability metrics for analysis."""
        return {
            "health_metrics": {
                name: [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "status": m.status.value
                    }
                    for m in metrics
                ]
                for name, metrics in self.health_monitor.metrics.items()
            },
            "failure_events": [
                {
                    "id": event.id,
                    "type": event.type.value,
                    "component": event.component,
                    "message": event.message,
                    "timestamp": event.timestamp.isoformat(),
                    "resolved": event.resolved,
                    "recovery_actions": event.recovery_actions
                }
                for event in self.failure_events
            ],
            "circuit_breakers": {
                name: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "failure_threshold": cb.failure_threshold
                }
                for name, cb in self.circuit_breakers.items()
            }
        }

# Example recovery strategies
def restart_component_strategy(failure_event: FailureEvent):
    """Example recovery strategy: restart component."""
    logging.info(f"🔄 Restarting component: {failure_event.component}")
    # Implementation would restart the specific component
    time.sleep(1)  # Simulate restart time

def fallback_strategy(failure_event: FailureEvent):
    """Example recovery strategy: fallback to backup system."""
    logging.info(f"🔄 Activating fallback for: {failure_event.component}")
    # Implementation would activate backup system
    time.sleep(0.5)

def cache_invalidation_strategy(failure_event: FailureEvent):
    """Example recovery strategy: invalidate cache."""
    logging.info(f"🔄 Invalidating cache for: {failure_event.component}")
    # Implementation would clear relevant caches
    pass

def main():
    """Demonstrate Autonomous Reliability Engine."""
    print("🛡️ Autonomous Reliability Engine - Generation 2")
    
    # Initialize engine
    engine = AutonomousReliabilityEngine()
    
    # Register metrics
    engine.health_monitor.register_metric("cpu_usage", 0.7, 0.9)
    engine.health_monitor.register_metric("memory_usage", 0.8, 0.95)
    
    # Register recovery strategies
    engine.register_recovery_strategy("firmware_scanner", restart_component_strategy)
    engine.register_recovery_strategy("pqc_patcher", fallback_strategy)
    engine.register_recovery_strategy("cache_system", cache_invalidation_strategy)
    
    # Register circuit breakers
    scanner_cb = engine.register_circuit_breaker("firmware_scanner", 3, 30)
    patcher_cb = engine.register_circuit_breaker("pqc_patcher", 5, 60)
    
    # Start monitoring
    engine.start_monitoring()
    
    # Simulate some operations with error boundaries
    print("\n📊 Simulating system operations...")
    
    try:
        # Simulate successful operation
        with engine.error_boundary("firmware_scanner"):
            time.sleep(0.1)
            print("✅ Firmware scan completed successfully")
        
        # Simulate failure with recovery
        with engine.error_boundary("pqc_patcher"):
            raise ConnectionError("Network timeout during patch download")
            
    except Exception as e:
        print(f"❌ Operation failed: {e}")
    
    # Generate health report
    time.sleep(2)  # Allow monitoring to collect some data
    
    print("\n📋 Health Report:")
    report = engine.get_health_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Export metrics
    print("\n📊 Reliability Metrics:")
    metrics = engine.export_reliability_metrics()
    print(f"  Health metrics tracked: {len(metrics['health_metrics'])}")
    print(f"  Failure events recorded: {len(metrics['failure_events'])}")
    print(f"  Circuit breakers active: {len(metrics['circuit_breakers'])}")
    
    # Stop monitoring
    engine.stop_monitoring()
    
    print("\n🎯 Generation 2 implementation complete!")
    print("💡 Ready for Generation 3: Performance optimization and scaling")

if __name__ == "__main__":
    main()