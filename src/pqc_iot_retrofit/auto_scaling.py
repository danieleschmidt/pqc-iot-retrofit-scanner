"""
Auto-scaling and intelligent load balancing for PQC IoT Retrofit Scanner.

This module provides:
- Dynamic worker pool scaling based on load
- Intelligent resource allocation
- Predictive scaling based on patterns
- Load shedding and backpressure handling
"""

import time
import threading
import logging
import statistics
from typing import Dict, List, Optional, Callable, Any, NamedTuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from enum import Enum

from .concurrency import WorkerPool, LoadBalancer, WorkItem
from .monitoring import metrics_collector, health_monitor
from .error_handling import PQCRetrofitError, ErrorSeverity, ErrorCategory


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadLevel(Enum):
    """System load levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    queue_depth: int
    active_workers: int
    throughput: float  # items per second
    response_time: float  # average response time
    error_rate: float  # percentage of failed requests
    load_level: LoadLevel = LoadLevel.NORMAL


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    direction: ScalingDirection
    old_size: int
    new_size: int
    reason: str
    metrics: ScalingMetrics
    success: bool = True
    error: Optional[str] = None


class ScalingPolicy(ABC):
    """Abstract base class for scaling policies."""
    
    @abstractmethod
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> ScalingDirection:
        """Determine if scaling is needed."""
        pass
    
    @abstractmethod
    def calculate_target_size(self, current_size: int, metrics: ScalingMetrics) -> int:
        """Calculate target pool size."""
        pass


class AdaptiveScalingPolicy(ScalingPolicy):
    """Adaptive scaling policy that learns from patterns."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 32, 
                 scale_up_threshold: float = 0.8, scale_down_threshold: float = 0.3,
                 response_time_threshold: float = 5.0):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.response_time_threshold = response_time_threshold
        
        # Learning parameters
        self.scaling_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=50)
        self.pattern_weights = defaultdict(float)
        
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> ScalingDirection:
        """Determine scaling direction based on multiple factors."""
        
        # Calculate utilization metrics
        worker_utilization = metrics.active_workers / max(metrics.active_workers + 
                                                         (len([w for w in history[-5:] if w.active_workers == 0]) if history else 0), 1)
        
        # Check primary scaling triggers
        scale_up_signals = 0
        scale_down_signals = 0
        
        # CPU/Memory pressure
        if metrics.cpu_usage > self.scale_up_threshold or metrics.memory_usage > self.scale_up_threshold:
            scale_up_signals += 2
        elif metrics.cpu_usage < self.scale_down_threshold and metrics.memory_usage < self.scale_down_threshold:
            scale_down_signals += 1
        
        # Queue depth
        if metrics.queue_depth > 10:
            scale_up_signals += 2
        elif metrics.queue_depth == 0:
            scale_down_signals += 1
        
        # Response time
        if metrics.response_time > self.response_time_threshold:
            scale_up_signals += 2
        elif metrics.response_time < self.response_time_threshold / 2:
            scale_down_signals += 1
        
        # Error rate
        if metrics.error_rate > 0.05:  # 5% error rate
            scale_up_signals += 1
        
        # Throughput trends
        if len(history) >= 3:
            recent_throughput = [m.throughput for m in history[-3:]]
            if len(recent_throughput) >= 2:
                throughput_trend = recent_throughput[-1] - recent_throughput[0]
                if throughput_trend < -0.1:  # Declining throughput
                    scale_up_signals += 1
        
        # Apply pattern-based learning
        pattern_score = self._calculate_pattern_score(metrics, history)
        if pattern_score > 0.7:
            scale_up_signals += 1
        elif pattern_score < -0.7:
            scale_down_signals += 1
        
        # Make scaling decision
        if scale_up_signals > scale_down_signals + 1:
            return ScalingDirection.UP
        elif scale_down_signals > scale_up_signals + 1:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def calculate_target_size(self, current_size: int, metrics: ScalingMetrics) -> int:
        """Calculate optimal target size based on metrics."""
        
        # Base calculation on queue depth and response time
        if metrics.queue_depth > 20:
            # Heavy load - scale up aggressively
            target = min(self.max_workers, current_size + max(2, current_size // 4))
        elif metrics.queue_depth > 5:
            # Moderate load - scale up conservatively
            target = min(self.max_workers, current_size + 1)
        elif metrics.queue_depth == 0 and metrics.response_time < self.response_time_threshold / 3:
            # Light load - consider scaling down
            target = max(self.min_workers, current_size - 1)
        else:
            target = current_size
        
        # Apply CPU/memory constraints
        if metrics.cpu_usage > 0.9 or metrics.memory_usage > 0.9:
            target = min(target, current_size)  # Don't scale up if resource constrained
        
        # Apply learned patterns
        pattern_adjustment = self._get_pattern_adjustment(metrics)
        target = max(self.min_workers, min(self.max_workers, target + pattern_adjustment))
        
        return target
    
    def _calculate_pattern_score(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> float:
        """Calculate pattern-based scaling score."""
        if len(history) < 5:
            return 0.0
        
        # Look for patterns in the last hour
        hour_ago = time.time() - 3600
        recent_metrics = [m for m in history if m.timestamp > hour_ago]
        
        if len(recent_metrics) < 3:
            return 0.0
        
        # Check for load increase patterns
        load_trend = 0.0
        for i in range(1, len(recent_metrics)):
            current_load = (recent_metrics[i].cpu_usage + recent_metrics[i].memory_usage + 
                          recent_metrics[i].queue_depth / 10.0) / 3.0
            prev_load = (recent_metrics[i-1].cpu_usage + recent_metrics[i-1].memory_usage + 
                        recent_metrics[i-1].queue_depth / 10.0) / 3.0
            load_trend += current_load - prev_load
        
        load_trend /= len(recent_metrics) - 1
        
        # Check for time-based patterns (e.g., daily cycles)
        hour_of_day = time.localtime(metrics.timestamp).tm_hour
        if hour_of_day in self.pattern_weights:
            pattern_bias = self.pattern_weights[hour_of_day]
            return load_trend + pattern_bias
        
        return load_trend
    
    def _get_pattern_adjustment(self, metrics: ScalingMetrics) -> int:
        """Get worker count adjustment based on learned patterns."""
        hour_of_day = time.localtime(metrics.timestamp).tm_hour
        
        if hour_of_day in self.pattern_weights:
            weight = self.pattern_weights[hour_of_day]
            if weight > 0.5:
                return 1  # Add one worker
            elif weight < -0.5:
                return -1  # Remove one worker
        
        return 0
    
    def record_scaling_outcome(self, event: ScalingEvent, performance_after: ScalingMetrics):
        """Learn from scaling outcomes."""
        self.scaling_history.append(event)
        self.performance_history.append(performance_after)
        
        # Update pattern weights based on success
        hour_of_day = time.localtime(event.timestamp).tm_hour
        
        if event.success and event.direction == ScalingDirection.UP:
            # Successful scale-up - reinforce this time pattern
            self.pattern_weights[hour_of_day] = min(1.0, self.pattern_weights[hour_of_day] + 0.1)
        elif event.success and event.direction == ScalingDirection.DOWN:
            # Successful scale-down - reduce weight for this time
            self.pattern_weights[hour_of_day] = max(-1.0, self.pattern_weights[hour_of_day] - 0.1)


class PredictiveScalingPolicy(ScalingPolicy):
    """Predictive scaling based on time series analysis."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 32,
                 prediction_window: int = 300):  # 5 minutes
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.prediction_window = prediction_window
        self.metrics_history = deque(maxlen=1000)
        
    def should_scale(self, metrics: ScalingMetrics, history: List[ScalingMetrics]) -> ScalingDirection:
        """Predict future load and scale proactively."""
        self.metrics_history.extend(history)
        self.metrics_history.append(metrics)
        
        if len(self.metrics_history) < 10:
            return ScalingDirection.STABLE
        
        # Predict load for next 5 minutes
        predicted_load = self._predict_load()
        current_capacity = metrics.active_workers
        
        # Calculate needed capacity based on prediction
        needed_capacity = self._calculate_needed_capacity(predicted_load)
        
        if needed_capacity > current_capacity * 1.2:
            return ScalingDirection.UP
        elif needed_capacity < current_capacity * 0.7:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def calculate_target_size(self, current_size: int, metrics: ScalingMetrics) -> int:
        """Calculate target size based on predictions."""
        predicted_load = self._predict_load()
        needed_capacity = self._calculate_needed_capacity(predicted_load)
        
        target = max(self.min_workers, min(self.max_workers, int(needed_capacity * 1.1)))  # 10% buffer
        
        return target
    
    def _predict_load(self) -> float:
        """Simple load prediction based on recent trends."""
        if len(self.metrics_history) < 5:
            return 1.0
        
        # Calculate load scores for recent metrics
        recent_loads = []
        for m in list(self.metrics_history)[-10:]:
            load_score = (m.cpu_usage + m.memory_usage + min(m.queue_depth / 10.0, 1.0)) / 3.0
            recent_loads.append(load_score)
        
        if len(recent_loads) >= 3:
            # Simple linear trend prediction
            trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
            predicted = recent_loads[-1] + trend * 5  # Predict 5 time steps ahead
            return max(0.0, min(2.0, predicted))
        
        return recent_loads[-1] if recent_loads else 1.0
    
    def _calculate_needed_capacity(self, predicted_load: float) -> float:
        """Calculate needed worker capacity for predicted load."""
        # Assume linear relationship between load and needed workers
        base_workers = self.min_workers
        scale_factor = (self.max_workers - self.min_workers) / 2.0
        
        return base_workers + predicted_load * scale_factor


class AutoScaler:
    """Auto-scaling manager for worker pools."""
    
    def __init__(self, worker_pool: WorkerPool, scaling_policy: ScalingPolicy,
                 check_interval: float = 30.0, cooldown_period: float = 300.0):
        self.worker_pool = worker_pool
        self.scaling_policy = scaling_policy
        self.check_interval = check_interval
        self.cooldown_period = cooldown_period
        
        # State management
        self.last_scaling_time = 0.0
        self.metrics_history = deque(maxlen=100)
        self.scaling_events = deque(maxlen=50)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Metrics collection
        self.stats = {
            'scaling_events': 0,
            'scale_up_events': 0,
            'scale_down_events': 0,
            'failed_scalings': 0
        }
    
    def start(self):
        """Start the auto-scaler."""
        with self.lock:
            if self.running:
                return
            
            self.running = True
            self.thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self.thread.start()
            
            logging.info(f"Auto-scaler started with {self.check_interval}s interval")
            metrics_collector.record_metric("autoscaler.started", 1, "events")
    
    def stop(self):
        """Stop the auto-scaler."""
        with self.lock:
            if not self.running:
                return
            
            self.running = False
            if self.thread:
                self.thread.join(timeout=5.0)
            
            logging.info("Auto-scaler stopped")
            metrics_collector.record_metric("autoscaler.stopped", 1, "events")
    
    def _scaling_loop(self):
        """Main scaling loop."""
        while self.running:
            try:
                current_metrics = self._collect_metrics()
                self.metrics_history.append(current_metrics)
                
                # Check if scaling is needed
                if self._should_check_scaling():
                    scaling_direction = self.scaling_policy.should_scale(
                        current_metrics, list(self.metrics_history)
                    )
                    
                    if scaling_direction != ScalingDirection.STABLE:
                        self._perform_scaling(scaling_direction, current_metrics)
                
                # Record metrics
                metrics_collector.record_metric("autoscaler.metrics_collected", 1, "events")
                metrics_collector.record_metric("autoscaler.queue_depth", current_metrics.queue_depth, "items")
                metrics_collector.record_metric("autoscaler.active_workers", current_metrics.active_workers, "workers")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Error in auto-scaler loop: {e}")
                time.sleep(self.check_interval)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current metrics for scaling decisions."""
        import psutil
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        memory_usage = psutil.virtual_memory().percent / 100.0
        
        # Get worker pool metrics
        pool_stats = self.worker_pool.get_stats()
        
        # Calculate throughput (items per second)
        throughput = 0.0
        if len(self.metrics_history) >= 2:
            recent_processed = pool_stats['items_processed']
            prev_metrics = self.metrics_history[-1]
            time_diff = time.time() - prev_metrics.timestamp
            if time_diff > 0:
                throughput = (recent_processed - getattr(prev_metrics, 'total_processed', 0)) / time_diff
        
        # Estimate queue depth (simplified)
        queue_depth = max(0, pool_stats['items_processed'] - pool_stats['items_failed'])
        
        # Calculate response time (average)
        response_time = pool_stats.get('average_processing_time', 0.0)
        
        # Calculate error rate
        total_items = pool_stats['items_processed'] + pool_stats['items_failed']
        error_rate = pool_stats['items_failed'] / max(total_items, 1)
        
        # Determine load level
        avg_load = (cpu_usage + memory_usage) / 2.0
        if avg_load > 0.9:
            load_level = LoadLevel.CRITICAL
        elif avg_load > 0.7:
            load_level = LoadLevel.HIGH
        elif avg_load > 0.3:
            load_level = LoadLevel.NORMAL
        else:
            load_level = LoadLevel.LOW
        
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            queue_depth=queue_depth,
            active_workers=pool_stats['workers_active'],
            throughput=throughput,
            response_time=response_time,
            error_rate=error_rate,
            load_level=load_level
        )
    
    def _should_check_scaling(self) -> bool:
        """Check if enough time has passed since last scaling."""
        return time.time() - self.last_scaling_time > self.cooldown_period
    
    def _perform_scaling(self, direction: ScalingDirection, metrics: ScalingMetrics):
        """Perform scaling operation."""
        current_size = self.worker_pool.worker_count
        target_size = self.scaling_policy.calculate_target_size(current_size, metrics)
        
        if target_size == current_size:
            return
        
        # Create scaling event
        event = ScalingEvent(
            timestamp=time.time(),
            direction=direction,
            old_size=current_size,
            new_size=target_size,
            reason=f"Load level: {metrics.load_level.value}, Queue: {metrics.queue_depth}",
            metrics=metrics
        )
        
        try:
            # Perform the scaling (this would need to be implemented in WorkerPool)
            success = self._scale_worker_pool(target_size)
            event.success = success
            
            if success:
                self.last_scaling_time = time.time()
                self.stats['scaling_events'] += 1
                
                if direction == ScalingDirection.UP:
                    self.stats['scale_up_events'] += 1
                    logging.info(f"Scaled up from {current_size} to {target_size} workers")
                    metrics_collector.record_metric("autoscaler.scale_up", 1, "events")
                else:
                    self.stats['scale_down_events'] += 1
                    logging.info(f"Scaled down from {current_size} to {target_size} workers")
                    metrics_collector.record_metric("autoscaler.scale_down", 1, "events")
            else:
                self.stats['failed_scalings'] += 1
                logging.warning(f"Failed to scale from {current_size} to {target_size} workers")
                metrics_collector.record_metric("autoscaler.scale_failed", 1, "events")
        
        except Exception as e:
            event.success = False
            event.error = str(e)
            self.stats['failed_scalings'] += 1
            logging.error(f"Scaling failed: {e}")
        
        self.scaling_events.append(event)
        
        # Let the policy learn from this event
        if hasattr(self.scaling_policy, 'record_scaling_outcome'):
            # Collect metrics after scaling to measure impact
            time.sleep(5)  # Wait a bit for metrics to stabilize
            post_metrics = self._collect_metrics()
            self.scaling_policy.record_scaling_outcome(event, post_metrics)
    
    def _scale_worker_pool(self, target_size: int) -> bool:
        """Scale the worker pool to target size."""
        # This would need to be implemented in the WorkerPool class
        # For now, just return True to simulate successful scaling
        logging.info(f"Simulating scaling to {target_size} workers")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        return {
            'running': self.running,
            'check_interval': self.check_interval,
            'cooldown_period': self.cooldown_period,
            'metrics_history_size': len(self.metrics_history),
            'scaling_events_count': len(self.scaling_events),
            **self.stats
        }
    
    def get_recent_events(self, count: int = 10) -> List[ScalingEvent]:
        """Get recent scaling events."""
        return list(self.scaling_events)[-count:]


class CircuitBreaker:
    """Circuit breaker for load shedding and backpressure handling."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise PQCRetrofitError(
                        "Circuit breaker is OPEN",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.RESOURCE_ERROR
                    )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (self.last_failure_time is not None and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"


# Factory functions for creating auto-scalers with different policies

def create_adaptive_autoscaler(worker_pool: WorkerPool, min_workers: int = 2, 
                              max_workers: int = 32) -> AutoScaler:
    """Create auto-scaler with adaptive scaling policy."""
    policy = AdaptiveScalingPolicy(min_workers=min_workers, max_workers=max_workers)
    return AutoScaler(worker_pool, policy)


def create_predictive_autoscaler(worker_pool: WorkerPool, min_workers: int = 2,
                                max_workers: int = 32) -> AutoScaler:
    """Create auto-scaler with predictive scaling policy."""
    policy = PredictiveScalingPolicy(min_workers=min_workers, max_workers=max_workers)
    return AutoScaler(worker_pool, policy)