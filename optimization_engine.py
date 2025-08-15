#!/usr/bin/env python3
"""
Performance Optimization Engine - Generation 3
Advanced optimization algorithms, memory management, and performance tuning
"""

import os
import time
import psutil
import gc
import sys
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json


class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios."""
    MEMORY_CONSTRAINED = "memory_constrained"
    CPU_INTENSIVE = "cpu_intensive"
    IO_BOUND = "io_bound"
    BALANCED = "balanced"
    REAL_TIME = "real_time"


@dataclass
class PerformanceProfile:
    """Performance profile for optimization decisions."""
    cpu_usage_percent: float
    memory_usage_mb: float
    io_wait_percent: float
    processing_rate: float
    error_rate: float
    response_time_p95: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationRule:
    """Rule for performance optimization."""
    name: str
    condition: Callable[[PerformanceProfile], bool]
    action: Callable[[], None]
    priority: int = 1
    cooldown_seconds: int = 30
    last_applied: float = 0


class MemoryManager:
    """Advanced memory management and optimization."""
    
    def __init__(self, target_memory_mb: int = 512):
        """Initialize memory manager."""
        self.target_memory_mb = target_memory_mb
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)
        self.allocation_stats = {
            "total_allocated": 0,
            "total_freed": 0,
            "current_usage": 0,
            "peak_usage": 0
        }
        self.gc_thresholds = (700, 10, 10)  # Optimized GC thresholds
        self.lock = threading.RLock()
        
        # Configure garbage collection
        gc.set_threshold(*self.gc_thresholds)
        
    def allocate_buffer(self, size: int, buffer_type: str = "default") -> bytearray:
        """Allocate memory buffer with pooling."""
        with self.lock:
            # Try to reuse from pool
            pool = self.memory_pools[buffer_type]
            for i, buffer in enumerate(pool):
                if len(buffer) >= size:
                    # Reuse existing buffer
                    reused = pool.pop(i)
                    self.allocation_stats["total_allocated"] += 1
                    return reused
            
            # Allocate new buffer
            buffer = bytearray(size)
            self.allocation_stats["total_allocated"] += 1
            self.allocation_stats["current_usage"] += size
            
            if self.allocation_stats["current_usage"] > self.allocation_stats["peak_usage"]:
                self.allocation_stats["peak_usage"] = self.allocation_stats["current_usage"]
            
            return buffer
    
    def free_buffer(self, buffer: bytearray, buffer_type: str = "default"):
        """Return buffer to pool for reuse."""
        with self.lock:
            if len(buffer) <= 1024 * 1024:  # Only pool buffers <= 1MB
                self.memory_pools[buffer_type].append(buffer)
            
            self.allocation_stats["total_freed"] += 1
            self.allocation_stats["current_usage"] -= len(buffer)
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return stats."""
        before_memory = self.get_memory_usage()
        
        # Clear buffer pools if memory pressure is high
        if before_memory > self.target_memory_mb:
            with self.lock:
                for pool_name in list(self.memory_pools.keys()):
                    if len(self.memory_pools[pool_name]) > 10:
                        # Keep only 10 most recent buffers
                        self.memory_pools[pool_name] = self.memory_pools[pool_name][-10:]
        
        # Force GC
        collected = [gc.collect(i) for i in range(3)]
        
        after_memory = self.get_memory_usage()
        
        return {
            "before_memory_mb": before_memory,
            "after_memory_mb": after_memory,
            "memory_freed_mb": before_memory - after_memory,
            "objects_collected": sum(collected),
            "gc_stats": collected
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            # Fallback method
            return sys.getsizeof(gc.get_objects()) / 1024 / 1024
    
    def optimize_memory_layout(self):
        """Optimize memory layout for better cache performance."""
        # Trigger garbage collection
        gc.collect()
        
        # Compact memory pools
        with self.lock:
            for pool_name, buffers in self.memory_pools.items():
                # Sort buffers by size for better locality
                buffers.sort(key=len)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.lock:
            pool_stats = {
                name: {"count": len(buffers), "total_size": sum(len(b) for b in buffers)}
                for name, buffers in self.memory_pools.items()
            }
            
            return {
                "current_usage_mb": self.get_memory_usage(),
                "target_memory_mb": self.target_memory_mb,
                "allocation_stats": self.allocation_stats,
                "pool_stats": pool_stats,
                "gc_thresholds": self.gc_thresholds,
                "gc_counts": gc.get_count()
            }


class CPUOptimizer:
    """CPU performance optimization and profiling."""
    
    def __init__(self):
        """Initialize CPU optimizer."""
        self.cpu_profiles: deque = deque(maxlen=100)
        self.optimization_history: List[Dict[str, Any]] = []
        self.cpu_affinity_set = False
        
    def profile_cpu_usage(self) -> Dict[str, float]:
        """Profile current CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            profile = {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "cpu_per_core": cpu_per_core,
                "load_1min": load_avg[0],
                "load_5min": load_avg[1],
                "load_15min": load_avg[2],
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True)
            }
            
            self.cpu_profiles.append(profile)
            return profile
            
        except Exception as e:
            logging.error(f"CPU profiling error: {e}")
            return {"error": str(e)}
    
    def optimize_cpu_affinity(self, process_count: int):
        """Optimize CPU affinity for better performance."""
        try:
            if self.cpu_affinity_set:
                return
            
            cpu_count = psutil.cpu_count()
            if cpu_count and cpu_count > 1:
                # Bind to specific CPUs for better cache locality
                current_process = psutil.Process()
                
                # Use first N CPUs for processing
                cpu_list = list(range(min(process_count, cpu_count)))
                current_process.cpu_affinity(cpu_list)
                
                self.cpu_affinity_set = True
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "optimization": "cpu_affinity",
                    "cpu_list": cpu_list,
                    "process_count": process_count
                })
                
                logging.info(f"Set CPU affinity to cores: {cpu_list}")
                
        except Exception as e:
            logging.error(f"CPU affinity optimization failed: {e}")
    
    def analyze_cpu_bottlenecks(self) -> Dict[str, Any]:
        """Analyze CPU usage patterns for bottlenecks."""
        if len(self.cpu_profiles) < 10:
            return {"error": "Insufficient data for analysis"}
        
        recent_profiles = list(self.cpu_profiles)[-10:]
        
        avg_cpu = sum(p["cpu_percent"] for p in recent_profiles) / len(recent_profiles)
        max_cpu = max(p["cpu_percent"] for p in recent_profiles)
        
        # Analyze per-core usage
        core_usage = defaultdict(list)
        for profile in recent_profiles:
            if "cpu_per_core" in profile:
                for i, usage in enumerate(profile["cpu_per_core"]):
                    core_usage[i].append(usage)
        
        core_averages = {
            core: sum(usages) / len(usages)
            for core, usages in core_usage.items()
        }
        
        # Detect bottlenecks
        bottlenecks = []
        if avg_cpu > 80:
            bottlenecks.append("High average CPU usage")
        
        if max_cpu > 95:
            bottlenecks.append("CPU spikes detected")
        
        # Check for uneven core usage
        if core_averages:
            min_core_usage = min(core_averages.values())
            max_core_usage = max(core_averages.values())
            if max_core_usage - min_core_usage > 30:
                bottlenecks.append("Uneven CPU core utilization")
        
        return {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "core_averages": core_averages,
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_cpu_recommendations(avg_cpu, bottlenecks)
        }
    
    def _generate_cpu_recommendations(self, avg_cpu: float, bottlenecks: List[str]) -> List[str]:
        """Generate CPU optimization recommendations."""
        recommendations = []
        
        if avg_cpu > 80:
            recommendations.append("Consider increasing worker pool size or reducing batch size")
        
        if "CPU spikes detected" in bottlenecks:
            recommendations.append("Implement CPU throttling or rate limiting")
        
        if "Uneven CPU core utilization" in bottlenecks:
            recommendations.append("Optimize task distribution across CPU cores")
        
        if not bottlenecks:
            recommendations.append("CPU performance is optimal")
        
        return recommendations


class IOOptimizer:
    """I/O performance optimization."""
    
    def __init__(self):
        """Initialize I/O optimizer."""
        self.io_stats: deque = deque(maxlen=100)
        self.file_cache: Dict[str, bytes] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def profile_io_usage(self) -> Dict[str, Any]:
        """Profile current I/O usage."""
        try:
            io_counters = psutil.disk_io_counters()
            
            if io_counters:
                profile = {
                    "timestamp": time.time(),
                    "read_bytes": io_counters.read_bytes,
                    "write_bytes": io_counters.write_bytes,
                    "read_count": io_counters.read_count,
                    "write_count": io_counters.write_count,
                    "read_time": io_counters.read_time,
                    "write_time": io_counters.write_time
                }
                
                self.io_stats.append(profile)
                return profile
            else:
                return {"error": "I/O counters not available"}
                
        except Exception as e:
            logging.error(f"I/O profiling error: {e}")
            return {"error": str(e)}
    
    def optimized_file_read(self, file_path: str, use_cache: bool = True) -> bytes:
        """Optimized file reading with caching."""
        if use_cache and file_path in self.file_cache:
            self.cache_hits += 1
            return self.file_cache[file_path]
        
        # Read file
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Cache if reasonable size (< 1MB)
        if use_cache and len(data) < 1024 * 1024:
            self.file_cache[file_path] = data
        
        self.cache_misses += 1
        return data
    
    def optimize_io_buffering(self, buffer_size: int = 64 * 1024):
        """Optimize I/O buffering parameters."""
        # This would typically configure system-level I/O parameters
        # For demo purposes, we'll track the optimization
        optimization = {
            "timestamp": time.time(),
            "optimization": "io_buffering",
            "buffer_size": buffer_size,
            "recommendation": f"Use {buffer_size} byte buffers for optimal I/O"
        }
        
        logging.info(f"I/O buffer size optimized to {buffer_size} bytes")
        return optimization
    
    def get_io_stats(self) -> Dict[str, Any]:
        """Get comprehensive I/O statistics."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / max(1, cache_total)) * 100
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            "cached_files": len(self.file_cache),
            "cache_size_bytes": sum(len(data) for data in self.file_cache.values()),
            "recent_io_samples": len(self.io_stats)
        }


class OptimizationEngine:
    """Main optimization engine coordinating all optimizations."""
    
    def __init__(self):
        """Initialize optimization engine."""
        self.memory_manager = MemoryManager()
        self.cpu_optimizer = CPUOptimizer()
        self.io_optimizer = IOOptimizer()
        
        self.performance_history: deque = deque(maxlen=100)
        self.optimization_rules: List[OptimizationRule] = []
        self.current_strategy = OptimizationStrategy.BALANCED
        
        self._setup_default_rules()
        self.logger = logging.getLogger('OptimizationEngine')
    
    def _setup_default_rules(self):
        """Setup default optimization rules."""
        # Memory pressure rule
        self.optimization_rules.append(OptimizationRule(
            name="memory_pressure",
            condition=lambda profile: profile.memory_usage_mb > 800,
            action=self._handle_memory_pressure,
            priority=1,
            cooldown_seconds=60
        ))
        
        # High CPU usage rule
        self.optimization_rules.append(OptimizationRule(
            name="high_cpu_usage",
            condition=lambda profile: profile.cpu_usage_percent > 90,
            action=self._handle_high_cpu_usage,
            priority=2,
            cooldown_seconds=30
        ))
        
        # Poor response time rule
        self.optimization_rules.append(OptimizationRule(
            name="poor_response_time",
            condition=lambda profile: profile.response_time_p95 > 5.0,
            action=self._handle_poor_response_time,
            priority=3,
            cooldown_seconds=45
        ))
    
    def analyze_performance(self) -> PerformanceProfile:
        """Analyze current system performance."""
        cpu_profile = self.cpu_optimizer.profile_cpu_usage()
        io_profile = self.io_optimizer.profile_io_usage()
        memory_usage = self.memory_manager.get_memory_usage()
        
        # Mock some metrics for demo
        profile = PerformanceProfile(
            cpu_usage_percent=cpu_profile.get("cpu_percent", 0),
            memory_usage_mb=memory_usage,
            io_wait_percent=5.0,  # Mock value
            processing_rate=50.0,  # Mock value
            error_rate=0.02,  # Mock value
            response_time_p95=2.5  # Mock value
        )
        
        self.performance_history.append(profile)
        return profile
    
    def apply_optimizations(self, profile: PerformanceProfile):
        """Apply optimization rules based on performance profile."""
        current_time = time.time()
        
        for rule in sorted(self.optimization_rules, key=lambda r: r.priority):
            # Check cooldown
            if current_time - rule.last_applied < rule.cooldown_seconds:
                continue
            
            # Check condition
            if rule.condition(profile):
                try:
                    rule.action()
                    rule.last_applied = current_time
                    self.logger.info(f"Applied optimization rule: {rule.name}")
                except Exception as e:
                    self.logger.error(f"Failed to apply rule {rule.name}: {e}")
    
    def _handle_memory_pressure(self):
        """Handle memory pressure situation."""
        gc_stats = self.memory_manager.force_garbage_collection()
        self.logger.info(f"Memory pressure handled: freed {gc_stats['memory_freed_mb']:.1f}MB")
    
    def _handle_high_cpu_usage(self):
        """Handle high CPU usage situation."""
        # This could implement CPU throttling or task prioritization
        self.logger.info("High CPU usage detected - implementing throttling")
        time.sleep(0.1)  # Simple throttling
    
    def _handle_poor_response_time(self):
        """Handle poor response time situation."""
        # This could adjust batch sizes or worker counts
        self.logger.info("Poor response time detected - optimizing processing")
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy):
        """Set optimization strategy based on workload characteristics."""
        self.current_strategy = strategy
        
        # Adjust parameters based on strategy
        if strategy == OptimizationStrategy.MEMORY_CONSTRAINED:
            self.memory_manager.target_memory_mb = 256
            self.memory_manager.gc_thresholds = (500, 5, 5)
        elif strategy == OptimizationStrategy.CPU_INTENSIVE:
            self.cpu_optimizer.optimize_cpu_affinity(psutil.cpu_count())
        elif strategy == OptimizationStrategy.IO_BOUND:
            self.io_optimizer.optimize_io_buffering(128 * 1024)
        
        self.logger.info(f"Optimization strategy set to: {strategy.value}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_profiles = list(self.performance_history)[-10:]
        
        avg_cpu = sum(p.cpu_usage_percent for p in recent_profiles) / len(recent_profiles)
        avg_memory = sum(p.memory_usage_mb for p in recent_profiles) / len(recent_profiles)
        avg_response_time = sum(p.response_time_p95 for p in recent_profiles) / len(recent_profiles)
        
        return {
            "current_strategy": self.current_strategy.value,
            "performance_summary": {
                "avg_cpu_percent": avg_cpu,
                "avg_memory_mb": avg_memory,
                "avg_response_time": avg_response_time
            },
            "memory_stats": self.memory_manager.get_memory_stats(),
            "cpu_analysis": self.cpu_optimizer.analyze_cpu_bottlenecks(),
            "io_stats": self.io_optimizer.get_io_stats(),
            "optimization_rules_applied": sum(
                1 for rule in self.optimization_rules if rule.last_applied > 0
            ),
            "recommendations": self._generate_optimization_recommendations(
                avg_cpu, avg_memory, avg_response_time
            )
        }
    
    def _generate_optimization_recommendations(self, avg_cpu: float, 
                                             avg_memory: float, avg_response_time: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if avg_cpu > 80:
            recommendations.append("Consider horizontal scaling or CPU optimization")
        
        if avg_memory > 600:
            recommendations.append("Implement memory optimization or increase memory")
        
        if avg_response_time > 3.0:
            recommendations.append("Optimize algorithms or increase parallelism")
        
        if avg_cpu < 30 and avg_memory < 200:
            recommendations.append("System is under-utilized - consider increasing workload")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations


def main():
    """Demo of optimization engine."""
    print("Performance Optimization Engine - Demo")
    print("=" * 50)
    
    # Initialize optimization engine
    engine = OptimizationEngine()
    
    # Set optimization strategy
    engine.set_optimization_strategy(OptimizationStrategy.BALANCED)
    
    # Simulate performance monitoring and optimization
    print("Simulating performance monitoring...")
    
    for i in range(10):
        # Analyze current performance
        profile = engine.analyze_performance()
        
        # Apply optimizations
        engine.apply_optimizations(profile)
        
        print(f"  Iteration {i+1}: CPU={profile.cpu_usage_percent:.1f}%, "
              f"Memory={profile.memory_usage_mb:.1f}MB, "
              f"Response={profile.response_time_p95:.2f}s")
        
        time.sleep(0.2)
    
    # Test memory management
    print("\nTesting memory management...")
    
    # Allocate some buffers
    buffers = []
    for i in range(10):
        buffer = engine.memory_manager.allocate_buffer(1024 * (i + 1), f"test_type_{i%3}")
        buffers.append(buffer)
    
    print(f"Allocated {len(buffers)} buffers")
    
    # Free half the buffers
    for buffer in buffers[:5]:
        engine.memory_manager.free_buffer(buffer, "test_type_0")
    
    print("Freed 5 buffers")
    
    # Test file I/O optimization
    print("\nTesting I/O optimization...")
    
    # Create test file
    test_file = "test_optimization.txt"
    with open(test_file, "w") as f:
        f.write("Test data for optimization" * 100)
    
    # Read file multiple times (should hit cache)
    for i in range(5):
        data = engine.io_optimizer.optimized_file_read(test_file)
        print(f"  Read {len(data)} bytes (attempt {i+1})")
    
    # Clean up
    os.remove(test_file)
    
    # Generate optimization report
    print("\nOptimization Report:")
    report = engine.get_optimization_report()
    print(json.dumps(report, indent=2, default=str))
    
    print("\nOptimization engine demo complete!")


if __name__ == '__main__':
    # Install psutil if not available
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not available, using mock values")
        # Create mock psutil module
        class MockPsutil:
            @staticmethod
            def cpu_percent(interval=None, percpu=False):
                return [25.0, 30.0, 20.0, 35.0] if percpu else 27.5
            @staticmethod
            def cpu_count(logical=True):
                return 8 if logical else 4
            @staticmethod
            def disk_io_counters():
                return None
            class Process:
                def memory_info(self):
                    class MemInfo:
                        rss = 128 * 1024 * 1024  # 128MB
                    return MemInfo()
                def cpu_affinity(self, cpus=None):
                    return [0, 1, 2, 3] if cpus is None else None
        
        sys.modules['psutil'] = MockPsutil()
        psutil = MockPsutil()
    
    main()