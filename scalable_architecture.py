#!/usr/bin/env python3
"""
Scalable Architecture System - Generation 3
High-performance, concurrent, and optimized PQC scanner with auto-scaling capabilities
"""

import os
import asyncio
import concurrent.futures
import multiprocessing
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import lru_cache, wraps
import threading
import queue
import logging


@dataclass 
class ProcessingTask:
    """Task for parallel processing."""
    task_id: str
    firmware_path: str
    priority: int = 1
    metadata: Dict[str, Any] = None


@dataclass
class ProcessingResult:
    """Result from parallel processing."""
    task_id: str
    firmware_path: str
    vulnerabilities: List[Dict[str, Any]]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class FirmwareCache:
    """High-performance firmware analysis cache."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with LRU eviction."""
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def _generate_cache_key(self, firmware_path: str) -> str:
        """Generate cache key from firmware path and modification time."""
        try:
            stat_info = os.stat(firmware_path)
            content_hash = hashlib.sha256(f"{firmware_path}_{stat_info.st_size}_{stat_info.st_mtime}".encode()).hexdigest()
            return content_hash[:16]
        except OSError:
            return hashlib.sha256(firmware_path.encode()).hexdigest()[:16]
    
    def get(self, firmware_path: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result."""
        with self.lock:
            cache_key = self._generate_cache_key(firmware_path)
            
            if cache_key in self.cache:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key].copy()
            
            return None
    
    def put(self, firmware_path: str, result: Dict[str, Any]):
        """Store analysis result in cache."""
        with self.lock:
            cache_key = self._generate_cache_key(firmware_path)
            
            # Evict LRU items if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[cache_key] = result.copy()
            self.access_times[cache_key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        # Sort by access time and remove oldest 20%
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        evict_count = max(1, len(sorted_keys) // 5)
        
        for key, _ in sorted_keys[:evict_count]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "utilization_percent": (len(self.cache) / self.max_size) * 100,
                "oldest_entry_age": time.time() - min(self.access_times.values()) if self.access_times else 0
            }


class ParallelProcessor:
    """High-performance parallel firmware processing."""
    
    def __init__(self, max_workers: Optional[int] = None, use_asyncio: bool = True):
        """Initialize parallel processor."""
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_asyncio = use_asyncio
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.processing_stats = {
            "tasks_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "errors": 0
        }
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.cache = FirmwareCache()
        self.logger = logging.getLogger('ParallelProcessor')
    
    async def process_firmware_batch(self, firmware_paths: List[str], 
                                   batch_size: int = 10) -> AsyncGenerator[ProcessingResult, None]:
        """Process firmware files in parallel batches."""
        tasks = []
        
        # Create processing tasks
        for i, path in enumerate(firmware_paths):
            task = ProcessingTask(
                task_id=f"task_{i:04d}",
                firmware_path=path,
                priority=1,
                metadata={"batch_index": i}
            )
            tasks.append(task)
        
        # Process in batches
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            
            if self.use_asyncio:
                batch_results = await self._process_batch_async(batch)
            else:
                batch_results = await self._process_batch_threaded(batch)
            
            for result in batch_results:
                yield result
    
    async def _process_batch_async(self, batch: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process batch using asyncio."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_task(task: ProcessingTask) -> ProcessingResult:
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor, self._process_single_firmware, task
                )
        
        return await asyncio.gather(
            *[process_single_task(task) for task in batch]
        )
    
    async def _process_batch_threaded(self, batch: List[ProcessingTask]) -> List[ProcessingResult]:
        """Process batch using thread pool."""
        loop = asyncio.get_event_loop()
        
        futures = [
            loop.run_in_executor(self.executor, self._process_single_firmware, task)
            for task in batch
        ]
        
        return await asyncio.gather(*futures)
    
    def _process_single_firmware(self, task: ProcessingTask) -> ProcessingResult:
        """Process a single firmware file."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.cache.get(task.firmware_path)
            if cached_result:
                self.logger.debug(f"Cache hit for {task.firmware_path}")
                return ProcessingResult(
                    task_id=task.task_id,
                    firmware_path=task.firmware_path,
                    vulnerabilities=cached_result["vulnerabilities"],
                    processing_time=cached_result["processing_time"],
                    success=True
                )
            
            # Perform actual analysis
            vulnerabilities = self._analyze_firmware_optimized(task.firmware_path)
            processing_time = time.time() - start_time
            
            # Cache result
            result_data = {
                "vulnerabilities": vulnerabilities,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
            self.cache.put(task.firmware_path, result_data)
            
            # Update stats
            self.processing_stats["tasks_processed"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            self.processing_stats["average_processing_time"] = (
                self.processing_stats["total_processing_time"] / 
                self.processing_stats["tasks_processed"]
            )
            
            return ProcessingResult(
                task_id=task.task_id,
                firmware_path=task.firmware_path,
                vulnerabilities=vulnerabilities,
                processing_time=processing_time,
                success=True
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats["errors"] += 1
            
            self.logger.error(f"Error processing {task.firmware_path}: {e}")
            
            return ProcessingResult(
                task_id=task.task_id,
                firmware_path=task.firmware_path,
                vulnerabilities=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    @lru_cache(maxsize=256)
    def _analyze_firmware_optimized(self, firmware_path: str) -> List[Dict[str, Any]]:
        """Optimized firmware analysis with caching."""
        # Simplified analysis for demo - in production would use actual scanner
        time.sleep(0.01)  # Simulate analysis work
        
        # Mock vulnerability detection
        vulnerabilities = []
        
        # Simulate finding vulnerabilities based on file characteristics
        file_size = os.path.getsize(firmware_path)
        file_hash = hashlib.md5(firmware_path.encode()).hexdigest()[:8]
        
        if file_size > 1000:  # Larger files more likely to have vulnerabilities
            vulnerabilities.append({
                "algorithm": "RSA-2048",
                "risk_level": "high",
                "location": int(file_hash[:4], 16) % file_size,
                "confidence": 0.85
            })
        
        if "vulnerable" in firmware_path:
            vulnerabilities.append({
                "algorithm": "ECDSA-P256", 
                "risk_level": "high",
                "location": int(file_hash[4:8], 16) % file_size,
                "confidence": 0.92
            })
        
        return vulnerabilities
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        stats["cache_stats"] = self.cache.get_stats()
        stats["max_workers"] = self.max_workers
        stats["success_rate"] = (
            (stats["tasks_processed"] - stats["errors"]) / max(1, stats["tasks_processed"])
        ) * 100
        return stats


class AutoScaler:
    """Automatic scaling based on workload and performance metrics."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 32):
        """Initialize auto-scaler."""
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.metrics_history: List[Dict[str, float]] = []
        self.scaling_cooldown = 30  # seconds
        self.last_scale_time = 0
        self.logger = logging.getLogger('AutoScaler')
    
    def analyze_workload(self, queue_size: int, processing_rate: float, 
                        error_rate: float, avg_response_time: float) -> int:
        """Analyze workload and determine optimal worker count."""
        current_time = time.time()
        
        # Collect metrics
        metrics = {
            "timestamp": current_time,
            "queue_size": queue_size,
            "processing_rate": processing_rate,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "worker_count": self.current_workers
        }
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics (last 10 minutes)
        cutoff_time = current_time - 600
        self.metrics_history = [
            m for m in self.metrics_history if m["timestamp"] >= cutoff_time
        ]
        
        # Check if we're in cooldown period
        if current_time - self.last_scale_time < self.scaling_cooldown:
            return self.current_workers
        
        # Scaling decisions
        target_workers = self.current_workers
        
        # Scale up conditions
        if queue_size > self.current_workers * 2:  # Queue backing up
            target_workers = min(self.max_workers, self.current_workers + 2)
            self.logger.info(f"Scaling up due to queue backlog: {queue_size}")
        
        elif avg_response_time > 5.0 and processing_rate > 0:  # Slow processing
            target_workers = min(self.max_workers, self.current_workers + 1)
            self.logger.info(f"Scaling up due to slow response time: {avg_response_time:.2f}s")
        
        # Scale down conditions
        elif queue_size == 0 and self.current_workers > self.min_workers:
            if len(self.metrics_history) >= 5:
                recent_queue_sizes = [m["queue_size"] for m in self.metrics_history[-5:]]
                if max(recent_queue_sizes) == 0:  # No work for a while
                    target_workers = max(self.min_workers, self.current_workers - 1)
                    self.logger.info("Scaling down due to low utilization")
        
        elif error_rate > 0.1:  # High error rate might indicate overload
            target_workers = max(self.min_workers, self.current_workers - 1)
            self.logger.info(f"Scaling down due to high error rate: {error_rate:.1%}")
        
        # Apply scaling
        if target_workers != self.current_workers:
            self.current_workers = target_workers
            self.last_scale_time = current_time
        
        return self.current_workers
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations and insights."""
        if len(self.metrics_history) < 3:
            return {"recommendation": "Insufficient data for recommendations"}
        
        recent_metrics = self.metrics_history[-10:]
        
        avg_queue_size = sum(m["queue_size"] for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m["avg_response_time"] for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m["error_rate"] for m in recent_metrics) / len(recent_metrics)
        
        recommendations = []
        
        if avg_queue_size > 10:
            recommendations.append("Consider increasing max_workers limit")
        
        if avg_response_time > 3.0:
            recommendations.append("Response time is high - consider performance optimization")
        
        if avg_error_rate > 0.05:
            recommendations.append("Error rate is elevated - investigate error causes")
        
        if not recommendations:
            recommendations.append("System is performing well")
        
        return {
            "current_workers": self.current_workers,
            "worker_range": f"{self.min_workers}-{self.max_workers}",
            "avg_queue_size": avg_queue_size,
            "avg_response_time": avg_response_time,
            "avg_error_rate": avg_error_rate,
            "recommendations": recommendations
        }


class LoadBalancer:
    """Intelligent load balancing for distributed processing."""
    
    def __init__(self, worker_endpoints: List[str] = None):
        """Initialize load balancer."""
        self.worker_endpoints = worker_endpoints or ["worker_1", "worker_2", "worker_3"]
        self.worker_loads: Dict[str, float] = {ep: 0.0 for ep in self.worker_endpoints}
        self.worker_response_times: Dict[str, List[float]] = {ep: [] for ep in self.worker_endpoints}
        self.request_count = 0
        self.lock = threading.RLock()
    
    def select_worker(self, task_size: int = 1) -> str:
        """Select optimal worker using weighted round-robin."""
        with self.lock:
            # Calculate worker scores (lower is better)
            worker_scores = {}
            
            for endpoint in self.worker_endpoints:
                load = self.worker_loads[endpoint]
                recent_response_times = self.worker_response_times[endpoint][-10:]
                avg_response_time = sum(recent_response_times) / max(1, len(recent_response_times))
                
                # Score based on load and response time
                score = load * 0.7 + avg_response_time * 0.3
                worker_scores[endpoint] = score
            
            # Select worker with lowest score
            selected_worker = min(worker_scores.items(), key=lambda x: x[1])[0]
            
            # Update load
            self.worker_loads[selected_worker] += task_size
            self.request_count += 1
            
            return selected_worker
    
    def report_task_completion(self, worker_endpoint: str, response_time: float, task_size: int = 1):
        """Report task completion and update worker metrics."""
        with self.lock:
            # Update load
            self.worker_loads[worker_endpoint] = max(0, self.worker_loads[worker_endpoint] - task_size)
            
            # Record response time
            self.worker_response_times[worker_endpoint].append(response_time)
            
            # Keep only recent response times
            if len(self.worker_response_times[worker_endpoint]) > 100:
                self.worker_response_times[worker_endpoint] = self.worker_response_times[worker_endpoint][-100:]
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        with self.lock:
            stats = {
                "total_requests": self.request_count,
                "worker_count": len(self.worker_endpoints),
                "worker_details": {}
            }
            
            for endpoint in self.worker_endpoints:
                recent_times = self.worker_response_times[endpoint][-10:]
                stats["worker_details"][endpoint] = {
                    "current_load": self.worker_loads[endpoint],
                    "avg_response_time": sum(recent_times) / max(1, len(recent_times)),
                    "request_count": len(self.worker_response_times[endpoint])
                }
            
            return stats


def performance_cache(maxsize: int = 128, ttl: int = 300):
    """Performance-optimized cache decorator with TTL."""
    def decorator(func):
        cache = {}
        cache_times = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Check if cached and not expired
            if key in cache and current_time - cache_times[key] < ttl:
                return cache[key]
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[key] = result
            cache_times[key] = current_time
            
            # Evict old entries if cache is full
            if len(cache) > maxsize:
                # Remove oldest 20% of entries
                old_keys = sorted(cache_times.items(), key=lambda x: x[1])[:maxsize//5]
                for old_key, _ in old_keys:
                    cache.pop(old_key, None)
                    cache_times.pop(old_key, None)
            
            return result
        
        wrapper.cache_info = lambda: {
            "cache_size": len(cache),
            "max_size": maxsize,
            "ttl": ttl
        }
        
        return wrapper
    return decorator


async def main():
    """Demo of scalable architecture system."""
    print("Scalable Architecture System - Demo")
    print("=" * 50)
    
    # Initialize components
    processor = ParallelProcessor(max_workers=8)
    auto_scaler = AutoScaler(min_workers=2, max_workers=16) 
    load_balancer = LoadBalancer()
    
    # Create test firmware files
    test_files = []
    for i in range(20):
        filename = f"test_data/firmware_{i:02d}.bin"
        os.makedirs("test_data", exist_ok=True)
        
        # Create files with varying sizes
        content = f"Test firmware {i} " * (100 + i * 50)
        if i % 3 == 0:
            content += " vulnerable_pattern"
        
        with open(filename, 'w') as f:
            f.write(content)
        
        test_files.append(filename)
    
    print(f"Created {len(test_files)} test firmware files")
    
    # Simulate auto-scaling analysis
    print("\nSimulating workload analysis...")
    for i in range(5):
        queue_size = 15 - i * 3  # Decreasing queue
        processing_rate = 2.5 + i * 0.5  # Increasing rate
        error_rate = 0.02 + i * 0.01  # Increasing errors
        avg_response_time = 3.0 - i * 0.5  # Decreasing response time
        
        optimal_workers = auto_scaler.analyze_workload(
            queue_size, processing_rate, error_rate, avg_response_time
        )
        print(f"  Iteration {i+1}: Queue={queue_size}, Workers={optimal_workers}")
        
        time.sleep(0.1)
    
    # Process firmware batch
    print("\nProcessing firmware batch with parallel processing...")
    start_time = time.time()
    
    results = []
    async for result in processor.process_firmware_batch(test_files[:10], batch_size=4):
        results.append(result)
        
        # Simulate load balancing
        worker = load_balancer.select_worker()
        load_balancer.report_task_completion(worker, result.processing_time)
    
    total_time = time.time() - start_time
    
    # Show results
    print(f"\nProcessing Results:")
    print(f"  Total files: {len(results)}")
    print(f"  Successful: {sum(1 for r in results if r.success)}")
    print(f"  Failed: {sum(1 for r in results if not r.success)}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Average time per file: {total_time/len(results):.3f}s")
    
    # Show vulnerabilities found
    total_vulns = sum(len(r.vulnerabilities) for r in results)
    print(f"  Total vulnerabilities: {total_vulns}")
    
    # Show performance stats
    print("\nProcessing Statistics:")
    stats = processor.get_processing_stats()
    print(json.dumps(stats, indent=2))
    
    # Show auto-scaling recommendations
    print("\nAuto-scaling Recommendations:")
    scaling_recs = auto_scaler.get_scaling_recommendations()
    print(json.dumps(scaling_recs, indent=2))
    
    # Show load balancing stats
    print("\nLoad Balancing Statistics:")
    lb_stats = load_balancer.get_load_balancing_stats()
    print(json.dumps(lb_stats, indent=2))
    
    # Demo performance cache
    @performance_cache(maxsize=64, ttl=60)
    def expensive_computation(n: int) -> int:
        time.sleep(0.01)  # Simulate expensive work
        return n * n
    
    print("\nTesting performance cache:")
    cache_start = time.time()
    for i in range(10):
        result = expensive_computation(i % 5)  # Will hit cache
    cache_time = time.time() - cache_start
    
    print(f"  Cache test time: {cache_time:.3f}s")
    print(f"  Cache info: {expensive_computation.cache_info()}")
    
    print("\nScalable architecture demo complete!")


if __name__ == '__main__':
    asyncio.run(main())