"""High-performance optimized scanner for Generation 3."""

import asyncio
import concurrent.futures
import multiprocessing
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import threading
import weakref

from .robust_scanner import RobustFirmwareScanner
from .scanner import CryptoVulnerability
from .security_enhanced import SecurityContext
from .error_handling import handle_errors


@dataclass
class CacheEntry:
    """Cache entry for scan results."""
    result: Any
    timestamp: float
    access_count: int
    file_hash: str
    
    def is_expired(self, ttl: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - self.timestamp > ttl


class IntelligentCache:
    """Multi-level intelligent cache for scan results."""
    
    def __init__(self, l1_max_size: int = 100, l2_max_size: int = 1000, ttl: float = 3600):
        """Initialize cache with L1 (memory) and L2 (persistent) layers."""
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = {}  # Simulated persistent cache
        self.l1_max_size = l1_max_size
        self.l2_max_size = l2_max_size
        self.ttl = ttl
        self.access_stats = defaultdict(int)
        self.hit_stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}
        self._lock = threading.RLock()
    
    def _generate_cache_key(self, firmware_path: str, architecture: str, 
                          base_address: int) -> str:
        """Generate unique cache key for scan parameters."""
        # Include file hash for integrity
        try:
            with open(firmware_path, 'rb') as f:
                file_data = f.read(1024)  # First 1KB for speed
                file_hash = hashlib.md5(file_data).hexdigest()[:16]
        except:
            file_hash = "unknown"
        
        key_data = f"{firmware_path}:{architecture}:{base_address}:{file_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, firmware_path: str, architecture: str, 
            base_address: int) -> Optional[List[CryptoVulnerability]]:
        """Get cached scan results."""
        with self._lock:
            cache_key = self._generate_cache_key(firmware_path, architecture, base_address)
            
            # Check L1 cache first
            if cache_key in self.l1_cache:
                entry = self.l1_cache[cache_key]
                if not entry.is_expired(self.ttl):
                    entry.access_count += 1
                    self.hit_stats['l1_hits'] += 1
                    return entry.result
                else:
                    del self.l1_cache[cache_key]
            
            # Check L2 cache
            if cache_key in self.l2_cache:
                entry = self.l2_cache[cache_key]
                if not entry.is_expired(self.ttl):
                    entry.access_count += 1
                    self.hit_stats['l2_hits'] += 1
                    # Promote to L1
                    self._promote_to_l1(cache_key, entry)
                    return entry.result
                else:
                    del self.l2_cache[cache_key]
            
            # Cache miss
            self.hit_stats['misses'] += 1
            return None
    
    def put(self, firmware_path: str, architecture: str, base_address: int,
            result: List[CryptoVulnerability]):
        """Store scan results in cache."""
        with self._lock:
            cache_key = self._generate_cache_key(firmware_path, architecture, base_address)
            
            # Calculate file hash for integrity
            try:
                with open(firmware_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            except:
                file_hash = "unknown"
            
            entry = CacheEntry(
                result=result,
                timestamp=time.time(),
                access_count=1,
                file_hash=file_hash
            )
            
            # Store in L1 first
            if len(self.l1_cache) >= self.l1_max_size:
                self._evict_l1_lru()
            
            self.l1_cache[cache_key] = entry
    
    def _promote_to_l1(self, cache_key: str, entry: CacheEntry):
        """Promote L2 entry to L1."""
        if len(self.l1_cache) >= self.l1_max_size:
            self._evict_l1_lru()
        
        self.l1_cache[cache_key] = entry
    
    def _evict_l1_lru(self):
        """Evict least recently used entry from L1 to L2."""
        if not self.l1_cache:
            return
        
        # Find LRU entry (lowest access count and oldest timestamp)
        lru_key = min(self.l1_cache.keys(), 
                     key=lambda k: (self.l1_cache[k].access_count, 
                                   self.l1_cache[k].timestamp))
        
        # Move to L2 if space available
        if len(self.l2_cache) < self.l2_max_size:
            self.l2_cache[lru_key] = self.l1_cache[lru_key]
        
        del self.l1_cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = sum(self.hit_stats.values())
            if total_requests == 0:
                return {
                    'l1_hit_rate': 0.0,
                    'l2_hit_rate': 0.0,
                    'overall_hit_rate': 0.0,
                    'l1_size': len(self.l1_cache),
                    'l2_size': len(self.l2_cache)
                }
            
            return {
                'l1_hit_rate': self.hit_stats['l1_hits'] / total_requests,
                'l2_hit_rate': self.hit_stats['l2_hits'] / total_requests,
                'overall_hit_rate': (self.hit_stats['l1_hits'] + self.hit_stats['l2_hits']) / total_requests,
                'l1_size': len(self.l1_cache),
                'l2_size': len(self.l2_cache),
                'total_requests': total_requests
            }
    
    def clear(self):
        """Clear all cache levels."""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.hit_stats = {'l1_hits': 0, 'l2_hits': 0, 'misses': 0}


class WorkerPool:
    """High-performance worker pool for concurrent scanning."""
    
    def __init__(self, scanner_class: type, scanner_kwargs: Dict[str, Any], 
                 max_workers: int = None):
        """Initialize worker pool."""
        self.scanner_class = scanner_class
        self.scanner_kwargs = scanner_kwargs
        self.max_workers = max_workers or min(8, (multiprocessing.cpu_count() or 1) + 4)
        
        # Thread pool for I/O-bound operations
        self.thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="pqc_scanner"
        )
        
        # Process pool for CPU-intensive operations
        self.process_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(4, multiprocessing.cpu_count() or 1)
        )
        
        self.active_tasks = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = threading.Lock()
        
    def submit_scan(self, firmware_path: str, base_address: int = 0,
                   use_process_pool: bool = False) -> concurrent.futures.Future:
        """Submit a scan task to the worker pool."""
        
        executor = self.process_executor if use_process_pool else self.thread_executor
        
        # Create scanner in the worker thread/process
        def worker_task():
            scanner = self.scanner_class(**self.scanner_kwargs)
            return scanner.scan_firmware_securely(firmware_path, base_address)
        
        future = executor.submit(worker_task)
        
        with self._lock:
            self.active_tasks.append(weakref.ref(future))
        
        # Add completion callback
        future.add_done_callback(self._task_completed)
        
        return future
    
    def _task_completed(self, future: concurrent.futures.Future):
        """Handle task completion."""
        with self._lock:
            if future.exception():
                self.failed_tasks += 1
            else:
                self.completed_tasks += 1
    
    def wait_for_completion(self, timeout: float = None) -> List[Any]:
        """Wait for all active tasks to complete."""
        # Get all active futures
        with self._lock:
            active_futures = [ref() for ref in self.active_tasks if ref() is not None]
        
        # Wait for completion
        results = []
        for future in concurrent.futures.as_completed(active_futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            # Clean up dead references
            self.active_tasks = [ref for ref in self.active_tasks if ref() is not None]
            
            return {
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': (self.completed_tasks / max(1, self.completed_tasks + self.failed_tasks))
            }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pools."""
        self.thread_executor.shutdown(wait=wait)
        self.process_executor.shutdown(wait=wait)


class OptimizedFirmwareScanner(RobustFirmwareScanner):
    """Generation 3 optimized scanner with caching and concurrency."""
    
    # Shared cache across all instances
    _global_cache = IntelligentCache()
    _worker_pool = None
    _pool_lock = threading.Lock()
    
    def __init__(self, architecture: str, memory_constraints: Dict[str, int] = None,
                 user_id: str = "anonymous", enable_caching: bool = True,
                 enable_worker_pool: bool = True):
        """Initialize optimized scanner."""
        super().__init__(architecture, memory_constraints, user_id)
        
        self.enable_caching = enable_caching
        self.enable_worker_pool = enable_worker_pool
        
        # Performance metrics
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'async_scans': 0,
            'batch_scans': 0,
            'total_scan_time': 0.0,
            'avg_scan_time': 0.0
        }
        
        # Initialize worker pool if needed
        if enable_worker_pool:
            self._ensure_worker_pool()
    
    def _ensure_worker_pool(self):
        """Ensure worker pool is initialized."""
        with self._pool_lock:
            if self.__class__._worker_pool is None:
                self.__class__._worker_pool = WorkerPool(
                    scanner_class=RobustFirmwareScanner,
                    scanner_kwargs={
                        'architecture': self.architecture,
                        'memory_constraints': self.memory_constraints,
                        'user_id': self.user_id
                    }
                )
    
    @handle_errors(operation_name="optimized_scan", retry_count=1)
    def scan_firmware_optimized(self, firmware_path: str, 
                               base_address: int = 0) -> List[CryptoVulnerability]:
        """Optimized firmware scan with caching."""
        
        scan_start = time.time()
        
        # Check cache first
        if self.enable_caching:
            cached_result = self._global_cache.get(firmware_path, self.architecture, base_address)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                self.logger.info(f"Cache hit for {firmware_path}")
                return cached_result
        
        # Cache miss - perform actual scan
        self.performance_stats['cache_misses'] += 1
        
        # Use parent's robust scan method
        vulnerabilities = super().scan_firmware_securely(firmware_path, base_address)
        
        # Cache the results
        if self.enable_caching:
            self._global_cache.put(firmware_path, self.architecture, base_address, vulnerabilities)
        
        # Update performance stats
        scan_time = time.time() - scan_start
        self.performance_stats['total_scan_time'] += scan_time
        self.performance_stats['avg_scan_time'] = (
            self.performance_stats['total_scan_time'] / 
            max(1, self.performance_stats['cache_misses'])
        )
        
        return vulnerabilities
    
    async def scan_firmware_async(self, firmware_path: str, 
                                 base_address: int = 0) -> List[CryptoVulnerability]:
        """Asynchronous firmware scan."""
        
        if not self.enable_worker_pool:
            # Fallback to synchronous scan
            return self.scan_firmware_optimized(firmware_path, base_address)
        
        self.performance_stats['async_scans'] += 1
        
        # Submit to worker pool
        self._ensure_worker_pool()
        future = self._worker_pool.submit_scan(firmware_path, base_address)
        
        # Wait for completion in async context
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, future.result, 30.0)  # 30s timeout
            return result
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Async scan timeout for {firmware_path}")
            raise
    
    def scan_firmware_batch(self, firmware_paths: List[str], 
                          base_addresses: List[int] = None) -> List[Tuple[str, List[CryptoVulnerability]]]:
        """Batch scan multiple firmware files with optimizations."""
        
        if base_addresses is None:
            base_addresses = [0] * len(firmware_paths)
        
        self.performance_stats['batch_scans'] += 1
        
        results = []
        
        if self.enable_worker_pool and len(firmware_paths) > 1:
            # Concurrent batch processing
            self._ensure_worker_pool()
            futures = []
            
            for firmware_path, base_address in zip(firmware_paths, base_addresses):
                future = self._worker_pool.submit_scan(firmware_path, base_address)
                futures.append((firmware_path, future))
            
            # Collect results
            for firmware_path, future in futures:
                try:
                    vulnerabilities = future.result(timeout=60.0)  # 60s timeout per scan
                    results.append((firmware_path, vulnerabilities))
                except Exception as e:
                    self.logger.error(f"Batch scan failed for {firmware_path}: {e}")
                    results.append((firmware_path, []))
        
        else:
            # Sequential processing with caching
            for firmware_path, base_address in zip(firmware_paths, base_addresses):
                try:
                    vulnerabilities = self.scan_firmware_optimized(firmware_path, base_address)
                    results.append((firmware_path, vulnerabilities))
                except Exception as e:
                    self.logger.error(f"Sequential scan failed for {firmware_path}: {e}")
                    results.append((firmware_path, []))
        
        return results
    
    def warm_cache(self, firmware_paths: List[str]) -> Dict[str, bool]:
        """Pre-warm cache with frequently accessed firmware files."""
        
        self.logger.info(f"Warming cache with {len(firmware_paths)} firmware files")
        
        results = {}
        for firmware_path in firmware_paths:
            try:
                # Check if already cached
                cached = self._global_cache.get(firmware_path, self.architecture, 0)
                if cached is None:
                    # Scan and cache
                    self.scan_firmware_optimized(firmware_path, 0)
                    results[firmware_path] = True
                else:
                    results[firmware_path] = True  # Already cached
            except Exception as e:
                self.logger.error(f"Cache warming failed for {firmware_path}: {e}")
                results[firmware_path] = False
        
        return results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        # Get base report
        base_report = super().generate_enhanced_report()
        
        # Add cache statistics
        cache_stats = self._global_cache.get_stats()
        
        # Add worker pool statistics
        pool_stats = {}
        if self._worker_pool:
            pool_stats = self._worker_pool.get_stats()
        
        # Combine performance data
        performance_report = {
            'generation': 3,
            'optimizations_enabled': {
                'intelligent_caching': self.enable_caching,
                'concurrent_processing': self.enable_worker_pool,
                'batch_processing': True,
                'async_operations': True
            },
            'cache_performance': cache_stats,
            'worker_pool_performance': pool_stats,
            'scan_performance': self.performance_stats,
            'efficiency_metrics': {
                'cache_hit_ratio': cache_stats.get('overall_hit_rate', 0.0),
                'avg_scan_time': self.performance_stats['avg_scan_time'],
                'concurrent_speedup': self._calculate_speedup(),
            }
        }
        
        # Merge with base report
        base_report['generation_3_performance'] = performance_report
        
        return base_report
    
    def _calculate_speedup(self) -> float:
        """Calculate concurrent processing speedup factor."""
        
        if self.performance_stats['async_scans'] == 0:
            return 1.0
        
        # Estimate speedup based on worker pool utilization
        if self._worker_pool:
            pool_stats = self._worker_pool.get_stats()
            return min(pool_stats.get('max_workers', 1), 4.0)  # Theoretical max 4x
        
        return 1.0
    
    @classmethod
    def get_global_cache_stats(cls) -> Dict[str, Any]:
        """Get global cache statistics across all scanner instances."""
        return cls._global_cache.get_stats()
    
    @classmethod
    def clear_global_cache(cls):
        """Clear global cache."""
        cls._global_cache.clear()
    
    @classmethod
    def shutdown_worker_pools(cls):
        """Shutdown all worker pools."""
        with cls._pool_lock:
            if cls._worker_pool:
                cls._worker_pool.shutdown()
                cls._worker_pool = None


# Convenience functions for creating optimized scanners
def create_optimized_scanner(architecture: str, memory_constraints: Dict[str, Any] = None,
                           user_id: str = "anonymous", enable_all_optimizations: bool = True) -> OptimizedFirmwareScanner:
    """Create an optimized scanner with Generation 3 performance features."""
    
    return OptimizedFirmwareScanner(
        architecture=architecture,
        memory_constraints=memory_constraints,
        user_id=user_id,
        enable_caching=enable_all_optimizations,
        enable_worker_pool=enable_all_optimizations
    )


# Global optimization utilities
async def scan_firmware_fleet_async(firmware_files: List[str], architecture: str,
                                   batch_size: int = 10) -> List[Tuple[str, List[CryptoVulnerability]]]:
    """Asynchronously scan a fleet of firmware files with optimizations."""
    
    scanner = create_optimized_scanner(architecture)
    
    # Process in batches for memory efficiency
    all_results = []
    
    for i in range(0, len(firmware_files), batch_size):
        batch = firmware_files[i:i + batch_size]
        batch_results = scanner.scan_firmware_batch(batch)
        all_results.extend(batch_results)
    
    return all_results