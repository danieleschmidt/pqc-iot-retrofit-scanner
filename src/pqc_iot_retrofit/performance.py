"""
Performance optimization and intelligent caching for PQC IoT Retrofit Scanner.

This module provides:
- Multi-level caching with intelligent eviction
- Performance optimization strategies
- Memory-efficient algorithms
- Adaptive resource management
"""

import time
import hashlib
import pickle
import threading
import gc
import multiprocessing
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from functools import wraps, lru_cache
from pathlib import Path
import weakref
import logging

from .monitoring import metrics_collector


T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""
    key: str
    value: T
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    tags: Optional[Dict[str, str]] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


class AdaptiveCache:
    """Intelligent multi-level cache with adaptive eviction."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100, 
                 default_ttl: Optional[float] = 3600):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # L1 Cache - In-memory, fast access
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # L2 Cache - Persistent, larger capacity
        self.l2_cache_dir = Path("cache/l2")
        self.l2_cache_dir.mkdir(parents=True, exist_ok=True)
        self.l2_index: Dict[str, Dict] = {}
        
        # Cache statistics
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'evictions': 0,
            'memory_usage': 0
        }
        
        # Threading support
        self.lock = threading.RLock()
        
        # Load L2 index
        self._load_l2_index()
        
        # Background cleanup thread
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            # Try L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    # Move to end (LRU)
                    self.l1_cache.move_to_end(key)
                    self.stats['l1_hits'] += 1
                    metrics_collector.record_metric("cache.l1.hits", 1, "hits")
                    return entry.value
                else:
                    # Expired, remove from L1
                    del self.l1_cache[key]
            
            self.stats['l1_misses'] += 1
            metrics_collector.record_metric("cache.l1.misses", 1, "misses")
            
            # Try L2 cache
            l2_value = self._get_from_l2(key)
            if l2_value is not None:
                self.stats['l2_hits'] += 1
                metrics_collector.record_metric("cache.l2.hits", 1, "hits")
                
                # Promote to L1 if there's space
                self._promote_to_l1(key, l2_value)
                return l2_value
            
            self.stats['l2_misses'] += 1
            metrics_collector.record_metric("cache.l2.misses", 1, "misses")
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            tags: Optional[Dict[str, str]] = None):
        """Store value in cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl,
                tags=tags
            )
            
            # Store in L1 if space available or evict
            if self._can_fit_in_l1(size_bytes):
                self._ensure_l1_space(size_bytes)
                self.l1_cache[key] = entry
                self.stats['memory_usage'] += size_bytes
            else:
                # Store directly in L2
                self._store_in_l2(key, entry)
            
            metrics_collector.record_metric("cache.puts", 1, "operations")
            metrics_collector.record_metric("cache.size_bytes", size_bytes, "bytes")
    
    def invalidate(self, key: str):
        """Remove key from cache."""
        with self.lock:
            # Remove from L1
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                self.stats['memory_usage'] -= entry.size_bytes
                del self.l1_cache[key]
            
            # Remove from L2  
            self._remove_from_l2(key)
    
    def invalidate_by_tags(self, tags: Dict[str, str]):
        """Invalidate all entries matching tags."""
        with self.lock:
            keys_to_remove = []
            
            # Check L1 cache
            for key, entry in self.l1_cache.items():
                if entry.tags and self._tags_match(entry.tags, tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.invalidate(key)
            
            # Check L2 cache
            for key, metadata in self.l2_index.items():
                if metadata.get('tags') and self._tags_match(metadata['tags'], tags):
                    self._remove_from_l2(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.l1_cache.clear()
            for file_path in self.l2_cache_dir.glob("*.cache"):
                file_path.unlink()
            self.l2_index.clear()
            self.stats['memory_usage'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            l1_hit_rate = (self.stats['l1_hits'] / 
                          max(self.stats['l1_hits'] + self.stats['l1_misses'], 1))
            l2_hit_rate = (self.stats['l2_hits'] / 
                          max(self.stats['l2_hits'] + self.stats['l2_misses'], 1))
            
            return {
                'l1_size': len(self.l1_cache),
                'l1_hit_rate': l1_hit_rate,
                'l2_size': len(self.l2_index),
                'l2_hit_rate': l2_hit_rate,
                'memory_usage_mb': self.stats['memory_usage'] / (1024 * 1024),
                'total_evictions': self.stats['evictions'],
                **self.stats
            }
    
    def _can_fit_in_l1(self, size_bytes: int) -> bool:
        """Check if entry can fit in L1 cache."""
        return (len(self.l1_cache) < self.max_size and 
                self.stats['memory_usage'] + size_bytes <= self.max_memory_bytes)
    
    def _ensure_l1_space(self, needed_bytes: int):
        """Ensure there's space in L1 cache."""
        while (len(self.l1_cache) >= self.max_size or 
               self.stats['memory_usage'] + needed_bytes > self.max_memory_bytes):
            
            if not self.l1_cache:
                break
            
            # Evict LRU item
            key, entry = self.l1_cache.popitem(last=False)
            self.stats['memory_usage'] -= entry.size_bytes
            self.stats['evictions'] += 1
            
            # Store evicted item in L2
            self._store_in_l2(key, entry)
    
    def _promote_to_l1(self, key: str, value: Any):
        """Promote L2 entry to L1 cache."""
        size_bytes = self._calculate_size(value)
        
        if self._can_fit_in_l1(size_bytes):
            self._ensure_l1_space(size_bytes)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes
            )
            
            self.l1_cache[key] = entry
            self.stats['memory_usage'] += size_bytes
    
    def _get_from_l2(self, key: str) -> Optional[Any]:
        """Get value from L2 cache."""
        if key not in self.l2_index:
            return None
        
        cache_file = self.l2_cache_dir / f"{key}.cache"
        
        try:
            with open(cache_file, 'rb') as f:
                entry_data = pickle.load(f)
                
            # Check expiration
            if entry_data.get('ttl') and time.time() - entry_data['created_at'] > entry_data['ttl']:
                self._remove_from_l2(key)
                return None
            
            return entry_data['value']
            
        except (FileNotFoundError, pickle.PickleError, KeyError):
            # Remove corrupted entry
            self._remove_from_l2(key)
            return None
    
    def _store_in_l2(self, key: str, entry: CacheEntry):
        """Store entry in L2 cache."""
        cache_file = self.l2_cache_dir / f"{key}.cache"
        
        try:
            entry_data = {
                'key': entry.key,
                'value': entry.value,
                'created_at': entry.created_at,
                'ttl': entry.ttl,
                'tags': entry.tags
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(entry_data, f)
            
            # Update index
            self.l2_index[key] = {
                'file_path': str(cache_file),
                'created_at': entry.created_at,
                'size_bytes': entry.size_bytes,
                'tags': entry.tags
            }
            
            self._save_l2_index()
            
        except Exception as e:
            logging.warning(f"Failed to store L2 cache entry {key}: {e}")
    
    def _remove_from_l2(self, key: str):
        """Remove entry from L2 cache."""
        if key in self.l2_index:
            cache_file = Path(self.l2_index[key]['file_path'])
            cache_file.unlink(missing_ok=True)
            del self.l2_index[key]
            self._save_l2_index()
    
    def _load_l2_index(self):
        """Load L2 cache index."""
        index_file = self.l2_cache_dir / "index.json"
        
        try:
            if index_file.exists():
                import json
                with open(index_file, 'r') as f:
                    self.l2_index = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load L2 cache index: {e}")
            self.l2_index = {}
    
    def _save_l2_index(self):
        """Save L2 cache index."""
        index_file = self.l2_cache_dir / "index.json"
        
        try:
            import json
            with open(index_file, 'w') as f:
                json.dump(self.l2_index, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save L2 cache index: {e}")
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _tags_match(self, entry_tags: Dict[str, str], match_tags: Dict[str, str]) -> bool:
        """Check if tags match."""
        for key, value in match_tags.items():
            if entry_tags.get(key) != value:
                return False
        return True
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while True:
                try:
                    time.sleep(300)  # Run every 5 minutes
                    self._cleanup_expired()
                except Exception as e:
                    logging.warning(f"Cache cleanup error: {e}")
        
        thread = threading.Thread(target=cleanup_loop, daemon=True)
        thread.start()
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            # Check L1 cache
            for key, entry in self.l1_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.l1_cache[key]
                self.stats['memory_usage'] -= entry.size_bytes
                del self.l1_cache[key]
            
            # Check L2 cache
            expired_l2_keys = []
            for key, metadata in self.l2_index.items():
                if (metadata.get('ttl') and 
                    current_time - metadata['created_at'] > metadata['ttl']):
                    expired_l2_keys.append(key)
            
            for key in expired_l2_keys:
                self._remove_from_l2(key)


class PerformanceOptimizer:
    """Performance optimization strategies and adaptive algorithms."""
    
    def __init__(self):
        self.cache = AdaptiveCache(max_size=5000, max_memory_mb=500)
        self.optimization_history = defaultdict(list)
        self.performance_profiles = {}
        
    def optimize_firmware_scanning(self, scanner_func: Callable) -> Callable:
        """Optimize firmware scanning with intelligent caching and preprocessing."""
        
        @wraps(scanner_func)
        def optimized_wrapper(self, firmware_path: str, base_address: int = 0):
            # Generate cache key
            firmware_stat = Path(firmware_path).stat()
            cache_key = hashlib.md5(
                f"{firmware_path}:{firmware_stat.st_mtime}:{firmware_stat.st_size}:{base_address}".encode()
            ).hexdigest()
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                metrics_collector.record_metric("scan.cache_hit", 1, "hits")
                return cached_result
            
            # Run actual scanning
            start_time = time.time()
            result = scanner_func(self, firmware_path, base_address)
            duration = time.time() - start_time
            
            # Cache result
            tags = {
                'architecture': getattr(self, 'architecture', 'unknown'),
                'firmware_size': str(firmware_stat.st_size)
            }
            self.cache.put(cache_key, result, ttl=3600, tags=tags)  # 1 hour TTL
            
            # Record performance
            metrics_collector.record_metric("scan.duration", duration, "seconds")
            metrics_collector.record_metric("scan.cache_miss", 1, "misses")
            
            return result
        
        return optimized_wrapper
    
    def optimize_pqc_generation(self, generator_func: Callable) -> Callable:
        """Optimize PQC implementation generation with caching."""
        
        @wraps(generator_func)
        def optimized_wrapper(self, optimization: str = "balanced"):
            # Generate cache key based on algorithm, architecture, and optimization
            cache_key = hashlib.md5(
                f"{self.__class__.__name__}:{self.target_arch}:{optimization}".encode()
            ).hexdigest()
            
            # Try cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                metrics_collector.record_metric("pqc_gen.cache_hit", 1, "hits")
                return cached_result
            
            # Generate implementation
            start_time = time.time()
            result = generator_func(self, optimization)
            duration = time.time() - start_time
            
            # Cache result (PQC implementations are deterministic)
            tags = {
                'algorithm': result.algorithm,
                'architecture': result.target_arch,
                'optimization': optimization
            }
            self.cache.put(cache_key, result, ttl=7200, tags=tags)  # 2 hour TTL
            
            metrics_collector.record_metric("pqc_gen.duration", duration, "seconds")
            metrics_collector.record_metric("pqc_gen.cache_miss", 1, "misses")
            
            return result
        
        return optimized_wrapper
    
    def create_memory_efficient_scanner(self, base_scanner_class):
        """Create memory-efficient version of scanner with streaming."""
        
        class MemoryEfficientScanner(base_scanner_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.chunk_size = 64 * 1024  # 64KB chunks
                self.max_memory_usage = 50 * 1024 * 1024  # 50MB limit
            
            def scan_firmware_streaming(self, firmware_path: str, base_address: int = 0):
                """Stream-based firmware scanning for large files."""
                firmware_size = Path(firmware_path).stat().st_size
                
                if firmware_size <= self.max_memory_usage:
                    # Use normal scanning for small files
                    return super().scan_firmware(firmware_path, base_address)
                
                # Stream-based scanning for large files
                self.logger.info(f"Using streaming scan for large file: {firmware_size / (1024*1024):.1f} MB")
                
                vulnerabilities = []
                
                with open(firmware_path, 'rb') as f:
                    offset = 0
                    
                    while True:
                        chunk = f.read(self.chunk_size)
                        if not chunk:
                            break
                        
                        # Scan chunk with overlap to catch cross-boundary patterns
                        overlap = 64  # 64 bytes overlap
                        if offset > 0:
                            f.seek(offset - overlap)
                            extended_chunk = f.read(len(chunk) + overlap)
                            scan_data = extended_chunk
                            scan_offset = base_address + offset - overlap
                        else:
                            scan_data = chunk
                            scan_offset = base_address + offset
                        
                        # Scan chunk
                        chunk_vulns = self._scan_chunk(scan_data, scan_offset)
                        
                        # Filter duplicates near boundaries
                        for vuln in chunk_vulns:
                            if not self._is_duplicate_vulnerability(vuln, vulnerabilities):
                                vulnerabilities.append(vuln)
                        
                        offset += len(chunk)
                        
                        # Force garbage collection periodically
                        if offset % (10 * self.chunk_size) == 0:
                            gc.collect()
                
                self._classify_vulnerabilities_list(vulnerabilities)
                return vulnerabilities
            
            def _scan_chunk(self, data: bytes, base_address: int):
                """Scan a single chunk of data."""
                # Temporarily store vulnerabilities
                old_vulns = self.vulnerabilities
                self.vulnerabilities = []
                
                # Use existing scan methods on chunk
                self._scan_crypto_constants(data, base_address)
                self._scan_crypto_strings(data, base_address)
                
                if self.disassembler:
                    self._scan_crypto_instructions(data, base_address)
                
                chunk_vulns = self.vulnerabilities
                self.vulnerabilities = old_vulns
                
                return chunk_vulns
            
            def _is_duplicate_vulnerability(self, vuln, existing_vulns):
                """Check if vulnerability is a duplicate (boundary artifact)."""
                for existing in existing_vulns:
                    if (abs(vuln.address - existing.address) < 64 and 
                        vuln.algorithm == existing.algorithm):
                        return True
                return False
            
            def _classify_vulnerabilities_list(self, vulnerabilities):
                """Classify a list of vulnerabilities."""
                for vuln in vulnerabilities:
                    # Apply same classification logic as parent
                    if vuln.algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.DH_1024]:
                        vuln.risk_level = RiskLevel.CRITICAL
                    elif vuln.key_size and vuln.key_size < 2048:
                        vuln.risk_level = RiskLevel.CRITICAL
        
        return MemoryEfficientScanner
    
    def adaptive_batch_size(self, operation_name: str, target_duration: float = 5.0) -> int:
        """Determine optimal batch size based on performance history."""
        
        if operation_name not in self.optimization_history:
            return 10  # Default batch size
        
        history = self.optimization_history[operation_name]
        
        if len(history) < 3:
            return min(50, len(history) * 10)  # Gradual increase
        
        # Calculate average processing time per item
        recent_history = history[-10:]  # Last 10 operations
        avg_time_per_item = sum(h['duration'] / h['items'] for h in recent_history) / len(recent_history)
        
        # Calculate optimal batch size
        optimal_batch = max(1, int(target_duration / avg_time_per_item))
        
        # Limit batch size based on memory constraints
        max_batch = min(1000, optimal_batch)
        
        return max_batch
    
    def record_batch_performance(self, operation_name: str, items_processed: int, duration: float):
        """Record batch processing performance."""
        self.optimization_history[operation_name].append({
            'timestamp': time.time(),
            'items': items_processed,
            'duration': duration,
            'throughput': items_processed / duration if duration > 0 else 0
        })
        
        # Keep only recent history
        if len(self.optimization_history[operation_name]) > 100:
            self.optimization_history[operation_name] = self.optimization_history[operation_name][-50:]


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()

# Decorators for performance optimization
def cached_result(ttl: int = 3600, tags: Optional[Dict[str, str]] = None):
    """Decorator for caching function results."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try cache
            result = performance_optimizer.cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            performance_optimizer.cache.put(cache_key, result, ttl=ttl, tags=tags)
            
            return result
        
        return wrapper
    return decorator


def memory_efficient(max_memory_mb: int = 100):
    """Decorator to ensure memory-efficient execution."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Monitor memory before execution
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                
                # Check memory usage after execution
                final_memory = process.memory_info().rss
                memory_used = (final_memory - initial_memory) / (1024 * 1024)  # MB
                
                if memory_used > max_memory_mb:
                    logging.warning(f"Function {func.__name__} used {memory_used:.1f} MB (limit: {max_memory_mb} MB)")
                
                # Force garbage collection if memory usage is high
                if memory_used > max_memory_mb / 2:
                    gc.collect()
                
                return result
                
            except MemoryError:
                # Force garbage collection and retry once
                gc.collect()
                logging.warning(f"Memory error in {func.__name__}, retrying after GC")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def auto_batch(target_duration: float = 5.0):
    """Decorator for automatic batch size optimization."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            
            # Get optimal batch size
            batch_size = performance_optimizer.adaptive_batch_size(operation_name, target_duration)
            
            results = []
            
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                start_time = time.time()
                batch_result = func(batch, *args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance
                performance_optimizer.record_batch_performance(operation_name, len(batch), duration)
                
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
            
            return results
        
        return wrapper
    return decorator


class ConcurrentProcessingPool:
    """Advanced concurrent processing with adaptive scaling."""
    
    def __init__(self, max_workers: int = None, enable_auto_scaling: bool = True):
        import concurrent.futures
        
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.enable_auto_scaling = enable_auto_scaling
        self.current_workers = min(2, self.max_workers)
        
        # Thread pool for I/O bound tasks
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="pqc_io"
        )
        
        # Process pool for CPU bound tasks
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.current_workers
        )
        
        # Performance tracking
        self.task_metrics = defaultdict(lambda: {'total_time': 0, 'count': 0})
        self.last_scale_check = time.time()
        
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O bound task to thread pool."""
        return self.thread_pool.submit(func, *args, **kwargs)
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit CPU bound task to process pool."""
        return self.process_pool.submit(func, *args, **kwargs)
    
    def map_parallel(self, func: Callable, items: List[Any], cpu_bound: bool = False, 
                    chunk_size: int = None) -> List[Any]:
        """Parallel map with automatic load balancing."""
        if not items:
            return []
        
        start_time = time.time()
        
        # Auto-scale if enabled
        if self.enable_auto_scaling:
            self._check_auto_scale(len(items))
        
        # Choose appropriate executor
        executor = self.process_pool if cpu_bound else self.thread_pool
        
        # Calculate optimal chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.current_workers * 4))
        
        # Submit work
        try:
            import concurrent.futures
            if cpu_bound:
                # Use process pool map for CPU-bound tasks
                results = list(executor.map(func, items, chunksize=chunk_size))
            else:
                # Use thread pool for I/O-bound tasks
                futures = [executor.submit(func, item) for item in items]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        except Exception as e:
            logging.error(f"Parallel processing error: {e}")
            # Fallback to sequential processing
            results = [func(item) for item in items]
        
        # Record metrics
        duration = time.time() - start_time
        task_name = func.__name__
        self.task_metrics[task_name]['total_time'] += duration
        self.task_metrics[task_name]['count'] += 1
        
        return results
    
    def _check_auto_scale(self, workload_size: int):
        """Check if we should scale workers up or down."""
        now = time.time()
        if now - self.last_scale_check < 30:  # Check every 30 seconds
            return
        
        self.last_scale_check = now
        
        # Calculate average task performance
        if not self.task_metrics:
            return
        
        avg_task_time = sum(m['total_time'] / max(m['count'], 1) for m in self.task_metrics.values()) / len(self.task_metrics)
        estimated_duration = (workload_size * avg_task_time) / self.current_workers
        
        # Scale up if tasks would take too long
        if estimated_duration > 60 and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 2, self.max_workers)
            self._scale_workers(new_workers)
        
        # Scale down if workers are underutilized
        elif estimated_duration < 10 and self.current_workers > 2:
            new_workers = max(self.current_workers - 1, 2)
            self._scale_workers(new_workers)
    
    def _scale_workers(self, new_count: int):
        """Scale worker pools."""
        if new_count == self.current_workers:
            return
        
        logging.info(f"Scaling workers from {self.current_workers} to {new_count}")
        
        # Update thread pool
        import concurrent.futures
        old_thread_pool = self.thread_pool
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=new_count,
            thread_name_prefix="pqc_io"
        )
        old_thread_pool.shutdown(wait=False)
        
        # Update process pool
        old_process_pool = self.process_pool
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=new_count
        )
        old_process_pool.shutdown(wait=False)
        
        self.current_workers = new_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return {
            'current_workers': self.current_workers,
            'max_workers': self.max_workers,
            'auto_scaling_enabled': self.enable_auto_scaling,
            'task_metrics': dict(self.task_metrics),
            'total_tasks': sum(m['count'] for m in self.task_metrics.values()),
            'total_processing_time': sum(m['total_time'] for m in self.task_metrics.values())
        }
    
    def shutdown(self):
        """Shutdown all worker pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


# Global concurrent processing pool
concurrent_pool = ConcurrentProcessingPool()


# Enhanced decorators for Generation 3 performance
def parallel_processing(cpu_bound: bool = False, chunk_size: int = None):
    """Decorator for automatic parallel processing."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(items: List[Any], *args, **kwargs):
            if len(items) <= 1:
                # Not worth parallelizing
                return [func(item, *args, **kwargs) for item in items]
            
            return concurrent_pool.map_parallel(func, items, cpu_bound, chunk_size)
        
        return wrapper
    return decorator


def optimized_for_embedded():
    """Decorator to optimize function for embedded constraints."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Enable aggressive garbage collection
            gc.collect()
            
            # Monitor memory before execution
            start_memory = 0
            try:
                import psutil
                process = psutil.Process()
                start_memory = process.memory_info().rss / (1024 * 1024)
            except ImportError:
                pass
            
            # Execute with timeout to prevent hangs
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Function execution timeout")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30-second timeout
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
                # Force cleanup after execution
                gc.collect()
            
            return result
        
        return wrapper
    return decorator