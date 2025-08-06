"""
Concurrent processing and resource pooling for PQC IoT Retrofit Scanner.

This module provides:
- Parallel firmware scanning across multiple threads/processes
- Resource pooling for expensive operations
- Load balancing and work distribution
- Async/await support for I/O bound operations
"""

import asyncio
import concurrent.futures
import multiprocessing
import threading
import time
import queue
import weakref
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from contextlib import contextmanager

from .monitoring import metrics_collector, health_monitor
from .error_handling import PQCRetrofitError, ErrorSeverity, ErrorCategory

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class WorkItem(Generic[T]):
    """Work item for processing queues."""
    id: str
    data: T
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    retries: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Support priority queue ordering."""
        return self.priority > other.priority  # Higher priority = lower number


@dataclass 
class ProcessingResult(Generic[R]):
    """Result from processing a work item."""
    item_id: str
    success: bool
    result: Optional[R] = None
    error: Optional[Exception] = None
    duration: float = 0.0
    worker_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourcePool(ABC):
    """Abstract base class for resource pools."""
    
    @abstractmethod
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        pass
    
    @abstractmethod
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get current pool size."""
        pass
    
    @abstractmethod
    def available(self) -> int:
        """Get number of available resources."""
        pass


class ThreadSafeResourcePool(ResourcePool):
    """Thread-safe resource pool with automatic cleanup."""
    
    def __init__(self, factory: Callable[[], Any], max_size: int = 10, 
                 cleanup_func: Optional[Callable[[Any], None]] = None):
        self.factory = factory
        self.max_size = max_size
        self.cleanup_func = cleanup_func
        self.pool = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = threading.RLock()
        self.active_resources = {}  # id(resource) -> resource
        
        # Pre-populate pool with initial resources
        initial_size = min(2, max_size)
        for _ in range(initial_size):
            resource = self._create_resource()
            self.pool.put(resource)
    
    def _create_resource(self) -> Any:
        """Create a new resource."""
        with self.lock:
            if self.created_count >= self.max_size:
                raise PQCRetrofitError(
                    f"Cannot create more resources, max size {self.max_size} reached",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.RESOURCE_ERROR
                )
            
            resource = self.factory()
            self.created_count += 1
            self.active_resources[id(resource)] = resource
            
            metrics_collector.record_metric("resource_pool.created", 1, "resources")
            return resource
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """Acquire a resource from the pool."""
        try:
            # Try to get existing resource
            resource = self.pool.get(block=True, timeout=timeout or 30.0)
            metrics_collector.record_metric("resource_pool.acquired", 1, "resources")
            return resource
            
        except queue.Empty:
            # Pool is empty, try to create new resource
            try:
                resource = self._create_resource()
                metrics_collector.record_metric("resource_pool.created_on_demand", 1, "resources")
                return resource
            except Exception as e:
                raise PQCRetrofitError(
                    f"Failed to acquire resource: {e}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.RESOURCE_ERROR
                ) from e
    
    def release(self, resource: Any):
        """Release a resource back to the pool."""
        if id(resource) in self.active_resources:
            try:
                # Put back in pool if there's space
                self.pool.put(resource, block=False)
                metrics_collector.record_metric("resource_pool.released", 1, "resources")
            except queue.Full:
                # Pool is full, cleanup the resource
                self._cleanup_resource(resource)
        else:
            # Resource not from this pool, cleanup anyway
            self._cleanup_resource(resource)
    
    def _cleanup_resource(self, resource: Any):
        """Clean up a resource."""
        try:
            if self.cleanup_func:
                self.cleanup_func(resource)
            
            with self.lock:
                self.active_resources.pop(id(resource), None)
                self.created_count = max(0, self.created_count - 1)
                
            metrics_collector.record_metric("resource_pool.cleaned_up", 1, "resources")
            
        except Exception as e:
            logging.warning(f"Error cleaning up resource: {e}")
    
    def size(self) -> int:
        """Get current pool size."""
        return self.created_count
    
    def available(self) -> int:
        """Get number of available resources."""
        return self.pool.qsize()
    
    @contextmanager
    def get_resource(self, timeout: Optional[float] = None):
        """Context manager for acquiring and releasing resources."""
        resource = self.acquire(timeout)
        try:
            yield resource
        finally:
            self.release(resource)


class WorkerPool:
    """Pool of worker threads/processes for concurrent processing."""
    
    def __init__(self, worker_count: int = None, use_processes: bool = False, 
                 queue_size: int = 1000):
        self.worker_count = worker_count or min(8, (multiprocessing.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.queue_size = queue_size
        
        # Work queues
        self.work_queue = queue.PriorityQueue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        
        # Worker management
        self.workers: List[Union[threading.Thread, multiprocessing.Process]] = []
        self.shutdown_event = threading.Event() if not use_processes else multiprocessing.Event()
        self.executor = None
        
        # Statistics
        self.stats = {
            'items_processed': 0,
            'items_failed': 0,
            'total_processing_time': 0.0,
            'workers_active': 0
        }
        self.stats_lock = threading.Lock()
        
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads or processes."""
        if self.use_processes:
            # Use ProcessPoolExecutor for CPU-intensive work
            self.executor = ProcessPoolExecutor(max_workers=self.worker_count)
        else:
            # Use ThreadPoolExecutor for I/O-intensive work
            self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        
        logging.info(f"Started worker pool with {self.worker_count} {'processes' if self.use_processes else 'threads'}")
        metrics_collector.record_metric("worker_pool.started", self.worker_count, "workers")
    
    def submit_work(self, work_item: WorkItem[T]) -> Future:
        """Submit work item for processing."""
        if self.shutdown_event.is_set():
            raise PQCRetrofitError(
                "Worker pool is shutting down",
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.RESOURCE_ERROR
            )
        
        # Submit to executor
        future = self.executor.submit(self._process_work_item, work_item)
        
        metrics_collector.record_metric("worker_pool.submitted", 1, "items")
        return future
    
    def submit_batch(self, work_items: List[WorkItem[T]]) -> List[Future]:
        """Submit multiple work items for processing."""
        futures = []
        
        for item in work_items:
            try:
                future = self.submit_work(item)
                futures.append(future)
            except Exception as e:
                logging.error(f"Failed to submit work item {item.id}: {e}")
                # Create a failed future
                failed_future = Future()
                failed_future.set_exception(e)
                futures.append(failed_future)
        
        return futures
    
    def _process_work_item(self, work_item: WorkItem[T]) -> ProcessingResult[R]:
        """Process a single work item."""
        worker_id = f"{threading.current_thread().ident}"
        start_time = time.time()
        
        try:
            with self.stats_lock:
                self.stats['workers_active'] += 1
            
            # Call the work item's callback if provided
            if work_item.callback:
                result = work_item.callback(work_item.data)
            else:
                # Default processing (should be overridden in subclasses)
                result = self._default_process(work_item.data)
            
            duration = time.time() - start_time
            
            # Update statistics
            with self.stats_lock:
                self.stats['items_processed'] += 1
                self.stats['total_processing_time'] += duration
                self.stats['workers_active'] -= 1
            
            # Record metrics
            metrics_collector.record_metric("worker_pool.processed", 1, "items")
            metrics_collector.record_metric("worker_pool.processing_time", duration, "seconds")
            
            return ProcessingResult(
                item_id=work_item.id,
                success=True,
                result=result,
                duration=duration,
                worker_id=worker_id,
                metadata=work_item.metadata
            )
            
        except Exception as e:
            duration = time.time() - start_time
            
            with self.stats_lock:
                self.stats['items_failed'] += 1
                self.stats['workers_active'] -= 1
            
            metrics_collector.record_metric("worker_pool.failed", 1, "items")
            
            logging.error(f"Work item {work_item.id} failed: {e}")
            
            return ProcessingResult(
                item_id=work_item.id,
                success=False,
                error=e,
                duration=duration,
                worker_id=worker_id,
                metadata=work_item.metadata
            )
    
    def _default_process(self, data: T) -> R:
        """Default processing function - should be overridden."""
        return data  # type: ignore
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self.stats_lock:
            avg_processing_time = (
                self.stats['total_processing_time'] / max(self.stats['items_processed'], 1)
            )
            
            return {
                'worker_count': self.worker_count,
                'use_processes': self.use_processes,
                'items_processed': self.stats['items_processed'],
                'items_failed': self.stats['items_failed'],
                'workers_active': self.stats['workers_active'],
                'average_processing_time': avg_processing_time,
                'success_rate': (
                    self.stats['items_processed'] / 
                    max(self.stats['items_processed'] + self.stats['items_failed'], 1)
                )
            }
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the worker pool."""
        if self.executor:
            logging.info("Shutting down worker pool...")
            self.shutdown_event.set()
            
            self.executor.shutdown(wait=wait)
            
            metrics_collector.record_metric("worker_pool.shutdown", 1, "events")
            logging.info("Worker pool shutdown complete")


class FirmwareScannerPool(WorkerPool):
    """Specialized worker pool for firmware scanning."""
    
    def __init__(self, scanner_class, scanner_kwargs: Dict[str, Any] = None, 
                 worker_count: int = None):
        super().__init__(worker_count=worker_count, use_processes=False)  # Use threads for I/O
        self.scanner_class = scanner_class
        self.scanner_kwargs = scanner_kwargs or {}
        
        # Create resource pool for scanner instances
        self.scanner_pool = ThreadSafeResourcePool(
            factory=self._create_scanner,
            max_size=self.worker_count * 2,  # Allow more scanners than workers
            cleanup_func=self._cleanup_scanner
        )
    
    def _create_scanner(self):
        """Create a new scanner instance."""
        return self.scanner_class(**self.scanner_kwargs)
    
    def _cleanup_scanner(self, scanner):
        """Clean up scanner instance."""
        if hasattr(scanner, 'cleanup'):
            scanner.cleanup()
    
    def _process_work_item(self, work_item: WorkItem[str]) -> ProcessingResult[List]:
        """Process firmware scanning work item."""
        firmware_path = work_item.data
        
        with self.scanner_pool.get_resource() as scanner:
            try:
                # Scan firmware
                vulnerabilities = scanner.scan_firmware(
                    firmware_path, 
                    base_address=work_item.metadata.get('base_address', 0)
                )
                
                return ProcessingResult(
                    item_id=work_item.id,
                    success=True,
                    result=vulnerabilities,
                    worker_id=f"{threading.current_thread().ident}",
                    metadata={
                        'firmware_path': firmware_path,
                        'vulnerability_count': len(vulnerabilities)
                    }
                )
                
            except Exception as e:
                raise PQCRetrofitError(
                    f"Failed to scan firmware {firmware_path}: {e}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.PROCESSING_ERROR
                ) from e


class PQCGeneratorPool(WorkerPool):
    """Specialized worker pool for PQC implementation generation."""
    
    def __init__(self, generator_class, worker_count: int = None):
        super().__init__(worker_count=worker_count, use_processes=True)  # Use processes for CPU-intensive work
        self.generator_class = generator_class
    
    def _process_work_item(self, work_item: WorkItem[Dict]) -> ProcessingResult[Any]:
        """Process PQC generation work item."""
        params = work_item.data
        
        try:
            # Create generator instance (in process)
            generator = self.generator_class(
                target_arch=params['target_arch'],
                optimization_level=params.get('optimization_level', 2)
            )
            
            # Generate PQC implementation
            if params['algorithm'] == 'kyber':
                result = generator.generate_kyber512(params.get('optimization', 'balanced'))
            elif params['algorithm'] == 'dilithium':
                result = generator.generate_dilithium2(params.get('optimization', 'balanced'))
            else:
                raise ValueError(f"Unsupported algorithm: {params['algorithm']}")
            
            return ProcessingResult(
                item_id=work_item.id,
                success=True,
                result=result,
                worker_id=f"{multiprocessing.current_process().pid}",
                metadata={
                    'algorithm': params['algorithm'],
                    'target_arch': params['target_arch'],
                    'code_size': len(result.c_code) if hasattr(result, 'c_code') else 0
                }
            )
            
        except Exception as e:
            raise PQCRetrofitError(
                f"Failed to generate PQC implementation: {e}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING_ERROR
            ) from e


class AsyncWorkManager:
    """Async/await based work manager for I/O-bound operations."""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.stats = {
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0
        }
    
    async def process_batch_async(self, items: List[T], 
                                 processor: Callable[[T], Awaitable[R]]) -> List[R]:
        """Process a batch of items asynchronously."""
        
        async def process_with_semaphore(item: T) -> R:
            async with self.semaphore:
                self.stats['tasks_started'] += 1
                try:
                    result = await processor(item)
                    self.stats['tasks_completed'] += 1
                    return result
                except Exception as e:
                    self.stats['tasks_failed'] += 1
                    logging.error(f"Async processing failed: {e}")
                    raise
        
        # Create tasks for all items
        tasks = [process_with_semaphore(item) for item in items]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Task failed with exception: {result}")
            else:
                successful_results.append(result)
        
        return successful_results
    
    async def scan_firmware_async(self, firmware_paths: List[str], 
                                 scanner_factory: Callable[[], Any]) -> List[List]:
        """Asynchronously scan multiple firmware files."""
        
        async def scan_single_firmware(firmware_path: str) -> List:
            # Run CPU-bound scanning in thread pool
            loop = asyncio.get_event_loop()
            
            def scan_sync():
                scanner = scanner_factory()
                return scanner.scan_firmware(firmware_path)
            
            return await loop.run_in_executor(None, scan_sync)
        
        return await self.process_batch_async(firmware_paths, scan_single_firmware)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get async manager statistics."""
        return {
            'max_concurrent': self.max_concurrent,
            'active_tasks': self.max_concurrent - self.semaphore._value,
            **self.stats,
            'success_rate': (
                self.stats['tasks_completed'] / 
                max(self.stats['tasks_started'], 1)
            )
        }


class LoadBalancer:
    """Simple load balancer for distributing work across multiple worker pools."""
    
    def __init__(self, worker_pools: List[WorkerPool], strategy: str = "round_robin"):
        self.worker_pools = worker_pools
        self.strategy = strategy
        self.current_index = 0
        self.lock = threading.Lock()
    
    def submit_work(self, work_item: WorkItem[T]) -> Future:
        """Submit work to the least loaded worker pool."""
        if self.strategy == "round_robin":
            return self._round_robin_submit(work_item)
        elif self.strategy == "least_loaded":
            return self._least_loaded_submit(work_item)
        else:
            raise ValueError(f"Unknown load balancing strategy: {self.strategy}")
    
    def _round_robin_submit(self, work_item: WorkItem[T]) -> Future:
        """Round-robin load balancing."""
        with self.lock:
            pool = self.worker_pools[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.worker_pools)
        
        return pool.submit_work(work_item)
    
    def _least_loaded_submit(self, work_item: WorkItem[T]) -> Future:
        """Submit to least loaded pool."""
        # Find pool with fewest active workers
        best_pool = min(self.worker_pools, 
                       key=lambda p: p.stats['workers_active'])
        
        return best_pool.submit_work(work_item)
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics from all worker pools."""
        total_stats = {
            'total_pools': len(self.worker_pools),
            'total_workers': 0,
            'total_processed': 0,
            'total_failed': 0,
            'total_active': 0
        }
        
        for pool in self.worker_pools:
            stats = pool.get_stats()
            total_stats['total_workers'] += stats['worker_count']
            total_stats['total_processed'] += stats['items_processed']
            total_stats['total_failed'] += stats['items_failed']
            total_stats['total_active'] += stats['workers_active']
        
        total_stats['overall_success_rate'] = (
            total_stats['total_processed'] / 
            max(total_stats['total_processed'] + total_stats['total_failed'], 1)
        )
        
        return total_stats
    
    def shutdown_all(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown all worker pools."""
        for pool in self.worker_pools:
            pool.shutdown(wait=wait, timeout=timeout)


# Global instances
firmware_scanner_pool = None
pqc_generator_pool = None
async_work_manager = AsyncWorkManager()


def initialize_pools(scanner_class=None, generator_class=None, 
                    scanner_workers: int = None, generator_workers: int = None,
                    scanner_kwargs: Dict[str, Any] = None, generator_kwargs: Dict[str, Any] = None):
    """Initialize global worker pools."""
    global firmware_scanner_pool, pqc_generator_pool
    
    if scanner_class:
        firmware_scanner_pool = FirmwareScannerPool(
            scanner_class=scanner_class,
            scanner_kwargs=scanner_kwargs or {},
            worker_count=scanner_workers
        )
        logging.info("Initialized firmware scanner pool")
    
    if generator_class:
        pqc_generator_pool = PQCGeneratorPool(
            generator_class=generator_class,
            worker_count=generator_workers
        )
        logging.info("Initialized PQC generator pool")


def shutdown_pools():
    """Shutdown all global worker pools."""
    global firmware_scanner_pool, pqc_generator_pool
    
    if firmware_scanner_pool:
        firmware_scanner_pool.shutdown()
        firmware_scanner_pool = None
    
    if pqc_generator_pool:
        pqc_generator_pool.shutdown()
        pqc_generator_pool = None
    
    logging.info("All worker pools shutdown")