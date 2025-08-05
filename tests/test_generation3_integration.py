"""
Integration tests for Generation 3 (Make it Scale) implementations.

Tests the complete scaling, concurrent processing, and performance optimization system.
"""

import pytest
import asyncio
import time
import threading
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
import sys
sys.path.insert(0, 'src')

from pqc_iot_retrofit.performance import (
    AdaptiveCache, PerformanceOptimizer, performance_optimizer,
    cached_result, memory_efficient, auto_batch
)
from pqc_iot_retrofit.concurrency import (
    WorkItem, WorkerPool, FirmwareScannerPool, PQCGeneratorPool,
    AsyncWorkManager, LoadBalancer, ThreadSafeResourcePool
)
from pqc_iot_retrofit.auto_scaling import (
    AdaptiveScalingPolicy, PredictiveScalingPolicy, AutoScaler,
    ScalingMetrics, LoadLevel, ScalingDirection, CircuitBreaker
)
from pqc_iot_retrofit.scanner import FirmwareScanner
from pqc_iot_retrofit.pqc_implementations import PQCImplementationGenerator


class TestAdaptiveCacheSystem:
    """Test the adaptive caching system."""
    
    def test_l1_cache_operations(self):
        """Test L1 cache basic operations."""
        cache = AdaptiveCache(max_size=3, max_memory_mb=1)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['l1_size'] == 1
        assert stats['l1_hit_rate'] > 0
    
    def test_l2_cache_promotion(self):
        """Test L2 cache promotion to L1."""
        cache = AdaptiveCache(max_size=2, max_memory_mb=1)
        
        # Fill L1 cache beyond capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1 to L2
        
        # Access key1 should promote it back to L1
        value = cache.get("key1")
        assert value == "value1"
        
        stats = cache.get_stats()
        assert stats['l2_hits'] > 0
    
    def test_cache_invalidation_by_tags(self):
        """Test cache invalidation by tags."""
        cache = AdaptiveCache()
        
        # Put items with tags
        cache.put("firmware1", "data1", tags={"arch": "arm", "type": "firmware"})
        cache.put("firmware2", "data2", tags={"arch": "x86", "type": "firmware"})
        cache.put("config1", "data3", tags={"type": "config"})
        
        # Invalidate by tag
        cache.invalidate_by_tags({"type": "firmware"})
        
        # Check invalidation
        assert cache.get("firmware1") is None
        assert cache.get("firmware2") is None
        assert cache.get("config1") == "data3"
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = AdaptiveCache()
        
        # Put item with short TTL
        cache.put("temp_key", "temp_value", ttl=0.1)
        
        # Should be available immediately
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired now
        assert cache.get("temp_key") is None


class TestPerformanceOptimizers:
    """Test performance optimization decorators."""
    
    def test_cached_result_decorator(self):
        """Test cached result decorator."""
        call_count = 0
        
        @cached_result(ttl=3600)
        def expensive_function(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return f"result_{arg1}_{arg2}"
        
        # First call
        result1 = expensive_function("a", "b")
        assert result1 == "result_a_b"
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function("a", "b")
        assert result2 == "result_a_b"
        assert call_count == 1  # Should not increment
        
        # Different args should call function
        result3 = expensive_function("c", "d")
        assert result3 == "result_c_d"
        assert call_count == 2
    
    def test_memory_efficient_decorator(self):
        """Test memory efficient decorator."""
        
        @memory_efficient(max_memory_mb=50)
        def memory_using_function():
            # Simulate some memory usage
            data = [0] * 1000
            return len(data)
        
        result = memory_using_function()
        assert result == 1000
    
    def test_auto_batch_decorator(self):
        """Test auto-batch decorator."""
        process_count = 0
        
        @auto_batch(target_duration=1.0)
        def batch_processor(items):
            nonlocal process_count
            process_count += 1
            return [f"processed_{item}" for item in items]
        
        items = ["item1", "item2", "item3", "item4", "item5"]
        results = batch_processor(items)
        
        assert len(results) == 5
        assert all("processed_" in result for result in results)
        assert process_count >= 1


class TestConcurrentProcessing:
    """Test concurrent processing capabilities."""
    
    def test_thread_safe_resource_pool(self):
        """Test thread-safe resource pool."""
        
        def create_resource():
            return {"id": threading.current_thread().ident, "data": "resource_data"}
        
        def cleanup_resource(resource):
            resource["cleaned"] = True
        
        pool = ThreadSafeResourcePool(
            factory=create_resource,
            max_size=3,
            cleanup_func=cleanup_resource
        )
        
        # Test resource acquisition
        with pool.get_resource() as resource:
            assert "id" in resource
            assert resource["data"] == "resource_data"
        
        # Test pool statistics
        assert pool.size() >= 1
        assert pool.available() >= 0
    
    def test_worker_pool_basic_operations(self):
        """Test basic worker pool operations."""
        
        def simple_processor(data):
            return f"processed_{data}"
        
        pool = WorkerPool(worker_count=2, use_processes=False)
        
        try:
            # Submit work
            work_item = WorkItem(id="test1", data="test_data", callback=simple_processor)
            future = pool.submit_work(work_item)
            
            # Get result
            result = future.result(timeout=5.0)
            assert result.success
            assert result.result == "processed_test_data"
            
            # Check statistics
            stats = pool.get_stats()
            assert stats['worker_count'] == 2
            assert stats['items_processed'] >= 1
            
        finally:
            pool.shutdown(wait=True, timeout=5.0)
    
    def test_firmware_scanner_pool(self):
        """Test firmware scanner pool."""
        
        # Mock scanner class
        class MockScanner:
            def __init__(self):
                pass
            
            def scan_firmware(self, firmware_path, base_address=0):
                return [{"type": "test_vulnerability", "path": firmware_path}]
        
        pool = FirmwareScannerPool(
            scanner_class=MockScanner,
            worker_count=2
        )
        
        try:
            # Create temp firmware file
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(b"fake firmware data")
                firmware_path = tmp.name
            
            # Submit scanning work
            work_item = WorkItem(id="scan1", data=firmware_path)
            future = pool.submit_work(work_item)
            
            result = future.result(timeout=5.0)
            assert result.success
            assert len(result.result) == 1
            assert result.result[0]["path"] == firmware_path
            
        finally:
            pool.shutdown(wait=True, timeout=5.0)
            os.unlink(firmware_path)
    
    def test_pqc_generator_pool(self):
        """Test PQC generator pool."""
        
        # Mock generator class
        class MockPQCGenerator:
            def __init__(self, target_arch, optimization_level=2):
                self.target_arch = target_arch
                self.optimization_level = optimization_level
            
            def generate_kyber512(self, optimization="balanced"):
                return Mock(
                    algorithm="kyber512",
                    target_arch=self.target_arch,
                    c_code="mock c code",
                    optimization=optimization
                )
        
        pool = PQCGeneratorPool(
            generator_class=MockPQCGenerator,
            worker_count=2
        )
        
        try:
            # Submit generation work
            work_data = {
                "algorithm": "kyber",
                "target_arch": "cortex-m4",
                "optimization": "speed"
            }
            work_item = WorkItem(id="gen1", data=work_data)
            future = pool.submit_work(work_item)
            
            result = future.result(timeout=10.0)
            assert result.success
            assert result.result.algorithm == "kyber512"
            assert result.result.target_arch == "cortex-m4"
            
        finally:
            pool.shutdown(wait=True, timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_async_work_manager(self):
        """Test async work manager."""
        
        async def async_processor(item):
            await asyncio.sleep(0.01)  # Simulate async work
            return f"async_processed_{item}"
        
        manager = AsyncWorkManager(max_concurrent=3)
        
        items = ["item1", "item2", "item3", "item4", "item5"]
        results = await manager.process_batch_async(items, async_processor)
        
        assert len(results) == 5
        assert all("async_processed_" in result for result in results)
        
        stats = manager.get_stats()
        assert stats['tasks_completed'] == 5
        assert stats['success_rate'] == 1.0
    
    def test_load_balancer(self):
        """Test load balancer across multiple pools."""
        
        def simple_processor(data):
            return f"processed_{data}"
        
        # Create multiple worker pools
        pool1 = WorkerPool(worker_count=1, use_processes=False)
        pool2 = WorkerPool(worker_count=1, use_processes=False)
        
        try:
            # Create load balancer
            balancer = LoadBalancer([pool1, pool2], strategy="round_robin")
            
            # Submit multiple work items
            futures = []
            for i in range(4):
                work_item = WorkItem(id=f"work{i}", data=f"data{i}", callback=simple_processor)
                future = balancer.submit_work(work_item)
                futures.append(future)
            
            # Collect results
            results = [f.result(timeout=5.0) for f in futures]
            
            assert len(results) == 4
            assert all(r.success for r in results)
            
            # Check aggregate stats
            stats = balancer.get_aggregate_stats()
            assert stats['total_pools'] == 2
            assert stats['total_workers'] == 2
            assert stats['total_processed'] >= 4
            
        finally:
            balancer.shutdown_all(wait=True, timeout=5.0)


class TestAutoScaling:
    """Test auto-scaling functionality."""
    
    def test_adaptive_scaling_policy(self):
        """Test adaptive scaling policy decisions."""
        policy = AdaptiveScalingPolicy(min_workers=2, max_workers=10)
        
        # Test scale-up conditions
        high_load_metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=0.9,
            memory_usage=0.8,
            queue_depth=15,
            active_workers=3,
            throughput=5.0,
            response_time=8.0,
            error_rate=0.02,
            load_level=LoadLevel.HIGH
        )
        
        direction = policy.should_scale(high_load_metrics, [])
        assert direction == ScalingDirection.UP
        
        target = policy.calculate_target_size(3, high_load_metrics)
        assert target > 3
        assert target <= 10
        
        # Test scale-down conditions
        low_load_metrics = ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=0.2,
            memory_usage=0.2,
            queue_depth=0,
            active_workers=5,
            throughput=1.0,
            response_time=1.0,
            error_rate=0.0,
            load_level=LoadLevel.LOW
        )
        
        direction = policy.should_scale(low_load_metrics, [])
        assert direction == ScalingDirection.DOWN
        
        target = policy.calculate_target_size(5, low_load_metrics)
        assert target < 5
        assert target >= 2
    
    def test_predictive_scaling_policy(self):
        """Test predictive scaling policy."""
        policy = PredictiveScalingPolicy(min_workers=2, max_workers=10)
        
        # Create metrics history showing increasing load
        history = []
        for i in range(10):
            metrics = ScalingMetrics(
                timestamp=time.time() - (10-i) * 60,  # 1 minute intervals
                cpu_usage=0.3 + i * 0.05,
                memory_usage=0.3 + i * 0.04,
                queue_depth=i,
                active_workers=2,
                throughput=2.0 + i * 0.1,
                response_time=2.0 + i * 0.2,
                error_rate=0.01,
                load_level=LoadLevel.NORMAL
            )
            history.append(metrics)
        
        current_metrics = history[-1]
        direction = policy.should_scale(current_metrics, history[:-1])
        
        # Should predict need for scaling up
        assert direction in [ScalingDirection.UP, ScalingDirection.STABLE]
        
        target = policy.calculate_target_size(2, current_metrics)
        assert target >= 2
    
    def test_circuit_breaker(self):
        """Test circuit breaker for load shedding."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def failing_function():
            raise Exception("Simulated failure")
        
        def success_function():
            return "success"
        
        # Test normal operation
        result = breaker.call(success_function)
        assert result == "success"
        
        # Test failure handling
        failure_count = 0
        for _ in range(5):
            try:
                breaker.call(failing_function)
            except Exception:
                failure_count += 1
        
        # Should have failed multiple times
        assert failure_count > 0
        
        # After enough failures, circuit should be open
        assert breaker.state == "OPEN"
        
        # Should reject calls when open
        with pytest.raises(Exception):
            breaker.call(success_function)
    
    def test_autoscaler_integration(self):
        """Test complete auto-scaler integration."""
        
        # Mock worker pool
        mock_pool = Mock()
        mock_pool.worker_count = 3
        mock_pool.get_stats.return_value = {
            'worker_count': 3,
            'items_processed': 100,
            'items_failed': 5,
            'workers_active': 2,
            'average_processing_time': 2.5
        }
        
        policy = AdaptiveScalingPolicy(min_workers=2, max_workers=8)
        autoscaler = AutoScaler(
            worker_pool=mock_pool,
            scaling_policy=policy,
            check_interval=0.1,  # Fast for testing
            cooldown_period=0.5   # Short cooldown for testing
        )
        
        try:
            # Start autoscaler
            autoscaler.start()
            assert autoscaler.running
            
            # Let it run for a short time
            time.sleep(0.3)
            
            # Check stats
            stats = autoscaler.get_stats()
            assert stats['running']
            assert stats['metrics_history_size'] > 0
            
        finally:
            autoscaler.stop()
            assert not autoscaler.running


class TestCompleteSystemIntegration:
    """Test complete Generation 3 system integration."""
    
    def test_end_to_end_scaling_workflow(self):
        """Test complete end-to-end scaling workflow."""
        
        # Mock components
        mock_scanner = Mock()
        mock_scanner.scan_firmware.return_value = [
            {"type": "rsa_key", "address": 0x1000, "size": 2048}
        ]
        
        # Create firmware scanner pool with auto-scaling
        scanner_pool = FirmwareScannerPool(
            scanner_class=lambda: mock_scanner,
            worker_count=2
        )
        
        try:
            # Create auto-scaler for the pool
            policy = AdaptiveScalingPolicy(min_workers=1, max_workers=5)
            autoscaler = AutoScaler(
                worker_pool=scanner_pool,
                scaling_policy=policy,
                check_interval=0.2,
                cooldown_period=0.5
            )
            
            # Start auto-scaler
            autoscaler.start()
            
            # Submit multiple scanning jobs to trigger scaling
            futures = []
            for i in range(10):
                work_item = WorkItem(id=f"scan_{i}", data=f"firmware_{i}.bin")
                future = scanner_pool.submit_work(work_item)
                futures.append(future)
            
            # Wait for completion
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=5.0)
                    results.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")
            
            # Verify results
            successful_results = [r for r in results if r.success]
            assert len(successful_results) >= 5  # At least half should succeed
            
            # Let auto-scaler run and collect metrics
            time.sleep(1.0)
            
            # Check auto-scaler collected metrics
            autoscaler_stats = autoscaler.get_stats()
            assert autoscaler_stats['metrics_history_size'] > 0
            
            # Stop auto-scaler
            autoscaler.stop()
            
        finally:
            scanner_pool.shutdown(wait=True, timeout=5.0)
    
    def test_performance_optimization_integration(self):
        """Test performance optimization features work together."""
        
        # Test cache integration with worker pools
        cache = AdaptiveCache(max_size=100, max_memory_mb=10)
        
        @cached_result(ttl=3600)
        def expensive_scan_operation(firmware_data):
            # Simulate expensive scanning
            time.sleep(0.01)
            return {"vulnerabilities": ["test_vuln"], "processed": True}
        
        # Test caching works
        result1 = expensive_scan_operation("firmware_data_1")
        start_time = time.time()
        result2 = expensive_scan_operation("firmware_data_1")  # Should be cached
        cache_time = time.time() - start_time
        
        assert result1 == result2
        assert cache_time < 0.005  # Should be much faster from cache
        
        # Test cache statistics
        cache_stats = performance_optimizer.cache.get_stats()
        assert cache_stats['l1_size'] > 0
    
    def test_resource_management_under_load(self):
        """Test resource management under high load conditions."""
        
        # Create resource pool
        def create_expensive_resource():
            time.sleep(0.01)  # Simulate expensive creation
            return {"id": time.time(), "connections": []}
        
        resource_pool = ThreadSafeResourcePool(
            factory=create_expensive_resource,
            max_size=3
        )
        
        # Simulate concurrent access
        results = []
        threads = []
        
        def worker_thread():
            try:
                with resource_pool.get_resource(timeout=2.0) as resource:
                    # Simulate work with resource
                    time.sleep(0.05)
                    results.append({"worker": threading.current_thread().ident, "success": True})
            except Exception as e:
                results.append({"worker": threading.current_thread().ident, "error": str(e)})
        
        # Start multiple workers
        for _ in range(8):
            thread = threading.Thread(target=worker_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Verify results
        successful_results = [r for r in results if r.get("success")]
        assert len(successful_results) >= 6  # Most should succeed
        
        # Check resource pool maintained constraints
        assert resource_pool.size() <= 3
    
    def test_fault_tolerance_and_recovery(self):
        """Test system fault tolerance and recovery."""
        
        # Test circuit breaker with recovery
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        
        failure_count = 0
        
        def unreliable_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Simulated failure")
            return "success_after_recovery"
        
        # Cause failures to trip circuit breaker
        for _ in range(3):
            try:
                breaker.call(unreliable_operation)
            except Exception:
                pass
        
        # Circuit should be open
        assert breaker.state == "OPEN"
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Should allow retry and succeed
        result = breaker.call(unreliable_operation)
        assert result == "success_after_recovery"
        assert breaker.state == "CLOSED"


if __name__ == "__main__":
    # Run specific test groups
    import subprocess
    
    print("Running Generation 3 Integration Tests...")
    
    # Run the tests
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", 
        "--tb=short", "--disable-warnings"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print("✅ All Generation 3 integration tests passed!")
    else:
        print("❌ Some tests failed.")
        print(f"Exit code: {result.returncode}")