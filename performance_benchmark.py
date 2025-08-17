#!/usr/bin/env python3
"""Performance benchmark for Generation 3 optimizations."""

import time
import sys
import os
sys.path.insert(0, 'src')

from pqc_iot_retrofit import FirmwareScanner
from pqc_iot_retrofit.performance import performance_optimizer, cached_result
from pqc_iot_retrofit.concurrency import WorkItem, WorkerPool

def benchmark_scanning_performance():
    """Benchmark firmware scanning with Generation 3 optimizations."""
    
    print("🚀 PQC IoT Retrofit Scanner - Generation 3 Performance Benchmark")
    print("=" * 60)
    
    # Create test firmware data
    test_firmware = b'\x7fELF' + b'\x00' * 1024  # Small test ELF
    test_firmware_path = "/tmp/test_firmware.bin"
    
    with open(test_firmware_path, 'wb') as f:
        f.write(test_firmware)
    
    try:
        # Test 1: Basic scanning
        print("\n📊 Test 1: Basic Firmware Scanning")
        scanner = FirmwareScanner(architecture="cortex-m4")
        
        start_time = time.time()
        for i in range(5):
            vulnerabilities = scanner.scan_firmware(test_firmware_path)
            print(f"  Scan {i+1}: {len(vulnerabilities)} vulnerabilities found")
        
        basic_duration = time.time() - start_time
        print(f"  Total time: {basic_duration:.2f}s")
        print(f"  Average per scan: {basic_duration/5:.2f}s")
        
        # Test 2: Cached scanning
        print("\n⚡ Test 2: Cached Scanning (Generation 3)")
        
        @cached_result(ttl=300)  # 5 minute cache
        def cached_scan(filepath, arch):
            scanner = FirmwareScanner(architecture=arch)
            return scanner.scan_firmware(filepath)
        
        start_time = time.time()
        for i in range(5):
            vulnerabilities = cached_scan(test_firmware_path, "cortex-m4")
            print(f"  Cached scan {i+1}: {len(vulnerabilities)} vulnerabilities found")
        
        cached_duration = time.time() - start_time
        print(f"  Total time: {cached_duration:.2f}s")
        print(f"  Average per scan: {cached_duration/5:.2f}s")
        print(f"  Speedup: {basic_duration/cached_duration:.1f}x faster")
        
        # Test 3: Performance stats
        print("\n📈 Test 3: Performance Statistics")
        cache_stats = performance_optimizer.cache.get_stats()
        print(f"  L1 cache hit rate: {cache_stats['l1_hit_rate']:.1%}")
        print(f"  L2 cache hit rate: {cache_stats['l2_hit_rate']:.1%}")
        print(f"  Memory usage: {cache_stats['memory_usage_mb']:.1f} MB")
        print(f"  Total evictions: {cache_stats['total_evictions']}")
        
        print("\n✅ Generation 3 performance benchmark completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(test_firmware_path):
            os.remove(test_firmware_path)

if __name__ == "__main__":
    success = benchmark_scanning_performance()
    sys.exit(0 if success else 1)