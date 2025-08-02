"""
Performance benchmark tests for PQC IoT Retrofit Scanner.

These tests measure and track performance metrics to detect regressions.
"""

import time
import statistics
import psutil
import os
from typing import Dict, List

import pytest
import pytest_benchmark

from pqc_iot_retrofit.scanner import FirmwareScanner
from pqc_iot_retrofit.patcher import PQCPatcher
from pqc_iot_retrofit.crypto.dilithium import Dilithium2Implementation
from pqc_iot_retrofit.crypto.kyber import Kyber512Implementation


class TestFirmwareAnalysisPerformance:
    """Benchmark firmware analysis performance."""
    
    @pytest.mark.benchmark
    def test_small_firmware_analysis_speed(self, benchmark, arm_rsa_sample, tmp_path):
        """Benchmark analysis speed for small firmware (< 1MB)."""
        firmware_file = tmp_path / "small_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        scanner = FirmwareScanner(architecture="arm")
        
        def analyze_firmware():
            return scanner.scan_firmware(str(firmware_file))
        
        result = benchmark(analyze_firmware)
        
        # Verify analysis completed successfully
        assert len(result) > 0, "Should detect vulnerabilities"
        
        # Performance expectations for small firmware
        assert benchmark.stats.mean < 5.0, \
            f"Small firmware analysis took {benchmark.stats.mean:.2f}s, should be < 5s"
    
    @pytest.mark.benchmark
    def test_medium_firmware_analysis_speed(self, benchmark, tmp_path):
        """Benchmark analysis speed for medium firmware (1-10MB)."""
        # Create a 5MB firmware file with crypto patterns
        firmware_data = b"\x00" * (5 * 1024 * 1024)  # 5MB of zeros
        # Add some RSA patterns
        firmware_data = firmware_data[:1000] + b"\x30\x82\x01\x22" * 100 + firmware_data[1400:]
        
        firmware_file = tmp_path / "medium_firmware.bin"
        firmware_file.write_bytes(firmware_data)
        
        scanner = FirmwareScanner(
            architecture="arm",
            analysis_timeout=60  # 1 minute timeout
        )
        
        def analyze_firmware():
            return scanner.scan_firmware(str(firmware_file))
        
        result = benchmark(analyze_firmware)
        
        # Performance expectations for medium firmware  
        assert benchmark.stats.mean < 30.0, \
            f"Medium firmware analysis took {benchmark.stats.mean:.2f}s, should be < 30s"
    
    @pytest.mark.benchmark
    def test_concurrent_analysis_performance(self, benchmark, firmware_samples, tmp_path):
        """Benchmark concurrent analysis of multiple firmware files."""
        import concurrent.futures
        
        # Create multiple firmware files
        firmware_files = []
        for i, (sample_name, sample) in enumerate(firmware_samples.items()):
            firmware_file = tmp_path / f"concurrent_{i}_{sample_name}.bin"
            firmware_file.write_bytes(sample.data)
            firmware_files.append((str(firmware_file), sample.architecture))
        
        def analyze_concurrent():
            def analyze_single(firmware_path_and_arch):
                firmware_path, arch = firmware_path_and_arch
                scanner = FirmwareScanner(architecture=arch)
                return scanner.scan_firmware(firmware_path)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(analyze_single, firmware_files))
            
            return results
        
        results = benchmark(analyze_concurrent)
        
        # Verify all analyses completed
        assert len(results) == len(firmware_files), "All concurrent analyses should complete"
        
        # Concurrent analysis should be faster than sequential
        expected_sequential_time = len(firmware_files) * 2.0  # Assume 2s per firmware
        assert benchmark.stats.mean < expected_sequential_time * 0.7, \
            f"Concurrent analysis should be faster than sequential"


class TestCryptographicPerformance:
    """Benchmark cryptographic operations performance."""
    
    @pytest.mark.benchmark
    def test_dilithium2_keygen_performance(self, benchmark):
        """Benchmark Dilithium2 key generation speed."""
        dilithium = Dilithium2Implementation()
        
        def generate_keypair():
            return dilithium.generate_keypair()
        
        public_key, secret_key = benchmark(generate_keypair)
        
        # Verify key generation succeeded
        assert public_key is not None and secret_key is not None
        assert len(public_key) > 0 and len(secret_key) > 0
        
        # Performance expectation: < 10ms on modern hardware
        assert benchmark.stats.mean < 0.01, \
            f"Dilithium2 key generation took {benchmark.stats.mean * 1000:.1f}ms, should be < 10ms"
    
    @pytest.mark.benchmark
    def test_dilithium2_signing_performance(self, benchmark):
        """Benchmark Dilithium2 signature generation speed."""
        dilithium = Dilithium2Implementation()
        public_key, secret_key = dilithium.generate_keypair()
        message = b"Performance test message for Dilithium2 signing" * 10  # 470 bytes
        
        def sign_message():
            return dilithium.sign(message, secret_key)
        
        signature = benchmark(sign_message)
        
        # Verify signature is valid
        assert dilithium.verify(message, signature, public_key)
        
        # Performance expectation: < 50ms for embedded systems
        assert benchmark.stats.mean < 0.05, \
            f"Dilithium2 signing took {benchmark.stats.mean * 1000:.1f}ms, should be < 50ms"
    
    @pytest.mark.benchmark
    def test_dilithium2_verification_performance(self, benchmark):
        """Benchmark Dilithium2 signature verification speed."""
        dilithium = Dilithium2Implementation()
        public_key, secret_key = dilithium.generate_keypair()
        message = b"Performance test message for Dilithium2 verification" * 10
        signature = dilithium.sign(message, secret_key)
        
        def verify_signature():
            return dilithium.verify(message, signature, public_key)
        
        is_valid = benchmark(verify_signature)
        
        # Verify signature verification succeeded
        assert is_valid is True
        
        # Performance expectation: < 20ms for embedded systems
        assert benchmark.stats.mean < 0.02, \
            f"Dilithium2 verification took {benchmark.stats.mean * 1000:.1f}ms, should be < 20ms"
    
    @pytest.mark.benchmark
    def test_kyber512_keygen_performance(self, benchmark):
        """Benchmark Kyber512 key generation speed."""
        kyber = Kyber512Implementation()
        
        def generate_keypair():
            return kyber.generate_keypair()
        
        public_key, secret_key = benchmark(generate_keypair)
        
        # Verify key generation succeeded
        assert public_key is not None and secret_key is not None
        
        # Performance expectation: < 5ms
        assert benchmark.stats.mean < 0.005, \
            f"Kyber512 key generation took {benchmark.stats.mean * 1000:.1f}ms, should be < 5ms"
    
    @pytest.mark.benchmark
    def test_kyber512_encapsulation_performance(self, benchmark):
        """Benchmark Kyber512 encapsulation speed."""
        kyber = Kyber512Implementation()
        public_key, secret_key = kyber.generate_keypair()
        
        def encapsulate():
            return kyber.encapsulate(public_key)
        
        shared_secret, ciphertext = benchmark(encapsulate)
        
        # Verify encapsulation succeeded
        assert shared_secret is not None and ciphertext is not None
        assert len(shared_secret) > 0 and len(ciphertext) > 0
        
        # Performance expectation: < 10ms
        assert benchmark.stats.mean < 0.01, \
            f"Kyber512 encapsulation took {benchmark.stats.mean * 1000:.1f}ms, should be < 10ms"
    
    @pytest.mark.benchmark
    def test_kyber512_decapsulation_performance(self, benchmark):
        """Benchmark Kyber512 decapsulation speed."""
        kyber = Kyber512Implementation()
        public_key, secret_key = kyber.generate_keypair()
        original_secret, ciphertext = kyber.encapsulate(public_key)
        
        def decapsulate():
            return kyber.decapsulate(ciphertext, secret_key)
        
        recovered_secret = benchmark(decapsulate)
        
        # Verify decapsulation succeeded and secret matches
        assert recovered_secret == original_secret
        
        # Performance expectation: < 15ms
        assert benchmark.stats.mean < 0.015, \
            f"Kyber512 decapsulation took {benchmark.stats.mean * 1000:.1f}ms, should be < 15ms"


class TestPatchGenerationPerformance:
    """Benchmark patch generation performance."""
    
    @pytest.mark.benchmark
    def test_dilithium_patch_generation_speed(self, benchmark, tmp_path):
        """Benchmark Dilithium patch generation speed."""
        # Create mock vulnerability
        from pqc_iot_retrofit.scanner import Vulnerability
        
        vulnerability = Vulnerability(
            algorithm="RSA-2048",
            function_name="rsa_verify",
            address=0x08001000,
            confidence=0.95,
            severity="high",
            description="RSA-2048 signature verification"
        )
        
        patcher = PQCPatcher(
            target_device="STM32L4",
            optimization_level="size"
        )
        
        def generate_patch():
            return patcher.create_dilithium_patch(
                vulnerability,
                security_level=2,
                stack_size=8192
            )
        
        patch = benchmark(generate_patch)
        
        # Verify patch generation succeeded
        assert patch is not None
        assert hasattr(patch, 'target_address')
        
        # Performance expectation: < 1 second
        assert benchmark.stats.mean < 1.0, \
            f"Dilithium patch generation took {benchmark.stats.mean:.2f}s, should be < 1s"
    
    @pytest.mark.benchmark
    def test_kyber_patch_generation_speed(self, benchmark):
        """Benchmark Kyber patch generation speed."""
        from pqc_iot_retrofit.scanner import Vulnerability
        
        vulnerability = Vulnerability(
            algorithm="ECDH-P256",
            function_name="ecdh_derive",
            address=0x08002000,
            confidence=0.90,
            severity="medium",
            description="ECDH P-256 key exchange"
        )
        
        patcher = PQCPatcher(
            target_device="ESP32",
            optimization_level="speed"
        )
        
        def generate_patch():
            return patcher.create_kyber_patch(
                vulnerability,
                security_level=1,
                shared_memory=True
            )
        
        patch = benchmark(generate_patch)
        
        # Verify patch generation succeeded
        assert patch is not None
        
        # Performance expectation: < 0.5 seconds
        assert benchmark.stats.mean < 0.5, \
            f"Kyber patch generation took {benchmark.stats.mean:.2f}s, should be < 0.5s"


class TestMemoryUsageProfile:
    """Profile memory usage patterns."""
    
    @pytest.mark.benchmark
    def test_firmware_analysis_memory_usage(self, tmp_path):
        """Profile memory usage during firmware analysis."""
        # Create a moderately sized firmware (1MB)
        firmware_data = b"\x00" * (1024 * 1024)
        firmware_file = tmp_path / "memory_test_firmware.bin"
        firmware_file.write_bytes(firmware_data)
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss
        
        scanner = FirmwareScanner(architecture="arm")
        
        # Measure memory during analysis
        peak_memory = baseline_memory
        
        def memory_monitor():
            nonlocal peak_memory
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
        
        # Start analysis
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        memory_monitor()
        
        # Calculate memory usage
        memory_increase = peak_memory - baseline_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory usage should be reasonable
        assert memory_increase_mb < 500, \
            f"Memory usage increased by {memory_increase_mb:.1f}MB, should be < 500MB"
        
        # Verify analysis completed
        assert isinstance(vulnerabilities, list)
    
    @pytest.mark.benchmark
    def test_crypto_operations_memory_efficiency(self):
        """Test memory efficiency of cryptographic operations."""
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss
        
        # Perform multiple crypto operations
        dilithium = Dilithium2Implementation()
        kyber = Kyber512Implementation()
        
        peak_memory = baseline_memory
        
        for i in range(10):
            # Dilithium operations
            pub_d, sec_d = dilithium.generate_keypair()
            message = f"test message {i}".encode()
            signature = dilithium.sign(message, sec_d)
            is_valid = dilithium.verify(message, signature, pub_d)
            assert is_valid
            
            # Kyber operations  
            pub_k, sec_k = kyber.generate_keypair()
            shared_secret, ciphertext = kyber.encapsulate(pub_k)
            recovered_secret = kyber.decapsulate(ciphertext, sec_k)
            assert shared_secret == recovered_secret
            
            # Monitor peak memory
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)
        
        memory_increase = peak_memory - baseline_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory increase should be minimal for crypto operations
        assert memory_increase_mb < 50, \
            f"Crypto operations increased memory by {memory_increase_mb:.1f}MB, should be < 50MB"


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Test scalability with increasing workloads."""
    
    def test_analysis_time_scaling(self, tmp_path):
        """Test how analysis time scales with firmware size."""
        sizes_kb = [64, 128, 256, 512, 1024]  # KB
        analysis_times = []
        
        for size_kb in sizes_kb:
            # Create firmware of specified size
            firmware_data = b"\x00" * (size_kb * 1024)
            # Add some crypto patterns
            crypto_pattern = b"\x30\x82\x01\x22" * (size_kb // 4)
            firmware_data = firmware_data[:len(crypto_pattern)] + crypto_pattern + \
                           firmware_data[len(crypto_pattern) * 2:]
            
            firmware_file = tmp_path / f"scale_test_{size_kb}kb.bin"
            firmware_file.write_bytes(firmware_data)
            
            scanner = FirmwareScanner(architecture="arm")
            
            # Measure analysis time
            start_time = time.time()
            vulnerabilities = scanner.scan_firmware(str(firmware_file))
            end_time = time.time()
            
            analysis_time = end_time - start_time
            analysis_times.append(analysis_time)
            
            # Clean up
            firmware_file.unlink()
        
        # Analysis time should scale reasonably (not exponentially)
        # Check that time doesn't increase too dramatically
        for i in range(1, len(analysis_times)):
            size_ratio = sizes_kb[i] / sizes_kb[i-1]
            time_ratio = analysis_times[i] / analysis_times[i-1]
            
            # Time should not increase more than 3x when size doubles
            assert time_ratio < size_ratio * 1.5, \
                f"Analysis time scaling is poor: {size_ratio}x size increase " \
                f"caused {time_ratio:.1f}x time increase"
    
    def test_concurrent_analysis_scaling(self, firmware_samples, tmp_path):
        """Test performance scaling with concurrent analyses."""
        import concurrent.futures
        
        # Create multiple copies of firmware samples
        firmware_files = []
        for i in range(8):  # 8 concurrent analyses
            for sample_name, sample in firmware_samples.items():
                firmware_file = tmp_path / f"concurrent_scale_{i}_{sample_name}.bin"
                firmware_file.write_bytes(sample.data)
                firmware_files.append((str(firmware_file), sample.architecture))
        
        def analyze_single(firmware_path_and_arch):
            firmware_path, arch = firmware_path_and_arch
            scanner = FirmwareScanner(architecture=arch)
            start_time = time.time()
            result = scanner.scan_firmware(firmware_path)
            end_time = time.time()
            return end_time - start_time, len(result)
        
        # Test with different numbers of workers
        worker_counts = [1, 2, 4, 8]
        total_times = []
        
        for worker_count in worker_counts:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(analyze_single, firmware_files))
            
            end_time = time.time()
            total_time = end_time - start_time
            total_times.append(total_time)
        
        # Concurrent execution should provide speedup
        sequential_time = total_times[0]  # 1 worker
        parallel_time = total_times[-1]   # Max workers
        
        speedup = sequential_time / parallel_time
        
        # Should see some speedup with parallel execution
        assert speedup > 1.5, \
            f"Parallel execution speedup is only {speedup:.1f}x, expected > 1.5x"