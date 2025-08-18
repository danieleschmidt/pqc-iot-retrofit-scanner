"""
Common testing utilities and helper functions.
"""

import os
import time
import contextlib
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import tempfile
import hashlib


class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        
    def time_operation(self, name: str):
        """Context manager to time an operation."""
        @contextlib.contextmanager
        def timer():
            start_time = time.perf_counter()
            try:
                yield
            finally:
                end_time = time.perf_counter()
                self.metrics[name] = end_time - start_time
        return timer()
    
    def get_metric(self, name: str) -> Optional[float]:
        """Get a recorded metric."""
        return self.metrics.get(name)
    
    def assert_performance(self, name: str, max_time: float):
        """Assert that an operation completed within time limit."""
        actual_time = self.metrics.get(name)
        if actual_time is None:
            raise ValueError(f"No timing data recorded for operation '{name}'")
        
        assert actual_time <= max_time, \
            f"Operation '{name}' took {actual_time:.3f}s, expected <= {max_time:.3f}s"


class TemporaryFirmware:
    """Create temporary firmware files for testing."""
    
    def __init__(self, 
                 content: bytes, 
                 filename: str = "test_firmware.bin",
                 cleanup: bool = True):
        self.content = content
        self.filename = filename
        self.cleanup = cleanup
        self._temp_dir = None
        self._file_path = None
    
    def __enter__(self) -> Path:
        self._temp_dir = tempfile.TemporaryDirectory()
        self._file_path = Path(self._temp_dir.name) / self.filename
        self._file_path.write_bytes(self.content)
        return self._file_path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup and self._temp_dir:
            self._temp_dir.cleanup()


def create_mock_firmware(
    architecture: str = "arm",
    size_bytes: int = 4096,
    crypto_patterns: Optional[List[str]] = None,
    add_header: bool = True
) -> bytes:
    """
    Create mock firmware data for testing.
    
    Args:
        architecture: Target architecture (arm, esp32, riscv, etc.)
        size_bytes: Size of firmware in bytes
        crypto_patterns: List of crypto patterns to embed
        add_header: Whether to add architecture-specific header
    
    Returns:
        Mock firmware bytes
    """
    if crypto_patterns is None:
        crypto_patterns = []
    
    firmware = bytearray()
    
    # Add architecture-specific header
    if add_header:
        if architecture == "arm":
            firmware.extend([0x7F, 0x45, 0x4C, 0x46])  # ELF magic
        elif architecture == "esp32":
            firmware.extend([0xE9, 0x00, 0x00, 0x02])  # ESP32 magic
        else:
            firmware.extend([0x00, 0x00, 0x00, 0x00])  # Generic header
    
    # Add crypto patterns
    for pattern in crypto_patterns:
        if pattern == "RSA-2048":
            firmware.extend([0x30, 0x82, 0x01, 0x22])  # RSA ASN.1
        elif pattern == "ECDSA-P256":
            firmware.extend([0x30, 0x45, 0x02, 0x21])  # ECDSA ASN.1
        elif pattern == "AES-128":
            firmware.extend([0x2B, 0x7E, 0x15, 0x16])  # AES S-box
        elif pattern == "SHA-256":
            firmware.extend([0x6A, 0x09, 0xE6, 0x67])  # SHA-256 constants
    
    # Pad to requested size
    current_size = len(firmware)
    if current_size < size_bytes:
        padding_size = size_bytes - current_size
        # Add some variety to padding
        for i in range(padding_size):
            firmware.append((i * 7) % 256)
    elif current_size > size_bytes:
        firmware = firmware[:size_bytes]
    
    return bytes(firmware)


def verify_test_environment():
    """Verify that the test environment is properly set up."""
    # Check required environment variables
    required_vars = ["PQC_TEST_MODE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise RuntimeError(
            f"Missing required test environment variables: {missing_vars}"
        )
    
    # Check if we're in test mode
    if os.getenv("PQC_TEST_MODE") != "1":
        raise RuntimeError("Tests must be run with PQC_TEST_MODE=1")


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def assert_firmware_integrity(
    firmware_path: Union[str, Path],
    expected_size: Optional[int] = None,
    expected_hash: Optional[str] = None,
    hash_algorithm: str = "sha256"
):
    """Assert firmware file integrity."""
    firmware_path = Path(firmware_path)
    
    assert firmware_path.exists(), f"Firmware file does not exist: {firmware_path}"
    assert firmware_path.is_file(), f"Path is not a file: {firmware_path}"
    
    if expected_size is not None:
        actual_size = firmware_path.stat().st_size
        assert actual_size == expected_size, \
            f"Firmware size mismatch: expected {expected_size}, got {actual_size}"
    
    if expected_hash is not None:
        actual_hash = calculate_file_hash(firmware_path, hash_algorithm)
        assert actual_hash == expected_hash, \
            f"Firmware hash mismatch: expected {expected_hash}, got {actual_hash}"


class MockHardwareInterface:
    """Mock hardware interface for testing without real hardware."""
    
    def __init__(self, 
                 device_type: str = "STM32L4",
                 flash_size: int = 512 * 1024,
                 ram_size: int = 128 * 1024):
        self.device_type = device_type
        self.flash_size = flash_size
        self.ram_size = ram_size
        self.connected = False
        self.memory_contents = {}
        
    def connect(self) -> bool:
        """Simulate hardware connection."""
        self.connected = True
        return True
    
    def disconnect(self):
        """Simulate hardware disconnection."""
        self.connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to hardware."""
        return self.connected
    
    def flash_firmware(self, firmware_data: bytes, address: int = 0) -> bool:
        """Simulate firmware flashing."""
        if not self.connected:
            return False
        
        if len(firmware_data) > self.flash_size:
            return False
        
        self.memory_contents[address] = firmware_data
        return True
    
    def read_memory(self, address: int, size: int) -> bytes:
        """Simulate memory reading."""
        if not self.connected:
            raise RuntimeError("Not connected to hardware")
        
        if address in self.memory_contents:
            data = self.memory_contents[address]
            return data[:size] if len(data) >= size else data + b"\x00" * (size - len(data))
        
        return b"\x00" * size
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information."""
        return {
            "device_type": self.device_type,
            "flash_size": self.flash_size,
            "ram_size": self.ram_size,
            "connected": self.connected
        }


def skip_if_no_hardware():
    """Skip test if no hardware is available."""
    import pytest
    
    has_hardware = os.getenv("PQC_ENABLE_HARDWARE_TESTS") == "1"
    return pytest.mark.skipif(
        not has_hardware,
        reason="Hardware tests disabled (set PQC_ENABLE_HARDWARE_TESTS=1 to enable)"
    )


def parametrize_architectures(architectures: Optional[List[str]] = None):
    """Parametrize test with different architectures."""
    import pytest
    
    if architectures is None:
        architectures = ["arm", "esp32", "riscv", "avr"]
    
    return pytest.mark.parametrize("architecture", architectures)


def parametrize_crypto_algorithms(algorithms: Optional[List[str]] = None):
    """Parametrize test with different crypto algorithms."""
    import pytest
    
    if algorithms is None:
        algorithms = ["RSA-2048", "ECDSA-P256", "AES-128", "SHA-256"]
    
    return pytest.mark.parametrize("crypto_algorithm", algorithms)


class SecurityTestHelper:
    """Helper for security-focused tests."""
    
    @staticmethod
    def generate_timing_samples(operation_func, sample_count: int = 100) -> List[float]:
        """Generate timing samples for an operation."""
        samples = []
        
        for _ in range(sample_count):
            start_time = time.perf_counter()
            operation_func()
            end_time = time.perf_counter()
            samples.append(end_time - start_time)
        
        return samples
    
    @staticmethod
    def analyze_timing_variance(samples: List[float], 
                              max_variance_ratio: float = 0.1) -> bool:
        """Analyze timing variance to detect potential side-channel leaks."""
        import statistics
        
        if len(samples) < 2:
            return True  # Can't analyze variance with < 2 samples
        
        mean_time = statistics.mean(samples)
        std_dev = statistics.stdev(samples)
        
        variance_ratio = std_dev / mean_time if mean_time > 0 else 0
        return variance_ratio <= max_variance_ratio
    
    @staticmethod
    def assert_constant_time(operation_func, 
                           sample_count: int = 100,
                           max_variance_ratio: float = 0.1):
        """Assert that an operation runs in constant time."""
        samples = SecurityTestHelper.generate_timing_samples(operation_func, sample_count)
        is_constant_time = SecurityTestHelper.analyze_timing_variance(samples, max_variance_ratio)
        
        assert is_constant_time, \
            f"Operation timing variance {max_variance_ratio:.1%} indicates potential timing leak"


def memory_leak_detector():
    """Context manager to detect memory leaks during testing."""
    @contextlib.contextmanager
    def detector():
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Force garbage collection before measurement
        gc.collect()
        initial_memory = process.memory_info().rss
        
        try:
            yield
        finally:
            # Force garbage collection after test
            gc.collect()
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Allow for some memory increase (10MB threshold)
            max_increase = 10 * 1024 * 1024  # 10MB
            
            assert memory_increase < max_increase, \
                f"Memory leak detected: {memory_increase / 1024 / 1024:.1f}MB increase"
    
    return detector()


class FirmwareFactory:
    """Factory for creating various types of test firmware."""
    
    @staticmethod
    def create_vulnerable_firmware(vulnerability_type: str, 
                                 architecture: str = "arm",
                                 size_kb: int = 64) -> bytes:
        """Create firmware with specific vulnerabilities."""
        patterns = []
        
        if vulnerability_type == "quantum_vulnerable":
            patterns = ["RSA-2048", "ECDSA-P256"]
        elif vulnerability_type == "weak_crypto":
            patterns = ["DES", "MD5"]  # These would be different patterns
        elif vulnerability_type == "mixed":
            patterns = ["RSA-2048", "AES-128", "SHA-256"]
        
        return create_mock_firmware(
            architecture=architecture,
            size_bytes=size_kb * 1024,
            crypto_patterns=patterns
        )
    
    @staticmethod
    def create_secure_firmware(architecture: str = "arm", 
                             size_kb: int = 64) -> bytes:
        """Create firmware with no known vulnerabilities."""
        return create_mock_firmware(
            architecture=architecture,
            size_bytes=size_kb * 1024,
            crypto_patterns=[]  # No vulnerable patterns
        )
    
    @staticmethod
    def create_large_firmware(architecture: str = "arm",
                            size_mb: int = 5) -> bytes:
        """Create large firmware for performance testing."""
        return create_mock_firmware(
            architecture=architecture,
            size_bytes=size_mb * 1024 * 1024,
            crypto_patterns=["RSA-2048"] * 10  # Multiple instances
        )