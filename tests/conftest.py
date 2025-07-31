"""Shared test configuration and fixtures for PQC IoT Retrofit Scanner."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
from unittest.mock import Mock, patch


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_firmware_dir(test_data_dir: Path) -> Path:
    """Return path to sample firmware directory."""
    return test_data_dir / "firmware"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_firmware_binary() -> bytes:
    """Return mock firmware binary data for testing."""
    # Simple ARM Cortex-M binary header pattern
    header = b"\x00\x00\x00\x20"  # Initial stack pointer
    header += b"\x01\x00\x00\x08"  # Reset vector
    # Add some padding and mock code
    mock_code = b"\x00\xBF" * 100  # NOP instructions
    return header + mock_code


@pytest.fixture
def sample_vulnerability_report() -> Dict[str, Any]:
    """Return sample vulnerability report for testing."""
    return {
        "vulnerabilities": [
            {
                "id": "vuln_001",
                "algorithm": "RSA-2048",
                "function_name": "rsa_sign",
                "address": 0x08001000,
                "severity": "high",
                "quantum_vulnerable": True,
                "stack_usage": 2048,
                "available_stack": 8192
            },
            {
                "id": "vuln_002", 
                "algorithm": "ECDSA-P256",
                "function_name": "ecdsa_verify",
                "address": 0x08002000,
                "severity": "medium",
                "quantum_vulnerable": True,
                "stack_usage": 1024,
                "available_stack": 8192
            }
        ],
        "metadata": {
            "architecture": "cortex-m4",
            "firmware_size": 262144,
            "scan_time": "2025-01-31T12:00:00Z"
        }
    }


@pytest.fixture
def mock_scanner():
    """Return mock FirmwareScanner instance."""
    from pqc_iot_retrofit.scanner import FirmwareScanner
    
    scanner = Mock(spec=FirmwareScanner)
    scanner.architecture = "cortex-m4"
    scanner.memory_constraints = {"flash": 512*1024, "ram": 128*1024}
    return scanner


@pytest.fixture
def mock_patcher():
    """Return mock PQCPatcher instance."""
    from pqc_iot_retrofit.patcher import PQCPatcher
    
    patcher = Mock(spec=PQCPatcher)
    patcher.target_device = "STM32L4"
    patcher.optimization_level = "size"
    return patcher


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment with proper logging and paths."""
    # Set up test environment variables
    os.environ["PQC_TEST_MODE"] = "1"
    os.environ["PQC_LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup
    os.environ.pop("PQC_TEST_MODE", None)
    os.environ.pop("PQC_LOG_LEVEL", None)


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "iterations": 1000,
        "timeout": 30,
        "memory_limit": 128 * 1024 * 1024,  # 128MB
        "min_rounds": 5
    }


# Hardware testing fixtures (for hardware-in-loop tests)
@pytest.fixture
def mock_hardware_interface():
    """Mock hardware interface for testing without real hardware."""
    interface = Mock()
    interface.is_connected.return_value = False
    interface.flash_firmware.return_value = True
    interface.read_memory.return_value = b"\x00" * 1024
    interface.get_device_info.return_value = {
        "device_type": "STM32L4",
        "flash_size": 512 * 1024,
        "ram_size": 128 * 1024
    }
    return interface


# Cryptographic testing fixtures
@pytest.fixture
def test_keypairs():
    """Generate test key pairs for crypto testing."""
    return {
        "rsa": {
            "public_key": "mock_rsa_public_key",
            "private_key": "mock_rsa_private_key",
            "key_size": 2048
        },
        "ecdsa": {
            "public_key": "mock_ecdsa_public_key", 
            "private_key": "mock_ecdsa_private_key",
            "curve": "P-256"
        },
        "dilithium": {
            "public_key": "mock_dilithium_public_key",
            "private_key": "mock_dilithium_private_key",
            "level": 2
        },
        "kyber": {
            "public_key": "mock_kyber_public_key",
            "private_key": "mock_kyber_private_key", 
            "level": 1
        }
    }


# Skip markers for different test environments
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_hardware: Test requires physical hardware")
    config.addinivalue_line("markers", "requires_network: Test requires network access")
    config.addinivalue_line("markers", "slow: Test is slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to apply markers based on test names."""
    for item in items:
        # Mark hardware tests
        if "hardware" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_hardware)
        
        # Mark network tests
        if "network" in item.nodeid.lower() or "download" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_network)
        
        # Mark slow tests
        if "benchmark" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)