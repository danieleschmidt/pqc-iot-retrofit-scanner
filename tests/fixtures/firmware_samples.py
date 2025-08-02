"""
Firmware sample fixtures for testing.

These are minimal, safe firmware samples for testing the analysis engine.
All samples are synthetic and contain no real sensitive data.
"""

import os
from pathlib import Path
from typing import Dict, List, NamedTuple

import pytest


class FirmwareSample(NamedTuple):
    """Represents a firmware sample for testing."""
    
    name: str
    architecture: str
    data: bytes
    expected_vulns: List[str]
    description: str


# ARM Cortex-M4 sample with RSA signature verification
ARM_RSA_SAMPLE = bytes([
    # ELF header (simplified)
    0x7F, 0x45, 0x4C, 0x46,  # ELF magic
    0x01, 0x01, 0x01, 0x00,  # 32-bit, little-endian, version 1
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  # padding
    0x02, 0x00, 0x28, 0x00,  # executable, ARM
    0x01, 0x00, 0x00, 0x00,  # version 1
    0x00, 0x80, 0x00, 0x08,  # entry point (0x08008000)
    # ... (truncated for brevity)
    # RSA signature pattern (mock)
    0x30, 0x82, 0x01, 0x22,  # ASN.1 SEQUENCE for RSA signature
    0x30, 0x0D, 0x06, 0x09,  # AlgorithmIdentifier
    0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B,  # RSA with SHA-256
    0x05, 0x00,  # NULL parameters
    # Mock signature bytes
    *([0xAA, 0xBB, 0xCC, 0xDD] * 64)  # 256 bytes of mock signature
])

# ESP32 sample with ECDSA key exchange
ESP32_ECDSA_SAMPLE = bytes([
    # ESP32 firmware header
    0xE9, 0x00, 0x00, 0x02,  # Magic number, segments, SPI mode, flash size
    0x00, 0x10, 0x00, 0x00,  # Entry point
    0x00, 0x00, 0x00, 0x00,  # Reserved
    0x01,  # Number of segments
    # Segment data with ECDSA patterns
    0x30, 0x45,  # ASN.1 SEQUENCE for ECDSA signature
    0x02, 0x21,  # INTEGER r (33 bytes including sign byte)
    0x00,  # Sign byte
    *([0x12, 0x34, 0x56, 0x78] * 8),  # Mock r value
    0x02, 0x20,  # INTEGER s (32 bytes)
    *([0x9A, 0xBC, 0xDE, 0xF0] * 8),  # Mock s value
    # P-256 curve parameters (mock)
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC,
])


# Collection of all firmware samples
FIRMWARE_SAMPLES = {
    "arm_cortex_m4_rsa": FirmwareSample(
        name="arm_cortex_m4_rsa",
        architecture="arm",
        data=ARM_RSA_SAMPLE,
        expected_vulns=["RSA-2048", "SHA-256"],
        description="ARM Cortex-M4 firmware with RSA-2048 signature verification"
    ),
    "esp32_ecdsa": FirmwareSample(
        name="esp32_ecdsa",
        architecture="esp32",
        data=ESP32_ECDSA_SAMPLE,
        expected_vulns=["ECDSA-P256"],
        description="ESP32 firmware with ECDSA P-256 key exchange"
    ),
}


@pytest.fixture(scope="session")
def firmware_samples_dir(tmp_path_factory):
    """Create a temporary directory with firmware samples."""
    samples_dir = tmp_path_factory.mktemp("firmware_samples")
    
    for sample_name, sample in FIRMWARE_SAMPLES.items():
        sample_file = samples_dir / f"{sample_name}.bin"
        sample_file.write_bytes(sample.data)
    
    return samples_dir


@pytest.fixture(scope="session")
def firmware_samples():
    """Provide access to firmware samples metadata."""
    return FIRMWARE_SAMPLES


@pytest.fixture
def arm_rsa_sample():
    """ARM Cortex-M4 sample with RSA vulnerability."""
    return FIRMWARE_SAMPLES["arm_cortex_m4_rsa"]


@pytest.fixture
def esp32_ecdsa_sample():
    """ESP32 sample with ECDSA vulnerability."""
    return FIRMWARE_SAMPLES["esp32_ecdsa"]


def create_test_firmware(
    architecture: str,
    crypto_algorithms: List[str],
    size_kb: int = 64
) -> bytes:
    """
    Create a synthetic firmware sample for testing.
    
    Args:
        architecture: Target architecture (arm, esp32, riscv, etc.)
        crypto_algorithms: List of crypto algorithms to embed
        size_kb: Size of firmware in KB
    
    Returns:
        Synthetic firmware bytes
    """
    # Start with basic architecture-specific header
    if architecture == "arm":
        firmware = bytearray([0x7F, 0x45, 0x4C, 0x46])  # ELF magic
    elif architecture == "esp32":
        firmware = bytearray([0xE9, 0x00, 0x00, 0x02])  # ESP32 magic
    else:
        firmware = bytearray([0x00, 0x00, 0x00, 0x00])  # Generic header
    
    # Add crypto algorithm signatures
    for algo in crypto_algorithms:
        if algo == "RSA-2048":
            firmware.extend([0x30, 0x82, 0x01, 0x22])  # RSA ASN.1 signature
        elif algo == "ECDSA-P256":
            firmware.extend([0x30, 0x45, 0x02, 0x21])  # ECDSA ASN.1 signature
        elif algo == "AES-128":
            firmware.extend([0x2B, 0x7E, 0x15, 0x16])  # AES S-box pattern
    
    # Pad to requested size
    target_size = size_kb * 1024
    if len(firmware) < target_size:
        padding = target_size - len(firmware)
        firmware.extend([0x00] * padding)
    
    return bytes(firmware)


def get_test_vector_data() -> Dict[str, bytes]:
    """
    Provide cryptographic test vectors for algorithm testing.
    
    Returns:
        Dictionary mapping algorithm names to test vector data
    """
    return {
        "dilithium2_test_vectors": bytes([
            # Dilithium2 test vectors (simplified)
            0x00, 0x01, 0x02, 0x03,  # Test case 1
            0x04, 0x05, 0x06, 0x07,  # Expected signature
        ]),
        "kyber512_test_vectors": bytes([
            # Kyber512 test vectors (simplified)
            0x10, 0x11, 0x12, 0x13,  # Test case 1
            0x14, 0x15, 0x16, 0x17,  # Expected ciphertext
        ]),
    }


@pytest.fixture
def test_vectors():
    """Provide cryptographic test vectors."""
    return get_test_vector_data()


@pytest.fixture
def mock_vulnerable_firmware():
    """Create a firmware sample with multiple vulnerabilities."""
    return create_test_firmware(
        architecture="arm",
        crypto_algorithms=["RSA-2048", "ECDSA-P256", "AES-128"],
        size_kb=128
    )


@pytest.fixture
def mock_secure_firmware():
    """Create a firmware sample with no known vulnerabilities."""
    return create_test_firmware(
        architecture="arm",
        crypto_algorithms=[],  # No vulnerable crypto
        size_kb=64
    )