"""Tests for firmware scanner module."""

import pytest
from pqc_iot_retrofit.scanner import FirmwareScanner


def test_scanner_initialization():
    """Test scanner initialization."""
    scanner = FirmwareScanner("cortex-m4")
    assert scanner.architecture == "cortex-m4"
    assert scanner.memory_constraints == {}


def test_scanner_with_constraints():
    """Test scanner initialization with memory constraints."""
    constraints = {"flash": 512 * 1024, "ram": 64 * 1024}
    scanner = FirmwareScanner("cortex-m4", constraints)
    assert scanner.memory_constraints == constraints


def test_scan_firmware_placeholder():
    """Test firmware scanning placeholder."""
    scanner = FirmwareScanner("cortex-m4")
    result = scanner.scan_firmware("dummy_path")
    assert isinstance(result, list)
    assert len(result) == 0