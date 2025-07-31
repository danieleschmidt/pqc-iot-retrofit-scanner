"""Unit tests for FirmwareScanner class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pqc_iot_retrofit.scanner import FirmwareScanner


class TestFirmwareScanner:
    """Unit tests for FirmwareScanner class."""

    def test_scanner_initialization(self):
        """Test scanner initialization with different parameters."""
        # Test basic initialization
        scanner = FirmwareScanner("cortex-m4")
        assert scanner.architecture == "cortex-m4"
        assert scanner.memory_constraints == {}

        # Test with memory constraints
        constraints = {"flash": 512*1024, "ram": 128*1024}
        scanner = FirmwareScanner("esp32", constraints)
        assert scanner.architecture == "esp32"
        assert scanner.memory_constraints == constraints

    def test_scan_firmware_empty_file(self, temp_dir):
        """Test scanning empty firmware file."""
        scanner = FirmwareScanner("cortex-m4")
        empty_file = temp_dir / "empty.bin"
        empty_file.write_bytes(b"")
        
        vulnerabilities = scanner.scan_firmware(str(empty_file))
        assert isinstance(vulnerabilities, list)
        assert len(vulnerabilities) == 0

    def test_scan_firmware_with_base_address(self, mock_firmware_binary, temp_dir):
        """Test scanning firmware with custom base address."""
        scanner = FirmwareScanner("cortex-m4")
        firmware_file = temp_dir / "test.bin"
        firmware_file.write_bytes(mock_firmware_binary)
        
        vulnerabilities = scanner.scan_firmware(str(firmware_file), base_address=0x08000000)
        assert isinstance(vulnerabilities, list)

    @pytest.mark.parametrize("architecture", [
        "cortex-m0", "cortex-m3", "cortex-m4", "cortex-m7",
        "esp32", "esp8266", "risc-v", "avr"
    ])
    def test_supported_architectures(self, architecture):
        """Test scanner initialization with supported architectures."""
        scanner = FirmwareScanner(architecture)
        assert scanner.architecture == architecture

    def test_memory_constraints_validation(self):
        """Test memory constraints validation."""
        # Valid constraints
        valid_constraints = {"flash": 1024*1024, "ram": 256*1024}
        scanner = FirmwareScanner("cortex-m4", valid_constraints)
        assert scanner.memory_constraints == valid_constraints

        # Empty constraints should default to empty dict
        scanner = FirmwareScanner("cortex-m4", None)
        assert scanner.memory_constraints == {}

    @patch('pqc_iot_retrofit.scanner.Path')
    def test_nonexistent_firmware_file(self, mock_path):
        """Test handling of nonexistent firmware file."""
        mock_path.return_value.exists.return_value = False
        
        scanner = FirmwareScanner("cortex-m4")
        
        # Should handle gracefully and return empty list
        vulnerabilities = scanner.scan_firmware("nonexistent.bin")
        assert vulnerabilities == []

    def test_scan_firmware_return_type(self, mock_firmware_binary, temp_dir):
        """Test that scan_firmware returns correct type."""
        scanner = FirmwareScanner("cortex-m4")
        firmware_file = temp_dir / "test.bin"
        firmware_file.write_bytes(mock_firmware_binary)
        
        result = scanner.scan_firmware(str(firmware_file))
        
        # Should return list of dictionaries
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict)

    def test_scanner_string_representation(self):
        """Test string representation of scanner."""
        scanner = FirmwareScanner("cortex-m4", {"flash": 512*1024})
        
        # Should contain architecture information
        str_repr = str(scanner)
        assert "cortex-m4" in str_repr.lower()

    @pytest.mark.unit
    def test_scanner_is_properly_isolated(self):
        """Test that scanner unit tests are properly isolated."""
        scanner1 = FirmwareScanner("cortex-m4")
        scanner2 = FirmwareScanner("esp32")
        
        # Each instance should be independent
        assert scanner1.architecture != scanner2.architecture
        
        scanner1.memory_constraints = {"flash": 1024}
        assert scanner2.memory_constraints == {}


class TestScannerEdgeCases:
    """Test edge cases for FirmwareScanner."""

    def test_large_firmware_file(self, temp_dir):
        """Test scanning very large firmware file."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Create large firmware file (1MB)
        large_firmware = temp_dir / "large.bin"
        large_data = b"\x00" * (1024 * 1024)
        large_firmware.write_bytes(large_data)
        
        vulnerabilities = scanner.scan_firmware(str(large_firmware))
        assert isinstance(vulnerabilities, list)

    def test_binary_with_unusual_patterns(self, temp_dir):
        """Test scanning firmware with unusual binary patterns."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Create firmware with unusual patterns
        unusual_data = b"\xFF" * 100 + b"\x00" * 100 + b"\xAA\x55" * 50
        firmware_file = temp_dir / "unusual.bin"
        firmware_file.write_bytes(unusual_data)
        
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        assert isinstance(vulnerabilities, list)

    def test_memory_constraints_edge_values(self):
        """Test memory constraints with edge values."""
        # Minimum values
        scanner = FirmwareScanner("cortex-m0", {"flash": 1024, "ram": 512})
        assert scanner.memory_constraints["flash"] == 1024
        assert scanner.memory_constraints["ram"] == 512
        
        # Large values
        scanner = FirmwareScanner("cortex-m7", {"flash": 16*1024*1024, "ram": 1024*1024})
        assert scanner.memory_constraints["flash"] == 16*1024*1024
        assert scanner.memory_constraints["ram"] == 1024*1024

    def test_unicode_filename_handling(self, temp_dir, mock_firmware_binary):
        """Test handling of Unicode filenames."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Create file with Unicode name
        unicode_file = temp_dir / "测试固件.bin"
        unicode_file.write_bytes(mock_firmware_binary)
        
        vulnerabilities = scanner.scan_firmware(str(unicode_file))
        assert isinstance(vulnerabilities, list)