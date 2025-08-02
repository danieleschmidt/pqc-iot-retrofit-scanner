"""
Integration tests for firmware analysis pipeline.

Tests the complete firmware analysis workflow from binary input to vulnerability report.
"""

import json
import tempfile
from pathlib import Path

import pytest

from pqc_iot_retrofit.scanner import FirmwareScanner
from tests.fixtures.firmware_samples import FIRMWARE_SAMPLES


class TestFirmwareAnalysisPipeline:
    """Test the complete firmware analysis pipeline."""
    
    def test_arm_cortex_m4_analysis(self, arm_rsa_sample, tmp_path):
        """Test analysis of ARM Cortex-M4 firmware with RSA vulnerability."""
        # Create temporary firmware file
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        # Initialize scanner
        scanner = FirmwareScanner(
            architecture="arm",
            memory_constraints={"flash": 512*1024, "ram": 64*1024}
        )
        
        # Analyze firmware
        vulnerabilities = scanner.scan_firmware(
            firmware_path=str(firmware_file),
            base_address=0x08000000
        )
        
        # Verify results
        assert len(vulnerabilities) > 0, "Should detect vulnerabilities"
        
        # Check for RSA vulnerability
        rsa_vulns = [v for v in vulnerabilities if "RSA" in v.algorithm]
        assert len(rsa_vulns) > 0, "Should detect RSA vulnerability"
        
        # Verify vulnerability details
        rsa_vuln = rsa_vulns[0]
        assert rsa_vuln.confidence > 0.8, "Should have high confidence"
        assert rsa_vuln.severity in ["high", "critical"], "RSA should be high/critical severity"
    
    def test_esp32_ecdsa_analysis(self, esp32_ecdsa_sample, tmp_path):
        """Test analysis of ESP32 firmware with ECDSA vulnerability."""
        firmware_file = tmp_path / "esp32_firmware.bin"
        firmware_file.write_bytes(esp32_ecdsa_sample.data)
        
        scanner = FirmwareScanner(
            architecture="esp32",
            memory_constraints={"flash": 4*1024*1024, "ram": 520*1024}
        )
        
        vulnerabilities = scanner.scan_firmware(
            firmware_path=str(firmware_file),
            base_address=0x400D0000
        )
        
        assert len(vulnerabilities) > 0, "Should detect vulnerabilities"
        
        # Check for ECDSA vulnerability
        ecdsa_vulns = [v for v in vulnerabilities if "ECDSA" in v.algorithm]
        assert len(ecdsa_vulns) > 0, "Should detect ECDSA vulnerability"
    
    @pytest.mark.parametrize("sample_name", FIRMWARE_SAMPLES.keys())
    def test_all_firmware_samples(self, sample_name, firmware_samples, tmp_path):
        """Test analysis of all firmware samples."""
        sample = firmware_samples[sample_name]
        firmware_file = tmp_path / f"{sample_name}.bin"
        firmware_file.write_bytes(sample.data)
        
        scanner = FirmwareScanner(architecture=sample.architecture)
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        
        # Verify expected vulnerabilities are detected
        detected_algorithms = {v.algorithm for v in vulnerabilities}
        for expected_vuln in sample.expected_vulns:
            assert any(expected_vuln in algo for algo in detected_algorithms), \
                f"Expected vulnerability {expected_vuln} not detected in {sample_name}"
    
    def test_empty_firmware_handling(self, tmp_path):
        """Test handling of empty or invalid firmware files."""
        # Test empty file
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        
        scanner = FirmwareScanner(architecture="arm")
        vulnerabilities = scanner.scan_firmware(str(empty_file))
        
        # Should handle gracefully without crashing
        assert isinstance(vulnerabilities, list), "Should return list even for empty firmware"
    
    def test_large_firmware_handling(self, tmp_path):
        """Test handling of large firmware files."""
        # Create a 10MB firmware file
        large_firmware = tmp_path / "large_firmware.bin"
        large_data = b"\x00" * (10 * 1024 * 1024)  # 10MB of zeros
        large_firmware.write_bytes(large_data)
        
        scanner = FirmwareScanner(
            architecture="arm",
            analysis_timeout=30  # 30 second timeout for large files
        )
        
        # Should handle large files without memory issues
        vulnerabilities = scanner.scan_firmware(str(large_firmware))
        assert isinstance(vulnerabilities, list), "Should handle large firmware files"
    
    def test_concurrent_analysis(self, firmware_samples, tmp_path):
        """Test concurrent analysis of multiple firmware files."""
        import concurrent.futures
        
        # Create multiple firmware files
        firmware_files = []
        for sample_name, sample in firmware_samples.items():
            firmware_file = tmp_path / f"concurrent_{sample_name}.bin"
            firmware_file.write_bytes(sample.data)
            firmware_files.append((str(firmware_file), sample.architecture))
        
        def analyze_firmware(firmware_path_and_arch):
            firmware_path, arch = firmware_path_and_arch
            scanner = FirmwareScanner(architecture=arch)
            return scanner.scan_firmware(firmware_path)
        
        # Analyze firmware files concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(analyze_firmware, firmware_files))
        
        # Verify all analyses completed successfully
        assert len(results) == len(firmware_files), "All concurrent analyses should complete"
        for result in results:
            assert isinstance(result, list), "Each analysis should return a list"


class TestAnalysisAccuracy:
    """Test the accuracy and reliability of vulnerability detection."""
    
    def test_false_positive_rate(self, mock_secure_firmware, tmp_path):
        """Test that secure firmware doesn't generate false positives."""
        firmware_file = tmp_path / "secure_firmware.bin"
        firmware_file.write_bytes(mock_secure_firmware)
        
        scanner = FirmwareScanner(architecture="arm")
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        
        # Should have very few or no vulnerabilities for secure firmware
        high_confidence_vulns = [v for v in vulnerabilities if v.confidence > 0.9]
        assert len(high_confidence_vulns) == 0, \
            "Secure firmware should not have high-confidence vulnerabilities"
    
    def test_vulnerability_confidence_scores(self, mock_vulnerable_firmware, tmp_path):
        """Test that vulnerability confidence scores are reasonable."""
        firmware_file = tmp_path / "vulnerable_firmware.bin"
        firmware_file.write_bytes(mock_vulnerable_firmware)
        
        scanner = FirmwareScanner(architecture="arm")
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        
        # Check confidence scores
        for vuln in vulnerabilities:
            assert 0.0 <= vuln.confidence <= 1.0, "Confidence should be between 0 and 1"
            
            # High-severity vulnerabilities should have reasonable confidence
            if vuln.severity in ["high", "critical"]:
                assert vuln.confidence >= 0.6, \
                    f"High-severity vulnerability should have confidence >= 0.6, got {vuln.confidence}"
    
    def test_algorithm_classification_accuracy(self, firmware_samples, tmp_path):
        """Test accuracy of cryptographic algorithm classification."""
        for sample_name, sample in firmware_samples.items():
            firmware_file = tmp_path / f"classify_{sample_name}.bin"
            firmware_file.write_bytes(sample.data)
            
            scanner = FirmwareScanner(architecture=sample.architecture)
            vulnerabilities = scanner.scan_firmware(str(firmware_file))
            
            # Check that detected algorithms match expected ones
            detected_algorithms = {v.algorithm for v in vulnerabilities}
            
            for expected_algo in sample.expected_vulns:
                # Allow partial matches (e.g., "RSA" matches "RSA-2048")
                assert any(expected_algo in detected for detected in detected_algorithms), \
                    f"Expected algorithm {expected_algo} not detected in {sample_name}"


class TestAnalysisPerformance:
    """Test performance characteristics of firmware analysis."""
    
    @pytest.mark.performance
    def test_analysis_speed(self, arm_rsa_sample, tmp_path):
        """Test that analysis completes within reasonable time."""
        import time
        
        firmware_file = tmp_path / "speed_test.bin"
        firmware_file.write_bytes(arm_rsa_sample.data)
        
        scanner = FirmwareScanner(architecture="arm")
        
        start_time = time.time()
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete analysis quickly for small firmware
        assert analysis_time < 10.0, f"Analysis took {analysis_time:.2f}s, should be < 10s"
        assert len(vulnerabilities) > 0, "Should still detect vulnerabilities"
    
    @pytest.mark.performance
    def test_memory_usage(self, mock_vulnerable_firmware, tmp_path):
        """Test that analysis doesn't consume excessive memory."""
        import psutil
        import os
        
        firmware_file = tmp_path / "memory_test.bin"
        firmware_file.write_bytes(mock_vulnerable_firmware)
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        scanner = FirmwareScanner(architecture="arm")
        vulnerabilities = scanner.scan_firmware(str(firmware_file))
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (< 100MB for small firmware)
        assert memory_increase < 100 * 1024 * 1024, \
            f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB, should be < 100MB"
        assert len(vulnerabilities) >= 0, "Should complete analysis"