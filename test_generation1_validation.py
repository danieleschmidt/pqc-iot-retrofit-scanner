"""Simple validation script for Generation 1 basic functionality.

Tests essential components without pytest dependency.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test that all Generation 1 modules import correctly."""
    print("Testing imports...")
    
    try:
        from pqc_iot_retrofit.utils import (
            ArchitectureDetector, FirmwareAnalyzer, MemoryLayoutAnalyzer,
            calculate_entropy, validate_firmware_path, format_size, format_address
        )
        print("✅ Utils module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import utils: {e}")
        return False
    
    try:
        from pqc_iot_retrofit.reporting import ReportGenerator, ReportFormat, ReportMetadata
        print("✅ Reporting module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import reporting: {e}")
        return False
    
    try:
        from pqc_iot_retrofit.cli_basic import main as cli_main
        print("✅ Basic CLI imported successfully")
    except Exception as e:
        print(f"❌ Failed to import basic CLI: {e}")
        return False
    
    return True

def test_scanner_basic():
    """Test basic scanner functionality."""
    print("\nTesting scanner functionality...")
    
    try:
        from pqc_iot_retrofit.scanner import FirmwareScanner, RiskLevel
        
        # Test scanner initialization
        scanner = FirmwareScanner("cortex-m4")
        print("✅ Scanner initialized successfully")
        
        # Test with memory constraints
        constraints = {"flash": 512*1024, "ram": 128*1024}
        scanner_with_constraints = FirmwareScanner("cortex-m4", constraints)
        assert scanner_with_constraints.memory_constraints == constraints
        print("✅ Scanner with memory constraints works")
        
        return True
        
    except Exception as e:
        print(f"❌ Scanner test failed: {e}")
        return False

def test_utils_functionality():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from pqc_iot_retrofit.utils import (
            calculate_entropy, format_size, format_address,
            ArchitectureDetector, FileFormat
        )
        
        # Test entropy calculation
        entropy = calculate_entropy(b"Hello World")
        assert entropy >= 0
        print(f"✅ Entropy calculation works: {entropy}")
        
        # Test size formatting
        formatted = format_size(1024*1024)
        assert "MB" in formatted
        print(f"✅ Size formatting works: {formatted}")
        
        # Test address formatting
        addr = format_address(0x08000000)
        assert addr == "0x08000000"
        print(f"✅ Address formatting works: {addr}")
        
        # Test file format detection
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"test data")
            tmp.flush()
            
            file_format = ArchitectureDetector.detect_file_format(tmp.name)
            assert file_format in [FileFormat.BIN, FileFormat.UNKNOWN]
            print(f"✅ File format detection works: {file_format}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def test_memory_analyzer():
    """Test memory layout analyzer."""
    print("\nTesting memory analyzer...")
    
    try:
        from pqc_iot_retrofit.utils import MemoryLayoutAnalyzer
        
        # Test memory layout retrieval
        layout = MemoryLayoutAnalyzer.get_memory_layout("cortex-m4")
        assert "flash" in layout
        assert "ram" in layout
        print("✅ Memory layout retrieval works")
        
        # Test memory estimation
        memory = MemoryLayoutAnalyzer.estimate_available_memory("cortex-m4", 256*1024)
        assert "flash_available" in memory
        assert "ram_available" in memory
        assert all(val >= 0 for val in memory.values())
        print("✅ Memory estimation works")
        
        # Test unknown architecture fallback
        unknown_memory = MemoryLayoutAnalyzer.estimate_available_memory("unknown_arch", 100*1024)
        assert unknown_memory["ram_available"] == 32*1024
        print("✅ Unknown architecture fallback works")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory analyzer test failed: {e}")
        return False

def test_reporting():
    """Test report generation."""
    print("\nTesting reporting functionality...")
    
    try:
        from pqc_iot_retrofit.reporting import ReportGenerator, ReportFormat, ReportMetadata
        from pqc_iot_retrofit.scanner import CryptoVulnerability, CryptoAlgorithm, RiskLevel
        
        # Create test data
        generator = ReportGenerator()
        metadata = ReportMetadata(firmware_path="/test/firmware.bin", architecture="cortex-m4")
        
        vuln = CryptoVulnerability(
            algorithm=CryptoAlgorithm.RSA_2048,
            address=0x08001000,
            function_name="test_function",
            risk_level=RiskLevel.HIGH,
            key_size=2048,
            description="Test vulnerability",
            mitigation="Test mitigation",
            stack_usage=256,
            available_stack=4096
        )
        
        scan_results = {
            "scan_summary": {"total_vulnerabilities": 1},
            "recommendations": ["Test recommendation"]
        }
        
        # Test JSON report generation
        json_report = generator.generate_report(
            scan_results, [vuln], metadata, ReportFormat.JSON
        )
        
        # Validate JSON structure
        report_data = json.loads(json_report)
        assert "metadata" in report_data
        assert "scan_summary" in report_data
        print("✅ JSON report generation works")
        
        # Test text report generation
        text_report = generator.generate_report(
            scan_results, [vuln], metadata, ReportFormat.TEXT
        )
        assert "PQC IOT" in text_report.upper()
        print("✅ Text report generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Reporting test failed: {e}")
        return False

def test_firmware_scanning():
    """Test actual firmware scanning with test file."""
    print("\nTesting firmware scanning...")
    
    try:
        from pqc_iot_retrofit.scanner import FirmwareScanner
        
        # Create test firmware with some patterns
        test_data = b'\x00' * 1000 + b'\x01\x00\x01\x00' + b'\x00' * 1000  # RSA pattern
        
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(test_data)
            tmp.flush()
            
            scanner = FirmwareScanner("cortex-m4")
            vulnerabilities = scanner.scan_firmware(tmp.name)
            
            assert isinstance(vulnerabilities, list)
            print(f"✅ Firmware scanning works: found {len(vulnerabilities)} potential issues")
            
            # Test report generation
            report = scanner.generate_report()
            assert "scan_summary" in report
            print("✅ Report generation from scan works")
        
        return True
        
    except Exception as e:
        print(f"❌ Firmware scanning test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")
    
    try:
        from pqc_iot_retrofit.scanner import FirmwareScanner
        from pqc_iot_retrofit.utils import create_firmware_info
        from pqc_iot_retrofit.reporting import ReportGenerator, ReportMetadata, ReportFormat
        
        # Create test firmware
        test_data = b"Test firmware content"
        
        with tempfile.NamedTemporaryFile(suffix='.bin') as tmp:
            tmp.write(test_data)
            tmp.flush()
            
            # Test firmware info creation
            info = create_firmware_info(tmp.name)
            assert info.size == len(test_data)
            print("✅ Firmware info creation works")
            
            # Test scanning
            scanner = FirmwareScanner("cortex-m4")
            vulnerabilities = scanner.scan_firmware(tmp.name)
            scan_results = scanner.generate_report()
            
            # Test integrated reporting
            generator = ReportGenerator()
            metadata = ReportMetadata(firmware_path=tmp.name, architecture="cortex-m4")
            
            report = generator.generate_report(
                scan_results, vulnerabilities, metadata, ReportFormat.JSON
            )
            
            report_data = json.loads(report)
            assert "metadata" in report_data
            print("✅ End-to-end integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Generation 1 Validation Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_scanner_basic,
        test_utils_functionality,
        test_memory_analyzer,
        test_reporting,
        test_firmware_scanning,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Generation 1 tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)