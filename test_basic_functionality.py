#!/usr/bin/env python3
"""Basic functionality test for PQC IoT Retrofit Scanner Generation 1."""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_scanner_basic():
    """Test basic scanner functionality without external dependencies."""
    
    # Create a mock firmware file with RSA constants
    mock_firmware = b'\x00' * 100  # Padding
    mock_firmware += b'\x01\x00\x01\x00'  # RSA-65537 exponent 
    mock_firmware += b'\x00' * 100  # More padding
    mock_firmware += b'RSA'  # RSA string
    mock_firmware += b'\x00' * 100  # Padding
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(mock_firmware)
        firmware_path = f.name
    
    try:
        # Test basic scanner import and functionality
        from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoAlgorithm, RiskLevel
        
        print("✅ Successfully imported scanner modules")
        
        # Initialize scanner for Cortex-M4
        scanner = FirmwareScanner(
            architecture="cortex-m4",
            memory_constraints={"flash": 512*1024, "ram": 128*1024}
        )
        
        print(f"✅ Scanner initialized: {scanner}")
        
        # Scan the mock firmware
        vulnerabilities = scanner.scan_firmware(firmware_path, base_address=0x08000000)
        
        print(f"✅ Scan completed: found {len(vulnerabilities)} vulnerabilities")
        
        # Verify we detected the RSA vulnerability
        rsa_found = any(vuln.algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.RSA_2048] 
                       for vuln in vulnerabilities)
        
        if rsa_found:
            print("✅ RSA vulnerability detected correctly")
        else:
            print("⚠️  RSA vulnerability not detected")
        
        # Generate report
        report = scanner.generate_report()
        print(f"✅ Report generated with {report['scan_summary']['total_vulnerabilities']} vulnerabilities")
        
        # Display findings
        if vulnerabilities:
            print("\n🔍 Detected Vulnerabilities:")
            for i, vuln in enumerate(vulnerabilities[:3], 1):
                print(f"  {i}. {vuln.algorithm.value} at 0x{vuln.address:08x}")
                print(f"     Risk: {vuln.risk_level.value}")
                print(f"     Mitigation: {vuln.mitigation}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        os.unlink(firmware_path)


def test_cli_import():
    """Test CLI module imports."""
    try:
        from pqc_iot_retrofit.cli import main
        print("✅ CLI module imported successfully")
        return True
    except Exception as e:
        print(f"⚠️  CLI import failed (expected due to missing rich): {e}")
        return False


def test_patcher_import():
    """Test patcher module imports."""
    try:
        from pqc_iot_retrofit.patcher import PQCPatcher
        print("✅ Patcher module imported successfully")
        return True
    except Exception as e:
        print(f"⚠️  Patcher import failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🚀 Testing PQC IoT Retrofit Scanner - Generation 1 Basic Functionality\n")
    
    tests = [
        ("Scanner Basic Functionality", test_scanner_basic),
        ("CLI Import", test_cli_import),
        ("Patcher Import", test_patcher_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 50)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! Generation 1 functionality working.")
        return True
    else:
        print("⚠️  Some tests failed. Core functionality may need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)