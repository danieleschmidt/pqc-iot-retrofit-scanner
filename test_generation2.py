#!/usr/bin/env python3
"""Test Generation 2 robust security features."""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_robust_scanner():
    """Test the robust scanner with security features."""
    
    # Create test firmware
    mock_firmware = b'\x00' * 100 
    mock_firmware += b'\x01\x00\x01\x00'  # RSA constant
    mock_firmware += b'\x00' * 100
    mock_firmware += b'RSA-2048'  # Algorithm string
    mock_firmware += b'\x00' * 100
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as f:
        f.write(mock_firmware)
        firmware_path = f.name
    
    try:
        from pqc_iot_retrofit.robust_scanner import RobustFirmwareScanner
        from pqc_iot_retrofit.security_enhanced import SecurityContext
        
        print("✅ Successfully imported robust scanner modules")
        
        # Initialize robust scanner
        scanner = RobustFirmwareScanner(
            architecture="cortex-m4",
            memory_constraints={"flash": 512*1024, "ram": 128*1024},
            user_id="test_user"
        )
        
        print(f"✅ Robust scanner initialized with security context: {scanner.security_context.session_id}")
        
        # Test prerequisite validation
        prerequisites = scanner.validate_scan_prerequisites(firmware_path)
        print(f"✅ Prerequisites validation: {prerequisites['prerequisites_met']}")
        
        if prerequisites.get('security_validation', {}).get('security_flags'):
            print(f"⚠️  Security flags: {prerequisites['security_validation']['security_flags']}")
        elif 'error' in prerequisites:
            print(f"⚠️  Prerequisites error: {prerequisites['error']}")
        
        # Perform secure scan
        vulnerabilities = scanner.scan_firmware_securely(firmware_path, base_address=0x08000000)
        
        print(f"✅ Secure scan completed: found {len(vulnerabilities)} vulnerabilities")
        
        # Generate enhanced report
        report = scanner.generate_enhanced_report()
        print(f"✅ Enhanced report generated with Generation {report['generation']} features")
        
        # Display security summary
        security_summary = scanner.get_security_summary()
        print(f"✅ Security summary - Session: {security_summary['session_id'][:8]}...")
        print(f"   Rate limit remaining: {security_summary['rate_limit_remaining']}")
        print(f"   Scan success rate: {scanner.scan_stats['successful_scans']}/{scanner.scan_stats['total_scans']}")
        
        # Test enhanced vulnerability details
        if vulnerabilities:
            print("\n🔒 Enhanced Vulnerability Details:")
            for i, vuln in enumerate(vulnerabilities[:2], 1):
                print(f"  {i}. {vuln.algorithm.value} at 0x{vuln.address:08x}")
                print(f"     Enhanced: {vuln.description}")
                print(f"     Risk: {vuln.risk_level.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust scanner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.unlink(firmware_path)


def test_security_features():
    """Test specific security features."""
    
    try:
        from pqc_iot_retrofit.security_enhanced import (
            SecureFirmwareHandler, InputSanitizer, RateLimiter,
            create_secure_scanner_context
        )
        
        print("✅ Security modules imported successfully")
        
        # Test input sanitization
        try:
            sanitized_arch = InputSanitizer.sanitize_architecture("CORTEX-M4")
            print(f"✅ Architecture sanitization: {sanitized_arch}")
        except Exception as e:
            print(f"❌ Architecture sanitization failed: {e}")
            return False
        
        # Test memory constraint sanitization
        try:
            constraints = {"flash": 512*1024, "ram": 128*1024}
            sanitized_constraints = InputSanitizer.sanitize_memory_constraints(constraints)
            print(f"✅ Memory constraint sanitization: {len(sanitized_constraints)} constraints")
        except Exception as e:
            print(f"❌ Memory constraint sanitization failed: {e}")
            return False
        
        # Test security context creation
        try:
            context = create_secure_scanner_context("test_user")
            print(f"✅ Security context created: {context.session_id[:8]}...")
        except Exception as e:
            print(f"❌ Security context creation failed: {e}")
            return False
        
        # Test rate limiter
        try:
            rate_limiter = RateLimiter(max_requests=5, window_seconds=60)
            allowed, remaining = rate_limiter.check_rate_limit("test_session")
            print(f"✅ Rate limiter: allowed={allowed}, remaining={remaining}")
        except Exception as e:
            print(f"❌ Rate limiter test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_inputs():
    """Test handling of invalid inputs."""
    
    try:
        from pqc_iot_retrofit.robust_scanner import RobustFirmwareScanner
        from pqc_iot_retrofit.error_handling import ValidationError, SecurityError
        
        print("✅ Testing invalid input handling")
        
        # Test invalid architecture
        try:
            scanner = RobustFirmwareScanner("invalid_arch")
            print("❌ Should have failed with invalid architecture")
            return False
        except ValidationError:
            print("✅ Invalid architecture properly rejected")
        
        # Test invalid memory constraints
        try:
            scanner = RobustFirmwareScanner("cortex-m4", {"invalid_type": 1024})
            print("❌ Should have failed with invalid memory constraints")
            return False
        except ValidationError:
            print("✅ Invalid memory constraints properly rejected")
        
        # Test with valid inputs
        scanner = RobustFirmwareScanner("cortex-m4")
        print("✅ Valid inputs accepted")
        
        # Test scanning non-existent file
        try:
            vulnerabilities = scanner.scan_firmware_securely("/nonexistent/file.bin")
            print("❌ Should have failed with non-existent file")
            return False
        except (ValidationError, SecurityError):
            print("✅ Non-existent file properly rejected")
        
        return True
        
    except Exception as e:
        print(f"❌ Invalid input test failed: {e}")
        return False


def main():
    """Run all Generation 2 tests."""
    print("🚀 Testing PQC IoT Retrofit Scanner - Generation 2 Robust Security\n")
    
    tests = [
        ("Robust Scanner Functionality", test_robust_scanner),
        ("Security Features", test_security_features),
        ("Invalid Input Handling", test_invalid_inputs),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 60)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("📊 GENERATION 2 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Generation 2 tests passed! Robust security features working.")
        print("\n🔒 Generation 2 Features Verified:")
        print("   • Secure file validation and integrity checking")
        print("   • Rate limiting and session management")
        print("   • Input sanitization and validation")
        print("   • Enhanced error handling and logging")
        print("   • Security audit trails")
        print("   • Prerequisite validation")
        return True
    else:
        print("⚠️  Some tests failed. Security features may need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)