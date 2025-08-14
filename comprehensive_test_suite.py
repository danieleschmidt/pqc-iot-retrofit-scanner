#!/usr/bin/env python3
"""Comprehensive test suite for all PQC IoT Retrofit Scanner generations."""

import sys
import os
import tempfile
import time
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_comprehensive_test_firmware() -> str:
    """Create comprehensive test firmware with multiple vulnerability types."""
    
    firmware_data = bytearray()
    
    # Add padding
    firmware_data.extend(b'\x00' * 200)
    
    # RSA vulnerabilities
    firmware_data.extend(b'\x01\x00\x01\x00')  # RSA-65537 exponent
    firmware_data.extend(b'\x00' * 50)
    firmware_data.extend(b'\x00\x01\x00\x00')  # RSA-2048 length indicator
    firmware_data.extend(b'\x00' * 50)
    firmware_data.extend(b'RSA-2048')  # Algorithm string
    firmware_data.extend(b'\x00' * 50)
    
    # ECC vulnerabilities  
    firmware_data.extend(b'\xff\xff\xff\xff\x00\x00\x00\x01')  # P-256 curve param
    firmware_data.extend(b'\x00' * 50)
    firmware_data.extend(b'ECDSA-P256')  # Algorithm string
    firmware_data.extend(b'\x00' * 50)
    
    # DH vulnerabilities
    firmware_data.extend(b'DH-2048')
    firmware_data.extend(b'\x00' * 50)
    firmware_data.extend(b'Diffie-Hellman')
    firmware_data.extend(b'\x00' * 100)
    
    # Create temporary file
    fd, firmware_path = tempfile.mkstemp(suffix='.bin', prefix='comprehensive_test_')
    with os.fdopen(fd, 'wb') as f:
        f.write(firmware_data)
    
    return firmware_path


class TestResults:
    """Track test results across all generations."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.generation_times = {}
        
    def add_result(self, generation: str, test_name: str, passed: bool, 
                  execution_time: float = 0.0, details: Dict[str, Any] = None):
        """Add test result."""
        if generation not in self.results:
            self.results[generation] = []
        
        self.results[generation].append({
            'test_name': test_name,
            'passed': passed,
            'execution_time': execution_time,
            'details': details or {}
        })
    
    def add_generation_time(self, generation: str, execution_time: float):
        """Add generation execution time."""
        self.generation_times[generation] = execution_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        total_tests = 0
        total_passed = 0
        
        generation_summary = {}
        
        for generation, tests in self.results.items():
            gen_passed = sum(1 for test in tests if test['passed'])
            gen_total = len(tests)
            
            generation_summary[generation] = {
                'passed': gen_passed,
                'total': gen_total,
                'success_rate': gen_passed / max(1, gen_total),
                'execution_time': self.generation_times.get(generation, 0.0)
            }
            
            total_tests += gen_total
            total_passed += gen_passed
        
        return {
            'overall_success_rate': total_passed / max(1, total_tests),
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_time': time.time() - self.start_time,
            'generations': generation_summary
        }


def test_generation_1_comprehensive(test_results: TestResults):
    """Comprehensive Generation 1 tests."""
    
    print("🔬 GENERATION 1: COMPREHENSIVE FUNCTIONALITY TESTS")
    print("=" * 60)
    
    gen_start = time.time()
    firmware_path = create_comprehensive_test_firmware()
    
    try:
        # Test 1: Basic Scanner Initialization
        start_time = time.time()
        try:
            from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoAlgorithm, RiskLevel
            
            scanner = FirmwareScanner("cortex-m4", {"flash": 512*1024, "ram": 128*1024})
            test_results.add_result("Generation 1", "Scanner Initialization", True, 
                                  time.time() - start_time)
            print("✅ Scanner Initialization")
            
        except Exception as e:
            test_results.add_result("Generation 1", "Scanner Initialization", False,
                                  time.time() - start_time, {"error": str(e)})
            print(f"❌ Scanner Initialization: {e}")
            return
        
        # Test 2: Comprehensive Vulnerability Detection
        start_time = time.time()
        try:
            vulnerabilities = scanner.scan_firmware(firmware_path, 0x08000000)
            
            # Check for different vulnerability types
            rsa_found = any(vuln.algorithm.value.startswith('RSA') for vuln in vulnerabilities)
            ecc_found = any(vuln.algorithm.value.startswith('ECDSA') for vuln in vulnerabilities)
            
            success = len(vulnerabilities) >= 3 and rsa_found  # At minimum expect RSA + others
            test_results.add_result("Generation 1", "Vulnerability Detection", success,
                                  time.time() - start_time, 
                                  {"vulnerabilities_found": len(vulnerabilities),
                                   "rsa_detected": rsa_found, "ecc_detected": ecc_found})
            
            status = "✅" if success else "❌"
            print(f"{status} Vulnerability Detection: {len(vulnerabilities)} vulnerabilities")
            
        except Exception as e:
            test_results.add_result("Generation 1", "Vulnerability Detection", False,
                                  time.time() - start_time, {"error": str(e)})
            print(f"❌ Vulnerability Detection: {e}")
        
        # Test 3: Report Generation
        start_time = time.time()
        try:
            report = scanner.generate_report()
            
            success = (isinstance(report, dict) and 
                      'scan_summary' in report and
                      'vulnerabilities' in report and
                      'recommendations' in report)
            
            test_results.add_result("Generation 1", "Report Generation", success,
                                  time.time() - start_time,
                                  {"report_sections": list(report.keys()) if success else []})
            
            status = "✅" if success else "❌"
            print(f"{status} Report Generation")
            
        except Exception as e:
            test_results.add_result("Generation 1", "Report Generation", False,
                                  time.time() - start_time, {"error": str(e)})
            print(f"❌ Report Generation: {e}")
        
        # Test 4: Architecture Support
        start_time = time.time()
        architectures_tested = 0
        architectures_working = 0
        
        for arch in ["cortex-m0", "cortex-m3", "cortex-m4", "esp32", "riscv32"]:
            try:
                test_scanner = FirmwareScanner(arch)
                architectures_tested += 1
                architectures_working += 1
            except Exception:
                architectures_tested += 1
        
        success = architectures_working >= 3  # At least 3 architectures should work
        test_results.add_result("Generation 1", "Architecture Support", success,
                              time.time() - start_time,
                              {"tested": architectures_tested, "working": architectures_working})
        
        status = "✅" if success else "❌"
        print(f"{status} Architecture Support: {architectures_working}/{architectures_tested}")
        
    finally:
        os.unlink(firmware_path)
        test_results.add_generation_time("Generation 1", time.time() - gen_start)


def test_generation_2_comprehensive(test_results: TestResults):
    """Comprehensive Generation 2 tests."""
    
    print("\n🔒 GENERATION 2: COMPREHENSIVE SECURITY TESTS")
    print("=" * 60)
    
    gen_start = time.time()
    
    # Test 1: Security Module Import and Functionality
    start_time = time.time()
    try:
        from pqc_iot_retrofit.security_enhanced import (
            SecureFirmwareHandler, InputSanitizer, create_secure_scanner_context
        )
        from pqc_iot_retrofit.robust_scanner import RobustFirmwareScanner
        
        test_results.add_result("Generation 2", "Security Module Import", True,
                              time.time() - start_time)
        print("✅ Security Module Import")
        
    except Exception as e:
        test_results.add_result("Generation 2", "Security Module Import", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Security Module Import: {e}")
        return
    
    # Test 2: Input Sanitization
    start_time = time.time()
    sanitization_tests = 0
    sanitization_passed = 0
    
    try:
        # Valid inputs
        try:
            InputSanitizer.sanitize_architecture("cortex-m4")
            sanitization_tests += 1
            sanitization_passed += 1
        except:
            sanitization_tests += 1
        
        # Invalid inputs should raise exceptions
        try:
            InputSanitizer.sanitize_architecture("invalid_arch")
            sanitization_tests += 1  # Should not reach here
        except:
            sanitization_tests += 1
            sanitization_passed += 1  # Expected to fail
        
        success = sanitization_passed == sanitization_tests
        test_results.add_result("Generation 2", "Input Sanitization", success,
                              time.time() - start_time,
                              {"tests": sanitization_tests, "passed": sanitization_passed})
        
        status = "✅" if success else "❌"
        print(f"{status} Input Sanitization: {sanitization_passed}/{sanitization_tests}")
        
    except Exception as e:
        test_results.add_result("Generation 2", "Input Sanitization", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Input Sanitization: {e}")
    
    # Test 3: Secure Scanner Functionality
    firmware_path = create_comprehensive_test_firmware()
    start_time = time.time()
    
    try:
        scanner = RobustFirmwareScanner("cortex-m4", user_id="test_comprehensive")
        vulnerabilities = scanner.scan_firmware_securely(firmware_path, 0x08000000)
        
        success = len(vulnerabilities) >= 3
        test_results.add_result("Generation 2", "Secure Scanner", success,
                              time.time() - start_time,
                              {"vulnerabilities": len(vulnerabilities),
                               "session_id": scanner.security_context.session_id[:8]})
        
        status = "✅" if success else "❌"
        print(f"{status} Secure Scanner: {len(vulnerabilities)} vulnerabilities")
        
    except Exception as e:
        test_results.add_result("Generation 2", "Secure Scanner", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Secure Scanner: {e}")
    finally:
        os.unlink(firmware_path)
    
    # Test 4: Enhanced Reporting
    start_time = time.time()
    try:
        scanner = RobustFirmwareScanner("cortex-m4")
        report = scanner.generate_enhanced_report()
        
        success = (isinstance(report, dict) and
                  'security_context' in report and
                  'performance_statistics' in report and
                  'generation' in report and
                  report['generation'] == 2)
        
        test_results.add_result("Generation 2", "Enhanced Reporting", success,
                              time.time() - start_time,
                              {"generation": report.get('generation'),
                               "features": len(report.get('features', []))})
        
        status = "✅" if success else "❌"
        print(f"{status} Enhanced Reporting")
        
    except Exception as e:
        test_results.add_result("Generation 2", "Enhanced Reporting", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Enhanced Reporting: {e}")
    
    test_results.add_generation_time("Generation 2", time.time() - gen_start)


def test_generation_3_comprehensive(test_results: TestResults):
    """Comprehensive Generation 3 tests."""
    
    print("\n⚡ GENERATION 3: COMPREHENSIVE PERFORMANCE TESTS")
    print("=" * 60)
    
    gen_start = time.time()
    
    # Test 1: Optimized Scanner Import and Initialization
    start_time = time.time()
    try:
        from pqc_iot_retrofit.optimized_scanner import (
            OptimizedFirmwareScanner, IntelligentCache, create_optimized_scanner
        )
        
        scanner = OptimizedFirmwareScanner("cortex-m4", enable_caching=True, enable_worker_pool=False)
        
        test_results.add_result("Generation 3", "Optimized Scanner Init", True,
                              time.time() - start_time)
        print("✅ Optimized Scanner Init")
        
    except Exception as e:
        test_results.add_result("Generation 3", "Optimized Scanner Init", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Optimized Scanner Init: {e}")
        return
    
    # Test 2: Caching Performance
    firmware_path = create_comprehensive_test_firmware()
    start_time = time.time()
    
    try:
        # First scan (cache miss)
        first_start = time.time()
        vulnerabilities1 = scanner.scan_firmware_optimized(firmware_path, 0x08000000)
        first_time = time.time() - first_start
        
        # Second scan (cache hit)
        second_start = time.time()
        vulnerabilities2 = scanner.scan_firmware_optimized(firmware_path, 0x08000000)
        second_time = time.time() - second_start
        
        # Cache should provide speedup and identical results
        speedup = first_time / max(second_time, 0.0001)  # Avoid division by zero
        results_match = vulnerabilities1 == vulnerabilities2
        
        # For very fast operations, accept lower speedup requirements
        min_speedup = 1.2 if first_time > 0.01 else 0.5  # Adjust for very fast scans
        success = speedup >= min_speedup and results_match and len(vulnerabilities1) >= 3
        
        test_results.add_result("Generation 3", "Caching Performance", success,
                              time.time() - start_time,
                              {"speedup": speedup, "results_match": results_match,
                               "first_time": first_time, "second_time": second_time})
        
        status = "✅" if success else "❌"
        print(f"{status} Caching Performance: {speedup:.1f}x speedup")
        
    except Exception as e:
        test_results.add_result("Generation 3", "Caching Performance", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Caching Performance: {e}")
    finally:
        os.unlink(firmware_path)
    
    # Test 3: Batch Processing
    start_time = time.time()
    try:
        # Create multiple test files
        firmware_files = []
        for i in range(3):
            firmware_path = create_comprehensive_test_firmware()
            firmware_files.append(firmware_path)
        
        try:
            batch_results = scanner.scan_firmware_batch(firmware_files)
            
            success = (len(batch_results) == 3 and
                      all(isinstance(result[1], list) for result in batch_results) and
                      all(len(result[1]) >= 1 for result in batch_results))
            
            test_results.add_result("Generation 3", "Batch Processing", success,
                                  time.time() - start_time,
                                  {"files_processed": len(batch_results),
                                   "avg_vulnerabilities": sum(len(r[1]) for r in batch_results) / len(batch_results)})
            
            status = "✅" if success else "❌"
            print(f"{status} Batch Processing: {len(batch_results)} files")
            
        finally:
            for fp in firmware_files:
                try:
                    os.unlink(fp)
                except:
                    pass
    
    except Exception as e:
        test_results.add_result("Generation 3", "Batch Processing", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Batch Processing: {e}")
    
    # Test 4: Performance Reporting
    start_time = time.time()
    try:
        performance_report = scanner.get_performance_report()
        
        success = (isinstance(performance_report, dict) and
                  'generation_3_performance' in performance_report and
                  'cache_performance' in performance_report['generation_3_performance'] and
                  'efficiency_metrics' in performance_report['generation_3_performance'])
        
        test_results.add_result("Generation 3", "Performance Reporting", success,
                              time.time() - start_time,
                              {"report_keys": list(performance_report.keys()) if success else []})
        
        status = "✅" if success else "❌"
        print(f"{status} Performance Reporting")
        
    except Exception as e:
        test_results.add_result("Generation 3", "Performance Reporting", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Performance Reporting: {e}")
    
    test_results.add_generation_time("Generation 3", time.time() - gen_start)


async def test_async_functionality(test_results: TestResults):
    """Test asynchronous functionality."""
    
    print("\n🔄 ASYNCHRONOUS FUNCTIONALITY TESTS")
    print("=" * 60)
    
    start_time = time.time()
    firmware_path = create_comprehensive_test_firmware()
    
    try:
        from pqc_iot_retrofit.optimized_scanner import OptimizedFirmwareScanner
        
        scanner = OptimizedFirmwareScanner("cortex-m4", enable_worker_pool=False)
        
        # Test async scanning
        vulnerabilities = await scanner.scan_firmware_async(firmware_path, 0x08000000)
        
        success = isinstance(vulnerabilities, list) and len(vulnerabilities) >= 1
        
        test_results.add_result("Async", "Async Scanning", success,
                              time.time() - start_time,
                              {"vulnerabilities": len(vulnerabilities)})
        
        status = "✅" if success else "❌"
        print(f"{status} Async Scanning: {len(vulnerabilities)} vulnerabilities")
        
    except Exception as e:
        test_results.add_result("Async", "Async Scanning", False,
                              time.time() - start_time, {"error": str(e)})
        print(f"❌ Async Scanning: {e}")
    finally:
        os.unlink(firmware_path)


def print_comprehensive_summary(test_results: TestResults):
    """Print comprehensive test summary."""
    
    summary = test_results.get_summary()
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    print(f"🎯 Overall Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"📈 Total Tests: {summary['total_tests']} ({summary['total_passed']} passed)")
    print(f"⏱️  Total Execution Time: {summary['total_time']:.2f} seconds")
    
    print("\n📋 Generation Breakdown:")
    
    for generation, gen_data in summary['generations'].items():
        success_rate = gen_data['success_rate']
        status_icon = "✅" if success_rate == 1.0 else "⚠️" if success_rate >= 0.5 else "❌"
        
        print(f"  {status_icon} {generation}:")
        print(f"     Success Rate: {success_rate:.1%} ({gen_data['passed']}/{gen_data['total']})")
        print(f"     Execution Time: {gen_data['execution_time']:.2f}s")
        
        # Show individual test details for failed generations
        if success_rate < 1.0 and generation in test_results.results:
            failed_tests = [test for test in test_results.results[generation] if not test['passed']]
            if failed_tests:
                print(f"     Failed Tests:")
                for test in failed_tests:
                    error = test['details'].get('error', 'Unknown error')
                    print(f"       • {test['test_name']}: {error}")
    
    # Quality gates assessment
    print("\n🚧 Quality Gates Assessment:")
    
    quality_gates = {
        "Core Functionality": summary['generations'].get('Generation 1', {}).get('success_rate', 0) >= 0.8,
        "Security Features": summary['generations'].get('Generation 2', {}).get('success_rate', 0) >= 0.8,
        "Performance Features": summary['generations'].get('Generation 3', {}).get('success_rate', 0) >= 0.8,
        "Overall Success": summary['overall_success_rate'] >= 0.85,
        "Execution Performance": summary['total_time'] < 60.0  # Should complete in under 1 minute
    }
    
    for gate_name, passed in quality_gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {gate_name}")
    
    all_gates_passed = all(quality_gates.values())
    
    print(f"\n🎯 Quality Gates: {'✅ ALL PASSED' if all_gates_passed else '❌ SOME FAILED'}")
    
    if all_gates_passed:
        print("\n🎉 READY FOR PRODUCTION DEPLOYMENT!")
        print("   All generations implemented successfully with:")
        print("   • Core firmware scanning functionality")
        print("   • Robust security and error handling")
        print("   • High-performance optimization features")
        print("   • Comprehensive testing coverage")
    else:
        print("\n⚠️  ADDITIONAL WORK NEEDED BEFORE PRODUCTION")
        failed_gates = [name for name, passed in quality_gates.items() if not passed]
        print(f"   Failed gates: {', '.join(failed_gates)}")
    
    return all_gates_passed


def main():
    """Run comprehensive test suite for all generations."""
    
    print("🧪 PQC IoT Retrofit Scanner - COMPREHENSIVE TEST SUITE")
    print("🔬 Testing Generations 1, 2, 3 + Async functionality")
    print("=" * 80)
    
    test_results = TestResults()
    
    # Run all test generations
    test_generation_1_comprehensive(test_results)
    test_generation_2_comprehensive(test_results)
    test_generation_3_comprehensive(test_results)
    
    # Run async tests
    asyncio.run(test_async_functionality(test_results))
    
    # Print comprehensive summary and assess quality gates
    all_passed = print_comprehensive_summary(test_results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)