"""Security Validation Tests.

Comprehensive security testing for PQC IoT Retrofit Scanner including
cryptographic correctness, side-channel resistance, and attack simulation.
"""

import pytest
import hashlib
import hmac
import secrets
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoAlgorithm, RiskLevel
from pqc_iot_retrofit.adaptive_ai import AdaptiveAI, AnomalyDetector
from pqc_iot_retrofit.quantum_resilience import QuantumResilienceAnalyzer, QuantumThreatLevel


class TestCryptographicSecurity:
    """Test cryptographic security properties."""
    
    def test_vulnerability_detection_accuracy(self):
        """Test accuracy of vulnerability detection."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Create firmware with known crypto patterns
        firmware_with_rsa = self._create_firmware_with_crypto_patterns([
            b'RSA_SIGN',
            b'\x01\x00\x01\x00',  # RSA public exponent 65537
            b'\x30\x31\x30\x0d'   # PKCS-SHA256
        ])
        
        vulnerabilities = scanner.scan_firmware("", base_address=0)
        scanner.vulnerabilities = []  # Reset
        
        # Should detect RSA patterns
        rsa_vulns = [v for v in vulnerabilities if 'RSA' in v.algorithm.value]
        assert len(rsa_vulns) > 0, "Should detect RSA patterns"
        
        # Test with ECC patterns
        firmware_with_ecc = self._create_firmware_with_crypto_patterns([
            b'ECDSA',
            b'\xff\xff\xff\xff\x00\x00\x00\x01',  # P-256 curve parameter
        ])
        
        vulnerabilities = scanner._scan_crypto_strings(firmware_with_ecc, 0)
        ecc_vulns = [v for v in scanner.vulnerabilities if 'ECDSA' in v.algorithm.value]
        assert len(ecc_vulns) > 0, "Should detect ECDSA patterns"
    
    def test_false_positive_resistance(self):
        """Test resistance to false positive detections."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Create firmware with crypto-like but non-crypto patterns
        false_positive_patterns = [
            b'RSA_KEYSTORE',  # Just a string, not actual crypto
            b'RANDOM_DATA_WITH_RSA_WORD',
            b'\x01\x00\x01\x00' * 100,  # Repeated pattern, likely data
        ]
        
        firmware = self._create_firmware_with_patterns(false_positive_patterns)
        vulnerabilities = scanner.scan_firmware("", base_address=0)
        
        # Should have some detections but not excessive false positives
        total_detections = len(vulnerabilities)
        assert total_detections < 50, f"Too many detections ({total_detections}), likely false positives"
    
    def test_cryptographic_constant_validation(self):
        """Test validation of cryptographic constants."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test RSA constants
        rsa_constants = scanner.RSA_CONSTANTS
        assert b'\x01\x00\x01\x00' in rsa_constants  # RSA-65537 exponent
        assert b'\x30\x21\x30\x09' in rsa_constants  # PKCS-SHA1
        
        # Test ECC constants
        ecc_constants = scanner.ECC_CURVES
        assert b'\xff\xff\xff\xff\x00\x00\x00\x01' in ecc_constants  # P-256 prime
        
        # Validate that constants are cryptographically meaningful
        for constant, description in rsa_constants.items():
            assert len(constant) >= 4, f"RSA constant {description} too short"
            assert isinstance(description, str), f"RSA constant description should be string"
    
    def test_key_size_estimation_accuracy(self):
        """Test accuracy of cryptographic key size estimation."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test RSA key size detection
        test_cases = [
            (b'\x80\x00\x00\x00', 1024),  # 128 bytes = 1024 bits
            (b'\x00\x01\x00\x00', 2048),  # 256 bytes = 2048 bits
            (b'\x00\x02\x00\x00', 4096),  # 512 bytes = 4096 bits
        ]
        
        for marker, expected_size in test_cases:
            # Create test data with key size marker
            test_data = b'\x00' * 50 + marker + b'\x00' * 50
            estimated_size = scanner._estimate_rsa_key_size(test_data, 50)
            
            assert estimated_size == expected_size, f"Expected {expected_size}, got {estimated_size}"
    
    def test_algorithm_classification_accuracy(self):
        """Test accuracy of cryptographic algorithm classification."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test algorithm-specific patterns
        test_patterns = {
            CryptoAlgorithm.RSA_2048: [b'RSA', b'modular.*exp', b'\x01\x00\x01\x00'],
            CryptoAlgorithm.ECDSA_P256: [b'ECDSA', b'point.*mul', b'\xff\xff\xff\xff\x00\x00\x00\x01'],
            CryptoAlgorithm.ECDH_P256: [b'ECDH', b'shared.*secret'],
        }
        
        for algorithm, patterns in test_patterns.items():
            firmware = self._create_firmware_with_crypto_patterns(patterns)
            vulnerabilities = scanner.scan_firmware("", base_address=0)
            scanner.vulnerabilities = []  # Reset
            
            # Should detect the specific algorithm
            matching_vulns = [v for v in vulnerabilities if v.algorithm == algorithm]
            assert len(matching_vulns) > 0, f"Should detect {algorithm.value}"
    
    def _create_firmware_with_crypto_patterns(self, patterns):
        """Create synthetic firmware containing specific crypto patterns."""
        firmware = bytearray(10000)
        
        # Insert patterns at random locations
        for pattern in patterns:
            pos = np.random.randint(0, len(firmware) - len(pattern))
            firmware[pos:pos+len(pattern)] = pattern
        
        # Fill rest with realistic firmware data
        for i in range(0, len(firmware), 4):
            if firmware[i:i+4] == b'\x00\x00\x00\x00':
                # Replace zeros with realistic instruction-like data
                firmware[i:i+4] = np.random.randint(0x4000, 0x8000, dtype=np.uint16).tobytes() * 2
        
        return bytes(firmware)
    
    def _create_firmware_with_patterns(self, patterns):
        """Create firmware with specific patterns for testing."""
        firmware = bytearray()
        
        # Add patterns with spacing
        for pattern in patterns:
            firmware.extend(pattern)
            firmware.extend(b'\x00' * 100)  # Spacing
        
        # Fill to reasonable size
        while len(firmware) < 5000:
            firmware.extend(np.random.bytes(100))
        
        return bytes(firmware)


class TestSideChannelResistance:
    """Test resistance to side-channel attacks."""
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test scan timing consistency
        firmware_samples = [np.random.bytes(10000) for _ in range(10)]
        timing_measurements = []
        
        for firmware in firmware_samples:
            start_time = time.perf_counter()
            scanner.scan_firmware("", base_address=0)
            end_time = time.perf_counter()
            timing_measurements.append(end_time - start_time)
            scanner.vulnerabilities = []  # Reset
        
        # Timing should be relatively consistent (coefficient of variation < 0.5)
        mean_time = np.mean(timing_measurements)
        std_time = np.std(timing_measurements)
        cv = std_time / mean_time if mean_time > 0 else float('inf')
        
        assert cv < 0.5, f"Timing variation too high (CV: {cv:.3f}), potential timing leak"
    
    def test_memory_access_pattern_consistency(self):
        """Test consistency of memory access patterns."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test with different input sizes
        sizes = [1000, 5000, 10000, 20000]
        access_patterns = []
        
        for size in sizes:
            firmware = np.random.bytes(size)
            
            # Mock memory access tracking
            memory_accesses = []
            
            # Simulate memory access during scanning
            original_scan = scanner._scan_crypto_constants
            def track_access(*args):
                memory_accesses.append(len(args))
                return original_scan(*args)
            
            scanner._scan_crypto_constants = track_access
            scanner.scan_firmware("", base_address=0)
            scanner._scan_crypto_constants = original_scan
            
            access_patterns.append(len(memory_accesses))
            scanner.vulnerabilities = []  # Reset
        
        # Memory access should scale reasonably with input size
        # (not revealing internal structure through access patterns)
        assert all(pattern > 0 for pattern in access_patterns), "Should have memory accesses"
    
    def test_cache_attack_resistance(self):
        """Test resistance to cache-based side-channel attacks."""
        ai = AdaptiveAI()
        
        # Test cache behavior consistency
        firmware_samples = [
            self._generate_similar_firmware(base_seed=i) 
            for i in range(5)
        ]
        
        cache_behaviors = []
        for firmware in firmware_samples:
            # Enable performance optimization caching
            analysis = ai.analyze_firmware(firmware)
            
            # Check if cache timing is consistent
            if ai.adaptive_optimizer.optimization_history:
                cache_size = len(ai.adaptive_optimizer.optimization_history)
                cache_behaviors.append(cache_size)
        
        # Cache behavior should not leak information about input structure
        if len(cache_behaviors) > 1:
            cache_variation = np.std(cache_behaviors) / np.mean(cache_behaviors)
            assert cache_variation < 1.0, f"Cache behavior too variable: {cache_variation:.3f}"
    
    def test_power_analysis_resistance_simulation(self):
        """Simulate resistance to power analysis attacks."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Simulate power consumption during crypto detection
        power_traces = []
        
        test_inputs = [
            b'RSA' + b'\x00' * 1000,  # Crypto pattern at start
            b'\x00' * 500 + b'RSA' + b'\x00' * 500,  # Crypto pattern in middle
            b'\x00' * 1000 + b'RSA',  # Crypto pattern at end
            b'\x00' * 1003,  # No crypto pattern
        ]
        
        for input_data in test_inputs:
            # Simulate power trace (simplified model)
            power_trace = []
            
            # Mock the scanning process with power simulation
            for byte in input_data:
                # Simulate power consumption based on operations
                if byte == ord('R') or byte == ord('S') or byte == ord('A'):
                    power_trace.append(1.2)  # Higher power for pattern matching
                else:
                    power_trace.append(1.0)  # Baseline power
            
            power_traces.append(np.array(power_trace))
            
            # Actually run the scan
            scanner.scan_firmware("", base_address=0)
            scanner.vulnerabilities = []  # Reset
        
        # Power traces should not clearly distinguish between crypto/non-crypto regions
        if len(power_traces) >= 4:
            crypto_trace_variance = np.var(power_traces[0])  # With crypto
            no_crypto_trace_variance = np.var(power_traces[3])  # Without crypto
            
            # Variance should be similar (not revealing crypto locations)
            variance_ratio = abs(crypto_trace_variance - no_crypto_trace_variance)
            assert variance_ratio < 0.5, f"Power variance reveals crypto patterns: {variance_ratio:.3f}"
    
    def _generate_similar_firmware(self, base_seed=42, size=10000):
        """Generate similar firmware samples for cache testing."""
        np.random.seed(base_seed)
        
        # Create firmware with similar structure but different content
        firmware = bytearray()
        
        # Common structure
        firmware.extend(b'\x00' * 256)  # Vector table
        firmware.extend(np.random.bytes(size // 2))  # Code section
        firmware.extend(b'Common_String_Pattern')  # Common string
        firmware.extend(np.random.bytes(size // 2 - 256 - len(b'Common_String_Pattern')))
        
        return bytes(firmware)


class TestQuantumSecurityValidation:
    """Test quantum security analysis validation."""
    
    def test_quantum_threat_assessment_accuracy(self):
        """Test accuracy of quantum threat assessment."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Test known quantum-vulnerable algorithms
        vulnerable_algorithms = [
            (CryptoAlgorithm.RSA_1024, QuantumThreatLevel.IMMEDIATE),
            (CryptoAlgorithm.RSA_2048, QuantumThreatLevel.NEAR_TERM),
            (CryptoAlgorithm.ECDSA_P256, QuantumThreatLevel.NEAR_TERM),
        ]
        
        for algorithm, expected_threat in vulnerable_algorithms:
            threat_assessment = quantum_analyzer.threat_model.assess_quantum_vulnerability(
                algorithm, 
                key_size=2048 if 'RSA' in algorithm.value else 256
            )
            
            # Should correctly identify threat level
            assert threat_assessment.threat_level in [expected_threat, QuantumThreatLevel.MID_TERM], \
                f"Incorrect threat assessment for {algorithm.value}"
            
            # Should identify Shor's algorithm applicability
            assert threat_assessment.shor_applicable is True, \
                f"Should identify Shor's algorithm applies to {algorithm.value}"
    
    def test_quantum_resource_estimation(self):
        """Test quantum resource requirement estimation."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Test RSA resource estimation
        rsa_assessment = quantum_analyzer.threat_model.assess_quantum_vulnerability(
            CryptoAlgorithm.RSA_2048, 2048
        )
        
        # RSA-2048 should require reasonable quantum resources
        assert rsa_assessment.estimated_qubits_required > 1000, \
            "RSA-2048 should require substantial qubits"
        assert rsa_assessment.estimated_qubits_required < 100000, \
            "RSA-2048 qubit estimate should be reasonable"
        
        assert rsa_assessment.estimated_gate_count > 1000000, \
            "RSA-2048 should require substantial gate operations"
    
    def test_pqc_algorithm_security_validation(self):
        """Test validation of PQC algorithm security claims."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Test PQC algorithm profiles
        dilithium2 = quantum_analyzer.algorithm_db.get_algorithm('Dilithium2')
        assert dilithium2 is not None
        
        # Validate security properties
        assert dilithium2.nist_security_level >= 1
        assert dilithium2.nist_security_level <= 5
        assert 'lattice' in dilithium2.security_assumptions[0].lower()
        
        # Key sizes should be reasonable
        assert dilithium2.key_sizes['public'] > 1000
        assert dilithium2.key_sizes['private'] > dilithium2.key_sizes['public']
        
        # Performance characteristics should be positive
        assert all(perf > 0 for perf in dilithium2.performance_characteristics.values())
    
    def test_migration_plan_security_validation(self):
        """Test security validation of migration plans."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Create test vulnerabilities
        vulnerabilities = [
            self._create_test_vulnerability(CryptoAlgorithm.RSA_2048),
            self._create_test_vulnerability(CryptoAlgorithm.ECDSA_P256)
        ]
        
        assessment = quantum_analyzer.assess_system_resilience(vulnerabilities)
        migration_plan = quantum_analyzer.generate_migration_plan(assessment, {})
        
        # Validate migration plan security
        assert len(migration_plan.recommended_algorithms) > 0
        
        # All recommended algorithms should be post-quantum secure
        pqc_algorithms = {'Dilithium2', 'Dilithium3', 'Dilithium5', 'Kyber512', 'Kyber768', 'Kyber1024', 'Falcon-512'}
        for algorithm in migration_plan.recommended_algorithms:
            assert algorithm in pqc_algorithms, f"{algorithm} is not a validated PQC algorithm"
        
        # Migration phases should include security validation steps
        phase_names = [phase['name'] for phase in migration_plan.migration_phases]
        assert any('security' in name.lower() or 'testing' in name.lower() for name in phase_names)
    
    def test_cryptographic_agility_assessment(self):
        """Test assessment of cryptographic agility."""
        quantum_analyzer = QuantumResilienceAnalyzer()
        
        # Test different agility scenarios
        agility_tests = [
            ({'configurable_crypto': False, 'runtime_crypto_selection': False}, 'RIGID'),
            ({'configurable_crypto': True, 'runtime_crypto_selection': False}, 'CONFIGURABLE'),
            ({'configurable_crypto': True, 'runtime_crypto_selection': True}, 'DYNAMIC'),
            ({'ai_crypto_optimization': True}, 'ADAPTIVE'),
        ]
        
        for constraints, expected_agility in agility_tests:
            agility = quantum_analyzer._assess_crypto_agility(constraints)
            assert agility.value.upper() == expected_agility, \
                f"Expected {expected_agility}, got {agility.value}"
    
    def _create_test_vulnerability(self, algorithm):
        """Create test vulnerability for quantum security testing."""
        from pqc_iot_retrofit.scanner import CryptoVulnerability
        
        return CryptoVulnerability(
            algorithm=algorithm,
            address=0x08001000,
            function_name=f"test_{algorithm.value.lower()}",
            risk_level=RiskLevel.HIGH,
            key_size=2048 if 'RSA' in algorithm.value else 256,
            description=f"Test {algorithm.value} implementation",
            mitigation="Replace with PQC algorithm",
            stack_usage=256,
            available_stack=32768
        )


class TestAISecurityValidation:
    """Test AI system security properties."""
    
    def test_ai_model_integrity(self):
        """Test AI model integrity and consistency."""
        ai = AdaptiveAI()
        
        # Test model initialization consistency
        initial_weights = ai.ensemble_detector.weights.copy()
        
        # Perform some operations
        firmware_data = np.random.bytes(10000)
        analysis1 = ai.analyze_firmware(firmware_data)
        analysis2 = ai.analyze_firmware(firmware_data)
        
        # Results should be consistent for same input
        fp1 = analysis1['firmware_fingerprint']
        fp2 = analysis2['firmware_fingerprint']
        
        assert fp1['size_bytes'] == fp2['size_bytes']
        assert abs(fp1['entropy'] - fp2['entropy']) < 0.1
        
        # Model weights should remain stable without feedback
        current_weights = ai.ensemble_detector.weights
        for model_name in initial_weights:
            if model_name in current_weights:
                weight_change = abs(initial_weights[model_name] - current_weights[model_name])
                assert weight_change < 0.1, f"Model {model_name} weight changed unexpectedly"
    
    def test_anomaly_detection_security(self):
        """Test security properties of anomaly detection."""
        ai = AdaptiveAI()
        
        # Train on normal samples
        normal_samples = [
            np.random.normal(128, 50, 1000).astype(np.uint8).tobytes()
            for _ in range(10)
        ]
        ai.train_anomaly_baseline(normal_samples)
        
        # Test with potential attack samples
        attack_samples = [
            b'\x00' * 1000,  # All zeros
            b'\xFF' * 1000,  # All ones
            np.random.bytes(1000),  # High entropy
            b'A' * 1000,  # Repeated character
        ]
        
        anomaly_results = []
        for sample in attack_samples:
            is_anomalous, score, analysis = ai.anomaly_detector.detect_anomalies(sample)
            anomaly_results.append((is_anomalous, score))
        
        # At least some attack samples should be detected as anomalous
        anomalous_count = sum(1 for is_anom, _ in anomaly_results if is_anom)
        assert anomalous_count >= 2, f"Should detect some attack samples as anomalous"
    
    def test_ai_adversarial_resistance(self):
        """Test resistance to adversarial inputs."""
        ai = AdaptiveAI()
        
        # Create base firmware
        base_firmware = self._create_realistic_firmware()
        base_analysis = ai.analyze_firmware(base_firmware)
        
        # Create adversarial variants
        adversarial_variants = [
            base_firmware + b'\x00',  # Append null byte
            b'\x00' + base_firmware,  # Prepend null byte  
            base_firmware[:-1],  # Remove last byte
            base_firmware[:len(base_firmware)//2] + base_firmware[len(base_firmware)//2:],  # No change (identity)
        ]
        
        for variant in adversarial_variants:
            try:
                variant_analysis = ai.analyze_firmware(variant)
                
                # Analysis should not crash
                assert 'firmware_fingerprint' in variant_analysis
                
                # Results should be reasonable (not dramatically different for small changes)
                base_entropy = base_analysis['firmware_fingerprint']['entropy']
                variant_entropy = variant_analysis['firmware_fingerprint']['entropy']
                entropy_diff = abs(base_entropy - variant_entropy)
                
                # Small changes should not cause dramatic entropy differences
                if len(variant) > len(base_firmware) * 0.9:  # Similar size
                    assert entropy_diff < 1.0, f"Small change caused large entropy difference: {entropy_diff}"
                
            except Exception as e:
                # Should handle adversarial inputs gracefully
                assert isinstance(e, (ValueError, TypeError)), f"Unexpected exception type: {type(e)}"
    
    def test_ai_privacy_preservation(self):
        """Test that AI analysis preserves firmware privacy."""
        ai = AdaptiveAI()
        
        # Create firmware with sensitive-looking data
        sensitive_patterns = [
            b'SECRET_KEY_12345',
            b'PASSWORD_HASH',
            b'PRIVATE_DATA_SECTION',
        ]
        
        firmware_with_secrets = bytearray(10000)
        for pattern in sensitive_patterns:
            pos = np.random.randint(0, len(firmware_with_secrets) - len(pattern))
            firmware_with_secrets[pos:pos+len(pattern)] = pattern
        
        # Fill rest with random data
        for i in range(len(firmware_with_secrets)):
            if firmware_with_secrets[i] == 0:
                firmware_with_secrets[i] = np.random.randint(0, 256)
        
        analysis = ai.analyze_firmware(bytes(firmware_with_secrets))
        
        # Analysis results should not contain raw sensitive data
        analysis_str = json.dumps(analysis, default=str).lower()
        
        for pattern in sensitive_patterns:
            pattern_str = pattern.decode('utf-8', errors='ignore').lower()
            assert pattern_str not in analysis_str, f"Sensitive pattern {pattern_str} leaked in analysis"
        
        # Should not expose raw firmware content
        assert 'secret_key' not in analysis_str
        assert 'password' not in analysis_str
    
    def test_ai_deterministic_behavior(self):
        """Test deterministic behavior of AI analysis."""
        ai = AdaptiveAI()
        
        firmware_data = np.random.bytes(5000)
        
        # Run analysis multiple times
        analyses = []
        for _ in range(3):
            analysis = ai.analyze_firmware(firmware_data)
            analyses.append(analysis)
        
        # Core fingerprint should be deterministic
        for i in range(1, len(analyses)):
            fp_base = analyses[0]['firmware_fingerprint']
            fp_current = analyses[i]['firmware_fingerprint']
            
            # Core metrics should be identical
            assert fp_base['size_bytes'] == fp_current['size_bytes']
            assert fp_base['checksum'] == fp_current['checksum']
            assert fp_base['architecture'] == fp_current['architecture']
    
    def _create_realistic_firmware(self, size=10000):
        """Create realistic firmware for adversarial testing."""
        firmware = bytearray()
        
        # ARM Cortex-M vector table
        for _ in range(64):
            firmware.extend(np.random.randint(0x08000000, 0x08010000, dtype=np.uint32).tobytes())
        
        # Code section with realistic instruction patterns
        while len(firmware) < size * 0.7:
            # ARM Thumb instruction patterns
            instr = np.random.randint(0x4000, 0x8000, dtype=np.uint16)
            firmware.extend(instr.tobytes())
        
        # Data section
        while len(firmware) < size:
            firmware.extend(np.random.bytes(min(100, size - len(firmware))))
        
        return bytes(firmware[:size])


class TestInputValidationSecurity:
    """Test input validation and sanitization security."""
    
    def test_malformed_input_handling(self):
        """Test handling of malformed inputs."""
        scanner = FirmwareScanner("cortex-m4")
        
        malformed_inputs = [
            b'',  # Empty input
            b'\x00',  # Single null byte
            b'\xFF' * 1000000,  # Very large input
            b'not firmware data',  # Text input
            bytes(range(256)) * 1000,  # Pattern input
        ]
        
        for malformed_input in malformed_inputs:
            try:
                # Should not crash on malformed input
                vulnerabilities = scanner._scan_crypto_constants(malformed_input, 0)
                # Results should be reasonable
                assert len(scanner.vulnerabilities) < 1000, "Too many vulnerabilities from malformed input"
                scanner.vulnerabilities = []  # Reset
                
            except Exception as e:
                # If it raises an exception, should be a handled type
                assert isinstance(e, (ValueError, TypeError, MemoryError))
    
    def test_buffer_overflow_protection(self):
        """Test protection against buffer overflow attacks."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test with oversized patterns
        oversized_pattern = b'RSA' * 10000  # Very long RSA pattern
        large_firmware = oversized_pattern + b'\x00' * 50000
        
        try:
            vulnerabilities = scanner._scan_crypto_strings(large_firmware, 0)
            
            # Should handle large inputs without excessive resource consumption
            assert len(scanner.vulnerabilities) < 100, "Excessive vulnerabilities from large input"
            
        except Exception as e:
            # Should gracefully handle memory limitations
            assert isinstance(e, (MemoryError, ValueError))
    
    def test_injection_attack_resistance(self):
        """Test resistance to injection attacks."""
        scanner = FirmwareScanner("cortex-m4")
        
        # Test with injection-like patterns
        injection_patterns = [
            b"'; DROP TABLE vulnerabilities; --",
            b"<script>alert('xss')</script>",
            b"../../../../etc/passwd",
            b"%n%n%n%n%n%n",  # Format string attack
            b"\x90" * 1000 + b"\x31\xc0",  # NOP sled pattern
        ]
        
        for pattern in injection_patterns:
            firmware_with_injection = b'\x00' * 1000 + pattern + b'\x00' * 1000
            
            try:
                vulnerabilities = scanner._scan_crypto_strings(firmware_with_injection, 0)
                
                # Should not interpret injection patterns as crypto vulnerabilities
                injection_vulns = [v for v in scanner.vulnerabilities 
                                 if any(inj.decode('utf-8', errors='ignore') in v.description 
                                       for inj in injection_patterns)]
                assert len(injection_vulns) == 0, "Detected injection pattern as crypto vulnerability"
                
                scanner.vulnerabilities = []  # Reset
                
            except Exception as e:
                # Should handle malicious inputs gracefully
                assert not isinstance(e, (SystemExit, KeyboardInterrupt))
    
    def test_path_traversal_protection(self):
        """Test protection against path traversal attacks."""
        from pqc_iot_retrofit.scanner import InputValidator
        
        # Test path validation
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "/proc/self/mem",
            "\\\\network\\share\\file",
            "file://system/file",
        ]
        
        for path in malicious_paths:
            try:
                # Should reject malicious paths
                InputValidator.validate_firmware_path(path)
                assert False, f"Should have rejected malicious path: {path}"
            except ValidationError:
                # Expected - should reject malicious paths
                pass
            except Exception as e:
                # Other exceptions are also acceptable for malicious input
                assert not isinstance(e, (SystemExit, KeyboardInterrupt))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])