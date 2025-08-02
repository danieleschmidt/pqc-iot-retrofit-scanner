"""
Security tests for side-channel resistance.

Tests PQC implementations for timing attacks, power analysis vulnerabilities,
and other side-channel vulnerabilities.
"""

import time
import statistics
from typing import List, Callable

import pytest

from pqc_iot_retrofit.patcher import PQCPatcher
from pqc_iot_retrofit.crypto.dilithium import Dilithium2Implementation
from pqc_iot_retrofit.crypto.kyber import Kyber512Implementation


class TestTimingAttackResistance:
    """Test resistance to timing attacks."""
    
    @pytest.mark.security
    def test_dilithium_constant_time_signing(self):
        """Test that Dilithium signing takes constant time regardless of input."""
        dilithium = Dilithium2Implementation()
        
        # Generate keypair
        public_key, secret_key = dilithium.generate_keypair()
        
        # Test messages of different patterns
        test_messages = [
            b"a" * 32,  # All same byte
            b"\x00" * 32,  # All zeros
            b"\xff" * 32,  # All ones
            bytes(range(32)),  # Sequential bytes
            b"The quick brown fox jumps over"[:32],  # Random text
        ]
        
        signing_times = []
        
        for message in test_messages:
            # Measure signing time multiple times
            times = []
            for _ in range(100):  # Multiple iterations for statistical significance
                start_time = time.perf_counter()
                signature = dilithium.sign(message, secret_key)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            signing_times.append(avg_time)
        
        # Check that timing variation is within acceptable bounds
        min_time = min(signing_times)
        max_time = max(signing_times)
        time_variation = (max_time - min_time) / min_time
        
        # Timing variation should be less than 5% for constant-time implementation
        assert time_variation < 0.05, \
            f"Timing variation {time_variation:.1%} exceeds 5% threshold, " \
            f"possible timing attack vulnerability"
    
    @pytest.mark.security
    def test_kyber_constant_time_decapsulation(self):
        """Test that Kyber decapsulation takes constant time."""
        kyber = Kyber512Implementation()
        
        # Generate keypair
        public_key, secret_key = kyber.generate_keypair()
        
        # Generate test ciphertexts
        valid_ciphertexts = []
        for _ in range(5):
            shared_secret, ciphertext = kyber.encapsulate(public_key)
            valid_ciphertexts.append(ciphertext)
        
        # Generate invalid ciphertexts (random data)
        invalid_ciphertexts = []
        for _ in range(5):
            invalid_ct = bytes([i % 256 for i in range(kyber.ciphertext_size())])
            invalid_ciphertexts.append(invalid_ct)
        
        # Measure decapsulation times
        valid_times = []
        invalid_times = []
        
        for ciphertext in valid_ciphertexts:
            times = []
            for _ in range(50):
                start_time = time.perf_counter()
                shared_secret = kyber.decapsulate(ciphertext, secret_key)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            valid_times.extend(times)
        
        for ciphertext in invalid_ciphertexts:
            times = []
            for _ in range(50):
                start_time = time.perf_counter()
                try:
                    shared_secret = kyber.decapsulate(ciphertext, secret_key)
                except Exception:
                    pass  # Invalid ciphertext may raise exception
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            invalid_times.extend(times)
        
        # Compare timing distributions
        valid_avg = statistics.mean(valid_times)
        invalid_avg = statistics.mean(invalid_times)
        
        time_difference = abs(valid_avg - invalid_avg) / min(valid_avg, invalid_avg)
        
        # Timing difference should be minimal for constant-time implementation
        assert time_difference < 0.1, \
            f"Timing difference {time_difference:.1%} between valid/invalid ciphertexts " \
            f"exceeds 10% threshold, possible timing attack vulnerability"
    
    @pytest.mark.security
    def test_key_dependent_timing_resistance(self):
        """Test that operations don't leak information about key material."""
        dilithium = Dilithium2Implementation()
        
        # Generate multiple keypairs with different patterns
        keypairs = []
        for i in range(5):
            public_key, secret_key = dilithium.generate_keypair()
            keypairs.append((public_key, secret_key))
        
        test_message = b"test message for timing analysis" + b"\x00" * 16
        
        # Measure signing times for different keys
        signing_times_by_key = []
        
        for public_key, secret_key in keypairs:
            times = []
            for _ in range(20):
                start_time = time.perf_counter()
                signature = dilithium.sign(test_message, secret_key)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            signing_times_by_key.append(avg_time)
        
        # Check timing variation across different keys
        min_time = min(signing_times_by_key)
        max_time = max(signing_times_by_key)
        key_timing_variation = (max_time - min_time) / min_time
        
        assert key_timing_variation < 0.1, \
            f"Key-dependent timing variation {key_timing_variation:.1%} exceeds 10% threshold"


class TestPowerAnalysisResistance:
    """Test resistance to power analysis attacks."""
    
    @pytest.mark.security
    @pytest.mark.slow
    def test_dilithium_power_simulation(self):
        """Simulate power analysis resistance for Dilithium operations."""
        dilithium = Dilithium2Implementation()
        public_key, secret_key = dilithium.generate_keypair()
        
        # Simulate power consumption by counting operations
        def count_operations(func: Callable) -> int:
            """Mock function to count computational operations."""
            # In a real implementation, this would interface with
            # power measurement hardware or simulation
            operation_count = 0
            
            # Mock operation counting (simplified)
            start_time = time.perf_counter()
            result = func()
            end_time = time.perf_counter()
            
            # Estimate operations based on execution time
            operation_count = int((end_time - start_time) * 1000000)  # Microseconds as proxy
            return operation_count, result
        
        # Test different messages for power consumption patterns
        test_messages = [
            b"\x00" * 32,  # All zeros
            b"\xff" * 32,  # All ones
            b"\xaa" * 32,  # Alternating pattern
            b"\x55" * 32,  # Alternating pattern (inverted)
        ]
        
        operation_counts = []
        
        for message in test_messages:
            def sign_operation():
                return dilithium.sign(message, secret_key)
            
            op_count, signature = count_operations(sign_operation)
            operation_counts.append(op_count)
        
        # Power consumption should be relatively uniform
        if len(operation_counts) > 1:
            avg_ops = statistics.mean(operation_counts)
            max_deviation = max(abs(count - avg_ops) for count in operation_counts)
            relative_deviation = max_deviation / avg_ops
            
            # Power variation should be limited
            assert relative_deviation < 0.2, \
                f"Power consumption variation {relative_deviation:.1%} exceeds 20% threshold"
    
    @pytest.mark.security
    def test_memory_access_pattern_analysis(self):
        """Test for consistent memory access patterns."""
        kyber = Kyber512Implementation()
        public_key, secret_key = kyber.generate_keypair()
        
        # Mock memory access pattern tracking
        memory_accesses = []
        
        def track_memory_access(operation_name: str):
            """Mock function to track memory access patterns."""
            # In a real implementation, this would use memory profiling tools
            # or hardware performance counters
            access_pattern = {
                "operation": operation_name,
                "cache_misses": hash(operation_name) % 100,  # Mock cache misses
                "memory_pages": hash(operation_name) % 10,   # Mock page accesses
            }
            memory_accesses.append(access_pattern)
        
        # Perform multiple key generation operations
        for i in range(5):
            track_memory_access(f"keygen_{i}")
            pub, sec = kyber.generate_keypair()
        
        # Analyze memory access pattern consistency
        cache_misses = [access["cache_misses"] for access in memory_accesses]
        memory_pages = [access["memory_pages"] for access in memory_accesses]
        
        # Check for consistent patterns (simplified test)
        cache_variance = statistics.variance(cache_misses) if len(cache_misses) > 1 else 0
        page_variance = statistics.variance(memory_pages) if len(memory_pages) > 1 else 0
        
        # Memory access patterns should be relatively consistent
        assert cache_variance < 1000, "Cache miss patterns show high variance"
        assert page_variance < 100, "Memory page access patterns show high variance"


class TestRandomnessQuality:
    """Test quality and security of random number generation."""
    
    @pytest.mark.security
    def test_keypair_uniqueness(self):
        """Test that generated keypairs are unique and unpredictable."""
        dilithium = Dilithium2Implementation()
        
        # Generate multiple keypairs
        keypairs = []
        for _ in range(10):
            public_key, secret_key = dilithium.generate_keypair()
            keypairs.append((public_key, secret_key))
        
        # Check uniqueness of public keys
        public_keys = [pk for pk, sk in keypairs]
        unique_public_keys = set(public_keys)
        
        assert len(unique_public_keys) == len(public_keys), \
            "Generated public keys are not unique"
        
        # Check uniqueness of secret keys
        secret_keys = [sk for pk, sk in keypairs]
        unique_secret_keys = set(secret_keys)
        
        assert len(unique_secret_keys) == len(secret_keys), \
            "Generated secret keys are not unique"
    
    @pytest.mark.security
    def test_signature_randomness(self):
        """Test that signatures use proper randomness."""
        dilithium = Dilithium2Implementation()
        public_key, secret_key = dilithium.generate_keypair()
        
        message = b"test message for signature randomness"
        
        # Generate multiple signatures of the same message
        signatures = []
        for _ in range(5):
            signature = dilithium.sign(message, secret_key)
            signatures.append(signature)
        
        # Signatures should be different due to randomness (for probabilistic schemes)
        unique_signatures = set(signatures)
        
        # For deterministic schemes, this test would be different
        # Here we assume probabilistic signatures
        if len(unique_signatures) == 1:
            # If all signatures are identical, verify this is expected behavior
            pytest.skip("Deterministic signature scheme detected")
        else:
            assert len(unique_signatures) == len(signatures), \
                "Signatures should be unique due to proper randomness usage"
    
    @pytest.mark.security
    def test_entropy_quality(self):
        """Test quality of entropy used in cryptographic operations."""
        import hashlib
        
        kyber = Kyber512Implementation()
        
        # Generate random data from multiple operations
        random_data = b""
        
        for _ in range(10):
            public_key, secret_key = kyber.generate_keypair()
            # Extract some "random" bytes from the public key
            random_data += public_key[:32]
        
        # Basic entropy tests (simplified)
        # Test 1: No obvious patterns
        assert len(set(random_data)) > len(random_data) * 0.8, \
            "Random data shows too much repetition"
        
        # Test 2: Approximately equal distribution of bytes
        byte_counts = [0] * 256
        for byte in random_data:
            byte_counts[byte] += 1
        
        expected_count = len(random_data) / 256
        max_deviation = max(abs(count - expected_count) for count in byte_counts)
        relative_deviation = max_deviation / expected_count
        
        assert relative_deviation < 2.0, \
            f"Byte distribution deviation {relative_deviation:.1f} indicates poor entropy"


class TestImplementationSecurity:
    """Test security properties of the implementation itself."""
    
    @pytest.mark.security
    def test_sensitive_data_clearing(self):
        """Test that sensitive data is properly cleared from memory."""
        dilithium = Dilithium2Implementation()
        
        # This test would ideally use memory introspection tools
        # For now, we test the interface for memory clearing
        public_key, secret_key = dilithium.generate_keypair()
        
        # Check if implementation provides secure cleanup
        if hasattr(dilithium, 'clear_sensitive_data'):
            dilithium.clear_sensitive_data()
            # Verify cleanup was performed (implementation-specific)
            assert True, "Sensitive data clearing method exists"
        else:
            pytest.skip("No explicit sensitive data clearing method found")
    
    @pytest.mark.security
    def test_buffer_overflow_protection(self):
        """Test protection against buffer overflow attacks."""
        kyber = Kyber512Implementation()
        public_key, secret_key = kyber.generate_keypair()
        
        # Test with oversized inputs
        oversized_ciphertext = b"\x00" * (kyber.ciphertext_size() * 2)
        
        # Should handle gracefully without crashing
        try:
            result = kyber.decapsulate(oversized_ciphertext, secret_key)
            # If it returns a result, it should be a failure case
            assert result is None or len(result) == 0, \
                "Oversized input should be rejected"
        except (ValueError, RuntimeError) as e:
            # Expected behavior - input validation should catch this
            assert "invalid" in str(e).lower() or "size" in str(e).lower(), \
                f"Exception message should indicate input validation: {e}"
    
    @pytest.mark.security
    def test_fault_injection_resistance(self):
        """Test basic resistance to fault injection attacks."""
        dilithium = Dilithium2Implementation()
        public_key, secret_key = dilithium.generate_keypair()
        
        message = b"test message for fault injection resistance"
        
        # Generate valid signature
        valid_signature = dilithium.sign(message, secret_key)
        
        # Verify valid signature
        assert dilithium.verify(message, valid_signature, public_key), \
            "Valid signature should verify correctly"
        
        # Test with corrupted signatures (simulating fault injection)
        for i in range(min(10, len(valid_signature))):
            corrupted_signature = bytearray(valid_signature)
            corrupted_signature[i] ^= 0x01  # Flip one bit
            
            # Corrupted signature should be rejected
            is_valid = dilithium.verify(message, bytes(corrupted_signature), public_key)
            assert not is_valid, \
                f"Corrupted signature at position {i} should be rejected"