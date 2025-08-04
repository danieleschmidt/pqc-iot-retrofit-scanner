"""
End-to-end integration tests for the PQC IoT Retrofit Scanner.

These tests verify the complete workflow from firmware scanning
to patch generation and binary modification.
"""

import pytest
import tempfile
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoVulnerability, CryptoAlgorithm, RiskLevel
from pqc_iot_retrofit.patcher import PQCPatcher, PQCAlgorithm
from pqc_iot_retrofit.pqc_implementations import create_pqc_implementation
from pqc_iot_retrofit.binary_patcher import BinaryPatcher, PatchType


class TestE2EPQCWorkflow:
    """End-to-end tests for the complete PQC retrofit workflow."""
    
    @pytest.fixture
    def sample_firmware(self):
        """Create a sample firmware binary with known crypto patterns."""
        
        # Create firmware with RSA signature pattern
        firmware_data = bytearray(4096)  # 4KB firmware
        
        # Add ARM Thumb function prologue at 0x100
        firmware_data[0x100:0x104] = struct.pack('<HH', 0xB580, 0xB083)  # push {r7, lr}; sub sp, #12
        
        # Add RSA-65537 constant at 0x110 (common RSA public exponent)
        firmware_data[0x110:0x114] = struct.pack('<I', 0x00010001)  # 65537 in little-endian
        
        # Add modular arithmetic operations (simplified RSA pattern)
        firmware_data[0x120:0x130] = bytes([
            0x00, 0xF0, 0x00, 0xF8,  # BL (call instruction)
            0x70, 0x47,              # BX LR (return)
            0x00, 0xBF,              # NOP
            0x00, 0xBF,              # NOP
        ])
        
        # Add ECC P-256 curve parameter at 0x200
        firmware_data[0x200:0x208] = b'\xff\xff\xff\xff\x00\x00\x00\x01'  # P-256 prime part
        
        # Add some ECDSA-like operations
        firmware_data[0x210:0x220] = bytes([
            0x40, 0xF2, 0x00, 0x00,  # movw r0, #0
            0xC0, 0xF2, 0x00, 0x00,  # movt r0, #0
            0x00, 0x68,              # ldr r0, [r0]
            0x40, 0x1C,              # add r0, #1 (point arithmetic pattern)
        ])
        
        return firmware_data
    
    @pytest.fixture
    def temp_firmware_file(self, sample_firmware):
        """Create temporary firmware file."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            f.write(sample_firmware)
            firmware_path = f.name
        
        yield firmware_path
        
        # Cleanup
        Path(firmware_path).unlink(missing_ok=True)
    
    def test_complete_scan_and_patch_workflow(self, temp_firmware_file):
        """Test the complete workflow from scanning to patching."""
        
        # Step 1: Scan firmware for vulnerabilities
        scanner = FirmwareScanner('cortex-m4', {'flash': 512*1024, 'ram': 128*1024})
        vulnerabilities = scanner.scan_firmware(temp_firmware_file)
        
        # Verify vulnerabilities were found
        assert len(vulnerabilities) > 0, "Should detect vulnerabilities in sample firmware"
        
        # Check that we found RSA vulnerability
        rsa_vulns = [v for v in vulnerabilities if 'RSA' in v.algorithm.value]
        assert len(rsa_vulns) > 0, "Should detect RSA vulnerability"
        
        # Step 2: Generate PQC patches
        patcher = PQCPatcher('STM32L4', 'balanced')
        
        patches_created = []
        for vuln in vulnerabilities[:2]:  # Test first 2 vulnerabilities
            try:
                if 'RSA' in vuln.algorithm.value or 'ECDSA' in vuln.algorithm.value:
                    patch = patcher.create_dilithium_patch(vuln, security_level=2)
                else:
                    patch = patcher.create_kyber_patch(vuln, security_level=1)
                
                patches_created.append(patch)
                
            except Exception as e:
                print(f"Note: Patch creation failed for {vuln.function_name}: {e}")
        
        assert len(patches_created) > 0, "Should create at least one patch"
        
        # Step 3: Verify patch metadata
        for patch in patches_created:
            assert patch.target_address > 0
            assert len(patch.replacement_code) > 0
            assert patch.verification_hash
            assert patch.patch_metadata['algorithm'] in ['dilithium2', 'dilithium3', 'kyber512', 'kyber768']
    
    def test_pqc_implementation_generation(self):
        """Test PQC implementation generation for different algorithms."""
        
        # Test Kyber-512 generation
        kyber_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'balanced')
        
        assert kyber_impl.algorithm == 'kyber512'
        assert kyber_impl.target_arch == 'cortex-m4'
        assert len(kyber_impl.c_code) > 1000  # Should be substantial implementation
        assert len(kyber_impl.header_code) > 500
        assert kyber_impl.performance_estimates['keygen_cycles'] > 0
        assert kyber_impl.memory_usage['public_key'] == 800  # Kyber-512 public key size
        
        # Test Dilithium-2 generation
        dilithium_impl = create_pqc_implementation('dilithium2', 'esp32', 'size')
        
        assert dilithium_impl.algorithm == 'dilithium2'
        assert dilithium_impl.target_arch == 'esp32'
        assert len(dilithium_impl.c_code) > 2000  # Dilithium is more complex
        assert dilithium_impl.memory_usage['signature'] == 2420  # Dilithium-2 signature size
    
    def test_binary_patcher_functionality(self, temp_firmware_file):
        """Test binary patching capabilities."""
        
        patcher = BinaryPatcher('arm')
        
        # Create a simple patch (modify bytes at specific location)
        original_data = b'\x00\xF0\x00\xF8'  # Original BL instruction
        replacement_data = b'\x00\xBF\x00\xBF'  # Two NOPs
        
        from pqc_iot_retrofit.binary_patcher import create_pqc_patch_info
        patch_info = create_pqc_patch_info(
            target_address=0x120,
            original_data=original_data,
            replacement_data=replacement_data,
            patch_type=PatchType.INLINE_PATCH
        )
        
        # Create output file
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            output_path = f.name
        
        try:
            # Apply patch
            success = patcher.patch_firmware(temp_firmware_file, [patch_info], output_path)
            assert success, "Patch should be applied successfully"
            
            # Verify patch was applied
            with open(output_path, 'rb') as f:
                patched_data = f.read()
            
            # Check that the patch was applied at the correct location
            assert patched_data[0x120:0x124] == replacement_data
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_ota_package_creation(self, temp_firmware_file):
        """Test OTA package creation and serialization."""
        
        # Create some mock vulnerabilities
        vuln1 = CryptoVulnerability(
            algorithm=CryptoAlgorithm.RSA_2048,
            address=0x100,
            function_name="rsa_sign_test",
            risk_level=RiskLevel.CRITICAL,
            key_size=2048,
            description="Test RSA vulnerability",
            mitigation="Replace with Dilithium",
            stack_usage=512,
            available_stack=8192
        )
        
        patcher = PQCPatcher('STM32L4', 'balanced')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the binary extraction and compilation for testing
            with patch.object(patcher.binary_patcher, 'extract_function_binary') as mock_extract:
                with patch.object(patcher, '_compile_implementation') as mock_compile:
                    
                    # Setup mocks
                    mock_extract.return_value = b'\x00\xF0\x00\xF8' * 10  # 40 bytes of mock binary
                    mock_compile.return_value = b'COMPILED_PQC_BINARY:100:' + b'mock_pqc_code' * 10
                    
                    # Create OTA package
                    ota_package = patcher.create_ota_update(
                        temp_firmware_file,
                        [vuln1],
                        temp_dir,
                        version="1.0.0"
                    )
                    
                    assert ota_package is not None
                    assert ota_package.version == "1.0.0"
                    assert ota_package.target_device == "STM32L4"
                    assert len(ota_package.patches) > 0
                    
                    # Verify files were created
                    ota_files = list(Path(temp_dir).glob("*.ota"))
                    assert len(ota_files) == 1
                    
                    firmware_files = list(Path(temp_dir).glob("firmware_*.bin"))
                    assert len(firmware_files) == 1
                    
                    script_files = list(Path(temp_dir).glob("*.sh"))
                    assert len(script_files) >= 1  # At least deploy.sh
    
    def test_performance_benchmarking(self):
        """Test performance estimation and benchmarking."""
        
        implementations = [
            ('kyber512', 'cortex-m4', 'speed'),
            ('kyber512', 'cortex-m4', 'size'), 
            ('dilithium2', 'esp32', 'balanced'),
            ('dilithium3', 'cortex-m4', 'speed'),
        ]
        
        for algo, arch, opt in implementations:
            impl = create_pqc_implementation(algo, arch, opt)
            
            # Verify performance estimates are reasonable
            perf = impl.performance_estimates
            
            if 'kyber' in algo:
                # Kyber should be faster than Dilithium
                assert perf['keygen_cycles'] < 2000000
                assert perf['encaps_cycles'] < 3000000
                assert perf['decaps_cycles'] < 3000000
            else:  # Dilithium
                # Dilithium signing is more expensive
                assert perf['sign_cycles'] > perf['keygen_cycles']
                assert perf['verify_cycles'] < perf['sign_cycles']
            
            # Memory usage should be reasonable for embedded systems
            memory = impl.memory_usage
            assert memory['stack_peak'] < 16*1024  # Less than 16KB stack
            assert memory['flash_usage'] < 200*1024  # Less than 200KB flash
    
    def test_architecture_specific_optimizations(self):
        """Test that different architectures generate different optimizations."""
        
        architectures = ['cortex-m4', 'esp32', 'riscv32']
        
        for arch in architectures:
            impl = create_pqc_implementation('kyber512', arch, 'speed')
            
            # Should generate architecture-specific code
            assert arch in impl.c_code or arch in impl.assembly_code
            
            # Architecture-specific optimizations should be present
            if arch == 'cortex-m4':
                # Should mention DSP instructions or ARM-specific optimizations
                assert any(keyword in impl.assembly_code.lower() 
                          for keyword in ['umull', 'umlal', 'sadd16', 'cortex'])
            elif arch == 'esp32':
                # Should mention Xtensa-specific optimizations
                assert any(keyword in impl.assembly_code.lower()
                          for keyword in ['xtensa', 'retw', 'movi'])
    
    def test_error_handling_and_validation(self, temp_firmware_file):
        """Test error handling for invalid inputs and edge cases."""
        
        # Test invalid architecture
        with pytest.raises(ValueError):
            FirmwareScanner('invalid-arch')
        
        # Test invalid firmware file
        scanner = FirmwareScanner('cortex-m4')
        with pytest.raises(FileNotFoundError):
            scanner.scan_firmware('nonexistent_file.bin')
        
        # Test invalid PQC algorithm
        with pytest.raises(ValueError):
            create_pqc_implementation('invalid_algo', 'cortex-m4', 'balanced')
        
        # Test insufficient memory constraints
        patcher = PQCPatcher('STM32L4')
        
        # Create vulnerability with very limited memory
        limited_vuln = CryptoVulnerability(
            algorithm=CryptoAlgorithm.RSA_2048,
            address=0x100,
            function_name="test_func",
            risk_level=RiskLevel.CRITICAL,
            key_size=2048,
            description="Test",
            mitigation="Test",
            stack_usage=512,
            available_stack=1024  # Very limited
        )
        
        # Should raise error for insufficient memory
        with pytest.raises(ValueError, match="Insufficient stack"):
            patcher.create_dilithium_patch(limited_vuln, security_level=5)  # Dilithium-5 needs more memory
    
    def test_integration_with_real_patterns(self, temp_firmware_file):
        """Test integration with realistic crypto patterns."""
        
        # Create more realistic firmware patterns
        firmware_data = bytearray(8192)  # 8KB firmware
        
        # Add realistic ARM function with RSA operations
        # Function prologue
        firmware_data[0x1000:0x1004] = struct.pack('<HH', 0xB580, 0xB083)  # push {r7, lr}; sub sp, #12
        
        # Montgomery multiplication pattern (common in RSA)
        firmware_data[0x1010:0x1020] = struct.pack('<HHHHHHHH',
            0x4340,  # mul r0, r0, r0  (squaring operation)
            0x4449,  # add r1, r1, r1  (doubling)
            0xEB01, 0x0100,  # add.w r1, r1, r0, lsl #0
            0xFBB0, 0xF0F1,  # udiv r0, r0, r1 (division in Montgomery)
            0x4770,  # bx lr (return)
            0xBF00,  # nop
        )
        
        # Update firmware file
        with open(temp_firmware_file, 'wb') as f:
            f.write(firmware_data)
        
        # Scan with enhanced patterns
        scanner = FirmwareScanner('cortex-m4', {'flash': 512*1024, 'ram': 128*1024})
        vulnerabilities = scanner.scan_firmware(temp_firmware_file)
        
        # Should detect the Montgomery multiplication pattern
        assert len(vulnerabilities) > 0
        
        # At least one should be high or critical risk
        high_risk_vulns = [v for v in vulnerabilities if v.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        assert len(high_risk_vulns) > 0


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for PQC implementations."""
    
    def test_kyber_performance_targets(self):
        """Test that Kyber implementations meet performance targets."""
        
        impl = create_pqc_implementation('kyber512', 'cortex-m4', 'speed')
        perf = impl.performance_estimates
        
        # Performance targets (cycles) for Cortex-M4 @ 80MHz
        assert perf['keygen_cycles'] < 1500000, "Key generation should be under 1.5M cycles"
        assert perf['encaps_cycles'] < 1800000, "Encapsulation should be under 1.8M cycles"  
        assert perf['decaps_cycles'] < 2000000, "Decapsulation should be under 2M cycles"
        
        # Memory targets
        memory = impl.memory_usage
        assert memory['stack_peak'] < 4096, "Stack usage should be under 4KB"
        assert memory['flash_usage'] < 25000, "Flash usage should be under 25KB"
    
    def test_dilithium_performance_targets(self):
        """Test that Dilithium implementations meet performance targets."""
        
        impl = create_pqc_implementation('dilithium2', 'cortex-m4', 'balanced')
        perf = impl.performance_estimates
        
        # Performance targets for Dilithium-2
        assert perf['keygen_cycles'] < 3000000, "Key generation should be under 3M cycles"
        assert perf['sign_cycles'] < 12000000, "Signing should be under 12M cycles"
        assert perf['verify_cycles'] < 4000000, "Verification should be under 4M cycles"
        
        # Memory targets
        memory = impl.memory_usage
        assert memory['stack_peak'] < 8192, "Stack usage should be under 8KB"
        assert memory['flash_usage'] < 130000, "Flash usage should be under 130KB"
    
    def test_optimization_levels_impact(self):
        """Test that optimization levels have measurable impact."""
        
        speed_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'speed')
        size_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'size')
        balanced_impl = create_pqc_implementation('kyber512', 'cortex-m4', 'balanced')
        
        # Speed optimization should be faster but larger
        assert speed_impl.performance_estimates['keygen_cycles'] <= balanced_impl.performance_estimates['keygen_cycles']
        assert speed_impl.memory_usage['flash_usage'] >= size_impl.memory_usage['flash_usage']
        
        # Size optimization should be smaller but potentially slower
        assert size_impl.memory_usage['flash_usage'] <= balanced_impl.memory_usage['flash_usage']
        
        # Balanced should be between speed and size
        balanced_cycles = balanced_impl.performance_estimates['keygen_cycles']
        speed_cycles = speed_impl.performance_estimates['keygen_cycles']
        
        # Balanced should not be much slower than speed-optimized
        assert balanced_cycles <= speed_cycles * 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])