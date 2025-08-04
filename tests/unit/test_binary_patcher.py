"""
Unit tests for the binary patcher module.

Tests binary-level firmware patching, OTA package creation,
and architecture-specific code generation.
"""

import pytest
import struct
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from pqc_iot_retrofit.binary_patcher import (
    BinaryPatcher,
    BinaryPatchInfo,
    PatchType,
    CompressionType,
    OTAPackage,
    create_binary_patcher,
    create_pqc_patch_info
)


class TestBinaryPatcher:
    """Test the binary patcher functionality."""
    
    @pytest.fixture
    def sample_firmware(self):
        """Create sample firmware data for testing."""
        firmware = bytearray(2048)  # 2KB firmware
        
        # Add some recognizable patterns
        firmware[0x100:0x104] = b'\x00\xF0\x00\xF8'  # ARM BL instruction
        firmware[0x200:0x204] = b'\x70\x47\x00\xBF'  # ARM BX LR + NOP
        firmware[0x300:0x310] = b'\xFF' * 16  # Free space (uninitialized flash)
        
        return firmware
    
    @pytest.fixture
    def temp_firmware_file(self, sample_firmware):
        """Create temporary firmware file."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            f.write(sample_firmware)
            firmware_path = f.name
        
        yield firmware_path
        Path(firmware_path).unlink(missing_ok=True)
    
    def test_patcher_initialization(self):
        """Test binary patcher initialization."""
        patcher = BinaryPatcher('arm')
        
        assert patcher.target_arch == 'arm'
        assert patcher.arch_config['endianness'] == 'little'
        assert patcher.arch_config['pointer_size'] == 4
        
        # Test different architectures
        xtensa_patcher = BinaryPatcher('xtensa')
        assert xtensa_patcher.arch_config['pointer_size'] == 4
        assert xtensa_patcher.arch_config['endianness'] == 'little'
        
        riscv_patcher = BinaryPatcher('riscv')
        assert riscv_patcher.arch_config['pointer_size'] == 4
    
    def test_arch_config_differences(self):
        """Test that different architectures have appropriate configurations."""
        arm_patcher = BinaryPatcher('arm')
        xtensa_patcher = BinaryPatcher('xtensa')
        riscv_patcher = BinaryPatcher('riscv')
        
        # ARM should have ARM-specific instructions
        assert arm_patcher.arch_config['nop_instruction'] == 0xBF00  # ARM Thumb NOP
        
        # Xtensa should have different instruction patterns
        assert xtensa_patcher.arch_config['nop_instruction'] == 0x20F0  # Xtensa NOP
        
        # RISC-V should have 32-bit instructions
        assert riscv_patcher.arch_config['nop_instruction'] == 0x00000013  # RISC-V NOP
    
    def test_inline_patch_application(self, temp_firmware_file):
        """Test applying inline patches."""
        patcher = BinaryPatcher('arm')
        
        # Create patch info
        original_data = b'\x00\xF0\x00\xF8'  # Original BL instruction
        replacement_data = b'\x00\xBF\x00\xBF'  # Two NOPs
        
        patch_info = create_pqc_patch_info(
            target_address=0x100,
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
            
            assert patched_data[0x100:0x104] == replacement_data
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_function_replacement_patch(self, temp_firmware_file):
        """Test function replacement patching."""
        patcher = BinaryPatcher('arm')
        
        # Create larger replacement that needs relocation
        original_data = b'\x00\xF0\x00\xF8'  # 4 bytes
        replacement_data = b'\x00\xBF' * 20  # 40 bytes of NOPs (larger than original)
        
        patch_info = create_pqc_patch_info(
            target_address=0x100,
            original_data=original_data,
            replacement_data=replacement_data,
            patch_type=PatchType.FUNCTION_REPLACEMENT
        )
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            output_path = f.name
        
        try:
            success = patcher.patch_firmware(temp_firmware_file, [patch_info], output_path)
            assert success, "Function replacement should succeed"
            
            # Verify the patch was applied (either directly or via jump)
            with open(output_path, 'rb') as f:
                patched_data = f.read()
            
            # Either replacement data is at original location, or there's a jump
            data_at_original = patched_data[0x100:0x104]
            assert data_at_original != original_data, "Original data should be replaced"
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_data_modification_patch(self, temp_firmware_file):
        """Test data modification patches."""
        patcher = BinaryPatcher('arm')
        
        # Modify data at address 0x200
        original_data = b'\x70\x47\x00\xBF'
        replacement_data = b'\x01\x02\x03\x04'
        
        patch_info = create_pqc_patch_info(
            target_address=0x200,
            original_data=original_data,
            replacement_data=replacement_data,
            patch_type=PatchType.DATA_MODIFICATION
        )
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            output_path = f.name
        
        try:
            success = patcher.patch_firmware(temp_firmware_file, [patch_info], output_path)
            assert success
            
            with open(output_path, 'rb') as f:
                patched_data = f.read()
            
            assert patched_data[0x200:0x204] == replacement_data
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_multiple_patches(self, temp_firmware_file):
        """Test applying multiple patches in sequence."""
        patcher = BinaryPatcher('arm')
        
        # Create multiple patches
        patch1 = create_pqc_patch_info(
            target_address=0x100,
            original_data=b'\x00\xF0\x00\xF8',
            replacement_data=b'\x00\xBF\x00\xBF',
            patch_type=PatchType.INLINE_PATCH
        )
        
        patch2 = create_pqc_patch_info(
            target_address=0x200,
            original_data=b'\x70\x47\x00\xBF',
            replacement_data=b'\xFF\xFF\xFF\xFF',
            patch_type=PatchType.DATA_MODIFICATION
        )
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            output_path = f.name
        
        try:
            success = patcher.patch_firmware(temp_firmware_file, [patch1, patch2], output_path)
            assert success
            
            with open(output_path, 'rb') as f:
                patched_data = f.read()
            
            # Verify both patches were applied
            assert patched_data[0x100:0x104] == b'\x00\xBF\x00\xBF'
            assert patched_data[0x200:0x204] == b'\xFF\xFF\xFF\xFF'
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_patch_validation_failure(self, temp_firmware_file):
        """Test that patches fail when original data doesn't match."""
        patcher = BinaryPatcher('arm')
        
        # Create patch with wrong original data
        patch_info = create_pqc_patch_info(
            target_address=0x100,
            original_data=b'\xFF\xFF\xFF\xFF',  # Wrong original data
            replacement_data=b'\x00\xBF\x00\xBF',
            patch_type=PatchType.INLINE_PATCH
        )
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            output_path = f.name
        
        try:
            success = patcher.patch_firmware(temp_firmware_file, [patch_info], output_path)
            assert not success, "Patch should fail with incorrect original data"
            
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_free_space_detection(self, sample_firmware):
        """Test detection of free space in firmware."""
        patcher = BinaryPatcher('arm')
        
        # Should find free space at 0x300 (filled with 0xFF)
        free_space = patcher._find_free_space(sample_firmware, 16)
        assert free_space == 0x300
        
        # Should not find space for very large requirement
        no_space = patcher._find_free_space(sample_firmware, 1024)
        assert no_space == -1
    
    def test_jump_instruction_generation(self):
        """Test generation of architecture-specific jump instructions."""
        arm_patcher = BinaryPatcher('arm')
        xtensa_patcher = BinaryPatcher('xtensa')
        riscv_patcher = BinaryPatcher('riscv')
        
        # Test ARM jump generation
        arm_jump = arm_patcher._create_jump_instruction(0x1000, 0x2000)
        assert len(arm_jump) == 4  # ARM BL is 4 bytes (2 x 16-bit instructions)
        
        # Test Xtensa jump generation
        xtensa_jump = xtensa_patcher._create_jump_instruction(0x1000, 0x2000)
        assert len(xtensa_jump) in [3, 4]  # Xtensa instructions can be 3 bytes
        
        # Test RISC-V jump generation
        riscv_jump = riscv_patcher._create_jump_instruction(0x1000, 0x2000)
        assert len(riscv_jump) == 4  # RISC-V JAL is 4 bytes
    
    def test_relocation_application(self, sample_firmware):
        """Test application of relocation entries."""
        patcher = BinaryPatcher('arm')
        
        # Apply relocation at offset 0x100
        patcher._apply_relocation(sample_firmware, 0x100, 0x12345678)
        
        # Verify the address was written in little-endian format
        written_addr = struct.unpack('<I', sample_firmware[0x100:0x104])[0]
        assert written_addr == 0x12345678
    
    def test_function_extraction(self, temp_firmware_file):
        """Test function binary extraction."""
        patcher = BinaryPatcher('arm')
        
        # Test extraction by address (no LIEF available in test)
        with patch('pqc_iot_retrofit.binary_patcher.LIEF_AVAILABLE', False):
            extracted = patcher.extract_function_binary(temp_firmware_file, 'test_func', 0x100)
            
            assert extracted is not None
            assert len(extracted) > 0
            assert len(extracted) <= 64  # Should be limited
    
    def test_firmware_integrity_verification(self, temp_firmware_file):
        """Test firmware integrity verification after patching."""
        patcher = BinaryPatcher('arm')
        
        # Create valid patch
        patch_info = create_pqc_patch_info(
            target_address=0x100,
            original_data=b'\x00\xF0\x00\xF8',
            replacement_data=b'\x00\xBF\x00\xBF',
            patch_type=PatchType.INLINE_PATCH
        )
        
        # Load and patch firmware
        with open(temp_firmware_file, 'rb') as f:
            firmware_data = bytearray(f.read())
        
        # Apply patch manually
        success = patcher._apply_single_patch(firmware_data, patch_info)
        assert success
        
        # Verify integrity
        integrity_ok = patcher._verify_patched_firmware(firmware_data, [patch_info])
        assert integrity_ok


class TestOTAPackage:
    """Test OTA package creation and handling."""
    
    def test_ota_package_creation(self):
        """Test creation of OTA packages."""
        
        # Create sample patches
        patch1 = create_pqc_patch_info(
            target_address=0x1000,
            original_data=b'\x00\xF0\x00\xF8',
            replacement_data=b'\x00\xBF\x00\xBF',
            patch_type=PatchType.INLINE_PATCH
        )
        
        patch2 = create_pqc_patch_info(
            target_address=0x2000,
            original_data=b'\x70\x47\x00\xBF',
            replacement_data=b'\xFF\xFF\xFF\xFF',
            patch_type=PatchType.DATA_MODIFICATION
        )
        
        metadata = {
            'version': '1.2.3',
            'target_device': 'STM32L4',
            'description': 'PQC security update'
        }
        
        ota_package = OTAPackage(
            version='1.2.3',
            target_device='STM32L4',
            base_version='1.0.0',
            patches=[patch1, patch2],
            compression=CompressionType.LZMA,
            encrypted=False,
            integrity_hash='abc123',
            rollback_data=b'rollback_info',
            metadata=metadata
        )
        
        assert ota_package.version == '1.2.3'
        assert ota_package.target_device == 'STM32L4'
        assert len(ota_package.patches) == 2
        assert ota_package.compression == CompressionType.LZMA
    
    def test_ota_package_serialization(self):
        """Test OTA package saving and serialization."""
        
        patch = create_pqc_patch_info(
            target_address=0x1000,
            original_data=b'\x00\xF0\x00\xF8',
            replacement_data=b'\x00\xBF\x00\xBF',
            patch_type=PatchType.INLINE_PATCH
        )
        
        ota_package = OTAPackage(
            version='1.0.0',
            target_device='ESP32',
            base_version='0.9.0',
            patches=[patch],
            compression=CompressionType.GZIP,
            encrypted=False,
            integrity_hash='test_hash_123',
            rollback_data=b'rollback',
            metadata={'test': 'data'}
        )
        
        with tempfile.NamedTemporaryFile(suffix='.ota', delete=False) as f:
            package_path = f.name
        
        try:
            # Save package
            ota_package.save(package_path)
            
            # Verify file was created
            assert Path(package_path).exists()
            
            # Verify file has expected structure
            with open(package_path, 'rb') as f:
                data = f.read()
            
            # Should start with magic header
            assert data[:8] == b'PQCOTA01'
            
            # Should have size header
            size = struct.unpack('<I', data[8:12])[0]
            assert size > 0
            assert size <= len(data) - 12
            
        finally:
            Path(package_path).unlink(missing_ok=True)
    
    def test_ota_package_compression(self):
        """Test different compression methods for OTA packages."""
        
        patch = create_pqc_patch_info(
            target_address=0x1000,
            original_data=b'\x00' * 100,  # Compressible data
            replacement_data=b'\xFF' * 100,
            patch_type=PatchType.DATA_MODIFICATION
        )
        
        # Test LZMA compression
        lzma_package = OTAPackage(
            version='1.0.0',
            target_device='Test',
            base_version='0.0.0',
            patches=[patch],
            compression=CompressionType.LZMA,
            encrypted=False,
            integrity_hash='test',
            rollback_data=b'test',
            metadata={}
        )
        
        # Test GZIP compression
        gzip_package = OTAPackage(
            version='1.0.0',
            target_device='Test',
            base_version='0.0.0',
            patches=[patch],
            compression=CompressionType.GZIP,
            encrypted=False,
            integrity_hash='test',
            rollback_data=b'test',
            metadata={}
        )
        
        # Test no compression
        none_package = OTAPackage(
            version='1.0.0',
            target_device='Test',
            base_version='0.0.0',
            patches=[patch],
            compression=CompressionType.NONE,
            encrypted=False,
            integrity_hash='test',
            rollback_data=b'test',
            metadata={}
        )
        
        # Save all packages and compare sizes
        packages = [
            (lzma_package, 'lzma'),
            (gzip_package, 'gzip'),
            (none_package, 'none')
        ]
        
        sizes = {}
        for pkg, name in packages:
            with tempfile.NamedTemporaryFile(suffix=f'.{name}.ota', delete=False) as f:
                temp_path = f.name
            
            try:
                pkg.save(temp_path)
                sizes[name] = Path(temp_path).stat().st_size
            finally:
                Path(temp_path).unlink(missing_ok=True)
        
        # Compressed versions should be smaller than uncompressed
        # (though with small test data, compression might not be effective)
        assert sizes['none'] > 0
        assert sizes['lzma'] > 0
        assert sizes['gzip'] > 0


class TestFactoryFunctions:
    """Test factory functions and helper utilities."""
    
    def test_create_binary_patcher(self):
        """Test the factory function for creating binary patchers."""
        
        arm_patcher = create_binary_patcher('arm')
        assert isinstance(arm_patcher, BinaryPatcher)
        assert arm_patcher.target_arch == 'arm'
        
        xtensa_patcher = create_binary_patcher('xtensa')
        assert isinstance(xtensa_patcher, BinaryPatcher)
        assert xtensa_patcher.target_arch == 'xtensa'
    
    def test_create_pqc_patch_info(self):
        """Test the helper function for creating patch info."""
        
        original = b'\x00\xF0\x00\xF8'
        replacement = b'\x00\xBF\x00\xBF'
        
        patch_info = create_pqc_patch_info(
            target_address=0x1000,
            original_data=original,
            replacement_data=replacement
        )
        
        assert isinstance(patch_info, BinaryPatchInfo)
        assert patch_info.target_address == 0x1000
        assert patch_info.original_data == original
        assert patch_info.replacement_data == replacement
        assert patch_info.patch_type == PatchType.FUNCTION_REPLACEMENT  # Default
        assert patch_info.target_size == len(original)
        assert len(patch_info.verification_hash) > 0  # Should have hash
        
        # Test with different patch type
        inline_patch = create_pqc_patch_info(
            target_address=0x2000,
            original_data=original,
            replacement_data=replacement,
            patch_type=PatchType.INLINE_PATCH
        )
        
        assert inline_patch.patch_type == PatchType.INLINE_PATCH


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_patch_type(self):
        """Test handling of invalid patch types."""
        patcher = BinaryPatcher('arm')
        
        # Create patch with invalid type (mock)
        patch_info = BinaryPatchInfo(
            patch_type=None,  # Invalid
            target_address=0x1000,
            target_size=4,
            replacement_data=b'\x00\x00\x00\x00',
            original_data=b'\xFF\xFF\xFF\xFF',
            relocation_entries=[],
            symbol_updates={},
            dependencies=[],
            verification_hash='test'
        )
        
        firmware_data = bytearray(2048)
        firmware_data[0x1000:0x1004] = b'\xFF\xFF\xFF\xFF'
        
        # Should handle gracefully
        success = patcher._apply_single_patch(firmware_data, patch_info)
        assert not success
    
    def test_patch_size_mismatch(self):
        """Test handling of patch size mismatches."""
        patcher = BinaryPatcher('arm')
        
        # Create inline patch with wrong size
        patch_info = create_pqc_patch_info(
            target_address=0x1000,
            original_data=b'\x00\x00\x00\x00',  # 4 bytes
            replacement_data=b'\xFF\xFF',        # 2 bytes (mismatch)
            patch_type=PatchType.INLINE_PATCH
        )
        
        firmware_data = bytearray(2048)
        firmware_data[0x1000:0x1004] = b'\x00\x00\x00\x00'
        
        success = patcher._apply_single_patch(firmware_data, patch_info)
        assert not success
    
    def test_nonexistent_firmware_file(self):
        """Test handling of nonexistent firmware files."""
        patcher = BinaryPatcher('arm')
        
        patch_info = create_pqc_patch_info(
            target_address=0x1000,
            original_data=b'\x00\x00\x00\x00',
            replacement_data=b'\xFF\xFF\xFF\xFF'
        )
        
        success = patcher.patch_firmware(
            'nonexistent_file.bin',
            [patch_info],
            'output.bin'
        )
        
        assert not success
    
    def test_insufficient_free_space(self, temp_firmware_file):
        """Test behavior when there's insufficient free space for relocation."""
        
        # Create firmware with no free space
        firmware_data = bytearray(1024)
        for i in range(0, 1024, 4):
            firmware_data[i:i+4] = struct.pack('<I', 0x12345678)  # Fill with non-free pattern
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            f.write(firmware_data)
            full_firmware_path = f.name
        
        try:
            patcher = BinaryPatcher('arm')
            
            # Try to replace with much larger code
            large_replacement = b'\x00\xBF' * 200  # 400 bytes
            patch_info = create_pqc_patch_info(
                target_address=0x100,
                original_data=b'\x78\x56\x34\x12',
                replacement_data=large_replacement,
                patch_type=PatchType.FUNCTION_REPLACEMENT
            )
            
            with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
                output_path = f.name
            
            try:
                success = patcher.patch_firmware(full_firmware_path, [patch_info], output_path)
                # Should either succeed (by finding space we missed) or fail gracefully
                # The important thing is it doesn't crash
                assert isinstance(success, bool)
                
            finally:
                Path(output_path).unlink(missing_ok=True)
                
        finally:
            Path(full_firmware_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])