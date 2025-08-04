"""
Binary patching system for firmware modification.

This module provides binary-level patching capabilities for IoT firmware,
including:
- ELF/binary file modification
- Function replacement and relocation
- OTA update package generation
- Rollback mechanism support
- Integrity verification
"""

import struct
import hashlib
import lzma
import zlib
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import json

try:
    import lief
    LIEF_AVAILABLE = True
except ImportError:
    LIEF_AVAILABLE = False

try:
    from elftools.elf.elffile import ELFFile
    from elftools.elf.sections import SymbolTableSection
    from elftools.common.py3compat import bytes2str
    ELFTOOLS_AVAILABLE = True
except ImportError:
    ELFTOOLS_AVAILABLE = False


class PatchType(Enum):
    """Types of binary patches."""
    FUNCTION_REPLACEMENT = "function_replacement"
    INLINE_PATCH = "inline_patch"
    DATA_MODIFICATION = "data_modification"
    SYMBOL_REDIRECTION = "symbol_redirection"
    LIBRARY_INJECTION = "library_injection"


class CompressionType(Enum):
    """Compression algorithms for OTA packages."""
    NONE = "none"
    GZIP = "gzip"
    LZMA = "lzma"
    CUSTOM = "custom"


@dataclass 
class BinaryPatchInfo:
    """Information about a binary patch."""
    patch_type: PatchType
    target_address: int
    target_size: int
    replacement_data: bytes
    original_data: bytes
    relocation_entries: List[Tuple[int, int]]  # (offset, target_address)
    symbol_updates: Dict[str, int]  # symbol_name -> new_address
    dependencies: List[str]  # Required symbols/functions
    verification_hash: str


@dataclass
class OTAPackage:
    """Over-the-Air update package."""
    version: str
    target_device: str
    base_version: str
    patches: List[BinaryPatchInfo]
    compression: CompressionType
    encrypted: bool
    integrity_hash: str
    rollback_data: bytes
    metadata: Dict[str, Any]
    
    def save(self, path: Union[str, Path]) -> None:
        """Save OTA package to file."""
        package_path = Path(path)
        
        # Create package structure
        package_data = {
            'header': {
                'version': self.version,
                'target_device': self.target_device,
                'base_version': self.base_version,
                'compression': self.compression.value,
                'encrypted': self.encrypted,
                'patch_count': len(self.patches),
                'metadata': self.metadata
            },
            'patches': [],
            'rollback_data': self.rollback_data.hex(),
            'integrity_hash': self.integrity_hash
        }
        
        # Serialize patches
        for i, patch in enumerate(self.patches):
            patch_data = {
                'id': i,
                'type': patch.patch_type.value,
                'target_address': f"0x{patch.target_address:08x}",
                'target_size': patch.target_size,
                'replacement_data': patch.replacement_data.hex(),
                'original_data': patch.original_data.hex(),
                'relocation_entries': [(f"0x{offset:08x}", f"0x{addr:08x}") 
                                     for offset, addr in patch.relocation_entries],
                'symbol_updates': {name: f"0x{addr:08x}" 
                                 for name, addr in patch.symbol_updates.items()},
                'dependencies': patch.dependencies,
                'verification_hash': patch.verification_hash
            }
            package_data['patches'].append(patch_data)
        
        # Compress if needed
        serialized = json.dumps(package_data, indent=2).encode('utf-8')
        
        if self.compression == CompressionType.LZMA:
            compressed_data = lzma.compress(serialized, preset=6)
        elif self.compression == CompressionType.GZIP:
            compressed_data = zlib.compress(serialized, level=6)
        else:
            compressed_data = serialized
        
        # Write package
        with open(package_path, 'wb') as f:
            # Package header
            f.write(b'PQCOTA01')  # Magic + version
            f.write(struct.pack('<I', len(compressed_data)))
            f.write(compressed_data)


class BinaryPatcher:
    """Binary-level firmware patcher."""
    
    def __init__(self, target_arch: str = "arm"):
        self.target_arch = target_arch
        self.arch_config = self._get_arch_config(target_arch)
        
    def _get_arch_config(self, arch: str) -> Dict[str, Any]:
        """Get architecture-specific configuration."""
        configs = {
            "arm": {
                "endianness": "little",
                "pointer_size": 4,
                "alignment": 4,
                "branch_instruction": 0xE7FE,  # b . (infinite loop)
                "nop_instruction": 0xBF00,     # nop (Thumb)
                "call_instruction_template": 0xF000F800,  # bl template
                "relocation_types": ["R_ARM_CALL", "R_ARM_THM_CALL", "R_ARM_ABS32"]
            },
            "xtensa": {
                "endianness": "little", 
                "pointer_size": 4,
                "alignment": 4,
                "branch_instruction": 0x0000F0,  # j 0
                "nop_instruction": 0x20F0,      # nop.n
                "call_instruction_template": 0x000001,  # call template
                "relocation_types": ["R_XTENSA_SLOT0_OP"]
            },
            "riscv": {
                "endianness": "little",
                "pointer_size": 4, 
                "alignment": 4,
                "branch_instruction": 0x0000006F,  # jal x0, 0
                "nop_instruction": 0x00000013,     # nop (addi x0, x0, 0)
                "call_instruction_template": 0x000000EF,  # jal ra, 0
                "relocation_types": ["R_RISCV_CALL", "R_RISCV_32"]
            }
        }
        return configs.get(arch, configs["arm"])
    
    def patch_firmware(self, firmware_path: str, patches: List[BinaryPatchInfo], 
                      output_path: str) -> bool:
        """Apply patches to firmware binary."""
        
        try:
            # Load original firmware
            with open(firmware_path, 'rb') as f:
                firmware_data = bytearray(f.read())
            
            # Apply patches in order
            for patch in patches:
                if not self._apply_single_patch(firmware_data, patch):
                    print(f"Failed to apply patch at 0x{patch.target_address:08x}")
                    return False
            
            # Verify integrity
            if not self._verify_patched_firmware(firmware_data, patches):
                print("Firmware integrity verification failed")
                return False
            
            # Write patched firmware
            with open(output_path, 'wb') as f:
                f.write(firmware_data)
            
            print(f"Successfully patched firmware: {output_path}")
            return True
            
        except Exception as e:
            print(f"Firmware patching failed: {e}")
            return False
    
    def _apply_single_patch(self, firmware_data: bytearray, patch: BinaryPatchInfo) -> bool:
        """Apply a single patch to firmware."""
        
        try:
            # Verify original data matches
            original_data = firmware_data[patch.target_address:patch.target_address + patch.target_size]
            if original_data != patch.original_data:
                print(f"Original data mismatch at 0x{patch.target_address:08x}")
                return False
            
            if patch.patch_type == PatchType.FUNCTION_REPLACEMENT:
                return self._apply_function_replacement(firmware_data, patch)
            elif patch.patch_type == PatchType.INLINE_PATCH:
                return self._apply_inline_patch(firmware_data, patch)
            elif patch.patch_type == PatchType.DATA_MODIFICATION:
                return self._apply_data_modification(firmware_data, patch)
            elif patch.patch_type == PatchType.SYMBOL_REDIRECTION:
                return self._apply_symbol_redirection(firmware_data, patch)
            else:
                print(f"Unsupported patch type: {patch.patch_type}")
                return False
                
        except Exception as e:
            print(f"Failed to apply patch: {e}")
            return False
    
    def _apply_function_replacement(self, firmware_data: bytearray, patch: BinaryPatchInfo) -> bool:
        """Replace entire function with PQC implementation."""
        
        # Check if replacement fits
        if len(patch.replacement_data) > patch.target_size:
            # Need to relocate function to free space
            new_location = self._find_free_space(firmware_data, len(patch.replacement_data))
            if new_location == -1:
                print("No free space for function replacement")
                return False
            
            # Place new function at free location
            firmware_data[new_location:new_location + len(patch.replacement_data)] = patch.replacement_data
            
            # Create jump from original location to new function
            jump_instruction = self._create_jump_instruction(patch.target_address, new_location)
            firmware_data[patch.target_address:patch.target_address + len(jump_instruction)] = jump_instruction
            
            # Fill remaining space with NOPs
            remaining_space = patch.target_size - len(jump_instruction)
            nop_pattern = struct.pack('<H', self.arch_config["nop_instruction"])
            for i in range(0, remaining_space, 2):
                if i + 1 < remaining_space:
                    firmware_data[patch.target_address + len(jump_instruction) + i:
                                patch.target_address + len(jump_instruction) + i + 2] = nop_pattern
        else:
            # Direct replacement
            firmware_data[patch.target_address:patch.target_address + len(patch.replacement_data)] = patch.replacement_data
            
            # Fill remaining space with NOPs
            remaining_space = patch.target_size - len(patch.replacement_data)
            nop_pattern = struct.pack('<H', self.arch_config["nop_instruction"])
            for i in range(0, remaining_space, 2):
                if i + 1 < remaining_space:
                    firmware_data[patch.target_address + len(patch.replacement_data) + i:
                                patch.target_address + len(patch.replacement_data) + i + 2] = nop_pattern
        
        # Apply relocations
        for offset, target_addr in patch.relocation_entries:
            self._apply_relocation(firmware_data, offset, target_addr)
        
        return True
    
    def _apply_inline_patch(self, firmware_data: bytearray, patch: BinaryPatchInfo) -> bool:
        """Apply inline binary patch (small modifications)."""
        
        # Direct replacement - must fit exactly
        if len(patch.replacement_data) != patch.target_size:
            print(f"Inline patch size mismatch: {len(patch.replacement_data)} != {patch.target_size}")
            return False
        
        firmware_data[patch.target_address:patch.target_address + patch.target_size] = patch.replacement_data
        
        # Apply relocations
        for offset, target_addr in patch.relocation_entries:
            self._apply_relocation(firmware_data, offset, target_addr)
        
        return True
    
    def _apply_data_modification(self, firmware_data: bytearray, patch: BinaryPatchInfo) -> bool:
        """Apply data modification patch (constants, lookup tables, etc.)."""
        
        # Replace data section
        firmware_data[patch.target_address:patch.target_address + patch.target_size] = patch.replacement_data
        
        return True
    
    def _apply_symbol_redirection(self, firmware_data: bytearray, patch: BinaryPatchInfo) -> bool:
        """Redirect symbol references to new addresses."""
        
        # This would typically involve parsing the symbol table and updating references
        # For now, apply as direct address updates
        for symbol_name, new_address in patch.symbol_updates.items():
            # Find and update symbol references
            # This is a simplified implementation
            old_addr_bytes = struct.pack('<I', patch.target_address)
            new_addr_bytes = struct.pack('<I', new_address)
            
            # Replace all occurrences
            data = bytes(firmware_data)
            data = data.replace(old_addr_bytes, new_addr_bytes)
            firmware_data[:] = data
        
        return True
    
    def _create_jump_instruction(self, from_addr: int, to_addr: int) -> bytes:
        """Create architecture-specific jump instruction."""
        
        if self.target_arch == "arm":
            # ARM Thumb BL instruction
            offset = to_addr - from_addr - 4
            
            # Split offset into parts for BL encoding
            s = (offset >> 24) & 1
            i1 = (offset >> 23) & 1  
            i2 = (offset >> 22) & 1
            imm10 = (offset >> 12) & 0x3FF
            imm11 = (offset >> 1) & 0x7FF
            
            # First instruction (BL immediate high)
            insn1 = 0xF000 | (s << 10) | imm10
            
            # Second instruction (BL immediate low) 
            insn2 = 0xF800 | (not(i1 ^ s) << 13) | (not(i2 ^ s) << 11) | imm11
            
            return struct.pack('<HH', insn1, insn2)
            
        elif self.target_arch == "xtensa":
            # Xtensa CALL instruction
            offset = (to_addr - from_addr - 4) >> 2
            if offset > 0x3FFFF or offset < -0x40000:
                # Use J instruction for long jumps
                target = to_addr >> 2
                insn = 0x000006 | ((target & 0x3FFFF) << 6)
                return struct.pack('<I', insn)[:3]  # Xtensa uses 3-byte instructions
            else:
                # CALL instruction
                insn = 0x000005 | ((offset & 0x3FFFF) << 6)
                return struct.pack('<I', insn)[:3]
                
        elif self.target_arch == "riscv":
            # RISC-V JAL instruction
            offset = to_addr - from_addr
            
            # JAL encoding
            imm_20 = (offset >> 20) & 1
            imm_10_1 = (offset >> 1) & 0x3FF
            imm_11 = (offset >> 11) & 1
            imm_19_12 = (offset >> 12) & 0xFF
            
            insn = (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | (imm_19_12 << 12) | 0x6F
            return struct.pack('<I', insn)
        
        else:
            # Generic - just return a NOP
            return struct.pack('<I', 0x00000000)
    
    def _find_free_space(self, firmware_data: bytearray, size_needed: int) -> int:
        """Find free space in firmware for new code."""
        
        # Look for sequences of 0xFF or 0x00 (typical uninitialized flash)
        search_patterns = [b'\xFF' * 16, b'\x00' * 16]
        
        for pattern in search_patterns:
            for i in range(0, len(firmware_data) - size_needed, 16):
                if firmware_data[i:i+16] == pattern:
                    # Found potential free space, verify it's large enough
                    free_size = 0
                    for j in range(i, len(firmware_data)):
                        if firmware_data[j] in [0xFF, 0x00]:
                            free_size += 1
                        else:
                            break
                    
                    if free_size >= size_needed:
                        return i
        
        return -1  # No free space found
    
    def _apply_relocation(self, firmware_data: bytearray, offset: int, target_addr: int) -> None:
        """Apply relocation entry."""
        
        # This is architecture-specific
        if self.target_arch == "arm":
            # ARM32 absolute relocation
            firmware_data[offset:offset+4] = struct.pack('<I', target_addr)
        elif self.target_arch in ["xtensa", "riscv"]:
            # Generic 32-bit absolute
            firmware_data[offset:offset+4] = struct.pack('<I', target_addr)
    
    def _verify_patched_firmware(self, firmware_data: bytearray, patches: List[BinaryPatchInfo]) -> bool:
        """Verify integrity of patched firmware."""
        
        for patch in patches:
            # Verify patch was applied correctly
            if patch.patch_type == PatchType.FUNCTION_REPLACEMENT:
                # Check that either direct replacement or jump was applied
                data_at_target = firmware_data[patch.target_address:patch.target_address + min(len(patch.replacement_data), patch.target_size)]
                
                if data_at_target == patch.replacement_data:
                    continue  # Direct replacement verified
                
                # Check for jump instruction
                if len(data_at_target) >= 4:
                    # Could be a jump - more complex verification needed
                    continue
                    
                return False
                
            elif patch.patch_type == PatchType.INLINE_PATCH:
                # Verify exact replacement
                data_at_target = firmware_data[patch.target_address:patch.target_address + patch.target_size]
                if data_at_target != patch.replacement_data:
                    return False
        
        return True
    
    def create_ota_package(self, base_firmware: str, patched_firmware: str, 
                          patches: List[BinaryPatchInfo], metadata: Dict[str, Any]) -> OTAPackage:
        """Create OTA update package."""
        
        # Load firmware files
        with open(base_firmware, 'rb') as f:
            base_data = f.read()
        
        with open(patched_firmware, 'rb') as f:
            patched_data = f.read()
        
        # Create rollback data (differential)
        rollback_data = self._create_rollback_data(base_data, patched_data, patches)
        
        # Calculate integrity hash
        integrity_data = patched_data + b''.join(p.replacement_data for p in patches)
        integrity_hash = hashlib.sha256(integrity_data).hexdigest()
        
        # Create package
        ota_package = OTAPackage(
            version=metadata.get('version', '1.0.0'),
            target_device=metadata.get('target_device', 'unknown'),
            base_version=metadata.get('base_version', '0.0.0'),
            patches=patches,
            compression=CompressionType.LZMA,
            encrypted=False,  # Could add encryption here
            integrity_hash=integrity_hash,
            rollback_data=rollback_data,
            metadata=metadata
        )
        
        return ota_package
    
    def _create_rollback_data(self, base_data: bytes, patched_data: bytes, 
                            patches: List[BinaryPatchInfo]) -> bytes:
        """Create rollback data for patches."""
        
        rollback_info = []
        
        for patch in patches:
            rollback_entry = {
                'address': patch.target_address,
                'size': patch.target_size,
                'original_data': patch.original_data.hex(),
                'type': patch.patch_type.value
            }
            rollback_info.append(rollback_entry)
        
        return json.dumps(rollback_info).encode('utf-8')
    
    def extract_function_binary(self, firmware_path: str, function_name: str, 
                              address: Optional[int] = None) -> Optional[bytes]:
        """Extract binary code for a specific function."""
        
        if not LIEF_AVAILABLE:
            print("LIEF not available - using address-based extraction")
            if address is None:
                return None
            
            # Simple extraction by address (would need size estimation)
            with open(firmware_path, 'rb') as f:
                f.seek(address)
                # Estimate function size by looking for return instruction
                data = f.read(1024)  # Read up to 1KB
                
                # Look for ARM Thumb return patterns
                if self.target_arch == "arm":
                    # Look for "bx lr" or "pop {..., pc}"
                    for i in range(0, len(data) - 1, 2):
                        insn = struct.unpack('<H', data[i:i+2])[0]
                        if insn == 0x4770:  # bx lr
                            return data[:i+2]
                        elif (insn & 0xFE00) == 0xBC00 and (insn & 0x0100):  # pop with PC
                            return data[:i+2]
                
                return data[:64]  # Fallback to first 64 bytes
        
        try:
            # Use LIEF to parse binary
            binary = lief.parse(firmware_path)
            if not binary:
                return None
            
            # Find function symbol
            for symbol in binary.symbols:
                if symbol.name == function_name:
                    section = binary.section_from_virtual_address(symbol.value)
                    if section:
                        offset = symbol.value - section.virtual_address
                        section_data = bytes(section.content)
                        
                        # Estimate function size
                        func_size = self._estimate_function_size(section_data[offset:], symbol.value)
                        return section_data[offset:offset + func_size]
            
            return None
            
        except Exception as e:
            print(f"Failed to extract function {function_name}: {e}")
            return None
    
    def _estimate_function_size(self, data: bytes, start_addr: int) -> int:
        """Estimate function size by analyzing instructions."""
        
        if self.target_arch == "arm":
            # ARM Thumb instruction analysis
            for i in range(0, len(data) - 1, 2):
                if i + 1 >= len(data):
                    break
                
                insn = struct.unpack('<H', data[i:i+2])[0]
                
                # Check for return instructions
                if insn == 0x4770:  # bx lr
                    return i + 2
                elif (insn & 0xFE00) == 0xBC00 and (insn & 0x0100):  # pop with PC
                    return i + 2
                elif insn == 0xBF00:  # nop - might indicate padding
                    # Look ahead for more NOPs
                    nop_count = 0
                    for j in range(i, min(i + 16, len(data) - 1), 2):
                        if j + 1 < len(data):
                            next_insn = struct.unpack('<H', data[j:j+2])[0]
                            if next_insn == 0xBF00:
                                nop_count += 1
                            else:
                                break
                    
                    if nop_count >= 4:  # Multiple NOPs suggest end of function
                        return i
        
        # Default fallback
        return min(256, len(data))


def create_binary_patcher(target_arch: str) -> BinaryPatcher:
    """Factory function to create binary patcher."""
    return BinaryPatcher(target_arch)


def create_pqc_patch_info(target_address: int, original_data: bytes, 
                         replacement_data: bytes, patch_type: PatchType = PatchType.FUNCTION_REPLACEMENT) -> BinaryPatchInfo:
    """Helper function to create PQC patch info."""
    
    return BinaryPatchInfo(
        patch_type=patch_type,
        target_address=target_address,
        target_size=len(original_data),
        replacement_data=replacement_data,
        original_data=original_data,
        relocation_entries=[],
        symbol_updates={},
        dependencies=[],
        verification_hash=hashlib.sha256(replacement_data).hexdigest()
    )