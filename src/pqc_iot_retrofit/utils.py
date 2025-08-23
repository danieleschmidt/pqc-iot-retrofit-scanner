"""Essential utility functions for PQC IoT Retrofit Scanner.

This module provides core utilities needed across the scanner:
- File format detection and validation
- Architecture detection helpers
- Memory management utilities
- Cryptographic pattern helpers
"""

import os
import struct
import hashlib
import logging
import mimetypes
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Configure logger
logger = logging.getLogger(__name__)


class FileFormat(Enum):
    """Supported firmware file formats."""
    ELF = "elf"
    HEX = "hex" 
    BIN = "bin"
    UF2 = "uf2"
    UNKNOWN = "unknown"


@dataclass
class FirmwareInfo:
    """Information about a firmware file."""
    path: str
    format: FileFormat
    size: int
    architecture: Optional[str]
    entry_point: Optional[int]
    sections: List[Dict[str, Any]]
    checksum: str


class ArchitectureDetector:
    """Automatic architecture detection from firmware files."""
    
    # ELF machine types to architecture mapping
    ELF_MACHINES = {
        0x28: "cortex-m",     # ARM
        0x40: "cortex-m",     # ARM (alternate)
        0xF3: "riscv32",      # RISC-V 32-bit
        0xF4: "riscv64",      # RISC-V 64-bit
        0x5E: "esp32",        # Xtensa (ESP32)
        0x83: "avr",          # AVR
    }
    
    # Magic byte signatures
    MAGIC_SIGNATURES = {
        b'\x7fELF': FileFormat.ELF,
        b':': FileFormat.HEX,        # Intel HEX starts with :
        b'UF2\n': FileFormat.UF2,    # UF2 magic
    }
    
    @classmethod
    def detect_file_format(cls, firmware_path: str) -> FileFormat:
        """Detect firmware file format from contents."""
        try:
            with open(firmware_path, 'rb') as f:
                header = f.read(16)
                
            # Check magic signatures
            for magic, format_type in cls.MAGIC_SIGNATURES.items():
                if header.startswith(magic):
                    return format_type
                    
            # Check file extension as fallback
            ext = Path(firmware_path).suffix.lower()
            if ext in ['.hex', '.ihex']:
                return FileFormat.HEX
            elif ext in ['.bin', '.fw']:
                return FileFormat.BIN
            elif ext == '.uf2':
                return FileFormat.UF2
                
            return FileFormat.UNKNOWN
            
        except Exception as e:
            logger.warning(f"Could not detect file format for {firmware_path}: {e}")
            return FileFormat.UNKNOWN
    
    @classmethod
    def detect_architecture(cls, firmware_path: str) -> Optional[str]:
        """Detect target architecture from firmware file."""
        try:
            file_format = cls.detect_file_format(firmware_path)
            
            if file_format == FileFormat.ELF:
                return cls._detect_elf_architecture(firmware_path)
            elif file_format == FileFormat.UF2:
                return cls._detect_uf2_architecture(firmware_path)
            
            # For raw binary/hex files, architecture cannot be reliably detected
            logger.info(f"Cannot auto-detect architecture for {file_format.value} format")
            return None
            
        except Exception as e:
            logger.error(f"Architecture detection failed for {firmware_path}: {e}")
            return None
    
    @classmethod
    def _detect_elf_architecture(cls, firmware_path: str) -> Optional[str]:
        """Extract architecture from ELF header."""
        try:
            with open(firmware_path, 'rb') as f:
                # Read ELF header
                header = f.read(52)  # ELF32 header size
                
                if len(header) < 20:
                    return None
                
                # Parse ELF header (simplified)
                magic, bit_class, endianness = struct.unpack('4sBB', header[:6])
                if magic != b'\x7fELF':
                    return None
                
                # Extract machine type (2 bytes at offset 18)
                machine_type = struct.unpack('<H' if endianness == 1 else '>H', 
                                           header[18:20])[0]
                
                return cls.ELF_MACHINES.get(machine_type)
                
        except Exception as e:
            logger.error(f"ELF architecture detection failed: {e}")
            return None
    
    @classmethod
    def _detect_uf2_architecture(cls, firmware_path: str) -> Optional[str]:
        """Extract architecture from UF2 family ID."""
        try:
            with open(firmware_path, 'rb') as f:
                # UF2 block structure
                header = f.read(32)
                
                if len(header) < 32:
                    return None
                    
                # UF2 family ID is at offset 28
                family_id = struct.unpack('<I', header[28:32])[0]
                
                # Known UF2 family IDs
                uf2_families = {
                    0x68ed2b88: "cortex-m",    # SAMD21
                    0x1851780a: "cortex-m",    # SAMD51  
                    0x621e937a: "cortex-m",    # nRF52840
                    0x57755a57: "cortex-m",    # STM32F4
                    0xe48bff56: "riscv32",     # ESP32-S2
                    0xbfdd4eee: "riscv32",     # ESP32-S3
                }
                
                return uf2_families.get(family_id)
                
        except Exception as e:
            logger.error(f"UF2 architecture detection failed: {e}")
            return None


class FirmwareAnalyzer:
    """Advanced firmware analysis utilities."""
    
    @staticmethod
    def calculate_checksum(firmware_path: str, algorithm: str = "sha256") -> str:
        """Calculate firmware checksum."""
        hash_func = getattr(hashlib, algorithm, None)
        if not hash_func:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
        hasher = hash_func()
        
        try:
            with open(firmware_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    @staticmethod
    def extract_strings(firmware_data: bytes, min_length: int = 4) -> List[str]:
        """Extract printable strings from firmware."""
        strings = []
        current_string = []
        
        for byte in firmware_data:
            if 32 <= byte <= 126:  # Printable ASCII range
                current_string.append(chr(byte))
            else:
                if len(current_string) >= min_length:
                    strings.append(''.join(current_string))
                current_string = []
                
        # Don't forget the last string
        if len(current_string) >= min_length:
            strings.append(''.join(current_string))
            
        return strings
    
    @staticmethod
    def analyze_entropy(data: bytes, block_size: int = 256) -> List[float]:
        """Analyze entropy distribution across firmware."""
        entropy_values = []
        
        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]
            if len(block) < block_size // 2:  # Skip very small blocks
                continue
                
            # Calculate Shannon entropy
            entropy = calculate_entropy(block)
            entropy_values.append(entropy)
            
        return entropy_values


class MemoryLayoutAnalyzer:
    """Memory layout analysis for embedded devices."""
    
    # Common memory layouts for popular MCUs
    MEMORY_LAYOUTS = {
        "cortex-m0": {
            "flash": {"start": 0x08000000, "size": 64*1024},
            "ram": {"start": 0x20000000, "size": 8*1024},
            "bootloader": {"start": 0x08000000, "size": 16*1024},
        },
        "cortex-m3": {
            "flash": {"start": 0x08000000, "size": 256*1024},
            "ram": {"start": 0x20000000, "size": 48*1024},
            "bootloader": {"start": 0x08000000, "size": 32*1024},
        },
        "cortex-m4": {
            "flash": {"start": 0x08000000, "size": 512*1024},
            "ram": {"start": 0x20000000, "size": 128*1024},
            "bootloader": {"start": 0x08000000, "size": 32*1024},
        },
        "esp32": {
            "flash": {"start": 0x400000, "size": 4*1024*1024},
            "ram": {"start": 0x3FFB0000, "size": 520*1024},
            "bootloader": {"start": 0x1000, "size": 28*1024},
        },
        "riscv32": {
            "flash": {"start": 0x10000000, "size": 1*1024*1024},
            "ram": {"start": 0x80000000, "size": 256*1024},
            "bootloader": {"start": 0x10000000, "size": 64*1024},
        }
    }
    
    @classmethod
    def get_memory_layout(cls, architecture: str) -> Dict[str, Any]:
        """Get standard memory layout for architecture."""
        # Direct lookup first
        if architecture in cls.MEMORY_LAYOUTS:
            return cls.MEMORY_LAYOUTS[architecture]
        
        # Fallback: try base architecture (e.g. "cortex-m" from "cortex-m4")
        if '-' in architecture:
            base_arch = '-'.join(architecture.split('-')[:-1])  # Keep "cortex-m" from "cortex-m4"
            if base_arch in cls.MEMORY_LAYOUTS:
                return cls.MEMORY_LAYOUTS[base_arch]
        
        return {}
    
    @classmethod
    def estimate_available_memory(cls, architecture: str, 
                                firmware_size: int) -> Dict[str, int]:
        """Estimate available memory for PQC implementations."""
        layout = cls.get_memory_layout(architecture)
        
        if not layout:
            # Conservative defaults
            return {
                "flash_available": max(0, 128*1024 - firmware_size),
                "ram_available": 32*1024,
                "stack_available": 8*1024,
            }
        
        flash_total = layout.get("flash", {}).get("size", 512*1024)
        ram_total = layout.get("ram", {}).get("size", 128*1024)
        bootloader_size = layout.get("bootloader", {}).get("size", 32*1024)
        
        # Conservative estimation
        flash_used = firmware_size + bootloader_size
        flash_available = max(0, flash_total - flash_used)
        
        # Reserve 50% of RAM for application, 25% for PQC, 25% for stack/heap
        ram_available = ram_total // 4
        stack_available = ram_total // 8
        
        return {
            "flash_available": flash_available,
            "ram_available": ram_available, 
            "stack_available": stack_available,
        }


def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of data."""
    if not data:
        return 0.0
        
    # Count byte frequencies
    byte_counts = [0] * 256
    for byte in data:
        byte_counts[byte] += 1
    
    # Calculate entropy
    data_len = len(data)
    entropy = 0.0
    
    for count in byte_counts:
        if count > 0:
            probability = count / data_len
            entropy -= probability * math.log2(probability)
            
    return entropy


def validate_firmware_path(firmware_path: str) -> bool:
    """Validate firmware file path and accessibility."""
    try:
        path = Path(firmware_path)
        
        if not path.exists():
            logger.error(f"Firmware file not found: {firmware_path}")
            return False
            
        if not path.is_file():
            logger.error(f"Path is not a file: {firmware_path}")
            return False
            
        if not os.access(firmware_path, os.R_OK):
            logger.error(f"Firmware file not readable: {firmware_path}")
            return False
            
        if path.stat().st_size == 0:
            logger.warning(f"Firmware file is empty: {firmware_path}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Firmware validation failed: {e}")
        return False


def format_size(size_bytes: int) -> str:
    """Format byte size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_address(address: int) -> str:
    """Format memory address in hex with proper padding."""
    return f"0x{address:08X}"


def create_firmware_info(firmware_path: str) -> FirmwareInfo:
    """Create comprehensive firmware information object."""
    try:
        path = Path(firmware_path)
        file_format = ArchitectureDetector.detect_file_format(firmware_path)
        architecture = ArchitectureDetector.detect_architecture(firmware_path)
        checksum = FirmwareAnalyzer.calculate_checksum(firmware_path)
        
        return FirmwareInfo(
            path=str(path.absolute()),
            format=file_format,
            size=path.stat().st_size,
            architecture=architecture,
            entry_point=None,  # Could be extracted from ELF
            sections=[],       # Could be populated from binary analysis
            checksum=checksum
        )
        
    except Exception as e:
        logger.error(f"Failed to create firmware info: {e}")
        return FirmwareInfo(
            path=firmware_path,
            format=FileFormat.UNKNOWN,
            size=0,
            architecture=None,
            entry_point=None,
            sections=[],
            checksum=""
        )


# Export commonly used functions and classes
__all__ = [
    'FileFormat', 'FirmwareInfo', 'ArchitectureDetector', 
    'FirmwareAnalyzer', 'MemoryLayoutAnalyzer',
    'calculate_entropy', 'validate_firmware_path', 
    'format_size', 'format_address', 'create_firmware_info'
]