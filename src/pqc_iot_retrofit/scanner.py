"""Firmware scanning module for detecting quantum-vulnerable cryptography."""

import struct
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

from .error_handling import (
    handle_errors, ValidationError, FirmwareAnalysisError, 
    InputValidator, global_error_handler
)
from .monitoring import track_performance, metrics_collector

try:
    import capstone
    CAPSTONE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False

try:
    import lief
    LIEF_AVAILABLE = True
except ImportError:
    LIEF_AVAILABLE = False


class CryptoAlgorithm(Enum):
    """Quantum-vulnerable cryptographic algorithms."""
    RSA_1024 = "RSA-1024"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"
    ECDSA_P256 = "ECDSA-P256"
    ECDSA_P384 = "ECDSA-P384"
    ECDH_P256 = "ECDH-P256"
    ECDH_P384 = "ECDH-P384"
    DH_1024 = "DH-1024"
    DH_2048 = "DH-2048"


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CryptoVulnerability:
    """Detected cryptographic vulnerability."""
    algorithm: CryptoAlgorithm
    address: int
    function_name: str
    risk_level: RiskLevel
    key_size: Optional[int]
    description: str
    mitigation: str
    stack_usage: int
    available_stack: int


@dataclass
class ArchitectureInfo:
    """Target architecture information."""
    name: str
    capstone_arch: int
    capstone_mode: int
    endianness: str
    pointer_size: int
    instruction_alignment: int


class FirmwareScanner:
    """Scanner for detecting quantum-vulnerable cryptographic implementations."""
    
    # Enhanced cryptographic constants and patterns
    RSA_CONSTANTS = {
        # RSA OAEP padding constants
        b'\x00\x01\xff\xff': 'RSA-PKCS1',
        b'\x00\x02': 'RSA-OAEP',
        # Common RSA public exponents (little-endian)
        b'\x01\x00\x01\x00': 'RSA-65537',
        b'\x03\x00\x00\x00': 'RSA-3',
        b'\x11\x00\x01\x00': 'RSA-65537-BE',
        # RSA key size indicators
        b'\x00\x01\x00\x00': 'RSA-2048-len',  # 256 bytes
        b'\x80\x00\x00\x00': 'RSA-1024-len',  # 128 bytes
        b'\x00\x02\x00\x00': 'RSA-4096-len',  # 512 bytes
        # PKCS padding indicators
        b'\x30\x21\x30\x09': 'PKCS-SHA1',
        b'\x30\x31\x30\x0d': 'PKCS-SHA256',
    }
    
    ECC_CURVES = {
        # NIST P-256 curve parameters
        b'\xff\xff\xff\xff\x00\x00\x00\x01': 'P-256-prime-1',
        b'\x00\x00\x00\x00\xff\xff\xff\xff': 'P-256-prime-2', 
        b'\xff\xff\xff\xff\xff\xff\xff\xfc': 'P-256-a-param',
        b'\x5a\xc6\x35\xd8\xaa\x3a\x93\xe7': 'P-256-b-param',
        # NIST P-384 curve parameters
        b'\xff\xff\xff\xff\xff\xff\xff\xff': 'P-384-prime',
        b'\xff\xff\xff\xff\x00\x00\x00\x00': 'P-384-partial',
        # secp256k1 (Bitcoin curve)
        b'\xff\xff\xff\xff\xff\xff\xff\xfe': 'secp256k1-prime',
        b'\xff\xff\xfc\x2f': 'secp256k1-suffix',
        # Curve25519
        b'\x7f\xff\xff\xff\xff\xff\xff\xed': 'Curve25519-prime',
    }
    
    # Hash function constants (for signature detection)
    HASH_CONSTANTS = {
        # SHA-1 initial values
        b'\x67\x45\x23\x01': 'SHA1-H0',
        b'\xEF\xCD\xAB\x89': 'SHA1-H1', 
        b'\x98\xBA\xDC\xFE': 'SHA1-H2',
        b'\x10\x32\x54\x76': 'SHA1-H3',
        b'\xC3\xD2\xE1\xF0': 'SHA1-H4',
        # SHA-256 initial values
        b'\x6a\x09\xe6\x67': 'SHA256-H0',
        b'\xbb\x67\xae\x85': 'SHA256-H1',
        b'\x3c\x6e\xf3\x72': 'SHA256-H2',
        # MD5 constants (deprecated but still found)
        b'\x01\x23\x45\x67': 'MD5-H0',
        b'\x89\xAB\xCD\xEF': 'MD5-H1',
    }
    
    CRYPTO_FUNCTION_PATTERNS = {
        'rsa_sign': [
            b'RSA.*sign',
            b'rsa.*crt',
            b'modular.*exp',
        ],
        'ecdsa_sign': [
            b'ecdsa.*sign',
            b'ecc.*sign',
            b'point.*mul',
        ],
        'ecdh_exchange': [
            b'ecdh.*compute',
            b'point.*multiply',
            b'shared.*secret',
        ],
        'dh_exchange': [
            b'dh.*compute',
            b'diffie.*hellman',
            b'modular.*pow',
        ]
    }
    
    def _get_architectures(self):
        """Get architecture definitions (handles missing capstone gracefully)."""
        if CAPSTONE_AVAILABLE:
            return {
                'cortex-m0': ArchitectureInfo('cortex-m0', capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB, 'little', 4, 2),
                'cortex-m3': ArchitectureInfo('cortex-m3', capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB, 'little', 4, 2),
                'cortex-m4': ArchitectureInfo('cortex-m4', capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB, 'little', 4, 2),
                'cortex-m7': ArchitectureInfo('cortex-m7', capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB, 'little', 4, 2),
                'esp32': ArchitectureInfo('esp32', capstone.CS_ARCH_XTENSA, 0, 'little', 4, 1),
                'riscv32': ArchitectureInfo('riscv32', capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV32, 'little', 4, 4),
                'avr': ArchitectureInfo('avr', capstone.CS_ARCH_ARM, capstone.CS_MODE_THUMB, 'little', 2, 2),
            }
        else:
            # Fallback when capstone is not available  
            return {
                'cortex-m0': ArchitectureInfo('cortex-m0', 0, 0, 'little', 4, 2),
                'cortex-m3': ArchitectureInfo('cortex-m3', 0, 0, 'little', 4, 2),
                'cortex-m4': ArchitectureInfo('cortex-m4', 0, 0, 'little', 4, 2),
                'cortex-m7': ArchitectureInfo('cortex-m7', 0, 0, 'little', 4, 2),
                'esp32': ArchitectureInfo('esp32', 0, 0, 'little', 4, 1),
                'riscv32': ArchitectureInfo('riscv32', 0, 0, 'little', 4, 4),
                'avr': ArchitectureInfo('avr', 0, 0, 'little', 2, 2),
            }
    
    def __init__(self, architecture: str, memory_constraints: Dict[str, int] = None):
        """Initialize firmware scanner.
        
        Args:
            architecture: Target device architecture (cortex-m4, esp32, etc.)
            memory_constraints: Flash and RAM constraints
        """
        # Validate inputs
        InputValidator.validate_architecture(architecture)
        if memory_constraints:
            InputValidator.validate_memory_constraints(memory_constraints)
        
        self.architecture = architecture
        self.memory_constraints = memory_constraints or {'flash': 512*1024, 'ram': 128*1024}
        self.logger = logging.getLogger(__name__)
        
        architectures = self._get_architectures()
        if architecture not in architectures:
            raise ValidationError(f"Unsupported architecture: {architecture}", 
                                field="architecture", value=architecture)
        
        self.arch_info = architectures[architecture]
        self.vulnerabilities = []
        
        # Initialize disassembler if available
        self.disassembler = None
        if CAPSTONE_AVAILABLE:
            try:
                self.disassembler = capstone.Cs(self.arch_info.capstone_arch, self.arch_info.capstone_mode)
                self.disassembler.detail = True
                self.logger.info(f"Initialized disassembler for {architecture}")
            except Exception as e:
                self.logger.warning(f"Could not initialize disassembler: {e}")
        else:
            self.logger.warning("Capstone not available - disassembly features disabled")
    
    @handle_errors(operation_name="firmware_scan", retry_count=1)
    @track_performance("firmware_scan")
    def scan_firmware(self, firmware_path: str, base_address: int = 0) -> List[CryptoVulnerability]:
        """Scan firmware for quantum-vulnerable cryptography.
        
        Args:
            firmware_path: Path to firmware binary
            base_address: Base memory address
            
        Returns:
            List of detected vulnerabilities
        """
        # Validate inputs
        InputValidator.validate_firmware_path(firmware_path)
        base_address = InputValidator.validate_address(base_address)
        
        self.logger.info(f"Starting firmware scan: {firmware_path}")
        metrics_collector.record_metric("scan.started", 1, "scans")
        
        try:
            # Check if file exists and is readable
            firmware_path_obj = Path(firmware_path)
            if not firmware_path_obj.exists():
                raise FirmwareAnalysisError(f"Firmware file not found: {firmware_path}", 
                                          firmware_path=firmware_path)
            
            if not firmware_path_obj.is_file():
                raise FirmwareAnalysisError(f"Path is not a file: {firmware_path}",
                                          firmware_path=firmware_path)
            
            # Read firmware data
            firmware_data = firmware_path_obj.read_bytes()
            firmware_size = len(firmware_data)
            
            self.logger.info(f"Loaded firmware: {firmware_size} bytes")
            metrics_collector.record_metric("firmware.size_bytes", firmware_size, "bytes")
            
            if firmware_size == 0:
                raise FirmwareAnalysisError("Firmware file is empty", firmware_path=firmware_path)
            
            if firmware_size > 100 * 1024 * 1024:  # 100MB limit
                self.logger.warning(f"Large firmware file: {firmware_size / (1024*1024):.1f} MB")
            
            self.vulnerabilities = []
            
            # Detect file format and extract code sections
            code_sections = self._extract_code_sections(firmware_data, firmware_path)
            self.logger.info(f"Extracted {len(code_sections)} code sections")
            
            for i, (section_data, section_address) in enumerate(code_sections):
                section_size = len(section_data)
                self.logger.debug(f"Scanning section {i}: {section_size} bytes at 0x{section_address:08x}")
                
                # Pattern-based detection
                self._scan_crypto_constants(section_data, section_address + base_address)
                
                # String-based detection
                self._scan_crypto_strings(section_data, section_address + base_address)
                
                # Disassembly-based detection
                if self.disassembler:
                    self._scan_crypto_instructions(section_data, section_address + base_address)
            
            # Analyze and classify findings
            self._classify_vulnerabilities()
            
            vuln_count = len(self.vulnerabilities)
            self.logger.info(f"Scan completed: found {vuln_count} vulnerabilities")
            metrics_collector.record_metric("scan.vulnerabilities_found", vuln_count, "vulnerabilities")
            metrics_collector.record_metric("scan.completed", 1, "scans")
            
            return self.vulnerabilities
            
        except Exception as e:
            self.logger.error(f"Firmware scan failed: {e}")
            metrics_collector.record_metric("scan.failed", 1, "scans")
            
            if isinstance(e, (ValidationError, FirmwareAnalysisError)):
                raise
            else:
                raise FirmwareAnalysisError(f"Unexpected error during scan: {str(e)}", 
                                          firmware_path=firmware_path)
    
    def _extract_code_sections(self, firmware_data: bytes, firmware_path: str) -> List[Tuple[bytes, int]]:
        """Extract executable code sections from firmware."""
        sections = []
        
        # Try LIEF first for structured binaries
        if LIEF_AVAILABLE:
            try:
                binary = lief.parse(firmware_path)
                if binary:
                    for section in binary.sections:
                        if section.has(lief.ELF.SECTION_FLAGS.EXECINSTR):
                            sections.append((bytes(section.content), section.virtual_address))
                    if sections:
                        return sections
            except Exception:
                pass  # Fall back to raw analysis
        
        # Fall back to treating entire file as code
        sections.append((firmware_data, 0))
        return sections
    
    def _scan_crypto_constants(self, data: bytes, base_address: int):
        """Scan for known cryptographic constants."""
        # RSA constants
        for constant, name in self.RSA_CONSTANTS.items():
            for match in re.finditer(re.escape(constant), data):
                address = base_address + match.start()
                key_size = self._estimate_rsa_key_size(data, match.start())
                
                vuln = CryptoVulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048 if key_size >= 2048 else CryptoAlgorithm.RSA_1024,
                    address=address,
                    function_name=f"rsa_function_0x{address:08x}",
                    risk_level=RiskLevel.CRITICAL,
                    key_size=key_size,
                    description=f"RSA implementation detected with {name} at 0x{address:08x}",
                    mitigation="Replace with Dilithium2 or Dilithium3 digital signatures",
                    stack_usage=key_size // 8,  # Estimate
                    available_stack=self.memory_constraints.get('ram', 0) // 4
                )
                self.vulnerabilities.append(vuln)
        
        # ECC curve constants
        for constant, curve_name in self.ECC_CURVES.items():
            for match in re.finditer(re.escape(constant), data):
                address = base_address + match.start()
                algorithm = CryptoAlgorithm.ECDSA_P256 if 'P-256' in curve_name else CryptoAlgorithm.ECDSA_P384
                
                vuln = CryptoVulnerability(
                    algorithm=algorithm,
                    address=address,
                    function_name=f"ecc_function_0x{address:08x}",
                    risk_level=RiskLevel.HIGH,
                    key_size=256 if 'P-256' in curve_name else 384,
                    description=f"ECC {curve_name} curve implementation detected at 0x{address:08x}",
                    mitigation="Replace with Kyber512 or Kyber768 key encapsulation",
                    stack_usage=64,  # Typical ECC stack usage
                    available_stack=self.memory_constraints.get('ram', 0) // 4
                )
                self.vulnerabilities.append(vuln)
    
    def _scan_crypto_strings(self, data: bytes, base_address: int):
        """Scan for cryptographic algorithm names in strings."""
        crypto_strings = [
            (b'RSA', CryptoAlgorithm.RSA_2048, RiskLevel.CRITICAL),
            (b'ECDSA', CryptoAlgorithm.ECDSA_P256, RiskLevel.HIGH),
            (b'ECDH', CryptoAlgorithm.ECDH_P256, RiskLevel.HIGH),
            (b'DH', CryptoAlgorithm.DH_2048, RiskLevel.MEDIUM),
            (b'Diffie-Hellman', CryptoAlgorithm.DH_2048, RiskLevel.MEDIUM),
        ]
        
        for crypto_string, algorithm, risk in crypto_strings:
            for match in re.finditer(re.escape(crypto_string), data, re.IGNORECASE):
                address = base_address + match.start()
                
                vuln = CryptoVulnerability(
                    algorithm=algorithm,
                    address=address,
                    function_name=f"crypto_string_0x{address:08x}",
                    risk_level=risk,
                    key_size=None,
                    description=f"Reference to {crypto_string.decode()} algorithm at 0x{address:08x}",
                    mitigation=self._get_pqc_mitigation(algorithm),
                    stack_usage=32,  # Conservative estimate
                    available_stack=self.memory_constraints.get('ram', 0) // 4
                )
                self.vulnerabilities.append(vuln)
    
    def _scan_crypto_instructions(self, data: bytes, base_address: int):
        """Scan for cryptographic instruction patterns."""
        if not self.disassembler:
            return
        
        try:
            instructions = list(self.disassembler.disasm(data, base_address))
            
            # Look for patterns indicating cryptographic operations
            for i, insn in enumerate(instructions):
                # Large integer operations (potential RSA)
                if self._is_large_integer_operation(insn, instructions[i:i+10]):
                    vuln = CryptoVulnerability(
                        algorithm=CryptoAlgorithm.RSA_2048,
                        address=insn.address,
                        function_name=f"suspected_rsa_0x{insn.address:08x}",
                        risk_level=RiskLevel.HIGH,
                        key_size=None,
                        description=f"Suspected RSA modular exponentiation at 0x{insn.address:08x}",
                        mitigation="Verify and replace with Dilithium digital signatures",
                        stack_usage=256,  # Conservative estimate
                        available_stack=self.memory_constraints.get('ram', 0) // 4
                    )
                    self.vulnerabilities.append(vuln)
                
                # Point multiplication patterns (potential ECC)
                if self._is_point_multiplication(insn, instructions[i:i+15]):
                    vuln = CryptoVulnerability(
                        algorithm=CryptoAlgorithm.ECDSA_P256,
                        address=insn.address,
                        function_name=f"suspected_ecc_0x{insn.address:08x}",
                        risk_level=RiskLevel.HIGH,
                        key_size=256,
                        description=f"Suspected ECC point multiplication at 0x{insn.address:08x}",
                        mitigation="Replace with Kyber key encapsulation mechanism",
                        stack_usage=128,
                        available_stack=self.memory_constraints.get('ram', 0) // 4
                    )
                    self.vulnerabilities.append(vuln)
        
        except Exception as e:
            print(f"Warning: Disassembly failed: {e}")
    
    def _estimate_rsa_key_size(self, data: bytes, offset: int) -> int:
        """Estimate RSA key size from surrounding data."""
        # Look for key length indicators around the constant
        search_range = data[max(0, offset-100):offset+100]
        
        # Common RSA key sizes as 32-bit little-endian integers
        key_size_markers = {
            b'\x00\x01\x00\x00': 256,   # 256 bytes = 2048 bits
            b'\x80\x00\x00\x00': 128,   # 128 bytes = 1024 bits
            b'\x00\x02\x00\x00': 512,   # 512 bytes = 4096 bits
        }
        
        for marker, size_bits in key_size_markers.items():
            if marker in search_range:
                return size_bits
        
        return 2048  # Default assumption
    
    def _is_large_integer_operation(self, insn, context) -> bool:
        """Detect patterns indicating large integer arithmetic (RSA)."""
        if not insn or not context:
            return False
        
        # Enhanced multi-precision arithmetic detection
        # Look for Montgomery multiplication patterns
        montgomery_patterns = {
            'umull': 0,  # Unsigned multiply long
            'umlal': 0,  # Unsigned multiply-accumulate long
            'adcs': 0,   # Add with carry
            'sbcs': 0,   # Subtract with carry
        }
        
        # Barrett reduction patterns
        barrett_patterns = {
            'umull': 0,  # High precision multiplication
            'lsr': 0,    # Logical shift right (for division)
            'mul': 0,    # Standard multiplication
            'sub': 0,    # Subtraction
        }
        
        # Count pattern occurrences in context
        for i in context[:10] if context else []:
            if i and hasattr(i, 'mnemonic'):
                mnemonic = i.mnemonic.lower()
                
                # Montgomery patterns
                for pattern in montgomery_patterns:
                    if pattern in mnemonic:
                        montgomery_patterns[pattern] += 1
                
                # Barrett patterns  
                for pattern in barrett_patterns:
                    if pattern in mnemonic:
                        barrett_patterns[pattern] += 1
        
        # Check for Montgomery ladder (RSA/ECC)
        montgomery_score = sum(montgomery_patterns.values())
        barrett_score = sum(barrett_patterns.values())
        
        # Look for specific RSA patterns
        has_carry_chain = montgomery_patterns['adcs'] >= 2 or montgomery_patterns['sbcs'] >= 2
        has_long_mul = montgomery_patterns['umull'] >= 1 or montgomery_patterns['umlal'] >= 1
        has_reduction = barrett_score >= 3
        
        return (montgomery_score >= 4 and has_carry_chain) or (has_long_mul and has_reduction)
    
    def _is_point_multiplication(self, insn, context) -> bool:
        """Detect patterns indicating elliptic curve point multiplication."""
        if not insn or not context:
            return False
        
        # Enhanced ECC point operation detection
        # Look for specific ECC patterns
        ecc_patterns = {
            'field_mul': 0,      # Field multiplication
            'field_sqr': 0,      # Field squaring
            'field_add': 0,      # Field addition
            'field_sub': 0,      # Field subtraction
            'conditional': 0,    # Conditional operations (constant-time)
            'doubling': 0,       # Point doubling patterns
        }
        
        # Analyze instruction patterns
        for i in context[:15] if context else []:
            if i and hasattr(i, 'mnemonic'):
                mnemonic = i.mnemonic.lower()
                op_str = getattr(i, 'op_str', '').lower()
                
                # Field multiplication patterns
                if 'mul' in mnemonic and ('r' in op_str or 'mem' in op_str):
                    ecc_patterns['field_mul'] += 1
                
                # Squaring patterns (mul with same operand)
                if 'mul' in mnemonic and self._is_squaring_operation(i):
                    ecc_patterns['field_sqr'] += 1
                
                # Field addition/subtraction
                if mnemonic in ['add', 'adc', 'adds', 'adcs']:
                    ecc_patterns['field_add'] += 1
                elif mnemonic in ['sub', 'sbc', 'subs', 'sbcs']:
                    ecc_patterns['field_sub'] += 1
                
                # Conditional moves (constant-time implementation)
                if mnemonic in ['csel', 'csinv', 'csneg', 'movne', 'moveq']:
                    ecc_patterns['conditional'] += 1
                
                # Point doubling indicator (left shift by 1)
                if mnemonic in ['lsl', 'shl'] and '1' in op_str:
                    ecc_patterns['doubling'] += 1
        
        # ECC scoring
        field_ops = ecc_patterns['field_mul'] + ecc_patterns['field_sqr']
        arith_ops = ecc_patterns['field_add'] + ecc_patterns['field_sub']
        
        # Check for classic ECC patterns
        has_field_ops = field_ops >= 3
        has_arithmetic = arith_ops >= 2
        has_conditional = ecc_patterns['conditional'] >= 1  # Constant-time indicator
        has_doubling = ecc_patterns['doubling'] >= 1
        
        # Montgomery ladder pattern for scalar multiplication
        montgomery_ladder = (field_ops >= 4 and arith_ops >= 3 and has_conditional)
        
        # Binary scalar multiplication pattern
        binary_method = (field_ops >= 2 and has_doubling and arith_ops >= 1)
        
        return montgomery_ladder or binary_method or (has_field_ops and has_arithmetic)
    
    def _is_squaring_operation(self, insn) -> bool:
        """Check if multiplication instruction is actually a squaring operation."""
        if not hasattr(insn, 'op_str'):
            return False
        
        op_str = insn.op_str.lower()
        # Look for patterns like "mul r1, r1, r1" or "mul r1, r2, r2"
        parts = [p.strip().rstrip(',') for p in op_str.split()]
        if len(parts) >= 3:
            # Check if two operands are the same (squaring)
            return parts[1] == parts[2]
        
        return False
    
    def _get_pqc_mitigation(self, algorithm: CryptoAlgorithm) -> str:
        """Get post-quantum mitigation recommendation."""
        mitigations = {
            CryptoAlgorithm.RSA_1024: "Replace with Dilithium2 (NIST Level 1)",
            CryptoAlgorithm.RSA_2048: "Replace with Dilithium3 (NIST Level 3)",
            CryptoAlgorithm.RSA_4096: "Replace with Dilithium5 (NIST Level 5)",
            CryptoAlgorithm.ECDSA_P256: "Replace with Dilithium2 for signatures",
            CryptoAlgorithm.ECDSA_P384: "Replace with Dilithium3 for signatures",
            CryptoAlgorithm.ECDH_P256: "Replace with Kyber512 key encapsulation",
            CryptoAlgorithm.ECDH_P384: "Replace with Kyber768 key encapsulation",
            CryptoAlgorithm.DH_1024: "Replace with Kyber512 key encapsulation",
            CryptoAlgorithm.DH_2048: "Replace with Kyber768 key encapsulation",
        }
        return mitigations.get(algorithm, "Assess and replace with appropriate PQC algorithm")
    
    def _classify_vulnerabilities(self):
        """Classify and prioritize detected vulnerabilities."""
        for vuln in self.vulnerabilities:
            # Adjust risk based on key size and algorithm
            if vuln.algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.DH_1024]:
                vuln.risk_level = RiskLevel.CRITICAL
            elif vuln.key_size and vuln.key_size < 2048:
                vuln.risk_level = RiskLevel.CRITICAL
            
            # Consider memory constraints
            if vuln.stack_usage > vuln.available_stack * 0.8:
                vuln.description += " (Limited memory for PQC replacement)"
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate vulnerability assessment report."""
        risk_counts = {level: 0 for level in RiskLevel}
        algorithm_counts = {alg: 0 for alg in CryptoAlgorithm}
        
        for vuln in self.vulnerabilities:
            risk_counts[vuln.risk_level] += 1
            algorithm_counts[vuln.algorithm] += 1
        
        return {
            'scan_summary': {
                'architecture': self.architecture,
                'total_vulnerabilities': len(self.vulnerabilities),
                'risk_distribution': {level.value: count for level, count in risk_counts.items()},
                'algorithm_distribution': {alg.value: count for alg, count in algorithm_counts.items()},
                'memory_constraints': self.memory_constraints,
            },
            'vulnerabilities': [
                {
                    'algorithm': vuln.algorithm.value,
                    'address': f"0x{vuln.address:08x}",
                    'function_name': vuln.function_name,
                    'risk_level': vuln.risk_level.value,
                    'key_size': vuln.key_size,
                    'description': vuln.description,
                    'mitigation': vuln.mitigation,
                    'memory_impact': {
                        'stack_usage': vuln.stack_usage,
                        'available_stack': vuln.available_stack,
                    }
                }
                for vuln in self.vulnerabilities
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []
        
        if any(vuln.risk_level == RiskLevel.CRITICAL for vuln in self.vulnerabilities):
            recommendations.append(
                "URGENT: Critical quantum vulnerabilities detected. "
                "Immediate PQC migration recommended."
            )
        
        total_stack_needed = sum(vuln.stack_usage for vuln in self.vulnerabilities)
        available_ram = self.memory_constraints.get('ram', 0)
        
        if total_stack_needed > available_ram * 0.6:
            recommendations.append(
                f"Memory optimization required: PQC implementations need ~{total_stack_needed} bytes, "
                f"but only {available_ram} bytes RAM available."
            )
        
        unique_algorithms = set(vuln.algorithm for vuln in self.vulnerabilities)
        if len(unique_algorithms) > 3:
            recommendations.append(
                "Multiple cryptographic algorithms detected. "
                "Consider standardizing on a single PQC suite (Dilithium + Kyber)."
            )
        
        return recommendations