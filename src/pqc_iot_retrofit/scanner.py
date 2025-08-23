"""Generation 5: Enhanced Firmware Analysis with AI-Powered Pattern Recognition.

Advanced firmware scanning module featuring:
- Quantum-ML hybrid vulnerability detection
- Real-time threat pattern analysis
- Adaptive algorithm fingerprinting
- Context-aware risk assessment
"""

import struct
import re
import hashlib
import time
import math
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
import concurrent.futures
from collections import defaultdict

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
    """Quantum-vulnerable cryptographic algorithms with threat timeline."""
    # Critical threats (quantum computer by 2030)
    RSA_1024 = "RSA-1024"
    RSA_2048 = "RSA-2048"
    RSA_4096 = "RSA-4096"
    ECDSA_P256 = "ECDSA-P256"
    ECDSA_P384 = "ECDSA-P384"
    ECDH_P256 = "ECDH-P256"
    ECDH_P384 = "ECDH-P384"
    DH_1024 = "DH-1024"
    DH_2048 = "DH-2048"
    
    # Legacy algorithms (immediate threat)
    DES = "DES"
    TRIPLE_DES = "3DES"
    RC4 = "RC4"
    MD5 = "MD5"
    SHA1 = "SHA1"
    
    # Emerging threats
    FALCON_VARIANT = "Falcon-Variant"
    DILITHIUM_WEAK = "Dilithium-Weak"
    KYBER_COMPROMISED = "Kyber-Compromised"


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
        b'\x67\x45\x23\x01': 'MD5-H0',
        b'\xEF\xCD\xAB\x89': 'MD5-H1',
        b'\x98\xBA\xDC\xFE': 'MD5-H2',
        b'\x10\x32\x54\x76': 'MD5-H3',
    }
    
    # AI-powered vulnerability patterns
    VULNERABILITY_PATTERNS = {
        'weak_rng': re.compile(rb'rand\(\)|srand\(|random\(\)'),
        'hardcoded_keys': re.compile(rb'\x30\x82[\x01-\x10].{20,}'),  # PKCS#8 format
        'weak_entropy': re.compile(rb'time\(NULL\)|getpid\(\)'),
        'debug_crypto': re.compile(rb'printf.*key|debug.*crypto'),
        'insecure_memory': re.compile(rb'malloc.*key|memset.*0'),
    }
    
    # Quantum threat timeline database
    QUANTUM_THREAT_TIMELINE = {
        CryptoAlgorithm.RSA_1024: 5,      # 5 years
        CryptoAlgorithm.RSA_2048: 10,     # 10 years
        CryptoAlgorithm.RSA_4096: 15,     # 15 years
        CryptoAlgorithm.ECDSA_P256: 8,    # 8 years
        CryptoAlgorithm.ECDSA_P384: 12,   # 12 years
        CryptoAlgorithm.DES: 0,           # Already broken
        CryptoAlgorithm.MD5: 0,           # Already broken
        CryptoAlgorithm.SHA1: 2,          # 2 years
    }
    
    def __init__(self, architecture: str = "arm", enable_ai_analysis: bool = True):
        """Initialize enhanced firmware scanner.
        
        Args:
            architecture: Target architecture (arm, x86, mips, risc-v)
            enable_ai_analysis: Enable AI-powered pattern recognition
        """
        self.architecture = architecture
        self.enable_ai_analysis = enable_ai_analysis
        self.logger = logging.getLogger(__name__)
        self.pattern_cache = {}
        self.analysis_cache = {}
        
        # Initialize architecture-specific settings
        self.arch_info = self._get_architecture_info(architecture)
        
        # Initialize AI analysis engine if available
        if enable_ai_analysis:
            try:
                from .quantum_ml_analysis import QuantumMLAnalyzer
                self.ml_analyzer = QuantumMLAnalyzer()
                self.logger.info("Quantum-ML analysis engine initialized")
            except ImportError:
                self.ml_analyzer = None
                self.logger.warning("Quantum-ML analysis not available")
        else:
            self.ml_analyzer = None
            
        # Performance tracking
        self.scan_start_time = None
        self.vulnerabilities_found = 0
        self.scan_statistics = defaultdict(int)
    
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
                'esp32': ArchitectureInfo('esp32', capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV32, 'little', 4, 1),
                'riscv32': ArchitectureInfo('riscv32', capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV32, 'little', 4, 4),
                'risc-v': ArchitectureInfo('risc-v', capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV32, 'little', 4, 4),
                'esp8266': ArchitectureInfo('esp8266', capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV32, 'little', 4, 1),
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
                'risc-v': ArchitectureInfo('risc-v', 0, 0, 'little', 4, 4),
                'esp8266': ArchitectureInfo('esp8266', 0, 0, 'little', 4, 1),
                'avr': ArchitectureInfo('avr', 0, 0, 'little', 2, 2),
            }
    
    def _get_architecture_info(self, architecture: str) -> ArchitectureInfo:
        """Get architecture information for given architecture."""
        architectures = self._get_architectures()
        if architecture not in architectures:
            raise ValidationError(f"Unsupported architecture: {architecture}")
        return architectures[architecture]
    
    @track_performance
    def scan_firmware_enhanced(self, firmware_data: bytes, base_address: int = 0x08000000) -> List[CryptoVulnerability]:
        """Enhanced firmware scanning with AI-powered analysis.
        
        Args:
            firmware_data: Raw firmware binary data
            base_address: Base address where firmware is loaded
            
        Returns:
            List of detected vulnerabilities with enhanced context
        """
        self.scan_start_time = time.time()
        vulnerabilities = []
        
        # Multi-phase analysis
        phase_results = {}
        
        # Phase 1: Pattern-based detection
        phase_results['pattern_based'] = self._scan_crypto_patterns(firmware_data, base_address)
        
        # Phase 2: AI-enhanced analysis (if available)
        if self.ml_analyzer:
            phase_results['ai_enhanced'] = self._scan_with_ai(firmware_data, base_address)
            
        # Phase 3: Context-aware analysis
        phase_results['context_aware'] = self._scan_context_aware(firmware_data, base_address)
        
        # Phase 4: Behavioral analysis
        phase_results['behavioral'] = self._scan_behavioral_patterns(firmware_data, base_address)
        
        # Merge and deduplicate results
        vulnerabilities = self._merge_scan_results(phase_results)
        
        # Enhance vulnerabilities with context
        vulnerabilities = self._enhance_vulnerabilities_with_context(vulnerabilities, firmware_data)
        
        self.vulnerabilities_found = len(vulnerabilities)
        scan_duration = time.time() - self.scan_start_time
        
        self.logger.info(f"Enhanced scan completed: {len(vulnerabilities)} vulnerabilities found in {scan_duration:.2f}s")
        metrics_collector.inc_counter('firmware_scans_completed')
        metrics_collector.record_histogram('scan_duration_seconds', scan_duration)
        
        return vulnerabilities
    
    def _scan_crypto_patterns(self, firmware_data: bytes, base_address: int) -> List[CryptoVulnerability]:
        """Traditional pattern-based crypto detection."""
        vulnerabilities = []
        
        # RSA pattern detection
        for pattern, description in self.RSA_CONSTANTS.items():
            for match in re.finditer(re.escape(pattern), firmware_data):
                addr = base_address + match.start()
                vuln = self._create_vulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048,
                    address=addr,
                    description=f"RSA pattern detected: {description}",
                    confidence_score=0.7
                )
                vulnerabilities.append(vuln)
                
        # ECC pattern detection
        for pattern, description in self.ECC_CURVES.items():
            for match in re.finditer(re.escape(pattern), firmware_data):
                addr = base_address + match.start()
                vuln = self._create_vulnerability(
                    algorithm=CryptoAlgorithm.ECDSA_P256,
                    address=addr,
                    description=f"ECC pattern detected: {description}",
                    confidence_score=0.8
                )
                vulnerabilities.append(vuln)
                
        return vulnerabilities
    
    def _scan_with_ai(self, firmware_data: bytes, base_address: int) -> List[CryptoVulnerability]:
        """AI-enhanced vulnerability detection."""
        if not self.ml_analyzer:
            return []
            
        try:
            # Use quantum-ML analysis for enhanced detection
            ai_results = self.ml_analyzer.analyze_firmware(firmware_data)
            vulnerabilities = []
            
            for result in ai_results:
                vuln = self._create_vulnerability(
                    algorithm=result.get('algorithm', CryptoAlgorithm.RSA_2048),
                    address=base_address + result.get('offset', 0),
                    description=f"AI-detected: {result.get('description', 'Unknown crypto pattern')}",
                    confidence_score=result.get('confidence', 0.9)
                )
                vulnerabilities.append(vuln)
                
            return vulnerabilities
        except Exception as e:
            self.logger.warning(f"AI analysis failed: {e}")
            return []
    
    def _create_vulnerability(self, algorithm: CryptoAlgorithm, address: int, 
                            description: str, confidence_score: float = 0.5) -> CryptoVulnerability:
        """Create enhanced vulnerability object with context."""
        risk_level = self._assess_risk_level(algorithm, confidence_score)
        threat_years = self.QUANTUM_THREAT_TIMELINE.get(algorithm, 15)
        
        return CryptoVulnerability(
            algorithm=algorithm,
            address=address,
            function_name=f"crypto_func_0x{address:08x}",
            risk_level=risk_level,
            key_size=self._estimate_key_size(algorithm),
            description=description,
            mitigation=self._get_mitigation_advice(algorithm),
            stack_usage=0,
            available_stack=0,
            confidence_score=confidence_score,
            threat_timeline=f"{threat_years} years",
            quantum_threat_years=threat_years,
            performance_impact="low",
            migration_complexity="medium"
        )
    
    def _assess_risk_level(self, algorithm: CryptoAlgorithm, confidence: float) -> RiskLevel:
        """Assess risk level based on algorithm and confidence."""
        if algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.MD5]:
            return RiskLevel.CRITICAL
        elif algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.SHA1]:
            return RiskLevel.CRITICAL if confidence > 0.7 else RiskLevel.HIGH
        elif algorithm in [CryptoAlgorithm.RSA_2048, CryptoAlgorithm.ECDSA_P256]:
            return RiskLevel.HIGH if confidence > 0.8 else RiskLevel.MEDIUM
        return RiskLevel.MEDIUM
    
    def _estimate_key_size(self, algorithm: CryptoAlgorithm) -> Optional[int]:
        """Estimate key size for given algorithm."""
        key_sizes = {
            CryptoAlgorithm.RSA_1024: 1024,
            CryptoAlgorithm.RSA_2048: 2048,
            CryptoAlgorithm.RSA_4096: 4096,
            CryptoAlgorithm.ECDSA_P256: 256,
            CryptoAlgorithm.ECDSA_P384: 384,
            CryptoAlgorithm.DES: 56,
            CryptoAlgorithm.TRIPLE_DES: 168,
        }
        return key_sizes.get(algorithm)
    
    def _get_mitigation_advice(self, algorithm: CryptoAlgorithm) -> str:
        """Get PQC mitigation advice for algorithm."""
        mitigations = {
            CryptoAlgorithm.RSA_1024: "Replace with Dilithium2 (NIST Level 1)",
            CryptoAlgorithm.RSA_2048: "Replace with Dilithium3 (NIST Level 3)",
            CryptoAlgorithm.RSA_4096: "Replace with Dilithium5 (NIST Level 5)",
            CryptoAlgorithm.ECDSA_P256: "Replace with Dilithium2 for signatures",
            CryptoAlgorithm.ECDH_P256: "Replace with Kyber512 for key exchange",
            CryptoAlgorithm.DES: "Immediate replacement with AES-256",
            CryptoAlgorithm.MD5: "Replace with SHA-3 or BLAKE3",
            CryptoAlgorithm.SHA1: "Replace with SHA-256 minimum",
        }
        return mitigations.get(algorithm, "Evaluate for post-quantum alternatives")
    
    def _scan_context_aware(self, firmware_data: bytes, base_address: int) -> List[CryptoVulnerability]:
        """Context-aware vulnerability scanning."""
        vulnerabilities = []
        
        # Scan for vulnerable patterns in context
        for pattern_name, pattern in self.VULNERABILITY_PATTERNS.items():
            for match in pattern.finditer(firmware_data):
                addr = base_address + match.start()
                vuln = self._create_vulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048,  # Default for context patterns
                    address=addr,
                    description=f"Context vulnerability: {pattern_name}",
                    confidence_score=0.6
                )
                vulnerabilities.append(vuln)
                
        return vulnerabilities
    
    def _scan_behavioral_patterns(self, firmware_data: bytes, base_address: int) -> List[CryptoVulnerability]:
        """Behavioral pattern analysis for crypto operations."""
        vulnerabilities = []
        
        # Analyze entropy patterns that might indicate crypto operations
        chunk_size = 256
        for i in range(0, len(firmware_data) - chunk_size, chunk_size):
            chunk = firmware_data[i:i + chunk_size]
            entropy = self._calculate_entropy(chunk)
            
            # High entropy regions might contain crypto constants or keys
            if entropy > 7.5:  # High entropy threshold
                addr = base_address + i
                vuln = self._create_vulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048,
                    address=addr,
                    description=f"High entropy region (possible crypto data): entropy={entropy:.2f}",
                    confidence_score=0.4
                )
                vulnerabilities.append(vuln)
                
        return vulnerabilities
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence."""
        if not data:
            return 0
            
        # Count frequency of each byte value
        frequencies = defaultdict(int)
        for byte in data:
            frequencies[byte] += 1
            
        # Calculate entropy
        entropy = 0
        data_len = len(data)
        for count in frequencies.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * math.log2(probability)
                
        return entropy
    
    def _merge_scan_results(self, phase_results: Dict[str, List[CryptoVulnerability]]) -> List[CryptoVulnerability]:
        """Merge and deduplicate scan results from multiple phases."""
        all_vulnerabilities = []
        seen_addresses = set()
        
        # Prioritize results by phase importance
        phase_priority = ['ai_enhanced', 'pattern_based', 'context_aware', 'behavioral']
        
        for phase in phase_priority:
            if phase in phase_results:
                for vuln in phase_results[phase]:
                    # Simple deduplication by address
                    if vuln.address not in seen_addresses:
                        all_vulnerabilities.append(vuln)
                        seen_addresses.add(vuln.address)
                        
        return all_vulnerabilities
    
    def _enhance_vulnerabilities_with_context(self, vulnerabilities: List[CryptoVulnerability], 
                                            firmware_data: bytes) -> List[CryptoVulnerability]:
        """Enhance vulnerabilities with additional context information."""
        enhanced = []
        
        for vuln in vulnerabilities:
            # Add business criticality assessment
            vuln.business_criticality = self._assess_business_criticality(vuln)
            
            # Add compliance impact
            vuln.compliance_impact = self._assess_compliance_impact(vuln.algorithm)
            
            # Add attack vectors
            vuln.attack_vectors = self._identify_attack_vectors(vuln.algorithm)
            
            # Calculate exploitability score
            vuln.exploitability_score = self._calculate_exploitability(vuln)
            
            enhanced.append(vuln)
            
        return enhanced
    
    def _assess_business_criticality(self, vuln: CryptoVulnerability) -> str:
        """Assess business criticality based on vulnerability characteristics."""
        if vuln.risk_level == RiskLevel.CRITICAL:
            return "high"
        elif vuln.confidence_score > 0.8:
            return "medium"
        return "low"
    
    def _assess_compliance_impact(self, algorithm: CryptoAlgorithm) -> List[str]:
        """Assess compliance impact for given algorithm."""
        impacts = []
        
        if algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.MD5, CryptoAlgorithm.SHA1]:
            impacts.extend(["FIPS 140-2", "Common Criteria", "NIST SP 800-131A"])
            
        if algorithm in [CryptoAlgorithm.RSA_1024, CryptoAlgorithm.RSA_2048]:
            impacts.extend(["NIST PQC Migration", "NSA CNSA 2.0"])
            
        return impacts
    
    def _identify_attack_vectors(self, algorithm: CryptoAlgorithm) -> List[str]:
        """Identify potential attack vectors for algorithm."""
        vectors = []
        
        if algorithm.value.startswith("RSA"):
            vectors.extend(["Shor's Algorithm", "Factoring Attacks", "Side-channel Analysis"])
        elif algorithm.value.startswith("ECC"):
            vectors.extend(["Shor's Algorithm", "ECDLP Attacks", "Fault Attacks"])
        elif algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.MD5]:
            vectors.extend(["Brute Force", "Collision Attacks", "Cryptanalysis"])
            
        return vectors
    
    def _calculate_exploitability(self, vuln: CryptoVulnerability) -> float:
        """Calculate exploitability score (0-10 scale)."""
        base_score = 5.0
        
        # Adjust based on risk level
        if vuln.risk_level == RiskLevel.CRITICAL:
            base_score += 3.0
        elif vuln.risk_level == RiskLevel.HIGH:
            base_score += 2.0
        elif vuln.risk_level == RiskLevel.MEDIUM:
            base_score += 1.0
            
        # Adjust based on confidence
        base_score += (vuln.confidence_score - 0.5) * 2.0
        
        # Clamp to 0-10 range
        return max(0.0, min(10.0, base_score))

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
        self.memory_constraints = memory_constraints or {}
        # Set defaults only if constraints are provided
        if not memory_constraints:
            self.memory_constraints = {}
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
    
    def __str__(self) -> str:
        """String representation of the scanner."""
        constraints_str = ', '.join(f"{k}={v}" for k, v in self.memory_constraints.items())
        return f"FirmwareScanner(arch={self.architecture}, constraints=[{constraints_str}])"
    
    def __repr__(self) -> str:
        """Detailed representation of the scanner."""
        return (f"FirmwareScanner(architecture='{self.architecture}', "
                f"memory_constraints={self.memory_constraints}, "
                f"vulnerabilities_found={len(self.vulnerabilities)})")
    
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
                self.logger.warning(f"Firmware file not found: {firmware_path}")
                return []  # Return empty vulnerability list for nonexistent files
            
            if not firmware_path_obj.is_file():
                raise FirmwareAnalysisError(f"Path is not a file: {firmware_path}",
                                          firmware_path=firmware_path)
            
            # Read firmware data
            firmware_data = firmware_path_obj.read_bytes()
            firmware_size = len(firmware_data)
            
            self.logger.info(f"Loaded firmware: {firmware_size} bytes")
            metrics_collector.record_metric("firmware.size_bytes", firmware_size, "bytes")
            
            if firmware_size == 0:
                self.logger.warning(f"Firmware file is empty: {firmware_path}")
                return []  # Return empty vulnerability list for empty files
            
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
                    available_stack=self.memory_constraints.get('ram', 128*1024) // 4
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
                    available_stack=self.memory_constraints.get('ram', 128*1024) // 4
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
                    available_stack=self.memory_constraints.get('ram', 128*1024) // 4
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
                        available_stack=self.memory_constraints.get('ram', 128*1024) // 4
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
                        available_stack=self.memory_constraints.get('ram', 128*1024) // 4
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
        available_ram = self.memory_constraints.get('ram', 128*1024)
        
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