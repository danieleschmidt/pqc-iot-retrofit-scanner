"""
Security-hardened components for Generation 2 robustness.

This module implements enhanced security measures:
- Input sanitization and validation
- Secure random number generation
- Memory protection and zeroization
- Constant-time operations
- Anti-tampering measures
"""

import os
import secrets
import hashlib
import hmac
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .error_handling import SecurityError, ValidationError, handle_errors, ErrorSeverity


class SecurityLevel(Enum):
    """Security protection levels."""
    BASIC = "basic"
    ENHANCED = "enhanced"
    PARANOID = "paranoid"


@dataclass
class SecurityContext:
    """Security context for operations."""
    level: SecurityLevel
    entropy_source: str
    memory_protection: bool
    timing_protection: bool
    anti_tampering: bool
    audit_enabled: bool


class SecureRandom:
    """Cryptographically secure random number generator."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.entropy_pool = bytearray()
        self.reseed_counter = 0
        self.last_reseed = time.time()
        self._lock = threading.Lock()
        
        # Initialize entropy sources
        self._initialize_entropy()
    
    def _initialize_entropy(self):
        """Initialize entropy sources."""
        try:
            # Collect entropy from various sources
            entropy_sources = [
                os.urandom(64),                    # OS entropy
                secrets.token_bytes(64),           # secrets module
                hashlib.sha256(str(time.time()).encode()).digest(),  # Time-based
            ]
            
            # Additional entropy for higher security levels
            if self.security_level in [SecurityLevel.ENHANCED, SecurityLevel.PARANOID]:
                entropy_sources.extend([
                    hashlib.sha256(str(threading.current_thread().ident).encode()).digest(),
                    hashlib.sha256(str(id(self)).encode()).digest(),
                ])
            
            # Mix all entropy sources
            for source in entropy_sources:
                self.entropy_pool.extend(source)
            
            # Final mixing
            self._mix_entropy()
            
        except Exception as e:
            raise SecurityError(f"Failed to initialize entropy: {e}", 
                              security_level="high", violation_type="entropy_failure")
    
    def _mix_entropy(self):
        """Mix entropy pool using cryptographic hash."""
        if len(self.entropy_pool) > 0:
            mixed = hashlib.sha512(bytes(self.entropy_pool)).digest()
            self.entropy_pool = bytearray(mixed)
    
    def _should_reseed(self) -> bool:
        """Check if reseeding is needed."""
        time_threshold = 3600 if self.security_level == SecurityLevel.BASIC else 1800  # 1 hour / 30 min
        count_threshold = 10000 if self.security_level == SecurityLevel.BASIC else 5000
        
        return (time.time() - self.last_reseed > time_threshold or 
                self.reseed_counter > count_threshold)
    
    def reseed(self):
        """Reseed the entropy pool."""
        with self._lock:
            self._initialize_entropy()
            self.reseed_counter = 0
            self.last_reseed = time.time()
    
    @handle_errors("secure_random_generation", retry_count=1)
    def generate_bytes(self, length: int) -> bytes:
        """Generate cryptographically secure random bytes."""
        if length <= 0:
            raise ValidationError("Random byte length must be positive", field="length", value=length)
        
        if length > 1024 * 1024:  # 1MB limit
            raise ValidationError("Random byte length too large", field="length", value=length)
        
        with self._lock:
            # Check if reseeding is needed
            if self._should_reseed():
                self.reseed()
            
            # Generate random bytes
            if self.security_level == SecurityLevel.PARANOID:
                # Use multiple sources and XOR them
                random_bytes = bytearray()
                for i in range(length):
                    # Multiple entropy sources
                    byte1 = secrets.randbits(8)
                    byte2 = os.urandom(1)[0]
                    byte3 = (self.entropy_pool[i % len(self.entropy_pool)] if self.entropy_pool else 0)
                    
                    # XOR combination
                    combined_byte = byte1 ^ byte2 ^ byte3
                    random_bytes.append(combined_byte)
                
                result = bytes(random_bytes)
            else:
                # Standard secure generation
                result = secrets.token_bytes(length)
            
            # Update counters
            self.reseed_counter += 1
            
            return result


class SecureMemory:
    """Secure memory management with zeroization."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.allocated_blocks: Dict[int, int] = {}  # id -> size mapping
        self._lock = threading.Lock()
    
    def allocate_secure(self, size: int) -> bytearray:
        """Allocate secure memory block."""
        if size <= 0:
            raise ValidationError("Memory size must be positive", field="size", value=size)
        
        if size > 100 * 1024 * 1024:  # 100MB limit
            raise ValidationError("Memory allocation too large", field="size", value=size)
        
        # Allocate memory block
        block = bytearray(size)
        
        # Track allocation
        with self._lock:
            self.allocated_blocks[id(block)] = size
        
        return block
    
    def secure_zero(self, memory_block: Union[bytearray, memoryview]) -> None:
        """Securely zero memory block."""
        if not memory_block:
            return
        
        block_size = len(memory_block)
        
        # Multiple pass zeroization for higher security
        passes = 1
        if self.security_level == SecurityLevel.ENHANCED:
            passes = 3
        elif self.security_level == SecurityLevel.PARANOID:
            passes = 7
        
        for pass_num in range(passes):
            if pass_num == 0:
                # First pass: zeros
                for i in range(block_size):
                    memory_block[i] = 0
            elif pass_num == 1:
                # Second pass: ones
                for i in range(block_size):
                    memory_block[i] = 0xFF
            else:
                # Subsequent passes: random data
                random_data = secrets.token_bytes(block_size)
                for i in range(block_size):
                    memory_block[i] = random_data[i]
        
        # Final zero pass
        for i in range(block_size):
            memory_block[i] = 0
    
    def deallocate_secure(self, memory_block: bytearray) -> None:
        """Securely deallocate memory block."""
        if not memory_block:
            return
        
        # Zero the memory first
        self.secure_zero(memory_block)
        
        # Remove from tracking
        with self._lock:
            block_id = id(memory_block)
            if block_id in self.allocated_blocks:
                del self.allocated_blocks[block_id]


class ConstantTimeOperations:
    """Constant-time cryptographic operations."""
    
    @staticmethod
    def constant_time_compare(a: bytes, b: bytes) -> bool:
        """Constant-time comparison of byte arrays."""
        if len(a) != len(b):
            return False
        
        result = 0
        for x, y in zip(a, b):
            result |= x ^ y
        
        return result == 0
    
    @staticmethod
    def constant_time_select(condition: bool, true_val: int, false_val: int) -> int:
        """Constant-time conditional selection."""
        # Convert boolean to mask (0x00 or 0xFF)
        mask = (condition & 1) * 0xFF
        
        # Use bitwise operations for selection
        return (mask & true_val) | (~mask & false_val)
    
    @staticmethod
    def constant_time_memcmp(a: bytes, b: bytes, length: int) -> int:
        """Constant-time memory comparison."""
        if len(a) < length or len(b) < length:
            raise ValidationError("Buffer length insufficient for comparison")
        
        result = 0
        for i in range(length):
            result |= a[i] ^ b[i]
        
        return result


class InputSanitizer:
    """Input sanitization and validation."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.max_input_sizes = {
            SecurityLevel.BASIC: 1024 * 1024,      # 1MB
            SecurityLevel.ENHANCED: 512 * 1024,    # 512KB
            SecurityLevel.PARANOID: 256 * 1024     # 256KB
        }
    
    @handle_errors("input_sanitization", retry_count=0)
    def sanitize_firmware_path(self, path: str) -> str:
        """Sanitize firmware file path."""
        if not path:
            raise ValidationError("Firmware path cannot be empty", field="path", value=path)
        
        if not isinstance(path, str):
            raise ValidationError("Firmware path must be string", field="path", value=type(path))
        
        # Length check
        if len(path) > 4096:
            raise ValidationError("Firmware path too long", field="path", value=len(path))
        
        # Character whitelist
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_./\\:')
        if not all(c in allowed_chars for c in path):
            raise SecurityError("Invalid characters in firmware path", 
                              security_level="medium", violation_type="path_traversal_attempt")
        
        # Path traversal protection
        dangerous_patterns = ['../', '..\\', '../', '..\\']
        normalized_path = path.lower()
        for pattern in dangerous_patterns:
            if pattern in normalized_path:
                raise SecurityError("Path traversal attempt detected", 
                                  security_level="high", violation_type="path_traversal")
        
        return path
    
    def sanitize_memory_constraints(self, constraints: Dict[str, int]) -> Dict[str, int]:
        """Sanitize memory constraint values."""
        if not isinstance(constraints, dict):
            raise ValidationError("Memory constraints must be dict", field="constraints", value=type(constraints))
        
        sanitized = {}
        max_memory = self.max_input_sizes[self.security_level]
        
        for key, value in constraints.items():
            # Validate key
            if key not in ['flash', 'ram', 'heap', 'stack']:
                raise ValidationError(f"Invalid constraint key: {key}", field="constraint_key", value=key)
            
            # Validate value
            if not isinstance(value, int):
                raise ValidationError(f"Constraint {key} must be integer", field=key, value=type(value))
            
            if value <= 0:
                raise ValidationError(f"Constraint {key} must be positive", field=key, value=value)
            
            if value > max_memory:
                raise SecurityError(f"Constraint {key} exceeds maximum allowed size", 
                                  security_level="medium", violation_type="resource_exhaustion")
            
            sanitized[key] = value
        
        return sanitized
    
    def sanitize_architecture(self, arch: str) -> str:
        """Sanitize architecture string."""
        if not arch:
            raise ValidationError("Architecture cannot be empty", field="architecture", value=arch)
        
        if not isinstance(arch, str):
            raise ValidationError("Architecture must be string", field="architecture", value=type(arch))
        
        # Whitelist of allowed architectures
        allowed_archs = {
            'cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7',
            'esp32', 'esp8266', 'riscv32', 'risc-v', 'avr'
        }
        
        if arch.lower() not in allowed_archs:
            raise SecurityError(f"Unsupported architecture: {arch}", 
                              security_level="medium", violation_type="invalid_architecture")
        
        return arch.lower()


class AntiTampering:
    """Anti-tampering and integrity protection."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.integrity_keys: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        
        # Generate integrity keys
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize integrity protection keys."""
        secure_random = SecureRandom(self.security_level)
        
        with self._lock:
            self.integrity_keys = {
                'firmware_integrity': secure_random.generate_bytes(32),
                'patch_integrity': secure_random.generate_bytes(32),
                'config_integrity': secure_random.generate_bytes(32)
            }
    
    def compute_integrity_hash(self, data: bytes, key_type: str = 'firmware_integrity') -> bytes:
        """Compute integrity hash for data."""
        if key_type not in self.integrity_keys:
            raise SecurityError(f"Unknown integrity key type: {key_type}", 
                              security_level="high", violation_type="integrity_violation")
        
        with self._lock:
            key = self.integrity_keys[key_type]
        
        # Use HMAC for authenticated integrity
        return hmac.new(key, data, hashlib.sha256).digest()
    
    def verify_integrity(self, data: bytes, expected_hash: bytes, key_type: str = 'firmware_integrity') -> bool:
        """Verify data integrity."""
        try:
            computed_hash = self.compute_integrity_hash(data, key_type)
            return ConstantTimeOperations.constant_time_compare(computed_hash, expected_hash)
        except Exception:
            return False
    
    def protect_data(self, data: bytes, key_type: str = 'firmware_integrity') -> Dict[str, Any]:
        """Protect data with integrity verification."""
        integrity_hash = self.compute_integrity_hash(data, key_type)
        
        return {
            'data': data,
            'integrity_hash': integrity_hash,
            'key_type': key_type,
            'timestamp': time.time(),
            'protection_level': self.security_level.value
        }
    
    def verify_protected_data(self, protected_data: Dict[str, Any]) -> bool:
        """Verify protected data integrity."""
        try:
            data = protected_data['data']
            expected_hash = protected_data['integrity_hash']
            key_type = protected_data['key_type']
            
            return self.verify_integrity(data, expected_hash, key_type)
        except KeyError as e:
            raise SecurityError(f"Missing protection field: {e}", 
                              security_level="high", violation_type="protection_bypass_attempt")


class SecurityAuditLogger:
    """Security event auditing and logging."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        self.logger = logging.getLogger("security_audit")
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Configure secure logging
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure security logging."""
        if not self.logger.handlers:
            # Create secure log handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = "medium") -> None:
        """Log security event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'thread_id': threading.current_thread().ident,
            'security_level': self.security_level.value
        }
        
        with self._lock:
            self.events.append(event)
            
            # Limit event history
            if len(self.events) > 10000:
                self.events = self.events[-5000:]  # Keep last 5000 events
        
        # Log to system logger
        log_message = f"[{event_type.upper()}] {details.get('message', 'Security event')}"
        
        if severity == "critical":
            self.logger.critical(log_message)
        elif severity == "high":
            self.logger.error(log_message)
        elif severity == "medium":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def get_security_events(self, event_type: Optional[str] = None, 
                           severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get security events matching criteria."""
        with self._lock:
            filtered_events = []
            
            for event in self.events:
                if event_type and event['event_type'] != event_type:
                    continue
                if severity and event['severity'] != severity:
                    continue
                
                filtered_events.append(event.copy())
        
        return filtered_events
    
    def clear_events(self) -> None:
        """Clear security event log."""
        with self._lock:
            self.events.clear()


class SecurityHardenedFirmwareScanner:
    """Security-hardened firmware scanner wrapper."""
    
    def __init__(self, architecture: str, memory_constraints: Dict[str, int] = None,
                 security_level: SecurityLevel = SecurityLevel.ENHANCED):
        self.security_level = security_level
        
        # Initialize security components
        self.sanitizer = InputSanitizer(security_level)
        self.memory_manager = SecureMemory(security_level)
        self.anti_tampering = AntiTampering(security_level)
        self.audit_logger = SecurityAuditLogger(security_level)
        
        # Sanitize inputs
        self.architecture = self.sanitizer.sanitize_architecture(architecture)
        self.memory_constraints = (
            self.sanitizer.sanitize_memory_constraints(memory_constraints) 
            if memory_constraints else {}
        )
        
        # Log initialization
        self.audit_logger.log_security_event(
            "scanner_initialization",
            {
                "message": "Security-hardened scanner initialized",
                "architecture": self.architecture,
                "security_level": security_level.value
            },
            "info"
        )
    
    @handle_errors("secure_firmware_scan", retry_count=1)
    def secure_scan_firmware(self, firmware_path: str, base_address: int = 0) -> Dict[str, Any]:
        """Securely scan firmware with enhanced protection."""
        # Sanitize inputs
        safe_path = self.sanitizer.sanitize_firmware_path(firmware_path)
        
        # Log scan start
        self.audit_logger.log_security_event(
            "firmware_scan_start",
            {
                "message": "Starting secure firmware scan",
                "firmware_path": safe_path,
                "base_address": f"0x{base_address:08x}"
            },
            "info"
        )
        
        try:
            # Import the actual scanner (avoid circular imports)
            from .scanner import FirmwareScanner
            
            # Create scanner with hardened configuration
            scanner = FirmwareScanner(self.architecture, self.memory_constraints)
            
            # Perform scan
            vulnerabilities = scanner.scan_firmware(safe_path, base_address)
            report = scanner.generate_report()
            
            # Protect report data
            protected_report = self.anti_tampering.protect_data(
                str(report).encode('utf-8'), 
                'firmware_integrity'
            )
            
            # Log successful scan
            self.audit_logger.log_security_event(
                "firmware_scan_complete",
                {
                    "message": "Firmware scan completed successfully",
                    "vulnerabilities_found": len(vulnerabilities),
                    "report_protected": True
                },
                "info"
            )
            
            return {
                'vulnerabilities': vulnerabilities,
                'report': report,
                'protected_report': protected_report,
                'security_context': {
                    'level': self.security_level.value,
                    'integrity_protected': True,
                    'audit_enabled': True
                }
            }
            
        except Exception as e:
            # Log security incident
            self.audit_logger.log_security_event(
                "firmware_scan_failure",
                {
                    "message": f"Firmware scan failed: {str(e)}",
                    "error_type": type(e).__name__,
                    "firmware_path": safe_path
                },
                "high"
            )
            raise
    
    def verify_scan_integrity(self, protected_data: Dict[str, Any]) -> bool:
        """Verify scan result integrity."""
        try:
            is_valid = self.anti_tampering.verify_protected_data(protected_data)
            
            self.audit_logger.log_security_event(
                "integrity_verification",
                {
                    "message": "Scan result integrity verification",
                    "result": "valid" if is_valid else "invalid"
                },
                "medium" if not is_valid else "info"
            )
            
            return is_valid
            
        except Exception as e:
            self.audit_logger.log_security_event(
                "integrity_verification_error",
                {
                    "message": f"Integrity verification failed: {str(e)}",
                    "error_type": type(e).__name__
                },
                "high"
            )
            return False


# Global security components
_default_security_level = SecurityLevel.ENHANCED
secure_random = SecureRandom(_default_security_level)
secure_memory = SecureMemory(_default_security_level)
constant_time_ops = ConstantTimeOperations()
input_sanitizer = InputSanitizer(_default_security_level)
anti_tampering = AntiTampering(_default_security_level)
security_audit = SecurityAuditLogger(_default_security_level)


def create_hardened_scanner(architecture: str, memory_constraints: Dict[str, int] = None,
                           security_level: SecurityLevel = SecurityLevel.ENHANCED) -> SecurityHardenedFirmwareScanner:
    """Create a security-hardened firmware scanner."""
    return SecurityHardenedFirmwareScanner(architecture, memory_constraints, security_level)


def get_security_status() -> Dict[str, Any]:
    """Get current security status."""
    return {
        'security_level': _default_security_level.value,
        'entropy_initialized': len(secure_random.entropy_pool) > 0,
        'integrity_keys_active': len(anti_tampering.integrity_keys) > 0,
        'audit_events': len(security_audit.events),
        'memory_blocks_tracked': len(secure_memory.allocated_blocks),
        'timestamp': time.time()
    }