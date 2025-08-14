"""Enhanced security module for PQC IoT Retrofit Scanner Generation 2."""

import hashlib
import hmac
import os
import secrets
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

from .error_handling import SecurityError, ValidationError, InputValidator


@dataclass
class SecurityContext:
    """Security context for firmware analysis operations."""
    session_id: str
    timestamp: float
    integrity_hash: str
    access_level: str = "standard"
    rate_limit_remaining: int = 100
    

class SecureFirmwareHandler:
    """Secure firmware file handling with integrity verification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._max_file_size = 100 * 1024 * 1024  # 100MB
        self._allowed_extensions = {'.bin', '.elf', '.hex', '.fw', '.img'}
        self._session_contexts = {}
        
    def create_security_context(self, user_id: str = "anonymous") -> SecurityContext:
        """Create a new security context for firmware analysis."""
        session_id = secrets.token_hex(16)
        timestamp = time.time()
        
        # Generate integrity hash
        context_data = f"{session_id}{timestamp}{user_id}".encode()
        integrity_hash = hashlib.sha256(context_data).hexdigest()
        
        context = SecurityContext(
            session_id=session_id,
            timestamp=timestamp,
            integrity_hash=integrity_hash,
            access_level="standard",
            rate_limit_remaining=100
        )
        
        self._session_contexts[session_id] = context
        self.logger.info(f"Created security context: {session_id}")
        
        return context
    
    def validate_firmware_security(self, firmware_path: str, context: SecurityContext) -> Dict[str, Any]:
        """Perform comprehensive security validation of firmware file."""
        
        if context.session_id not in self._session_contexts:
            raise SecurityError("Invalid security context")
        
        # Rate limiting
        if context.rate_limit_remaining <= 0:
            raise SecurityError("Rate limit exceeded")
        
        context.rate_limit_remaining -= 1
        
        firmware_path_obj = Path(firmware_path)
        
        # File existence and accessibility
        if not firmware_path_obj.exists():
            raise ValidationError(f"Firmware file not found: {firmware_path}")
        
        if not firmware_path_obj.is_file():
            raise SecurityError(f"Path is not a regular file: {firmware_path}")
        
        # File extension validation
        if firmware_path_obj.suffix.lower() not in self._allowed_extensions:
            raise SecurityError(f"Unsafe file extension: {firmware_path_obj.suffix}")
        
        # File size validation
        file_size = firmware_path_obj.stat().st_size
        if file_size > self._max_file_size:
            raise SecurityError(f"File too large: {file_size} bytes (max: {self._max_file_size})")
        
        if file_size == 0:
            raise ValidationError("Firmware file is empty")
        
        # Content-based validation
        try:
            with open(firmware_path_obj, 'rb') as f:
                # Read first few bytes for magic number detection
                header = f.read(512)
                
                # Basic file format detection
                file_type = self._detect_file_format(header)
                
                # Calculate file hash for integrity
                f.seek(0)
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                
                file_integrity = file_hash.hexdigest()
                
        except IOError as e:
            raise SecurityError(f"Cannot read firmware file: {e}")
        
        # Check for suspicious patterns (basic malware detection)
        security_flags = self._check_security_patterns(header)
        
        validation_result = {
            'file_path': str(firmware_path_obj),
            'file_size': file_size,
            'file_type': file_type,
            'integrity_hash': file_integrity,
            'security_flags': security_flags,
            'validation_time': time.time(),
            'context_id': context.session_id
        }
        
        self.logger.info(f"Firmware security validation complete: {firmware_path}")
        
        return validation_result
    
    def _detect_file_format(self, header: bytes) -> str:
        """Detect firmware file format from header."""
        
        # ELF magic number
        if header.startswith(b'\x7fELF'):
            return 'ELF'
        
        # Intel HEX format
        if header.startswith(b':'):
            return 'Intel HEX'
        
        # Motorola S-record
        if header.startswith(b'S0'):
            return 'Motorola S-record'
        
        # ARM Cortex-M vector table (common pattern)
        if len(header) >= 8:
            # Check for reasonable stack pointer (typical range)
            stack_ptr = int.from_bytes(header[0:4], 'little')
            reset_vector = int.from_bytes(header[4:8], 'little')
            
            if (0x20000000 <= stack_ptr <= 0x20100000 and  # RAM range
                0x08000000 <= reset_vector <= 0x08100000):  # Flash range
                return 'ARM Cortex-M Binary'
        
        # ESP32 binary
        if header[0:1] == b'\xe9':
            return 'ESP32 Binary'
        
        return 'Unknown Binary'
    
    def _check_security_patterns(self, data: bytes) -> List[str]:
        """Check for suspicious patterns in firmware data."""
        
        flags = []
        
        # Check for potential backdoor strings
        suspicious_strings = [
            b'backdoor', b'debug_shell', b'secret_key',
            b'telnet', b'ftp_server', b'admin_password'
        ]
        
        data_lower = data.lower()
        for pattern in suspicious_strings:
            if pattern in data_lower:
                flags.append(f"Suspicious string detected: {pattern.decode()}")
        
        # Check for high entropy regions (potential encrypted backdoors)
        if len(data) >= 256:
            entropy = self._calculate_entropy(data[:256])
            if entropy > 7.5:  # High entropy threshold
                flags.append("High entropy region detected (potential encrypted content)")
        
        # Check for executable signatures
        if b'MZ' in data[:100]:  # DOS executable header
            flags.append("Windows executable signature detected")
        
        return flags
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        
        if not data:
            return 0.0
        
        # Count byte frequencies
        frequency = {}
        for byte in data:
            frequency[byte] = frequency.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in frequency.values():
            probability = count / data_len
            if probability > 0:
                import math
                entropy -= probability * math.log2(probability)
        
        return entropy


class RateLimiter:
    """Rate limiting for API operations."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}  # session_id -> List[timestamp]
        
    def check_rate_limit(self, session_id: str) -> Tuple[bool, int]:
        """Check if request is within rate limit."""
        
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        # Clean old requests
        if session_id in self._requests:
            self._requests[session_id] = [
                req_time for req_time in self._requests[session_id]
                if req_time > window_start
            ]
        else:
            self._requests[session_id] = []
        
        # Check limit
        current_requests = len(self._requests[session_id])
        
        if current_requests >= self.max_requests:
            return False, 0
        
        # Add current request
        self._requests[session_id].append(current_time)
        remaining = self.max_requests - (current_requests + 1)
        
        return True, remaining


class InputSanitizer:
    """Enhanced input sanitization for Generation 2."""
    
    @staticmethod
    def sanitize_architecture(arch: str) -> str:
        """Sanitize architecture input."""
        
        if not isinstance(arch, str):
            raise ValidationError("Architecture must be a string")
        
        arch = arch.lower().strip()
        
        # Whitelist of allowed architectures
        allowed_archs = {
            'cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7',
            'esp32', 'esp8266', 'riscv32', 'risc-v', 'avr'
        }
        
        if arch not in allowed_archs:
            raise ValidationError(f"Unsupported architecture: {arch}")
        
        return arch
    
    @staticmethod
    def sanitize_memory_constraints(constraints: Dict[str, int]) -> Dict[str, int]:
        """Sanitize memory constraint inputs."""
        
        if not isinstance(constraints, dict):
            raise ValidationError("Memory constraints must be a dictionary")
        
        sanitized = {}
        
        for key, value in constraints.items():
            # Validate key
            if key not in ['flash', 'ram', 'eeprom']:
                raise ValidationError(f"Invalid memory type: {key}")
            
            # Validate value
            if not isinstance(value, int):
                raise ValidationError(f"Memory size must be integer: {key}")
            
            if value <= 0:
                raise ValidationError(f"Memory size must be positive: {key}")
            
            if value > 1024 * 1024 * 1024:  # 1GB limit
                raise ValidationError(f"Memory size too large: {key}")
            
            sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def sanitize_address(address: Any) -> int:
        """Sanitize memory address input."""
        
        if isinstance(address, str):
            # Handle hex strings
            if address.startswith('0x') or address.startswith('0X'):
                try:
                    address = int(address, 16)
                except ValueError:
                    raise ValidationError(f"Invalid hex address: {address}")
            else:
                try:
                    address = int(address)
                except ValueError:
                    raise ValidationError(f"Invalid address: {address}")
        elif isinstance(address, int):
            pass  # Already valid
        else:
            raise ValidationError("Address must be string or integer")
        
        # Validate address range
        if address < 0:
            raise ValidationError("Address cannot be negative")
        
        if address > 0xFFFFFFFF:  # 32-bit limit
            raise ValidationError("Address exceeds 32-bit range")
        
        return address


class SecurityLogger:
    """Security-focused logging for audit trails."""
    
    def __init__(self):
        self.logger = logging.getLogger('security_audit')
        
        # Create security log handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_scan_start(self, session_id: str, firmware_path: str, user_info: str = "anonymous"):
        """Log start of firmware scan."""
        self.logger.info(f"SCAN_START - Session: {session_id} - File: {firmware_path} - User: {user_info}")
    
    def log_scan_complete(self, session_id: str, vulnerabilities_found: int, duration: float):
        """Log completion of firmware scan."""
        self.logger.info(f"SCAN_COMPLETE - Session: {session_id} - Vulns: {vulnerabilities_found} - Duration: {duration:.2f}s")
    
    def log_security_violation(self, session_id: str, violation_type: str, details: str):
        """Log security violation."""
        self.logger.warning(f"SECURITY_VIOLATION - Session: {session_id} - Type: {violation_type} - Details: {details}")
    
    def log_rate_limit_exceeded(self, session_id: str, endpoint: str):
        """Log rate limit exceeded."""
        self.logger.warning(f"RATE_LIMIT_EXCEEDED - Session: {session_id} - Endpoint: {endpoint}")


# Global instances
security_handler = SecureFirmwareHandler()
rate_limiter = RateLimiter()
security_logger = SecurityLogger()


def create_secure_scanner_context(user_id: str = "anonymous") -> SecurityContext:
    """Create a secure context for firmware scanning operations."""
    return security_handler.create_security_context(user_id)


def validate_firmware_securely(firmware_path: str, context: SecurityContext) -> Dict[str, Any]:
    """Perform secure validation of firmware file."""
    return security_handler.validate_firmware_security(firmware_path, context)