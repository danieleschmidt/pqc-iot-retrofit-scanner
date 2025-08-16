#!/usr/bin/env python3
"""
Security Validation System - Generation 2
Comprehensive security checks, input sanitization, and vulnerability detection
"""

import os
import re
import hashlib
import hmac
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import secrets
import logging


class SecurityLevel(Enum):
    """Security validation levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SecurityThreat(Enum):
    """Types of security threats."""
    MALICIOUS_FIRMWARE = "malicious_firmware"
    INJECTION_ATTACK = "injection_attack"
    BUFFER_OVERFLOW = "buffer_overflow"
    PATH_TRAVERSAL = "path_traversal"
    CODE_INJECTION = "code_injection"
    WEAK_CRYPTO = "weak_crypto"
    SIDE_CHANNEL = "side_channel"
    TIMING_ATTACK = "timing_attack"


@dataclass
class SecurityIssue:
    """Security issue details."""
    threat_type: SecurityThreat
    severity: SecurityLevel
    description: str
    location: Optional[str] = None
    mitigation: Optional[str] = None
    cve_references: List[str] = None


class SecurityValidator:
    """Comprehensive security validation system."""
    
    # Malicious patterns to detect
    MALICIOUS_PATTERNS = {
        'shellcode': [
            rb'\x31\xc0\x50\x68',  # Common x86 shellcode
            rb'\x48\x31\xff\x57',  # Common x64 shellcode
            rb'\x90\x90\x90\x90',  # NOP sled
        ],
        'backdoor_strings': [
            b'backdoor',
            b'rootkit',
            b'keylogger',
            b'password',
            b'admin123',
            b'secret_key'
        ],
        'suspicious_urls': [
            rb'http://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP URLs
            rb'\.onion',  # Tor addresses
            rb'bit\.ly',  # URL shorteners
        ]
    }
    
    # Weak cryptographic patterns
    WEAK_CRYPTO_PATTERNS = {
        'md5': rb'MD5|md5',
        'sha1': rb'SHA1|sha1',
        'des': rb'DES|3DES',
        'rc4': rb'RC4|rc4',
        'weak_rsa': rb'RSA-1024|RSA-512',
        'weak_dh': rb'DH-1024|DH-512'
    }
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        rb'\.\./|\.\.\x5c',  # ../ or ..\
        rb'%2e%2e%2f|%2e%2e%5c',  # URL encoded ../
        rb'\x00',  # Null bytes
    ]
    
    def __init__(self):
        """Initialize security validator."""
        self.logger = logging.getLogger('SecurityValidator')
        self.detected_issues: List[SecurityIssue] = []
        self.validation_rules = self._load_validation_rules()
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load security validation rules."""
        return {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'allowed_extensions': ['.bin', '.hex', '.elf', '.img'],
            'max_path_length': 255,
            'max_filename_length': 100,
            'require_checksum_validation': True,
            'scan_for_malware': True,
            'validate_certificates': True
        }
    
    def validate_firmware_security(self, firmware_path: str, 
                                 expected_checksum: Optional[str] = None) -> List[SecurityIssue]:
        """Comprehensive firmware security validation."""
        issues = []
        
        try:
            # Basic file validation
            issues.extend(self._validate_file_security(firmware_path))
            
            # Read firmware data
            with open(firmware_path, 'rb') as f:
                firmware_data = f.read()
            
            # Checksum validation
            if expected_checksum:
                issues.extend(self._validate_checksum(firmware_data, expected_checksum))
            
            # Malware scanning
            issues.extend(self._scan_for_malware(firmware_data))
            
            # Weak crypto detection
            issues.extend(self._detect_weak_crypto(firmware_data))
            
            # Buffer overflow detection
            issues.extend(self._detect_buffer_overflows(firmware_data))
            
            # Timing attack susceptibility
            issues.extend(self._analyze_timing_vulnerabilities(firmware_data))
            
        except Exception as e:
            issues.append(SecurityIssue(
                threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                severity=SecurityLevel.HIGH,
                description=f"Security validation failed: {str(e)}",
                mitigation="Manual security review required"
            ))
        
        self.detected_issues.extend(issues)
        return issues
    
    def _validate_file_security(self, file_path: str) -> List[SecurityIssue]:
        """Validate file-level security."""
        issues = []
        
        # Path traversal check
        if any(pattern in file_path.encode() for patterns in self.PATH_TRAVERSAL_PATTERNS for pattern in [patterns]):
            issues.append(SecurityIssue(
                threat_type=SecurityThreat.PATH_TRAVERSAL,
                severity=SecurityLevel.CRITICAL,
                description=f"Path traversal detected in: {file_path}",
                location=file_path,
                mitigation="Sanitize file paths and use absolute paths only"
            ))
        
        # File size check
        file_size = os.path.getsize(file_path)
        if file_size > self.validation_rules['max_file_size']:
            issues.append(SecurityIssue(
                threat_type=SecurityThreat.BUFFER_OVERFLOW,
                severity=SecurityLevel.HIGH,
                description=f"File size exceeds limit: {file_size} bytes",
                location=file_path,
                mitigation="Implement file size limits and streaming processing"
            ))
        
        # Extension validation
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.validation_rules['allowed_extensions']:
            issues.append(SecurityIssue(
                threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                severity=SecurityLevel.MEDIUM,
                description=f"Unexpected file extension: {file_ext}",
                location=file_path,
                mitigation="Verify file type and content match expected format"
            ))
        
        return issues
    
    def _validate_checksum(self, data: bytes, expected_checksum: str) -> List[SecurityIssue]:
        """Validate file integrity using checksum."""
        issues = []
        
        # Calculate SHA-256 checksum
        actual_checksum = hashlib.sha256(data).hexdigest()
        
        if actual_checksum != expected_checksum:
            issues.append(SecurityIssue(
                threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                severity=SecurityLevel.CRITICAL,
                description=f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}",
                mitigation="Verify firmware integrity and source authenticity"
            ))
        
        return issues
    
    def _scan_for_malware(self, data: bytes) -> List[SecurityIssue]:
        """Scan firmware for malware patterns."""
        issues = []
        
        # Check for shellcode patterns
        for pattern in self.MALICIOUS_PATTERNS['shellcode']:
            if pattern in data:
                offset = data.find(pattern)
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                    severity=SecurityLevel.CRITICAL,
                    description=f"Shellcode pattern detected at offset 0x{offset:x}",
                    location=f"offset_0x{offset:x}",
                    mitigation="Quarantine firmware and perform detailed analysis"
                ))
        
        # Check for backdoor strings
        for pattern in self.MALICIOUS_PATTERNS['backdoor_strings']:
            if pattern in data:
                offset = data.find(pattern)
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                    severity=SecurityLevel.HIGH,
                    description=f"Suspicious string '{pattern.decode()}' at offset 0x{offset:x}",
                    location=f"offset_0x{offset:x}",
                    mitigation="Review string context and verify legitimacy"
                ))
        
        # Check for suspicious URLs
        for pattern in self.MALICIOUS_PATTERNS['suspicious_urls']:
            matches = re.finditer(pattern, data)
            for match in matches:
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.MALICIOUS_FIRMWARE,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Suspicious URL pattern at offset 0x{match.start():x}",
                    location=f"offset_0x{match.start():x}",
                    mitigation="Verify URL legitimacy and network behavior"
                ))
        
        return issues
    
    def _detect_weak_crypto(self, data: bytes) -> List[SecurityIssue]:
        """Detect weak cryptographic implementations."""
        issues = []
        
        for crypto_type, pattern in self.WEAK_CRYPTO_PATTERNS.items():
            matches = re.finditer(pattern, data, re.IGNORECASE)
            for match in matches:
                severity = SecurityLevel.CRITICAL if 'weak' in crypto_type else SecurityLevel.HIGH
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.WEAK_CRYPTO,
                    severity=severity,
                    description=f"Weak cryptography detected: {crypto_type} at offset 0x{match.start():x}",
                    location=f"offset_0x{match.start():x}",
                    mitigation=f"Replace {crypto_type} with post-quantum secure alternatives"
                ))
        
        return issues
    
    def _detect_buffer_overflows(self, data: bytes) -> List[SecurityIssue]:
        """Detect potential buffer overflow vulnerabilities."""
        issues = []
        
        # Look for dangerous C functions
        dangerous_functions = [
            b'strcpy', b'strcat', b'sprintf', b'gets',
            b'scanf', b'strncpy', b'strncat'
        ]
        
        for func in dangerous_functions:
            if func in data:
                offset = data.find(func)
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.BUFFER_OVERFLOW,
                    severity=SecurityLevel.HIGH,
                    description=f"Dangerous function '{func.decode()}' detected at offset 0x{offset:x}",
                    location=f"offset_0x{offset:x}",
                    mitigation=f"Replace {func.decode()} with safer alternatives"
                ))
        
        # Look for large stack allocations (heuristic)
        stack_patterns = [
            rb'alloca\([0-9]{4,}\)',  # Large alloca calls
            rb'char\s+\w+\[[0-9]{4,}\]'  # Large stack arrays
        ]
        
        for pattern in stack_patterns:
            matches = re.finditer(pattern, data)
            for match in matches:
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.BUFFER_OVERFLOW,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Large stack allocation detected at offset 0x{match.start():x}",
                    location=f"offset_0x{match.start():x}",
                    mitigation="Validate buffer sizes and implement bounds checking"
                ))
        
        return issues
    
    def _analyze_timing_vulnerabilities(self, data: bytes) -> List[SecurityIssue]:
        """Analyze for timing attack vulnerabilities."""
        issues = []
        
        # Look for non-constant-time operations
        timing_vulnerable_patterns = [
            rb'memcmp',     # Non-constant-time comparison
            rb'strcmp',     # String comparison
            rb'if.*==.*key', # Key comparison in conditional
        ]
        
        for pattern in timing_vulnerable_patterns:
            matches = re.finditer(pattern, data, re.IGNORECASE)
            for match in matches:
                issues.append(SecurityIssue(
                    threat_type=SecurityThreat.TIMING_ATTACK,
                    severity=SecurityLevel.MEDIUM,
                    description=f"Timing-vulnerable operation detected at offset 0x{match.start():x}",
                    location=f"offset_0x{match.start():x}",
                    mitigation="Use constant-time comparison functions"
                ))
        
        return issues
    
    def sanitize_input(self, input_string: str) -> str:
        """Sanitize user input to prevent injection attacks."""
        if not isinstance(input_string, str):
            raise TypeError("Input must be a string")
        
        # Remove null bytes
        sanitized = input_string.replace('\x00', '')
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '`', '|', ';', '$']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000]
        
        return sanitized
    
    def validate_path(self, path: str) -> bool:
        """Validate file path for security."""
        # Convert to absolute path
        abs_path = os.path.abspath(path)
        
        # Check for path traversal
        if '..' in path or abs_path != os.path.normpath(abs_path):
            raise ValueError(f"Path traversal detected: {path}")
        
        # Check path length
        if len(abs_path) > self.validation_rules['max_path_length']:
            raise ValueError(f"Path too long: {len(abs_path)} characters")
        
        return True
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token."""
        return secrets.token_hex(length)
    
    def secure_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison."""
        return hmac.compare_digest(a.encode(), b.encode())
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        if not self.detected_issues:
            return {
                "security_status": "clean",
                "total_issues": 0,
                "recommendation": "No security issues detected"
            }
        
        # Count issues by severity
        severity_counts = {}
        threat_counts = {}
        
        for issue in self.detected_issues:
            sev = issue.severity.value
            threat = issue.threat_type.value
            
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            threat_counts[threat] = threat_counts.get(threat, 0) + 1
        
        # Determine overall security status
        if severity_counts.get('critical', 0) > 0:
            security_status = "critical"
            recommendation = "CRITICAL security issues found - do not deploy"
        elif severity_counts.get('high', 0) > 0:
            security_status = "high_risk"
            recommendation = "High-risk security issues - review before deployment"
        elif severity_counts.get('medium', 0) > 0:
            security_status = "medium_risk"
            recommendation = "Medium-risk issues - address when possible"
        else:
            security_status = "low_risk"
            recommendation = "Low-risk issues - monitor and address"
        
        return {
            "security_status": security_status,
            "total_issues": len(self.detected_issues),
            "severity_breakdown": severity_counts,
            "threat_breakdown": threat_counts,
            "recommendation": recommendation,
            "detailed_issues": [
                {
                    "threat": issue.threat_type.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "location": issue.location,
                    "mitigation": issue.mitigation
                }
                for issue in self.detected_issues
            ]
        }


class SecureFileHandler:
    """Secure file handling with validation."""
    
    def __init__(self, validator: SecurityValidator):
        """Initialize with security validator."""
        self.validator = validator
        self.temp_dir = "/tmp/pqc_scanner_secure"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def secure_read(self, file_path: str, max_size: int = 10*1024*1024) -> bytes:
        """Securely read file with validation."""
        # Validate path
        self.validator.validate_path(file_path)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes")
        
        # Read file
        with open(file_path, 'rb') as f:
            return f.read()
    
    def secure_write(self, data: bytes, filename: str) -> str:
        """Securely write data to temporary file."""
        # Sanitize filename
        safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]
        
        # Generate unique file path
        secure_token = self.validator.generate_secure_token(8)
        file_path = os.path.join(self.temp_dir, f"{secure_token}_{safe_filename}")
        
        # Write file with secure permissions
        with open(file_path, 'wb') as f:
            f.write(data)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(file_path, 0o600)
        
        return file_path
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        import glob
        temp_files = glob.glob(os.path.join(self.temp_dir, "*"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except OSError:
                pass


def main():
    """Demo of security validation system."""
    print("Security Validation System - Demo")
    print("=" * 50)
    
    # Initialize validator
    validator = SecurityValidator()
    
    # Create test data with vulnerabilities
    test_firmware = (
        b"ARM firmware header\x00\x00\x00\x00" +
        b"strcpy(buffer, user_input);" +  # Buffer overflow
        b"MD5_Update(&ctx, data, len);" +  # Weak crypto
        b"\x31\xc0\x50\x68" +  # Shellcode pattern
        b"admin123" +  # Backdoor string
        b"http://192.168.1.100/evil"  # Suspicious URL
    )
    
    # Write test firmware
    secure_handler = SecureFileHandler(validator)
    test_file = secure_handler.secure_write(test_firmware, "test_firmware.bin")
    
    print(f"Created test firmware: {test_file}")
    
    # Perform security validation
    print("\nPerforming security validation...")
    issues = validator.validate_firmware_security(test_file)
    
    # Generate security report
    report = validator.get_security_report()
    print("\nSecurity Report:")
    print(json.dumps(report, indent=2))
    
    # Test input sanitization
    print("\nTesting input sanitization:")
    malicious_input = "'; DROP TABLE users; --<script>alert('xss')</script>"
    sanitized = validator.sanitize_input(malicious_input)
    print(f"Original: {malicious_input}")
    print(f"Sanitized: {sanitized}")
    
    # Test secure token generation
    print(f"\nSecure token: {validator.generate_secure_token()}")
    
    # Cleanup
    secure_handler.cleanup_temp_files()
    print("\nTemporary files cleaned up")


if __name__ == '__main__':
    main()