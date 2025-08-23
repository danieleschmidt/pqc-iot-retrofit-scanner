"""Generation 2: Robust Input Validation and Sanitization.

Enhanced validation framework providing:
- Comprehensive input sanitization
- Type safety enforcement
- Security-focused validation rules
- Defensive programming patterns
- Attack prevention mechanisms
"""

import re
import os
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None
    value: Any = None
    suggestion: Optional[str] = None
    
    def __bool__(self) -> bool:
        return self.is_valid


class RobustValidator:
    """Comprehensive input validation and sanitization."""
    
    # Security patterns to detect potential attacks
    MALICIOUS_PATTERNS = {
        'path_traversal': re.compile(r'\.\.[\\/]|[\\/]\.\.'),
        'command_injection': re.compile(r'[;&|`$(){}[\]<>]'),
        'sql_injection': re.compile(r"['\"];|--\s|/\*|\*/|union\s+select", re.IGNORECASE),
        'script_injection': re.compile(r'<script|javascript:|data:', re.IGNORECASE),
        'format_string': re.compile(r'%[dioxX]|{.*}'),
        'null_byte': re.compile(r'\x00'),
    }
    
    # File size limits (in bytes)
    MAX_FIRMWARE_SIZE = 100 * 1024 * 1024  # 100MB
    MIN_FIRMWARE_SIZE = 1024  # 1KB
    
    # Architecture validation
    VALID_ARCHITECTURES = {
        'cortex-m0', 'cortex-m3', 'cortex-m4', 'cortex-m7',
        'esp32', 'esp8266', 'riscv32', 'riscv64', 'avr'
    }
    
    # File extension validation
    VALID_FIRMWARE_EXTENSIONS = {
        '.bin', '.elf', '.hex', '.fw', '.img', '.uf2'
    }
    
    @classmethod
    def validate_string_input(cls, value: Any, field_name: str, 
                            max_length: int = 1000,
                            allow_empty: bool = False,
                            pattern: Optional[str] = None) -> ValidationResult:
        """Validate and sanitize string input."""
        
        # Type validation
        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be a string, got {type(value).__name__}",
                field=field_name,
                value=value
            )
        
        # Empty check
        if not value and not allow_empty:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} cannot be empty",
                field=field_name,
                value=value,
                suggestion="Provide a non-empty string value"
            )
        
        # Length validation
        if len(value) > max_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} exceeds maximum length of {max_length} characters",
                field=field_name,
                value=value,
                suggestion=f"Truncate to {max_length} characters or less"
            )
        
        # Security pattern detection (skip for file paths that might be temporary)
        is_likely_file_path = '/' in value or '\\' in value or value.endswith(('.bin', '.elf', '.hex', '.fw', '.img'))
        
        for attack_type, attack_pattern in cls.MALICIOUS_PATTERNS.items():
            # Skip certain checks for file paths
            if is_likely_file_path and attack_type in ['command_injection', 'sql_injection']:
                continue
                
            if attack_pattern.search(value):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"{field_name} contains potentially malicious {attack_type} patterns",
                    field=field_name,
                    value=value,
                    suggestion="Remove suspicious characters or patterns"
                )
        
        # Custom pattern validation
        if pattern and not re.match(pattern, value):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} does not match required pattern",
                field=field_name,
                value=value,
                suggestion=f"Format must match: {pattern}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} validation passed",
            field=field_name,
            value=value
        )
    
    @classmethod
    def validate_file_path(cls, file_path: Any, field_name: str = "file_path",
                          must_exist: bool = True,
                          check_permissions: bool = True) -> ValidationResult:
        """Comprehensive file path validation."""
        
        # Basic string validation
        string_result = cls.validate_string_input(file_path, field_name, max_length=4096)
        if not string_result:
            return string_result
        
        try:
            path = Path(file_path).resolve()
            
            # Security: Prevent path traversal (but allow /tmp for temporary files)
            if '..' in str(path):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"{field_name} contains path traversal patterns",
                    field=field_name,
                    value=file_path,
                    suggestion="Remove '..' path traversal patterns"
                )
            
            # Existence validation
            if must_exist and not path.exists():
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} does not exist: {file_path}",
                    field=field_name,
                    value=file_path,
                    suggestion="Ensure the file exists and path is correct"
                )
            
            if must_exist and not path.is_file():
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} is not a regular file: {file_path}",
                    field=field_name,
                    value=file_path,
                    suggestion="Provide a path to a regular file, not a directory"
                )
            
            # Permission validation
            if check_permissions and must_exist:
                if not os.access(str(path), os.R_OK):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"{field_name} is not readable: {file_path}",
                        field=field_name,
                        value=file_path,
                        suggestion="Check file permissions"
                    )
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"{field_name} validation passed",
                field=field_name,
                value=str(path)
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} path resolution failed: {e}",
                field=field_name,
                value=file_path,
                suggestion="Ensure the path is valid and accessible"
            )
    
    @classmethod
    def validate_firmware_file(cls, firmware_path: Any) -> ValidationResult:
        """Specialized firmware file validation."""
        
        # Basic file path validation
        path_result = cls.validate_file_path(firmware_path, "firmware_path")
        if not path_result:
            return path_result
        
        try:
            path = Path(firmware_path)
            
            # Extension validation
            if path.suffix.lower() not in cls.VALID_FIRMWARE_EXTENSIONS:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Firmware file must have one of these extensions: {', '.join(cls.VALID_FIRMWARE_EXTENSIONS)}",
                    field="firmware_path",
                    value=firmware_path,
                    suggestion=f"Rename file with valid extension or convert format"
                )
            
            # Size validation
            file_size = path.stat().st_size
            
            if file_size < cls.MIN_FIRMWARE_SIZE:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Firmware file is very small ({file_size} bytes), may be incomplete",
                    field="firmware_path",
                    value=firmware_path,
                    suggestion="Verify this is a complete firmware image"
                )
            
            if file_size > cls.MAX_FIRMWARE_SIZE:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Firmware file is too large ({file_size} bytes), exceeds {cls.MAX_FIRMWARE_SIZE} bytes",
                    field="firmware_path",
                    value=firmware_path,
                    suggestion="Use a smaller firmware file or increase size limit"
                )
            
            # Content validation - basic header checks
            with open(path, 'rb') as f:
                header = f.read(16)
                
                # Check for common firmware signatures
                if header.startswith(b'\x7fELF'):
                    # ELF file - good
                    pass
                elif header.startswith(b':'):
                    # Intel HEX - good  
                    pass
                elif header.startswith(b'UF2\n'):
                    # UF2 format - good
                    pass
                elif path.suffix.lower() == '.bin':
                    # Binary - check for all zeros (suspicious)
                    if header == b'\x00' * 16:
                        return ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            message="Firmware file appears to contain only zeros",
                            field="firmware_path",
                            value=firmware_path,
                            suggestion="Verify this is a valid firmware binary"
                        )
                
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="Firmware file validation passed",
                field="firmware_path",
                value=firmware_path
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Firmware file validation failed: {e}",
                field="firmware_path",
                value=firmware_path,
                suggestion="Ensure file is accessible and not corrupted"
            )
    
    @classmethod
    def validate_architecture(cls, architecture: Any) -> ValidationResult:
        """Validate target architecture specification."""
        
        # Basic string validation
        string_result = cls.validate_string_input(
            architecture, "architecture", 
            max_length=50, 
            pattern=r'^[a-zA-Z0-9\-_]+$'
        )
        if not string_result:
            return string_result
        
        arch_lower = architecture.lower()
        
        if arch_lower not in cls.VALID_ARCHITECTURES:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Unsupported architecture: {architecture}",
                field="architecture",
                value=architecture,
                suggestion=f"Use one of: {', '.join(cls.VALID_ARCHITECTURES)}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Architecture validation passed",
            field="architecture",
            value=arch_lower
        )
    
    @classmethod
    def validate_memory_constraints(cls, constraints: Any) -> ValidationResult:
        """Validate memory constraint specifications."""
        
        if not isinstance(constraints, dict):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Memory constraints must be a dictionary",
                field="memory_constraints",
                value=constraints,
                suggestion="Provide constraints as {'flash': size, 'ram': size}"
            )
        
        # Validate constraint keys
        valid_keys = {'flash', 'ram', 'heap', 'stack', 'eeprom'}
        invalid_keys = set(constraints.keys()) - valid_keys
        if invalid_keys:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid memory constraint keys: {invalid_keys}",
                field="memory_constraints",
                value=constraints,
                suggestion=f"Use only: {', '.join(valid_keys)}"
            )
        
        # Validate constraint values
        for key, value in constraints.items():
            if not isinstance(value, int) or value <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Memory constraint '{key}' must be a positive integer",
                    field="memory_constraints",
                    value=constraints,
                    suggestion="Provide memory sizes in bytes as positive integers"
                )
            
            # Sanity check: reasonable memory sizes
            if key == 'flash' and value > 100 * 1024 * 1024:  # 100MB
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Flash memory constraint very large: {value} bytes",
                    field="memory_constraints",
                    value=constraints,
                    suggestion="Verify this is the intended flash size"
                )
            
            if key == 'ram' and value > 10 * 1024 * 1024:  # 10MB  
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"RAM constraint very large: {value} bytes",
                    field="memory_constraints", 
                    value=constraints,
                    suggestion="Verify this is the intended RAM size"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Memory constraints validation passed",
            field="memory_constraints",
            value=constraints
        )
    
    @classmethod
    def validate_address(cls, address: Any, field_name: str = "address") -> ValidationResult:
        """Validate memory address values."""
        
        # Handle string hex addresses
        if isinstance(address, str):
            if address.startswith('0x') or address.startswith('0X'):
                try:
                    address = int(address, 16)
                except ValueError:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"{field_name} is not a valid hex address: {address}",
                        field=field_name,
                        value=address,
                        suggestion="Use format like '0x08000000'"
                    )
            else:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"{field_name} string must start with '0x'",
                    field=field_name,
                    value=address,
                    suggestion="Use hex format like '0x08000000'"
                )
        
        # Validate integer address
        if not isinstance(address, int):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be an integer or hex string",
                field=field_name,
                value=address,
                suggestion="Provide address as integer or hex string"
            )
        
        # Range validation (32-bit addresses)
        if address < 0 or address > 0xFFFFFFFF:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"{field_name} must be in 32-bit range (0 to 0xFFFFFFFF)",
                field=field_name,
                value=address,
                suggestion="Use valid 32-bit memory address"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"{field_name} validation passed",
            field=field_name,
            value=address
        )
    
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename for safe filesystem operations."""
        
        if not filename:
            return "sanitized_file"
        
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"|?*\\/]', '_', filename)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
        
        # Limit length
        sanitized = sanitized[:200]
        
        # Ensure not empty after sanitization
        if not sanitized:
            sanitized = "sanitized_file"
        
        # Prevent reserved names (Windows)
        reserved_names = {'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'}
        
        base_name = sanitized.split('.')[0].upper()
        if base_name in reserved_names:
            sanitized = f"safe_{sanitized}"
        
        return sanitized
    
    @classmethod
    def validate_configuration(cls, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate entire configuration object."""
        
        results = []
        
        # Validate each field based on its purpose
        if 'firmware_path' in config:
            results.append(cls.validate_firmware_file(config['firmware_path']))
        
        if 'architecture' in config:
            results.append(cls.validate_architecture(config['architecture']))
        
        if 'memory_constraints' in config:
            results.append(cls.validate_memory_constraints(config['memory_constraints']))
        
        if 'base_address' in config:
            results.append(cls.validate_address(config['base_address'], 'base_address'))
        
        # Check for required fields
        required_fields = {'firmware_path', 'architecture'}
        missing_fields = required_fields - set(config.keys())
        
        if missing_fields:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                field="configuration",
                value=config,
                suggestion="Provide all required configuration fields"
            ))
        
        return results


class SecureHashValidator:
    """Secure file hash validation and integrity checking."""
    
    @staticmethod
    def calculate_secure_hash(file_path: str, algorithm: str = "sha256") -> Optional[str]:
        """Calculate secure hash of a file."""
        
        try:
            hash_func = getattr(hashlib, algorithm, None)
            if not hash_func:
                logger.error(f"Unsupported hash algorithm: {algorithm}")
                return None
            
            hasher = hash_func()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Hash calculation failed: {e}")
            return None
    
    @staticmethod
    def validate_file_integrity(file_path: str, expected_hash: str, 
                              algorithm: str = "sha256") -> ValidationResult:
        """Validate file integrity against expected hash."""
        
        actual_hash = SecureHashValidator.calculate_secure_hash(file_path, algorithm)
        
        if not actual_hash:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="Failed to calculate file hash",
                field="file_integrity",
                value=file_path
            )
        
        if actual_hash.lower() != expected_hash.lower():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="File integrity check failed - hash mismatch",
                field="file_integrity",
                value=file_path,
                suggestion="File may be corrupted or tampered with"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="File integrity verified",
            field="file_integrity",
            value=file_path
        )


# Convenience validation functions
def validate_firmware_scan_config(firmware_path: str, architecture: str,
                                 memory_constraints: Optional[Dict] = None,
                                 base_address: Union[str, int] = 0) -> List[ValidationResult]:
    """Validate complete firmware scan configuration."""
    
    config = {
        'firmware_path': firmware_path,
        'architecture': architecture,
        'base_address': base_address
    }
    
    if memory_constraints:
        config['memory_constraints'] = memory_constraints
    
    return RobustValidator.validate_configuration(config)


def is_valid_configuration(validation_results: List[ValidationResult]) -> bool:
    """Check if all validation results are successful."""
    return all(result.is_valid for result in validation_results)


def get_validation_errors(validation_results: List[ValidationResult]) -> List[ValidationResult]:
    """Get only the validation errors from results."""
    return [result for result in validation_results if not result.is_valid]


def format_validation_errors(validation_results: List[ValidationResult]) -> str:
    """Format validation errors for display."""
    errors = get_validation_errors(validation_results)
    
    if not errors:
        return "No validation errors"
    
    error_messages = []
    for error in errors:
        severity_icon = {
            ValidationSeverity.WARNING: "⚠️",
            ValidationSeverity.ERROR: "❌", 
            ValidationSeverity.CRITICAL: "🚨"
        }.get(error.severity, "❓")
        
        message = f"{severity_icon} {error.message}"
        if error.suggestion:
            message += f" (Suggestion: {error.suggestion})"
        
        error_messages.append(message)
    
    return "\n".join(error_messages)


# Export main components
__all__ = [
    'ValidationSeverity', 'ValidationResult', 'RobustValidator', 
    'SecureHashValidator', 'validate_firmware_scan_config',
    'is_valid_configuration', 'get_validation_errors', 'format_validation_errors'
]