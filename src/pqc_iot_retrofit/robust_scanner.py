"""Robust scanner with enhanced error handling and security for Generation 2."""

import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .scanner import FirmwareScanner, CryptoVulnerability
from .security_enhanced import (
    create_secure_scanner_context, validate_firmware_securely,
    SecurityContext, InputSanitizer, security_logger, rate_limiter
)
from .error_handling import handle_errors, SecurityError, ValidationError


class RobustFirmwareScanner(FirmwareScanner):
    """Enhanced firmware scanner with Generation 2 security and robustness features."""
    
    def __init__(self, architecture: str, memory_constraints: Dict[str, int] = None, 
                 user_id: str = "anonymous"):
        """Initialize robust scanner with security context.
        
        Args:
            architecture: Target device architecture
            memory_constraints: Memory limitations
            user_id: User identifier for audit logging
        """
        
        # Sanitize inputs
        architecture = InputSanitizer.sanitize_architecture(architecture)
        if memory_constraints:
            memory_constraints = InputSanitizer.sanitize_memory_constraints(memory_constraints)
        
        # Initialize parent scanner
        super().__init__(architecture, memory_constraints)
        
        # Create security context
        self.security_context = create_secure_scanner_context(user_id)
        self.user_id = user_id
        
        # Enhanced logging
        self.security_logger = security_logger
        self.logger = logging.getLogger(__name__)
        
        # Scan statistics
        self.scan_stats = {
            'total_scans': 0,
            'successful_scans': 0,
            'failed_scans': 0,
            'vulnerabilities_total': 0,
            'avg_scan_time': 0.0
        }
        
        self.logger.info(f"RobustFirmwareScanner initialized for {architecture} with security context {self.security_context.session_id}")
    
    @handle_errors(operation_name="secure_firmware_scan", retry_count=2)
    def scan_firmware_securely(self, firmware_path: str, base_address: int = 0) -> List[CryptoVulnerability]:
        """Perform secure firmware scan with enhanced validation and error handling.
        
        Args:
            firmware_path: Path to firmware binary
            base_address: Base memory address
            
        Returns:
            List of detected vulnerabilities
            
        Raises:
            SecurityError: If security validation fails
            ValidationError: If input validation fails
        """
        
        scan_start_time = time.time()
        
        # Rate limiting check
        allowed, remaining = rate_limiter.check_rate_limit(self.security_context.session_id)
        if not allowed:
            self.security_logger.log_rate_limit_exceeded(
                self.security_context.session_id, 
                "scan_firmware_securely"
            )
            raise SecurityError("Rate limit exceeded for this session")
        
        # Update remaining rate limit
        self.security_context.rate_limit_remaining = remaining
        
        # Security logging
        self.security_logger.log_scan_start(
            self.security_context.session_id,
            firmware_path,
            self.user_id
        )
        
        try:
            # Comprehensive security validation
            security_validation = validate_firmware_securely(firmware_path, self.security_context)
            
            # Check for security flags
            if security_validation['security_flags']:
                self.security_logger.log_security_violation(
                    self.security_context.session_id,
                    "suspicious_firmware_content",
                    f"Flags: {security_validation['security_flags']}"
                )
                
                # Continue with warning for now, but log the issue
                self.logger.warning(f"Security flags detected: {security_validation['security_flags']}")
            
            # Sanitize base address
            base_address = InputSanitizer.sanitize_address(base_address)
            
            # Perform the actual scan using parent method
            self.logger.info(f"Starting secure scan of {firmware_path} (type: {security_validation['file_type']})")
            vulnerabilities = super().scan_firmware(firmware_path, base_address)
            
            # Enhanced vulnerability analysis
            vulnerabilities = self._enhance_vulnerability_analysis(vulnerabilities, security_validation)
            
            # Update statistics
            scan_duration = time.time() - scan_start_time
            self._update_scan_statistics(True, len(vulnerabilities), scan_duration)
            
            # Security logging
            self.security_logger.log_scan_complete(
                self.security_context.session_id,
                len(vulnerabilities),
                scan_duration
            )
            
            self.logger.info(f"Secure scan completed: {len(vulnerabilities)} vulnerabilities found in {scan_duration:.2f}s")
            
            return vulnerabilities
            
        except Exception as e:
            # Update failure statistics
            scan_duration = time.time() - scan_start_time
            self._update_scan_statistics(False, 0, scan_duration)
            
            # Enhanced error logging
            self.security_logger.log_security_violation(
                self.security_context.session_id,
                "scan_failure",
                f"Error: {str(e)}"
            )
            
            self.logger.error(f"Secure scan failed after {scan_duration:.2f}s: {e}")
            
            # Re-raise with context
            if isinstance(e, (SecurityError, ValidationError)):
                raise
            else:
                raise SecurityError(f"Scan failed due to unexpected error: {str(e)}")
    
    def _enhance_vulnerability_analysis(self, vulnerabilities: List[CryptoVulnerability], 
                                      security_validation: Dict[str, Any]) -> List[CryptoVulnerability]:
        """Enhance vulnerability analysis with security context."""
        
        enhanced_vulnerabilities = []
        
        for vuln in vulnerabilities:
            # Add security context to vulnerability description
            enhanced_description = vuln.description
            
            # Add file type context
            file_type = security_validation.get('file_type', 'Unknown')
            enhanced_description += f" (detected in {file_type} firmware)"
            
            # Add integrity information
            integrity_hash = security_validation.get('integrity_hash', '')[:16]
            enhanced_description += f" [integrity: {integrity_hash}...]"
            
            # Create enhanced vulnerability
            enhanced_vuln = CryptoVulnerability(
                algorithm=vuln.algorithm,
                address=vuln.address,
                function_name=vuln.function_name,
                risk_level=vuln.risk_level,
                key_size=vuln.key_size,
                description=enhanced_description,
                mitigation=vuln.mitigation,
                stack_usage=vuln.stack_usage,
                available_stack=vuln.available_stack
            )
            
            enhanced_vulnerabilities.append(enhanced_vuln)
        
        return enhanced_vulnerabilities
    
    def _update_scan_statistics(self, success: bool, vuln_count: int, duration: float):
        """Update internal scan statistics."""
        
        self.scan_stats['total_scans'] += 1
        
        if success:
            self.scan_stats['successful_scans'] += 1
            self.scan_stats['vulnerabilities_total'] += vuln_count
        else:
            self.scan_stats['failed_scans'] += 1
        
        # Update average scan time
        total_scans = self.scan_stats['total_scans']
        current_avg = self.scan_stats['avg_scan_time']
        self.scan_stats['avg_scan_time'] = ((current_avg * (total_scans - 1)) + duration) / total_scans
    
    def generate_enhanced_report(self) -> Dict[str, Any]:
        """Generate enhanced report with security and performance metrics."""
        
        # Get base report
        base_report = super().generate_report()
        
        # Add security context
        security_context = {
            'session_id': self.security_context.session_id,
            'user_id': self.user_id,
            'security_level': 'enhanced',
            'rate_limit_remaining': self.security_context.rate_limit_remaining,
            'timestamp': self.security_context.timestamp
        }
        
        # Add performance statistics
        performance_stats = self.scan_stats.copy()
        
        # Calculate additional metrics
        if self.scan_stats['total_scans'] > 0:
            success_rate = (self.scan_stats['successful_scans'] / self.scan_stats['total_scans']) * 100
            avg_vulns_per_scan = (self.scan_stats['vulnerabilities_total'] / 
                                max(1, self.scan_stats['successful_scans']))
        else:
            success_rate = 0.0
            avg_vulns_per_scan = 0.0
        
        performance_stats.update({
            'success_rate_percent': success_rate,
            'avg_vulnerabilities_per_scan': avg_vulns_per_scan
        })
        
        # Enhanced report
        enhanced_report = base_report.copy()
        enhanced_report.update({
            'security_context': security_context,
            'performance_statistics': performance_stats,
            'generation': 2,
            'features': [
                'secure_file_validation',
                'rate_limiting',
                'audit_logging',
                'input_sanitization',
                'enhanced_error_handling'
            ]
        })
        
        return enhanced_report
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security-focused summary of scanner state."""
        
        return {
            'session_id': self.security_context.session_id,
            'user_id': self.user_id,
            'architecture': self.architecture,
            'rate_limit_remaining': self.security_context.rate_limit_remaining,
            'scan_statistics': self.scan_stats,
            'security_features': {
                'file_validation': True,
                'rate_limiting': True,
                'audit_logging': True,
                'input_sanitization': True,
                'error_handling': True
            }
        }
    
    def validate_scan_prerequisites(self, firmware_path: str) -> Dict[str, Any]:
        """Validate all prerequisites before performing scan."""
        
        try:
            # Security validation
            security_validation = validate_firmware_securely(firmware_path, self.security_context)
            
            # Rate limit check
            allowed, remaining = rate_limiter.check_rate_limit(self.security_context.session_id)
            
            # Architecture compatibility check
            file_type = security_validation.get('file_type', 'Unknown')
            compatibility_warnings = []
            
            if self.architecture.startswith('cortex-m') and file_type != 'ARM Cortex-M Binary':
                compatibility_warnings.append(f"File type '{file_type}' may not be compatible with {self.architecture}")
            
            return {
                'prerequisites_met': allowed and not security_validation['security_flags'],
                'security_validation': security_validation,
                'rate_limit_allowed': allowed,
                'rate_limit_remaining': remaining,
                'compatibility_warnings': compatibility_warnings,
                'recommendations': self._generate_prerequisite_recommendations(
                    security_validation, allowed, compatibility_warnings
                )
            }
            
        except Exception as e:
            return {
                'prerequisites_met': False,
                'error': str(e),
                'recommendations': [
                    "Fix the reported error before attempting to scan",
                    "Ensure firmware file is accessible and valid",
                    "Check file permissions and format"
                ]
            }
    
    def _generate_prerequisite_recommendations(self, security_validation: Dict[str, Any], 
                                            rate_allowed: bool, 
                                            compatibility_warnings: List[str]) -> List[str]:
        """Generate recommendations based on prerequisite validation."""
        
        recommendations = []
        
        if not rate_allowed:
            recommendations.append("Wait before scanning again due to rate limiting")
        
        if security_validation['security_flags']:
            recommendations.append("Review security flags before proceeding with scan")
        
        if compatibility_warnings:
            recommendations.extend([
                f"Warning: {warning}" for warning in compatibility_warnings
            ])
        
        if not recommendations:
            recommendations.append("All prerequisites met - ready to scan")
        
        return recommendations


# Convenience function for creating robust scanners
def create_robust_scanner(architecture: str, memory_constraints: Dict[str, int] = None,
                         user_id: str = "anonymous") -> RobustFirmwareScanner:
    """Create a robust firmware scanner with Generation 2 security features.
    
    Args:
        architecture: Target device architecture
        memory_constraints: Memory limitations
        user_id: User identifier for audit logging
        
    Returns:
        RobustFirmwareScanner instance
    """
    return RobustFirmwareScanner(architecture, memory_constraints, user_id)