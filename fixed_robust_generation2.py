#!/usr/bin/env python3
"""Fixed Robust Generation 2 Firmware Analyzer.

Enhanced with enterprise-grade robustness:
- Comprehensive exception handling and recovery
- Input validation and sanitization  
- Structured logging with correlation IDs
- Health checks and system monitoring
- Graceful degradation when dependencies missing
- Security validation and threat detection
"""

import sys
import os
import uuid
import logging
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import hashlib
import struct

# Add source path
sys.path.insert(0, 'src')

from pqc_iot_retrofit.scanner import FirmwareScanner, CryptoVulnerability, RiskLevel


class AnalysisStatus(Enum):
    """Analysis execution status."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SECURITY_VIOLATION = "security_violation"
    INVALID_INPUT = "invalid_input"


@dataclass 
class HealthCheckResult:
    """System health check result."""
    status: str
    dependencies: Dict[str, bool] = field(default_factory=dict)
    capabilities: Dict[str, bool] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class RobustAnalysisResult:
    """Comprehensive analysis result with error tracking."""
    correlation_id: str
    status: AnalysisStatus
    firmware_path: str
    architecture: str
    vulnerabilities: List[CryptoVulnerability]
    risk_score: float
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class FixedRobustFirmwareAnalyzer:
    """Generation 2 analyzer with comprehensive error handling (fixed)."""
    
    def __init__(self, architecture: str, memory_constraints: Optional[Dict[str, int]] = None):
        self.correlation_id = str(uuid.uuid4())[:8]
        self.architecture = architecture
        self.memory_constraints = memory_constraints or {}
        
        # Initialize structured logging
        self._setup_logging()
        
        # Validate system health
        self.health_status = self._perform_health_check()
        
        # Initialize scanner with error handling
        try:
            self.scanner = FirmwareScanner(architecture, memory_constraints)
            self.logger.info("Scanner initialized successfully", 
                           extra={"architecture": architecture, "correlation_id": self.correlation_id})
        except Exception as e:
            self.logger.error(f"Scanner initialization failed: {e}", 
                            extra={"correlation_id": self.correlation_id})
            raise RuntimeError(f"Failed to initialize scanner: {e}")
    
    def _setup_logging(self):
        """Configure structured logging with correlation IDs."""
        self.logger = logging.getLogger(f"robust_analyzer_{self.correlation_id}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _perform_health_check(self) -> HealthCheckResult:
        """Comprehensive system health validation."""
        health = HealthCheckResult(status="healthy")
        
        # Check dependencies
        try:
            import capstone
            health.dependencies["capstone"] = True
            health.capabilities["disassembly"] = True
        except ImportError:
            health.dependencies["capstone"] = False
            health.capabilities["disassembly"] = False
            health.warnings.append("Capstone not available - disassembly disabled")
        
        try:
            import lief
            health.dependencies["lief"] = True
            health.capabilities["binary_analysis"] = True
        except ImportError:
            health.dependencies["lief"] = False
            health.capabilities["binary_analysis"] = False
            health.warnings.append("LIEF not available - advanced binary analysis disabled")
        
        # Check file system permissions
        try:
            temp_file = Path("/tmp/pqc_test_write")
            temp_file.write_text("test")
            temp_file.unlink()
            health.capabilities["file_operations"] = True
        except Exception as e:
            health.capabilities["file_operations"] = False
            health.errors.append(f"File system access error: {e}")
            health.status = "degraded"
        
        # Check memory availability
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.available < 100 * 1024 * 1024:  # < 100MB
                health.warnings.append("Low system memory - performance may be impacted")
                health.status = "degraded"
            health.capabilities["memory_monitoring"] = True
        except ImportError:
            health.capabilities["memory_monitoring"] = False
        
        self.logger.info(f"Health check completed: {health.status}", 
                        extra={"correlation_id": self.correlation_id, "health": health.__dict__})
        
        return health
    
    def analyze_firmware(self, firmware_path: str, **kwargs) -> Optional[RobustAnalysisResult]:
        """Perform robust firmware analysis with comprehensive error handling."""
        
        start_time = time.time()
        result = RobustAnalysisResult(
            correlation_id=self.correlation_id,
            status=AnalysisStatus.FAILED,
            firmware_path=firmware_path,
            architecture=self.architecture,
            vulnerabilities=[],
            risk_score=0.0,
            recommendations=[]
        )
        
        try:
            self.logger.info(f"Starting firmware analysis", 
                           extra={"firmware_path": firmware_path, "correlation_id": self.correlation_id})
            
            # Input validation with security checks
            if not self._validate_and_sanitize_input(firmware_path, result):
                return result
            
            # Load and validate firmware
            firmware_data = self._load_firmware_safely(firmware_path, result)
            if firmware_data is None:
                return result
            
            # Perform analysis with graceful degradation
            vulnerabilities = self._scan_with_fallbacks(firmware_path, result)
            
            # Calculate risk metrics
            risk_score = self._calculate_risk_score_robust(vulnerabilities, result)
            
            # Generate contextual recommendations
            recommendations = self._generate_recommendations_robust(vulnerabilities, result)
            
            # Update successful result
            result.vulnerabilities = vulnerabilities
            result.risk_score = risk_score
            result.recommendations = recommendations
            result.status = AnalysisStatus.SUCCESS if not result.warnings else AnalysisStatus.PARTIAL_SUCCESS
            result.metadata.update({
                "firmware_size": len(firmware_data),
                "firmware_hash": hashlib.sha256(firmware_data).hexdigest()[:16],
                "analysis_duration": time.time() - start_time,
                "health_status": self.health_status.status
            })
            
            self.logger.info("Analysis completed successfully", 
                           extra={"correlation_id": self.correlation_id, 
                                 "vulnerabilities_found": len(vulnerabilities),
                                 "risk_score": risk_score})
            
            return result
            
        except PermissionError as e:
            result.status = AnalysisStatus.SECURITY_VIOLATION
            result.errors.append(f"Permission denied: {e}")
            self.logger.error(f"Security violation: {e}", 
                            extra={"correlation_id": self.correlation_id})
            
        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.errors.append(f"Unexpected error: {e}")
            self.logger.error(f"Unexpected error: {e}", 
                            extra={"correlation_id": self.correlation_id, 
                                  "traceback": traceback.format_exc()})
        
        finally:
            result.performance_metrics["total_duration"] = time.time() - start_time
            
        return result
    
    def _validate_and_sanitize_input(self, firmware_path: str, result: RobustAnalysisResult) -> bool:
        """Comprehensive input validation and security checks."""
        
        try:
            # Path validation
            firmware_file = Path(firmware_path).resolve()
            
            # Security: Check for path traversal
            if ".." in str(firmware_file):
                current_dir = Path.cwd()
                try:
                    firmware_file.resolve().relative_to(current_dir)
                except ValueError:
                    result.errors.append("Path traversal detected - security violation")
                    return False
            
            # File existence and readability
            if not firmware_file.exists():
                result.errors.append(f"Firmware file not found: {firmware_path}")
                return False
            
            if not firmware_file.is_file():
                result.errors.append(f"Path is not a file: {firmware_path}")
                return False
            
            # File size validation
            file_size = firmware_file.stat().st_size
            if file_size == 0:
                result.errors.append("Firmware file is empty")
                return False
            
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                result.warnings.append(f"Large firmware file ({file_size / 1024 / 1024:.1f}MB) - analysis may be slow")
            
            # Architecture validation
            valid_architectures = [
                "cortex-m0", "cortex-m3", "cortex-m4", "cortex-m7",
                "esp32", "riscv32", "avr", "x86", "x86_64", "arm64"
            ]
            
            if self.architecture not in valid_architectures:
                result.warnings.append(f"Unrecognized architecture: {self.architecture}")
            
            return True
            
        except Exception as e:
            result.errors.append(f"Input validation error: {e}")
            return False
    
    def _load_firmware_safely(self, firmware_path: str, result: RobustAnalysisResult) -> Optional[bytes]:
        """Safely load firmware with memory limits and validation."""
        
        try:
            firmware_file = Path(firmware_path)
            
            # Read with size limit for safety
            max_size = 100 * 1024 * 1024  # 100MB limit
            file_size = firmware_file.stat().st_size
            
            if file_size > max_size:
                result.errors.append(f"Firmware too large: {file_size / 1024 / 1024:.1f}MB > 100MB limit")
                return None
            
            with open(firmware_file, 'rb') as f:
                firmware_data = f.read()
            
            # Basic file type validation
            if len(firmware_data) < 16:
                result.warnings.append("Very small firmware file - may not be valid")
            
            # Check for obvious text files (likely not firmware)
            try:
                sample = firmware_data[:1024].decode('utf-8')
                if sample.isprintable() and len(sample.strip()) > 100:
                    result.warnings.append("File appears to be text - may not be firmware binary")
            except UnicodeDecodeError:
                pass  # Good - likely binary
            
            return firmware_data
            
        except MemoryError:
            result.errors.append("Insufficient memory to load firmware")
            return None
        except IOError as e:
            result.errors.append(f"I/O error reading firmware: {e}")
            return None
        except Exception as e:
            result.errors.append(f"Unexpected error loading firmware: {e}")
            return None
    
    def _scan_with_fallbacks(self, firmware_path: str, result: RobustAnalysisResult) -> List[CryptoVulnerability]:
        """Scan with graceful degradation and fallback mechanisms."""
        
        vulnerabilities = []
        
        try:
            # Primary scanning with full capabilities
            if self.health_status.capabilities.get("disassembly", False):
                vulnerabilities = self.scanner.scan_firmware(firmware_path)
                result.metadata["scan_method"] = "full_disassembly"
            else:
                # Fallback: Pattern-based scanning
                result.warnings.append("Using pattern-based scanning (limited accuracy)")
                vulnerabilities = self._pattern_based_scan(firmware_path)
                result.metadata["scan_method"] = "pattern_based"
                
        except Exception as e:
            result.warnings.append(f"Primary scan failed: {e}")
            
            # Emergency fallback: Basic pattern matching
            try:
                vulnerabilities = self._basic_pattern_scan(firmware_path)
                result.metadata["scan_method"] = "basic_patterns"
                result.warnings.append("Using basic pattern matching - limited accuracy")
            except Exception as fallback_error:
                result.errors.append(f"All scanning methods failed: {fallback_error}")
        
        return vulnerabilities
    
    def _pattern_based_scan(self, firmware_path: str) -> List[CryptoVulnerability]:
        """Pattern-based vulnerability detection fallback."""
        
        firmware_data = Path(firmware_path).read_bytes()
        vulnerabilities = []
        
        # Define crypto signature patterns
        patterns = {
            b"RSA": ("RSA-2048", RiskLevel.CRITICAL),
            b"ECDSA": ("ECDSA-P256", RiskLevel.HIGH),
            b"ECDH": ("ECDH-P256", RiskLevel.HIGH),
            b"DH_": ("DH-2048", RiskLevel.HIGH),
        }
        
        for pattern, (algorithm, risk) in patterns.items():
            offset = 0
            while True:
                pos = firmware_data.find(pattern, offset)
                if pos == -1:
                    break
                
                # Create vulnerability using correct enum
                from pqc_iot_retrofit.scanner import CryptoAlgorithm
                
                # Map string to enum
                algo_mapping = {
                    "RSA-2048": CryptoAlgorithm.RSA_2048,
                    "ECDSA-P256": CryptoAlgorithm.ECDSA_P256,
                    "ECDH-P256": CryptoAlgorithm.ECDH_P256,
                    "DH-2048": CryptoAlgorithm.DH_2048,
                }
                
                crypto_algo = algo_mapping.get(algorithm, CryptoAlgorithm.RSA_2048)
                
                vuln = CryptoVulnerability(
                    algorithm=crypto_algo,
                    address=pos,
                    function_name=f"crypto_pattern_0x{pos:08x}",
                    risk_level=risk,
                    description=f"Detected {algorithm} pattern at offset {pos}",
                    mitigation=f"Replace {algorithm} with quantum-resistant alternative",
                    stack_usage=1024,  # Estimated
                    available_stack=self.memory_constraints.get("ram", 64*1024) // 4
                )
                vulnerabilities.append(vuln)
                offset = pos + 1
        
        return vulnerabilities
    
    def _basic_pattern_scan(self, firmware_path: str) -> List[CryptoVulnerability]:
        """Basic pattern scan as last resort."""
        
        firmware_data = Path(firmware_path).read_bytes()
        vulnerabilities = []
        
        # Ultra-basic: just look for common crypto strings
        basic_patterns = [b"RSA", b"ECC", b"ECDSA", b"DH"]
        
        for pattern in basic_patterns:
            if pattern in firmware_data:
                pos = firmware_data.find(pattern)
                
                from pqc_iot_retrofit.scanner import CryptoAlgorithm
                
                vuln = CryptoVulnerability(
                    algorithm=CryptoAlgorithm.RSA_2048,  # Default fallback
                    address=pos,
                    function_name=f"basic_pattern_0x{pos:08x}",
                    risk_level=RiskLevel.MEDIUM,
                    description=f"Basic pattern detection: {pattern.decode()}",
                    mitigation="Manual analysis recommended",
                    stack_usage=512,
                    available_stack=16*1024
                )
                vulnerabilities.append(vuln)
        
        return vulnerabilities
    
    def _calculate_risk_score_robust(self, vulnerabilities: List[CryptoVulnerability], 
                                   result: RobustAnalysisResult) -> float:
        """Calculate risk score with error handling."""
        
        try:
            if not vulnerabilities:
                return 0.0
            
            risk_weights = {
                RiskLevel.CRITICAL: 25,
                RiskLevel.HIGH: 15,
                RiskLevel.MEDIUM: 10,
                RiskLevel.LOW: 5
            }
            
            total_score = 0
            for vuln in vulnerabilities:
                weight = risk_weights.get(vuln.risk_level, 5)
                total_score += weight
            
            # Apply constraints penalty
            if self.memory_constraints:
                available_memory = sum(self.memory_constraints.values())
                if available_memory < 32 * 1024:  # Very constrained
                    total_score *= 1.2  # Higher risk for constrained devices
            
            final_score = min(total_score, 100.0)
            result.metadata["risk_calculation"] = {
                "base_score": total_score,
                "capped_score": final_score,
                "vulnerability_count": len(vulnerabilities)
            }
            
            return final_score
            
        except Exception as e:
            result.warnings.append(f"Risk calculation error: {e}")
            return 50.0  # Conservative default
    
    def _generate_recommendations_robust(self, vulnerabilities: List[CryptoVulnerability],
                                       result: RobustAnalysisResult) -> List[str]:
        """Generate robust recommendations with error handling."""
        
        try:
            if not vulnerabilities:
                return ["✅ No quantum vulnerabilities detected - firmware appears secure"]
            
            recommendations = []
            
            # Categorize vulnerabilities
            critical_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.CRITICAL)
            high_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.HIGH)
            
            # Critical issues
            if critical_count > 0:
                recommendations.append(f"🚨 URGENT: {critical_count} critical quantum vulnerabilities require immediate attention")
                recommendations.append("🔄 Begin quantum-resistant migration planning immediately")
            
            # High-priority issues  
            if high_count > 0:
                recommendations.append(f"⚠️ {high_count} high-risk vulnerabilities should be addressed within 30 days")
            
            # Algorithm-specific guidance
            algorithms = {str(v.algorithm.value) for v in vulnerabilities}
            
            if any("RSA" in alg for alg in algorithms):
                recommendations.append("🔐 Replace RSA signatures with Dilithium2/3 (NIST PQC standard)")
            
            if any("ECC" in alg or "ECDH" in alg or "ECDSA" in alg for alg in algorithms):
                recommendations.append("🔑 Replace ECC operations with Kyber KEM + Dilithium signatures")
            
            # Memory-aware recommendations
            if self.memory_constraints:
                total_memory = sum(self.memory_constraints.values())
                if total_memory < 64 * 1024:
                    recommendations.append("💾 Consider Kyber512/Dilithium2 for memory-constrained devices")
                elif total_memory < 256 * 1024:
                    recommendations.append("💾 Kyber768/Dilithium3 recommended for moderate memory constraints")
                else:
                    recommendations.append("💾 Full Kyber1024/Dilithium5 available with sufficient memory")
            
            # Implementation guidance
            recommendations.append("📊 Generate detailed patches with 'pqc-iot patch <scan_report>'")
            recommendations.append("🧪 Test patches in isolated environment before production deployment")
            
            # Health-based recommendations
            if not self.health_status.capabilities.get("disassembly", True):
                recommendations.append("⚙️ Install 'capstone' for enhanced analysis accuracy")
            
            return recommendations
            
        except Exception as e:
            result.warnings.append(f"Recommendation generation error: {e}")
            return ["❌ Error generating recommendations - manual analysis required"]


def demonstrate_fixed_robust_analysis():
    """Demonstrate Generation 2 robust error handling (fixed)."""
    
    print("=" * 70)
    print("🛡️ PQC IoT Retrofit Scanner - Generation 2 Fixed Robust Analysis")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "Valid STM32L4 Firmware",
            "arch": "cortex-m4",
            "constraints": {"flash": 512*1024, "ram": 128*1024},
            "create_file": True,
            "file_content": b"ARM_FIRMWARE" + b"RSA_SIGNATURE_2048" + b"ECDSA_P256" + b"\x00" * 500
        },
        {
            "name": "Non-existent File",
            "arch": "esp32",
            "constraints": {"flash": 4*1024*1024, "ram": 520*1024},
            "create_file": False,
            "firmware_path": "nonexistent_firmware.bin"
        },
        {
            "name": "Empty File",
            "arch": "cortex-m0",
            "constraints": {"flash": 32*1024, "ram": 8*1024},
            "create_file": True,
            "file_content": b""
        },
        {
            "name": "Text File Misidentified as Firmware",
            "arch": "riscv32",
            "constraints": {"flash": 128*1024, "ram": 32*1024},
            "create_file": True,
            "file_content": b"This is clearly a text file, not firmware binary data at all!"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        try:
            analyzer = FixedRobustFirmwareAnalyzer(
                architecture=test_case["arch"],
                memory_constraints=test_case["constraints"]
            )
            
            # Setup test file
            if test_case["create_file"]:
                test_file = Path(f"test_firmware_robust_{i}.bin")
                test_file.write_bytes(test_case["file_content"])
                firmware_path = str(test_file)
            else:
                firmware_path = test_case.get("firmware_path", "missing.bin")
                test_file = None
            
            # Perform analysis
            result = analyzer.analyze_firmware(firmware_path)
            
            if result:
                # Display results
                status_icon = {
                    AnalysisStatus.SUCCESS: "✅",
                    AnalysisStatus.PARTIAL_SUCCESS: "⚠️",
                    AnalysisStatus.FAILED: "❌",
                    AnalysisStatus.SECURITY_VIOLATION: "🚨",
                    AnalysisStatus.INVALID_INPUT: "📋"
                }.get(result.status, "❓")
                
                print(f"{status_icon} Status: {result.status.value}")
                print(f"🔍 Correlation ID: {result.correlation_id}")
                print(f"📊 Risk Score: {result.risk_score:.1f}/100")
                print(f"🔍 Vulnerabilities: {len(result.vulnerabilities)}")
                
                if result.warnings:
                    print("⚠️ Warnings:")
                    for warning in result.warnings:
                        print(f"  • {warning}")
                
                if result.errors:
                    print("❌ Errors:")
                    for error in result.errors:
                        print(f"  • {error}")
                
                if result.recommendations:
                    print("💡 Recommendations:")
                    for rec in result.recommendations[:3]:  # Show first 3
                        print(f"  • {rec}")
                    if len(result.recommendations) > 3:
                        print(f"  • ... and {len(result.recommendations) - 3} more")
                
                print(f"⏱️ Duration: {result.performance_metrics.get('total_duration', 0):.3f}s")
            else:
                print("❌ Analysis returned no result")
            
            # Cleanup
            if test_file and test_file.exists():
                test_file.unlink()
                
        except Exception as e:
            print(f"❌ Test case failed: {e}")
    
    print(f"\n🎉 Generation 2 robust error handling demonstration complete!")


if __name__ == "__main__":
    demonstrate_fixed_robust_analysis()