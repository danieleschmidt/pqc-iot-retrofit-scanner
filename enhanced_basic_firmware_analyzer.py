#!/usr/bin/env python3
"""Enhanced Basic Firmware Analyzer - Generation 1 Implementation.

Demonstrates working basic functionality with essential improvements:
- Firmware binary analysis with pattern matching
- Crypto vulnerability detection  
- Basic PQC patch recommendation
- Memory constraint validation
- Multi-architecture support
"""

import sys
import struct
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Add source path for imports
sys.path.insert(0, 'src')

from pqc_iot_retrofit.scanner import (
    FirmwareScanner, CryptoAlgorithm, RiskLevel, CryptoVulnerability
)


@dataclass
class FirmwareAnalysisResult:
    """Comprehensive firmware analysis result."""
    firmware_path: str
    architecture: str
    file_size: int
    file_hash: str
    vulnerabilities: List[CryptoVulnerability]
    memory_constraints: Dict[str, int]
    risk_score: float
    recommendations: List[str]


class BasicFirmwareAnalyzer:
    """Enhanced basic firmware analyzer with essential features."""
    
    def __init__(self, architecture: str, memory_constraints: Optional[Dict[str, int]] = None):
        self.architecture = architecture
        self.memory_constraints = memory_constraints or {}
        self.scanner = FirmwareScanner(architecture, memory_constraints)
        
    def analyze_firmware(self, firmware_path: str) -> FirmwareAnalysisResult:
        """Perform comprehensive firmware analysis."""
        
        firmware_file = Path(firmware_path)
        if not firmware_file.exists():
            raise FileNotFoundError(f"Firmware file not found: {firmware_path}")
        
        # Read firmware binary
        firmware_data = firmware_file.read_bytes()
        file_hash = hashlib.sha256(firmware_data).hexdigest()
        
        print(f"🔍 Analyzing {firmware_file.name} ({len(firmware_data):,} bytes)")
        print(f"📋 Architecture: {self.architecture}")
        print(f"🔐 SHA256: {file_hash[:16]}...")
        
        # Scan for crypto vulnerabilities
        vulnerabilities = self.scanner.scan_firmware(firmware_path)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(vulnerabilities)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(vulnerabilities)
        
        return FirmwareAnalysisResult(
            firmware_path=str(firmware_file),
            architecture=self.architecture,
            file_size=len(firmware_data),
            file_hash=file_hash,
            vulnerabilities=vulnerabilities,
            memory_constraints=self.memory_constraints,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _calculate_risk_score(self, vulnerabilities: List[CryptoVulnerability]) -> float:
        """Calculate overall risk score (0-100)."""
        if not vulnerabilities:
            return 0.0
        
        risk_weights = {
            RiskLevel.CRITICAL: 25,
            RiskLevel.HIGH: 15,
            RiskLevel.MEDIUM: 10,
            RiskLevel.LOW: 5
        }
        
        total_score = sum(risk_weights.get(vuln.risk_level, 0) for vuln in vulnerabilities)
        return min(total_score, 100.0)
    
    def _generate_recommendations(self, vulnerabilities: List[CryptoVulnerability]) -> List[str]:
        """Generate actionable recommendations."""
        if not vulnerabilities:
            return ["✅ No quantum vulnerabilities detected - firmware appears secure"]
        
        recommendations = []
        critical_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.HIGH)
        
        if critical_count > 0:
            recommendations.append(f"🚨 URGENT: Address {critical_count} critical vulnerabilities immediately")
            recommendations.append("🔄 Consider immediate quantum-resistant algorithm migration")
        
        if high_count > 0:
            recommendations.append(f"⚠️ Prioritize {high_count} high-risk vulnerabilities")
        
        # Algorithm-specific recommendations
        algorithms = {v.algorithm for v in vulnerabilities}
        if CryptoAlgorithm.RSA_1024 in algorithms or CryptoAlgorithm.RSA_2048 in algorithms:
            recommendations.append("🔐 Replace RSA signatures with Dilithium2/3")
        
        if CryptoAlgorithm.ECDH_P256 in algorithms or CryptoAlgorithm.ECDSA_P256 in algorithms:
            recommendations.append("🔑 Replace ECDH/ECDSA with Kyber512 + Dilithium2")
        
        # Memory-aware recommendations
        total_memory = sum(self.memory_constraints.values())
        if total_memory > 0 and total_memory < 64 * 1024:  # < 64KB
            recommendations.append("💾 Consider lightweight PQC variants for memory-constrained devices")
        
        recommendations.append("📊 Run detailed analysis with --generate-patches for specific solutions")
        
        return recommendations


def demonstrate_basic_analysis():
    """Demonstrate enhanced basic firmware analysis functionality."""
    
    print("=" * 60)
    print("🚀 PQC IoT Retrofit Scanner - Generation 1 Demo")
    print("=" * 60)
    
    # Test with different architectures
    test_cases = [
        {
            "arch": "cortex-m4",
            "constraints": {"flash": 512*1024, "ram": 128*1024},
            "description": "STM32L4 Smart Meter"
        },
        {
            "arch": "esp32", 
            "constraints": {"flash": 4*1024*1024, "ram": 520*1024},
            "description": "ESP32 IoT Sensor"
        },
        {
            "arch": "cortex-m0",
            "constraints": {"flash": 64*1024, "ram": 16*1024}, 
            "description": "Constrained MCU Device"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['description']}")
        print("-" * 40)
        
        try:
            analyzer = BasicFirmwareAnalyzer(
                architecture=test_case["arch"],
                memory_constraints=test_case["constraints"]
            )
            
            # Create synthetic firmware for testing
            synthetic_firmware = create_synthetic_firmware(test_case["arch"])
            test_file = Path(f"test_firmware_{test_case['arch']}.bin")
            test_file.write_bytes(synthetic_firmware)
            
            # Analyze firmware
            result = analyzer.analyze_firmware(str(test_file))
            
            # Display results
            print(f"📊 Risk Score: {result.risk_score:.1f}/100")
            print(f"🔍 Vulnerabilities Found: {len(result.vulnerabilities)}")
            
            if result.vulnerabilities:
                for vuln in result.vulnerabilities:
                    risk_icon = "🚨" if vuln.risk_level == RiskLevel.CRITICAL else "⚠️"
                    print(f"  {risk_icon} {vuln.algorithm.value} at {vuln.function_name}")
            
            print("\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
                
            # Cleanup
            test_file.unlink()
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    print(f"\n🎉 Generation 1 basic functionality demonstration complete!")


def create_synthetic_firmware(architecture: str) -> bytes:
    """Create synthetic firmware with crypto patterns for testing."""
    
    # Base firmware structure
    firmware = bytearray(1024)  # 1KB firmware
    
    # Add architecture-specific patterns
    if architecture == "cortex-m4":
        # ARM Cortex-M4 patterns
        firmware[0:4] = struct.pack("<I", 0x20020000)  # Stack pointer
        firmware[4:8] = struct.pack("<I", 0x08000009)  # Reset vector (thumb mode)
        
        # Simulate RSA-2048 signature pattern
        firmware[100:116] = b"RSA_SIGNATURE_2048_HERE"  # RSA pattern
        firmware[200:220] = b"ECDSA_P256_VALIDATION"     # ECDSA pattern
        
    elif architecture == "esp32":
        # ESP32 firmware header pattern  
        firmware[0:4] = b"\xe9"  # ESP32 image header magic
        firmware[4:8] = struct.pack("<I", 0x40080000)  # Entry point
        
        # ESP32 crypto patterns
        firmware[150:170] = b"ESP_ECDH_P256_EXCHANGE"
        firmware[250:270] = b"MBEDTLS_RSA_VERIFY_2048"
        
    elif architecture == "cortex-m0":
        # Simple ARM Cortex-M0 
        firmware[0:4] = struct.pack("<I", 0x20004000)  # Smaller stack
        firmware[4:8] = struct.pack("<I", 0x08000009)  # Reset vector
        
        # Limited crypto (due to constraints)
        firmware[80:100] = b"SIMPLE_ECC_P256_AUTH"
    
    return bytes(firmware)


if __name__ == "__main__":
    demonstrate_basic_analysis()