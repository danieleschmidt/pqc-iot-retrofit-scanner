#!/usr/bin/env python3
"""
Simple Firmware Analyzer - Generation 1 Implementation
Minimal viable PQC vulnerability scanner for IoT firmware
"""

import os
import sys
import hashlib
import struct
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class CryptoVulnerability:
    """Detected cryptographic vulnerability."""
    algorithm: str
    risk_level: str
    location: int
    key_size: Optional[int] = None
    function_name: Optional[str] = None
    recommendation: str = ""


@dataclass
class FirmwareAnalysisResult:
    """Complete firmware analysis result."""
    filename: str
    file_size: int
    file_hash: str
    architecture: str
    vulnerabilities: List[CryptoVulnerability]
    analysis_time: float
    pqc_recommendation: str


class SimpleFirmwareAnalyzer:
    """Minimal firmware analyzer for quantum vulnerabilities."""
    
    # Cryptographic patterns to detect
    CRYPTO_PATTERNS = {
        'rsa_1024': {
            'patterns': [b'\x82\x01\x00', b'\x30\x82\x01\x0a'],  # ASN.1 RSA-1024 patterns
            'risk': 'critical',
            'algorithm': 'RSA-1024',
            'pqc_replacement': 'Dilithium2'
        },
        'rsa_2048': {
            'patterns': [b'\x82\x02\x00', b'\x30\x82\x02\x0a'],  # ASN.1 RSA-2048 patterns
            'risk': 'high',
            'algorithm': 'RSA-2048', 
            'pqc_replacement': 'Dilithium3'
        },
        'ecdsa_p256': {
            'patterns': [b'\x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07'],  # NIST P-256 OID
            'risk': 'high',
            'algorithm': 'ECDSA-P256',
            'pqc_replacement': 'Dilithium2'
        },
        'ecdh_p256': {
            'patterns': [b'\x30\x59\x30\x13\x06\x07'],  # ECDH P-256 patterns
            'risk': 'high', 
            'algorithm': 'ECDH-P256',
            'pqc_replacement': 'Kyber512'
        }
    }
    
    # Architecture detection patterns
    ARCH_PATTERNS = {
        'arm_cortexm': {
            'patterns': [b'\x08\x00\x00\x20', b'\x00\x00\x00\x08'],  # ARM Cortex-M vector table
            'name': 'ARM Cortex-M'
        },
        'esp32': {
            'patterns': [b'\xe9\x00', b'\x40\x00\x10\x00'],  # ESP32 bootloader patterns
            'name': 'ESP32'
        },
        'avr': {
            'patterns': [b'\x0c\x94', b'\x95\x08'],  # AVR instruction patterns
            'name': 'AVR'
        },
        'risc_v': {
            'patterns': [b'\x97\x02', b'\x13\x01'],  # RISC-V instruction patterns
            'name': 'RISC-V'
        }
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self.stats = {
            'files_analyzed': 0,
            'vulnerabilities_found': 0,
            'critical_issues': 0
        }
    
    def detect_architecture(self, firmware_data: bytes) -> str:
        """Detect firmware architecture from binary patterns."""
        for arch_name, arch_info in self.ARCH_PATTERNS.items():
            for pattern in arch_info['patterns']:
                if pattern in firmware_data[:1024]:  # Check first 1KB
                    return arch_info['name']
        return 'Unknown'
    
    def scan_crypto_patterns(self, firmware_data: bytes) -> List[CryptoVulnerability]:
        """Scan firmware for cryptographic patterns."""
        vulnerabilities = []
        
        for crypto_name, crypto_info in self.CRYPTO_PATTERNS.items():
            for pattern in crypto_info['patterns']:
                offset = 0
                while True:
                    pos = firmware_data.find(pattern, offset)
                    if pos == -1:
                        break
                    
                    vuln = CryptoVulnerability(
                        algorithm=crypto_info['algorithm'],
                        risk_level=crypto_info['risk'],
                        location=pos,
                        function_name=f"crypto_function_at_0x{pos:x}",
                        recommendation=f"Replace with {crypto_info['pqc_replacement']}"
                    )
                    vulnerabilities.append(vuln)
                    self.stats['vulnerabilities_found'] += 1
                    
                    if crypto_info['risk'] == 'critical':
                        self.stats['critical_issues'] += 1
                    
                    offset = pos + 1
        
        return vulnerabilities
    
    def analyze_firmware(self, firmware_path: str) -> FirmwareAnalysisResult:
        """Analyze a firmware file for quantum vulnerabilities."""
        import time
        start_time = time.time()
        
        if not os.path.exists(firmware_path):
            raise FileNotFoundError(f"Firmware file not found: {firmware_path}")
        
        # Read firmware data
        with open(firmware_path, 'rb') as f:
            firmware_data = f.read()
        
        # Basic file analysis
        file_size = len(firmware_data)
        file_hash = hashlib.sha256(firmware_data).hexdigest()
        
        # Detect architecture
        architecture = self.detect_architecture(firmware_data)
        
        # Scan for crypto vulnerabilities
        vulnerabilities = self.scan_crypto_patterns(firmware_data)
        
        # Generate PQC recommendation
        pqc_recommendation = self._generate_pqc_recommendation(vulnerabilities, architecture)
        
        analysis_time = time.time() - start_time
        self.stats['files_analyzed'] += 1
        
        return FirmwareAnalysisResult(
            filename=os.path.basename(firmware_path),
            file_size=file_size,
            file_hash=file_hash,
            architecture=architecture,
            vulnerabilities=vulnerabilities,
            analysis_time=analysis_time,
            pqc_recommendation=pqc_recommendation
        )
    
    def _generate_pqc_recommendation(self, vulnerabilities: List[CryptoVulnerability], 
                                   architecture: str) -> str:
        """Generate post-quantum cryptography recommendations."""
        if not vulnerabilities:
            return "No quantum-vulnerable cryptography detected. Consider proactive PQC adoption."
        
        critical_count = sum(1 for v in vulnerabilities if v.risk_level == 'critical')
        high_count = sum(1 for v in vulnerabilities if v.risk_level == 'high')
        
        recommendations = []
        
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} critical vulnerabilities require immediate PQC migration")
        
        if high_count > 0:
            recommendations.append(f"HIGH PRIORITY: {high_count} high-risk algorithms need PQC replacement")
        
        # Architecture-specific recommendations
        arch_recommendations = {
            'ARM Cortex-M': "Recommended: Dilithium2 + Kyber512 (optimized for constrained devices)",
            'ESP32': "Recommended: Dilithium3 + Kyber768 (leverage hardware acceleration)",
            'RISC-V': "Recommended: Dilithium2 + Kyber512 (minimal memory footprint)",
            'AVR': "Recommended: Hybrid mode with gradual PQC migration"
        }
        
        if architecture in arch_recommendations:
            recommendations.append(arch_recommendations[architecture])
        
        return "; ".join(recommendations)
    
    def generate_report(self, results: List[FirmwareAnalysisResult], 
                       output_format: str = 'json') -> str:
        """Generate analysis report in specified format."""
        if output_format == 'json':
            return self._generate_json_report(results)
        elif output_format == 'text':
            return self._generate_text_report(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_json_report(self, results: List[FirmwareAnalysisResult]) -> str:
        """Generate JSON report."""
        report = {
            'scan_summary': {
                'files_analyzed': len(results),
                'total_vulnerabilities': sum(len(r.vulnerabilities) for r in results),
                'critical_issues': sum(1 for r in results for v in r.vulnerabilities 
                                     if v.risk_level == 'critical'),
                'scan_timestamp': None  # Would add timestamp in real implementation
            },
            'results': [asdict(result) for result in results],
            'analyzer_stats': self.stats
        }
        return json.dumps(report, indent=2)
    
    def _generate_text_report(self, results: List[FirmwareAnalysisResult]) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("PQC IoT Retrofit Scanner - Analysis Report")
        lines.append("=" * 50)
        lines.append("")
        
        # Summary
        total_vulns = sum(len(r.vulnerabilities) for r in results)
        critical_vulns = sum(1 for r in results for v in r.vulnerabilities 
                           if v.risk_level == 'critical')
        
        lines.append(f"Files Analyzed: {len(results)}")
        lines.append(f"Total Vulnerabilities: {total_vulns}")
        lines.append(f"Critical Issues: {critical_vulns}")
        lines.append("")
        
        # Detailed results
        for result in results:
            lines.append(f"File: {result.filename}")
            lines.append(f"  Architecture: {result.architecture}")
            lines.append(f"  Size: {result.file_size:,} bytes")
            lines.append(f"  Vulnerabilities: {len(result.vulnerabilities)}")
            
            for vuln in result.vulnerabilities:
                lines.append(f"    - {vuln.algorithm} at 0x{vuln.location:x} ({vuln.risk_level} risk)")
            
            lines.append(f"  Recommendation: {result.pqc_recommendation}")
            lines.append("")
        
        return "\n".join(lines)


def main():
    """CLI entry point for simple firmware analyzer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple PQC vulnerability scanner for IoT firmware"
    )
    parser.add_argument('firmware_files', nargs='+', help='Firmware files to analyze')
    parser.add_argument('--output', '-o', help='Output file for report')
    parser.add_argument('--format', '-f', choices=['json', 'text'], 
                       default='text', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    analyzer = SimpleFirmwareAnalyzer()
    results = []
    
    print("PQC IoT Retrofit Scanner - Simple Analysis Mode")
    print("=" * 50)
    
    for firmware_file in args.firmware_files:
        if args.verbose:
            print(f"Analyzing: {firmware_file}")
        
        try:
            result = analyzer.analyze_firmware(firmware_file)
            results.append(result)
            
            if args.verbose:
                print(f"  Found {len(result.vulnerabilities)} vulnerabilities")
                print(f"  Architecture: {result.architecture}")
        
        except Exception as e:
            print(f"Error analyzing {firmware_file}: {e}")
            continue
    
    # Generate report
    report = analyzer.generate_report(results, args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print("\n" + report)
    
    # Summary statistics
    total_vulns = sum(len(r.vulnerabilities) for r in results)
    critical_vulns = sum(1 for r in results for v in r.vulnerabilities 
                        if v.risk_level == 'critical')
    
    print(f"\nScan Complete:")
    print(f"  Files: {len(results)}")
    print(f"  Vulnerabilities: {total_vulns}")
    print(f"  Critical Issues: {critical_vulns}")
    
    if critical_vulns > 0:
        print("\n⚠️  CRITICAL: Immediate PQC migration recommended!")
        return 1
    elif total_vulns > 0:
        print("\n⚠️  WARNING: PQC migration planning recommended")
        return 0
    else:
        print("\n✅ No quantum vulnerabilities detected")
        return 0


if __name__ == '__main__':
    sys.exit(main())