"""Enhanced reporting module for PQC IoT Retrofit Scanner.

Provides comprehensive reporting capabilities:
- Multiple output formats (JSON, HTML, PDF, CSV)
- Executive summaries
- Technical details
- Risk assessments
- Compliance mapping
"""

import json
import csv
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Import scanner components
from .scanner import CryptoVulnerability, RiskLevel, CryptoAlgorithm
from .utils import format_size, format_address

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    TEXT = "text"
    EXECUTIVE = "executive"


@dataclass
class ReportMetadata:
    """Report metadata information."""
    tool_name: str = "PQC IoT Retrofit Scanner"
    tool_version: str = "1.0.0"
    report_format: str = "standard"
    generation_time: str = None
    scan_duration: float = 0.0
    firmware_path: str = ""
    architecture: str = ""
    
    def __post_init__(self):
        if self.generation_time is None:
            self.generation_time = datetime.datetime.now().isoformat()


class ReportGenerator:
    """Enhanced report generation with multiple formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_report(self, 
                       scan_results: Dict[str, Any],
                       vulnerabilities: List[CryptoVulnerability],
                       metadata: ReportMetadata,
                       format_type: ReportFormat = ReportFormat.JSON) -> str:
        """Generate comprehensive report in specified format."""
        
        try:
            if format_type == ReportFormat.JSON:
                return self._generate_json_report(scan_results, vulnerabilities, metadata)
            elif format_type == ReportFormat.CSV:
                return self._generate_csv_report(vulnerabilities, metadata)
            elif format_type == ReportFormat.HTML:
                return self._generate_html_report(scan_results, vulnerabilities, metadata)
            elif format_type == ReportFormat.TEXT:
                return self._generate_text_report(scan_results, vulnerabilities, metadata)
            elif format_type == ReportFormat.EXECUTIVE:
                return self._generate_executive_report(scan_results, vulnerabilities, metadata)
            else:
                raise ValueError(f"Unsupported report format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _generate_json_report(self, 
                             scan_results: Dict[str, Any],
                             vulnerabilities: List[CryptoVulnerability],
                             metadata: ReportMetadata) -> str:
        """Generate comprehensive JSON report."""
        
        report = {
            "metadata": asdict(metadata),
            "scan_summary": scan_results.get("scan_summary", {}),
            "vulnerability_analysis": self._analyze_vulnerabilities(vulnerabilities),
            "risk_assessment": self._generate_risk_assessment(vulnerabilities),
            "compliance_mapping": self._generate_compliance_mapping(vulnerabilities),
            "remediation_plan": self._generate_remediation_plan(vulnerabilities),
            "vulnerabilities": [self._serialize_vulnerability(v) for v in vulnerabilities],
            "recommendations": scan_results.get("recommendations", [])
        }
        
        return json.dumps(report, indent=2, default=str)
    
    def _generate_csv_report(self, 
                           vulnerabilities: List[CryptoVulnerability],
                           metadata: ReportMetadata) -> str:
        """Generate CSV report for spreadsheet analysis."""
        
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Function Name", "Algorithm", "Address", "Risk Level",
            "Key Size", "Description", "Mitigation", "Stack Usage",
            "Available Stack", "Business Criticality", "Exploitability Score"
        ])
        
        # Data rows
        for vuln in vulnerabilities:
            writer.writerow([
                vuln.function_name,
                vuln.algorithm.value,
                format_address(vuln.address),
                vuln.risk_level.value,
                vuln.key_size or "Unknown",
                vuln.description,
                vuln.mitigation,
                vuln.stack_usage,
                vuln.available_stack,
                getattr(vuln, 'business_criticality', 'medium'),
                getattr(vuln, 'exploitability_score', 5.0)
            ])
        
        return output.getvalue()
    
    def _generate_html_report(self, 
                            scan_results: Dict[str, Any],
                            vulnerabilities: List[CryptoVulnerability],
                            metadata: ReportMetadata) -> str:
        """Generate HTML report for web viewing."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PQC IoT Security Assessment Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .risk-critical {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .risk-high {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .risk-medium {{ background: #d1ecf1; border-left: 4px solid #17a2b8; }}
        .vulnerability {{ margin: 10px 0; padding: 15px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        .meta {{ font-size: 0.9em; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ PQC IoT Security Assessment Report</h1>
        <p>Firmware: {firmware_path}</p>
        <p>Architecture: {architecture}</p>
        <p>Generated: {generation_time}</p>
    </div>

    <div class="summary">
        <h2>📊 Executive Summary</h2>
        <p><strong>Total Vulnerabilities Found:</strong> {total_vulnerabilities}</p>
        <p><strong>Critical Risks:</strong> {critical_count}</p>
        <p><strong>High Risks:</strong> {high_count}</p>
        <p><strong>Scan Duration:</strong> {scan_duration:.2f} seconds</p>
    </div>

    <h2>🚨 Vulnerability Details</h2>
    {vulnerability_list}

    <div class="meta">
        <p>Report generated by {tool_name} v{tool_version}</p>
        <p>For questions or support, visit our documentation.</p>
    </div>
</body>
</html>
        """
        
        # Count risks
        risk_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for vuln in vulnerabilities:
            risk_counts[vuln.risk_level.value] += 1
        
        # Generate vulnerability HTML
        vuln_html = ""
        for vuln in vulnerabilities:
            risk_class = f"risk-{vuln.risk_level.value}"
            vuln_html += f"""
            <div class="vulnerability {risk_class}">
                <h3>{vuln.function_name} - {vuln.algorithm.value}</h3>
                <p><strong>Address:</strong> {format_address(vuln.address)}</p>
                <p><strong>Risk Level:</strong> {vuln.risk_level.value.upper()}</p>
                <p><strong>Description:</strong> {vuln.description}</p>
                <p><strong>Recommended Fix:</strong> {vuln.mitigation}</p>
            </div>
            """
        
        return html_template.format(
            firmware_path=metadata.firmware_path,
            architecture=metadata.architecture,
            generation_time=metadata.generation_time,
            tool_name=metadata.tool_name,
            tool_version=metadata.tool_version,
            total_vulnerabilities=len(vulnerabilities),
            critical_count=risk_counts["critical"],
            high_count=risk_counts["high"],
            scan_duration=metadata.scan_duration,
            vulnerability_list=vuln_html
        )
    
    def _generate_text_report(self, 
                            scan_results: Dict[str, Any],
                            vulnerabilities: List[CryptoVulnerability],
                            metadata: ReportMetadata) -> str:
        """Generate plain text report."""
        
        report_lines = [
            "=" * 80,
            f"PQC IoT SECURITY ASSESSMENT REPORT",
            "=" * 80,
            f"",
            f"Firmware: {metadata.firmware_path}",
            f"Architecture: {metadata.architecture}",
            f"Generated: {metadata.generation_time}",
            f"Tool: {metadata.tool_name} v{metadata.tool_version}",
            f"",
            f"EXECUTIVE SUMMARY",
            f"-" * 40,
            f"Total vulnerabilities found: {len(vulnerabilities)}",
        ]
        
        # Risk distribution
        risk_counts = {}
        for vuln in vulnerabilities:
            risk_level = vuln.risk_level.value
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        for risk, count in sorted(risk_counts.items()):
            report_lines.append(f"{risk.capitalize()} risks: {count}")
        
        report_lines.extend([
            f"",
            f"VULNERABILITY DETAILS",
            f"-" * 40
        ])
        
        # Vulnerability details
        for i, vuln in enumerate(vulnerabilities, 1):
            report_lines.extend([
                f"",
                f"{i}. {vuln.function_name}",
                f"   Algorithm: {vuln.algorithm.value}",
                f"   Address: {format_address(vuln.address)}",
                f"   Risk: {vuln.risk_level.value.upper()}",
                f"   Description: {vuln.description}",
                f"   Fix: {vuln.mitigation}",
            ])
        
        # Recommendations
        recommendations = scan_results.get("recommendations", [])
        if recommendations:
            report_lines.extend([
                f"",
                f"RECOMMENDATIONS",
                f"-" * 40
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        report_lines.extend([
            f"",
            f"=" * 80,
            f"End of Report"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_executive_report(self, 
                                 scan_results: Dict[str, Any],
                                 vulnerabilities: List[CryptoVulnerability],
                                 metadata: ReportMetadata) -> str:
        """Generate executive summary report."""
        
        # Calculate key metrics
        total_vulns = len(vulnerabilities)
        critical_vulns = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.CRITICAL)
        high_vulns = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.HIGH)
        
        # Business impact assessment
        business_impact = "High" if critical_vulns > 0 else "Medium" if high_vulns > 0 else "Low"
        
        # Compliance risks
        compliance_risks = []
        for vuln in vulnerabilities:
            if vuln.algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.MD5]:
                compliance_risks.append("FIPS 140-2 non-compliance")
            if vuln.algorithm.value.startswith("RSA"):
                compliance_risks.append("NIST PQC migration required")
        
        compliance_risks = list(set(compliance_risks))  # Remove duplicates
        
        executive_summary = f"""
EXECUTIVE SUMMARY - PQC IoT Security Assessment

OVERVIEW
========
Firmware: {Path(metadata.firmware_path).name}
Assessment Date: {metadata.generation_time[:10]}
Architecture: {metadata.architecture}

KEY FINDINGS
============
• Total security vulnerabilities identified: {total_vulns}
• Critical risk vulnerabilities: {critical_vulns}
• High risk vulnerabilities: {high_vulns}
• Estimated business impact: {business_impact}

RISK SUMMARY
============
The firmware contains quantum-vulnerable cryptographic implementations that 
pose significant security risks in a post-quantum computing environment.

COMPLIANCE IMPACT
=================
"""
        
        if compliance_risks:
            for risk in compliance_risks:
                executive_summary += f"• {risk}\n"
        else:
            executive_summary += "• No immediate compliance issues identified\n"
        
        executive_summary += f"""

RECOMMENDED ACTIONS
==================
1. Immediate: Address all critical risk vulnerabilities
2. Short-term: Migrate to post-quantum cryptographic algorithms
3. Long-term: Implement quantum-safe security architecture

MIGRATION TIMELINE
==================
• Critical vulnerabilities: Immediate action required
• High-risk algorithms: 6-12 months migration timeline
• Medium-risk items: 12-24 months for complete transition

This assessment was generated by {metadata.tool_name} v{metadata.tool_version}
For detailed technical analysis, refer to the complete technical report.
        """
        
        return executive_summary.strip()
    
    def _analyze_vulnerabilities(self, vulnerabilities: List[CryptoVulnerability]) -> Dict[str, Any]:
        """Analyze vulnerability patterns and trends."""
        
        analysis = {
            "total_count": len(vulnerabilities),
            "by_risk_level": {},
            "by_algorithm": {},
            "memory_impact": {
                "total_stack_usage": sum(v.stack_usage for v in vulnerabilities),
                "average_stack_usage": 0,
                "max_stack_usage": max((v.stack_usage for v in vulnerabilities), default=0)
            },
            "attack_surface": {
                "unique_algorithms": len(set(v.algorithm for v in vulnerabilities)),
                "unique_functions": len(set(v.function_name for v in vulnerabilities))
            }
        }
        
        # Risk level distribution
        for vuln in vulnerabilities:
            risk = vuln.risk_level.value
            analysis["by_risk_level"][risk] = analysis["by_risk_level"].get(risk, 0) + 1
            
            # Algorithm distribution
            alg = vuln.algorithm.value
            analysis["by_algorithm"][alg] = analysis["by_algorithm"].get(alg, 0) + 1
        
        # Average stack usage
        if vulnerabilities:
            analysis["memory_impact"]["average_stack_usage"] = (
                analysis["memory_impact"]["total_stack_usage"] / len(vulnerabilities)
            )
        
        return analysis
    
    def _generate_risk_assessment(self, vulnerabilities: List[CryptoVulnerability]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment."""
        
        risk_assessment = {
            "overall_risk_score": 0,
            "quantum_threat_timeline": {},
            "business_criticality": {
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "exploitability": {
                "average_score": 0,
                "max_score": 0,
                "distribution": {}
            }
        }
        
        if not vulnerabilities:
            return risk_assessment
        
        # Calculate overall risk score (0-10 scale)
        total_score = 0
        exploit_scores = []
        
        for vuln in vulnerabilities:
            # Risk scoring based on level
            if vuln.risk_level == RiskLevel.CRITICAL:
                total_score += 10
            elif vuln.risk_level == RiskLevel.HIGH:
                total_score += 7
            elif vuln.risk_level == RiskLevel.MEDIUM:
                total_score += 4
            else:
                total_score += 1
            
            # Business criticality
            criticality = getattr(vuln, 'business_criticality', 'medium')
            risk_assessment["business_criticality"][criticality] += 1
            
            # Exploitability scores
            exploit_score = getattr(vuln, 'exploitability_score', 5.0)
            exploit_scores.append(exploit_score)
            
            # Quantum threat timeline
            threat_years = getattr(vuln, 'quantum_threat_years', 15)
            timeline_key = f"{threat_years}_years"
            risk_assessment["quantum_threat_timeline"][timeline_key] = (
                risk_assessment["quantum_threat_timeline"].get(timeline_key, 0) + 1
            )
        
        # Overall risk score (normalized to 0-10)
        risk_assessment["overall_risk_score"] = min(10, total_score / len(vulnerabilities))
        
        # Exploitability metrics
        if exploit_scores:
            risk_assessment["exploitability"]["average_score"] = sum(exploit_scores) / len(exploit_scores)
            risk_assessment["exploitability"]["max_score"] = max(exploit_scores)
        
        return risk_assessment
    
    def _generate_compliance_mapping(self, vulnerabilities: List[CryptoVulnerability]) -> Dict[str, Any]:
        """Map vulnerabilities to compliance frameworks."""
        
        compliance_mapping = {
            "frameworks": {
                "NIST_SP_800_131A": {"violations": 0, "requirements": []},
                "FIPS_140_2": {"violations": 0, "requirements": []},
                "Common_Criteria": {"violations": 0, "requirements": []},
                "NIST_PQC": {"migration_required": 0, "algorithms": []}
            }
        }
        
        for vuln in vulnerabilities:
            algorithm = vuln.algorithm
            
            # NIST SP 800-131A (deprecated algorithms)
            if algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.MD5, CryptoAlgorithm.SHA1]:
                compliance_mapping["frameworks"]["NIST_SP_800_131A"]["violations"] += 1
                compliance_mapping["frameworks"]["NIST_SP_800_131A"]["requirements"].append(
                    f"Replace {algorithm.value} with approved algorithm"
                )
            
            # FIPS 140-2 
            if algorithm in [CryptoAlgorithm.DES, CryptoAlgorithm.RC4]:
                compliance_mapping["frameworks"]["FIPS_140_2"]["violations"] += 1
            
            # NIST PQC migration
            if algorithm.value.startswith("RSA") or algorithm.value.startswith("ECC"):
                compliance_mapping["frameworks"]["NIST_PQC"]["migration_required"] += 1
                compliance_mapping["frameworks"]["NIST_PQC"]["algorithms"].append(algorithm.value)
        
        return compliance_mapping
    
    def _generate_remediation_plan(self, vulnerabilities: List[CryptoVulnerability]) -> Dict[str, Any]:
        """Generate remediation plan with priorities and timelines."""
        
        remediation_plan = {
            "immediate_actions": [],
            "short_term_plan": [],
            "long_term_strategy": [],
            "estimated_effort": {
                "development_weeks": 0,
                "testing_weeks": 0,
                "deployment_weeks": 0
            }
        }
        
        critical_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.CRITICAL)
        high_count = sum(1 for v in vulnerabilities if v.risk_level == RiskLevel.HIGH)
        
        # Immediate actions for critical vulnerabilities
        if critical_count > 0:
            remediation_plan["immediate_actions"].append(
                f"Address {critical_count} critical vulnerabilities immediately"
            )
            remediation_plan["immediate_actions"].append(
                "Implement temporary mitigations for production systems"
            )
        
        # Short-term plan
        if high_count > 0:
            remediation_plan["short_term_plan"].append(
                f"Migrate {high_count} high-risk cryptographic implementations"
            )
        
        remediation_plan["short_term_plan"].extend([
            "Implement post-quantum cryptographic algorithms",
            "Update security architecture documentation",
            "Train development team on PQC implementations"
        ])
        
        # Long-term strategy
        remediation_plan["long_term_strategy"].extend([
            "Establish crypto-agility architecture",
            "Implement automated vulnerability scanning in CI/CD",
            "Regular security assessments and updates"
        ])
        
        # Effort estimation
        total_vulns = len(vulnerabilities)
        remediation_plan["estimated_effort"]["development_weeks"] = max(2, total_vulns // 2)
        remediation_plan["estimated_effort"]["testing_weeks"] = max(1, total_vulns // 4)
        remediation_plan["estimated_effort"]["deployment_weeks"] = max(1, total_vulns // 6)
        
        return remediation_plan
    
    def _serialize_vulnerability(self, vuln: CryptoVulnerability) -> Dict[str, Any]:
        """Serialize vulnerability to dictionary."""
        
        vuln_dict = {
            "algorithm": vuln.algorithm.value,
            "address": format_address(vuln.address),
            "function_name": vuln.function_name,
            "risk_level": vuln.risk_level.value,
            "key_size": vuln.key_size,
            "description": vuln.description,
            "mitigation": vuln.mitigation,
            "memory_impact": {
                "stack_usage": vuln.stack_usage,
                "available_stack": vuln.available_stack
            }
        }
        
        # Add optional fields if available
        for attr in ['business_criticality', 'compliance_impact', 'attack_vectors', 
                    'exploitability_score', 'threat_timeline', 'quantum_threat_years',
                    'performance_impact', 'migration_complexity']:
            if hasattr(vuln, attr):
                vuln_dict[attr] = getattr(vuln, attr)
        
        return vuln_dict
    
    def save_report(self, 
                   report_content: str,
                   output_path: str,
                   format_type: ReportFormat = ReportFormat.JSON):
        """Save report to file."""
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Set appropriate file extension
            if format_type == ReportFormat.JSON and not output_path.endswith('.json'):
                output_file = output_file.with_suffix('.json')
            elif format_type == ReportFormat.HTML and not output_path.endswith('.html'):
                output_file = output_file.with_suffix('.html')
            elif format_type == ReportFormat.CSV and not output_path.endswith('.csv'):
                output_file = output_file.with_suffix('.csv')
            elif format_type == ReportFormat.TEXT and not output_path.endswith('.txt'):
                output_file = output_file.with_suffix('.txt')
            
            output_file.write_text(report_content, encoding='utf-8')
            self.logger.info(f"Report saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            raise


# Convenience functions
def generate_json_report(scan_results: Dict[str, Any], 
                        vulnerabilities: List[CryptoVulnerability],
                        metadata: ReportMetadata) -> str:
    """Generate JSON report."""
    generator = ReportGenerator()
    return generator.generate_report(scan_results, vulnerabilities, metadata, ReportFormat.JSON)


def generate_html_report(scan_results: Dict[str, Any], 
                        vulnerabilities: List[CryptoVulnerability],
                        metadata: ReportMetadata) -> str:
    """Generate HTML report."""
    generator = ReportGenerator()
    return generator.generate_report(scan_results, vulnerabilities, metadata, ReportFormat.HTML)


def generate_executive_summary(scan_results: Dict[str, Any], 
                              vulnerabilities: List[CryptoVulnerability],
                              metadata: ReportMetadata) -> str:
    """Generate executive summary."""
    generator = ReportGenerator()
    return generator.generate_report(scan_results, vulnerabilities, metadata, ReportFormat.EXECUTIVE)


# Export classes and functions
__all__ = [
    'ReportFormat', 'ReportMetadata', 'ReportGenerator',
    'generate_json_report', 'generate_html_report', 'generate_executive_summary'
]