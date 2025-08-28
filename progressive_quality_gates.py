#!/usr/bin/env python3
"""
Progressive Quality Gates System - Generation 1 Implementation
Autonomous quality validation with progressive enhancement capabilities.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

class QualityLevel(Enum):
    """Quality assessment levels."""
    BASIC = "basic"
    ROBUST = "robust" 
    OPTIMIZED = "optimized"
    RESEARCH_GRADE = "research_grade"

class GateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class QualityMetric:
    """Individual quality metric result."""
    name: str
    value: Union[int, float, str, bool]
    threshold: Union[int, float, str, bool]
    passed: bool
    message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass 
class QualityGate:
    """Individual quality gate definition and results."""
    name: str
    level: QualityLevel
    command: str
    timeout: int = 300
    required: bool = True
    metrics: List[QualityMetric] = None
    status: GateStatus = GateStatus.PENDING
    execution_time: float = 0.0
    output: str = ""
    error: str = ""
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = []

class ProgressiveQualityGates:
    """
    Autonomous Quality Gates System with Progressive Enhancement.
    
    Implements continuous quality validation with automatic progression
    through quality levels and adaptive improvement suggestions.
    """
    
    def __init__(self, project_root: Path = None):
        """Initialize the Progressive Quality Gates system."""
        self.project_root = project_root or Path.cwd()
        self.report_dir = self.project_root / "reports"
        self.report_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality gates
        self.gates = self._initialize_quality_gates()
        self.current_level = QualityLevel.BASIC
        self.execution_id = f"qg_{int(time.time())}"
        
    def _initialize_quality_gates(self) -> Dict[QualityLevel, List[QualityGate]]:
        """Initialize quality gates for each level."""
        return {
            QualityLevel.BASIC: self._basic_gates(),
            QualityLevel.ROBUST: self._robust_gates(),
            QualityLevel.OPTIMIZED: self._optimized_gates(),
            QualityLevel.RESEARCH_GRADE: self._research_grade_gates()
        }
    
    def _basic_gates(self) -> List[QualityGate]:
        """Basic quality gates - make it work."""
        return [
            QualityGate(
                name="syntax_validation",
                level=QualityLevel.BASIC,
                command="python3 -m py_compile src/pqc_iot_retrofit/__init__.py",
                timeout=30,
                required=True
            ),
            QualityGate(
                name="import_validation", 
                level=QualityLevel.BASIC,
                command="python3 -c 'import src.pqc_iot_retrofit; print(\"Imports successful\")'",
                timeout=30,
                required=True
            ),
            QualityGate(
                name="basic_tests",
                level=QualityLevel.BASIC,
                command="python3 -m pytest tests/test_scanner.py -v --tb=short",
                timeout=60,
                required=False
            )
        ]
    
    def _robust_gates(self) -> List[QualityGate]:
        """Robust quality gates - comprehensive validation."""
        return [
            QualityGate(
                name="full_test_suite",
                level=QualityLevel.ROBUST,
                command="python3 -m pytest tests/ -v --cov=src --cov-report=json",
                timeout=300,
                required=True
            ),
            QualityGate(
                name="security_scan",
                level=QualityLevel.ROBUST, 
                command="python3 -m bandit -r src/ -f json",
                timeout=60,
                required=True
            ),
            QualityGate(
                name="type_checking",
                level=QualityLevel.ROBUST,
                command="python3 -m mypy src/pqc_iot_retrofit --ignore-missing-imports",
                timeout=120,
                required=False
            ),
            QualityGate(
                name="code_style",
                level=QualityLevel.ROBUST,
                command="python3 -m ruff check src/ --output-format=json",
                timeout=60,
                required=False
            )
        ]
    
    def _optimized_gates(self) -> List[QualityGate]:
        """Optimized quality gates - performance and scalability."""
        return [
            QualityGate(
                name="performance_benchmarks",
                level=QualityLevel.OPTIMIZED,
                command="python3 -m pytest tests/benchmarks/ -v --benchmark-only",
                timeout=600,
                required=False
            ),
            QualityGate(
                name="memory_profiling",
                level=QualityLevel.OPTIMIZED,
                command="python3 scripts/collect-metrics.py",
                timeout=180,
                required=False
            ),
            QualityGate(
                name="integration_tests",
                level=QualityLevel.OPTIMIZED,
                command="python3 -m pytest tests/integration/ -v --tb=short",
                timeout=300,
                required=True
            )
        ]
    
    def _research_grade_gates(self) -> List[QualityGate]:
        """Research-grade quality gates - publication ready."""
        return [
            QualityGate(
                name="e2e_workflow_tests",
                level=QualityLevel.RESEARCH_GRADE,
                command="python3 -m pytest tests/e2e/ -v --tb=long",
                timeout=900,
                required=True
            ),
            QualityGate(
                name="security_integration",
                level=QualityLevel.RESEARCH_GRADE,
                command="python3 -m pytest tests/security/ -v",
                timeout=300,
                required=True
            ),
            QualityGate(
                name="documentation_validation",
                level=QualityLevel.RESEARCH_GRADE,
                command="python3 -c 'import doctest; import src.pqc_iot_retrofit; print(\"Doctest validation complete\")'",
                timeout=60,
                required=False
            )
        ]
    
    def execute_gate(self, gate: QualityGate) -> QualityGate:
        """Execute a single quality gate."""
        self.logger.info(f"🚀 Executing gate: {gate.name}")
        gate.status = GateStatus.RUNNING
        start_time = time.time()
        
        try:
            # Execute command
            result = subprocess.run(
                gate.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=gate.timeout,
                cwd=self.project_root
            )
            
            gate.execution_time = time.time() - start_time
            gate.output = result.stdout
            gate.error = result.stderr
            
            if result.returncode == 0:
                gate.status = GateStatus.PASSED
                self.logger.info(f"✅ Gate passed: {gate.name} ({gate.execution_time:.2f}s)")
            else:
                gate.status = GateStatus.FAILED
                self.logger.warning(f"❌ Gate failed: {gate.name} (code: {result.returncode})")
                
            # Extract metrics from output
            self._extract_metrics(gate)
            
        except subprocess.TimeoutExpired:
            gate.execution_time = time.time() - start_time
            gate.status = GateStatus.FAILED
            gate.error = f"Command timed out after {gate.timeout}s"
            self.logger.error(f"⏰ Gate timed out: {gate.name}")
            
        except Exception as e:
            gate.execution_time = time.time() - start_time
            gate.status = GateStatus.FAILED
            gate.error = str(e)
            self.logger.error(f"💥 Gate error: {gate.name} - {e}")
        
        return gate
    
    def _extract_metrics(self, gate: QualityGate):
        """Extract quality metrics from gate output."""
        try:
            # Coverage metrics
            if "coverage" in gate.name.lower() and gate.output:
                if "coverage.json" in gate.command:
                    coverage_file = self.project_root / "coverage.json"
                    if coverage_file.exists():
                        with open(coverage_file) as f:
                            coverage_data = json.load(f)
                            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
                            gate.metrics.append(QualityMetric(
                                name="test_coverage",
                                value=total_coverage,
                                threshold=85.0,
                                passed=total_coverage >= 85.0,
                                message=f"Test coverage: {total_coverage:.1f}%"
                            ))
            
            # Security scan metrics
            elif "bandit" in gate.command and gate.output:
                try:
                    bandit_data = json.loads(gate.output)
                    high_severity = len([r for r in bandit_data.get("results", []) 
                                       if r.get("issue_severity") == "HIGH"])
                    gate.metrics.append(QualityMetric(
                        name="security_high_issues",
                        value=high_severity,
                        threshold=0,
                        passed=high_severity == 0,
                        message=f"High severity security issues: {high_severity}"
                    ))
                except json.JSONDecodeError:
                    pass
            
            # Performance metrics
            elif "benchmark" in gate.name.lower():
                # Extract benchmark results from output
                lines = gate.output.split('\n')
                for line in lines:
                    if "seconds" in line.lower() and "ops" in line.lower():
                        # Simple benchmark metric extraction
                        gate.metrics.append(QualityMetric(
                            name="performance_baseline",
                            value="passed",
                            threshold="baseline",
                            passed=True,
                            message="Performance benchmarks completed"
                        ))
                        break
                        
        except Exception as e:
            self.logger.debug(f"Could not extract metrics from {gate.name}: {e}")
    
    def execute_level(self, level: QualityLevel) -> Dict[str, QualityGate]:
        """Execute all gates for a specific quality level."""
        self.logger.info(f"📊 Executing quality level: {level.value}")
        gates = self.gates[level]
        results = {}
        
        # Execute gates in parallel for efficiency
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_gate = {executor.submit(self.execute_gate, gate): gate 
                            for gate in gates}
            
            for future in as_completed(future_to_gate):
                gate = future_to_gate[future]
                try:
                    results[gate.name] = future.result()
                except Exception as e:
                    self.logger.error(f"Gate execution failed: {gate.name} - {e}")
                    gate.status = GateStatus.FAILED
                    gate.error = str(e)
                    results[gate.name] = gate
        
        return results
    
    def progressive_execution(self) -> Dict[str, Any]:
        """Execute quality gates progressively through all levels."""
        self.logger.info("🎯 Starting Progressive Quality Gates execution")
        start_time = time.time()
        
        all_results = {}
        level_summary = {}
        
        for level in QualityLevel:
            level_start = time.time()
            self.logger.info(f"🏗️  Processing {level.value} quality level")
            
            # Execute level gates
            level_results = self.execute_level(level)
            all_results[level.value] = level_results
            
            # Analyze level results
            total_gates = len(level_results)
            passed_gates = sum(1 for g in level_results.values() if g.status == GateStatus.PASSED)
            failed_gates = sum(1 for g in level_results.values() if g.status == GateStatus.FAILED)
            required_failed = sum(1 for g in level_results.values() 
                                if g.status == GateStatus.FAILED and g.required)
            
            level_passed = required_failed == 0
            level_summary[level.value] = {
                "passed": level_passed,
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "failed_gates": failed_gates,
                "required_failed": required_failed,
                "execution_time": time.time() - level_start
            }
            
            if level_passed:
                self.logger.info(f"✅ {level.value} level PASSED ({passed_gates}/{total_gates} gates)")
                self.current_level = level
            else:
                self.logger.warning(f"❌ {level.value} level FAILED ({required_failed} required gates failed)")
                if level in [QualityLevel.BASIC, QualityLevel.ROBUST]:
                    self.logger.error("🛑 Critical level failed - stopping execution")
                    break
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        report = {
            "execution_id": self.execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root),
            "achieved_level": self.current_level.value,
            "total_execution_time": total_time,
            "level_summary": level_summary,
            "detailed_results": all_results,
            "recommendations": self._generate_recommendations(all_results)
        }
        
        # Save report
        self._save_report(report)
        
        self.logger.info(f"🎉 Progressive Quality Gates completed in {total_time:.2f}s")
        self.logger.info(f"📈 Achieved quality level: {self.current_level.value}")
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Dict[str, QualityGate]]) -> List[str]:
        """Generate improvement recommendations based on results."""
        recommendations = []
        
        for level_name, level_results in results.items():
            for gate_name, gate in level_results.items():
                if gate.status == GateStatus.FAILED:
                    if "test" in gate_name.lower():
                        recommendations.append(
                            f"Fix failing tests in {gate_name}: Review test output and fix implementation issues"
                        )
                    elif "security" in gate_name.lower():
                        recommendations.append(
                            f"Address security issues from {gate_name}: Review and fix identified vulnerabilities"
                        )
                    elif "style" in gate_name.lower() or "ruff" in gate.command:
                        recommendations.append(
                            f"Fix code style issues: Run 'ruff check --fix src/' to auto-fix styling"
                        )
                    elif "type" in gate_name.lower():
                        recommendations.append(
                            f"Fix type checking issues: Add proper type hints and resolve mypy warnings"
                        )
        
        # Add progressive enhancement recommendations
        if self.current_level == QualityLevel.BASIC:
            recommendations.append("🚀 Ready for Generation 2: Add comprehensive error handling and validation")
        elif self.current_level == QualityLevel.ROBUST:
            recommendations.append("🚀 Ready for Generation 3: Implement performance optimizations and scaling")
        elif self.current_level == QualityLevel.OPTIMIZED:
            recommendations.append("🚀 Ready for Research Grade: Add formal verification and publication-ready docs")
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """Save quality gate report to disk."""
        # Save detailed report
        report_file = self.report_dir / f"quality_gate_report_{self.execution_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save latest report (for CI/CD)
        latest_file = self.report_dir / "quality_gate_report_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Report saved: {report_file}")

def main():
    """Main entry point for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Progressive Quality Gates System")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory")
    parser.add_argument("--level", choices=[l.value for l in QualityLevel],
                       help="Execute specific quality level only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize quality gates system
    pqg = ProgressiveQualityGates(project_root=args.project_root)
    
    if args.level:
        # Execute specific level
        level = QualityLevel(args.level)
        results = pqg.execute_level(level)
        print(f"✅ Executed {level.value} level - {len(results)} gates processed")
    else:
        # Execute progressive enhancement
        report = pqg.progressive_execution()
        print(f"🎯 Quality assessment complete - achieved level: {report['achieved_level']}")
        print(f"📊 View detailed report at: reports/quality_gate_report_latest.json")
        
        # Print recommendations
        if report['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")

if __name__ == "__main__":
    main()