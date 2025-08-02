#!/usr/bin/env python3
"""
Automated metrics collection script for PQC IoT Retrofit Scanner.

Collects comprehensive project metrics from various sources and updates
the project metrics dashboard and tracking systems.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Represents a single metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.results: List[MetricResult] = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load metrics configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "github": {
                "owner": os.getenv("GITHUB_REPOSITORY_OWNER", ""),
                "repo": os.getenv("GITHUB_REPOSITORY", "").split("/")[-1] if "/" in os.getenv("GITHUB_REPOSITORY", "") else "",
                "token": os.getenv("GITHUB_TOKEN", "")
            },
            "codecov": {
                "token": os.getenv("CODECOV_TOKEN", "")
            },
            "output": {
                "file": ".github/metrics-results.json",
                "format": "json"
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_all_metrics(self) -> List[MetricResult]:
        """Collect all available metrics."""
        self.logger.info("Starting comprehensive metrics collection")
        
        # Code quality metrics
        self._collect_test_coverage()
        self._collect_code_complexity()
        self._collect_technical_debt()
        self._collect_code_duplication()
        
        # Security metrics
        self._collect_security_vulnerabilities()
        self._collect_dependency_vulnerabilities()
        
        # Performance metrics
        self._collect_build_performance()
        self._collect_test_performance()
        self._collect_docker_metrics()
        
        # Maintainability metrics
        self._collect_maintainability_index()
        self._collect_documentation_coverage()
        self._collect_code_churn()
        
        # Delivery metrics
        self._collect_deployment_metrics()
        self._collect_github_metrics()
        
        self.logger.info(f"Collected {len(self.results)} metrics")
        return self.results
    
    def _collect_test_coverage(self):
        """Collect test coverage metrics."""
        try:
            # Try to read coverage report
            coverage_files = [
                "coverage.xml",
                "htmlcov/index.html",
                ".coverage"
            ]
            
            for coverage_file in coverage_files:
                if Path(coverage_file).exists():
                    if coverage_file.endswith(".xml"):
                        coverage = self._parse_coverage_xml(coverage_file)
                    else:
                        # Run coverage report to get percentage
                        result = subprocess.run(
                            ["python", "-m", "coverage", "report", "--show-missing"],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            coverage = self._parse_coverage_text(result.stdout)
                        else:
                            coverage = 0
                    
                    self.results.append(MetricResult(
                        name="test_coverage",
                        value=coverage,
                        unit="percentage",
                        timestamp=datetime.now(timezone.utc),
                        source="coverage",
                        metadata={"file": coverage_file}
                    ))
                    break
            else:
                self.logger.warning("No coverage reports found")
                
        except Exception as e:
            self.logger.error(f"Failed to collect test coverage: {e}")
    
    def _collect_code_complexity(self):
        """Collect code complexity metrics using radon."""
        try:
            result = subprocess.run([
                "python", "-m", "radon", "cc", "src/", "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                avg_complexity = self._calculate_average_complexity(complexity_data)
                
                self.results.append(MetricResult(
                    name="code_complexity",
                    value=avg_complexity,
                    unit="cyclomatic_complexity",
                    timestamp=datetime.now(timezone.utc),
                    source="radon",
                    metadata={"total_functions": len(complexity_data)}
                ))
            else:
                self.logger.warning("Failed to run radon complexity analysis")
                
        except Exception as e:
            self.logger.error(f"Failed to collect code complexity: {e}")
    
    def _collect_security_vulnerabilities(self):
        """Collect security vulnerability metrics."""
        try:
            # Run bandit security scan
            result = subprocess.run([
                "python", "-m", "bandit", "-r", "src/", "-f", "json"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode in [0, 1]:  # 0 = no issues, 1 = issues found
                if result.stdout:
                    bandit_data = json.loads(result.stdout)
                    vulnerability_count = len(bandit_data.get("results", []))
                    
                    # Count by severity
                    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                    for issue in bandit_data.get("results", []):
                        severity = issue.get("issue_severity", "LOW")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    self.results.append(MetricResult(
                        name="security_vulnerabilities",
                        value=vulnerability_count,
                        unit="count",
                        timestamp=datetime.now(timezone.utc),
                        source="bandit",
                        metadata={"severity_breakdown": severity_counts}
                    ))
                else:
                    # No vulnerabilities found
                    self.results.append(MetricResult(
                        name="security_vulnerabilities",
                        value=0,
                        unit="count",
                        timestamp=datetime.now(timezone.utc),
                        source="bandit"
                    ))
            else:
                self.logger.warning("Bandit scan failed or returned unexpected code")
                
        except Exception as e:
            self.logger.error(f"Failed to collect security vulnerabilities: {e}")
    
    def _collect_dependency_vulnerabilities(self):
        """Collect dependency vulnerability metrics."""
        try:
            # Run safety check
            result = subprocess.run([
                "python", "-m", "safety", "check", "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode in [0, 64]:  # 0 = safe, 64 = vulnerabilities found
                if result.stdout:
                    try:
                        safety_data = json.loads(result.stdout)
                        vuln_count = len(safety_data)
                    except json.JSONDecodeError:
                        # Sometimes safety outputs plain text
                        vuln_count = 0 if "No known security vulnerabilities found" in result.stdout else 1
                else:
                    vuln_count = 0
                
                self.results.append(MetricResult(
                    name="dependency_vulnerabilities",
                    value=vuln_count,
                    unit="count",
                    timestamp=datetime.now(timezone.utc),
                    source="safety"
                ))
            else:
                self.logger.warning("Safety check failed")
                
        except Exception as e:
            self.logger.error(f"Failed to collect dependency vulnerabilities: {e}")
    
    def _collect_build_performance(self):
        """Collect build performance metrics from CI."""
        try:
            if self.config["github"]["token"] and self.config["github"]["repo"]:
                github_api = GitHubMetricsAPI(
                    self.config["github"]["token"],
                    self.config["github"]["owner"],
                    self.config["github"]["repo"]
                )
                
                workflow_runs = github_api.get_recent_workflow_runs(limit=10)
                if workflow_runs:
                    avg_duration = sum(run["duration"] for run in workflow_runs) / len(workflow_runs)
                    
                    self.results.append(MetricResult(
                        name="build_time",
                        value=avg_duration,
                        unit="seconds",
                        timestamp=datetime.now(timezone.utc),
                        source="github-actions",
                        metadata={"sample_size": len(workflow_runs)}
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to collect build performance: {e}")
    
    def _collect_docker_metrics(self):
        """Collect Docker image metrics."""
        try:
            # Check if Docker image exists
            result = subprocess.run([
                "docker", "images", "--format", "{{.Size}}", "pqc-iot-retrofit-scanner:latest"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and result.stdout.strip():
                size_str = result.stdout.strip()
                size_mb = self._parse_docker_size(size_str)
                
                self.results.append(MetricResult(
                    name="docker_image_size",
                    value=size_mb,
                    unit="MB",
                    timestamp=datetime.now(timezone.utc),
                    source="docker",
                    metadata={"raw_size": size_str}
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to collect Docker metrics: {e}")
    
    def _collect_github_metrics(self):
        """Collect GitHub repository metrics."""
        try:
            if self.config["github"]["token"] and self.config["github"]["repo"]:
                github_api = GitHubMetricsAPI(
                    self.config["github"]["token"],
                    self.config["github"]["owner"],
                    self.config["github"]["repo"]
                )
                
                # Collect various GitHub metrics
                repo_info = github_api.get_repository_info()
                if repo_info:
                    self.results.extend([
                        MetricResult(
                            name="github_stars",
                            value=repo_info.get("stargazers_count", 0),
                            unit="count",
                            timestamp=datetime.now(timezone.utc),
                            source="github-api"
                        ),
                        MetricResult(
                            name="github_forks",
                            value=repo_info.get("forks_count", 0),
                            unit="count",
                            timestamp=datetime.now(timezone.utc),
                            source="github-api"
                        ),
                        MetricResult(
                            name="github_open_issues",
                            value=repo_info.get("open_issues_count", 0),
                            unit="count",
                            timestamp=datetime.now(timezone.utc),
                            source="github-api"
                        )
                    ])
                
                # Collect PR and commit metrics
                pr_metrics = github_api.get_pull_request_metrics()
                if pr_metrics:
                    self.results.append(MetricResult(
                        name="pr_lead_time",
                        value=pr_metrics["average_lead_time_hours"],
                        unit="hours",
                        timestamp=datetime.now(timezone.utc),
                        source="github-api",
                        metadata={"sample_size": pr_metrics["sample_size"]}
                    ))
                    
        except Exception as e:
            self.logger.error(f"Failed to collect GitHub metrics: {e}")
    
    def _collect_technical_debt(self):
        """Estimate technical debt using various heuristics."""
        try:
            # Count TODO, FIXME, HACK comments
            result = subprocess.run([
                "grep", "-r", "-i", "-E", "(TODO|FIXME|HACK|XXX)", "src/", "--include=*.py"
            ], capture_output=True, text=True, timeout=30)
            
            debt_comments = len(result.stdout.splitlines()) if result.returncode == 0 else 0
            
            # Get total lines of code
            loc_result = subprocess.run([
                "find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True, timeout=30)
            
            total_loc = 0
            if loc_result.returncode == 0:
                for line in loc_result.stdout.splitlines():
                    if line.strip() and not line.strip().endswith("total"):
                        try:
                            total_loc += int(line.split()[0])
                        except (ValueError, IndexError):
                            continue
            
            # Calculate debt ratio as percentage
            debt_ratio = (debt_comments / max(total_loc, 1)) * 100 if total_loc > 0 else 0
            
            self.results.append(MetricResult(
                name="technical_debt_ratio",
                value=debt_ratio,
                unit="percentage",
                timestamp=datetime.now(timezone.utc),
                source="static-analysis",
                metadata={
                    "debt_comments": debt_comments,
                    "total_loc": total_loc
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to estimate technical debt: {e}")
    
    def _collect_code_duplication(self):
        """Collect code duplication metrics."""
        # This is a placeholder - would need a proper duplication detection tool
        self.results.append(MetricResult(
            name="code_duplication",
            value=0,  # Default to 0 for now
            unit="percentage",
            timestamp=datetime.now(timezone.utc),
            source="placeholder",
            metadata={"note": "Requires duplication detection tool"}
        ))
    
    def _collect_maintainability_index(self):
        """Collect maintainability index using radon."""
        try:
            result = subprocess.run([
                "python", "-m", "radon", "mi", "src/", "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                mi_data = json.loads(result.stdout)
                avg_mi = self._calculate_average_maintainability(mi_data)
                
                self.results.append(MetricResult(
                    name="maintainability_index",
                    value=avg_mi,
                    unit="index",
                    timestamp=datetime.now(timezone.utc),
                    source="radon"
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to collect maintainability index: {e}")
    
    def _collect_documentation_coverage(self):
        """Collect documentation coverage metrics."""
        try:
            result = subprocess.run([
                "python", "-m", "pydocstyle", "src/", "--count"
            ], capture_output=True, text=True, timeout=60)
            
            # Pydocstyle returns error count, we need to calculate coverage
            # This is a simplified calculation
            undocumented_count = len(result.stdout.splitlines()) if result.returncode != 0 else 0
            
            # Count total functions/classes that should be documented
            python_files = list(Path("src/").rglob("*.py"))
            total_definitions = 0
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    # Simple count of def and class keywords
                    total_definitions += content.count("def ") + content.count("class ")
                except Exception:
                    continue
            
            if total_definitions > 0:
                doc_coverage = max(0, (total_definitions - undocumented_count) / total_definitions * 100)
            else:
                doc_coverage = 100  # No definitions to document
            
            self.results.append(MetricResult(
                name="documentation_coverage",
                value=doc_coverage,
                unit="percentage",
                timestamp=datetime.now(timezone.utc),
                source="pydocstyle",
                metadata={
                    "total_definitions": total_definitions,
                    "undocumented": undocumented_count
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Failed to collect documentation coverage: {e}")
    
    def _collect_code_churn(self):
        """Collect code churn metrics from git history."""
        try:
            # Get recent commits (last 30 days)
            result = subprocess.run([
                "git", "log", "--since=30.days.ago", "--numstat", "--pretty=format:"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_additions = 0
                total_deletions = 0
                
                for line in lines:
                    if line and '\t' in line:
                        parts = line.split('\t')
                        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                            total_additions += int(parts[0])
                            total_deletions += int(parts[1])
                
                total_changes = total_additions + total_deletions
                
                # Get total lines of code for churn percentage
                loc_result = subprocess.run([
                    "find", "src/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"
                ], capture_output=True, text=True, timeout=30)
                
                total_loc = 1000  # Default fallback
                if loc_result.returncode == 0:
                    total_loc = sum(int(line.split()[0]) for line in loc_result.stdout.splitlines() 
                                  if line.strip() and line.split()[0].isdigit())
                
                churn_percentage = (total_changes / max(total_loc, 1)) * 100
                
                self.results.append(MetricResult(
                    name="code_churn",
                    value=churn_percentage,
                    unit="percentage",
                    timestamp=datetime.now(timezone.utc),
                    source="git",
                    metadata={
                        "additions": total_additions,
                        "deletions": total_deletions,
                        "total_loc": total_loc
                    }
                ))
                
        except Exception as e:
            self.logger.error(f"Failed to collect code churn: {e}")
    
    def _collect_test_performance(self):
        """Collect test execution performance."""
        # This would typically come from CI/CD or test reports
        self.results.append(MetricResult(
            name="test_execution_time",
            value=0,  # Placeholder
            unit="seconds",
            timestamp=datetime.now(timezone.utc),
            source="placeholder",
            metadata={"note": "Requires test timing data"}
        ))
    
    def _collect_deployment_metrics(self):
        """Collect deployment frequency and other delivery metrics."""
        # This would typically come from deployment tracking systems
        self.results.append(MetricResult(
            name="deployment_frequency",
            value=0,  # Placeholder
            unit="days",
            timestamp=datetime.now(timezone.utc),
            source="placeholder",
            metadata={"note": "Requires deployment tracking"}
        ))
    
    # Helper methods
    def _parse_coverage_xml(self, xml_file: str) -> float:
        """Parse coverage percentage from XML file."""
        # Simplified XML parsing - would need proper XML parser for production
        try:
            content = Path(xml_file).read_text()
            if 'line-rate=' in content:
                import re
                match = re.search(r'line-rate="([0-9.]+)"', content)
                if match:
                    return float(match.group(1)) * 100
        except Exception:
            pass
        return 0
    
    def _parse_coverage_text(self, coverage_text: str) -> float:
        """Parse coverage percentage from text output."""
        try:
            lines = coverage_text.splitlines()
            for line in lines:
                if "TOTAL" in line and "%" in line:
                    import re
                    match = re.search(r'(\d+)%', line)
                    if match:
                        return float(match.group(1))
        except Exception:
            pass
        return 0
    
    def _calculate_average_complexity(self, complexity_data: Dict) -> float:
        """Calculate average cyclomatic complexity."""
        total_complexity = 0
        function_count = 0
        
        for file_data in complexity_data.values():
            for item in file_data:
                if isinstance(item, dict) and 'complexity' in item:
                    total_complexity += item['complexity']
                    function_count += 1
        
        return total_complexity / max(function_count, 1)
    
    def _calculate_average_maintainability(self, mi_data: Dict) -> float:
        """Calculate average maintainability index."""
        total_mi = 0
        file_count = 0
        
        for file_path, mi_value in mi_data.items():
            if isinstance(mi_value, (int, float)):
                total_mi += mi_value
                file_count += 1
        
        return total_mi / max(file_count, 1)
    
    def _parse_docker_size(self, size_str: str) -> float:
        """Parse Docker image size string to MB."""
        size_str = size_str.upper()
        if 'GB' in size_str:
            return float(size_str.replace('GB', '').strip()) * 1024
        elif 'MB' in size_str:
            return float(size_str.replace('MB', '').strip())
        elif 'KB' in size_str:
            return float(size_str.replace('KB', '').strip()) / 1024
        else:
            # Assume bytes
            try:
                return float(size_str) / (1024 * 1024)
            except ValueError:
                return 0
    
    def save_results(self, output_file: Optional[str] = None):
        """Save metrics results to file."""
        if not output_file:
            output_file = self.config["output"]["file"]
        
        output_data = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_count": len(self.results),
            "metrics": [
                {
                    "name": result.name,
                    "value": result.value,
                    "unit": result.unit,
                    "timestamp": result.timestamp.isoformat(),
                    "source": result.source,
                    "metadata": result.metadata or {}
                }
                for result in self.results
            ]
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Metrics saved to {output_file}")


class GitHubMetricsAPI:
    """GitHub API client for metrics collection."""
    
    def __init__(self, token: str, owner: str, repo: str):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
    
    def get_repository_info(self) -> Optional[Dict]:
        """Get basic repository information."""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Failed to get repository info: {e}")
            return None
    
    def get_recent_workflow_runs(self, limit: int = 10) -> List[Dict]:
        """Get recent workflow run metrics."""
        try:
            response = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/runs",
                headers=self.headers,
                params={"per_page": limit},
                timeout=30
            )
            response.raise_for_status()
            
            runs = response.json().get("workflow_runs", [])
            results = []
            
            for run in runs:
                if run.get("conclusion") == "success":
                    created_at = datetime.fromisoformat(run["created_at"].replace('Z', '+00:00'))
                    updated_at = datetime.fromisoformat(run["updated_at"].replace('Z', '+00:00'))
                    duration = (updated_at - created_at).total_seconds()
                    
                    results.append({
                        "id": run["id"],
                        "name": run["name"],
                        "duration": duration,
                        "created_at": created_at,
                        "status": run["conclusion"]
                    })
            
            return results
        except Exception as e:
            logging.error(f"Failed to get workflow runs: {e}")
            return []
    
    def get_pull_request_metrics(self, limit: int = 20) -> Optional[Dict]:
        """Get pull request lead time metrics."""
        try:
            # Get recently merged PRs
            response = requests.get(
                f"{self.base_url}/repos/{self.owner}/{self.repo}/pulls",
                headers=self.headers,
                params={"state": "closed", "per_page": limit, "sort": "updated"},
                timeout=30
            )
            response.raise_for_status()
            
            prs = response.json()
            lead_times = []
            
            for pr in prs:
                if pr.get("merged_at"):
                    created_at = datetime.fromisoformat(pr["created_at"].replace('Z', '+00:00'))
                    merged_at = datetime.fromisoformat(pr["merged_at"].replace('Z', '+00:00'))
                    lead_time_hours = (merged_at - created_at).total_seconds() / 3600
                    lead_times.append(lead_time_hours)
            
            if lead_times:
                return {
                    "average_lead_time_hours": sum(lead_times) / len(lead_times),
                    "median_lead_time_hours": sorted(lead_times)[len(lead_times) // 2],
                    "sample_size": len(lead_times)
                }
            
            return None
        except Exception as e:
            logging.error(f"Failed to get PR metrics: {e}")
            return None


def main():
    """Main entry point for metrics collection."""
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(config_path=args.config)
        collector.collect_all_metrics()
        collector.save_results(output_file=args.output)
        
        print(f"Successfully collected {len(collector.results)} metrics")
        
        # Print summary
        for result in collector.results:
            print(f"  {result.name}: {result.value} {result.unit} (from {result.source})")
        
        return 0
        
    except Exception as e:
        logging.error(f"Metrics collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())