#!/usr/bin/env python3
"""
Repository health monitoring script for PQC IoT Retrofit Scanner.

Monitors various aspects of repository health and generates alerts
when thresholds are exceeded or trends indicate potential issues.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import requests
from dataclasses import dataclass
from enum import Enum


class HealthStatus(Enum):
    """Health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Represents a health monitoring alert."""
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    recommendation: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class HealthMetric:
    """Represents a health metric with thresholds."""
    name: str
    current_value: float
    unit: str
    status: HealthStatus
    thresholds: Dict[str, float]
    trend: str = "stable"
    last_updated: datetime = None
    metadata: Dict[str, Any] = None


class RepositoryHealthMonitor:
    """Main repository health monitoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.alerts: List[HealthAlert] = []
        self.metrics: List[HealthMetric] = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load health monitoring configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Load default configuration from project metrics
        project_metrics_path = Path(".github/project-metrics.json")
        if project_metrics_path.exists():
            with open(project_metrics_path, 'r') as f:
                project_config = json.load(f)
                return {
                    "thresholds": {
                        metric_name: metric_data.get("thresholds", {})
                        for category in project_config.get("metrics", {}).values()
                        for metric_name, metric_data in category.items()
                        if isinstance(metric_data, dict) and "thresholds" in metric_data
                    },
                    "alerts": {
                        "slack_webhook": os.getenv("SLACK_WEBHOOK_URL", ""),
                        "email_recipients": os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(","),
                        "github_issues": os.getenv("CREATE_GITHUB_ISSUES", "false").lower() == "true"
                    }
                }
        
        return {"thresholds": {}, "alerts": {}}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_latest_metrics(self, metrics_file: str = ".github/metrics-results.json") -> bool:
        """Load the latest metrics from file."""
        metrics_path = Path(metrics_file)
        if not metrics_path.exists():
            self.logger.warning(f"Metrics file not found: {metrics_file}")
            return False
        
        try:
            with open(metrics_path, 'r') as f:
                metrics_data = json.load(f)
            
            self.metrics = []
            for metric in metrics_data.get("metrics", []):
                health_metric = self._create_health_metric(metric)
                if health_metric:
                    self.metrics.append(health_metric)
            
            self.logger.info(f"Loaded {len(self.metrics)} metrics for health monitoring")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            return False
    
    def _create_health_metric(self, metric_data: Dict) -> Optional[HealthMetric]:
        """Create a health metric from metric data."""
        try:
            metric_name = metric_data["name"]
            current_value = metric_data["value"]
            unit = metric_data.get("unit", "")
            
            # Get thresholds from config
            thresholds = self.config.get("thresholds", {}).get(metric_name, {})
            if not thresholds:
                # Skip metrics without thresholds
                return None
            
            # Determine status based on thresholds
            status = self._determine_status(current_value, thresholds)
            
            return HealthMetric(
                name=metric_name,
                current_value=current_value,
                unit=unit,
                status=status,
                thresholds=thresholds,
                last_updated=datetime.fromisoformat(metric_data["timestamp"]),
                metadata=metric_data.get("metadata", {})
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create health metric: {e}")
            return None
    
    def _determine_status(self, value: float, thresholds: Dict[str, float]) -> HealthStatus:
        """Determine health status based on value and thresholds."""
        try:
            if value <= thresholds.get("excellent", float('inf')):
                return HealthStatus.EXCELLENT
            elif value <= thresholds.get("good", float('inf')):
                return HealthStatus.GOOD
            elif value <= thresholds.get("warning", float('inf')):
                return HealthStatus.WARNING
            else:
                return HealthStatus.CRITICAL
        except Exception:
            return HealthStatus.UNKNOWN
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        self.logger.info("Starting repository health check")
        
        health_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": HealthStatus.GOOD.value,
            "metrics_count": len(self.metrics),
            "alerts_count": 0,
            "status_breakdown": {
                "excellent": 0,
                "good": 0,
                "warning": 0,
                "critical": 0,
                "unknown": 0
            },
            "recommendations": []
        }
        
        # Analyze each metric
        critical_count = 0
        warning_count = 0
        
        for metric in self.metrics:
            status_value = metric.status.value
            health_summary["status_breakdown"][status_value] += 1
            
            # Generate alerts for problematic metrics
            if metric.status == HealthStatus.CRITICAL:
                critical_count += 1
                self._create_alert(
                    AlertSeverity.CRITICAL,
                    f"Critical: {metric.name}",
                    f"{metric.name} is in critical state: {metric.current_value} {metric.unit}",
                    metric
                )
            elif metric.status == HealthStatus.WARNING:
                warning_count += 1
                self._create_alert(
                    AlertSeverity.WARNING,
                    f"Warning: {metric.name}",
                    f"{metric.name} exceeds warning threshold: {metric.current_value} {metric.unit}",
                    metric
                )
        
        # Determine overall status
        if critical_count > 0:
            health_summary["overall_status"] = HealthStatus.CRITICAL.value
        elif warning_count > 0:
            health_summary["overall_status"] = HealthStatus.WARNING.value
        elif health_summary["status_breakdown"]["excellent"] > len(self.metrics) * 0.8:
            health_summary["overall_status"] = HealthStatus.EXCELLENT.value
        
        health_summary["alerts_count"] = len(self.alerts)
        health_summary["recommendations"] = self._generate_recommendations()
        
        # Perform additional health checks
        self._check_repository_freshness(health_summary)
        self._check_dependency_health(health_summary)
        self._check_workflow_health(health_summary)
        
        self.logger.info(f"Health check completed. Status: {health_summary['overall_status']}")
        return health_summary
    
    def _create_alert(self, severity: AlertSeverity, title: str, description: str, metric: HealthMetric):
        """Create a health alert."""
        # Determine appropriate threshold for comparison
        threshold_value = 0
        if severity == AlertSeverity.CRITICAL:
            threshold_value = metric.thresholds.get("critical", 0)
        elif severity == AlertSeverity.WARNING:
            threshold_value = metric.thresholds.get("warning", 0)
        
        recommendation = self._get_recommendation(metric.name, metric.status)
        
        alert = HealthAlert(
            severity=severity,
            title=title,
            description=description,
            metric_name=metric.name,
            current_value=metric.current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(timezone.utc),
            recommendation=recommendation,
            metadata=metric.metadata
        )
        
        self.alerts.append(alert)
    
    def _get_recommendation(self, metric_name: str, status: HealthStatus) -> str:
        """Get recommendation for improving metric."""
        recommendations = {
            "test_coverage": {
                HealthStatus.WARNING: "Add more unit tests to increase coverage above 70%",
                HealthStatus.CRITICAL: "Critical: Test coverage is below 50%. Immediately add comprehensive tests."
            },
            "security_vulnerabilities": {
                HealthStatus.WARNING: "Review and fix security vulnerabilities found by static analysis",
                HealthStatus.CRITICAL: "Critical security vulnerabilities detected. Fix immediately before next release."
            },
            "dependency_vulnerabilities": {
                HealthStatus.WARNING: "Update vulnerable dependencies using 'pip-audit' recommendations",
                HealthStatus.CRITICAL: "Critical dependency vulnerabilities. Update or replace vulnerable packages immediately."
            },
            "build_time": {
                HealthStatus.WARNING: "Build time is increasing. Consider optimizing CI pipeline or reducing dependencies",
                HealthStatus.CRITICAL: "Build time is excessive. Investigate performance bottlenecks in CI/CD pipeline."
            },
            "code_complexity": {
                HealthStatus.WARNING: "Code complexity is high. Consider refactoring complex functions",
                HealthStatus.CRITICAL: "Code complexity is very high. Refactoring is needed to improve maintainability."
            },
            "technical_debt_ratio": {
                HealthStatus.WARNING: "Technical debt is accumulating. Address TODO/FIXME comments",
                HealthStatus.CRITICAL: "High technical debt detected. Plan a dedicated refactoring sprint."
            }
        }
        
        return recommendations.get(metric_name, {}).get(status, "Monitor this metric and take action if it worsens.")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate overall health recommendations."""
        recommendations = []
        
        # Count status types
        status_counts = {status.value: 0 for status in HealthStatus}
        for metric in self.metrics:
            status_counts[metric.status.value] += 1
        
        if status_counts["critical"] > 0:
            recommendations.append(f"Address {status_counts['critical']} critical issues immediately")
        
        if status_counts["warning"] > 0:
            recommendations.append(f"Plan to resolve {status_counts['warning']} warning-level issues")
        
        if status_counts["excellent"] < len(self.metrics) * 0.5:
            recommendations.append("Focus on improving overall code quality metrics")
        
        # Specific recommendations based on metrics
        coverage_metrics = [m for m in self.metrics if "coverage" in m.name.lower()]
        if coverage_metrics and all(m.status in [HealthStatus.WARNING, HealthStatus.CRITICAL] for m in coverage_metrics):
            recommendations.append("Invest in comprehensive testing strategy")
        
        security_metrics = [m for m in self.metrics if "security" in m.name.lower() or "vulnerabilities" in m.name.lower()]
        if security_metrics and any(m.status == HealthStatus.CRITICAL for m in security_metrics):
            recommendations.append("Conduct immediate security review and remediation")
        
        return recommendations
    
    def _check_repository_freshness(self, health_summary: Dict):
        """Check if repository is being actively maintained."""
        try:
            import subprocess
            
            # Check last commit date
            result = subprocess.run([
                "git", "log", "-1", "--format=%ct"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                last_commit_timestamp = int(result.stdout.strip())
                last_commit_date = datetime.fromtimestamp(last_commit_timestamp, tz=timezone.utc)
                days_since_commit = (datetime.now(timezone.utc) - last_commit_date).days
                
                if days_since_commit > 30:
                    self.alerts.append(HealthAlert(
                        severity=AlertSeverity.WARNING,
                        title="Repository Staleness",
                        description=f"No commits in {days_since_commit} days",
                        metric_name="repository_freshness",
                        current_value=days_since_commit,
                        threshold_value=30,
                        timestamp=datetime.now(timezone.utc),
                        recommendation="Consider making regular commits or creating a maintenance schedule"
                    ))
                
                health_summary["last_commit_days"] = days_since_commit
                
        except Exception as e:
            self.logger.warning(f"Failed to check repository freshness: {e}")
    
    def _check_dependency_health(self, health_summary: Dict):
        """Check health of project dependencies."""
        try:
            import subprocess
            
            # Check for outdated packages
            result = subprocess.run([
                "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                outdated_count = len(outdated_packages)
                
                health_summary["outdated_dependencies"] = outdated_count
                
                if outdated_count > 10:
                    self.alerts.append(HealthAlert(
                        severity=AlertSeverity.WARNING,
                        title="Many Outdated Dependencies",
                        description=f"{outdated_count} packages are outdated",
                        metric_name="outdated_dependencies",
                        current_value=outdated_count,
                        threshold_value=10,
                        timestamp=datetime.now(timezone.utc),
                        recommendation="Schedule regular dependency updates"
                    ))
                
        except Exception as e:
            self.logger.warning(f"Failed to check dependency health: {e}")
    
    def _check_workflow_health(self, health_summary: Dict):
        """Check GitHub Actions workflow health."""
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            if not github_token:
                return
            
            repo = os.getenv("GITHUB_REPOSITORY", "")
            if not repo:
                return
            
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get recent workflow runs
            response = requests.get(
                f"https://api.github.com/repos/{repo}/actions/runs",
                headers=headers,
                params={"per_page": 10},
                timeout=30
            )
            
            if response.status_code == 200:
                runs = response.json().get("workflow_runs", [])
                
                failed_runs = [run for run in runs if run.get("conclusion") == "failure"]
                failure_rate = len(failed_runs) / max(len(runs), 1) * 100
                
                health_summary["workflow_failure_rate"] = failure_rate
                
                if failure_rate > 20:
                    self.alerts.append(HealthAlert(
                        severity=AlertSeverity.WARNING,
                        title="High Workflow Failure Rate",
                        description=f"Workflow failure rate is {failure_rate:.1f}%",
                        metric_name="workflow_failure_rate",
                        current_value=failure_rate,
                        threshold_value=20,
                        timestamp=datetime.now(timezone.utc),
                        recommendation="Investigate and fix failing workflows"
                    ))
                
        except Exception as e:
            self.logger.warning(f"Failed to check workflow health: {e}")
    
    def send_alerts(self):
        """Send alerts through configured channels."""
        if not self.alerts:
            self.logger.info("No alerts to send")
            return
        
        # Send to Slack
        slack_webhook = self.config.get("alerts", {}).get("slack_webhook")
        if slack_webhook:
            self._send_slack_alerts(slack_webhook)
        
        # Create GitHub issues
        if self.config.get("alerts", {}).get("github_issues"):
            self._create_github_issues()
        
        # Log all alerts
        for alert in self.alerts:
            self.logger.log(
                self._alert_severity_to_log_level(alert.severity),
                f"{alert.title}: {alert.description}"
            )
    
    def _send_slack_alerts(self, webhook_url: str):
        """Send alerts to Slack."""
        try:
            critical_alerts = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
            warning_alerts = [a for a in self.alerts if a.severity == AlertSeverity.WARNING]
            
            color = "danger" if critical_alerts else "warning"
            
            message = {
                "text": "Repository Health Alert",
                "attachments": [{
                    "color": color,
                    "title": f"Health Check Results - {len(self.alerts)} alerts",
                    "fields": [
                        {
                            "title": "Critical Issues",
                            "value": str(len(critical_alerts)),
                            "short": True
                        },
                        {
                            "title": "Warnings",
                            "value": str(len(warning_alerts)),
                            "short": True
                        }
                    ],
                    "text": "\n".join([
                        f"• {alert.title}: {alert.description}"
                        for alert in self.alerts[:5]  # Limit to first 5 alerts
                    ])
                }]
            }
            
            response = requests.post(webhook_url, json=message, timeout=10)
            response.raise_for_status()
            
            self.logger.info("Alerts sent to Slack successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alerts: {e}")
    
    def _create_github_issues(self):
        """Create GitHub issues for critical alerts."""
        try:
            github_token = os.getenv("GITHUB_TOKEN")
            repo = os.getenv("GITHUB_REPOSITORY")
            
            if not github_token or not repo:
                self.logger.warning("GitHub token or repository not configured")
                return
            
            headers = {
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            critical_alerts = [a for a in self.alerts if a.severity == AlertSeverity.CRITICAL]
            
            for alert in critical_alerts:
                issue_body = f"""
**Description:** {alert.description}

**Metric:** {alert.metric_name}
**Current Value:** {alert.current_value}
**Threshold:** {alert.threshold_value}
**Timestamp:** {alert.timestamp.isoformat()}

**Recommendation:** {alert.recommendation}

---
*This issue was automatically created by the repository health monitor.*
                """.strip()
                
                issue_data = {
                    "title": f"Health Alert: {alert.title}",
                    "body": issue_body,
                    "labels": ["health-alert", "critical", "automated"]
                }
                
                response = requests.post(
                    f"https://api.github.com/repos/{repo}/issues",
                    headers=headers,
                    json=issue_data,
                    timeout=30
                )
                
                if response.status_code == 201:
                    self.logger.info(f"Created GitHub issue for {alert.title}")
                else:
                    self.logger.warning(f"Failed to create GitHub issue: {response.status_code}")
                    
        except Exception as e:
            self.logger.error(f"Failed to create GitHub issues: {e}")
    
    def _alert_severity_to_log_level(self, severity: AlertSeverity) -> int:
        """Convert alert severity to logging level."""
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        return mapping.get(severity, logging.INFO)
    
    def generate_health_report(self, output_file: str = ".github/health-report.json"):
        """Generate comprehensive health report."""
        health_summary = self.perform_health_check()
        
        report = {
            "health_summary": health_summary,
            "metrics": [
                {
                    "name": metric.name,
                    "current_value": metric.current_value,
                    "unit": metric.unit,
                    "status": metric.status.value,
                    "thresholds": metric.thresholds,
                    "trend": metric.trend,
                    "last_updated": metric.last_updated.isoformat() if metric.last_updated else None,
                    "metadata": metric.metadata
                }
                for metric in self.metrics
            ],
            "alerts": [
                {
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "threshold_value": alert.threshold_value,
                    "timestamp": alert.timestamp.isoformat(),
                    "recommendation": alert.recommendation,
                    "metadata": alert.metadata
                }
                for alert in self.alerts
            ]
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health report saved to {output_file}")
        return report


def main():
    """Main entry point for health monitoring."""
    parser = argparse.ArgumentParser(description="Monitor repository health")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--metrics-file", default=".github/metrics-results.json",
                        help="Path to metrics file")
    parser.add_argument("--output", default=".github/health-report.json",
                        help="Output file path")
    parser.add_argument("--send-alerts", action="store_true",
                        help="Send alerts through configured channels")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        monitor = RepositoryHealthMonitor(config_path=args.config)
        
        if not monitor.load_latest_metrics(args.metrics_file):
            print("Failed to load metrics file")
            return 1
        
        report = monitor.generate_health_report(args.output)
        
        if args.send_alerts:
            monitor.send_alerts()
        
        # Print summary
        health_summary = report["health_summary"]
        print(f"Repository Health Status: {health_summary['overall_status'].upper()}")
        print(f"Metrics analyzed: {health_summary['metrics_count']}")
        print(f"Alerts generated: {health_summary['alerts_count']}")
        
        if health_summary["alerts_count"] > 0:
            print("\nTop recommendations:")
            for i, rec in enumerate(health_summary["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        
        # Exit with appropriate code
        if health_summary["overall_status"] == "critical":
            return 2
        elif health_summary["overall_status"] == "warning":
            return 1
        else:
            return 0
        
    except Exception as e:
        logging.error(f"Health monitoring failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())