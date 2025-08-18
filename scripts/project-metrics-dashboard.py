#!/usr/bin/env python3
"""
Project Metrics Dashboard Generator for PQC IoT Retrofit Scanner.

Creates interactive dashboards and reports from collected project metrics,
including trend analysis, health scoring, and actionable insights.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import tempfile
import webbrowser

# Try to import optional dependencies
try:
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting libraries not available. Install pandas and plotly for dashboard generation.")


@dataclass
class ProjectHealth:
    """Overall project health assessment."""
    score: float  # 0-100
    grade: str   # A, B, C, D, F
    trend: str   # improving, stable, declining
    critical_issues: List[str]
    recommendations: List[str]
    last_updated: str


class MetricsDashboard:
    """Project metrics dashboard generator."""
    
    def __init__(self, data_dir: str = ".github"):
        self.data_dir = Path(data_dir)
        self.metrics_history = []
        self.current_metrics = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_metrics_data(self, metrics_files: List[str] = None) -> bool:
        """Load metrics data from files."""
        if not metrics_files:
            # Look for metrics files in data directory
            metrics_files = list(self.data_dir.glob("*metrics*.json"))
            metrics_files.extend(list(self.data_dir.glob("*debt*.json")))
        
        if not metrics_files:
            self.logger.warning("No metrics files found")
            return False
        
        self.logger.info(f"Loading metrics from {len(metrics_files)} files")
        
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                if "collection_timestamp" in data:
                    # This is a metrics collection file
                    self.metrics_history.append(data)
                    if not self.current_metrics or data["collection_timestamp"] > self.current_metrics.get("collection_timestamp", ""):
                        self.current_metrics = data
                
            except Exception as e:
                self.logger.error(f"Failed to load metrics from {metrics_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.metrics_history)} metrics snapshots")
        return len(self.metrics_history) > 0
    
    def calculate_project_health(self) -> ProjectHealth:
        """Calculate overall project health score."""
        if not self.current_metrics:
            return ProjectHealth(
                score=0,
                grade="F",
                trend="unknown",
                critical_issues=["No metrics data available"],
                recommendations=["Run metrics collection first"],
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        
        metrics = {m["name"]: m["value"] for m in self.current_metrics.get("metrics", [])}
        
        # Health scoring algorithm
        health_score = 100.0
        critical_issues = []
        recommendations = []
        
        # Code quality factors (40% weight)
        if "test_coverage" in metrics:
            coverage = metrics["test_coverage"]
            if coverage < 50:
                health_score -= 20
                critical_issues.append(f"Low test coverage: {coverage:.1f}%")
                recommendations.append("Increase test coverage to at least 80%")
            elif coverage < 80:
                health_score -= 10
                recommendations.append("Improve test coverage")
        
        if "security_vulnerabilities" in metrics:
            vuln_count = metrics["security_vulnerabilities"]
            if vuln_count > 0:
                health_score -= min(vuln_count * 5, 25)
                critical_issues.append(f"Security vulnerabilities found: {vuln_count}")
                recommendations.append("Address all security vulnerabilities")
        
        if "code_complexity" in metrics:
            complexity = metrics["code_complexity"]
            if complexity > 15:
                health_score -= 15
                critical_issues.append(f"High code complexity: {complexity:.1f}")
                recommendations.append("Refactor complex functions")
            elif complexity > 10:
                health_score -= 5
        
        # Maintainability factors (30% weight)
        if "technical_debt_ratio" in metrics:
            debt_ratio = metrics["technical_debt_ratio"]
            if debt_ratio > 5:
                health_score -= 15
                critical_issues.append(f"High technical debt: {debt_ratio:.1f}%")
                recommendations.append("Address technical debt systematically")
            elif debt_ratio > 2:
                health_score -= 8
        
        if "documentation_coverage" in metrics:
            doc_coverage = metrics["documentation_coverage"]
            if doc_coverage < 60:
                health_score -= 10
                recommendations.append("Improve documentation coverage")
        
        # Operational factors (20% weight)
        if "build_time" in metrics:
            build_time = metrics["build_time"]
            if build_time > 600:  # 10 minutes
                health_score -= 10
                recommendations.append("Optimize build performance")
        
        if "dependency_vulnerabilities" in metrics:
            dep_vulns = metrics["dependency_vulnerabilities"]
            if dep_vulns > 0:
                health_score -= min(dep_vulns * 3, 15)
                critical_issues.append(f"Dependency vulnerabilities: {dep_vulns}")
                recommendations.append("Update vulnerable dependencies")
        
        # Community factors (10% weight)
        if "github_open_issues" in metrics:
            open_issues = metrics["github_open_issues"]
            if open_issues > 50:
                health_score -= 5
                recommendations.append("Triage and close stale issues")
        
        # Ensure score is within bounds
        health_score = max(0, min(100, health_score))
        
        # Calculate grade
        if health_score >= 90:
            grade = "A"
        elif health_score >= 80:
            grade = "B"
        elif health_score >= 70:
            grade = "C"
        elif health_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        # Calculate trend if we have historical data
        trend = "stable"
        if len(self.metrics_history) >= 2:
            previous_metrics = self.metrics_history[-2]
            prev_score = self._calculate_historical_score(previous_metrics)
            
            if health_score > prev_score + 5:
                trend = "improving"
            elif health_score < prev_score - 5:
                trend = "declining"
        
        return ProjectHealth(
            score=health_score,
            grade=grade,
            trend=trend,
            critical_issues=critical_issues,
            recommendations=recommendations,
            last_updated=self.current_metrics.get("collection_timestamp", "")
        )
    
    def _calculate_historical_score(self, metrics_data: Dict) -> float:
        """Calculate health score for historical metrics data."""
        # Simplified version of the health calculation
        metrics = {m["name"]: m["value"] for m in metrics_data.get("metrics", [])}
        score = 100.0
        
        if "test_coverage" in metrics and metrics["test_coverage"] < 80:
            score -= 15
        if "security_vulnerabilities" in metrics:
            score -= min(metrics["security_vulnerabilities"] * 5, 25)
        if "technical_debt_ratio" in metrics and metrics["technical_debt_ratio"] > 2:
            score -= 10
        
        return max(0, min(100, score))
    
    def generate_html_dashboard(self, output_file: str = "project-dashboard.html") -> str:
        """Generate interactive HTML dashboard."""
        if not PLOTTING_AVAILABLE:
            self.logger.error("Plotting libraries not available for dashboard generation")
            return self._generate_text_dashboard(output_file.replace('.html', '.txt'))
        
        self.logger.info("Generating HTML dashboard...")
        
        # Calculate project health
        health = self.calculate_project_health()
        
        # Create dashboard with multiple charts
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                "Project Health Score", "Metrics Trend",
                "Code Quality Metrics", "Security & Dependencies",
                "Performance Metrics", "Repository Activity",
                "Technical Debt Breakdown", "Recommendations"
            ],
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "table"}]
            ],
            vertical_spacing=0.08
        )
        
        # Health score gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=health.score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Health Score (Grade: {health.grade})"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_health_color(health.score)},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Metrics trend
        if len(self.metrics_history) > 1:
            timestamps = [data["collection_timestamp"] for data in self.metrics_history]
            health_scores = [self._calculate_historical_score(data) for data in self.metrics_history]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=health_scores,
                    mode='lines+markers',
                    name='Health Score Trend',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
        
        # Code quality metrics
        if self.current_metrics:
            metrics = {m["name"]: m["value"] for m in self.current_metrics.get("metrics", [])}
            
            quality_metrics = {
                "Test Coverage": metrics.get("test_coverage", 0),
                "Documentation": metrics.get("documentation_coverage", 0),
                "Maintainability": metrics.get("maintainability_index", 0)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(quality_metrics.keys()),
                    y=list(quality_metrics.values()),
                    marker_color=['green', 'blue', 'orange'],
                    name='Quality Metrics'
                ),
                row=2, col=1
            )
            
            # Security metrics
            security_metrics = {
                "Vulnerabilities": metrics.get("security_vulnerabilities", 0),
                "Dep. Vulnerabilities": metrics.get("dependency_vulnerabilities", 0),
                "Open Issues": metrics.get("github_open_issues", 0)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(security_metrics.keys()),
                    y=list(security_metrics.values()),
                    marker_color=['red', 'orange', 'yellow'],
                    name='Security & Issues'
                ),
                row=2, col=2
            )
            
            # Performance metrics over time
            if len(self.metrics_history) > 1:
                build_times = []
                timestamps = []
                for data in self.metrics_history:
                    metrics_dict = {m["name"]: m["value"] for m in data.get("metrics", [])}
                    if "build_time" in metrics_dict:
                        build_times.append(metrics_dict["build_time"])
                        timestamps.append(data["collection_timestamp"])
                
                if build_times:
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=build_times,
                            mode='lines+markers',
                            name='Build Time (seconds)',
                            line=dict(color='purple')
                        ),
                        row=3, col=1
                    )
            
            # Repository activity
            activity_metrics = {
                "Stars": metrics.get("github_stars", 0),
                "Forks": metrics.get("github_forks", 0),
                "Contributors": metrics.get("github_contributors", 1)
            }
            
            fig.add_trace(
                go.Bar(
                    x=list(activity_metrics.keys()),
                    y=list(activity_metrics.values()),
                    marker_color=['gold', 'silver', 'bronze'],
                    name='Repository Activity'
                ),
                row=3, col=2
            )
            
            # Technical debt breakdown (pie chart)
            debt_categories = ["Comment Debt", "Complexity", "Duplication", "Large Files"]
            debt_values = [25, 35, 20, 20]  # Example values - would come from actual data
            
            fig.add_trace(
                go.Pie(
                    labels=debt_categories,
                    values=debt_values,
                    name="Technical Debt"
                ),
                row=4, col=1
            )
        
        # Recommendations table
        if health.recommendations:
            fig.add_trace(
                go.Table(
                    header=dict(values=['Priority', 'Recommendation']),
                    cells=dict(values=[
                        ['High'] * len(health.recommendations[:5]),
                        health.recommendations[:5]
                    ])
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"PQC IoT Retrofit Scanner - Project Dashboard (Health: {health.grade})",
            showlegend=False
        )
        
        # Add health trend indicator
        trend_color = {"improving": "green", "stable": "blue", "declining": "red"}.get(health.trend, "gray")
        fig.add_annotation(
            text=f"Trend: {health.trend.upper()}",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=16, color=trend_color),
            bgcolor="white",
            bordercolor=trend_color,
            borderwidth=2
        )
        
        # Save dashboard
        output_path = Path(output_file)
        pyo.plot(fig, filename=str(output_path), auto_open=False)
        
        # Add custom CSS and metadata
        self._enhance_html_dashboard(output_path, health)
        
        self.logger.info(f"Dashboard saved to {output_file}")
        return str(output_path)
    
    def _enhance_html_dashboard(self, html_file: Path, health: ProjectHealth):
        """Add custom styling and metadata to HTML dashboard."""
        try:
            content = html_file.read_text()
            
            # Add custom CSS and metadata
            custom_header = f"""
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>PQC IoT Retrofit Scanner - Project Dashboard</title>
                <meta name="description" content="Project health dashboard with metrics and insights">
                <meta name="last-updated" content="{health.last_updated}">
                <style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
                    .header-info {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 8px;
                    }}
                    .health-badge {{
                        display: inline-block;
                        padding: 8px 16px;
                        border-radius: 20px;
                        font-weight: bold;
                        background: {self._get_health_color(health.score)};
                        color: white;
                    }}
                    .critical-issue {{
                        background: #ffebee;
                        border-left: 4px solid #f44336;
                        padding: 10px;
                        margin: 5px 0;
                    }}
                    .recommendation {{
                        background: #e8f5e8;
                        border-left: 4px solid #4caf50;
                        padding: 10px;
                        margin: 5px 0;
                    }}
                </style>
            </head>
            """
            
            # Add dashboard info section
            info_section = f"""
            <div class="header-info">
                <h1>🔒 PQC IoT Retrofit Scanner - Project Dashboard</h1>
                <p><strong>Health Score:</strong> <span class="health-badge">{health.score:.1f}/100 (Grade: {health.grade})</span></p>
                <p><strong>Trend:</strong> {health.trend.upper()}</p>
                <p><strong>Last Updated:</strong> {health.last_updated}</p>
            </div>
            """
            
            if health.critical_issues:
                info_section += "<h3>🚨 Critical Issues</h3>"
                for issue in health.critical_issues:
                    info_section += f'<div class="critical-issue">{issue}</div>'
            
            if health.recommendations:
                info_section += "<h3>💡 Recommendations</h3>"
                for rec in health.recommendations[:5]:
                    info_section += f'<div class="recommendation">{rec}</div>'
            
            # Insert custom content
            content = content.replace('<head>', custom_header, 1)
            content = content.replace('<body>', f'<body>{info_section}', 1)
            
            html_file.write_text(content)
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance HTML dashboard: {e}")
    
    def _get_health_color(self, score: float) -> str:
        """Get color based on health score."""
        if score >= 90:
            return "#4caf50"  # Green
        elif score >= 80:
            return "#8bc34a"  # Light green
        elif score >= 70:
            return "#ffeb3b"  # Yellow
        elif score >= 60:
            return "#ff9800"  # Orange
        else:
            return "#f44336"  # Red
    
    def _generate_text_dashboard(self, output_file: str) -> str:
        """Generate text-based dashboard when plotting is not available."""
        self.logger.info("Generating text dashboard...")
        
        health = self.calculate_project_health()
        
        dashboard_content = f"""
PQC IoT Retrofit Scanner - Project Dashboard
==========================================

HEALTH OVERVIEW
--------------
Overall Score: {health.score:.1f}/100 (Grade: {health.grade})
Trend: {health.trend.upper()}
Last Updated: {health.last_updated}

CRITICAL ISSUES ({len(health.critical_issues)})
{'-' * 20}
"""
        
        if health.critical_issues:
            for i, issue in enumerate(health.critical_issues, 1):
                dashboard_content += f"{i}. {issue}\n"
        else:
            dashboard_content += "No critical issues found.\n"
        
        dashboard_content += f"""
RECOMMENDATIONS ({len(health.recommendations)})
{'-' * 20}
"""
        
        if health.recommendations:
            for i, rec in enumerate(health.recommendations, 1):
                dashboard_content += f"{i}. {rec}\n"
        else:
            dashboard_content += "No specific recommendations at this time.\n"
        
        # Add current metrics summary
        if self.current_metrics:
            dashboard_content += "\nCURRENT METRICS\n"
            dashboard_content += "-" * 20 + "\n"
            
            metrics = {m["name"]: (m["value"], m["unit"]) for m in self.current_metrics.get("metrics", [])}
            
            for name, (value, unit) in metrics.items():
                formatted_name = name.replace("_", " ").title()
                dashboard_content += f"{formatted_name}: {value} {unit}\n"
        
        # Add trend analysis if historical data available
        if len(self.metrics_history) > 1:
            dashboard_content += "\nTREND ANALYSIS\n"
            dashboard_content += "-" * 20 + "\n"
            dashboard_content += f"Historical data points: {len(self.metrics_history)}\n"
            
            # Calculate key metric trends
            current_metrics = {m["name"]: m["value"] for m in self.current_metrics.get("metrics", [])}
            previous_metrics = {m["name"]: m["value"] for m in self.metrics_history[-2].get("metrics", [])}
            
            for metric_name in ["test_coverage", "security_vulnerabilities", "code_complexity"]:
                if metric_name in current_metrics and metric_name in previous_metrics:
                    current = current_metrics[metric_name]
                    previous = previous_metrics[metric_name]
                    change = current - previous
                    trend_arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                    dashboard_content += f"{metric_name.replace('_', ' ').title()}: {current} {trend_arrow} ({change:+.1f})\n"
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(dashboard_content)
        
        self.logger.info(f"Text dashboard saved to {output_file}")
        return output_file
    
    def generate_slack_report(self) -> str:
        """Generate Slack-formatted health report."""
        health = self.calculate_project_health()
        
        # Emoji based on health grade
        grade_emoji = {
            "A": "🟢", "B": "🔵", "C": "🟡", "D": "🟠", "F": "🔴"
        }.get(health.grade, "⚪")
        
        trend_emoji = {
            "improving": "📈", "stable": "➡️", "declining": "📉"
        }.get(health.trend, "❓")
        
        slack_report = f"""
{grade_emoji} *PQC IoT Retrofit Scanner Health Report*

*Overall Health:* {health.score:.1f}/100 (Grade {health.grade}) {trend_emoji}
*Trend:* {health.trend.title()}
"""
        
        if health.critical_issues:
            slack_report += f"\n:warning: *Critical Issues ({len(health.critical_issues)}):*\n"
            for issue in health.critical_issues[:3]:  # Limit to 3 for Slack
                slack_report += f"• {issue}\n"
        
        if health.recommendations:
            slack_report += f"\n:bulb: *Top Recommendations:*\n"
            for rec in health.recommendations[:3]:  # Limit to 3 for Slack
                slack_report += f"• {rec}\n"
        
        # Add key metrics
        if self.current_metrics:
            metrics = {m["name"]: m["value"] for m in self.current_metrics.get("metrics", [])}
            
            key_metrics = {}
            if "test_coverage" in metrics:
                key_metrics["Test Coverage"] = f"{metrics['test_coverage']:.1f}%"
            if "security_vulnerabilities" in metrics:
                key_metrics["Security Issues"] = str(int(metrics["security_vulnerabilities"]))
            if "github_stars" in metrics:
                key_metrics["GitHub Stars"] = str(int(metrics["github_stars"]))
            
            if key_metrics:
                slack_report += "\n*Key Metrics:*\n"
                for name, value in key_metrics.items():
                    slack_report += f"• {name}: {value}\n"
        
        return slack_report
    
    def generate_json_summary(self, output_file: str = "dashboard-summary.json") -> str:
        """Generate JSON summary for API consumption."""
        health = self.calculate_project_health()
        
        summary = {
            "dashboard_generated": datetime.now(timezone.utc).isoformat(),
            "project_health": {
                "score": health.score,
                "grade": health.grade,
                "trend": health.trend,
                "last_updated": health.last_updated
            },
            "critical_issues": health.critical_issues,
            "recommendations": health.recommendations,
            "metrics_summary": {},
            "historical_data_points": len(self.metrics_history)
        }
        
        # Add current metrics summary
        if self.current_metrics:
            metrics = {m["name"]: {"value": m["value"], "unit": m["unit"]} 
                      for m in self.current_metrics.get("metrics", [])}
            summary["metrics_summary"] = metrics
        
        # Add trend analysis
        if len(self.metrics_history) > 1:
            trends = {}
            current_metrics = {m["name"]: m["value"] for m in self.current_metrics.get("metrics", [])}
            previous_metrics = {m["name"]: m["value"] for m in self.metrics_history[-2].get("metrics", [])}
            
            for metric_name in current_metrics:
                if metric_name in previous_metrics:
                    change = current_metrics[metric_name] - previous_metrics[metric_name]
                    trends[metric_name] = {
                        "current": current_metrics[metric_name],
                        "previous": previous_metrics[metric_name],
                        "change": change,
                        "direction": "up" if change > 0 else "down" if change < 0 else "stable"
                    }
            
            summary["trends"] = trends
        
        # Save JSON summary
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"JSON summary saved to {output_file}")
        return output_file


def main():
    """Main entry point for dashboard generation."""
    parser = argparse.ArgumentParser(description="Generate project metrics dashboard")
    parser.add_argument("--data-dir", default=".github", help="Directory containing metrics data")
    parser.add_argument("--output", default="project-dashboard.html", help="Output file path")
    parser.add_argument("--format", choices=["html", "text", "json", "slack"], default="html", help="Output format")
    parser.add_argument("--open", action="store_true", help="Open dashboard in browser")
    parser.add_argument("--metrics-files", nargs="*", help="Specific metrics files to load")
    
    args = parser.parse_args()
    
    try:
        dashboard = MetricsDashboard(data_dir=args.data_dir)
        
        if not dashboard.load_metrics_data(args.metrics_files):
            print("No metrics data found. Run metrics collection first.")
            return 1
        
        if args.format == "html":
            output_file = dashboard.generate_html_dashboard(args.output)
            if args.open:
                webbrowser.open(f"file://{Path(output_file).absolute()}")
        elif args.format == "text":
            output_file = dashboard._generate_text_dashboard(args.output.replace('.html', '.txt'))
        elif args.format == "json":
            output_file = dashboard.generate_json_summary(args.output.replace('.html', '.json'))
        elif args.format == "slack":
            slack_report = dashboard.generate_slack_report()
            print(slack_report)
            return 0
        
        # Calculate and display health summary
        health = dashboard.calculate_project_health()
        print(f"\nProject Health Summary:")
        print(f"  Score: {health.score:.1f}/100 (Grade: {health.grade})")
        print(f"  Trend: {health.trend}")
        print(f"  Critical Issues: {len(health.critical_issues)}")
        print(f"  Recommendations: {len(health.recommendations)}")
        
        if args.format != "slack":
            print(f"  Dashboard saved to: {output_file}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Dashboard generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())