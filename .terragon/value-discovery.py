#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Continuously discovers, scores, and executes highest-value work items.
"""

import json
import subprocess
import re
import os
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod


@dataclass
class WorkItem:
    """Represents a discovered work item with value scoring."""
    id: str
    title: str
    description: str
    category: str
    type: str
    estimated_effort: float  # hours
    confidence: float  # 0-1
    impact: float  # 0-10
    ease: float  # 0-10
    wsjf: float = 0.0
    ice: float = 0.0
    technical_debt: float = 0.0
    composite_score: float = 0.0
    discovered_at: str = ""
    source: str = ""
    files_affected: List[str] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.files_affected is None:
            self.files_affected = []
        if self.dependencies is None:
            self.dependencies = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()


class ValueDiscoverySource(ABC):
    """Abstract base class for value discovery sources."""
    
    @abstractmethod
    def discover(self) -> List[WorkItem]:
        """Discover work items from this source."""
        pass


class GitHistorySource(ValueDiscoverySource):
    """Discovers work items from git history analysis."""
    
    def discover(self) -> List[WorkItem]:
        items = []
        
        # Find TODO/FIXME comments
        result = subprocess.run([
            'git', 'grep', '-n', '-i', '-E', 
            '(TODO|FIXME|HACK|XXX|DEPRECATED):'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.strip():
                    parts = line.split(':', 3)
                    if len(parts) >= 3:
                        file_path, line_num, content = parts[0], parts[1], parts[2]
                        
                        # Extract marker type
                        marker_match = re.search(r'(TODO|FIXME|HACK|XXX|DEPRECATED)', 
                                               content, re.IGNORECASE)
                        if marker_match:
                            marker = marker_match.group(1).upper()
                            
                            # Score based on marker type
                            effort_map = {
                                'TODO': 2.0, 'FIXME': 4.0, 'HACK': 6.0,
                                'XXX': 3.0, 'DEPRECATED': 8.0
                            }
                            impact_map = {
                                'TODO': 4.0, 'FIXME': 7.0, 'HACK': 8.0,
                                'XXX': 6.0, 'DEPRECATED': 9.0
                            }
                            
                            items.append(WorkItem(
                                id=f"git-{marker.lower()}-{len(items)+1}",
                                title=f"Address {marker} in {file_path}:{line_num}",
                                description=content.strip(),
                                category="technical_debt",
                                type=marker.lower(),
                                estimated_effort=effort_map.get(marker, 3.0),
                                confidence=0.8,
                                impact=impact_map.get(marker, 5.0),
                                ease=6.0,
                                source="git_history",
                                files_affected=[file_path]
                            ))
        
        return items


class StaticAnalysisSource(ValueDiscoverySource):
    """Discovers work items from static analysis tools."""
    
    def discover(self) -> List[WorkItem]:
        items = []
        
        # Run ruff for Python issues
        if Path("pyproject.toml").exists():
            result = subprocess.run([
                'python3', '-m', 'ruff', 'check', '--output-format=json', 'src/'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                try:
                    ruff_issues = json.loads(result.stdout)
                    for issue in ruff_issues:
                        severity = issue.get('level', 'warning')
                        
                        # Score based on severity
                        effort_map = {'error': 3.0, 'warning': 1.5}
                        impact_map = {'error': 8.0, 'warning': 5.0}
                        
                        items.append(WorkItem(
                            id=f"ruff-{issue['code']}-{len(items)+1}",
                            title=f"Fix {severity}: {issue['message']}",
                            description=f"{issue['message']} in {issue['filename']}:{issue['location']['row']}",
                            category="code_quality",
                            type="lint_issue",
                            estimated_effort=effort_map.get(severity, 2.0),
                            confidence=0.9,
                            impact=impact_map.get(severity, 6.0),
                            ease=7.0,
                            source="static_analysis",
                            files_affected=[issue['filename']]
                        ))
                except json.JSONDecodeError:
                    pass
        
        return items


class SecurityScanSource(ValueDiscoverySource):
    """Discovers security vulnerabilities and compliance gaps."""
    
    def discover(self) -> List[WorkItem]:
        items = []
        
        # Run safety check for Python dependencies
        result = subprocess.run([
            'python3', '-m', 'safety', 'check', '--json'
        ], capture_output=True, text=True)
        
        if result.returncode != 0 and result.stdout:
            try:
                safety_issues = json.loads(result.stdout)
                for issue in safety_issues:
                    severity = issue.get('severity', 'medium')
                    
                    # High priority for security issues
                    effort_map = {'high': 4.0, 'medium': 2.0, 'low': 1.0}
                    impact_map = {'high': 9.0, 'medium': 7.0, 'low': 5.0}
                    
                    items.append(WorkItem(
                        id=f"security-{issue['id']}-{len(items)+1}",
                        title=f"Security: Update {issue['package_name']}",
                        description=f"Vulnerability {issue['id']}: {issue['advisory']}",
                        category="security",
                        type="vulnerability",
                        estimated_effort=effort_map.get(severity, 2.0),
                        confidence=0.95,
                        impact=impact_map.get(severity, 7.0),
                        ease=8.0,  # Usually just dependency updates
                        source="security_scan"
                    ))
            except json.JSONDecodeError:
                pass
        
        return items


class DependencyUpdateSource(ValueDiscoverySource):
    """Discovers outdated dependencies."""
    
    def discover(self) -> List[WorkItem]:
        items = []
        
        # Check for outdated pip packages
        result = subprocess.run([
            'python3', '-m', 'pip', 'list', '--outdated', '--format=json'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            try:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:10]:  # Limit to top 10
                    items.append(WorkItem(
                        id=f"dep-update-{pkg['name']}-{len(items)+1}",
                        title=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        description=f"Dependency update: {pkg['name']} {pkg['version']} → {pkg['latest_version']}",
                        category="maintenance",
                        type="dependency_update",
                        estimated_effort=1.0,
                        confidence=0.7,
                        impact=4.0,
                        ease=9.0,
                        source="dependency_scan"
                    ))
            except json.JSONDecodeError:
                pass
        
        return items


class ScoringEngine:
    """Advanced scoring engine implementing WSJF + ICE + Technical Debt."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = config['scoring']['weights']['maturing']
        self.thresholds = config['scoring']['thresholds']
        self.multipliers = config['scoring']['multipliers']
    
    def calculate_wsjf(self, item: WorkItem) -> float:
        """Calculate Weighted Shortest Job First score."""
        # Cost of Delay components
        user_business_value = item.impact * 0.4
        time_criticality = self._get_time_criticality(item) * 0.3
        risk_reduction = self._get_risk_reduction(item) * 0.2
        opportunity_enablement = self._get_opportunity_enablement(item) * 0.1
        
        cost_of_delay = (user_business_value + time_criticality + 
                        risk_reduction + opportunity_enablement)
        
        # Job size (effort)
        job_size = max(item.estimated_effort, 0.5)  # Avoid division by zero
        
        return cost_of_delay / job_size
    
    def calculate_ice(self, item: WorkItem) -> float:
        """Calculate Impact, Confidence, Ease score."""
        return item.impact * item.confidence * item.ease
    
    def calculate_technical_debt(self, item: WorkItem) -> float:
        """Calculate technical debt score."""
        if item.category != "technical_debt":
            return 0.0
        
        # Base debt impact
        debt_impact = item.impact * 5
        
        # Interest rate (future cost if not addressed)
        debt_interest = self._calculate_debt_interest(item)
        
        # Hotspot multiplier
        hotspot_multiplier = self._get_hotspot_multiplier(item)
        
        return (debt_impact + debt_interest) * hotspot_multiplier
    
    def calculate_composite_score(self, item: WorkItem) -> float:
        """Calculate final composite score."""
        item.wsjf = self.calculate_wsjf(item)
        item.ice = self.calculate_ice(item)
        item.technical_debt = self.calculate_technical_debt(item)
        
        # Normalize scores (simple min-max for now)
        norm_wsjf = min(item.wsjf / 50.0, 1.0)
        norm_ice = min(item.ice / 1000.0, 1.0)
        norm_debt = min(item.technical_debt / 100.0, 1.0)
        
        # Apply weights
        composite = (
            self.weights['wsjf'] * norm_wsjf +
            self.weights['ice'] * norm_ice +
            self.weights['technicalDebt'] * norm_debt +
            self.weights['security'] * (1.0 if item.category == 'security' else 0.0)
        )
        
        # Apply category-specific boosts
        if item.category == 'security':
            composite *= self.thresholds['securityBoost']
        elif item.type == 'vulnerability':
            composite *= self.thresholds['securityBoost']
        
        item.composite_score = composite * 100  # Scale to 0-100
        return item.composite_score
    
    def _get_time_criticality(self, item: WorkItem) -> float:
        """Calculate time criticality factor."""
        if item.category == 'security':
            return 9.0
        elif item.type == 'deprecated':
            return 7.0
        elif item.category == 'compliance':
            return 8.0
        return 5.0
    
    def _get_risk_reduction(self, item: WorkItem) -> float:
        """Calculate risk reduction factor."""
        risk_map = {
            'security': 9.0,
            'technical_debt': 6.0,
            'performance': 5.0,
            'maintenance': 3.0
        }
        return risk_map.get(item.category, 4.0)
    
    def _get_opportunity_enablement(self, item: WorkItem) -> float:
        """Calculate opportunity enablement factor."""
        if item.category in ['infrastructure', 'tooling']:
            return 7.0
        elif item.category == 'performance':
            return 6.0
        return 4.0
    
    def _calculate_debt_interest(self, item: WorkItem) -> float:
        """Calculate debt interest (future cost)."""
        # Simple model: debt grows over time
        days_old = 30  # Assume 30 days for now
        return item.impact * (days_old / 30.0) * 2.0
    
    def _get_hotspot_multiplier(self, item: WorkItem) -> float:
        """Calculate hotspot multiplier based on file churn."""
        if not item.files_affected:
            return 1.0
        
        # Simple heuristic: files in src/ are likely hotspots
        hotspot_multiplier = 1.0
        for file_path in item.files_affected:
            if 'src/' in file_path:
                hotspot_multiplier = max(hotspot_multiplier, 2.0)
            if any(core in file_path for core in ['cli.py', 'scanner.py', '__init__.py']):
                hotspot_multiplier = max(hotspot_multiplier, 3.0)
        
        return hotspot_multiplier


class ValueDiscoveryEngine:
    """Main engine for autonomous value discovery and execution."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scoring_engine = ScoringEngine(self.config)
        self.sources = [
            GitHistorySource(),
            StaticAnalysisSource(),
            SecurityScanSource(),
            DependencyUpdateSource()
        ]
        
        self.metrics_file = Path(".terragon/value-metrics.json")
        self.backlog_file = Path("BACKLOG.md")
    
    def discover_work_items(self) -> List[WorkItem]:
        """Discover all work items from configured sources."""
        all_items = []
        
        for source in self.sources:
            try:
                items = source.discover()
                all_items.extend(items)
                print(f"Discovered {len(items)} items from {source.__class__.__name__}")
            except Exception as e:
                print(f"Error in {source.__class__.__name__}: {e}")
        
        return all_items
    
    def score_and_prioritize(self, items: List[WorkItem]) -> List[WorkItem]:
        """Score all items and sort by composite score."""
        for item in items:
            self.scoring_engine.calculate_composite_score(item)
        
        # Sort by composite score (descending)
        return sorted(items, key=lambda x: x.composite_score, reverse=True)
    
    def select_next_best_value(self, items: List[WorkItem]) -> Optional[WorkItem]:
        """Select the next best value item for execution."""
        for item in items:
            # Skip if below minimum score
            if item.composite_score < self.config['scoring']['thresholds']['minScore']:
                continue
            
            # Skip if dependencies not met
            if item.dependencies and not self._are_dependencies_met(item.dependencies):
                continue
            
            # Skip if risk too high
            if self._assess_risk(item) > self.config['scoring']['thresholds']['maxRisk']:
                continue
            
            return item
        
        return None
    
    def update_backlog(self, items: List[WorkItem]):
        """Update BACKLOG.md with current items and metrics."""
        now = datetime.now()
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now.isoformat()}
Repository: {self.config['repository']['name']}
Maturity Level: {self.config['repository']['maturity_level'].title()}

## 🎯 Next Best Value Item
"""
        
        if items:
            next_item = items[0]
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf:.1f} | **ICE**: {next_item.ice:.0f} | **Tech Debt**: {next_item.technical_debt:.1f}
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Estimated Effort**: {next_item.estimated_effort:.1f} hours
- **Source**: {next_item.source.replace('_', ' ').title()}

"""
        else:
            content += "No high-value items currently identified.\n\n"
        
        content += f"""## 📋 Top {min(len(items), 20)} Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours | Source |
|------|-----|--------|---------|----------|------------|---------|
"""
        
        for i, item in enumerate(items[:20], 1):
            title_short = item.title[:50] + "..." if len(item.title) > 50 else item.title
            content += f"| {i} | {item.id} | {title_short} | {item.composite_score:.1f} | {item.category.replace('_', ' ').title()} | {item.estimated_effort:.1f} | {item.source.replace('_', ' ').title()} |\n"
        
        content += f"""

## 📈 Discovery Statistics
- **Total Items Discovered**: {len(items)}
- **Security Items**: {len([i for i in items if i.category == 'security'])}
- **Technical Debt Items**: {len([i for i in items if i.category == 'technical_debt'])}
- **Maintenance Items**: {len([i for i in items if i.category == 'maintenance'])}
- **Average Score**: {sum(i.composite_score for i in items) / len(items) if items else 0:.1f}
- **High-Value Items (>50)**: {len([i for i in items if i.composite_score > 50])}

## 🔄 Discovery Sources Breakdown
"""
        
        source_counts = {}
        for item in items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
        
        for source, count in source_counts.items():
            percentage = (count / len(items)) * 100 if items else 0
            content += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        content += f"""

## ⚙️ Configuration
- **Scoring Model**: WSJF + ICE + Technical Debt
- **Repository Maturity**: {self.config['repository']['maturity_level'].title()}
- **Min Score Threshold**: {self.config['scoring']['thresholds']['minScore']}
- **Security Boost**: {self.config['scoring']['thresholds']['securityBoost']}x
- **Max Concurrent Tasks**: {self.config['execution']['maxConcurrentTasks']}

---
*Generated by Terragon Autonomous SDLC Value Discovery Engine*
"""
        
        self.backlog_file.write_text(content)
        print(f"Updated {self.backlog_file} with {len(items)} items")
    
    def save_metrics(self, items: List[WorkItem], execution_time: float):
        """Save execution metrics for learning and optimization."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "total_items_discovered": len(items),
            "items_by_category": {},
            "items_by_source": {},
            "score_distribution": {
                "min": min((i.composite_score for i in items), default=0),
                "max": max((i.composite_score for i in items), default=0),
                "avg": sum(i.composite_score for i in items) / len(items) if items else 0,
                "high_value_count": len([i for i in items if i.composite_score > 50])
            },
            "top_10_items": [asdict(item) for item in items[:10]]
        }
        
        # Calculate category and source breakdowns
        for item in items:
            metrics["items_by_category"][item.category] = metrics["items_by_category"].get(item.category, 0) + 1
            metrics["items_by_source"][item.source] = metrics["items_by_source"].get(item.source, 0) + 1
        
        # Load existing metrics if available
        existing_metrics = []
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except json.JSONDecodeError:
                existing_metrics = []
        
        # Add new metrics
        existing_metrics.append(metrics)
        
        # Keep only last 100 executions
        existing_metrics = existing_metrics[-100:]
        
        # Save updated metrics
        with open(self.metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2)
    
    def _are_dependencies_met(self, dependencies: List[str]) -> bool:
        """Check if dependencies are met (placeholder)."""
        # For now, assume all dependencies are met
        return True
    
    def _assess_risk(self, item: WorkItem) -> float:
        """Assess risk of executing this item."""
        # Simple risk assessment
        base_risk = 0.1
        
        if item.confidence < 0.5:
            base_risk += 0.3
        
        if item.category == 'infrastructure':
            base_risk += 0.2
        
        if 'critical' in item.description.lower():
            base_risk += 0.3
        
        return min(base_risk, 1.0)
    
    def run_discovery_cycle(self):
        """Run a complete discovery and prioritization cycle."""
        start_time = datetime.now()
        
        print("🔍 Starting value discovery cycle...")
        
        # Discover work items
        items = self.discover_work_items()
        
        # Score and prioritize
        prioritized_items = self.score_and_prioritize(items)
        
        # Update backlog
        self.update_backlog(prioritized_items)
        
        # Save metrics
        execution_time = (datetime.now() - start_time).total_seconds()
        self.save_metrics(prioritized_items, execution_time)
        
        print(f"✅ Discovery cycle complete: {len(items)} items discovered in {execution_time:.1f}s")
        
        # Show next best value item
        next_item = self.select_next_best_value(prioritized_items)
        if next_item:
            print(f"🎯 Next best value: [{next_item.id}] {next_item.title} (Score: {next_item.composite_score:.1f})")
        else:
            print("ℹ️ No high-value items above threshold")
        
        return prioritized_items


def main():
    """Main entry point for value discovery."""
    engine = ValueDiscoveryEngine()
    items = engine.run_discovery_cycle()
    
    # Print summary
    if items:
        print(f"\n📊 Summary:")
        print(f"Total items: {len(items)}")
        print(f"Security items: {len([i for i in items if i.category == 'security'])}")
        print(f"Technical debt: {len([i for i in items if i.category == 'technical_debt'])}")
        print(f"High-value items (>50): {len([i for i in items if i.composite_score > 50])}")


if __name__ == "__main__":
    main()