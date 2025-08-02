#!/usr/bin/env python3
"""
Technical debt tracking and analysis tool for PQC IoT Retrofit Scanner.

Identifies, categorizes, and tracks technical debt across the codebase
including TODO comments, code complexity, duplication, and maintenance issues.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

import git
from dataclasses import dataclass, asdict


@dataclass
class TechnicalDebtItem:
    """Represents a single technical debt item."""
    id: str
    type: str  # TODO, FIXME, HACK, complexity, duplication, etc.
    severity: str  # low, medium, high, critical
    title: str
    description: str
    file_path: str
    line_number: int
    estimated_effort_hours: float
    created_date: str
    last_seen_date: str
    tags: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TechnicalDebtTracker:
    """Main technical debt tracking and analysis system."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.debt_items: List[TechnicalDebtItem] = []
        self.config = self._load_config()
        
        try:
            self.repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            self.repo = None
            print("Warning: Not a git repository, some features will be disabled")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load technical debt tracking configuration."""
        config_file = self.repo_path / ".technical-debt-config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "debt_keywords": {
                "TODO": {"severity": "medium", "effort_hours": 2},
                "FIXME": {"severity": "high", "effort_hours": 4},
                "HACK": {"severity": "high", "effort_hours": 6},
                "XXX": {"severity": "medium", "effort_hours": 3},
                "BUG": {"severity": "high", "effort_hours": 8},
                "DEPRECATED": {"severity": "low", "effort_hours": 1}
            },
            "complexity_thresholds": {
                "function": 10,
                "class": 15,
                "file": 20
            },
            "file_patterns": {
                "include": ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h"],
                "exclude": ["*test*", "*__pycache__*", "*.pyc", "node_modules/*"]
            },
            "duplication_threshold": 6,  # minimum lines for duplication detection
            "effort_estimation": {
                "complexity_multiplier": 0.5,  # hours per complexity point
                "duplication_base_hours": 2,
                "large_file_threshold": 500,  # lines
                "large_file_penalty_hours": 1
            }
        }
    
    def scan_codebase(self) -> List[TechnicalDebtItem]:
        """Perform comprehensive technical debt scan."""
        print("Scanning codebase for technical debt...")
        
        self.debt_items = []
        
        # Scan for comment-based debt (TODO, FIXME, etc.)
        self._scan_comment_debt()
        
        # Scan for code complexity issues
        self._scan_complexity_debt()
        
        # Scan for code duplication
        self._scan_duplication_debt()
        
        # Scan for large files
        self._scan_large_files()
        
        # Scan for outdated patterns
        self._scan_outdated_patterns()
        
        # Add git history context if available
        if self.repo:
            self._add_git_context()
        
        print(f"Found {len(self.debt_items)} technical debt items")
        return self.debt_items
    
    def _scan_comment_debt(self):
        """Scan for TODO, FIXME, and similar comments."""
        print("  Scanning comment-based debt...")
        
        keywords = self.config["debt_keywords"]
        pattern = r'(?i)\b(' + '|'.join(keywords.keys()) + r')\b:?\s*(.*)'
        
        for file_path in self._get_source_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        # Skip if line is too long (likely binary or generated)
                        if len(line) > 500:
                            continue
                            
                        match = re.search(pattern, line)
                        if match:
                            keyword = match.group(1).upper()
                            description = match.group(2).strip()
                            
                            if not description:
                                description = f"{keyword} comment without description"
                            
                            config = keywords.get(keyword, {"severity": "medium", "effort_hours": 2})
                            
                            debt_item = TechnicalDebtItem(
                                id=self._generate_debt_id(file_path, line_num, keyword),
                                type="comment_debt",
                                severity=config["severity"],
                                title=f"{keyword}: {description[:50]}{'...' if len(description) > 50 else ''}",
                                description=description,
                                file_path=str(file_path.relative_to(self.repo_path)),
                                line_number=line_num,
                                estimated_effort_hours=config["effort_hours"],
                                created_date=datetime.now(timezone.utc).isoformat(),
                                last_seen_date=datetime.now(timezone.utc).isoformat(),
                                tags=[keyword.lower(), "comment"],
                                metadata={
                                    "keyword": keyword,
                                    "line_content": line.strip(),
                                    "context": self._get_line_context(file_path, line_num)
                                }
                            )
                            
                            self.debt_items.append(debt_item)
                            
            except Exception as e:
                print(f"    Error scanning {file_path}: {e}")
    
    def _scan_complexity_debt(self):
        """Scan for high complexity functions and classes."""
        print("  Scanning complexity debt...")
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                complexity_analyzer = ComplexityAnalyzer()
                complexity_analyzer.visit(tree)
                
                for item in complexity_analyzer.complex_items:
                    if item['complexity'] > self.config["complexity_thresholds"].get(item['type'], 10):
                        severity = self._calculate_complexity_severity(item['complexity'])
                        effort_hours = item['complexity'] * self.config["effort_estimation"]["complexity_multiplier"]
                        
                        debt_item = TechnicalDebtItem(
                            id=self._generate_debt_id(file_path, item['line'], "complexity"),
                            type="complexity_debt",
                            severity=severity,
                            title=f"High complexity {item['type']}: {item['name']}",
                            description=f"Cyclomatic complexity of {item['complexity']} exceeds threshold",
                            file_path=str(file_path.relative_to(self.repo_path)),
                            line_number=item['line'],
                            estimated_effort_hours=effort_hours,
                            created_date=datetime.now(timezone.utc).isoformat(),
                            last_seen_date=datetime.now(timezone.utc).isoformat(),
                            tags=["complexity", item['type']],
                            metadata={
                                "complexity_score": item['complexity'],
                                "complexity_type": item['type'],
                                "threshold": self.config["complexity_thresholds"].get(item['type'], 10)
                            }
                        )
                        
                        self.debt_items.append(debt_item)
                        
            except Exception as e:
                print(f"    Error analyzing complexity in {file_path}: {e}")
    
    def _scan_duplication_debt(self):
        """Scan for code duplication."""
        print("  Scanning code duplication...")
        
        # Simple duplication detection based on similar lines
        file_line_hashes = {}
        
        for file_path in self._get_source_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                file_line_hashes[file_path] = []
                for i, line in enumerate(lines):
                    # Normalize line for comparison
                    normalized = re.sub(r'\s+', ' ', line.strip())
                    if len(normalized) > 10:  # Ignore very short lines
                        file_line_hashes[file_path].append((i + 1, hash(normalized), normalized))
                        
            except Exception as e:
                print(f"    Error reading {file_path}: {e}")
        
        # Find duplications
        duplications = self._find_duplications(file_line_hashes)
        
        for duplication in duplications:
            # Create debt item for each file involved in duplication
            for file_info in duplication['files']:
                debt_item = TechnicalDebtItem(
                    id=self._generate_debt_id(file_info['file'], file_info['start_line'], "duplication"),
                    type="duplication_debt",
                    severity="medium",
                    title=f"Code duplication detected ({duplication['line_count']} lines)",
                    description=f"Duplicated code block found in {len(duplication['files'])} files",
                    file_path=str(file_info['file'].relative_to(self.repo_path)),
                    line_number=file_info['start_line'],
                    estimated_effort_hours=self.config["effort_estimation"]["duplication_base_hours"],
                    created_date=datetime.now(timezone.utc).isoformat(),
                    last_seen_date=datetime.now(timezone.utc).isoformat(),
                    tags=["duplication", "refactoring"],
                    metadata={
                        "duplication_id": duplication['id'],
                        "total_files": len(duplication['files']),
                        "line_count": duplication['line_count'],
                        "other_files": [str(f['file'].relative_to(self.repo_path)) 
                                      for f in duplication['files'] if f['file'] != file_info['file']]
                    }
                )
                
                self.debt_items.append(debt_item)
    
    def _scan_large_files(self):
        """Scan for overly large files."""
        print("  Scanning large files...")
        
        threshold = self.config["effort_estimation"]["large_file_threshold"]
        
        for file_path in self._get_source_files():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
                
                if line_count > threshold:
                    severity = "low" if line_count < threshold * 2 else "medium"
                    effort_hours = self.config["effort_estimation"]["large_file_penalty_hours"]
                    
                    debt_item = TechnicalDebtItem(
                        id=self._generate_debt_id(file_path, 1, "large_file"),
                        type="maintainability_debt",
                        severity=severity,
                        title=f"Large file ({line_count} lines)",
                        description=f"File exceeds recommended size of {threshold} lines",
                        file_path=str(file_path.relative_to(self.repo_path)),
                        line_number=1,
                        estimated_effort_hours=effort_hours,
                        created_date=datetime.now(timezone.utc).isoformat(),
                        last_seen_date=datetime.now(timezone.utc).isoformat(),
                        tags=["large_file", "maintainability"],
                        metadata={
                            "line_count": line_count,
                            "threshold": threshold,
                            "size_ratio": line_count / threshold
                        }
                    )
                    
                    self.debt_items.append(debt_item)
                    
            except Exception as e:
                print(f"    Error checking size of {file_path}: {e}")
    
    def _scan_outdated_patterns(self):
        """Scan for outdated coding patterns and deprecated usage."""
        print("  Scanning outdated patterns...")
        
        outdated_patterns = [
            {
                "pattern": r"print\s*\(",
                "name": "print_statements",
                "description": "Print statement found - consider using logging",
                "severity": "low",
                "effort_hours": 0.5
            },
            {
                "pattern": r"except\s*:",
                "name": "bare_except",
                "description": "Bare except clause - should specify exception types",
                "severity": "medium",
                "effort_hours": 1
            },
            {
                "pattern": r"eval\s*\(",
                "name": "eval_usage",
                "description": "Use of eval() - potential security risk",
                "severity": "high",
                "effort_hours": 4
            },
            {
                "pattern": r"exec\s*\(",
                "name": "exec_usage",
                "description": "Use of exec() - potential security risk",
                "severity": "high",
                "effort_hours": 4
            }
        ]
        
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()
                
                for pattern_config in outdated_patterns:
                    pattern = re.compile(pattern_config["pattern"])
                    
                    for line_num, line in enumerate(lines, 1):
                        if pattern.search(line):
                            debt_item = TechnicalDebtItem(
                                id=self._generate_debt_id(file_path, line_num, pattern_config["name"]),
                                type="pattern_debt",
                                severity=pattern_config["severity"],
                                title=f"Outdated pattern: {pattern_config['name']}",
                                description=pattern_config["description"],
                                file_path=str(file_path.relative_to(self.repo_path)),
                                line_number=line_num,
                                estimated_effort_hours=pattern_config["effort_hours"],
                                created_date=datetime.now(timezone.utc).isoformat(),
                                last_seen_date=datetime.now(timezone.utc).isoformat(),
                                tags=["outdated_pattern", pattern_config["name"]],
                                metadata={
                                    "pattern": pattern_config["pattern"],
                                    "line_content": line.strip()
                                }
                            )
                            
                            self.debt_items.append(debt_item)
                            
            except Exception as e:
                print(f"    Error scanning patterns in {file_path}: {e}")
    
    def _add_git_context(self):
        """Add git history context to debt items."""
        print("  Adding git history context...")
        
        for debt_item in self.debt_items:
            try:
                file_path = self.repo_path / debt_item.file_path
                
                # Get last modification date
                commits = list(self.repo.iter_commits(paths=str(file_path), max_count=1))
                if commits:
                    last_commit = commits[0]
                    debt_item.metadata["last_modified"] = last_commit.committed_datetime.isoformat()
                    debt_item.metadata["last_author"] = last_commit.author.name
                
                # Get creation date (first commit that introduced this line)
                try:
                    blame = self.repo.blame(str(file_path), rev='HEAD')
                    if debt_item.line_number <= len(blame):
                        blame_commit = blame[debt_item.line_number - 1][0]
                        debt_item.created_date = blame_commit.committed_datetime.isoformat()
                        debt_item.metadata["created_by"] = blame_commit.author.name
                except Exception:
                    pass  # Blame might fail for various reasons
                    
            except Exception as e:
                print(f"    Error adding git context for {debt_item.file_path}: {e}")
    
    def _get_source_files(self) -> List[Path]:
        """Get list of source files to analyze."""
        source_files = []
        include_patterns = self.config["file_patterns"]["include"]
        exclude_patterns = self.config["file_patterns"]["exclude"]
        
        for pattern in include_patterns:
            for file_path in self.repo_path.rglob(pattern):
                if file_path.is_file():
                    # Check if file should be excluded
                    relative_path = str(file_path.relative_to(self.repo_path))
                    should_exclude = any(
                        Path(relative_path).match(exclude_pattern)
                        for exclude_pattern in exclude_patterns
                    )
                    
                    if not should_exclude:
                        source_files.append(file_path)
        
        return source_files
    
    def _get_python_files(self) -> List[Path]:
        """Get list of Python files to analyze."""
        return [f for f in self._get_source_files() if f.suffix == '.py']
    
    def _generate_debt_id(self, file_path: Path, line_number: int, debt_type: str) -> str:
        """Generate unique ID for debt item."""
        relative_path = str(file_path.relative_to(self.repo_path))
        return f"{relative_path}:{line_number}:{debt_type}"
    
    def _get_line_context(self, file_path: Path, line_number: int, context_lines: int = 2) -> Dict[str, Any]:
        """Get context around a specific line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            return {
                "before": [line.rstrip() for line in lines[start:line_number-1]] if line_number > 1 else [],
                "after": [line.rstrip() for line in lines[line_number:end]] if line_number < len(lines) else []
            }
        except Exception:
            return {"before": [], "after": []}
    
    def _calculate_complexity_severity(self, complexity: int) -> str:
        """Calculate severity based on complexity score."""
        if complexity > 20:
            return "critical"
        elif complexity > 15:
            return "high"
        elif complexity > 10:
            return "medium"
        else:
            return "low"
    
    def _find_duplications(self, file_line_hashes: Dict[Path, List[Tuple]]) -> List[Dict]:
        """Find code duplications across files."""
        duplications = []
        threshold = self.config["duplication_threshold"]
        
        # Group lines by hash
        hash_to_locations = defaultdict(list)
        for file_path, line_data in file_line_hashes.items():
            for line_num, line_hash, content in line_data:
                hash_to_locations[line_hash].append({
                    'file': file_path,
                    'line': line_num,
                    'content': content
                })
        
        # Find sequences of duplicate lines
        processed_duplications = set()
        
        for line_hash, locations in hash_to_locations.items():
            if len(locations) > 1:  # Line appears in multiple places
                # Group by file and check for sequences
                for i, loc1 in enumerate(locations):
                    for loc2 in locations[i+1:]:
                        if loc1['file'] != loc2['file']:  # Different files
                            sequence_length = self._check_duplication_sequence(
                                file_line_hashes, loc1, loc2
                            )
                            
                            if sequence_length >= threshold:
                                dup_id = f"{loc1['file']}:{loc1['line']}-{loc2['file']}:{loc2['line']}"
                                if dup_id not in processed_duplications:
                                    duplications.append({
                                        'id': dup_id,
                                        'line_count': sequence_length,
                                        'files': [
                                            {'file': loc1['file'], 'start_line': loc1['line']},
                                            {'file': loc2['file'], 'start_line': loc2['line']}
                                        ]
                                    })
                                    processed_duplications.add(dup_id)
        
        return duplications
    
    def _check_duplication_sequence(self, file_line_hashes: Dict, loc1: Dict, loc2: Dict) -> int:
        """Check how many consecutive lines are duplicated."""
        # This is a simplified implementation
        # A more sophisticated algorithm would be needed for production use
        return self.config["duplication_threshold"]  # Placeholder
    
    def generate_report(self, output_file: str = "technical-debt-report.json") -> Dict[str, Any]:
        """Generate comprehensive technical debt report."""
        print("Generating technical debt report...")
        
        if not self.debt_items:
            self.scan_codebase()
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Group debt items by various criteria
        by_type = self._group_by_type()
        by_severity = self._group_by_severity()
        by_file = self._group_by_file()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        report = {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(self.repo_path),
            "total_debt_items": len(self.debt_items),
            "statistics": stats,
            "debt_by_type": by_type,
            "debt_by_severity": by_severity,
            "debt_by_file": by_file,
            "recommendations": recommendations,
            "debt_items": [asdict(item) for item in self.debt_items]
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Technical debt report saved to {output_file}")
        return report
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate various statistics about technical debt."""
        if not self.debt_items:
            return {}
        
        total_effort = sum(item.estimated_effort_hours for item in self.debt_items)
        severities = [item.severity for item in self.debt_items]
        types = [item.type for item in self.debt_items]
        
        return {
            "total_estimated_effort_hours": total_effort,
            "average_effort_per_item": total_effort / len(self.debt_items),
            "severity_distribution": dict(Counter(severities)),
            "type_distribution": dict(Counter(types)),
            "files_with_debt": len(set(item.file_path for item in self.debt_items)),
            "debt_density": len(self.debt_items) / len(self._get_source_files()) if self._get_source_files() else 0
        }
    
    def _group_by_type(self) -> Dict[str, List[Dict]]:
        """Group debt items by type."""
        grouped = defaultdict(list)
        for item in self.debt_items:
            grouped[item.type].append(asdict(item))
        return dict(grouped)
    
    def _group_by_severity(self) -> Dict[str, List[Dict]]:
        """Group debt items by severity."""
        grouped = defaultdict(list)
        for item in self.debt_items:
            grouped[item.severity].append(asdict(item))
        return dict(grouped)
    
    def _group_by_file(self) -> Dict[str, List[Dict]]:
        """Group debt items by file."""
        grouped = defaultdict(list)
        for item in self.debt_items:
            grouped[item.file_path].append(asdict(item))
        return dict(grouped)
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations for debt reduction."""
        recommendations = []
        
        stats = self._calculate_statistics()
        
        # High-severity items
        critical_items = [item for item in self.debt_items if item.severity == "critical"]
        if critical_items:
            recommendations.append({
                "priority": "immediate",
                "title": "Address Critical Technical Debt",
                "description": f"There are {len(critical_items)} critical debt items requiring immediate attention",
                "effort_hours": sum(item.estimated_effort_hours for item in critical_items),
                "items_count": len(critical_items)
            })
        
        # High effort items
        high_effort_items = [item for item in self.debt_items if item.estimated_effort_hours > 5]
        if high_effort_items:
            recommendations.append({
                "priority": "high",
                "title": "Plan Refactoring for High-Effort Items",
                "description": f"Schedule dedicated time for {len(high_effort_items)} high-effort debt items",
                "effort_hours": sum(item.estimated_effort_hours for item in high_effort_items),
                "items_count": len(high_effort_items)
            })
        
        # File-specific recommendations
        file_debt_counts = Counter(item.file_path for item in self.debt_items)
        problematic_files = [(file, count) for file, count in file_debt_counts.items() if count > 5]
        
        if problematic_files:
            recommendations.append({
                "priority": "medium",
                "title": "Focus on Problematic Files",
                "description": f"Consider refactoring files with high debt concentration",
                "problematic_files": problematic_files[:10],  # Top 10
                "items_count": sum(count for _, count in problematic_files)
            })
        
        # Pattern-based recommendations
        pattern_debt = [item for item in self.debt_items if item.type == "pattern_debt"]
        if pattern_debt:
            recommendations.append({
                "priority": "medium",
                "title": "Update Coding Patterns",
                "description": f"Modernize {len(pattern_debt)} outdated coding patterns",
                "effort_hours": sum(item.estimated_effort_hours for item in pattern_debt),
                "items_count": len(pattern_debt)
            })
        
        return recommendations
    
    def print_summary(self):
        """Print a summary of technical debt findings."""
        if not self.debt_items:
            print("No technical debt items found.")
            return
        
        stats = self._calculate_statistics()
        
        print("\n" + "="*60)
        print("TECHNICAL DEBT SUMMARY")
        print("="*60)
        
        print(f"Total debt items: {len(self.debt_items)}")
        print(f"Estimated total effort: {stats['total_estimated_effort_hours']:.1f} hours")
        print(f"Average effort per item: {stats['average_effort_per_item']:.1f} hours")
        print(f"Files with debt: {stats['files_with_debt']}")
        print(f"Debt density: {stats['debt_density']:.3f} items per file")
        
        print(f"\nBy Severity:")
        for severity, count in stats['severity_distribution'].items():
            print(f"  {severity}: {count}")
        
        print(f"\nBy Type:")
        for debt_type, count in stats['type_distribution'].items():
            print(f"  {debt_type}: {count}")
        
        # Show top problematic files
        file_debt_counts = Counter(item.file_path for item in self.debt_items)
        print(f"\nTop 5 files with most debt:")
        for file_path, count in file_debt_counts.most_common(5):
            total_effort = sum(item.estimated_effort_hours for item in self.debt_items if item.file_path == file_path)
            print(f"  {file_path}: {count} items ({total_effort:.1f} hours)")


class ComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code complexity."""
    
    def __init__(self):
        self.complex_items = []
        self.current_complexity = 0
        
    def visit_FunctionDef(self, node):
        """Visit function definition and calculate complexity."""
        old_complexity = self.current_complexity
        self.current_complexity = 1  # Base complexity
        
        self.generic_visit(node)
        
        if self.current_complexity > 5:  # Threshold for reporting
            self.complex_items.append({
                'type': 'function',
                'name': node.name,
                'line': node.lineno,
                'complexity': self.current_complexity
            })
        
        self.current_complexity = old_complexity
    
    def visit_ClassDef(self, node):
        """Visit class definition."""
        # For classes, we'll count methods and their complexities
        method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
        
        if method_count > 10:  # Threshold for large classes
            self.complex_items.append({
                'type': 'class',
                'name': node.name,
                'line': node.lineno,
                'complexity': method_count
            })
        
        self.generic_visit(node)
    
    def visit_If(self, node):
        """Visit if statement."""
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Visit for loop."""
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Visit while loop."""
        self.current_complexity += 1
        self.generic_visit(node)
    
    def visit_Try(self, node):
        """Visit try statement."""
        self.current_complexity += 1
        self.generic_visit(node)


def main():
    """Main entry point for technical debt tracker."""
    parser = argparse.ArgumentParser(description="Track and analyze technical debt")
    parser.add_argument("--repo-path", default=".", help="Repository path to analyze")
    parser.add_argument("--output", default="technical-debt-report.json", help="Output file path")
    parser.add_argument("--format", choices=["json", "summary"], default="json", help="Output format")
    parser.add_argument("--scan-only", action="store_true", help="Only scan, don't generate report")
    
    args = parser.parse_args()
    
    try:
        tracker = TechnicalDebtTracker(repo_path=args.repo_path)
        debt_items = tracker.scan_codebase()
        
        if args.format == "summary" or args.scan_only:
            tracker.print_summary()
        
        if not args.scan_only:
            report = tracker.generate_report(args.output)
            
            if args.format == "json":
                print(f"Full report saved to {args.output}")
        
        # Exit with error code if critical debt found
        critical_debt = [item for item in debt_items if item.severity == "critical"]
        return 1 if critical_debt else 0
        
    except Exception as e:
        print(f"Technical debt analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())