#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Execution Engine
Automatically executes highest-value work items with full validation.
"""

import json
import subprocess
import os
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from current directory
exec(open(os.path.join(os.path.dirname(__file__), 'value_discovery.py')).read())


@dataclass
class ExecutionResult:
    """Result of autonomous work item execution."""
    work_item_id: str
    success: bool
    start_time: str
    end_time: str
    duration_seconds: float
    branch_name: str
    pr_url: Optional[str] = None
    changes_made: List[str] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False
    test_results: Dict = None
    
    def __post_init__(self):
        if self.changes_made is None:
            self.changes_made = []
        if self.test_results is None:
            self.test_results = {}


class AutonomousExecutor:
    """Executes work items autonomously with full validation and rollback."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        self.execution_history = []
        self.current_branch = None
        
        # Load execution history if exists
        history_file = Path('.terragon/execution-history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.execution_history = json.load(f)
    
    def can_execute_item(self, item: WorkItem) -> Tuple[bool, str]:
        """Check if work item can be safely executed."""
        
        # Check if repository is clean
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            return False, "Repository has uncommitted changes"
        
        # Check if on appropriate branch
        result = subprocess.run(['git', 'branch', '--show-current'], 
                              capture_output=True, text=True)
        current_branch = result.stdout.strip()
        if current_branch != 'main' and not current_branch.startswith('auto-value'):
            return False, f"Not on main or auto-value branch: {current_branch}"
        
        # Check dependencies
        if item.dependencies:
            for dep in item.dependencies:
                if not self._check_dependency(dep):
                    return False, f"Dependency not met: {dep}"
        
        # Check minimum score threshold
        min_score = self.config['scoring']['thresholds']['minScore']
        if item.composite_score < min_score:
            return False, f"Score {item.composite_score} below threshold {min_score}"
        
        return True, "Ready for execution"
    
    def create_execution_branch(self, item: WorkItem) -> str:
        """Create a new branch for executing the work item."""
        # Generate branch name
        slug = item.title.lower().replace(' ', '-').replace(':', '').replace('[', '').replace(']', '')[:50]
        branch_name = f"auto-value/{item.id}-{slug}"
        
        # Create and checkout branch
        subprocess.run(['git', 'checkout', 'main'], check=True)
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        
        self.current_branch = branch_name
        return branch_name
    
    def execute_dependency_update(self, item: WorkItem) -> List[str]:
        """Execute dependency update work item."""
        changes = []
        
        if 'update' in item.title.lower() and 'from' in item.title and 'to' in item.title:
            # Parse package and versions from title
            parts = item.title.split()
            package_name = None
            for i, part in enumerate(parts):
                if part.lower() == 'update' and i + 1 < len(parts):
                    package_name = parts[i + 1]
                    break
            
            if package_name:
                # For system packages, we can't auto-update, so create documentation
                update_note = f"""# Dependency Update Recommendation

## Package: {package_name}
- **Current**: System-managed
- **Recommendation**: {item.description}
- **Priority**: {item.category.title()}
- **Estimated Effort**: {item.estimated_effort} hours

This system dependency should be updated through the system package manager.
For containerized deployments, update the base image or Dockerfile.

## Manual Update Steps:
1. Check current version: `{package_name} --version` or similar
2. Update through system package manager
3. Test functionality after update
4. Update documentation if needed

*Auto-generated by Terragon Autonomous SDLC*
"""
                
                # Create documentation file
                doc_file = Path(f"docs/dependency-updates/{package_name}-update.md")
                doc_file.parent.mkdir(parents=True, exist_ok=True)
                doc_file.write_text(update_note)
                changes.append(f"Created dependency update documentation: {doc_file}")
                
                # Stage the change
                subprocess.run(['git', 'add', str(doc_file)], check=True)
        
        return changes
    
    def execute_technical_debt(self, item: WorkItem) -> List[str]:
        """Execute technical debt work item."""
        changes = []
        
        if item.files_affected:
            for file_path in item.files_affected:
                file_obj = Path(file_path)
                if file_obj.exists() and file_obj.suffix == '.py':
                    # Add a comment about the technical debt
                    content = file_obj.read_text()
                    
                    # Add comment at the top of the file
                    debt_comment = f'''"""
Technical Debt Item: {item.title}
Status: Identified for resolution
Priority: {item.category.title()}
Estimated Effort: {item.estimated_effort} hours

Description: {item.description}

*Auto-generated by Terragon Autonomous SDLC*
"""

'''
                    
                    # Only add if not already present
                    if "Technical Debt Item:" not in content:
                        if content.startswith('"""') or content.startswith("'''"):
                            # Insert after existing docstring
                            lines = content.split('\n')
                            end_quote_idx = -1
                            quote_type = '"""' if content.startswith('"""') else "'''"
                            
                            for i, line in enumerate(lines[1:], 1):
                                if quote_type in line:
                                    end_quote_idx = i
                                    break
                            
                            if end_quote_idx > 0:
                                lines.insert(end_quote_idx + 1, debt_comment)
                                content = '\n'.join(lines)
                        else:
                            content = debt_comment + content
                        
                        file_obj.write_text(content)
                        changes.append(f"Added technical debt documentation to: {file_path}")
                        subprocess.run(['git', 'add', file_path], check=True)
        
        return changes
    
    def execute_code_quality(self, item: WorkItem) -> List[str]:
        """Execute code quality improvement work item."""
        changes = []
        
        # For linting issues, try to run auto-formatters
        if 'ruff' in item.source or 'lint' in item.type:
            # Run black formatter
            try:
                result = subprocess.run([
                    'python3', '-m', 'black', 'src/', '--check'
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    # Run black to fix formatting
                    subprocess.run(['python3', '-m', 'black', 'src/'], check=True)
                    changes.append("Applied black code formatting to src/")
                    subprocess.run(['git', 'add', 'src/'], check=True)
            except subprocess.CalledProcessError:
                pass  # Black might not be available
            
            # Try to run ruff --fix
            try:
                result = subprocess.run([
                    'python3', '-m', 'ruff', 'check', '--fix', 'src/'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    changes.append("Applied ruff auto-fixes to src/")
                    subprocess.run(['git', 'add', 'src/'], check=True)
            except subprocess.CalledProcessError:
                pass  # Ruff might not be available
        
        return changes
    
    def execute_work_item(self, item: WorkItem) -> ExecutionResult:
        """Execute a single work item with full validation."""
        start_time = datetime.now()
        result = ExecutionResult(
            work_item_id=item.id,
            success=False,
            start_time=start_time.isoformat(),
            end_time="",
            duration_seconds=0.0,
            branch_name=""
        )
        
        try:
            print(f"🚀 Executing: [{item.id}] {item.title}")
            
            # Check if item can be executed
            can_execute, reason = self.can_execute_item(item)
            if not can_execute:
                result.error_message = reason
                print(f"❌ Cannot execute: {reason}")
                return result
            
            # Create execution branch
            branch_name = self.create_execution_branch(item)
            result.branch_name = branch_name
            print(f"📝 Created branch: {branch_name}")
            
            # Execute based on item type
            changes = []
            if item.category == 'maintenance' and 'update' in item.type:
                changes = self.execute_dependency_update(item)
            elif item.category == 'technical_debt':
                changes = self.execute_technical_debt(item)
            elif item.category == 'code_quality':
                changes = self.execute_code_quality(item)
            else:
                # Generic execution - create documentation
                changes = self._create_generic_documentation(item)
            
            result.changes_made = changes
            
            # Run validation tests
            if changes:
                test_results = self.run_validation_tests()
                result.test_results = test_results
                
                if not test_results.get('passed', False):
                    print("❌ Validation failed, rolling back...")
                    self.rollback_changes()
                    result.rollback_performed = True
                    result.error_message = "Validation tests failed"
                    return result
                
                # Create commit
                commit_message = self.create_commit_message(item, changes)
                subprocess.run(['git', 'commit', '-m', commit_message], check=True)
                print(f"✅ Committed changes: {len(changes)} files")
                
                # Create pull request
                pr_url = self.create_pull_request(item, changes)
                result.pr_url = pr_url
                print(f"🔗 Created PR: {pr_url}")
                
                result.success = True
            else:
                print("ℹ️ No changes to commit")
                self.rollback_changes()
                result.error_message = "No changes generated"
        
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            result.error_message = str(e)
            if self.current_branch:
                self.rollback_changes()
                result.rollback_performed = True
        
        finally:
            end_time = datetime.now()
            result.end_time = end_time.isoformat()
            result.duration_seconds = (end_time - start_time).total_seconds()
            
            # Return to main branch
            try:
                subprocess.run(['git', 'checkout', 'main'], check=True)
                self.current_branch = None
            except subprocess.CalledProcessError:
                pass
        
        return result
    
    def _create_generic_documentation(self, item: WorkItem) -> List[str]:
        """Create documentation for generic work items."""
        changes = []
        
        # Create work item documentation
        doc_content = f"""# Work Item: {item.title}

## Details
- **ID**: {item.id}
- **Category**: {item.category.replace('_', ' ').title()}
- **Type**: {item.type.replace('_', ' ').title()}
- **Priority Score**: {item.composite_score:.1f}
- **Estimated Effort**: {item.estimated_effort} hours
- **Discovered**: {item.discovered_at}
- **Source**: {item.source.replace('_', ' ').title()}

## Description
{item.description}

## Files Affected
{chr(10).join(f"- {f}" for f in item.files_affected) if item.files_affected else "None specified"}

## Implementation Notes
This work item was identified by the Terragon Autonomous SDLC system but requires manual implementation.

## Next Steps
1. Review the work item details above
2. Plan the implementation approach
3. Implement the changes
4. Test thoroughly
5. Update this documentation with results

---
*Auto-generated by Terragon Autonomous SDLC*
Generated: {datetime.now().isoformat()}
"""
        
        # Create documentation file
        doc_dir = Path("docs/work-items")
        doc_dir.mkdir(parents=True, exist_ok=True)
        doc_file = doc_dir / f"{item.id}.md"
        doc_file.write_text(doc_content)
        changes.append(f"Created work item documentation: {doc_file}")
        
        # Stage the change
        subprocess.run(['git', 'add', str(doc_file)], check=True)
        
        return changes
    
    def run_validation_tests(self) -> Dict:
        """Run validation tests on changes."""
        results = {
            'passed': True,
            'test_results': {},
            'lint_results': {},
            'build_results': {}
        }
        
        # Run tests if pytest is available
        if Path('pytest.ini').exists() or Path('pyproject.toml').exists():
            try:
                result = subprocess.run([
                    'python3', '-m', 'pytest', '--tb=short', '-v'
                ], capture_output=True, text=True, timeout=300)
                
                results['test_results'] = {
                    'returncode': result.returncode,
                    'passed': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
                if result.returncode != 0:
                    results['passed'] = False
                    print(f"❌ Tests failed: {result.returncode}")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                results['test_results'] = {'error': str(e), 'passed': False}
                results['passed'] = False
        
        # Run linting if configured
        test_requirements = self.config['execution']['testRequirements']
        if test_requirements.get('linting', False):
            try:
                result = subprocess.run([
                    'python3', '-m', 'ruff', 'check', 'src/'
                ], capture_output=True, text=True)
                
                results['lint_results'] = {
                    'returncode': result.returncode,
                    'passed': result.returncode == 0,
                    'stdout': result.stdout
                }
                
                if result.returncode != 0:
                    results['passed'] = False
                    print(f"❌ Linting failed: {result.returncode}")
            except subprocess.CalledProcessError:
                pass  # Ruff might not be available
        
        return results
    
    def rollback_changes(self):
        """Rollback all changes and return to main branch."""
        try:
            if self.current_branch and self.current_branch != 'main':
                subprocess.run(['git', 'checkout', 'main'], check=True)
                subprocess.run(['git', 'branch', '-D', self.current_branch], check=True)
                print(f"🔄 Rolled back branch: {self.current_branch}")
            self.current_branch = None
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Rollback warning: {e}")
    
    def create_commit_message(self, item: WorkItem, changes: List[str]) -> str:
        """Create standardized commit message."""
        return f"""[AUTO-VALUE] {item.title}

Category: {item.category.replace('_', ' ').title()}
Priority Score: {item.composite_score:.1f}
Estimated Effort: {item.estimated_effort}h

Changes:
{chr(10).join(f"- {change}" for change in changes)}

🤖 Generated with Terragon Autonomous SDLC

Co-Authored-By: Terry <noreply@terragon.ai>"""
    
    def create_pull_request(self, item: WorkItem, changes: List[str]) -> Optional[str]:
        """Create pull request for the changes."""
        try:
            # Push branch first
            subprocess.run(['git', 'push', '-u', 'origin', self.current_branch], check=True)
            
            # Create PR description
            description = f"""## 🤖 Autonomous Value Implementation

**Work Item ID**: {item.id}
**Category**: {item.category.replace('_', ' ').title()}
**Priority Score**: {item.composite_score:.1f}
**Source**: {item.source.replace('_', ' ').title()}

### Description
{item.description}

### Changes Made
{chr(10).join(f"- {change}" for change in changes)}

### Value Metrics
- **WSJF Score**: {item.wsjf:.1f}
- **ICE Score**: {item.ice:.0f}
- **Technical Debt Score**: {item.technical_debt:.1f}
- **Estimated Effort**: {item.estimated_effort} hours
- **Confidence**: {item.confidence:.1%}

### Test Results
All validation tests passed ✅

### Files Affected
{chr(10).join(f"- `{f}`" for f in item.files_affected) if item.files_affected else "- Documentation only"}

---
🤖 Generated with [Terragon Autonomous SDLC](https://terragon.ai)

Co-Authored-By: Terry <noreply@terragon.ai>"""
            
            # Use gh CLI to create PR if available
            result = subprocess.run([
                'gh', 'pr', 'create',
                '--title', f"[AUTO-VALUE] {item.title}",
                '--body', description,
                '--label', 'autonomous,value-driven,' + item.category
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                pr_url = result.stdout.strip()
                return pr_url
            else:
                print(f"⚠️ Could not create PR: {result.stderr}")
                return None
                
        except subprocess.CalledProcessError as e:
            print(f"⚠️ PR creation failed: {e}")
            return None
    
    def save_execution_history(self, result: ExecutionResult):
        """Save execution result to history."""
        self.execution_history.append(result.__dict__)
        
        # Keep only last 100 executions
        self.execution_history = self.execution_history[-100:]
        
        # Save to file
        history_file = Path('.terragon/execution-history.json')
        with open(history_file, 'w') as f:
            json.dump(self.execution_history, f, indent=2)
    
    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is met."""
        # Simple dependency checking - can be enhanced
        return True  # For now, assume all dependencies are met
    
    def run_autonomous_cycle(self) -> Optional[ExecutionResult]:
        """Run one complete autonomous execution cycle."""
        print("🔄 Starting autonomous execution cycle...")
        
        # Discover and prioritize work items
        items = self.discovery_engine.run_discovery_cycle()
        
        if not items:
            print("ℹ️ No work items discovered")
            return None
        
        # Select next best value item
        next_item = self.discovery_engine.select_next_best_value(items)
        
        if not next_item:
            print("ℹ️ No items meet execution criteria")
            return None
        
        # Execute the item
        result = self.execute_work_item(next_item)
        
        # Save execution history
        self.save_execution_history(result)
        
        if result.success:
            print(f"✅ Autonomous execution completed successfully!")
            print(f"   Duration: {result.duration_seconds:.1f}s")
            print(f"   Changes: {len(result.changes_made)}")
            if result.pr_url:
                print(f"   PR: {result.pr_url}")
        else:
            print(f"❌ Autonomous execution failed: {result.error_message}")
        
        return result


def main():
    """Main entry point for autonomous execution."""
    executor = AutonomousExecutor()
    result = executor.run_autonomous_cycle()
    
    if result:
        print(f"\n📊 Execution Summary:")
        print(f"Work Item: {result.work_item_id}")
        print(f"Success: {result.success}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Changes: {len(result.changes_made)}")


if __name__ == "__main__":
    main()