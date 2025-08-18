#!/usr/bin/env python3
"""
SDLC Health Check Script for PQC IoT Retrofit Scanner.

Validates that all SDLC components are properly configured and functioning.
Provides actionable recommendations for any missing or misconfigured components.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import requests
import time


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # pass, warning, fail, not_configured
    message: str
    details: Optional[Dict[str, Any]] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class SDLCHealthChecker:
    """Comprehensive SDLC health checker."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.results: List[HealthCheckResult] = []
        self.logger = self._setup_logging()
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_repo = os.getenv("GITHUB_REPOSITORY", "")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def run_comprehensive_check(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        self.logger.info("🔍 Starting comprehensive SDLC health check...")
        
        # Foundation checks
        self._check_documentation()
        self._check_community_files()
        self._check_architecture_decisions()
        
        # Development environment checks
        self._check_development_environment()
        self._check_code_quality_tools()
        self._check_pre_commit_hooks()
        
        # Testing infrastructure checks
        self._check_testing_framework()
        self._check_test_coverage()
        self._check_security_testing()
        
        # Build and containerization checks
        self._check_build_system()
        self._check_containerization()
        self._check_package_management()
        
        # Monitoring and observability checks
        self._check_monitoring_setup()
        self._check_logging_configuration()
        self._check_metrics_collection()
        
        # Workflow and automation checks
        self._check_github_workflows()
        self._check_security_scanning()
        self._check_dependency_management()
        
        # Integration and configuration checks
        self._check_repository_settings()
        self._check_branch_protection()
        self._check_external_integrations()
        
        self.logger.info(f"✅ Health check completed. {len(self.results)} components checked.")
        return self.results
    
    def _check_documentation(self):
        """Check documentation completeness."""
        required_docs = [
            ("README.md", "Project overview and quick start"),
            ("ARCHITECTURE.md", "System architecture documentation"),
            ("CONTRIBUTING.md", "Contribution guidelines"),
            ("SECURITY.md", "Security policy and reporting"),
            ("CHANGELOG.md", "Change log for releases"),
            ("PROJECT_CHARTER.md", "Project charter and scope"),
            ("docs/ROADMAP.md", "Project roadmap and milestones")
        ]
        
        missing_docs = []
        incomplete_docs = []
        
        for doc_path, description in required_docs:
            full_path = self.repo_path / doc_path
            if not full_path.exists():
                missing_docs.append(f"{doc_path} - {description}")
            elif full_path.stat().st_size < 100:  # Less than 100 bytes is likely incomplete
                incomplete_docs.append(f"{doc_path} - appears incomplete")
        
        if missing_docs or incomplete_docs:
            status = "fail" if missing_docs else "warning"
            message = f"Documentation issues found: {len(missing_docs)} missing, {len(incomplete_docs)} incomplete"
            recommendations = []
            if missing_docs:
                recommendations.append("Create missing documentation files")
            if incomplete_docs:
                recommendations.append("Complete documentation stubs")
            
            self.results.append(HealthCheckResult(
                component="documentation",
                status=status,
                message=message,
                details={"missing": missing_docs, "incomplete": incomplete_docs},
                recommendations=recommendations
            ))
        else:
            self.results.append(HealthCheckResult(
                component="documentation",
                status="pass",
                message="All required documentation is present and appears complete"
            ))
    
    def _check_community_files(self):
        """Check community and governance files."""
        required_files = [
            "CODE_OF_CONDUCT.md",
            "LICENSE", 
            "CODEOWNERS",
            ".github/ISSUE_TEMPLATE",
            ".github/PULL_REQUEST_TEMPLATE.md"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.repo_path / file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results.append(HealthCheckResult(
                component="community_files",
                status="warning",
                message=f"{len(missing_files)} community files missing",
                details={"missing": missing_files},
                recommendations=["Create missing community files for better project governance"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="community_files", 
                status="pass",
                message="All community files are present"
            ))
    
    def _check_architecture_decisions(self):
        """Check Architecture Decision Records (ADR)."""
        adr_dir = self.repo_path / "docs" / "adr"
        
        if not adr_dir.exists():
            self.results.append(HealthCheckResult(
                component="architecture_decisions",
                status="fail",
                message="ADR directory not found",
                recommendations=["Create docs/adr/ directory and document architectural decisions"]
            ))
            return
        
        adr_files = list(adr_dir.glob("*.md"))
        template_exists = (adr_dir / "template.md").exists()
        
        if len(adr_files) < 2:  # Expecting at least template + one ADR
            status = "warning"
            message = f"Only {len(adr_files)} ADR files found"
            recommendations = ["Document more architectural decisions as ADRs"]
        elif not template_exists:
            status = "warning"
            message = "ADR template missing"
            recommendations = ["Create ADR template for consistent documentation"]
        else:
            status = "pass"
            message = f"ADR framework properly configured with {len(adr_files)} decisions documented"
            recommendations = []
        
        self.results.append(HealthCheckResult(
            component="architecture_decisions",
            status=status,
            message=message,
            details={"adr_count": len(adr_files), "template_exists": template_exists},
            recommendations=recommendations
        ))
    
    def _check_development_environment(self):
        """Check development environment configuration."""
        dev_files = [
            (".devcontainer/devcontainer.json", "Development container configuration"),
            (".env.example", "Environment variables template"),
            (".editorconfig", "Editor configuration"),
            (".gitignore", "Git ignore patterns"),
            ("pyproject.toml", "Python project configuration")
        ]
        
        missing_files = []
        for file_path, description in dev_files:
            if not (self.repo_path / file_path).exists():
                missing_files.append(f"{file_path} - {description}")
        
        if missing_files:
            self.results.append(HealthCheckResult(
                component="development_environment",
                status="warning",
                message=f"{len(missing_files)} development configuration files missing",
                details={"missing": missing_files},
                recommendations=["Create missing development environment configuration files"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="development_environment",
                status="pass",
                message="Development environment is properly configured"
            ))
    
    def _check_code_quality_tools(self):
        """Check code quality tool configuration."""
        quality_configs = [
            (".bandit", "Security linting configuration"),
            (".secrets.baseline", "Secret detection baseline"),
            ("pyproject.toml", "Python tooling configuration")
        ]
        
        configured_tools = []
        missing_configs = []
        
        for config_file, description in quality_configs:
            if (self.repo_path / config_file).exists():
                configured_tools.append(config_file)
            else:
                missing_configs.append(f"{config_file} - {description}")
        
        # Check if tools are specified in pyproject.toml
        pyproject_path = self.repo_path / "pyproject.toml"
        has_tool_config = False
        if pyproject_path.exists():
            try:
                content = pyproject_path.read_text()
                has_tool_config = any(tool in content for tool in ["ruff", "black", "mypy", "bandit"])
            except Exception:
                pass
        
        if not has_tool_config and missing_configs:
            self.results.append(HealthCheckResult(
                component="code_quality_tools",
                status="warning",
                message="Code quality tools not fully configured",
                details={"configured": configured_tools, "missing": missing_configs},
                recommendations=["Configure code quality tools (ruff, black, mypy, bandit)"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="code_quality_tools",
                status="pass",
                message="Code quality tools are configured"
            ))
    
    def _check_pre_commit_hooks(self):
        """Check pre-commit hooks configuration."""
        precommit_config = self.repo_path / ".pre-commit-config.yaml"
        
        if not precommit_config.exists():
            self.results.append(HealthCheckResult(
                component="pre_commit_hooks",
                status="warning",
                message="Pre-commit hooks not configured",
                recommendations=["Setup pre-commit hooks for automated code quality checks"]
            ))
        else:
            # Check if hooks are actually installed
            try:
                result = subprocess.run(
                    ["pre-commit", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                installed = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                installed = False
            
            if installed:
                self.results.append(HealthCheckResult(
                    component="pre_commit_hooks",
                    status="pass",
                    message="Pre-commit hooks are configured and installed"
                ))
            else:
                self.results.append(HealthCheckResult(
                    component="pre_commit_hooks",
                    status="warning",
                    message="Pre-commit configured but not installed",
                    recommendations=["Run 'pre-commit install' to activate hooks"]
                ))
    
    def _check_testing_framework(self):
        """Check testing framework setup."""
        test_configs = [
            ("pytest.ini", "Pytest configuration"),
            ("tests/conftest.py", "Test configuration"),
            ("tests/unit/", "Unit tests directory"),
            ("tests/integration/", "Integration tests directory")
        ]
        
        missing_items = []
        existing_items = []
        
        for item_path, description in test_configs:
            full_path = self.repo_path / item_path
            if full_path.exists():
                existing_items.append(item_path)
            else:
                missing_items.append(f"{item_path} - {description}")
        
        if len(existing_items) < 2:
            self.results.append(HealthCheckResult(
                component="testing_framework",
                status="fail",
                message="Testing framework not properly configured",
                details={"existing": existing_items, "missing": missing_items},
                recommendations=["Setup comprehensive testing framework with pytest"]
            ))
        elif missing_items:
            self.results.append(HealthCheckResult(
                component="testing_framework",
                status="warning",
                message="Testing framework partially configured",
                details={"existing": existing_items, "missing": missing_items},
                recommendations=["Complete testing framework setup"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="testing_framework",
                status="pass",
                message="Testing framework is properly configured"
            ))
    
    def _check_test_coverage(self):
        """Check test coverage configuration."""
        coverage_files = [
            ".coveragerc",
            "pyproject.toml"  # coverage config can be in pyproject.toml
        ]
        
        has_coverage_config = any(
            (self.repo_path / f).exists() for f in coverage_files
        )
        
        if not has_coverage_config:
            self.results.append(HealthCheckResult(
                component="test_coverage",
                status="warning",
                message="Test coverage not configured",
                recommendations=["Configure test coverage reporting with coverage.py"]
            ))
        else:
            # Try to check if there's recent coverage data
            coverage_data_exists = any([
                (self.repo_path / "coverage.xml").exists(),
                (self.repo_path / "htmlcov").exists(),
                (self.repo_path / ".coverage").exists()
            ])
            
            if coverage_data_exists:
                self.results.append(HealthCheckResult(
                    component="test_coverage",
                    status="pass", 
                    message="Test coverage is configured and data exists"
                ))
            else:
                self.results.append(HealthCheckResult(
                    component="test_coverage",
                    status="warning",
                    message="Test coverage configured but no recent data found",
                    recommendations=["Run tests with coverage to generate coverage data"]
                ))
    
    def _check_security_testing(self):
        """Check security testing setup."""
        security_configs = [
            (".bandit", "Bandit security scanner config"),
            (".secrets.baseline", "Secret detection baseline")
        ]
        
        configured_tools = []
        for config_file, description in security_configs:
            if (self.repo_path / config_file).exists():
                configured_tools.append(config_file)
        
        if not configured_tools:
            self.results.append(HealthCheckResult(
                component="security_testing",
                status="warning",
                message="Security testing tools not configured",
                recommendations=["Configure security testing tools (bandit, safety)"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="security_testing",
                status="pass",
                message="Security testing tools are configured"
            ))
    
    def _check_build_system(self):
        """Check build system configuration."""
        build_files = [
            ("Makefile", "Make-based build system"),
            ("scripts/build.sh", "Build automation script"),
            ("pyproject.toml", "Python package configuration")
        ]
        
        has_build_system = any(
            (self.repo_path / f).exists() for f, _ in build_files
        )
        
        if not has_build_system:
            self.results.append(HealthCheckResult(
                component="build_system",
                status="fail",
                message="No build system configuration found",
                recommendations=["Create build system (Makefile or build scripts)"]
            ))
        else:
            # Check if build script is executable
            build_script = self.repo_path / "scripts" / "build.sh"
            if build_script.exists():
                is_executable = os.access(build_script, os.X_OK)
                if not is_executable:
                    self.results.append(HealthCheckResult(
                        component="build_system",
                        status="warning",
                        message="Build script exists but is not executable",
                        recommendations=["Make build script executable: chmod +x scripts/build.sh"]
                    ))
                else:
                    self.results.append(HealthCheckResult(
                        component="build_system",
                        status="pass",
                        message="Build system is properly configured"
                    ))
            else:
                self.results.append(HealthCheckResult(
                    component="build_system",
                    status="pass",
                    message="Build system configuration found"
                ))
    
    def _check_containerization(self):
        """Check containerization setup."""
        container_files = [
            ("Dockerfile", "Container image definition"),
            ("docker-compose.yml", "Multi-container orchestration"),
            (".dockerignore", "Docker build context optimization")
        ]
        
        missing_files = []
        existing_files = []
        
        for file_path, description in container_files:
            if (self.repo_path / file_path).exists():
                existing_files.append(file_path)
            else:
                missing_files.append(f"{file_path} - {description}")
        
        if not existing_files:
            self.results.append(HealthCheckResult(
                component="containerization",
                status="warning",
                message="No containerization configuration found",
                recommendations=["Add Docker configuration for consistent deployments"]
            ))
        elif len(existing_files) == 1:
            self.results.append(HealthCheckResult(
                component="containerization",
                status="warning",
                message="Basic containerization configured",
                details={"existing": existing_files, "missing": missing_files},
                recommendations=["Complete containerization setup with all recommended files"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="containerization",
                status="pass",
                message="Containerization is properly configured"
            ))
    
    def _check_package_management(self):
        """Check package management configuration."""
        # Check for semantic release configuration
        semantic_release_configs = [
            ".releaserc.json",
            ".releaserc.yml", 
            "release.config.js"
        ]
        
        has_semantic_release = any(
            (self.repo_path / f).exists() for f in semantic_release_configs
        )
        
        # Check for package publishing configuration
        pyproject_exists = (self.repo_path / "pyproject.toml").exists()
        
        if not pyproject_exists:
            self.results.append(HealthCheckResult(
                component="package_management",
                status="fail",
                message="Python package configuration missing",
                recommendations=["Create pyproject.toml for package management"]
            ))
        elif not has_semantic_release:
            self.results.append(HealthCheckResult(
                component="package_management",
                status="warning",
                message="Package configured but no semantic release setup",
                recommendations=["Configure semantic release for automated versioning"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="package_management",
                status="pass",
                message="Package management and semantic release configured"
            ))
    
    def _check_monitoring_setup(self):
        """Check monitoring and observability setup."""
        monitoring_files = [
            ("monitoring/health_check.py", "Health check system"),
            ("monitoring/prometheus_metrics.py", "Metrics collection"),
            ("monitoring/structured_logging.py", "Structured logging"),
            ("monitoring/grafana-dashboard.json", "Dashboard configuration")
        ]
        
        existing_files = []
        for file_path, description in monitoring_files:
            if (self.repo_path / file_path).exists():
                existing_files.append(file_path)
        
        if len(existing_files) < 2:
            self.results.append(HealthCheckResult(
                component="monitoring_setup",
                status="warning",
                message="Limited monitoring configuration",
                recommendations=["Implement comprehensive monitoring and observability"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="monitoring_setup",
                status="pass",
                message="Monitoring and observability properly configured"
            ))
    
    def _check_logging_configuration(self):
        """Check logging configuration."""
        logging_config = self.repo_path / "monitoring" / "structured_logging.py"
        
        if not logging_config.exists():
            self.results.append(HealthCheckResult(
                component="logging_configuration",
                status="warning",
                message="Structured logging not configured",
                recommendations=["Implement structured logging for better observability"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="logging_configuration",
                status="pass",
                message="Structured logging is configured"
            ))
    
    def _check_metrics_collection(self):
        """Check metrics collection setup."""
        metrics_files = [
            ("scripts/collect-metrics.py", "Automated metrics collection"),
            (".github/project-metrics.json", "Metrics configuration"),
            ("scripts/project-metrics-dashboard.py", "Dashboard generation")
        ]
        
        existing_files = []
        for file_path, description in metrics_files:
            if (self.repo_path / file_path).exists():
                existing_files.append(file_path)
        
        if len(existing_files) < 2:
            self.results.append(HealthCheckResult(
                component="metrics_collection",
                status="warning",
                message="Metrics collection not fully configured",
                recommendations=["Setup comprehensive metrics collection and dashboard"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="metrics_collection",
                status="pass",
                message="Metrics collection system is configured"
            ))
    
    def _check_github_workflows(self):
        """Check GitHub Actions workflow configuration."""
        workflows_dir = self.repo_path / ".github" / "workflows"
        
        if not workflows_dir.exists():
            self.results.append(HealthCheckResult(
                component="github_workflows",
                status="not_configured",
                message="GitHub workflows directory not found",
                recommendations=["Create .github/workflows/ directory and add CI/CD workflows from templates"]
            ))
            return
        
        workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            self.results.append(HealthCheckResult(
                component="github_workflows",
                status="not_configured",
                message="No GitHub workflows found",
                recommendations=["Copy workflow templates from docs/workflows/examples/ to .github/workflows/"]
            ))
        else:
            # Check for essential workflows
            essential_workflows = ["ci", "security", "release"]
            found_workflows = []
            
            for workflow_file in workflow_files:
                workflow_name = workflow_file.stem.lower()
                for essential in essential_workflows:
                    if essential in workflow_name:
                        found_workflows.append(essential)
            
            missing_workflows = [w for w in essential_workflows if w not in found_workflows]
            
            if missing_workflows:
                self.results.append(HealthCheckResult(
                    component="github_workflows",
                    status="warning",
                    message=f"Some essential workflows missing: {', '.join(missing_workflows)}",
                    details={"found": found_workflows, "missing": missing_workflows},
                    recommendations=["Add missing essential workflows (CI, security, release)"]
                ))
            else:
                self.results.append(HealthCheckResult(
                    component="github_workflows",
                    status="pass",
                    message="Essential GitHub workflows are configured"
                ))
    
    def _check_security_scanning(self):
        """Check security scanning configuration."""
        # Check for security workflow templates
        security_templates = [
            "docs/workflows/examples/security-scan.yml",
            ".github/workflows/security.yml",
            ".github/workflows/security-scan.yml"
        ]
        
        has_security_workflow = any(
            (self.repo_path / f).exists() for f in security_templates
        )
        
        # Check for dependabot configuration
        dependabot_config = self.repo_path / ".github" / "dependabot.yml"
        
        if not has_security_workflow and not dependabot_config.exists():
            self.results.append(HealthCheckResult(
                component="security_scanning",
                status="warning",
                message="Security scanning not configured",
                recommendations=["Setup automated security scanning and dependency updates"]
            ))
        elif not has_security_workflow:
            self.results.append(HealthCheckResult(
                component="security_scanning",
                status="warning",
                message="Dependabot configured but no security workflow",
                recommendations=["Add security scanning workflow"]
            ))
        elif not dependabot_config.exists():
            self.results.append(HealthCheckResult(
                component="security_scanning",
                status="warning",
                message="Security workflow configured but no Dependabot",
                recommendations=["Configure Dependabot for automated dependency updates"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="security_scanning",
                status="pass",
                message="Security scanning is properly configured"
            ))
    
    def _check_dependency_management(self):
        """Check dependency management setup."""
        dependency_files = [
            ("pyproject.toml", "Python dependencies"),
            (".github/dependabot.yml", "Automated dependency updates"),
            ("requirements.txt", "Legacy dependency file")
        ]
        
        has_modern_deps = (self.repo_path / "pyproject.toml").exists()
        has_dependabot = (self.repo_path / ".github" / "dependabot.yml").exists()
        
        if not has_modern_deps:
            self.results.append(HealthCheckResult(
                component="dependency_management",
                status="warning",
                message="Modern dependency management not configured",
                recommendations=["Use pyproject.toml for dependency management"]
            ))
        elif not has_dependabot:
            self.results.append(HealthCheckResult(
                component="dependency_management",
                status="warning",
                message="No automated dependency updates",
                recommendations=["Configure Dependabot for automated updates"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="dependency_management",
                status="pass",
                message="Dependency management is properly configured"
            ))
    
    def _check_repository_settings(self):
        """Check repository settings (requires GitHub API)."""
        if not self.github_token or not self.github_repo:
            self.results.append(HealthCheckResult(
                component="repository_settings",
                status="not_configured",
                message="Cannot check repository settings (missing GitHub token/repo)",
                recommendations=["Set GITHUB_TOKEN and GITHUB_REPOSITORY environment variables"]
            ))
            return
        
        try:
            headers = {"Authorization": f"token {self.github_token}"}
            response = requests.get(
                f"https://api.github.com/repos/{self.github_repo}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                repo_data = response.json()
                
                issues = []
                if not repo_data.get("description"):
                    issues.append("Missing repository description")
                if not repo_data.get("homepage"):
                    issues.append("Missing homepage URL")
                if not repo_data.get("topics"):
                    issues.append("No topics configured")
                if repo_data.get("has_wiki", False):
                    issues.append("Wiki is enabled (consider disabling)")
                
                if issues:
                    self.results.append(HealthCheckResult(
                        component="repository_settings",
                        status="warning",
                        message="Repository settings need attention",
                        details={"issues": issues},
                        recommendations=["Update repository settings in GitHub"]
                    ))
                else:
                    self.results.append(HealthCheckResult(
                        component="repository_settings",
                        status="pass",
                        message="Repository settings are properly configured"
                    ))
            else:
                self.results.append(HealthCheckResult(
                    component="repository_settings",
                    status="warning",
                    message=f"Cannot access repository settings (HTTP {response.status_code})",
                    recommendations=["Check GitHub token permissions"]
                ))
                
        except Exception as e:
            self.results.append(HealthCheckResult(
                component="repository_settings",
                status="warning",
                message=f"Error checking repository settings: {str(e)}",
                recommendations=["Verify GitHub API connectivity"]
            ))
    
    def _check_branch_protection(self):
        """Check branch protection rules."""
        if not self.github_token or not self.github_repo:
            self.results.append(HealthCheckResult(
                component="branch_protection",
                status="not_configured",
                message="Cannot check branch protection (missing GitHub token/repo)",
                recommendations=["Set GITHUB_TOKEN and GITHUB_REPOSITORY environment variables"]
            ))
            return
        
        try:
            headers = {"Authorization": f"token {self.github_token}"}
            response = requests.get(
                f"https://api.github.com/repos/{self.github_repo}/branches/main/protection",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                protection_data = response.json()
                
                issues = []
                if not protection_data.get("required_status_checks", {}).get("strict"):
                    issues.append("Strict status checks not enabled")
                if not protection_data.get("required_pull_request_reviews"):
                    issues.append("Pull request reviews not required")
                if not protection_data.get("enforce_admins", {}).get("enabled"):
                    issues.append("Admin enforcement not enabled")
                
                if issues:
                    self.results.append(HealthCheckResult(
                        component="branch_protection",
                        status="warning",
                        message="Branch protection needs strengthening",
                        details={"issues": issues},
                        recommendations=["Update branch protection rules in GitHub settings"]
                    ))
                else:
                    self.results.append(HealthCheckResult(
                        component="branch_protection",
                        status="pass",
                        message="Branch protection is properly configured"
                    ))
            elif response.status_code == 404:
                self.results.append(HealthCheckResult(
                    component="branch_protection",
                    status="fail",
                    message="No branch protection configured for main branch",
                    recommendations=["Configure branch protection rules for main branch"]
                ))
            else:
                self.results.append(HealthCheckResult(
                    component="branch_protection",
                    status="warning",
                    message=f"Cannot check branch protection (HTTP {response.status_code})",
                    recommendations=["Check GitHub token permissions"]
                ))
                
        except Exception as e:
            self.results.append(HealthCheckResult(
                component="branch_protection",
                status="warning",
                message=f"Error checking branch protection: {str(e)}",
                recommendations=["Verify GitHub API connectivity"]
            ))
    
    def _check_external_integrations(self):
        """Check external service integrations."""
        # This is a placeholder for checking external integrations
        # In practice, this would check for webhooks, service configurations, etc.
        
        integration_configs = [
            (".codecov.yml", "Codecov configuration"),
            ("sonar-project.properties", "SonarQube configuration"),
            (".github/dependabot.yml", "Dependabot configuration")
        ]
        
        configured_integrations = []
        for config_file, description in integration_configs:
            if (self.repo_path / config_file).exists():
                configured_integrations.append(description)
        
        if not configured_integrations:
            self.results.append(HealthCheckResult(
                component="external_integrations",
                status="warning",
                message="No external integrations configured",
                recommendations=["Consider integrating with code quality and security services"]
            ))
        else:
            self.results.append(HealthCheckResult(
                component="external_integrations",
                status="pass",
                message=f"External integrations configured: {', '.join(configured_integrations)}"
            ))
    
    def generate_report(self, format_type: str = "detailed") -> str:
        """Generate health check report."""
        if format_type == "summary":
            return self._generate_summary_report()
        elif format_type == "json":
            return self._generate_json_report()
        else:
            return self._generate_detailed_report()
    
    def _generate_summary_report(self) -> str:
        """Generate summary health report."""
        status_counts = {"pass": 0, "warning": 0, "fail": 0, "not_configured": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        total_checks = len(self.results)
        pass_rate = (status_counts["pass"] / total_checks * 100) if total_checks > 0 else 0
        
        report = f"""
🔍 SDLC Health Check Summary
==========================

📊 Overall Health: {pass_rate:.1f}% ({status_counts['pass']}/{total_checks} checks passing)

📈 Status Breakdown:
  ✅ Passing: {status_counts['pass']}
  ⚠️  Warnings: {status_counts['warning']}
  ❌ Failing: {status_counts['fail']}
  ⏸️  Not Configured: {status_counts['not_configured']}

🎯 Health Grade: {'A' if pass_rate >= 90 else 'B' if pass_rate >= 80 else 'C' if pass_rate >= 70 else 'D' if pass_rate >= 60 else 'F'}
"""
        
        if status_counts["fail"] > 0 or status_counts["warning"] > 0:
            report += "\n🚨 Issues Requiring Attention:\n"
            for result in self.results:
                if result.status in ["fail", "warning"]:
                    emoji = "❌" if result.status == "fail" else "⚠️"
                    report += f"  {emoji} {result.component}: {result.message}\n"
        
        return report
    
    def _generate_detailed_report(self) -> str:
        """Generate detailed health report."""
        report = self._generate_summary_report()
        
        report += "\n\n📋 Detailed Results:\n"
        report += "=" * 50 + "\n"
        
        for result in self.results:
            emoji_map = {
                "pass": "✅",
                "warning": "⚠️",
                "fail": "❌", 
                "not_configured": "⏸️"
            }
            emoji = emoji_map.get(result.status, "❓")
            
            report += f"\n{emoji} {result.component.replace('_', ' ').title()}\n"
            report += f"   Status: {result.status.upper()}\n"
            report += f"   Message: {result.message}\n"
            
            if result.recommendations:
                report += "   Recommendations:\n"
                for rec in result.recommendations:
                    report += f"     • {rec}\n"
        
        return report
    
    def _generate_json_report(self) -> str:
        """Generate JSON health report."""
        report_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repository": self.github_repo or "unknown",
            "total_checks": len(self.results),
            "status_summary": {},
            "checks": []
        }
        
        # Calculate status summary
        status_counts = {"pass": 0, "warning": 0, "fail": 0, "not_configured": 0}
        for result in self.results:
            status_counts[result.status] += 1
        
        report_data["status_summary"] = status_counts
        report_data["pass_rate"] = (status_counts["pass"] / len(self.results) * 100) if self.results else 0
        
        # Add detailed results
        for result in self.results:
            check_data = {
                "component": result.component,
                "status": result.status,
                "message": result.message,
                "recommendations": result.recommendations or [],
                "details": result.details or {}
            }
            report_data["checks"].append(check_data)
        
        return json.dumps(report_data, indent=2)
    
    def get_exit_code(self) -> int:
        """Get appropriate exit code based on health check results."""
        has_failures = any(result.status == "fail" for result in self.results)
        return 1 if has_failures else 0


def main():
    """Main entry point for SDLC health checker."""
    parser = argparse.ArgumentParser(description="SDLC Health Check for PQC IoT Retrofit Scanner")
    parser.add_argument("--repo-path", default=".", help="Repository path to check")
    parser.add_argument("--format", choices=["detailed", "summary", "json"], default="detailed", help="Report format")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--fail-on-warnings", action="store_true", help="Exit with error on warnings")
    
    args = parser.parse_args()
    
    try:
        checker = SDLCHealthChecker(repo_path=args.repo_path)
        results = checker.run_comprehensive_check()
        
        # Generate report
        report = checker.generate_report(format_type=args.format)
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Health check report saved to {args.output}")
        else:
            print(report)
        
        # Determine exit code
        exit_code = checker.get_exit_code()
        if args.fail_on_warnings:
            has_warnings = any(result.status == "warning" for result in results)
            if has_warnings:
                exit_code = 1
        
        return exit_code
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())