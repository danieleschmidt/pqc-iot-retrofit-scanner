"""Health check and monitoring utilities for PQC IoT Retrofit Scanner."""

import os
import sys
import time
import json
import logging
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
from pathlib import Path


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = 0
    duration_ms: float = 0

    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class HealthChecker:
    """Comprehensive health checking for the application."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.checks: Dict[str, callable] = {
            "python_version": self._check_python_version,
            "dependencies": self._check_dependencies,
            "memory": self._check_memory,
            "disk_space": self._check_disk_space,
            "permissions": self._check_permissions,
            "crypto_libraries": self._check_crypto_libraries,
            "container_env": self._check_container_environment,
            "system_resources": self._check_system_resources
        }
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all health checks and return results."""
        results = []
        
        for check_name, check_func in self.checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.duration_ms = (time.time() - start_time) * 1000
                results.append(result)
            except Exception as e:
                self.logger.exception(f"Health check {check_name} failed with exception")
                results.append(HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check failed with exception: {str(e)}",
                    duration_ms=(time.time() - start_time) * 1000 if 'start_time' in locals() else 0
                ))
        
        return results
    
    def run_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if check_name not in self.checks:
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.UNKNOWN,
                message=f"Unknown check: {check_name}"
            )
        
        try:
            start_time = time.time()
            result = self.checks[check_name]()
            result.duration_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            self.logger.exception(f"Health check {check_name} failed")
            return HealthCheckResult(
                name=check_name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}"
            )
    
    def _check_python_version(self) -> HealthCheckResult:
        """Check Python version compatibility."""
        current_version = sys.version_info
        required_major, required_minor = 3, 8
        
        if current_version.major < required_major or current_version.minor < required_minor:
            return HealthCheckResult(
                name="python_version",
                status=HealthStatus.CRITICAL,
                message=f"Python {required_major}.{required_minor}+ required, got {current_version.major}.{current_version.minor}",
                details={
                    "current": f"{current_version.major}.{current_version.minor}.{current_version.micro}",
                    "required": f"{required_major}.{required_minor}+",
                    "executable": sys.executable
                }
            )
        
        return HealthCheckResult(
            name="python_version",
            status=HealthStatus.HEALTHY,
            message=f"Python version {current_version.major}.{current_version.minor}.{current_version.micro} is compatible",
            details={
                "version": f"{current_version.major}.{current_version.minor}.{current_version.micro}",
                "executable": sys.executable
            }
        )
    
    def _check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies are available."""
        critical_deps = [
            "click", "cryptography", "pyyaml", "requests", "rich"
        ]
        
        missing_deps = []
        available_deps = {}
        
        for dep in critical_deps:
            try:
                __import__(dep)
                module = sys.modules[dep]
                version = getattr(module, "__version__", "unknown")
                available_deps[dep] = version
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            return HealthCheckResult(
                name="dependencies",
                status=HealthStatus.CRITICAL,
                message=f"Missing critical dependencies: {', '.join(missing_deps)}",
                details={
                    "missing": missing_deps,
                    "available": available_deps
                }
            )
        
        return HealthCheckResult(
            name="dependencies",
            status=HealthStatus.HEALTHY,
            message="All critical dependencies are available",
            details={"dependencies": available_deps}
        )
    
    def _check_memory(self) -> HealthCheckResult:
        """Check system memory availability."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            percent_used = memory.percent
            
            if available_gb < 0.5:  # Less than 500MB available
                status = HealthStatus.CRITICAL
                message = f"Low memory: only {available_gb:.1f}GB available"
            elif percent_used > 90:
                status = HealthStatus.WARNING
                message = f"High memory usage: {percent_used:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory OK: {available_gb:.1f}GB available ({100-percent_used:.1f}% free)"
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                details={
                    "total_gb": memory.total / (1024**3),
                    "available_gb": available_gb,
                    "percent_used": percent_used
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check memory: {str(e)}"
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage(".")
            available_gb = disk_usage.free / (1024**3)
            percent_used = (disk_usage.used / disk_usage.total) * 100
            
            if available_gb < 1.0:  # Less than 1GB available
                status = HealthStatus.CRITICAL
                message = f"Low disk space: only {available_gb:.1f}GB available"
            elif percent_used > 90:
                status = HealthStatus.WARNING
                message = f"High disk usage: {percent_used:.1f}% used"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {available_gb:.1f}GB available"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "total_gb": disk_usage.total / (1024**3),
                    "available_gb": available_gb,
                    "percent_used": percent_used
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check disk space: {str(e)}"
            )
    
    def _check_permissions(self) -> HealthCheckResult:
        """Check file system permissions."""
        test_locations = [
            ("current_directory", "."),
            ("temp_directory", tempfile.gettempdir())
        ]
        
        permission_issues = []
        working_locations = []
        
        for name, path in test_locations:
            try:
                # Test write permission
                test_file = os.path.join(path, f".pqc_health_check_{os.getpid()}")
                with open(test_file, "w") as f:
                    f.write("health check")
                os.remove(test_file)
                working_locations.append(name)
            except Exception as e:
                permission_issues.append(f"{name}: {str(e)}")
        
        if permission_issues:
            status = HealthStatus.WARNING if working_locations else HealthStatus.CRITICAL
            message = f"Permission issues: {'; '.join(permission_issues)}"
        else:
            status = HealthStatus.HEALTHY
            message = "File system permissions OK"
        
        return HealthCheckResult(
            name="permissions",
            status=status,
            message=message,
            details={
                "working_locations": working_locations,
                "issues": permission_issues
            }
        )
    
    def _check_crypto_libraries(self) -> HealthCheckResult:
        """Check cryptographic library functionality."""
        try:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.backends import default_backend
            
            # Test basic cryptographic operations
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            test_data = b"health check test"
            digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
            digest.update(test_data)
            hash_result = digest.finalize()
            
            return HealthCheckResult(
                name="crypto_libraries",
                status=HealthStatus.HEALTHY,
                message="Cryptographic libraries working correctly",
                details={
                    "test_hash_length": len(hash_result),
                    "key_size": private_key.key_size
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="crypto_libraries",
                status=HealthStatus.CRITICAL,
                message=f"Cryptographic library test failed: {str(e)}"
            )
    
    def _check_container_environment(self) -> HealthCheckResult:
        """Check if running in container and container health."""
        is_container = (
            os.path.exists("/.dockerenv") or
            os.getenv("container") is not None or
            os.getenv("PQC_CONTAINER") == "1"
        )
        
        details = {
            "is_container": is_container,
            "cgroup_info": self._get_cgroup_info() if is_container else None
        }
        
        if is_container:
            # Additional container-specific checks
            container_issues = []
            
            # Check for proper resource limits
            if details["cgroup_info"]:
                memory_limit = details["cgroup_info"].get("memory_limit")
                if memory_limit and memory_limit < 128 * 1024 * 1024:  # Less than 128MB
                    container_issues.append("Memory limit too low")
            
            if container_issues:
                status = HealthStatus.WARNING
                message = f"Container issues: {'; '.join(container_issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Container environment OK"
        else:
            status = HealthStatus.HEALTHY
            message = "Running in native environment"
        
        return HealthCheckResult(
            name="container_env",
            status=status,
            message=message,
            details=details
        )
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            issues = []
            if cpu_percent > 95:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if load_avg[0] > psutil.cpu_count() * 2:
                issues.append(f"High load average: {load_avg[0]:.2f}")
            
            if issues:
                status = HealthStatus.WARNING
                message = f"System resource issues: {'; '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources OK"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "load_average": load_avg,
                    "cpu_count": psutil.cpu_count()
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check system resources: {str(e)}"
            )
    
    def _get_cgroup_info(self) -> Optional[Dict[str, Any]]:
        """Get container cgroup information."""
        try:
            cgroup_info = {}
            
            # Try to read memory limit
            try:
                with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r") as f:
                    cgroup_info["memory_limit"] = int(f.read().strip())
            except:
                pass
            
            # Try to read CPU quota
            try:
                with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r") as f:
                    quota = int(f.read().strip())
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r") as f:
                    period = int(f.read().strip())
                if quota > 0:
                    cgroup_info["cpu_quota"] = quota / period
            except:
                pass
            
            return cgroup_info if cgroup_info else None
        except:
            return None
    
    def get_health_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Generate health summary from check results."""
        status_counts = {status.value: 0 for status in HealthStatus}
        
        for result in results:
            status_counts[result.status.value] += 1
        
        # Determine overall health
        if status_counts["critical"] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts["warning"] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts["unknown"] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "total_checks": len(results),
            "status_counts": status_counts,
            "checks": [
                {
                    "name": result.name,
                    "status": result.status.value,
                    "message": result.message,
                    "duration_ms": result.duration_ms
                }
                for result in results
            ]
        }


# Global health checker instance
health_checker = HealthChecker()


def check_health() -> Dict[str, Any]:
    """Convenience function to run all health checks."""
    results = health_checker.run_all_checks()
    return health_checker.get_health_summary(results)