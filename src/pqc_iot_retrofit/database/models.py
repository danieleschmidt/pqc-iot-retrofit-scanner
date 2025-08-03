"""Data models for PQC IoT Retrofit Scanner database."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import json


class SessionStatus(Enum):
    """Analysis session status."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionType(Enum):
    """Analysis session type."""
    SCAN = "scan"
    PATCH = "patch"
    ANALYZE = "analyze"
    BATCH = "batch"


@dataclass
class FirmwareMetadata:
    """Firmware metadata record."""
    file_path: str
    file_hash: str
    file_size: int
    architecture: str
    base_address: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to strings
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod 
    def from_dict(cls, data: Dict[str, Any]) -> 'FirmwareMetadata':
        """Create from dictionary."""
        # Convert string timestamps back to datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class AnalysisSession:
    """Analysis session record."""
    firmware_id: int
    session_type: SessionType
    configuration: Dict[str, Any]
    status: SessionStatus = SessionStatus.RUNNING
    id: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'firmware_id': self.firmware_id,
            'session_type': self.session_type.value,
            'configuration': json.dumps(self.configuration),
            'status': self.status.value,
            'error_message': self.error_message
        }
        
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSession':
        """Create from dictionary."""
        # Parse JSON configuration
        if isinstance(data['configuration'], str):
            data['configuration'] = json.loads(data['configuration'])
        
        # Convert enum strings
        data['session_type'] = SessionType(data['session_type'])
        data['status'] = SessionStatus(data['status'])
        
        # Convert timestamps
        if 'started_at' in data and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            
        return cls(**data)
    
    def duration_seconds(self) -> Optional[float]:
        """Get session duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def mark_completed(self):
        """Mark session as completed."""
        self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self, error_message: str):
        """Mark session as failed."""
        self.status = SessionStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.now()


@dataclass
class ScanResult:
    """Scan result record."""
    session_id: int
    total_vulnerabilities: int
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    scan_duration_ms: Optional[int] = None
    memory_constraints: Optional[Dict[str, int]] = None
    recommendations: Optional[List[str]] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'session_id': self.session_id,
            'total_vulnerabilities': self.total_vulnerabilities,
            'critical_count': self.critical_count,
            'high_count': self.high_count,
            'medium_count': self.medium_count,
            'low_count': self.low_count,
            'scan_duration_ms': self.scan_duration_ms,
            'memory_constraints': json.dumps(self.memory_constraints) if self.memory_constraints else None,
            'recommendations': json.dumps(self.recommendations) if self.recommendations else None
        }
        
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScanResult':
        """Create from dictionary."""
        # Parse JSON fields
        if data.get('memory_constraints') and isinstance(data['memory_constraints'], str):
            data['memory_constraints'] = json.loads(data['memory_constraints'])
        if data.get('recommendations') and isinstance(data['recommendations'], str):
            data['recommendations'] = json.loads(data['recommendations'])
        
        # Convert timestamp
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            
        return cls(**data)
    
    def risk_distribution(self) -> Dict[str, int]:
        """Get risk level distribution."""
        return {
            'critical': self.critical_count,
            'high': self.high_count,
            'medium': self.medium_count,
            'low': self.low_count
        }
    
    def risk_score(self) -> float:
        """Calculate weighted risk score (0-100)."""
        weights = {'critical': 10, 'high': 7, 'medium': 4, 'low': 1}
        total_score = (
            self.critical_count * weights['critical'] +
            self.high_count * weights['high'] +
            self.medium_count * weights['medium'] +
            self.low_count * weights['low']
        )
        
        if self.total_vulnerabilities == 0:
            return 0.0
        
        max_possible = self.total_vulnerabilities * weights['critical']
        return min(100.0, (total_score / max_possible) * 100)


@dataclass
class VulnerabilityRecord:
    """Vulnerability record."""
    scan_result_id: int
    algorithm: str
    address: int
    function_name: str
    risk_level: str
    description: str
    mitigation: str
    stack_usage: int
    available_stack: int
    key_size: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VulnerabilityRecord':
        """Create from dictionary."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    def severity_score(self) -> int:
        """Get numerical severity score (1-10)."""
        scores = {'low': 2, 'medium': 5, 'high': 8, 'critical': 10}
        return scores.get(self.risk_level.lower(), 1)
    
    def memory_pressure(self) -> float:
        """Calculate memory pressure ratio (0-1)."""
        if self.available_stack <= 0:
            return 1.0
        return min(1.0, self.stack_usage / self.available_stack)
    
    def is_patchable(self) -> bool:
        """Check if vulnerability is patchable based on memory constraints."""
        # Rough heuristic: need at least 2KB available for PQC patches
        return self.available_stack >= 2048


@dataclass
class PatchRecord:
    """Patch record."""
    vulnerability_id: int
    pqc_algorithm: str
    target_device: str
    security_level: int
    optimization_level: str
    patch_size: int
    verification_hash: str
    metadata: Dict[str, Any]
    installation_script: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'vulnerability_id': self.vulnerability_id,
            'pqc_algorithm': self.pqc_algorithm,
            'target_device': self.target_device,
            'security_level': self.security_level,
            'optimization_level': self.optimization_level,
            'patch_size': self.patch_size,
            'verification_hash': self.verification_hash,
            'metadata': json.dumps(self.metadata),
            'installation_script': self.installation_script
        }
        
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatchRecord':
        """Create from dictionary."""
        # Parse JSON metadata
        if isinstance(data['metadata'], str):
            data['metadata'] = json.loads(data['metadata'])
        
        # Convert timestamp
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            
        return cls(**data)
    
    def estimated_performance_impact(self) -> Dict[str, Any]:
        """Estimate performance impact from metadata."""
        metadata = self.metadata
        performance = metadata.get('performance', {})
        
        # Calculate relative impact compared to classical crypto
        # These are rough estimates for demonstration
        classical_cycles = {
            'rsa': {'keygen': 100000, 'sign': 50000, 'verify': 5000},
            'ecdsa': {'keygen': 50000, 'sign': 25000, 'verify': 30000}
        }
        
        impact = {}
        for operation in ['keygen', 'sign', 'verify']:
            pqc_cycles = performance.get(f'{operation}_cycles', 0)
            if pqc_cycles > 0:
                # Assume RSA baseline for comparison
                classical = classical_cycles['rsa'].get(operation, 1)
                impact[f'{operation}_slowdown'] = pqc_cycles / classical
        
        return impact
    
    def memory_requirements(self) -> Dict[str, int]:
        """Get memory requirements from metadata."""
        memory = self.metadata.get('memory', {})
        return {
            'stack_usage': memory.get('stack_usage', 0),
            'flash_usage': memory.get('flash_usage', 0),
            'public_key_size': memory.get('public_key_size', 0),
            'private_key_size': memory.get('private_key_size', 0),
            'signature_size': memory.get('signature_size', 0),
            'ciphertext_size': memory.get('ciphertext_size', 0)
        }


@dataclass 
class BatchAnalysisResult:
    """Result of batch firmware analysis."""
    total_files: int
    successful_scans: int
    failed_scans: int
    total_vulnerabilities: int
    unique_algorithms: List[str]
    risk_distribution: Dict[str, int]
    processing_time_seconds: float
    session_ids: List[int]
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_scans / self.total_files) * 100
    
    def average_vulnerabilities_per_file(self) -> float:
        """Calculate average vulnerabilities per successfully scanned file."""
        if self.successful_scans == 0:
            return 0.0
        return self.total_vulnerabilities / self.successful_scans
    
    def most_common_algorithm(self) -> Optional[str]:
        """Get most commonly found vulnerable algorithm."""
        if not self.unique_algorithms:
            return None
        return max(self.unique_algorithms, key=self.unique_algorithms.count)


@dataclass
class AnalysisMetrics:
    """Performance and usage metrics."""
    total_scans: int
    total_patches_generated: int
    average_scan_time_ms: float
    average_vulnerabilities_per_scan: float
    most_vulnerable_architecture: str
    most_common_vulnerability: str
    patch_success_rate: float
    cache_hit_rate: float
    
    def efficiency_score(self) -> float:
        """Calculate efficiency score based on metrics."""
        # Weighted score based on various factors
        time_score = max(0, 100 - (self.average_scan_time_ms / 1000))  # Penalty for slow scans
        success_score = self.patch_success_rate
        cache_score = self.cache_hit_rate
        
        return (time_score * 0.3 + success_score * 0.5 + cache_score * 0.2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)