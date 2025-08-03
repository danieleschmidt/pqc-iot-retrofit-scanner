"""Database and persistence layer for PQC IoT Retrofit Scanner."""

from .connection import DatabaseManager
from .models import (
    ScanResult,
    VulnerabilityRecord,
    PatchRecord,
    FirmwareMetadata,
    AnalysisSession
)
from .repositories import (
    ScanResultRepository,
    VulnerabilityRepository,
    PatchRepository,
    FirmwareRepository
)

__all__ = [
    'DatabaseManager',
    'ScanResult',
    'VulnerabilityRecord', 
    'PatchRecord',
    'FirmwareMetadata',
    'AnalysisSession',
    'ScanResultRepository',
    'VulnerabilityRepository',
    'PatchRepository',
    'FirmwareRepository'
]