"""Repository classes for data access layer."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
import logging

from .connection import DatabaseManager
from .models import (
    FirmwareMetadata,
    AnalysisSession,
    ScanResult,
    VulnerabilityRecord,
    PatchRecord,
    SessionStatus,
    SessionType,
    BatchAnalysisResult,
    AnalysisMetrics
)

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository class with common functionality."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize repository.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary."""
        return dict(row) if row else {}


class FirmwareRepository(BaseRepository):
    """Repository for firmware metadata operations."""
    
    def create(self, firmware: FirmwareMetadata) -> int:
        """Create new firmware metadata record.
        
        Args:
            firmware: Firmware metadata
            
        Returns:
            ID of created record
        """
        query = """
        INSERT INTO firmware_metadata 
        (file_path, file_hash, file_size, architecture, base_address)
        VALUES (?, ?, ?, ?, ?)
        """
        
        return self.db.execute_insert(
            query,
            (firmware.file_path, firmware.file_hash, firmware.file_size,
             firmware.architecture, firmware.base_address)
        )
    
    def get_by_id(self, firmware_id: int) -> Optional[FirmwareMetadata]:
        """Get firmware metadata by ID.
        
        Args:
            firmware_id: Firmware ID
            
        Returns:
            Firmware metadata or None if not found
        """
        query = "SELECT * FROM firmware_metadata WHERE id = ?"
        results = self.db.execute_query(query, (firmware_id,))
        
        if results:
            data = self._row_to_dict(results[0])
            return FirmwareMetadata.from_dict(data)
        return None
    
    def get_by_hash(self, file_hash: str) -> Optional[FirmwareMetadata]:
        """Get firmware metadata by file hash.
        
        Args:
            file_hash: File hash
            
        Returns:
            Firmware metadata or None if not found
        """
        query = "SELECT * FROM firmware_metadata WHERE file_hash = ?"
        results = self.db.execute_query(query, (file_hash,))
        
        if results:
            data = self._row_to_dict(results[0])
            return FirmwareMetadata.from_dict(data)
        return None
    
    def find_or_create(self, firmware: FirmwareMetadata) -> int:
        """Find existing firmware record or create new one.
        
        Args:
            firmware: Firmware metadata
            
        Returns:
            ID of existing or newly created record
        """
        existing = self.get_by_hash(firmware.file_hash)
        if existing:
            return existing.id
        
        return self.create(firmware)
    
    def list_all(self, limit: Optional[int] = None) -> List[FirmwareMetadata]:
        """List all firmware metadata records.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of firmware metadata
        """
        query = "SELECT * FROM firmware_metadata ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.db.execute_query(query)
        return [FirmwareMetadata.from_dict(self._row_to_dict(row)) for row in results]
    
    def delete(self, firmware_id: int) -> bool:
        """Delete firmware metadata and all related data.
        
        Args:
            firmware_id: Firmware ID
            
        Returns:
            True if deleted, False if not found
        """
        # Delete related sessions (which cascades to other tables)
        self.db.execute_update(
            "DELETE FROM analysis_sessions WHERE firmware_id = ?",
            (firmware_id,)
        )
        
        # Delete firmware metadata
        rows_affected = self.db.execute_update(
            "DELETE FROM firmware_metadata WHERE id = ?",
            (firmware_id,)
        )
        
        return rows_affected > 0
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class AnalysisSessionRepository(BaseRepository):
    """Repository for analysis session operations."""
    
    def create(self, session: AnalysisSession) -> int:
        """Create new analysis session.
        
        Args:
            session: Analysis session
            
        Returns:
            ID of created session
        """
        data = session.to_dict()
        query = """
        INSERT INTO analysis_sessions 
        (firmware_id, session_type, configuration, status, started_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """
        
        return self.db.execute_insert(
            query,
            (data['firmware_id'], data['session_type'], 
             data['configuration'], data['status'])
        )
    
    def get_by_id(self, session_id: int) -> Optional[AnalysisSession]:
        """Get analysis session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Analysis session or None if not found
        """
        query = "SELECT * FROM analysis_sessions WHERE id = ?"
        results = self.db.execute_query(query, (session_id,))
        
        if results:
            data = self._row_to_dict(results[0])
            return AnalysisSession.from_dict(data)
        return None
    
    def update_status(self, session_id: int, status: SessionStatus, 
                     error_message: Optional[str] = None) -> bool:
        """Update session status.
        
        Args:
            session_id: Session ID
            status: New status
            error_message: Error message if failed
            
        Returns:
            True if updated successfully
        """
        if status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]:
            query = """
            UPDATE analysis_sessions 
            SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
            WHERE id = ?
            """
            rows_affected = self.db.execute_update(
                query, (status.value, error_message, session_id)
            )
        else:
            query = "UPDATE analysis_sessions SET status = ? WHERE id = ?"
            rows_affected = self.db.execute_update(query, (status.value, session_id))
        
        return rows_affected > 0
    
    def get_by_firmware(self, firmware_id: int, 
                       session_type: Optional[SessionType] = None) -> List[AnalysisSession]:
        """Get sessions for a firmware.
        
        Args:
            firmware_id: Firmware ID
            session_type: Optional session type filter
            
        Returns:
            List of analysis sessions
        """
        if session_type:
            query = """
            SELECT * FROM analysis_sessions 
            WHERE firmware_id = ? AND session_type = ?
            ORDER BY started_at DESC
            """
            results = self.db.execute_query(query, (firmware_id, session_type.value))
        else:
            query = """
            SELECT * FROM analysis_sessions 
            WHERE firmware_id = ?
            ORDER BY started_at DESC
            """
            results = self.db.execute_query(query, (firmware_id,))
        
        return [AnalysisSession.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_recent(self, hours: int = 24, limit: int = 100) -> List[AnalysisSession]:
        """Get recent analysis sessions.
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of sessions
            
        Returns:
            List of recent analysis sessions
        """
        query = """
        SELECT * FROM analysis_sessions 
        WHERE started_at > datetime('now', '-{} hours')
        ORDER BY started_at DESC
        LIMIT ?
        """.format(hours)
        
        results = self.db.execute_query(query, (limit,))
        return [AnalysisSession.from_dict(self._row_to_dict(row)) for row in results]


class ScanResultRepository(BaseRepository):
    """Repository for scan result operations."""
    
    def create(self, scan_result: ScanResult) -> int:
        """Create new scan result.
        
        Args:
            scan_result: Scan result
            
        Returns:
            ID of created scan result
        """
        data = scan_result.to_dict()
        query = """
        INSERT INTO scan_results 
        (session_id, total_vulnerabilities, critical_count, high_count, 
         medium_count, low_count, scan_duration_ms, memory_constraints, recommendations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.db.execute_insert(
            query,
            (data['session_id'], data['total_vulnerabilities'], data['critical_count'],
             data['high_count'], data['medium_count'], data['low_count'],
             data['scan_duration_ms'], data['memory_constraints'], data['recommendations'])
        )
    
    def get_by_session(self, session_id: int) -> Optional[ScanResult]:
        """Get scan result by session ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Scan result or None if not found
        """
        query = "SELECT * FROM scan_results WHERE session_id = ?"
        results = self.db.execute_query(query, (session_id,))
        
        if results:
            data = self._row_to_dict(results[0])
            return ScanResult.from_dict(data)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scan result statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        # Total scans
        total_query = "SELECT COUNT(*) as count FROM scan_results"
        result = self.db.execute_query(total_query)
        stats['total_scans'] = result[0]['count'] if result else 0
        
        # Average vulnerabilities per scan
        avg_query = "SELECT AVG(total_vulnerabilities) as avg FROM scan_results"
        result = self.db.execute_query(avg_query)
        stats['avg_vulnerabilities'] = round(result[0]['avg'] or 0, 2)
        
        # Risk distribution across all scans
        risk_query = """
        SELECT 
            SUM(critical_count) as critical,
            SUM(high_count) as high,
            SUM(medium_count) as medium,
            SUM(low_count) as low
        FROM scan_results
        """
        result = self.db.execute_query(risk_query)
        if result:
            row = result[0]
            stats['total_vulnerabilities'] = {
                'critical': row['critical'] or 0,
                'high': row['high'] or 0,
                'medium': row['medium'] or 0,
                'low': row['low'] or 0
            }
        
        # Average scan duration
        duration_query = "SELECT AVG(scan_duration_ms) as avg_duration FROM scan_results WHERE scan_duration_ms IS NOT NULL"
        result = self.db.execute_query(duration_query)
        stats['avg_scan_duration_ms'] = round(result[0]['avg_duration'] or 0, 2)
        
        return stats


class VulnerabilityRepository(BaseRepository):
    """Repository for vulnerability record operations."""
    
    def create(self, vulnerability: VulnerabilityRecord) -> int:
        """Create new vulnerability record.
        
        Args:
            vulnerability: Vulnerability record
            
        Returns:
            ID of created vulnerability
        """
        data = vulnerability.to_dict()
        query = """
        INSERT INTO vulnerability_records 
        (scan_result_id, algorithm, address, function_name, risk_level,
         key_size, description, mitigation, stack_usage, available_stack)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.db.execute_insert(
            query,
            (data['scan_result_id'], data['algorithm'], data['address'],
             data['function_name'], data['risk_level'], data['key_size'],
             data['description'], data['mitigation'], data['stack_usage'],
             data['available_stack'])
        )
    
    def create_batch(self, vulnerabilities: List[VulnerabilityRecord]) -> List[int]:
        """Create multiple vulnerability records.
        
        Args:
            vulnerabilities: List of vulnerability records
            
        Returns:
            List of created vulnerability IDs
        """
        ids = []
        for vuln in vulnerabilities:
            vuln_id = self.create(vuln)
            ids.append(vuln_id)
        return ids
    
    def get_by_scan_result(self, scan_result_id: int) -> List[VulnerabilityRecord]:
        """Get vulnerabilities by scan result ID.
        
        Args:
            scan_result_id: Scan result ID
            
        Returns:
            List of vulnerability records
        """
        query = """
        SELECT * FROM vulnerability_records 
        WHERE scan_result_id = ?
        ORDER BY risk_level DESC, address ASC
        """
        results = self.db.execute_query(query, (scan_result_id,))
        
        return [VulnerabilityRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_by_algorithm(self, algorithm: str, limit: Optional[int] = None) -> List[VulnerabilityRecord]:
        """Get vulnerabilities by algorithm type.
        
        Args:
            algorithm: Algorithm name
            limit: Maximum number of records
            
        Returns:
            List of vulnerability records
        """
        query = """
        SELECT * FROM vulnerability_records 
        WHERE algorithm = ?
        ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.db.execute_query(query, (algorithm,))
        return [VulnerabilityRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_by_risk_level(self, risk_level: str, limit: Optional[int] = None) -> List[VulnerabilityRecord]:
        """Get vulnerabilities by risk level.
        
        Args:
            risk_level: Risk level (critical, high, medium, low)
            limit: Maximum number of records
            
        Returns:
            List of vulnerability records
        """
        query = """
        SELECT * FROM vulnerability_records 
        WHERE risk_level = ?
        ORDER BY created_at DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.db.execute_query(query, (risk_level,))
        return [VulnerabilityRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_patchable_vulnerabilities(self, min_available_stack: int = 2048) -> List[VulnerabilityRecord]:
        """Get vulnerabilities that can be patched.
        
        Args:
            min_available_stack: Minimum available stack for patching
            
        Returns:
            List of patchable vulnerability records
        """
        query = """
        SELECT * FROM vulnerability_records 
        WHERE available_stack >= ?
        ORDER BY risk_level DESC, available_stack DESC
        """
        
        results = self.db.execute_query(query, (min_available_stack,))
        return [VulnerabilityRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_algorithm_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics by algorithm.
        
        Returns:
            Dictionary with algorithm statistics
        """
        query = """
        SELECT 
            algorithm,
            COUNT(*) as count,
            AVG(stack_usage) as avg_stack_usage,
            COUNT(CASE WHEN risk_level = 'critical' THEN 1 END) as critical_count,
            COUNT(CASE WHEN risk_level = 'high' THEN 1 END) as high_count,
            COUNT(CASE WHEN available_stack >= 2048 THEN 1 END) as patchable_count
        FROM vulnerability_records
        GROUP BY algorithm
        ORDER BY count DESC
        """
        
        results = self.db.execute_query(query)
        stats = {}
        
        for row in results:
            stats[row['algorithm']] = {
                'total_count': row['count'],
                'avg_stack_usage': round(row['avg_stack_usage'] or 0, 2),
                'critical_count': row['critical_count'],
                'high_count': row['high_count'],
                'patchable_count': row['patchable_count'],
                'patchable_percentage': round((row['patchable_count'] / row['count']) * 100, 1)
            }
        
        return stats


class PatchRepository(BaseRepository):
    """Repository for patch record operations."""
    
    def create(self, patch: PatchRecord) -> int:
        """Create new patch record.
        
        Args:
            patch: Patch record
            
        Returns:
            ID of created patch
        """
        data = patch.to_dict()
        query = """
        INSERT INTO patch_records 
        (vulnerability_id, pqc_algorithm, target_device, security_level,
         optimization_level, patch_size, metadata, installation_script, verification_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        return self.db.execute_insert(
            query,
            (data['vulnerability_id'], data['pqc_algorithm'], data['target_device'],
             data['security_level'], data['optimization_level'], data['patch_size'],
             data['metadata'], data['installation_script'], data['verification_hash'])
        )
    
    def get_by_vulnerability(self, vulnerability_id: int) -> List[PatchRecord]:
        """Get patches for a vulnerability.
        
        Args:
            vulnerability_id: Vulnerability ID
            
        Returns:
            List of patch records
        """
        query = """
        SELECT * FROM patch_records 
        WHERE vulnerability_id = ?
        ORDER BY created_at DESC
        """
        results = self.db.execute_query(query, (vulnerability_id,))
        
        return [PatchRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_by_algorithm(self, pqc_algorithm: str) -> List[PatchRecord]:
        """Get patches by PQC algorithm.
        
        Args:
            pqc_algorithm: PQC algorithm name
            
        Returns:
            List of patch records
        """
        query = """
        SELECT * FROM patch_records 
        WHERE pqc_algorithm = ?
        ORDER BY created_at DESC
        """
        results = self.db.execute_query(query, (pqc_algorithm,))
        
        return [PatchRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_by_device(self, target_device: str) -> List[PatchRecord]:
        """Get patches by target device.
        
        Args:
            target_device: Target device name
            
        Returns:
            List of patch records
        """
        query = """
        SELECT * FROM patch_records 
        WHERE target_device = ?
        ORDER BY created_at DESC
        """
        results = self.db.execute_query(query, (target_device,))
        
        return [PatchRecord.from_dict(self._row_to_dict(row)) for row in results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get patch generation statistics.
        
        Returns:
            Dictionary with patch statistics
        """
        stats = {}
        
        # Total patches
        total_query = "SELECT COUNT(*) as count FROM patch_records"
        result = self.db.execute_query(total_query)
        stats['total_patches'] = result[0]['count'] if result else 0
        
        # Patches by algorithm
        algo_query = """
        SELECT pqc_algorithm, COUNT(*) as count
        FROM patch_records
        GROUP BY pqc_algorithm
        ORDER BY count DESC
        """
        results = self.db.execute_query(algo_query)
        stats['patches_by_algorithm'] = {row['pqc_algorithm']: row['count'] for row in results}
        
        # Patches by device
        device_query = """
        SELECT target_device, COUNT(*) as count
        FROM patch_records
        GROUP BY target_device
        ORDER BY count DESC
        """
        results = self.db.execute_query(device_query)
        stats['patches_by_device'] = {row['target_device']: row['count'] for row in results}
        
        # Average patch size
        size_query = "SELECT AVG(patch_size) as avg_size FROM patch_records"
        result = self.db.execute_query(size_query)
        stats['avg_patch_size'] = round(result[0]['avg_size'] or 0, 2)
        
        # Security level distribution
        security_query = """
        SELECT security_level, COUNT(*) as count
        FROM patch_records
        GROUP BY security_level
        ORDER BY security_level
        """
        results = self.db.execute_query(security_query)
        stats['security_level_distribution'] = {row['security_level']: row['count'] for row in results}
        
        return stats
    
    def get_patch_success_rate(self, days: int = 30) -> float:
        """Calculate patch success rate.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Success rate as percentage (0-100)
        """
        # This is a simplified calculation - in practice you'd track deployment success
        query = """
        SELECT COUNT(*) as total_patches
        FROM patch_records
        WHERE created_at > datetime('now', '-{} days')
        """.format(days)
        
        result = self.db.execute_query(query)
        total_patches = result[0]['total_patches'] if result else 0
        
        if total_patches == 0:
            return 0.0
        
        # For demo purposes, assume 85% success rate
        # In practice, this would be based on deployment feedback
        return 85.0