"""Unit tests for database layer components."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime

from pqc_iot_retrofit.database.connection import DatabaseManager
from pqc_iot_retrofit.database.models import (
    FirmwareMetadata, AnalysisSession, ScanResult, VulnerabilityRecord, 
    PatchRecord, SessionStatus, SessionType
)
from pqc_iot_retrofit.database.repositories import (
    FirmwareRepository, AnalysisSessionRepository, ScanResultRepository,
    VulnerabilityRepository, PatchRepository
)


class TestDatabaseManager:
    """Test database manager functionality."""
    
    def test_init_in_memory_database(self):
        """Test initialization of in-memory database."""
        db_manager = DatabaseManager()
        
        # Test basic connectivity
        with db_manager.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
    
    def test_init_file_database(self):
        """Test initialization of file-based database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            db_manager = DatabaseManager(db_path)
            
            # Test that database file was created
            assert Path(db_path).exists()
            
            # Test connectivity
            with db_manager.get_connection() as conn:
                result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                table_names = [row[0] for row in result]
                
                # Check that all expected tables exist
                expected_tables = [
                    'firmware_metadata', 'analysis_sessions', 'scan_results',
                    'vulnerability_records', 'patch_records', 'analysis_cache'
                ]
                
                for table in expected_tables:
                    assert table in table_names
        finally:
            Path(db_path).unlink(missing_ok=True)
    
    def test_cache_operations(self):
        """Test cache operations."""
        db_manager = DatabaseManager()
        
        # Test set and get
        test_data = {'key': 'value', 'number': 42}
        db_manager.set_cache('test_key', test_data, ttl_minutes=60)
        
        retrieved_data = db_manager.get_cache('test_key')
        assert retrieved_data == test_data
        
        # Test non-existent key
        assert db_manager.get_cache('non_existent') is None
        
        # Test clear cache
        db_manager.clear_cache('test_key')
        assert db_manager.get_cache('test_key') is None
    
    def test_statistics(self):
        """Test database statistics."""
        db_manager = DatabaseManager()
        stats = db_manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'firmware_metadata_count' in stats
        assert 'vulnerability_by_risk' in stats
        assert 'sessions_last_7_days' in stats
    
    def test_health_check(self):
        """Test database health check."""
        db_manager = DatabaseManager()
        health = db_manager.health_check()
        
        assert health['status'] in ['healthy', 'warning', 'error']
        assert isinstance(health['issues'], list)
        assert isinstance(health['recommendations'], list)


class TestFirmwareRepository:
    """Test firmware repository operations."""
    
    @pytest.fixture
    def repo(self):
        """Create firmware repository with in-memory database."""
        db_manager = DatabaseManager()
        return FirmwareRepository(db_manager)
    
    @pytest.fixture
    def sample_firmware(self):
        """Create sample firmware metadata."""
        return FirmwareMetadata(
            file_path="/path/to/firmware.bin",
            file_hash="abc123def456",
            file_size=262144,
            architecture="cortex-m4",
            base_address=0x08000000
        )
    
    def test_create_firmware(self, repo, sample_firmware):
        """Test creating firmware metadata."""
        firmware_id = repo.create(sample_firmware)
        
        assert firmware_id > 0
        
        # Verify the firmware was created
        retrieved = repo.get_by_id(firmware_id)
        assert retrieved is not None
        assert retrieved.file_path == sample_firmware.file_path
        assert retrieved.file_hash == sample_firmware.file_hash
        assert retrieved.architecture == sample_firmware.architecture
    
    def test_get_by_hash(self, repo, sample_firmware):
        """Test retrieving firmware by hash."""
        firmware_id = repo.create(sample_firmware)
        
        retrieved = repo.get_by_hash(sample_firmware.file_hash)
        assert retrieved is not None
        assert retrieved.id == firmware_id
        assert retrieved.file_hash == sample_firmware.file_hash
    
    def test_find_or_create(self, repo, sample_firmware):
        """Test find or create functionality."""
        # First call should create
        firmware_id1 = repo.find_or_create(sample_firmware)
        assert firmware_id1 > 0
        
        # Second call should find existing
        firmware_id2 = repo.find_or_create(sample_firmware)
        assert firmware_id2 == firmware_id1
    
    def test_list_all(self, repo, sample_firmware):
        """Test listing all firmware."""
        # Create multiple firmware entries
        firmware1 = sample_firmware
        firmware2 = FirmwareMetadata(
            file_path="/path/to/firmware2.bin",
            file_hash="def456ghi789",
            file_size=131072,
            architecture="esp32"
        )
        
        repo.create(firmware1)
        repo.create(firmware2)
        
        all_firmware = repo.list_all()
        assert len(all_firmware) >= 2
        
        # Test with limit
        limited = repo.list_all(limit=1)
        assert len(limited) == 1
    
    def test_delete_firmware(self, repo, sample_firmware):
        """Test deleting firmware."""
        firmware_id = repo.create(sample_firmware)
        
        # Verify it exists
        assert repo.get_by_id(firmware_id) is not None
        
        # Delete it
        success = repo.delete(firmware_id)
        assert success is True
        
        # Verify it's gone
        assert repo.get_by_id(firmware_id) is None


class TestAnalysisSessionRepository:
    """Test analysis session repository operations."""
    
    @pytest.fixture
    def repo(self):
        """Create repositories with in-memory database."""
        db_manager = DatabaseManager()
        firmware_repo = FirmwareRepository(db_manager)
        session_repo = AnalysisSessionRepository(db_manager)
        
        # Create test firmware
        firmware = FirmwareMetadata(
            file_path="/test/firmware.bin",
            file_hash="test_hash_123",
            file_size=1024,
            architecture="cortex-m4"
        )
        firmware_id = firmware_repo.create(firmware)
        
        return session_repo, firmware_id
    
    def test_create_session(self, repo):
        """Test creating analysis session."""
        session_repo, firmware_id = repo
        
        session = AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.SCAN,
            configuration={'architecture': 'cortex-m4', 'deep_scan': True}
        )
        
        session_id = session_repo.create(session)
        assert session_id > 0
        
        # Verify the session was created
        retrieved = session_repo.get_by_id(session_id)
        assert retrieved is not None
        assert retrieved.firmware_id == firmware_id
        assert retrieved.session_type == SessionType.SCAN
        assert retrieved.status == SessionStatus.RUNNING
    
    def test_update_status(self, repo):
        """Test updating session status."""
        session_repo, firmware_id = repo
        
        session = AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.SCAN,
            configuration={}
        )
        
        session_id = session_repo.create(session)
        
        # Update to completed
        success = session_repo.update_status(session_id, SessionStatus.COMPLETED)
        assert success is True
        
        # Verify status was updated
        retrieved = session_repo.get_by_id(session_id)
        assert retrieved.status == SessionStatus.COMPLETED
        assert retrieved.completed_at is not None
        
        # Update to failed with error message
        success = session_repo.update_status(session_id, SessionStatus.FAILED, "Test error")
        assert success is True
        
        retrieved = session_repo.get_by_id(session_id)
        assert retrieved.status == SessionStatus.FAILED
        assert retrieved.error_message == "Test error"
    
    def test_get_by_firmware(self, repo):
        """Test getting sessions by firmware."""
        session_repo, firmware_id = repo
        
        # Create multiple sessions
        session1 = AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.SCAN,
            configuration={}
        )
        session2 = AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.PATCH,
            configuration={}
        )
        
        session_repo.create(session1)
        session_repo.create(session2)
        
        # Get all sessions for firmware
        all_sessions = session_repo.get_by_firmware(firmware_id)
        assert len(all_sessions) == 2
        
        # Get only scan sessions
        scan_sessions = session_repo.get_by_firmware(firmware_id, SessionType.SCAN)
        assert len(scan_sessions) == 1
        assert scan_sessions[0].session_type == SessionType.SCAN


class TestScanResultRepository:
    """Test scan result repository operations."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup database with test data."""
        db_manager = DatabaseManager()
        
        # Create repositories
        firmware_repo = FirmwareRepository(db_manager)
        session_repo = AnalysisSessionRepository(db_manager)
        scan_repo = ScanResultRepository(db_manager)
        
        # Create test firmware and session
        firmware = FirmwareMetadata(
            file_path="/test/firmware.bin",
            file_hash="test_hash_123",
            file_size=1024,
            architecture="cortex-m4"
        )
        firmware_id = firmware_repo.create(firmware)
        
        session = AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.SCAN,
            configuration={}
        )
        session_id = session_repo.create(session)
        
        return scan_repo, session_id
    
    def test_create_scan_result(self, setup_data):
        """Test creating scan result."""
        scan_repo, session_id = setup_data
        
        scan_result = ScanResult(
            session_id=session_id,
            total_vulnerabilities=5,
            critical_count=1,
            high_count=2,
            medium_count=2,
            low_count=0,
            scan_duration_ms=1500,
            memory_constraints={'flash': 512*1024, 'ram': 128*1024},
            recommendations=['Upgrade to PQC algorithms', 'Review memory usage']
        )
        
        result_id = scan_repo.create(scan_result)
        assert result_id > 0
        
        # Verify the result was created
        retrieved = scan_repo.get_by_session(session_id)
        assert retrieved is not None
        assert retrieved.total_vulnerabilities == 5
        assert retrieved.critical_count == 1
        assert retrieved.scan_duration_ms == 1500
        assert len(retrieved.recommendations) == 2
    
    def test_risk_score_calculation(self, setup_data):
        """Test risk score calculation."""
        scan_repo, session_id = setup_data
        
        scan_result = ScanResult(
            session_id=session_id,
            total_vulnerabilities=10,
            critical_count=2,  # 2 * 10 = 20 points
            high_count=3,      # 3 * 7 = 21 points  
            medium_count=3,    # 3 * 4 = 12 points
            low_count=2        # 2 * 1 = 2 points
        )
        
        # Total: 55 points out of max 100 (10 * 10) = 55%
        risk_score = scan_result.risk_score()
        assert risk_score == 55.0
        
        # Test empty scan
        empty_scan = ScanResult(
            session_id=session_id,
            total_vulnerabilities=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0
        )
        
        assert empty_scan.risk_score() == 0.0


class TestVulnerabilityRepository:
    """Test vulnerability repository operations."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup database with test data."""
        db_manager = DatabaseManager()
        
        # Create full chain: firmware -> session -> scan_result
        firmware_repo = FirmwareRepository(db_manager)
        session_repo = AnalysisSessionRepository(db_manager)
        scan_repo = ScanResultRepository(db_manager)
        vuln_repo = VulnerabilityRepository(db_manager)
        
        firmware_id = firmware_repo.create(FirmwareMetadata(
            file_path="/test/firmware.bin",
            file_hash="test_hash",
            file_size=1024,
            architecture="cortex-m4"
        ))
        
        session_id = session_repo.create(AnalysisSession(
            firmware_id=firmware_id,
            session_type=SessionType.SCAN,
            configuration={}
        ))
        
        scan_result_id = scan_repo.create(ScanResult(
            session_id=session_id,
            total_vulnerabilities=1
        ))
        
        return vuln_repo, scan_result_id
    
    def test_create_vulnerability(self, setup_data):
        """Test creating vulnerability record."""
        vuln_repo, scan_result_id = setup_data
        
        vulnerability = VulnerabilityRecord(
            scan_result_id=scan_result_id,
            algorithm="RSA-2048",
            address=0x08001000,
            function_name="rsa_sign",
            risk_level="critical",
            description="RSA signature vulnerable to quantum attacks",
            mitigation="Replace with Dilithium2",
            stack_usage=2048,
            available_stack=8192,
            key_size=2048
        )
        
        vuln_id = vuln_repo.create(vulnerability)
        assert vuln_id > 0
        
        # Verify vulnerability was created
        vulns = vuln_repo.get_by_scan_result(scan_result_id)
        assert len(vulns) == 1
        assert vulns[0].algorithm == "RSA-2048"
        assert vulns[0].risk_level == "critical"
    
    def test_batch_create(self, setup_data):
        """Test batch vulnerability creation."""
        vuln_repo, scan_result_id = setup_data
        
        vulnerabilities = [
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001000,
                function_name="rsa_sign",
                risk_level="critical",
                description="RSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=2048,
                available_stack=8192
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="ECDSA-P256",
                address=0x08002000,
                function_name="ecdsa_verify",
                risk_level="high",
                description="ECDSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=1024,
                available_stack=8192
            )
        ]
        
        vuln_ids = vuln_repo.create_batch(vulnerabilities)
        assert len(vuln_ids) == 2
        assert all(vid > 0 for vid in vuln_ids)
        
        # Verify all vulnerabilities were created
        all_vulns = vuln_repo.get_by_scan_result(scan_result_id)
        assert len(all_vulns) == 2
    
    def test_get_by_algorithm(self, setup_data):
        """Test getting vulnerabilities by algorithm."""
        vuln_repo, scan_result_id = setup_data
        
        # Create vulnerabilities with different algorithms
        vulns = [
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001000,
                function_name="rsa_sign1",
                risk_level="critical",
                description="RSA vulnerability 1",
                mitigation="Use Dilithium2",
                stack_usage=2048,
                available_stack=8192
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001100,
                function_name="rsa_sign2",
                risk_level="critical",
                description="RSA vulnerability 2",
                mitigation="Use Dilithium2",
                stack_usage=2048,
                available_stack=8192
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="ECDSA-P256",
                address=0x08002000,
                function_name="ecdsa_verify",
                risk_level="high",
                description="ECDSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=1024,
                available_stack=8192
            )
        ]
        
        vuln_repo.create_batch(vulns)
        
        # Get RSA vulnerabilities
        rsa_vulns = vuln_repo.get_by_algorithm("RSA-2048")
        assert len(rsa_vulns) == 2
        
        # Get ECDSA vulnerabilities
        ecdsa_vulns = vuln_repo.get_by_algorithm("ECDSA-P256")
        assert len(ecdsa_vulns) == 1
    
    def test_get_patchable_vulnerabilities(self, setup_data):
        """Test getting patchable vulnerabilities."""
        vuln_repo, scan_result_id = setup_data
        
        # Create vulnerabilities with different stack availability
        vulns = [
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001000,
                function_name="patchable_rsa",
                risk_level="critical",
                description="Patchable RSA",
                mitigation="Use Dilithium2",
                stack_usage=1024,
                available_stack=4096  # Enough for patching
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="ECDSA-P256",
                address=0x08002000,
                function_name="unpatchable_ecdsa",
                risk_level="high",
                description="Unpatchable ECDSA",
                mitigation="Hardware upgrade needed",
                stack_usage=1024,
                available_stack=1024  # Not enough for patching
            )
        ]
        
        vuln_repo.create_batch(vulns)
        
        # Get patchable vulnerabilities (min 2KB available stack)
        patchable = vuln_repo.get_patchable_vulnerabilities(min_available_stack=2048)
        assert len(patchable) == 1
        assert patchable[0].function_name == "patchable_rsa"
    
    def test_algorithm_statistics(self, setup_data):
        """Test algorithm statistics."""
        vuln_repo, scan_result_id = setup_data
        
        # Create diverse vulnerabilities
        vulns = [
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001000,
                function_name="rsa1",
                risk_level="critical",
                description="RSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=2048,
                available_stack=8192
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="RSA-2048",
                address=0x08001100,
                function_name="rsa2",
                risk_level="high",
                description="RSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=2048,
                available_stack=4096  # Patchable
            ),
            VulnerabilityRecord(
                scan_result_id=scan_result_id,
                algorithm="ECDSA-P256",
                address=0x08002000,
                function_name="ecdsa1",
                risk_level="critical",
                description="ECDSA vulnerability",
                mitigation="Use Dilithium2",
                stack_usage=1024,
                available_stack=1024  # Not patchable
            )
        ]
        
        vuln_repo.create_batch(vulns)
        
        stats = vuln_repo.get_algorithm_statistics()
        
        assert 'RSA-2048' in stats
        assert 'ECDSA-P256' in stats
        
        rsa_stats = stats['RSA-2048']
        assert rsa_stats['total_count'] == 2
        assert rsa_stats['critical_count'] == 1
        assert rsa_stats['high_count'] == 1
        assert rsa_stats['patchable_count'] == 1  # Only one has enough stack
        assert rsa_stats['patchable_percentage'] == 50.0
        
        ecdsa_stats = stats['ECDSA-P256']
        assert ecdsa_stats['total_count'] == 1
        assert ecdsa_stats['critical_count'] == 1
        assert ecdsa_stats['patchable_count'] == 0  # Not enough stack