"""Database connection and management for PQC IoT Retrofit Scanner."""

import sqlite3
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import json


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and schema initialization."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._connection = None
        self._initialized = False
        
        # Initialize database schema
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema."""
        schema_sql = """
        -- Firmware metadata table
        CREATE TABLE IF NOT EXISTS firmware_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            file_size INTEGER NOT NULL,
            architecture TEXT NOT NULL,
            base_address INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Analysis sessions table
        CREATE TABLE IF NOT EXISTS analysis_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            firmware_id INTEGER NOT NULL,
            session_type TEXT NOT NULL, -- 'scan', 'patch', 'analyze'
            status TEXT DEFAULT 'pending',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (firmware_id) REFERENCES firmware_metadata (id)
        );
        
        -- Scan results table
        CREATE TABLE IF NOT EXISTS scan_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            scan_type TEXT NOT NULL,
            result_data TEXT, -- JSON data
            risk_score REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
        );
        
        -- Vulnerabilities table
        CREATE TABLE IF NOT EXISTS vulnerabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            address INTEGER,
            severity TEXT DEFAULT 'medium',
            description TEXT,
            patchable BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
        );
        """
        
        with self.get_connection() as conn:
            conn.executescript(schema_sql)
            logger.info("Database schema initialized successfully")
        
        self._initialized = True
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of query results
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            ID of the inserted row
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            return cursor.lastrowid
    
    def get_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data.
        
        Args:
            cache_key: Unique cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        query = """
        SELECT cache_data, expires_at FROM analysis_cache
        WHERE cache_key = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        """
        
        results = self.execute_query(query, (cache_key,))
        if results:
            try:
                return json.loads(results[0]['cache_data'])
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode cached data for key: {cache_key}")
                return None
        return None
    
    def set_cache(self, cache_key: str, data: Dict[str, Any], ttl_minutes: Optional[int] = None):
        """Store data in cache.
        
        Args:
            cache_key: Unique cache key
            data: Data to cache
            ttl_minutes: Time to live in minutes (None for no expiration)
        """
        expires_at = None
        if ttl_minutes:
            expires_at = datetime.now().timestamp() + (ttl_minutes * 60)
        
        query = """
        INSERT OR REPLACE INTO analysis_cache (cache_key, cache_data, expires_at)
        VALUES (?, ?, ?)
        """
        
        self.execute_update(query, (cache_key, json.dumps(data), expires_at))
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cached data.
        
        Args:
            pattern: SQL LIKE pattern to match keys (None clears all)
        """
        if pattern:
            query = "DELETE FROM analysis_cache WHERE cache_key LIKE ?"
            self.execute_update(query, (pattern,))
        else:
            query = "DELETE FROM analysis_cache"
            self.execute_update(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        # Count records in each table
        tables = [
            'firmware_metadata',
            'analysis_sessions', 
            'scan_results',
            'vulnerability_records',
            'patch_records',
            'analysis_cache'
        ]
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query)
            stats[f"{table}_count"] = result[0]['count'] if result else 0
        
        # Vulnerability statistics
        vuln_stats_query = """
        SELECT 
            risk_level,
            COUNT(*) as count
        FROM vulnerability_records
        GROUP BY risk_level
        """
        vuln_results = self.execute_query(vuln_stats_query)
        stats['vulnerability_by_risk'] = {row['risk_level']: row['count'] for row in vuln_results}
        
        # Algorithm statistics
        algo_stats_query = """
        SELECT 
            algorithm,
            COUNT(*) as count
        FROM vulnerability_records
        GROUP BY algorithm
        ORDER BY count DESC
        """
        algo_results = self.execute_query(algo_stats_query)
        stats['vulnerabilities_by_algorithm'] = {row['algorithm']: row['count'] for row in algo_results}
        
        # Recent activity
        recent_query = """
        SELECT COUNT(*) as count
        FROM analysis_sessions
        WHERE started_at > datetime('now', '-7 days')
        """
        recent_result = self.execute_query(recent_query)
        stats['sessions_last_7_days'] = recent_result[0]['count'] if recent_result else 0
        
        return stats
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to maintain database size.
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_query = "datetime('now', '-{} days')".format(days_to_keep)
        
        # Clean expired cache entries
        cache_query = f"DELETE FROM analysis_cache WHERE expires_at < {cutoff_query}"
        cache_deleted = self.execute_update(cache_query)
        
        # Clean old sessions and related data (cascading)
        session_query = f"DELETE FROM analysis_sessions WHERE started_at < {cutoff_query}"
        sessions_deleted = self.execute_update(session_query)
        
        logger.info(f"Cleaned up database: {cache_deleted} cache entries, {sessions_deleted} old sessions")
        
        # Vacuum database to reclaim space
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
    
    def export_data(self, output_path: str, include_cache: bool = False):
        """Export database data to JSON file.
        
        Args:
            output_path: Path to output JSON file
            include_cache: Whether to include cache data
        """
        export_data = {}
        
        # Export main tables
        tables = [
            'firmware_metadata',
            'analysis_sessions',
            'scan_results', 
            'vulnerability_records',
            'patch_records'
        ]
        
        if include_cache:
            tables.append('analysis_cache')
        
        for table in tables:
            query = f"SELECT * FROM {table}"
            results = self.execute_query(query)
            export_data[table] = [dict(row) for row in results]
        
        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Database exported to {output_path}")
    
    def import_data(self, input_path: str):
        """Import data from JSON file.
        
        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            import_data = json.load(f)
        
        # Import data in dependency order
        table_order = [
            'firmware_metadata',
            'analysis_sessions',
            'scan_results',
            'vulnerability_records', 
            'patch_records',
            'analysis_cache'
        ]
        
        for table in table_order:
            if table in import_data:
                records = import_data[table]
                if records:
                    # Build INSERT query
                    columns = list(records[0].keys())
                    placeholders = ', '.join(['?' for _ in columns])
                    query = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    # Execute batch insert
                    with self.get_connection() as conn:
                        for record in records:
                            values = [record[col] for col in columns]
                            conn.execute(query, values)
                        conn.commit()
        
        logger.info(f"Database imported from {input_path}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check.
        
        Returns:
            Health check results
        """
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Test basic connectivity
            with self.get_connection() as conn:
                conn.execute("SELECT 1")
            
            # Check database size
            if self.db_path != ":memory:":
                db_size = Path(self.db_path).stat().st_size
                health['database_size_mb'] = db_size / (1024 * 1024)
                
                if db_size > 100 * 1024 * 1024:  # 100MB
                    health['recommendations'].append("Database size is large, consider cleanup")
            
            # Check for data integrity
            stats = self.get_statistics()
            health['statistics'] = stats
            
            # Check for orphaned records
            orphan_query = """
            SELECT COUNT(*) as count FROM vulnerability_records v
            LEFT JOIN scan_results s ON v.scan_result_id = s.id
            WHERE s.id IS NULL
            """
            orphan_result = self.execute_query(orphan_query)
            orphan_count = orphan_result[0]['count'] if orphan_result else 0
            
            if orphan_count > 0:
                health['issues'].append(f"Found {orphan_count} orphaned vulnerability records")
                health['status'] = 'warning'
            
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Database error: {str(e)}")
        
        return health