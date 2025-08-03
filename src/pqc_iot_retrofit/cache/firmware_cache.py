"""Firmware-specific caching functionality."""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

from .cache_manager import CacheManager

logger = logging.getLogger(__name__)


class FirmwareCache:
    """Cache for firmware metadata and analysis artifacts."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize firmware cache.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache = cache_manager
        self.key_prefix = "firmware:"
    
    def get_file_metadata(self, firmware_path: str) -> Optional[Dict[str, Any]]:
        """Get cached firmware file metadata.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Cached metadata or None if not found
        """
        # Use file modification time and size as part of cache key
        try:
            stat = os.stat(firmware_path)
            metadata_key = f"{self.key_prefix}metadata:{firmware_path}:{stat.st_mtime}:{stat.st_size}"
            return self.cache.get(metadata_key)
        except OSError:
            return None
    
    def cache_file_metadata(self, firmware_path: str, metadata: Dict[str, Any], 
                           ttl_minutes: int = 1440) -> None:
        """Cache firmware file metadata.
        
        Args:
            firmware_path: Path to firmware file
            metadata: File metadata
            ttl_minutes: Time to live in minutes (default: 24 hours)
        """
        try:
            stat = os.stat(firmware_path)
            metadata_key = f"{self.key_prefix}metadata:{firmware_path}:{stat.st_mtime}:{stat.st_size}"
            
            # Add caching timestamp
            metadata_with_timestamp = {
                **metadata,
                'cached_at': datetime.now().isoformat(),
                'file_path': firmware_path,
                'file_size': stat.st_size,
                'file_mtime': stat.st_mtime
            }
            
            self.cache.set(metadata_key, metadata_with_timestamp, ttl_minutes)
            logger.debug(f"Cached metadata for firmware: {firmware_path}")
        except OSError as e:
            logger.warning(f"Failed to cache metadata for {firmware_path}: {e}")
    
    def get_binary_sections(self, firmware_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached binary section information.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Cached section information or None
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return None
        
        sections_key = f"{self.key_prefix}sections:{file_hash}"
        return self.cache.get(sections_key)
    
    def cache_binary_sections(self, firmware_path: str, sections: List[Dict[str, Any]],
                             ttl_minutes: int = 720) -> None:
        """Cache binary section information.
        
        Args:
            firmware_path: Path to firmware file
            sections: List of section information
            ttl_minutes: Time to live in minutes (default: 12 hours)
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return
        
        sections_key = f"{self.key_prefix}sections:{file_hash}"
        
        # Add metadata to sections
        cached_sections = {
            'sections': sections,
            'total_sections': len(sections),
            'cached_at': datetime.now().isoformat(),
            'source_file': firmware_path
        }
        
        self.cache.set(sections_key, cached_sections, ttl_minutes)
        logger.debug(f"Cached {len(sections)} sections for firmware: {firmware_path}")
    
    def get_string_table(self, firmware_path: str) -> Optional[List[str]]:
        """Get cached string table from firmware.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Cached strings or None
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return None
        
        strings_key = f"{self.key_prefix}strings:{file_hash}"
        cached_data = self.cache.get(strings_key)
        
        if cached_data:
            return cached_data.get('strings', [])
        return None
    
    def cache_string_table(self, firmware_path: str, strings: List[str],
                          ttl_minutes: int = 480) -> None:
        """Cache extracted strings from firmware.
        
        Args:
            firmware_path: Path to firmware file
            strings: List of extracted strings
            ttl_minutes: Time to live in minutes (default: 8 hours)
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return
        
        strings_key = f"{self.key_prefix}strings:{file_hash}"
        
        # Filter and categorize strings
        crypto_strings = [s for s in strings if self._is_crypto_related(s)]
        
        cached_data = {
            'strings': strings,
            'total_strings': len(strings),
            'crypto_strings': crypto_strings,
            'crypto_string_count': len(crypto_strings),
            'cached_at': datetime.now().isoformat(),
            'source_file': firmware_path
        }
        
        self.cache.set(strings_key, cached_data, ttl_minutes)
        logger.debug(f"Cached {len(strings)} strings ({len(crypto_strings)} crypto-related) for firmware: {firmware_path}")
    
    def get_function_list(self, firmware_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached function list from firmware.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Cached function information or None
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return None
        
        functions_key = f"{self.key_prefix}functions:{file_hash}"
        cached_data = self.cache.get(functions_key)
        
        if cached_data:
            return cached_data.get('functions', [])
        return None
    
    def cache_function_list(self, firmware_path: str, functions: List[Dict[str, Any]],
                           ttl_minutes: int = 360) -> None:
        """Cache function information from firmware.
        
        Args:
            firmware_path: Path to firmware file
            functions: List of function information
            ttl_minutes: Time to live in minutes (default: 6 hours)
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return
        
        functions_key = f"{self.key_prefix}functions:{file_hash}"
        
        # Categorize functions
        crypto_functions = [f for f in functions if self._is_crypto_function(f)]
        
        cached_data = {
            'functions': functions,
            'total_functions': len(functions),
            'crypto_functions': crypto_functions,
            'crypto_function_count': len(crypto_functions),
            'cached_at': datetime.now().isoformat(),
            'source_file': firmware_path
        }
        
        self.cache.set(functions_key, cached_data, ttl_minutes)
        logger.debug(f"Cached {len(functions)} functions ({len(crypto_functions)} crypto-related) for firmware: {firmware_path}")
    
    def get_architecture_info(self, firmware_path: str) -> Optional[Dict[str, Any]]:
        """Get cached architecture detection results.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Cached architecture info or None
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return None
        
        arch_key = f"{self.key_prefix}architecture:{file_hash}"
        return self.cache.get(arch_key)
    
    def cache_architecture_info(self, firmware_path: str, arch_info: Dict[str, Any],
                               ttl_minutes: int = 2880) -> None:
        """Cache architecture detection results.
        
        Args:
            firmware_path: Path to firmware file
            arch_info: Architecture information
            ttl_minutes: Time to live in minutes (default: 48 hours)
        """
        file_hash = self._get_file_hash(firmware_path)
        if not file_hash:
            return
        
        arch_key = f"{self.key_prefix}architecture:{file_hash}"
        
        cached_data = {
            **arch_info,
            'cached_at': datetime.now().isoformat(),
            'source_file': firmware_path
        }
        
        self.cache.set(arch_key, cached_data, ttl_minutes)
        logger.debug(f"Cached architecture info for firmware: {firmware_path}")
    
    def get_similar_firmware(self, firmware_path: str, similarity_threshold: float = 0.8) -> List[str]:
        """Get list of similar firmware files from cache.
        
        Args:
            firmware_path: Path to firmware file
            similarity_threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar firmware file paths
        """
        # This is a simplified implementation
        # In practice, you'd use more sophisticated similarity metrics
        
        file_size = self._get_file_size(firmware_path)
        if not file_size:
            return []
        
        similar_files = []
        
        # Look for files with similar sizes (within 20%)
        size_range = (file_size * 0.8, file_size * 1.2)
        
        # This would query cached metadata for files in size range
        # For now, return empty list
        return similar_files
    
    def invalidate_firmware_cache(self, firmware_path: str) -> None:
        """Invalidate all cache entries for a firmware file.
        
        Args:
            firmware_path: Path to firmware file
        """
        file_hash = self._get_file_hash(firmware_path)
        if file_hash:
            # Clear all entries with this file hash
            pattern = f"{self.key_prefix}*:{file_hash}*"
            self.cache.clear(pattern)
        
        # Also clear path-based entries
        path_pattern = f"{self.key_prefix}*{firmware_path}*"
        self.cache.clear(path_pattern)
        
        logger.debug(f"Invalidated all cache entries for firmware: {firmware_path}")
    
    def get_cache_info(self, firmware_path: str) -> Dict[str, Any]:
        """Get cache information for a firmware file.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            Dictionary with cache information
        """
        file_hash = self._get_file_hash(firmware_path)
        
        cache_info = {
            'firmware_path': firmware_path,
            'file_hash': file_hash,
            'cached_items': {
                'metadata': self.get_file_metadata(firmware_path) is not None,
                'sections': self.get_binary_sections(firmware_path) is not None,
                'strings': self.get_string_table(firmware_path) is not None,
                'functions': self.get_function_list(firmware_path) is not None,
                'architecture': self.get_architecture_info(firmware_path) is not None
            }
        }
        
        # Count total cached items
        cache_info['total_cached_items'] = sum(cache_info['cached_items'].values())
        
        return cache_info
    
    def _get_file_hash(self, firmware_path: str) -> Optional[str]:
        """Get file hash for caching purposes.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            File hash or None if file doesn't exist
        """
        try:
            from ..database.repositories import FirmwareRepository
            repo = FirmwareRepository(self.cache.db)
            return repo.calculate_file_hash(firmware_path)
        except Exception as e:
            logger.warning(f"Failed to calculate file hash for {firmware_path}: {e}")
            return None
    
    def _get_file_size(self, firmware_path: str) -> Optional[int]:
        """Get file size.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            File size in bytes or None
        """
        try:
            return os.path.getsize(firmware_path)
        except OSError:
            return None
    
    def _is_crypto_related(self, string: str) -> bool:
        """Check if string is cryptography-related.
        
        Args:
            string: String to check
            
        Returns:
            True if crypto-related
        """
        crypto_keywords = [
            'rsa', 'ecdsa', 'ecdh', 'aes', 'des', 'sha', 'md5',
            'crypto', 'cipher', 'key', 'sign', 'verify', 'encrypt',
            'decrypt', 'hash', 'digest', 'certificate', 'x509',
            'ssl', 'tls', 'pkcs', 'pem', 'der'
        ]
        
        string_lower = string.lower()
        return any(keyword in string_lower for keyword in crypto_keywords)
    
    def _is_crypto_function(self, function_info: Dict[str, Any]) -> bool:
        """Check if function is cryptography-related.
        
        Args:
            function_info: Function information dictionary
            
        Returns:
            True if crypto-related
        """
        function_name = function_info.get('name', '').lower()
        return self._is_crypto_related(function_name)
    
    def preload_firmware_cache(self, firmware_paths: List[str]) -> Dict[str, Any]:
        """Preload cache for multiple firmware files.
        
        Args:
            firmware_paths: List of firmware file paths
            
        Returns:
            Summary of preloading results
        """
        results = {
            'total_files': len(firmware_paths),
            'processed_files': 0,
            'cached_files': 0,
            'errors': []
        }
        
        for firmware_path in firmware_paths:
            try:
                # Check if file exists and get basic metadata
                if not os.path.exists(firmware_path):
                    results['errors'].append(f"File not found: {firmware_path}")
                    continue
                
                results['processed_files'] += 1
                
                # Check if already cached
                cache_info = self.get_cache_info(firmware_path)
                if cache_info['total_cached_items'] > 0:
                    results['cached_files'] += 1
                    continue
                
                # Basic metadata caching
                stat = os.stat(firmware_path)
                basic_metadata = {
                    'file_size': stat.st_size,
                    'file_mtime': stat.st_mtime,
                    'file_mode': stat.st_mode,
                    'preloaded': True
                }
                
                self.cache_file_metadata(firmware_path, basic_metadata)
                results['cached_files'] += 1
                
            except Exception as e:
                results['errors'].append(f"Error processing {firmware_path}: {str(e)}")
        
        logger.info(f"Preloaded firmware cache: {results['cached_files']}/{results['total_files']} files")
        return results