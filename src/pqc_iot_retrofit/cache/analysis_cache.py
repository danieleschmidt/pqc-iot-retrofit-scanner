"""Analysis-specific caching functionality."""

import hashlib
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from .cache_manager import CacheManager
from ..scanner import CryptoVulnerability

logger = logging.getLogger(__name__)


class AnalysisCache:
    """Cache for firmware analysis results."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize analysis cache.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache = cache_manager
        self.key_prefix = "analysis:"
    
    def get_firmware_hash(self, firmware_path: str) -> str:
        """Calculate firmware file hash for caching.
        
        Args:
            firmware_path: Path to firmware file
            
        Returns:
            SHA256 hash of firmware file
        """
        sha256_hash = hashlib.sha256()
        
        try:
            with open(firmware_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {firmware_path}: {e}")
            # Fallback to path-based hash
            return hashlib.sha256(firmware_path.encode()).hexdigest()
    
    def cache_key_for_scan(self, firmware_path: str, architecture: str, 
                          base_address: int = 0, memory_constraints: Optional[Dict] = None) -> str:
        """Generate cache key for scan results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture
            base_address: Base memory address
            memory_constraints: Memory constraints
            
        Returns:
            Cache key for scan results
        """
        file_hash = self.get_firmware_hash(firmware_path)
        constraints_hash = hashlib.md5(str(memory_constraints or {}).encode()).hexdigest()[:8]
        
        return f"{self.key_prefix}scan:{file_hash}:{architecture}:{base_address:x}:{constraints_hash}"
    
    def get_scan_results(self, firmware_path: str, architecture: str,
                        base_address: int = 0, memory_constraints: Optional[Dict] = None) -> Optional[List[CryptoVulnerability]]:
        """Get cached scan results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture  
            base_address: Base memory address
            memory_constraints: Memory constraints
            
        Returns:
            Cached vulnerabilities or None if not found
        """
        cache_key = self.cache_key_for_scan(firmware_path, architecture, base_address, memory_constraints)
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            try:
                # Reconstruct vulnerability objects from cached data
                vulnerabilities = []
                for vuln_data in cached_data:
                    # Import here to avoid circular dependency
                    from ..scanner import CryptoVulnerability, CryptoAlgorithm, RiskLevel
                    
                    vuln = CryptoVulnerability(
                        algorithm=CryptoAlgorithm(vuln_data['algorithm']),
                        address=vuln_data['address'],
                        function_name=vuln_data['function_name'],
                        risk_level=RiskLevel(vuln_data['risk_level']),
                        key_size=vuln_data.get('key_size'),
                        description=vuln_data['description'],
                        mitigation=vuln_data['mitigation'],
                        stack_usage=vuln_data['stack_usage'],
                        available_stack=vuln_data['available_stack']
                    )
                    vulnerabilities.append(vuln)
                
                logger.debug(f"Cache hit for scan: {firmware_path} ({len(vulnerabilities)} vulnerabilities)")
                return vulnerabilities
                
            except Exception as e:
                logger.warning(f"Failed to deserialize cached scan results: {e}")
                # Remove corrupted cache entry
                self.cache.delete(cache_key)
        
        return None
    
    def cache_scan_results(self, firmware_path: str, architecture: str,
                          vulnerabilities: List[CryptoVulnerability],
                          base_address: int = 0, memory_constraints: Optional[Dict] = None,
                          ttl_minutes: int = 120) -> None:
        """Cache scan results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture
            vulnerabilities: List of detected vulnerabilities
            base_address: Base memory address
            memory_constraints: Memory constraints
            ttl_minutes: Time to live in minutes
        """
        cache_key = self.cache_key_for_scan(firmware_path, architecture, base_address, memory_constraints)
        
        # Serialize vulnerabilities for caching
        cached_data = []
        for vuln in vulnerabilities:
            vuln_data = {
                'algorithm': vuln.algorithm.value,
                'address': vuln.address,
                'function_name': vuln.function_name,
                'risk_level': vuln.risk_level.value,
                'key_size': vuln.key_size,
                'description': vuln.description,
                'mitigation': vuln.mitigation,
                'stack_usage': vuln.stack_usage,
                'available_stack': vuln.available_stack
            }
            cached_data.append(vuln_data)
        
        self.cache.set(cache_key, cached_data, ttl_minutes)
        logger.debug(f"Cached scan results: {firmware_path} ({len(vulnerabilities)} vulnerabilities)")
    
    def cache_key_for_disassembly(self, firmware_path: str, architecture: str, 
                                 section_offset: int = 0, section_size: int = 0) -> str:
        """Generate cache key for disassembly results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture
            section_offset: Section offset
            section_size: Section size
            
        Returns:
            Cache key for disassembly results
        """
        file_hash = self.get_firmware_hash(firmware_path)
        section_hash = hashlib.md5(f"{section_offset}:{section_size}".encode()).hexdigest()[:8]
        
        return f"{self.key_prefix}disasm:{file_hash}:{architecture}:{section_hash}"
    
    def get_disassembly_cache(self, firmware_path: str, architecture: str,
                             section_offset: int = 0, section_size: int = 0) -> Optional[List[Dict]]:
        """Get cached disassembly results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture
            section_offset: Section offset
            section_size: Section size
            
        Returns:
            Cached disassembly instructions or None
        """
        cache_key = self.cache_key_for_disassembly(firmware_path, architecture, section_offset, section_size)
        return self.cache.get(cache_key)
    
    def cache_disassembly(self, firmware_path: str, architecture: str, instructions: List[Dict],
                         section_offset: int = 0, section_size: int = 0, ttl_minutes: int = 240) -> None:
        """Cache disassembly results.
        
        Args:
            firmware_path: Path to firmware file
            architecture: Target architecture
            instructions: Disassembly instructions
            section_offset: Section offset
            section_size: Section size
            ttl_minutes: Time to live in minutes
        """
        cache_key = self.cache_key_for_disassembly(firmware_path, architecture, section_offset, section_size)
        
        # Convert instructions to serializable format
        serializable_instructions = []
        for insn in instructions:
            if hasattr(insn, '__dict__'):
                # Convert capstone instruction to dict
                insn_dict = {
                    'address': insn.address,
                    'mnemonic': insn.mnemonic,
                    'op_str': insn.op_str,
                    'bytes': insn.bytes,
                    'size': insn.size
                }
            else:
                insn_dict = insn
            
            serializable_instructions.append(insn_dict)
        
        self.cache.set(cache_key, serializable_instructions, ttl_minutes)
        logger.debug(f"Cached disassembly: {firmware_path} ({len(instructions)} instructions)")
    
    def cache_key_for_patch(self, vulnerability_algorithm: str, target_device: str,
                           security_level: int, optimization_level: str) -> str:
        """Generate cache key for patch templates.
        
        Args:
            vulnerability_algorithm: Original vulnerable algorithm
            target_device: Target device
            security_level: Security level
            optimization_level: Optimization level
            
        Returns:
            Cache key for patch template
        """
        key_data = f"{vulnerability_algorithm}:{target_device}:{security_level}:{optimization_level}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"{self.key_prefix}patch_template:{key_hash}"
    
    def get_patch_template(self, vulnerability_algorithm: str, target_device: str,
                          security_level: int, optimization_level: str) -> Optional[Dict[str, Any]]:
        """Get cached patch template.
        
        Args:
            vulnerability_algorithm: Original vulnerable algorithm
            target_device: Target device
            security_level: Security level
            optimization_level: Optimization level
            
        Returns:
            Cached patch template or None
        """
        cache_key = self.cache_key_for_patch(vulnerability_algorithm, target_device, 
                                           security_level, optimization_level)
        return self.cache.get(cache_key)
    
    def cache_patch_template(self, vulnerability_algorithm: str, target_device: str,
                           security_level: int, optimization_level: str,
                           template_data: Dict[str, Any], ttl_minutes: int = 480) -> None:
        """Cache patch template.
        
        Args:
            vulnerability_algorithm: Original vulnerable algorithm
            target_device: Target device
            security_level: Security level
            optimization_level: Optimization level
            template_data: Patch template data
            ttl_minutes: Time to live in minutes
        """
        cache_key = self.cache_key_for_patch(vulnerability_algorithm, target_device,
                                           security_level, optimization_level)
        self.cache.set(cache_key, template_data, ttl_minutes)
    
    def invalidate_firmware_cache(self, firmware_path: str) -> None:
        """Invalidate all cache entries for a firmware file.
        
        Args:
            firmware_path: Path to firmware file
        """
        file_hash = self.get_firmware_hash(firmware_path)
        pattern = f"{self.key_prefix}*{file_hash}*"
        self.cache.clear(pattern)
        logger.debug(f"Invalidated cache for firmware: {firmware_path}")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        base_stats = self.cache.get_stats()
        
        # Count analysis-specific cache entries
        analysis_stats = {
            'scan_results_cached': 0,
            'disassembly_cached': 0,
            'patch_templates_cached': 0
        }
        
        # This is a simplified count - in practice you'd query the cache more efficiently
        cache_info = self.cache.db.get_statistics()
        
        analysis_stats['total_cache_entries'] = cache_info.get('analysis_cache_count', 0)
        
        return {
            **base_stats,
            'analysis_specific': analysis_stats
        }
    
    def preload_common_patterns(self) -> None:
        """Preload common cryptographic patterns into cache."""
        common_patterns = {
            'rsa_constants': [
                b'\x01\x00\x01',  # RSA-65537
                b'\x03',           # RSA-3
                b'\x00\x01\xff\xff',  # PKCS1 padding
            ],
            'ecc_curves': [
                b'\xff\xff\xff\xff\x00\x00\x00\x01',  # P-256
                b'\xff\xff\xff\xff\xff\xff\xff\xff',  # P-384
            ],
            'common_algorithms': ['RSA', 'ECDSA', 'ECDH', 'DH']
        }
        
        cache_key = f"{self.key_prefix}common_patterns"
        self.cache.set(cache_key, common_patterns, ttl_minutes=1440)  # 24 hours
        logger.debug("Preloaded common cryptographic patterns into cache")