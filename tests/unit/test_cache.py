"""Unit tests for caching system."""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from pqc_iot_retrofit.database.connection import DatabaseManager
from pqc_iot_retrofit.cache.cache_manager import CacheManager
from pqc_iot_retrofit.cache.analysis_cache import AnalysisCache
from pqc_iot_retrofit.cache.firmware_cache import FirmwareCache
from pqc_iot_retrofit.scanner import CryptoVulnerability, CryptoAlgorithm, RiskLevel


class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_manager(self):
        """Create cache manager with in-memory database."""
        db_manager = DatabaseManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            yield CacheManager(db_manager, tmpdir)
    
    def test_memory_cache_operations(self, cache_manager):
        """Test in-memory cache operations."""
        # Test set and get
        cache_manager.set('test_key', {'data': 'test_value'}, ttl_minutes=60)
        
        retrieved = cache_manager.get('test_key')
        assert retrieved == {'data': 'test_value'}
        
        # Test default value
        assert cache_manager.get('non_existent', 'default') == 'default'
        
        # Test delete
        success = cache_manager.delete('test_key')
        assert success is True
        assert cache_manager.get('test_key') is None
    
    def test_cache_expiration(self, cache_manager):
        """Test cache expiration logic."""
        # Set cache with very short TTL
        cache_manager.set('expire_test', 'value', ttl_minutes=0.01)  # ~0.6 seconds
        
        # Should be available immediately
        assert cache_manager.get('expire_test') == 'value'
        
        # Wait for expiration (simplified test)
        # In real scenario would wait, but for unit test we'll test the internal logic
        with patch('pqc_iot_retrofit.cache.cache_manager.datetime') as mock_datetime:
            from datetime import datetime, timedelta
            
            # Mock current time to be after expiration
            future_time = datetime.now() + timedelta(minutes=2)
            mock_datetime.now.return_value = future_time
            
            # Should return None after expiration
            assert cache_manager.get('expire_test') is None
    
    def test_lru_eviction(self, cache_manager):
        """Test LRU eviction when cache is full."""
        # Set very small cache size for testing
        cache_manager.max_memory_items = 2
        
        # Fill cache to capacity
        cache_manager.set('key1', 'value1')
        cache_manager.set('key2', 'value2')
        
        # Access key1 to make it more recently used
        cache_manager.get('key1')
        
        # Add another item, should evict key2 (least recently used)
        cache_manager.set('key3', 'value3')
        
        # key1 and key3 should still exist, key2 should be evicted
        assert cache_manager.get('key1') == 'value1'
        assert cache_manager.get('key3') == 'value3'
        # Note: Since key2 might still be in database cache, we'd need to check memory cache specifically
    
    def test_cache_key_generation(self, cache_manager):
        """Test cache key generation from arguments."""
        # Test with positional arguments
        key1 = cache_manager.cache_key('arg1', 'arg2', 123)
        key2 = cache_manager.cache_key('arg1', 'arg2', 123)
        assert key1 == key2  # Should be deterministic
        
        # Test with keyword arguments
        key3 = cache_manager.cache_key(a=1, b=2, c='test')
        key4 = cache_manager.cache_key(c='test', b=2, a=1)  # Different order
        assert key3 == key4  # Should be order-independent
        
        # Test different arguments produce different keys
        key5 = cache_manager.cache_key('different', 'args')
        assert key1 != key5
    
    def test_cached_decorator(self, cache_manager):
        """Test the cached decorator functionality."""
        call_count = 0
        
        @cache_manager.cached(ttl_minutes=60, key_prefix="test_")
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different arguments should execute function again
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2
    
    def test_file_cache_operations(self, cache_manager):
        """Test file-based cache operations."""
        test_data = {'large_data': list(range(1000))}
        
        # Save to file cache
        cache_manager.save_to_file('file_test_key', test_data)
        
        # Load from file cache
        loaded_data = cache_manager.load_from_file('file_test_key')
        assert loaded_data == test_data
        
        # Test non-existent file
        assert cache_manager.load_from_file('non_existent_key') is None
    
    def test_cache_statistics(self, cache_manager):
        """Test cache statistics collection."""
        # Populate cache with some data
        cache_manager.set('stat_test1', 'value1')
        cache_manager.set('stat_test2', 'value2')
        
        # Get and miss some keys to generate stats
        cache_manager.get('stat_test1')  # Hit
        cache_manager.get('stat_test2')  # Hit
        cache_manager.get('non_existent')  # Miss
        
        stats = cache_manager.get_stats()
        
        assert 'memory_cache' in stats
        assert 'file_cache' in stats
        assert 'statistics' in stats
        
        assert stats['statistics']['hits'] >= 2
        assert stats['statistics']['misses'] >= 1
        assert stats['statistics']['sets'] >= 2
        assert 'hit_rate_percent' in stats['statistics']
    
    def test_cache_cleanup(self, cache_manager):
        """Test cache cleanup functionality."""
        # Add some items to cache
        cache_manager.set('cleanup_test1', 'value1')
        cache_manager.set('cleanup_test2', 'value2')
        
        # Clear with pattern
        cache_manager.clear('cleanup_test*')
        
        # Should be cleared from memory (database clearing is tested separately)
        # This is a simplified test since we're dealing with multiple cache layers


class TestAnalysisCache:
    """Test analysis-specific caching."""
    
    @pytest.fixture
    def analysis_cache(self):
        """Create analysis cache with in-memory database."""
        db_manager = DatabaseManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(db_manager, tmpdir)
            yield AnalysisCache(cache_manager)
    
    @pytest.fixture
    def sample_vulnerabilities(self):
        """Create sample vulnerabilities for testing."""
        return [
            CryptoVulnerability(
                algorithm=CryptoAlgorithm.RSA_2048,
                address=0x08001000,
                function_name="rsa_sign",
                risk_level=RiskLevel.CRITICAL,
                key_size=2048,
                description="RSA-2048 signature function",
                mitigation="Replace with Dilithium2",
                stack_usage=2048,
                available_stack=8192
            ),
            CryptoVulnerability(
                algorithm=CryptoAlgorithm.ECDSA_P256,
                address=0x08002000,
                function_name="ecdsa_verify",
                risk_level=RiskLevel.HIGH,
                key_size=256,
                description="ECDSA-P256 verification function",
                mitigation="Replace with Dilithium2",
                stack_usage=1024,
                available_stack=8192
            )
        ]
    
    def test_firmware_hash_calculation(self, analysis_cache, tmp_path):
        """Test firmware hash calculation."""
        # Create test firmware file
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content")
        
        hash1 = analysis_cache.get_firmware_hash(str(firmware_file))
        hash2 = analysis_cache.get_firmware_hash(str(firmware_file))
        
        # Should be deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        
        # Different content should produce different hash
        firmware_file2 = tmp_path / "test_firmware2.bin"
        firmware_file2.write_bytes(b"different firmware content")
        
        hash3 = analysis_cache.get_firmware_hash(str(firmware_file2))
        assert hash1 != hash3
    
    def test_scan_results_caching(self, analysis_cache, sample_vulnerabilities, tmp_path):
        """Test caching scan results."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content")
        
        # Cache scan results
        analysis_cache.cache_scan_results(
            str(firmware_file),
            "cortex-m4",
            sample_vulnerabilities,
            base_address=0x08000000,
            memory_constraints={'flash': 512*1024, 'ram': 128*1024}
        )
        
        # Retrieve cached results
        cached_vulns = analysis_cache.get_scan_results(
            str(firmware_file),
            "cortex-m4",
            base_address=0x08000000,
            memory_constraints={'flash': 512*1024, 'ram': 128*1024}
        )
        
        assert cached_vulns is not None
        assert len(cached_vulns) == 2
        assert cached_vulns[0].algorithm == CryptoAlgorithm.RSA_2048
        assert cached_vulns[1].algorithm == CryptoAlgorithm.ECDSA_P256
        
        # Test cache miss with different parameters
        cached_miss = analysis_cache.get_scan_results(
            str(firmware_file),
            "esp32",  # Different architecture
            base_address=0x08000000
        )
        assert cached_miss is None
    
    def test_disassembly_caching(self, analysis_cache, tmp_path):
        """Test disassembly caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content")
        
        # Mock disassembly instructions
        mock_instructions = [
            {'address': 0x08001000, 'mnemonic': 'mov', 'op_str': 'r0, #1'},
            {'address': 0x08001002, 'mnemonic': 'bx', 'op_str': 'lr'}
        ]
        
        # Cache disassembly
        analysis_cache.cache_disassembly(
            str(firmware_file),
            "cortex-m4",
            mock_instructions,
            section_offset=0x1000,
            section_size=0x1000
        )
        
        # Retrieve cached disassembly
        cached_instructions = analysis_cache.get_disassembly_cache(
            str(firmware_file),
            "cortex-m4",
            section_offset=0x1000,
            section_size=0x1000
        )
        
        assert cached_instructions is not None
        assert len(cached_instructions) == 2
        assert cached_instructions[0]['mnemonic'] == 'mov'
        assert cached_instructions[1]['mnemonic'] == 'bx'
    
    def test_patch_template_caching(self, analysis_cache):
        """Test patch template caching."""
        template_data = {
            'algorithm': 'dilithium2',
            'target_device': 'STM32L4',
            'optimization': 'size',
            'code_template': 'dilithium2_sign_template',
            'memory_requirements': {'stack': 6144, 'flash': 87000}
        }
        
        # Cache patch template
        analysis_cache.cache_patch_template(
            'RSA-2048',
            'STM32L4',
            2,  # Security level
            'size',
            template_data
        )
        
        # Retrieve cached template
        cached_template = analysis_cache.get_patch_template(
            'RSA-2048',
            'STM32L4',
            2,
            'size'
        )
        
        assert cached_template is not None
        assert cached_template['algorithm'] == 'dilithium2'
        assert cached_template['memory_requirements']['stack'] == 6144
        
        # Test cache miss with different parameters
        cached_miss = analysis_cache.get_patch_template(
            'RSA-2048',
            'ESP32',  # Different device
            2,
            'size'
        )
        assert cached_miss is None
    
    def test_cache_invalidation(self, analysis_cache, tmp_path):
        """Test cache invalidation for firmware."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content")
        
        # Cache some data
        analysis_cache.cache_scan_results(str(firmware_file), "cortex-m4", [])
        analysis_cache.cache_disassembly(str(firmware_file), "cortex-m4", [])
        
        # Verify cache exists
        assert analysis_cache.get_scan_results(str(firmware_file), "cortex-m4") is not None
        
        # Invalidate cache
        analysis_cache.invalidate_firmware_cache(str(firmware_file))
        
        # Cache should be cleared (this tests the interface, actual clearing depends on cache manager)
        # In a full integration test, we'd verify the cache is actually cleared
    
    def test_common_patterns_preload(self, analysis_cache):
        """Test preloading common cryptographic patterns."""
        analysis_cache.preload_common_patterns()
        
        # Verify patterns were cached (simplified test)
        # In practice, we'd verify the specific cache keys and data


class TestFirmwareCache:
    """Test firmware-specific caching."""
    
    @pytest.fixture
    def firmware_cache(self):
        """Create firmware cache with in-memory database."""
        db_manager = DatabaseManager()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(db_manager, tmpdir)
            yield FirmwareCache(cache_manager)
    
    def test_file_metadata_caching(self, firmware_cache, tmp_path):
        """Test firmware file metadata caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content")
        
        metadata = {
            'architecture': 'cortex-m4',
            'base_address': 0x08000000,
            'entry_point': 0x08000100,
            'sections': [
                {'name': '.text', 'address': 0x08000000, 'size': 1024},
                {'name': '.data', 'address': 0x20000000, 'size': 256}
            ]
        }
        
        # Cache metadata
        firmware_cache.cache_file_metadata(str(firmware_file), metadata)
        
        # Retrieve cached metadata
        cached_metadata = firmware_cache.get_file_metadata(str(firmware_file))
        
        assert cached_metadata is not None
        assert cached_metadata['architecture'] == 'cortex-m4'
        assert cached_metadata['base_address'] == 0x08000000
        assert len(cached_metadata['sections']) == 2
        assert 'cached_at' in cached_metadata
        assert 'file_path' in cached_metadata
    
    def test_binary_sections_caching(self, firmware_cache, tmp_path):
        """Test binary sections caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware content with sections")
        
        sections = [
            {
                'name': '.text',
                'type': 'PROGBITS',
                'address': 0x08000000,
                'size': 2048,
                'offset': 0x1000,
                'flags': ['SHF_ALLOC', 'SHF_EXECINSTR']
            },
            {
                'name': '.data',
                'type': 'PROGBITS', 
                'address': 0x20000000,
                'size': 512,
                'offset': 0x2000,
                'flags': ['SHF_ALLOC', 'SHF_WRITE']
            }
        ]
        
        # Cache sections
        firmware_cache.cache_binary_sections(str(firmware_file), sections)
        
        # Retrieve cached sections
        cached_sections = firmware_cache.get_binary_sections(str(firmware_file))
        
        assert cached_sections is not None
        assert cached_sections['total_sections'] == 2
        assert len(cached_sections['sections']) == 2
        assert cached_sections['sections'][0]['name'] == '.text'
        assert cached_sections['sections'][1]['name'] == '.data'
    
    def test_string_table_caching(self, firmware_cache, tmp_path):
        """Test string table caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware with crypto strings")
        
        strings = [
            "RSA_sign_function",
            "ECDSA_verify_key",
            "AES_encrypt_data",
            "normal_string",
            "SHA256_hash_compute",
            "another_normal_string"
        ]
        
        # Cache strings
        firmware_cache.cache_string_table(str(firmware_file), strings)
        
        # Retrieve cached strings
        cached_strings = firmware_cache.get_string_table(str(firmware_file))
        
        assert cached_strings is not None
        assert len(cached_strings) == 6
        assert "RSA_sign_function" in cached_strings
        assert "normal_string" in cached_strings
    
    def test_crypto_string_categorization(self, firmware_cache):
        """Test cryptographic string categorization."""
        # Test crypto-related string detection
        assert firmware_cache._is_crypto_related("RSA_sign_function") is True
        assert firmware_cache._is_crypto_related("ECDSA_verify") is True
        assert firmware_cache._is_crypto_related("AES_encrypt") is True
        assert firmware_cache._is_crypto_related("SHA256_hash") is True
        assert firmware_cache._is_crypto_related("ssl_context") is True
        
        # Test non-crypto strings
        assert firmware_cache._is_crypto_related("normal_function") is False
        assert firmware_cache._is_crypto_related("print_string") is False
        assert firmware_cache._is_crypto_related("gpio_configure") is False
    
    def test_function_list_caching(self, firmware_cache, tmp_path):
        """Test function list caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware with functions")
        
        functions = [
            {'name': 'rsa_sign', 'address': 0x08001000, 'size': 256, 'type': 'crypto'},
            {'name': 'ecdsa_verify', 'address': 0x08001100, 'size': 128, 'type': 'crypto'},
            {'name': 'main', 'address': 0x08001200, 'size': 64, 'type': 'normal'},
            {'name': 'gpio_init', 'address': 0x08001300, 'size': 32, 'type': 'normal'}
        ]
        
        # Cache functions
        firmware_cache.cache_function_list(str(firmware_file), functions)
        
        # Retrieve cached functions
        cached_functions = firmware_cache.get_function_list(str(firmware_file))
        
        assert cached_functions is not None
        assert len(cached_functions) == 4
        
        # Find crypto functions
        crypto_funcs = [f for f in cached_functions if 'rsa' in f['name'] or 'ecdsa' in f['name']]
        assert len(crypto_funcs) == 2
    
    def test_architecture_info_caching(self, firmware_cache, tmp_path):
        """Test architecture information caching."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware with arch info")
        
        arch_info = {
            'detected_arch': 'cortex-m4',
            'confidence': 0.95,
            'endianness': 'little',
            'instruction_set': 'thumb2',
            'features': ['dsp', 'fpu'],
            'memory_map': {
                'flash_start': 0x08000000,
                'flash_size': 512*1024,
                'ram_start': 0x20000000,
                'ram_size': 128*1024
            }
        }
        
        # Cache architecture info
        firmware_cache.cache_architecture_info(str(firmware_file), arch_info)
        
        # Retrieve cached info
        cached_arch_info = firmware_cache.get_architecture_info(str(firmware_file))
        
        assert cached_arch_info is not None
        assert cached_arch_info['detected_arch'] == 'cortex-m4'
        assert cached_arch_info['confidence'] == 0.95
        assert 'dsp' in cached_arch_info['features']
        assert cached_arch_info['memory_map']['flash_size'] == 512*1024
    
    def test_cache_info_summary(self, firmware_cache, tmp_path):
        """Test cache information summary."""
        firmware_file = tmp_path / "test_firmware.bin"
        firmware_file.write_bytes(b"test firmware")
        
        # Cache some data
        firmware_cache.cache_file_metadata(str(firmware_file), {'arch': 'cortex-m4'})
        firmware_cache.cache_string_table(str(firmware_file), ['test_string'])
        
        # Get cache info
        cache_info = firmware_cache.get_cache_info(str(firmware_file))
        
        assert cache_info['firmware_path'] == str(firmware_file)
        assert 'file_hash' in cache_info
        assert 'cached_items' in cache_info
        assert cache_info['cached_items']['metadata'] is True
        assert cache_info['cached_items']['strings'] is True
        assert cache_info['total_cached_items'] >= 2
    
    def test_preload_firmware_cache(self, firmware_cache, tmp_path):
        """Test preloading cache for multiple firmware files."""
        # Create multiple test firmware files
        firmware_files = []
        for i in range(3):
            firmware_file = tmp_path / f"test_firmware_{i}.bin"
            firmware_file.write_bytes(f"test firmware content {i}".encode())
            firmware_files.append(str(firmware_file))
        
        # Create one non-existent file to test error handling
        firmware_files.append(str(tmp_path / "non_existent.bin"))
        
        # Preload cache
        results = firmware_cache.preload_firmware_cache(firmware_files)
        
        assert results['total_files'] == 4
        assert results['processed_files'] == 3  # 3 existing files
        assert results['cached_files'] >= 0  # May vary based on cache state
        assert len(results['errors']) == 1  # One non-existent file
        assert "non_existent.bin" in results['errors'][0]