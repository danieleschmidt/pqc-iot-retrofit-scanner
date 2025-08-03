"""Cache management for performance optimization."""

import hashlib
import json
import logging
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import threading
from functools import wraps

from ..database.connection import DatabaseManager

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages various caches for performance optimization."""
    
    def __init__(self, db_manager: DatabaseManager, cache_dir: Optional[str] = None):
        """Initialize cache manager.
        
        Args:
            db_manager: Database manager for persistent cache
            cache_dir: Directory for file-based caches
        """
        self.db = db_manager
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / '.cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for frequently accessed data
        self._memory_cache = {}
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
        self._lock = threading.RLock()
        
        # Cache configuration
        self.max_memory_items = 1000
        self.default_ttl_minutes = 60
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Try memory cache first
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if not self._is_expired(entry):
                    self._cache_stats['hits'] += 1
                    return entry['value']
                else:
                    # Remove expired entry
                    del self._memory_cache[key]
        
        # Try database cache
        cached_data = self.db.get_cache(key)
        if cached_data:
            # Store in memory cache for faster access
            self._set_memory_cache(key, cached_data, ttl_minutes=self.default_ttl_minutes)
            self._cache_stats['hits'] += 1
            return cached_data
        
        self._cache_stats['misses'] += 1
        return default
    
    def set(self, key: str, value: Any, ttl_minutes: Optional[int] = None) -> None:
        """Set cached value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_minutes: Time to live in minutes
        """
        ttl = ttl_minutes or self.default_ttl_minutes
        
        # Store in memory cache
        self._set_memory_cache(key, value, ttl)
        
        # Store in database cache
        self.db.set_cache(key, {'data': value}, ttl)
        
        self._cache_stats['sets'] += 1
    
    def _set_memory_cache(self, key: str, value: Any, ttl_minutes: int) -> None:
        """Set value in memory cache."""
        with self._lock:
            # Evict old entries if cache is full
            if len(self._memory_cache) >= self.max_memory_items:
                self._evict_lru()
            
            expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
            self._memory_cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'accessed_at': datetime.now()
            }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._memory_cache:
            return
        
        # Find least recently used item
        lru_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k]['accessed_at']
        )
        
        del self._memory_cache[lru_key]
        self._cache_stats['evictions'] += 1
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > entry['expires_at']
    
    def delete(self, key: str) -> bool:
        """Delete cached value.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted
        """
        deleted = False
        
        # Remove from memory cache
        with self._lock:
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
        
        # Remove from database cache
        self.db.clear_cache(key)
        
        return deleted
    
    def clear(self, pattern: Optional[str] = None) -> None:
        """Clear cached values.
        
        Args:
            pattern: Pattern to match keys (None clears all)
        """
        # Clear memory cache
        with self._lock:
            if pattern:
                # Simple pattern matching (contains)
                keys_to_delete = [k for k in self._memory_cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self._memory_cache[key]
            else:
                self._memory_cache.clear()
        
        # Clear database cache
        self.db.clear_cache(pattern)
    
    def cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            SHA256 hash as cache key
        """
        # Create deterministic string from arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())  # Sort for consistency
        }
        
        key_string = json.dumps(key_data, default=str, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def cached(self, ttl_minutes: Optional[int] = None, key_prefix: str = ""):
        """Decorator for caching function results.
        
        Args:
            ttl_minutes: Time to live in minutes
            key_prefix: Prefix for cache key
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = f"{key_prefix}{func.__name__}_{self.cache_key(*args, **kwargs)}"
                
                # Try to get cached result
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result['data']
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl_minutes)
                
                return result
            
            return wrapper
        return decorator
    
    def file_cache_path(self, key: str) -> Path:
        """Get file path for cache key.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Use first two characters of key for subdirectory
        subdir = key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        
        return cache_subdir / f"{key}.cache"
    
    def save_to_file(self, key: str, data: Any) -> None:
        """Save data to file cache.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        cache_file = self.file_cache_path(key)
        
        try:
            cache_data = {
                'data': data,
                'created_at': datetime.now().isoformat(),
                'key': key
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to save file cache {key}: {e}")
    
    def load_from_file(self, key: str, max_age_hours: Optional[int] = None) -> Any:
        """Load data from file cache.
        
        Args:
            key: Cache key
            max_age_hours: Maximum age in hours (None for no limit)
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_file = self.file_cache_path(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check age if specified
            if max_age_hours:
                created_at = datetime.fromisoformat(cache_data['created_at'])
                if datetime.now() - created_at > timedelta(hours=max_age_hours):
                    # Remove expired file
                    cache_file.unlink()
                    return None
            
            return cache_data['data']
            
        except Exception as e:
            logger.warning(f"Failed to load file cache {key}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            memory_stats = {
                'items': len(self._memory_cache),
                'max_items': self.max_memory_items,
                'usage_percent': (len(self._memory_cache) / self.max_memory_items) * 100
            }
        
        # Count file cache items
        file_cache_count = 0
        total_file_size = 0
        
        for cache_file in self.cache_dir.rglob('*.cache'):
            file_cache_count += 1
            total_file_size += cache_file.stat().st_size
        
        file_stats = {
            'items': file_cache_count,
            'total_size_mb': total_file_size / (1024 * 1024)
        }
        
        # Calculate hit rate
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = (self._cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_cache': memory_stats,
            'file_cache': file_stats,
            'statistics': {
                **self._cache_stats,
                'hit_rate_percent': round(hit_rate, 2)
            }
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        
        # Clean memory cache
        with self._lock:
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
                cleaned_count += 1
        
        # Clean file cache (files older than 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        
        for cache_file in self.cache_dir.rglob('*.cache'):
            try:
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to clean cache file {cache_file}: {e}")
        
        # Clean database cache
        self.db.cleanup_old_data(days_to_keep=7)
        
        logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        return cleaned_count
    
    def warm_up(self, firmware_paths: List[str]) -> None:
        """Warm up cache with commonly used data.
        
        Args:
            firmware_paths: List of firmware file paths to pre-analyze
        """
        logger.info(f"Warming up cache with {len(firmware_paths)} firmware files")
        
        # This would pre-calculate common analysis results
        # For demonstration, just log the action
        for path in firmware_paths[:5]:  # Limit to first 5 files
            cache_key = f"firmware_metadata_{hashlib.md5(path.encode()).hexdigest()}"
            
            # Check if already cached
            if self.get(cache_key) is None:
                # In practice, this would pre-calculate file metadata
                metadata = {
                    'path': path,
                    'analyzed_at': datetime.now().isoformat(),
                    'pre_warmed': True
                }
                self.set(cache_key, metadata, ttl_minutes=120)
        
        logger.info("Cache warm-up completed")