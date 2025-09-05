"""
Caching utilities for SyftBox NSAI SDK
"""
import json
import hashlib
import time
import logging

from typing import Any, Dict, Optional, List, Callable
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> Any:
        """Access cache entry and update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'expires_at': self.expires_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(
            key=data['key'],
            value=data['value'],
            created_at=data['created_at'],
            expires_at=data.get('expires_at'),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed'),
            metadata=data.get('metadata', {})
        )


class InMemoryCache:
    """In-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds (5 minutes)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _evict_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _evict_lru(self):
        """Remove least recently used entries if cache is full."""
        while len(self._cache) >= self.max_size:
            if not self._access_order:
                break
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)
    
    def _remove_entry(self, key: str):
        """Remove entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _update_access_order(self, key: str):
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired():
            self._remove_entry(key)
            return None
        
        self._update_access_order(key)
        return entry.access()
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None, 
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            metadata: Optional metadata
        """
        # Clean up expired entries first
        self._evict_expired()
        
        # Remove old entry if exists
        if key in self._cache:
            self._remove_entry(key)
        
        # Evict LRU if needed
        self._evict_lru()
        
        # Calculate expiration
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
            metadata=metadata
        )
        
        self._cache[key] = entry
        self._update_access_order(key)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        if key in self._cache:
            self._remove_entry(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        return list(self._cache.keys())
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._evict_expired()
        
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        entries_with_access = [e for e in self._cache.values() if e.access_count > 0]
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'total_accesses': total_accesses,
            'avg_accesses_per_entry': total_accesses / len(self._cache) if self._cache else 0,
            'entries_accessed': len(entries_with_access),
            'hit_rate': len(entries_with_access) / len(self._cache) if self._cache else 0
        }


class FileCache:
    """File-based cache for persistent storage."""
    
    def __init__(self, cache_dir: Path, max_size_mb: float = 100.0, default_ttl: float = 3600.0):
        """Initialize file cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
            default_ttl: Default TTL in seconds (1 hour)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        
        # Index file to track cache entries
        self.index_file = self.cache_dir / "cache_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load cache index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index_data = json.load(f)
                    self.index = {
                        key: CacheEntry.from_dict(data)
                        for key, data in index_data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
                self.index = {}
        else:
            self.index = {}
    
    def _save_index(self):
        """Save cache index to file."""
        try:
            index_data = {
                key: entry.to_dict()
                for key, entry in self.index.items()
            }
            with open(self.index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _get_cache_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"
    
    def _cleanup_expired(self):
        """Remove expired cache files."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.index.items():
            if entry.expires_at and current_time > entry.expires_at:
                expired_keys.append(key)
                cache_file = self._get_cache_file_path(key)
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete expired cache file {cache_file}: {e}")
        
        for key in expired_keys:
            del self.index[key]
        
        if expired_keys:
            self._save_index()
    
    def _cleanup_size(self):
        """Remove old files if cache size exceeds limit."""
        total_size = sum(
            self._get_cache_file_path(key).stat().st_size
            for key in self.index.keys()
            if self._get_cache_file_path(key).exists()
        )
        
        if total_size > self.max_size_bytes:
            # Sort by last accessed time (oldest first)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
            
            for key, entry in sorted_entries:
                if total_size <= self.max_size_bytes * 0.8:  # Keep some headroom
                    break
                
                cache_file = self._get_cache_file_path(key)
                if cache_file.exists():
                    file_size = cache_file.stat().st_size
                    try:
                        cache_file.unlink()
                        total_size -= file_size
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                
                del self.index[key]
            
            self._save_index()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        if key not in self.index:
            return None
        
        entry = self.index[key]
        
        # Check expiration
        if entry.is_expired():
            self.delete(key)
            return None
        
        # Load from file
        cache_file = self._get_cache_file_path(key)
        if not cache_file.exists():
            del self.index[key]
            self._save_index()
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Update access info
            entry.access()
            self._save_index()
            
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_file}: {e}")
            self.delete(key)
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Put value in file cache."""
        # Cleanup first
        self._cleanup_expired()
        self._cleanup_size()
        
        # Calculate expiration
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl if ttl > 0 else None
        
        # Save to file
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'w') as f:
                json.dump(value, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")
            return
        
        # Update index
        entry = CacheEntry(
            key=key,
            value=None,  # Don't store value in index, it's in the file
            created_at=time.time(),
            expires_at=expires_at,
            metadata=metadata
        )
        
        self.index[key] = entry
        self._save_index()
    
    def delete(self, key: str) -> bool:
        """Delete entry from file cache."""
        if key not in self.index:
            return False
        
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        del self.index[key]
        self._save_index()
        return True
    
    def clear(self) -> None:
        """Clear all cache files."""
        for key in list(self.index.keys()):
            self.delete(key)
    
    def size(self) -> int:
        """Get number of cache entries."""
        self._cleanup_expired()
        return len(self.index)
    
    def size_bytes(self) -> int:
        """Get total cache size in bytes."""
        return sum(
            self._get_cache_file_path(key).stat().st_size
            for key in self.index.keys()
            if self._get_cache_file_path(key).exists()
        )


class ResponseCache:
    """High-level cache for API responses."""
    
    def __init__(self, 
                 use_memory: bool = True,
                 use_file: bool = False,
                 cache_dir: Optional[Path] = None,
                 memory_size: int = 500,
                 file_size_mb: float = 50.0):
        """Initialize response cache.
        
        Args:
            use_memory: Enable in-memory cache
            use_file: Enable file cache
            cache_dir: Directory for file cache
            memory_size: Max entries for memory cache
            file_size_mb: Max size for file cache in MB
        """
        self.memory_cache = InMemoryCache(max_size=memory_size) if use_memory else None
        
        if use_file:
            cache_dir = cache_dir or Path.home() / ".syftbox" / "cache"
            self.file_cache = FileCache(cache_dir, max_size_mb=file_size_mb)
        else:
            self.file_cache = None
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key for method call."""
        key_data = {
            'method': method,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_response(self, method: str, *args, **kwargs) -> Optional[Any]:
        """Get cached response."""
        key = self._get_cache_key(method, *args, **kwargs)
        
        # Try memory cache first
        if self.memory_cache:
            result = self.memory_cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit (memory): {method}")
                return result
        
        # Try file cache
        if self.file_cache:
            result = self.file_cache.get(key)
            if result is not None:
                logger.debug(f"Cache hit (file): {method}")
                # Store in memory cache for faster future access
                if self.memory_cache:
                    self.memory_cache.put(key, result, ttl=300)  # 5 min in memory
                return result
        
        logger.debug(f"Cache miss: {method}")
        return None
    
    def cache_response(self, method: str, response: Any, ttl: Optional[float] = None,
                      memory_only: bool = False, *args, **kwargs) -> None:
        """Cache response."""
        key = self._get_cache_key(method, *args, **kwargs)
        
        # Store in memory cache
        if self.memory_cache:
            memory_ttl = ttl if memory_only else min(ttl or 300, 300)  # Max 5 min in memory
            self.memory_cache.put(key, response, ttl=memory_ttl)
        
        # Store in file cache (unless memory_only)
        if self.file_cache and not memory_only:
            self.file_cache.put(key, response, ttl=ttl)
        
        logger.debug(f"Cached response: {method}")
    
    def invalidate(self, method: str, *args, **kwargs) -> None:
        """Invalidate cached response."""
        key = self._get_cache_key(method, *args, **kwargs)
        
        if self.memory_cache:
            self.memory_cache.delete(key)
        
        if self.file_cache:
            self.file_cache.delete(key)
        
        logger.debug(f"Invalidated cache: {method}")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.file_cache:
            self.file_cache.clear()
        
        logger.debug("Cleared all cache")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}
        
        if self.memory_cache:
            stats['memory'] = self.memory_cache.stats()
        
        if self.file_cache:
            stats['file'] = {
                'size': self.file_cache.size(),
                'size_bytes': self.file_cache.size_bytes(),
                'size_mb': self.file_cache.size_bytes() / (1024 * 1024)
            }
        
        return stats


# Decorator for caching function results
def cached_response(cache: ResponseCache, ttl: Optional[float] = None, memory_only: bool = False):
    """Decorator to cache function responses.
    
    Args:
        cache: ResponseCache instance
        ttl: Time to live for cache entry
        memory_only: Store only in memory cache
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Try to get from cache
            cached = cache.get_response(func.__name__, *args, **kwargs)
            if cached is not None:
                return cached
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.cache_response(func.__name__, result, ttl=ttl, memory_only=memory_only, *args, **kwargs)
            
            return result
        return wrapper
    return decorator


def cached_async_response(cache: ResponseCache, ttl: Optional[float] = None, memory_only: bool = False):
    """Decorator to cache async function responses.
    
    Args:
        cache: ResponseCache instance
        ttl: Time to live for cache entry
        memory_only: Store only in memory cache
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Try to get from cache
            cached = cache.get_response(func.__name__, *args, **kwargs)
            if cached is not None:
                return cached
            
            # Call function and cache result
            result = await func(*args, **kwargs)
            cache.cache_response(func.__name__, result, ttl=ttl, memory_only=memory_only, *args, **kwargs)
            
            return result
        return wrapper
    return decorator


# Cache configuration for different types of data
CACHE_CONFIGS = {
    'service_discovery': {'ttl': 300, 'memory_only': False},      # 5 minutes
    'health_check': {'ttl': 60, 'memory_only': True},          # 1 minute, memory only
    'chat_response': {'ttl': 0, 'memory_only': True},          # No cache for chat
    'search_response': {'ttl': 600, 'memory_only': False},     # 10 minutes
    'account_info': {'ttl': 120, 'memory_only': True},         # 2 minutes, memory only
    'service_metadata': {'ttl': 1800, 'memory_only': False},   # 30 minutes
}


def get_cache_config(cache_type: str) -> Dict[str, Any]:
    """Get cache configuration for a specific type.
    
    Args:
        cache_type: Type of cache (service_discovery, health_check, etc.)
        
    Returns:
        Cache configuration dictionary
    """
    return CACHE_CONFIGS.get(cache_type, {'ttl': 300, 'memory_only': False})