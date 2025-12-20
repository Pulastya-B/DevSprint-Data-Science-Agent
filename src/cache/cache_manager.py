"""
Cache Manager for Data Science Copilot
Uses SQLite for persistent caching of API responses and computation results.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional
import pickle


class CacheManager:
    """
    Manages caching of LLM responses and expensive computations.
    
    Uses SQLite for persistence and supports TTL-based invalidation.
    Cache keys are generated from file hashes and operation parameters.
    """
    
    def __init__(self, db_path: str = "./cache_db/cache.db", ttl_seconds: int = 86400):
        """
        Initialize cache manager.
        
        Args:
            db_path: Path to SQLite database file
            ttl_seconds: Time-to-live for cache entries (default 24 hours)
        """
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        
        # Ensure cache directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self) -> None:
        """Create cache table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create index on expires_at for efficient cleanup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)
            
            conn.commit()
            conn.close()
            print(f"✅ Cache database initialized at {self.db_path}")
        except Exception as e:
            print(f"⚠️ Error initializing cache database: {e}")
            print(f"   Attempting to recreate database...")
            try:
                # Remove corrupted database and recreate
                if self.db_path.exists():
                    self.db_path.unlink()
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE cache (
                        key TEXT PRIMARY KEY,
                        value BLOB NOT NULL,
                        created_at INTEGER NOT NULL,
                        expires_at INTEGER NOT NULL,
                        metadata TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_expires_at 
                    ON cache(expires_at)
                """)
                
                conn.commit()
                conn.close()
                print(f"✅ Cache database recreated successfully")
            except Exception as e2:
                print(f"❌ Failed to recreate cache database: {e2}")
                print(f"   Cache functionality will be disabled")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.
        
        Args:
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash
            
        Returns:
            MD5 hash of the arguments
        """
        # Combine args and kwargs into a single string
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value if exists and not expired, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            
            cursor.execute("""
                SELECT value, expires_at 
                FROM cache 
                WHERE key = ? AND expires_at > ?
            """, (key, current_time))
            
            result = cursor.fetchone()
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"⚠️ Cache read error: {e}")
            print(f"   Reinitializing cache database...")
            self._init_db()
            return None
        except Exception as e:
            print(f"⚠️ Unexpected cache error: {e}")
            return None
        
        if result:
            value_blob, expires_at = result
            # Deserialize using pickle for complex Python objects
            return pickle.loads(value_blob)
        
        return None
    
    def set(self, key: str, value: Any, ttl_override: Optional[int] = None, 
            metadata: Optional[dict] = None) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be pickleable)
            ttl_override: Optional override for TTL (seconds)
            metadata: Optional metadata to store with cache entry
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            ttl = ttl_override if ttl_override is not None else self.ttl_seconds
            expires_at = current_time + ttl
            
            # Serialize value using pickle
            value_blob = pickle.dumps(value)
            
            # Serialize metadata as JSON
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (key, value_blob, current_time, expires_at, metadata_json))
            
            conn.commit()
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"⚠️ Cache write error: {e}")
            print(f"   Reinitializing cache database...")
            self._init_db()
        except Exception as e:
            print(f"⚠️ Unexpected cache error during write: {e}")
    
    def invalidate(self, key: str) -> bool:
        """
        Remove specific entry from cache.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if entry was removed, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def clear_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(time.time())
        cursor.execute("DELETE FROM cache WHERE expires_at <= ?", (current_time,))
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def clear_all(self) -> None:
        """Remove all entries from cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM cache")
        
        conn.commit()
        conn.close()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (total entries, expired, size)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_time = int(time.time())
        
        # Total entries
        cursor.execute("SELECT COUNT(*) FROM cache")
        total = cursor.fetchone()[0]
        
        # Valid entries
        cursor.execute("SELECT COUNT(*) FROM cache WHERE expires_at > ?", (current_time,))
        valid = cursor.fetchone()[0]
        
        # Database size
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": total - valid,
            "size_mb": round(size_bytes / (1024 * 1024), 2)
        }
    
    def generate_file_hash(self, file_path: str) -> str:
        """
        Generate hash of file contents for cache key.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash of file contents
        """
        hasher = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
