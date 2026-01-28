"""
Cache Manager for Data Science Copilot
Uses SQLite for persistent caching with hierarchical support.
Supports individual tool result caching and cache warming.
"""

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
import pickle


class CacheManager:
    """
    Manages caching of LLM responses and expensive computations.
    
    Features:
    - Hierarchical caching: file_hash → [profile, quality, features, etc.]
    - Individual tool result caching (not full workflows)
    - Cache warming on file upload
    - TTL-based invalidation
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
        """Create cache tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main cache table for individual tool results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Hierarchical cache table for file-based operations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hierarchical_cache (
                    file_hash TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args TEXT,
                    result BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    PRIMARY KEY (file_hash, tool_name, tool_args)
                )
            """)
            
            # Create indices for efficient lookup
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON cache(expires_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_file_hash 
                ON hierarchical_cache(file_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hierarchical_expires 
                ON hierarchical_cache(expires_at)
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
                    CREATE TABLE hierarchical_cache (
                        file_hash TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        tool_args TEXT,
                        result BLOB NOT NULL,
                        created_at INTEGER NOT NULL,
                        expires_at INTEGER NOT NULL,
                        PRIMARY KEY (file_hash, tool_name, tool_args)
                    )
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_expires_at 
                    ON cache(expires_at)
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_file_hash 
                    ON hierarchical_cache(file_hash)
                """)
                
                cursor.execute("""
                    CREATE INDEX idx_hierarchical_expires 
                    ON hierarchical_cache(expires_at)
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
    
    # ========================================
    # HIERARCHICAL CACHING (NEW)
    # ========================================
    
    def get_tool_result(self, file_hash: str, tool_name: str, tool_args: Dict[str, Any] = None) -> Optional[Any]:
        """
        Get cached result for a specific tool applied to a file.
        
        Args:
            file_hash: MD5 hash of the file
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool (excluding file_path)
            
        Returns:
            Cached tool result if exists and not expired, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            tool_args_str = json.dumps(tool_args or {}, sort_keys=True)
            
            cursor.execute("""
                SELECT result, expires_at 
                FROM hierarchical_cache 
                WHERE file_hash = ? AND tool_name = ? AND tool_args = ? AND expires_at > ?
            """, (file_hash, tool_name, tool_args_str, current_time))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                result_blob, expires_at = result
                cached_result = pickle.loads(result_blob)
                print(f"📦 Cache HIT: {tool_name} for file {file_hash[:8]}...")
                return cached_result
            else:
                print(f"📭 Cache MISS: {tool_name} for file {file_hash[:8]}...")
                return None
                
        except Exception as e:
            print(f"⚠️ Hierarchical cache read error: {e}")
            return None
    
    def set_tool_result(self, file_hash: str, tool_name: str, result: Any, 
                       tool_args: Dict[str, Any] = None, ttl_override: Optional[int] = None) -> None:
        """
        Cache result for a specific tool applied to a file.
        
        Args:
            file_hash: MD5 hash of the file
            tool_name: Name of the tool
            result: Tool result to cache
            tool_args: Arguments passed to the tool (excluding file_path)
            ttl_override: Optional override for TTL (seconds)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            ttl = ttl_override if ttl_override is not None else self.ttl_seconds
            expires_at = current_time + ttl
            
            tool_args_str = json.dumps(tool_args or {}, sort_keys=True)
            result_blob = pickle.dumps(result)
            
            cursor.execute("""
                INSERT OR REPLACE INTO hierarchical_cache 
                (file_hash, tool_name, tool_args, result, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (file_hash, tool_name, tool_args_str, result_blob, current_time, expires_at))
            
            conn.commit()
            conn.close()
            print(f"💾 Cached: {tool_name} for file {file_hash[:8]}...")
            
        except Exception as e:
            print(f"⚠️ Hierarchical cache write error: {e}")
    
    def get_all_tool_results_for_file(self, file_hash: str) -> Dict[str, Any]:
        """
        Get all cached tool results for a specific file.
        
        Args:
            file_hash: MD5 hash of the file
            
        Returns:
            Dictionary mapping tool_name → result for all cached results
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            
            cursor.execute("""
                SELECT tool_name, tool_args, result
                FROM hierarchical_cache
                WHERE file_hash = ? AND expires_at > ?
            """, (file_hash, current_time))
            
            results = {}
            for row in cursor.fetchall():
                tool_name, tool_args_str, result_blob = row
                tool_args = json.loads(tool_args_str)
                result = pickle.loads(result_blob)
                
                # Create unique key for tool + args combination
                if tool_args:
                    key = f"{tool_name}_{hashlib.md5(tool_args_str.encode()).hexdigest()[:8]}"
                else:
                    key = tool_name
                    
                results[key] = {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result
                }
            
            conn.close()
            
            if results:
                print(f"📦 Found {len(results)} cached results for file {file_hash[:8]}...")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error retrieving file cache results: {e}")
            return {}
    
    def warm_cache_for_file(self, file_path: str, tools_to_warm: List[str] = None) -> Dict[str, bool]:
        """
        Warm cache by pre-computing common tool results for a file.
        
        This is typically called on file upload to speed up first analysis.
        
        Args:
            file_path: Path to the file
            tools_to_warm: List of tool names to pre-compute (defaults to basic profiling tools)
            
        Returns:
            Dictionary mapping tool_name → success status
        """
        if tools_to_warm is None:
            # Default tools to warm: basic profiling operations
            tools_to_warm = [
                "profile_dataset",
                "detect_data_quality_issues",
                "analyze_correlations"
            ]
        
        file_hash = self.generate_file_hash(file_path)
        results = {}
        
        print(f"🔥 Warming cache for file {file_hash[:8]}... ({len(tools_to_warm)} tools)")
        
        # Import here to avoid circular dependency
        from ..orchestrator import DataScienceOrchestrator
        
        try:
            # Create temporary orchestrator for cache warming
            orchestrator = DataScienceOrchestrator(use_cache=False)  # Don't use cache during warming
            
            for tool_name in tools_to_warm:
                try:
                    # Execute tool
                    result = orchestrator._execute_tool(tool_name, {"file_path": file_path})
                    
                    # Cache the result
                    if result.get("success", True):
                        self.set_tool_result(file_hash, tool_name, result)
                        results[tool_name] = True
                        print(f"   ✓ Warmed: {tool_name}")
                    else:
                        results[tool_name] = False
                        print(f"   ✗ Failed: {tool_name}")
                        
                except Exception as e:
                    results[tool_name] = False
                    print(f"   ✗ Error warming {tool_name}: {e}")
            
            print(f"✅ Cache warming complete: {sum(results.values())}/{len(tools_to_warm)} successful")
            
        except Exception as e:
            print(f"❌ Cache warming failed: {e}")
        
        return results
    
    def invalidate_file_cache(self, file_hash: str) -> int:
        """
        Invalidate all cached results for a specific file.
        
        Args:
            file_hash: MD5 hash of the file
            
        Returns:
            Number of entries invalidated
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM hierarchical_cache WHERE file_hash = ?", (file_hash,))
            deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted > 0:
                print(f"🗑️ Invalidated {deleted} cached results for file {file_hash[:8]}...")
            
            return deleted
            
        except Exception as e:
            print(f"⚠️ Error invalidating file cache: {e}")
            return 0
