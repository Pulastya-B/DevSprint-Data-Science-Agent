"""
Session Storage Manager
Persists session memory to SQLite database for cross-session continuity.

Enables users to resume conversations even after restarting the agent.
"""

import sqlite3
import json
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from session_memory import SessionMemory


class SessionStore:
    """
    Persistent storage for session memory using SQLite.
    
    Features:
    - Save/load sessions by ID
    - Resume most recent session automatically
    - Cleanup old sessions
    - List all sessions
    
    Storage location: ./cache_db/sessions.db
    """
    
    def __init__(self, db_path: str = "./cache_db/sessions.db"):
        """
        Initialize session store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_database()
    
    def _init_database(self):
        """Create sessions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                context_json TEXT NOT NULL
            )
        """)
        
        # Create index on last_active for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_active 
            ON sessions(last_active DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        Handles matplotlib Figures, plotly Figures, numpy arrays, and other non-serializable types.
        """
        try:
            import numpy as np
        except ImportError:
            np = None
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        
        # Handle lists recursively
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle matplotlib Figure objects
        elif hasattr(obj, '__class__') and 'Figure' in obj.__class__.__name__:
            return f"<{obj.__class__.__name__} object: {id(obj)}>"
        
        # Handle numpy arrays
        elif np and isinstance(obj, np.ndarray):
            return f"<NumPy array: shape={obj.shape}>"
        
        # Handle numpy scalar types
        elif hasattr(obj, 'item') and callable(obj.item):
            try:
                return obj.item()
            except:
                return str(obj)
        
        # Handle other non-serializable objects (dataframes, models, etc.)
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            # Check if it's a common non-serializable type
            class_name = obj.__class__.__name__
            if class_name in ['DataFrame', 'Series', 'Model', 'Pipeline', 'Figure']:
                return f"<{class_name} object: {id(obj)}>"
            return f"<{class_name} object>"
        
        # Already serializable
        return obj
    
    def save(self, session: SessionMemory):
        """
        Save session to database.
        
        Args:
            session: SessionMemory instance to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize session to JSON - clean non-serializable objects first
        data = session.to_dict()
        clean_data = self._make_json_serializable(data)
        
        cursor.execute("""
            INSERT OR REPLACE INTO sessions (session_id, created_at, last_active, context_json)
            VALUES (?, ?, ?, ?)
        """, (
            session.session_id,
            session.created_at.isoformat(),
            session.last_active.isoformat(),
            json.dumps(clean_data)
        ))
        
        conn.commit()
        conn.close()
    
    def load(self, session_id: str) -> Optional[SessionMemory]:
        """
        Load session from database by ID.
        
        Args:
            session_id: Unique session identifier
        
        Returns:
            SessionMemory instance or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT context_json FROM sessions WHERE session_id = ?
        """, (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        # Deserialize JSON to SessionMemory
        data = json.loads(result[0])
        return SessionMemory.from_dict(data)
    
    def get_recent_session(self, max_age_hours: int = 24) -> Optional[SessionMemory]:
        """
        Get most recent active session within time window.
        
        Useful for automatic session resumption when user returns.
        
        Args:
            max_age_hours: Maximum age in hours (default: 24)
        
        Returns:
            Most recent SessionMemory or None if no recent sessions
        
        Example:
            # Resume conversation from yesterday
            session = store.get_recent_session(max_age_hours=24)
            if session:
                print(f"Resuming session: {session.last_dataset}")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        
        cursor.execute("""
            SELECT context_json FROM sessions
            WHERE last_active > ?
            ORDER BY last_active DESC
            LIMIT 1
        """, (cutoff_time,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        data = json.loads(result[0])
        return SessionMemory.from_dict(data)
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, str]]:
        """
        List recent sessions with basic info.
        
        Args:
            limit: Maximum number of sessions to return
        
        Returns:
            List of session info dicts with id, created_at, last_active
        
        Example:
            sessions = store.list_sessions(limit=5)
            for s in sessions:
                print(f"{s['session_id']}: {s['last_active']}")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, created_at, last_active
            FROM sessions
            ORDER BY last_active DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "session_id": row[0],
                "created_at": row[1],
                "last_active": row[2]
            }
            for row in results
        ]
    
    def delete(self, session_id: str) -> bool:
        """
        Delete session from database.
        
        Args:
            session_id: Session to delete
        
        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        rows_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_deleted > 0
    
    def cleanup_old_sessions(self, days: int = 7) -> int:
        """
        Delete sessions older than specified days.
        
        Args:
            days: Age threshold in days
        
        Returns:
            Number of sessions deleted
        
        Example:
            # Delete sessions older than 7 days
            deleted = store.cleanup_old_sessions(days=7)
            print(f"Cleaned up {deleted} old sessions")
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff_time,))
        rows_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return rows_deleted
    
    def get_session_count(self) -> int:
        """
        Get total number of sessions in database.
        
        Returns:
            Session count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
