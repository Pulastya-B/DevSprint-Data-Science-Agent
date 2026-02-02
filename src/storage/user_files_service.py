"""
User Files Service - Manages file metadata in Supabase

This service:
1. Tracks all user files (plots, CSVs, reports, models) in Supabase
2. Provides file listing for the Assets panel
3. Handles file expiration and cleanup coordination
4. Works with R2StorageService for actual file storage
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Supabase client import
try:
    from supabase import create_client, Client
except ImportError:
    print("Warning: supabase package not installed. Run: pip install supabase")
    Client = None

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Use service key for backend


class FileType(Enum):
    PLOT = "plot"
    CSV = "csv"
    REPORT = "report"
    MODEL = "model"


@dataclass
class UserFile:
    """Represents a user file record."""
    id: str
    user_id: str
    session_id: Optional[str]
    file_type: FileType
    file_name: str
    r2_key: str
    size_bytes: int
    mime_type: str
    metadata: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    download_url: Optional[str] = None


class UserFilesService:
    """Service for managing user file metadata in Supabase."""
    
    def __init__(self):
        """Initialize Supabase client."""
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        
        self.client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        self.table = "user_files"
    
    # ==================== CREATE ====================
    
    def create_file_record(
        self,
        user_id: str,
        file_type: FileType,
        file_name: str,
        r2_key: str,
        size_bytes: int,
        session_id: Optional[str] = None,
        mime_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_days: int = 7
    ) -> UserFile:
        """
        Create a file record in Supabase.
        
        Args:
            user_id: User ID
            file_type: Type of file
            file_name: Display name
            r2_key: R2 storage key
            size_bytes: File size
            session_id: Optional chat session ID
            mime_type: MIME type
            metadata: Additional metadata (plot type, metrics, etc.)
            expires_in_days: Days until file expires
            
        Returns:
            Created UserFile record
        """
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        data = {
            "user_id": user_id,
            "session_id": session_id,
            "file_type": file_type.value,
            "file_name": file_name,
            "r2_key": r2_key,
            "size_bytes": size_bytes,
            "mime_type": mime_type,
            "metadata": metadata or {},
            "expires_at": expires_at.isoformat()
        }
        
        result = self.client.table(self.table).insert(data).execute()
        
        if result.data:
            return self._to_user_file(result.data[0])
        raise Exception("Failed to create file record")
    
    # ==================== READ ====================
    
    def get_user_files(
        self,
        user_id: str,
        file_type: Optional[FileType] = None,
        session_id: Optional[str] = None,
        include_expired: bool = False
    ) -> List[UserFile]:
        """
        Get all files for a user.
        
        Args:
            user_id: User ID
            file_type: Optional filter by type
            session_id: Optional filter by session
            include_expired: Include expired files
            
        Returns:
            List of UserFile records
        """
        query = self.client.table(self.table)\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("is_deleted", False)
        
        if file_type:
            query = query.eq("file_type", file_type.value)
        
        if session_id:
            query = query.eq("session_id", session_id)
        
        if not include_expired:
            query = query.gt("expires_at", datetime.utcnow().isoformat())
        
        query = query.order("created_at", desc=True)
        
        result = query.execute()
        
        return [self._to_user_file(row) for row in (result.data or [])]
    
    def get_file_by_id(self, file_id: str) -> Optional[UserFile]:
        """Get a specific file by ID."""
        result = self.client.table(self.table)\
            .select("*")\
            .eq("id", file_id)\
            .single()\
            .execute()
        
        if result.data:
            return self._to_user_file(result.data)
        return None
    
    def get_file_by_r2_key(self, r2_key: str) -> Optional[UserFile]:
        """Get a file by R2 key."""
        result = self.client.table(self.table)\
            .select("*")\
            .eq("r2_key", r2_key)\
            .single()\
            .execute()
        
        if result.data:
            return self._to_user_file(result.data)
        return None
    
    def get_session_files(self, session_id: str) -> List[UserFile]:
        """Get all files for a chat session."""
        result = self.client.table(self.table)\
            .select("*")\
            .eq("session_id", session_id)\
            .eq("is_deleted", False)\
            .order("created_at", desc=True)\
            .execute()
        
        return [self._to_user_file(row) for row in (result.data or [])]
    
    # ==================== UPDATE ====================
    
    def extend_expiration(self, file_id: str, additional_days: int = 7) -> bool:
        """Extend file expiration date."""
        file = self.get_file_by_id(file_id)
        if not file:
            return False
        
        new_expires = datetime.utcnow() + timedelta(days=additional_days)
        
        result = self.client.table(self.table)\
            .update({"expires_at": new_expires.isoformat()})\
            .eq("id", file_id)\
            .execute()
        
        return bool(result.data)
    
    # ==================== DELETE ====================
    
    def soft_delete_file(self, file_id: str) -> bool:
        """Soft delete a file (mark as deleted)."""
        result = self.client.table(self.table)\
            .update({"is_deleted": True})\
            .eq("id", file_id)\
            .execute()
        
        return bool(result.data)
    
    def hard_delete_file(self, file_id: str) -> bool:
        """Permanently delete a file record."""
        result = self.client.table(self.table)\
            .delete()\
            .eq("id", file_id)\
            .execute()
        
        return bool(result.data)
    
    def get_expired_files(self) -> List[UserFile]:
        """Get all expired files for cleanup."""
        result = self.client.table(self.table)\
            .select("*")\
            .lt("expires_at", datetime.utcnow().isoformat())\
            .eq("is_deleted", False)\
            .execute()
        
        return [self._to_user_file(row) for row in (result.data or [])]
    
    # ==================== STATS ====================
    
    def get_user_storage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get storage statistics for a user."""
        files = self.get_user_files(user_id, include_expired=False)
        
        stats = {
            "total_files": len(files),
            "total_size_bytes": sum(f.size_bytes for f in files),
            "by_type": {}
        }
        
        for file_type in FileType:
            type_files = [f for f in files if f.file_type == file_type]
            stats["by_type"][file_type.value] = {
                "count": len(type_files),
                "size_bytes": sum(f.size_bytes for f in type_files)
            }
        
        stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
        
        return stats
    
    # ==================== HELPERS ====================
    
    def _to_user_file(self, row: Dict[str, Any]) -> UserFile:
        """Convert database row to UserFile object."""
        return UserFile(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row.get("session_id"),
            file_type=FileType(row["file_type"]),
            file_name=row["file_name"],
            r2_key=row["r2_key"],
            size_bytes=row.get("size_bytes", 0),
            mime_type=row.get("mime_type", "application/octet-stream"),
            metadata=row.get("metadata", {}),
            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            expires_at=datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        )


# ==================== SINGLETON ====================

_files_service: Optional[UserFilesService] = None

def get_files_service() -> UserFilesService:
    """Get or create UserFilesService singleton."""
    global _files_service
    if _files_service is None:
        _files_service = UserFilesService()
    return _files_service
