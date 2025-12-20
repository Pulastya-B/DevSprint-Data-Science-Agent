"""
Artifact Storage Abstraction Layer

Provides unified interface for saving models, plots, reports, and data files
to either local filesystem or Google Cloud Storage (GCS).

Design Principles:
- Backend chosen via environment variable (ARTIFACT_BACKEND=local|gcs)
- Tools never know which backend is used (clean separation)
- GCS paths versioned with timestamps for reproducibility
- Consistent return format: local paths or GCS URIs
- Graceful fallback to local if GCS unavailable

Architecture:
    Tool → ArtifactStore → LocalBackend / GCSBackend

Usage:
    from storage import get_artifact_store
    
    store = get_artifact_store()
    
    # Save model
    path = store.save_model("model.pkl", metadata={"accuracy": 0.95})
    
    # Save plot
    path = store.save_plot("correlation_heatmap.html")
    
    # Save report
    path = store.save_report("eda_report.html")
    
    # Save data file
    path = store.save_data("cleaned_data.csv")
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def save_file(
        self, 
        local_path: Union[str, Path], 
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save file to backend storage.
        
        Args:
            local_path: Path to local file to save
            artifact_type: Type of artifact (model, plot, report, data)
            metadata: Optional metadata to save alongside artifact
            
        Returns:
            Storage path or URI where file was saved
        """
        pass
    
    @abstractmethod
    def list_artifacts(self, artifact_type: str) -> list[str]:
        """List all artifacts of given type."""
        pass
    
    @abstractmethod
    def get_artifact_path(self, artifact_type: str, filename: str) -> str:
        """Get full path/URI for an artifact."""
        pass


class LocalBackend(StorageBackend):
    """
    Local filesystem storage backend.
    
    Preserves existing behavior - saves to ./outputs/ directory structure.
    """
    
    def __init__(self, base_dir: str = "./outputs"):
        """
        Initialize local backend.
        
        Args:
            base_dir: Base directory for all artifacts (default: ./outputs)
        """
        self.base_dir = Path(base_dir)
        
        # Create subdirectories
        self.subdirs = {
            "model": self.base_dir / "models",
            "plot": self.base_dir / "plots",
            "report": self.base_dir / "reports",
            "data": self.base_dir / "data",
            "code": self.base_dir / "code"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
    
    def save_file(
        self, 
        local_path: Union[str, Path], 
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save file to local filesystem.
        
        Args:
            local_path: Path to source file
            artifact_type: Type (model, plot, report, data, code)
            metadata: Optional metadata (saved as JSON sidecar)
            
        Returns:
            Absolute path where file was saved
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Source file not found: {local_path}")
        
        # Determine target directory
        target_dir = self.subdirs.get(artifact_type)
        if target_dir is None:
            raise ValueError(
                f"Unknown artifact type: {artifact_type}. "
                f"Must be one of: {list(self.subdirs.keys())}"
            )
        
        # Preserve filename
        target_path = target_dir / local_path.name
        
        # Copy file (if not already in target location)
        if local_path.resolve() != target_path.resolve():
            import shutil
            shutil.copy2(local_path, target_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = target_path.with_suffix(target_path.suffix + ".meta.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "artifact_type": artifact_type,
                    "filename": local_path.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "backend": "local",
                    **metadata
                }, f, indent=2)
        
        return str(target_path.resolve())
    
    def list_artifacts(self, artifact_type: str) -> list[str]:
        """List all artifacts of given type in local storage."""
        # Validate artifact type
        valid_types = ["model", "plot", "report", "data", "code"]
        if artifact_type not in valid_types:
            raise ValueError(
                f"Invalid artifact type: {artifact_type}. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
        target_dir = self.subdirs.get(artifact_type)
        if target_dir is None or not target_dir.exists():
            return []
        
        # Exclude metadata files
        return [
            str(f.resolve()) 
            for f in target_dir.iterdir() 
            if f.is_file() and not f.name.endswith(".meta.json")
        ]
    
    def get_artifact_path(self, artifact_type: str, filename: str) -> str:
        """Get full local path for artifact."""
        target_dir = self.subdirs.get(artifact_type)
        if target_dir is None:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
        
        return str((target_dir / filename).resolve())


class GCSBackend(StorageBackend):
    """
    Google Cloud Storage backend.
    
    Saves artifacts to GCS bucket with versioned paths.
    """
    
    def __init__(
        self, 
        bucket_name: Optional[str] = None,
        project_id: Optional[str] = None,
        base_prefix: str = "artifacts"
    ):
        """
        Initialize GCS backend.
        
        Args:
            bucket_name: GCS bucket name (from env: GCS_BUCKET_NAME)
            project_id: GCP project ID (from env: GCP_PROJECT_ID)
            base_prefix: Base prefix for all artifacts (default: artifacts)
        """
        try:
            from google.cloud import storage
            from google.auth import default as gcp_default
        except ImportError:
            raise ImportError(
                "GCS backend requires google-cloud-storage. "
                "Install with: pip install google-cloud-storage"
            )
        
        # Get configuration from environment
        self.bucket_name = bucket_name or os.getenv("GCS_BUCKET_NAME")
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.base_prefix = base_prefix
        
        if not self.bucket_name:
            raise ValueError(
                "GCS bucket name not specified. "
                "Set GCS_BUCKET_NAME environment variable or pass bucket_name."
            )
        
        # Initialize GCS client
        try:
            if self.project_id:
                self.client = storage.Client(project=self.project_id)
            else:
                # Use default credentials
                credentials, project = gcp_default()
                self.client = storage.Client(credentials=credentials, project=project)
                self.project_id = project
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize GCS client: {e}\n"
                "Ensure credentials are configured (GOOGLE_APPLICATION_CREDENTIALS "
                "or gcloud auth application-default login)"
            )
        
        # Get bucket
        try:
            self.bucket = self.client.bucket(self.bucket_name)
            # Verify bucket exists
            if not self.bucket.exists():
                raise ValueError(f"Bucket '{self.bucket_name}' does not exist")
        except Exception as e:
            raise RuntimeError(f"Failed to access bucket '{self.bucket_name}': {e}")
    
    def _get_versioned_path(self, artifact_type: str, filename: str) -> str:
        """
        Generate versioned GCS path.
        
        Format: artifacts/{type}/{YYYY-MM-DD}/{timestamp}_{filename}
        
        Example: artifacts/models/2025-12-23/20251223_143052_model.pkl
        """
        timestamp = datetime.utcnow()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        versioned_filename = f"{time_str}_{filename}"
        
        return f"{self.base_prefix}/{artifact_type}/{date_str}/{versioned_filename}"
    
    def save_file(
        self, 
        local_path: Union[str, Path], 
        artifact_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload file to GCS with versioned path.
        
        Args:
            local_path: Path to local file to upload
            artifact_type: Type (model, plot, report, data, code)
            metadata: Optional metadata (stored as blob metadata)
            
        Returns:
            GCS URI (gs://bucket/path)
        """
        local_path = Path(local_path)
        
        if not local_path.exists():
            raise FileNotFoundError(f"Source file not found: {local_path}")
        
        # Generate versioned path
        gcs_path = self._get_versioned_path(artifact_type, local_path.name)
        
        # Create blob
        blob = self.bucket.blob(gcs_path)
        
        # Set metadata
        if metadata:
            blob.metadata = {
                "artifact_type": artifact_type,
                "filename": local_path.name,
                "timestamp": datetime.utcnow().isoformat(),
                "backend": "gcs",
                **{k: str(v) for k, v in metadata.items()}  # Convert all to strings
            }
        
        # Upload file
        try:
            blob.upload_from_filename(str(local_path))
        except Exception as e:
            raise RuntimeError(f"Failed to upload to GCS: {e}")
        
        # Return GCS URI
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        
        return gcs_uri
    
    def list_artifacts(self, artifact_type: str) -> list[str]:
        """List all artifacts of given type in GCS."""
        # Validate artifact type
        valid_types = ["model", "plot", "report", "data", "code"]
        if artifact_type not in valid_types:
            raise ValueError(
                f"Invalid artifact type: {artifact_type}. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
        prefix = f"{self.base_prefix}/{artifact_type}/"
        
        try:
            blobs = self.client.list_blobs(self.bucket, prefix=prefix)
            return [f"gs://{self.bucket_name}/{blob.name}" for blob in blobs]
        except Exception as e:
            raise RuntimeError(f"Failed to list GCS artifacts: {e}")
    
    def get_artifact_path(self, artifact_type: str, filename: str) -> str:
        """Get latest GCS path for artifact (most recent version)."""
        artifacts = self.list_artifacts(artifact_type)
        
        # Filter by filename (strip timestamp prefix)
        matching = [
            uri for uri in artifacts 
            if uri.endswith(f"_{filename}") or uri.endswith(f"/{filename}")
        ]
        
        if not matching:
            raise FileNotFoundError(
                f"No artifact found with filename '{filename}' in type '{artifact_type}'"
            )
        
        # Return most recent (last in sorted list)
        return sorted(matching)[-1]


class ArtifactStore:
    """
    Unified interface for artifact storage.
    
    Automatically routes to correct backend based on configuration.
    Tools use this class and never directly interact with backends.
    """
    
    def __init__(self, backend: Optional[StorageBackend] = None):
        """
        Initialize artifact store with backend.
        
        Args:
            backend: Storage backend (auto-detected if None)
        """
        if backend is None:
            backend = self._detect_backend()
        
        self.backend = backend
    
    def _detect_backend(self) -> StorageBackend:
        """
        Detect and initialize appropriate backend.
        
        Detection logic:
        1. Check ARTIFACT_BACKEND env var (local|gcs)
        2. If GCS, check for GCS_BUCKET_NAME
        3. Fall back to local if anything fails
        
        Returns:
            Initialized storage backend
        """
        backend_type = os.getenv("ARTIFACT_BACKEND", "local").lower()
        
        if backend_type == "gcs":
            try:
                # Try to initialize GCS
                bucket_name = os.getenv("GCS_BUCKET_NAME")
                if not bucket_name:
                    print("⚠️  GCS backend requested but GCS_BUCKET_NAME not set. Falling back to local.")
                    return LocalBackend()
                
                print(f"🔵 Initializing GCS backend (bucket: {bucket_name})")
                return GCSBackend(bucket_name=bucket_name)
                
            except Exception as e:
                print(f"⚠️  GCS backend initialization failed: {e}")
                print("   Falling back to local storage.")
                return LocalBackend()
        
        elif backend_type == "local":
            print("📁 Using local filesystem backend")
            return LocalBackend()
        
        else:
            print(f"⚠️  Unknown ARTIFACT_BACKEND: {backend_type}. Using local.")
            return LocalBackend()
    
    def save_model(
        self, 
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save machine learning model.
        
        Args:
            local_path: Path to model file (e.g., model.pkl)
            metadata: Optional metadata (accuracy, hyperparameters, etc.)
            
        Returns:
            Storage path or URI where model was saved
            
        Example:
            store = ArtifactStore()
            path = store.save_model(
                "model.pkl",
                metadata={"accuracy": 0.95, "model_type": "RandomForest"}
            )
        """
        return self.backend.save_file(local_path, "model", metadata)
    
    def save_plot(
        self,
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save visualization plot.
        
        Args:
            local_path: Path to plot file (e.g., plot.html, plot.png)
            metadata: Optional metadata (plot type, columns, etc.)
            
        Returns:
            Storage path or URI where plot was saved
            
        Example:
            store = ArtifactStore()
            path = store.save_plot(
                "correlation_heatmap.html",
                metadata={"plot_type": "heatmap", "columns": ["age", "income"]}
            )
        """
        return self.backend.save_file(local_path, "plot", metadata)
    
    def save_report(
        self,
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save analysis report.
        
        Args:
            local_path: Path to report file (e.g., report.html)
            metadata: Optional metadata (report type, dataset, etc.)
            
        Returns:
            Storage path or URI where report was saved
            
        Example:
            store = ArtifactStore()
            path = store.save_report(
                "eda_report.html",
                metadata={"report_type": "sweetviz", "dataset": "titanic"}
            )
        """
        return self.backend.save_file(local_path, "report", metadata)
    
    def save_data(
        self,
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save processed data file.
        
        Args:
            local_path: Path to data file (e.g., cleaned.csv)
            metadata: Optional metadata (transformation steps, row count, etc.)
            
        Returns:
            Storage path or URI where data was saved
            
        Example:
            store = ArtifactStore()
            path = store.save_data(
                "cleaned_data.csv",
                metadata={"rows": 1000, "columns": 20, "transformations": ["drop_na", "encode"]}
            )
        """
        return self.backend.save_file(local_path, "data", metadata)
    
    def save_code(
        self,
        local_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save code interpreter output.
        
        Args:
            local_path: Path to code output file
            metadata: Optional metadata (execution time, etc.)
            
        Returns:
            Storage path or URI where file was saved
        """
        return self.backend.save_file(local_path, "code", metadata)
    
    def list_artifacts(self, artifact_type: str) -> list[str]:
        """
        List all artifacts of a specific type.
        
        Args:
            artifact_type: Type of artifact (model, plot, report, data, code)
            
        Returns:
            List of artifact paths or URIs
            
        Example:
            store = ArtifactStore()
            models = store.list_artifacts("model")
            plots = store.list_artifacts("plot")
        """
        return self.backend.list_artifacts(artifact_type)
    
    def list_models(self) -> list[str]:
        """List all saved models."""
        return self.backend.list_artifacts("model")
    
    def list_plots(self) -> list[str]:
        """List all saved plots."""
        return self.backend.list_artifacts("plot")
    
    def list_reports(self) -> list[str]:
        """List all saved reports."""
        return self.backend.list_artifacts("report")
    
    def list_data_files(self) -> list[str]:
        """List all saved data files."""
        return self.backend.list_artifacts("data")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about current backend.
        
        Returns:
            Backend configuration details
        """
        if isinstance(self.backend, LocalBackend):
            return {
                "type": "local",
                "base_path": str(self.backend.base_dir.resolve()),
                "base_dir": str(self.backend.base_dir.resolve()),
                "subdirs": {k: str(v) for k, v in self.backend.subdirs.items()}
            }
        elif isinstance(self.backend, GCSBackend):
            return {
                "type": "gcs",
                "base_path": f"gs://{self.backend.bucket_name}/{self.backend.base_prefix}",
                "bucket": self.backend.bucket_name,
                "project": self.backend.project_id,
                "base_prefix": self.backend.base_prefix
            }
        else:
            return {"type": "unknown", "base_path": "unknown"}


# Singleton instance
_artifact_store_instance: Optional[ArtifactStore] = None


def get_artifact_store(backend: Optional[StorageBackend] = None) -> ArtifactStore:
    """
    Get singleton instance of ArtifactStore.
    
    This ensures all tools use the same backend configuration.
    
    Args:
        backend: Optional backend (for testing or custom configuration)
        
    Returns:
        Singleton ArtifactStore instance
        
    Example:
        from storage import get_artifact_store
        
        store = get_artifact_store()
        path = store.save_model("model.pkl", metadata={"accuracy": 0.95})
    """
    global _artifact_store_instance
    
    if _artifact_store_instance is None or backend is not None:
        _artifact_store_instance = ArtifactStore(backend=backend)
    
    return _artifact_store_instance


def reset_artifact_store():
    """
    Reset singleton instance (useful for testing).
    """
    global _artifact_store_instance
    _artifact_store_instance = None
