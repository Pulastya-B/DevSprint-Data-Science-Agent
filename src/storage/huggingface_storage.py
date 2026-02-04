"""
HuggingFace Storage Service

Stores user artifacts (datasets, models, plots, reports) directly to the user's
HuggingFace account, enabling:
1. Persistent storage at no cost
2. Easy model deployment
3. User ownership of data
4. Version control via Git
"""

import os
import json
import gzip
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, BinaryIO, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Optional: huggingface_hub for HF operations
try:
    from huggingface_hub import HfApi, upload_folder
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")


class HuggingFaceStorage:
    """
    Manages file storage on HuggingFace for user artifacts.
    
    Storage structure on HuggingFace:
    - Datasets repo: {username}/ds-agent-data
      - /datasets/{session_id}/cleaned_data.csv.gz
      - /datasets/{session_id}/encoded_data.csv.gz
    
    - Models repo: {username}/ds-agent-models  
      - /models/{session_id}/{model_name}.pkl
      - /models/{session_id}/model_config.json
    
    - Spaces repo (for reports/plots): {username}/ds-agent-outputs
      - /plots/{session_id}/correlation_heatmap.json
      - /reports/{session_id}/eda_report.html.gz
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize HuggingFace storage.
        
        Args:
            hf_token: HuggingFace API token with write permissions
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
        
        self.token = hf_token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("HuggingFace token is required")
        
        self.api = HfApi(token=self.token)
        self._username: Optional[str] = None
        
        # Repo names
        self.DATA_REPO_SUFFIX = "ds-agent-data"
        self.MODELS_REPO_SUFFIX = "ds-agent-models"
        self.OUTPUTS_REPO_SUFFIX = "ds-agent-outputs"
    
    @property
    def username(self) -> str:
        """Get the authenticated user's username."""
        if self._username is None:
            user_info = self.api.whoami()
            self._username = user_info["name"]
        return self._username
    
    def _get_repo_id(self, repo_type: str) -> str:
        """Get the full repo ID for a given type."""
        suffix_map = {
            "data": self.DATA_REPO_SUFFIX,
            "models": self.MODELS_REPO_SUFFIX,
            "outputs": self.OUTPUTS_REPO_SUFFIX
        }
        suffix = suffix_map.get(repo_type, self.OUTPUTS_REPO_SUFFIX)
        return f"{self.username}/{suffix}"
    
    def _ensure_repo_exists(self, repo_type: str, repo_kind: str = "dataset") -> str:
        """
        Ensure the repository exists, create if not.
        
        Args:
            repo_type: "data", "models", or "outputs"
            repo_kind: "dataset", "model", or "space"
        
        Returns:
            The repo ID
        """
        repo_id = self._get_repo_id(repo_type)
        
        try:
            self.api.repo_info(repo_id=repo_id, repo_type=repo_kind)
            logger.info(f"Repo {repo_id} exists")
        except RepositoryNotFoundError:
            logger.info(f"Creating repo {repo_id}")
            self.api.create_repo(
                repo_id=repo_id,
                repo_type=repo_kind,
                private=True,  # Default to private
                exist_ok=True  # Don't fail if already exists
            )
        
        return repo_id
    
    def upload_dataset(
        self,
        file_path: str,
        session_id: str,
        file_name: Optional[str] = None,
        compress: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a dataset (CSV, Parquet) to user's HuggingFace.
        
        Args:
            file_path: Local path to the file
            session_id: Session ID for organizing files
            file_name: Optional custom filename
            compress: Whether to gzip compress the file
            metadata: Optional metadata to store alongside
        
        Returns:
            Dict with upload info (url, path, size, etc.)
        """
        repo_id = self._ensure_repo_exists("data", "dataset")
        
        original_path = Path(file_path)
        file_name = file_name or original_path.name
        
        # Compress if requested and not already compressed
        if compress and not file_name.endswith('.gz'):
            with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as tmp:
                with open(file_path, 'rb') as f_in:
                    with gzip.open(tmp.name, 'wb') as f_out:
                        f_out.write(f_in.read())
                upload_path = tmp.name
                file_name = f"{file_name}.gz"
        else:
            upload_path = file_path
        
        # Upload to HuggingFace
        path_in_repo = f"datasets/{session_id}/{file_name}"
        
        try:
            result = self.api.upload_file(
                path_or_fileobj=upload_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                                commit_message=f"Add dataset: {file_name}"
            )
            
            # Upload metadata if provided
            if metadata:
                metadata_path = f"datasets/{session_id}/{file_name}.meta.json"
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                    json.dump({
                        **metadata,
                        "uploaded_at": datetime.now().isoformat(),
                        "original_name": original_path.name,
                        "compressed": compress
                    }, tmp)
                    tmp.flush()
                    
                    self.api.upload_file(
                        path_or_fileobj=tmp.name,
                        path_in_repo=metadata_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                                                commit_message=f"Add metadata for {file_name}"
                    )
            
            file_size = os.path.getsize(upload_path)
            
            return {
                "success": True,
                "repo_id": repo_id,
                "path": path_in_repo,
                "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
                "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}",
                "size_bytes": file_size,
                "compressed": compress
            }
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up temp file if we created one
            if compress and upload_path != file_path:
                try:
                    os.unlink(upload_path)
                except:
                    pass
    
    def upload_model(
        self,
        model_path: str,
        session_id: str,
        model_name: str,
        model_type: str = "sklearn",
        metrics: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a trained model to user's HuggingFace.
        
        Args:
            model_path: Local path to the model file (.pkl, .joblib, .pt, etc.)
            session_id: Session ID
            model_name: Name for the model
            model_type: Type of model (sklearn, xgboost, pytorch, etc.)
            metrics: Model performance metrics
            feature_names: List of feature names the model expects
            target_column: Target column name
        
        Returns:
            Dict with upload info
        """
        repo_id = self._ensure_repo_exists("models", "model")
        
        path_in_repo = f"models/{session_id}/{model_name}"
        model_file_name = Path(model_path).name
        
        try:
            # Upload the model file
            self.api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo=f"{path_in_repo}/{model_file_name}",
                repo_id=repo_id,
                repo_type="model",
                                commit_message=f"Add model: {model_name}"
            )
            
            # Create and upload model card
            model_card = self._generate_model_card(
                model_name=model_name,
                model_type=model_type,
                metrics=metrics,
                feature_names=feature_names,
                target_column=target_column
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
                tmp.write(model_card)
                tmp.flush()
                
                self.api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=f"{path_in_repo}/README.md",
                    repo_id=repo_id,
                    repo_type="model",
                                        commit_message=f"Add model card for {model_name}"
                )
            
            # Upload config
            config = {
                "model_name": model_name,
                "model_type": model_type,
                "model_file": model_file_name,
                "metrics": metrics or {},
                "feature_names": feature_names or [],
                "target_column": target_column,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                json.dump(config, tmp, indent=2)
                tmp.flush()
                
                self.api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=f"{path_in_repo}/config.json",
                    repo_id=repo_id,
                    repo_type="model",
                                        commit_message=f"Add config for {model_name}"
                )
            
            return {
                "success": True,
                "repo_id": repo_id,
                "path": path_in_repo,
                "url": f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}",
                "model_type": model_type,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_plot(
        self,
        plot_data: Union[str, Dict],
        session_id: str,
        plot_name: str,
        plot_type: str = "plotly"
    ) -> Dict[str, Any]:
        """
        Upload plot data (as JSON) to user's HuggingFace.
        
        For Plotly charts, we store the JSON data and render client-side,
        which is much smaller than storing full HTML.
        
        Args:
            plot_data: Either JSON string or dict of plot data
            session_id: Session ID
            plot_name: Name for the plot
            plot_type: Type of plot (plotly, matplotlib, etc.)
        
        Returns:
            Dict with upload info
        """
        repo_id = self._ensure_repo_exists("outputs", "dataset")
        
        # Ensure we have JSON string
        if isinstance(plot_data, dict):
            plot_json = json.dumps(plot_data)
        else:
            plot_json = plot_data
        
        path_in_repo = f"plots/{session_id}/{plot_name}.json"
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp.write(plot_json)
                tmp.flush()
                
                self.api.upload_file(
                    path_or_fileobj=tmp.name,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="dataset",
                                        commit_message=f"Add plot: {plot_name}"
                )
            
            return {
                "success": True,
                "repo_id": repo_id,
                "path": path_in_repo,
                "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
                "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}",
                "plot_type": plot_type,
                "size_bytes": len(plot_json.encode())
            }
            
        except Exception as e:
            logger.error(f"Failed to upload plot: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_report(
        self,
        report_path: str,
        session_id: str,
        report_name: str,
        compress: bool = True
    ) -> Dict[str, Any]:
        """
        Upload an HTML report to user's HuggingFace.
        
        Args:
            report_path: Local path to the HTML report
            session_id: Session ID
            report_name: Name for the report
            compress: Whether to gzip compress
        
        Returns:
            Dict with upload info
        """
        repo_id = self._ensure_repo_exists("outputs", "dataset")
        
        file_name = f"{report_name}.html"
        
        # Compress if requested
        if compress:
            with tempfile.NamedTemporaryFile(suffix='.html.gz', delete=False) as tmp:
                with open(report_path, 'rb') as f_in:
                    with gzip.open(tmp.name, 'wb') as f_out:
                        f_out.write(f_in.read())
                upload_path = tmp.name
                file_name = f"{file_name}.gz"
        else:
            upload_path = report_path
        
        path_in_repo = f"reports/{session_id}/{file_name}"
        
        try:
            self.api.upload_file(
                path_or_fileobj=upload_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                                commit_message=f"Add report: {report_name}"
            )
            
            file_size = os.path.getsize(upload_path)
            
            return {
                "success": True,
                "repo_id": repo_id,
                "path": path_in_repo,
                "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
                "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}",
                "size_bytes": file_size,
                "compressed": compress
            }
            
        except Exception as e:
            logger.error(f"Failed to upload report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if compress and upload_path != report_path:
                try:
                    os.unlink(upload_path)
                except:
                    pass
    
    def upload_generic_file(
        self,
        file_path: str,
        session_id: str,
        subfolder: str = "files"
    ) -> Dict[str, Any]:
        """
        Upload any file to user's HuggingFace outputs repo.
        
        Args:
            file_path: Local path to the file
            session_id: Session ID
            subfolder: Subfolder within outputs (e.g., "plots", "images", "files")
        
        Returns:
            Dict with upload info
        """
        repo_id = self._ensure_repo_exists("outputs", "dataset")
        
        file_name = Path(file_path).name
        path_in_repo = f"{subfolder}/{session_id}/{file_name}"
        
        try:
            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                                commit_message=f"Add {subfolder}: {file_name}"
            )
            
            file_size = os.path.getsize(file_path)
            
            return {
                "success": True,
                "repo_id": repo_id,
                "path": path_in_repo,
                "url": f"https://huggingface.co/datasets/{repo_id}/blob/main/{path_in_repo}",
                "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path_in_repo}",
                "size_bytes": file_size
            }
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def list_user_files(
        self,
        session_id: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all files for the user, optionally filtered by session or type.
        
        Args:
            session_id: Optional session ID to filter by
            file_type: Optional type ("datasets", "models", "plots", "reports")
        
        Returns:
            Dict with lists of files by type
        """
        result = {
            "datasets": [],
            "models": [],
            "plots": [],
            "reports": []
        }
        
        try:
            # List datasets
            if file_type is None or file_type == "datasets":
                repo_id = self._get_repo_id("data")
                try:
                    files = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                    for f in files:
                        if f.startswith("datasets/") and not f.endswith(".meta.json"):
                            if session_id is None or f"/{session_id}/" in f:
                                result["datasets"].append({
                                    "path": f,
                                    "name": Path(f).name,
                                    "session_id": f.split("/")[1] if len(f.split("/")) > 1 else None,
                                    "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                                })
                except:
                    pass
            
            # List models
            if file_type is None or file_type == "models":
                repo_id = self._get_repo_id("models")
                try:
                    files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
                    for f in files:
                        if f.startswith("models/") and f.endswith("config.json"):
                            if session_id is None or f"/{session_id}/" in f:
                                model_path = "/".join(f.split("/")[:-1])
                                result["models"].append({
                                    "path": model_path,
                                    "name": f.split("/")[-2] if len(f.split("/")) > 2 else None,
                                    "session_id": f.split("/")[1] if len(f.split("/")) > 1 else None,
                                    "url": f"https://huggingface.co/{repo_id}/tree/main/{model_path}"
                                })
                except:
                    pass
            
            # List plots and reports
            if file_type is None or file_type in ["plots", "reports"]:
                repo_id = self._get_repo_id("outputs")
                try:
                    files = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
                    for f in files:
                        if f.startswith("plots/"):
                            if session_id is None or f"/{session_id}/" in f:
                                result["plots"].append({
                                    "path": f,
                                    "name": Path(f).stem,
                                    "session_id": f.split("/")[1] if len(f.split("/")) > 1 else None,
                                    "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                                })
                        elif f.startswith("reports/"):
                            if session_id is None or f"/{session_id}/" in f:
                                result["reports"].append({
                                    "path": f,
                                    "name": Path(f).stem.replace(".html", ""),
                                    "session_id": f.split("/")[1] if len(f.split("/")) > 1 else None,
                                    "download_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}"
                                })
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
        
        return result
    
    def _generate_model_card(
        self,
        model_name: str,
        model_type: str,
        metrics: Optional[Dict[str, float]] = None,
        feature_names: Optional[List[str]] = None,
        target_column: Optional[str] = None
    ) -> str:
        """Generate a HuggingFace model card."""
        
        metrics_str = ""
        if metrics:
            metrics_str = "\n".join([f"- **{k}**: {v:.4f}" for k, v in metrics.items()])
        
        features_str = ""
        if feature_names:
            features_str = ", ".join(f"`{f}`" for f in feature_names[:20])
            if len(feature_names) > 20:
                features_str += f" ... and {len(feature_names) - 20} more"
        
        return f"""---
license: apache-2.0
tags:
- tabular
- {model_type}
- ds-agent
---

# {model_name}

This model was trained using [DS Agent](https://huggingface.co/spaces/Pulastya0/Data-Science-Agent), 
an AI-powered data science assistant.

## Model Details

- **Model Type**: {model_type}
- **Target Column**: {target_column or "Not specified"}
- **Created**: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Performance Metrics

{metrics_str or "No metrics recorded"}

## Features

{features_str or "Feature names not recorded"}

## Usage

```python
import joblib

# Load the model
model = joblib.load("model.pkl")

# Make predictions
predictions = model.predict(X_new)
```

## Training

This model was automatically trained using DS Agent's ML pipeline which includes:
- Automated data cleaning
- Feature engineering
- Hyperparameter optimization with Optuna
- Cross-validation

---

*Generated by DS Agent*
"""
    
    def get_user_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the user."""
        stats = {
            "datasets_count": 0,
            "models_count": 0,
            "plots_count": 0,
            "reports_count": 0,
            "total_files": 0
        }
        
        files = self.list_user_files()
        stats["datasets_count"] = len(files["datasets"])
        stats["models_count"] = len(files["models"])
        stats["plots_count"] = len(files["plots"])
        stats["reports_count"] = len(files["reports"])
        stats["total_files"] = sum(stats.values()) - stats["total_files"]
        
        return stats


# Convenience function for creating storage instance
def get_hf_storage(token: str) -> Optional[HuggingFaceStorage]:
    """
    Create a HuggingFace storage instance.
    
    Args:
        token: HuggingFace API token
    
    Returns:
        HuggingFaceStorage instance or None if not available
    """
    if not HF_AVAILABLE:
        logger.error("huggingface_hub not installed")
        return None
    
    try:
        return HuggingFaceStorage(hf_token=token)
    except Exception as e:
        logger.error(f"Failed to create HF storage: {e}")
        return None
