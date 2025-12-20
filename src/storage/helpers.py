"""
Helper utilities for artifact storage integration
"""

import os
import tempfile
import joblib
from typing import Any, Dict, Optional
from pathlib import Path


def save_model_with_store(
    model_data: Any,
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
    fallback_dir: str = "./outputs/models"
) -> str:
    """
    Save model using artifact store if available, otherwise use fallback path.
    
    Args:
        model_data: Model object or dict to save
        filename: Name of the model file (e.g., "model.pkl")
        metadata: Optional metadata to attach
        fallback_dir: Directory to use if artifact store unavailable
        
    Returns:
        Path where model was saved
    """
    try:
        from storage import get_artifact_store
        store = get_artifact_store()
        
        # Save to temp file first
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as tmp:
            joblib.dump(model_data, tmp.name)
            model_path = store.save_model(tmp.name, metadata=metadata)
        os.unlink(tmp.name)
        
        return model_path
        
    except ImportError:
        # Fallback to local path
        model_path = os.path.join(fallback_dir, filename)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, model_path)
        return model_path


def save_plot_with_store(
    plot_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save plot using artifact store if available.
    
    Args:
        plot_path: Path to existing plot file
        metadata: Optional metadata to attach
        
    Returns:
        Path where plot was saved
    """
    try:
        from storage import get_artifact_store
        store = get_artifact_store()
        return store.save_plot(plot_path, metadata=metadata)
    except ImportError:
        # Already saved locally
        return plot_path


def save_report_with_store(
    report_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save report using artifact store if available.
    
    Args:
        report_path: Path to existing report file
        metadata: Optional metadata to attach
        
    Returns:
        Path where report was saved
    """
    try:
        from storage import get_artifact_store
        store = get_artifact_store()
        return store.save_report(report_path, metadata=metadata)
    except ImportError:
        # Already saved locally
        return report_path


def save_data_with_store(
    data_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save data file using artifact store if available.
    
    Args:
        data_path: Path to existing data file
        metadata: Optional metadata to attach
        
    Returns:
        Path where data was saved
    """
    try:
        from storage import get_artifact_store
        store = get_artifact_store()
        return store.save_data(data_path, metadata=metadata)
    except ImportError:
        # Already saved locally
        return data_path
