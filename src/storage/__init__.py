"""Storage abstraction for artifacts (models, plots, reports)."""

from .artifact_store import ArtifactStore, get_artifact_store, reset_artifact_store
from .helpers import (
    save_model_with_store,
    save_plot_with_store,
    save_report_with_store,
    save_data_with_store
)

__all__ = [
    "ArtifactStore",
    "get_artifact_store",
    "reset_artifact_store",
    "save_model_with_store",
    "save_plot_with_store",
    "save_report_with_store",
    "save_data_with_store"
]
