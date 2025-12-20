"""Utils module initialization."""

from .polars_helpers import (
    load_dataframe,
    save_dataframe,
    get_numeric_columns,
    get_categorical_columns,
    get_datetime_columns,
    detect_id_columns,
    get_column_info,
    calculate_memory_usage,
    split_features_target,
)

from .validation import (
    ValidationError,
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
    validate_column_exists,
    validate_columns_exist,
    validate_target_column,
)

__all__ = [
    "load_dataframe",
    "save_dataframe",
    "get_numeric_columns",
    "get_categorical_columns",
    "get_datetime_columns",
    "detect_id_columns",
    "get_column_info",
    "calculate_memory_usage",
    "split_features_target",
    "ValidationError",
    "validate_file_exists",
    "validate_file_format",
    "validate_dataframe",
    "validate_column_exists",
    "validate_columns_exist",
    "validate_target_column",
]
