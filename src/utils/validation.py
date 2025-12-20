"""
Validation utilities for data science operations.
"""

import polars as pl
from typing import List, Dict, Any, Optional
from pathlib import Path


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_file_exists(file_path: str) -> None:
    """
    Validate that a file exists.
    
    Args:
        file_path: Path to file
        
    Raises:
        ValidationError: If file doesn't exist
    """
    if not Path(file_path).exists():
        raise ValidationError(f"File not found: {file_path}")


def validate_file_format(file_path: str, allowed_formats: List[str] = None) -> None:
    """
    Validate file format.
    
    Args:
        file_path: Path to file
        allowed_formats: List of allowed extensions (default: ['.csv', '.parquet'])
        
    Raises:
        ValidationError: If file format is not supported
    """
    if allowed_formats is None:
        allowed_formats = ['.csv', '.parquet']
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in allowed_formats:
        raise ValidationError(
            f"Unsupported file format: {file_ext}. Allowed: {', '.join(allowed_formats)}"
        )


def validate_dataframe(df: pl.DataFrame) -> None:
    """
    Validate that dataframe is valid and not empty.
    
    Args:
        df: Polars DataFrame
        
    Raises:
        ValidationError: If dataframe is invalid or empty
    """
    if df is None:
        raise ValidationError("DataFrame is None")
    
    if len(df) == 0:
        raise ValidationError("DataFrame is empty (0 rows)")
    
    if len(df.columns) == 0:
        raise ValidationError("DataFrame has no columns")


def validate_column_exists(df: pl.DataFrame, column: str) -> None:
    """
    Validate that a column exists in dataframe.
    
    Args:
        df: Polars DataFrame
        column: Column name
        
    Raises:
        ValidationError: If column doesn't exist
    """
    if column not in df.columns:
        raise ValidationError(
            f"Column '{column}' not found. Available columns: {', '.join(df.columns)}"
        )


def validate_columns_exist(df: pl.DataFrame, columns: List[str]) -> None:
    """
    Validate that multiple columns exist in dataframe.
    
    Args:
        df: Polars DataFrame
        columns: List of column names
        
    Raises:
        ValidationError: If any column doesn't exist
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValidationError(
            f"Columns not found: {', '.join(missing)}. "
            f"Available: {', '.join(df.columns)}"
        )


def validate_numeric_column(df: pl.DataFrame, column: str) -> None:
    """
    Validate that a column is numeric.
    
    Args:
        df: Polars DataFrame
        column: Column name
        
    Raises:
        ValidationError: If column is not numeric
    """
    validate_column_exists(df, column)
    
    if df[column].dtype not in pl.NUMERIC_DTYPES:
        raise ValidationError(
            f"Column '{column}' is not numeric (dtype: {df[column].dtype})"
        )


def validate_categorical_column(df: pl.DataFrame, column: str) -> None:
    """
    Validate that a column is categorical.
    
    Args:
        df: Polars DataFrame
        column: Column name
        
    Raises:
        ValidationError: If column is not categorical
    """
    validate_column_exists(df, column)
    
    if df[column].dtype not in [pl.Utf8, pl.Categorical]:
        raise ValidationError(
            f"Column '{column}' is not categorical (dtype: {df[column].dtype})"
        )


def validate_datetime_column(df: pl.DataFrame, column: str) -> None:
    """
    Validate that a column is datetime.
    
    Args:
        df: Polars DataFrame
        column: Column name
        
    Raises:
        ValidationError: If column is not datetime
    """
    validate_column_exists(df, column)
    
    if df[column].dtype not in [pl.Date, pl.Datetime]:
        raise ValidationError(
            f"Column '{column}' is not datetime (dtype: {df[column].dtype})"
        )


def validate_target_column(df: pl.DataFrame, target_col: str, 
                          task_type: Optional[str] = None) -> str:
    """
    Validate target column and infer task type if not provided.
    
    Args:
        df: Polars DataFrame
        target_col: Target column name
        task_type: Optional task type ('classification' or 'regression')
        
    Returns:
        Inferred or validated task type
        
    Raises:
        ValidationError: If target column is invalid
    """
    validate_column_exists(df, target_col)
    
    target = df[target_col]
    n_unique = target.n_unique()
    
    # Infer task type if not provided
    if task_type is None:
        if target.dtype in pl.NUMERIC_DTYPES and n_unique > 10:
            task_type = "regression"
        else:
            task_type = "classification"
    
    # Validate task type
    if task_type not in ["classification", "regression"]:
        raise ValidationError(
            f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'"
        )
    
    # Validate target column matches task type
    if task_type == "classification":
        if n_unique > 100:
            raise ValidationError(
                f"Classification target has too many unique values ({n_unique}). "
                f"Consider regression or check if this is the correct target."
            )
    
    if task_type == "regression":
        if target.dtype not in pl.NUMERIC_DTYPES:
            raise ValidationError(
                f"Regression target must be numeric (dtype: {target.dtype})"
            )
    
    return task_type


def validate_train_test_split(X_train: Any, X_test: Any, 
                               y_train: Any, y_test: Any) -> None:
    """
    Validate train/test split data.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Raises:
        ValidationError: If split data is invalid
    """
    if len(X_train) == 0:
        raise ValidationError("X_train is empty")
    
    if len(X_test) == 0:
        raise ValidationError("X_test is empty")
    
    if len(y_train) == 0:
        raise ValidationError("y_train is empty")
    
    if len(y_test) == 0:
        raise ValidationError("y_test is empty")
    
    if len(X_train) != len(y_train):
        raise ValidationError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) have different lengths"
        )
    
    if len(X_test) != len(y_test):
        raise ValidationError(
            f"X_test ({len(X_test)}) and y_test ({len(y_test)}) have different lengths"
        )


def validate_strategy_config(strategy: Dict[str, Any], 
                             required_keys: List[str]) -> None:
    """
    Validate strategy configuration dictionary.
    
    Args:
        strategy: Strategy configuration
        required_keys: List of required keys
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(strategy, dict):
        raise ValidationError(f"Strategy must be a dictionary, got {type(strategy)}")
    
    missing = [key for key in required_keys if key not in strategy]
    if missing:
        raise ValidationError(
            f"Missing required strategy keys: {', '.join(missing)}"
        )
