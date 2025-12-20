"""
Polars utility functions for data manipulation.
"""

import polars as pl
from typing import List, Dict, Any, Optional


def load_dataframe(file_path: str) -> pl.DataFrame:
    """
    Load a dataframe from CSV or Parquet file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Polars DataFrame
    """
    if file_path.endswith('.parquet'):
        return pl.read_parquet(file_path)
    elif file_path.endswith('.csv'):
        # Use longer schema inference to handle mixed types better
        # and ignore errors to handle problematic rows gracefully
        return pl.read_csv(
            file_path, 
            try_parse_dates=True,
            infer_schema_length=10000,  # Scan more rows for better type inference
            ignore_errors=True  # Skip problematic rows instead of failing
        )
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def save_dataframe(df: pl.DataFrame, file_path: str) -> None:
    """
    Save dataframe to CSV or Parquet file.
    
    Args:
        df: Polars DataFrame
        file_path: Output path
    """
    if file_path.endswith('.parquet'):
        df.write_parquet(file_path)
    elif file_path.endswith('.csv'):
        df.write_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def get_numeric_columns(df: pl.DataFrame) -> List[str]:
    """
    Get list of numeric column names.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        List of numeric column names
    """
    return [col for col in df.columns if df[col].dtype in pl.NUMERIC_DTYPES]


def get_categorical_columns(df: pl.DataFrame) -> List[str]:
    """
    Get list of categorical/string column names.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        List of categorical column names
    """
    return [col for col in df.columns if df[col].dtype in [pl.Utf8, pl.Categorical]]


def get_datetime_columns(df: pl.DataFrame) -> List[str]:
    """
    Get list of datetime column names.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        List of datetime column names
    """
    return [col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]]


def detect_id_columns(df: pl.DataFrame) -> List[str]:
    """
    Detect columns that are likely IDs (unique values, low information content).
    
    Args:
        df: Polars DataFrame
        
    Returns:
        List of likely ID column names
    """
    id_columns = []
    
    for col in df.columns:
        # Check if column name suggests it's an ID
        col_lower = col.lower()
        if any(id_term in col_lower for id_term in ['id', '_id', 'key', 'index']):
            id_columns.append(col)
            continue
        
        # Check if column has mostly unique values (>95% unique)
        n_unique = df[col].n_unique()
        n_total = len(df)
        if n_total > 0 and (n_unique / n_total) > 0.95:
            id_columns.append(col)
    
    return id_columns


def safe_cast_numeric(df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
    """
    Safely cast columns to numeric, handling errors gracefully.
    
    Args:
        df: Polars DataFrame
        columns: List of columns to cast
        
    Returns:
        DataFrame with columns cast to numeric where possible
    """
    result = df.clone()
    
    for col in columns:
        try:
            result = result.with_columns(
                pl.col(col).cast(pl.Float64).alias(col)
            )
        except Exception:
            # If casting fails, keep original column
            pass
    
    return result


def get_column_info(df: pl.DataFrame, col: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a column.
    
    Args:
        df: Polars DataFrame
        col: Column name
        
    Returns:
        Dictionary with column statistics
    """
    col_data = df[col]
    
    info = {
        "name": col,
        "dtype": str(col_data.dtype),
        "null_count": col_data.null_count(),
        "null_percentage": round(col_data.null_count() / len(df) * 100, 2),
        "unique_count": col_data.n_unique(),
        "unique_percentage": round(col_data.n_unique() / len(df) * 100, 2),
    }
    
    # Add numeric-specific stats
    if col_data.dtype in pl.NUMERIC_DTYPES:
        info.update({
            "mean": float(col_data.mean()) if col_data.mean() is not None else None,
            "std": float(col_data.std()) if col_data.std() is not None else None,
            "min": float(col_data.min()) if col_data.min() is not None else None,
            "max": float(col_data.max()) if col_data.max() is not None else None,
            "median": float(col_data.median()) if col_data.median() is not None else None,
        })
    
    # Add categorical-specific stats
    if col_data.dtype in [pl.Utf8, pl.Categorical]:
        value_counts = col_data.value_counts().limit(5)
        info["top_values"] = [
            {"value": str(row[0]), "count": int(row[1])}
            for row in value_counts.iter_rows()
        ]
    
    return info


def calculate_memory_usage(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Calculate memory usage of dataframe.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    total_bytes = df.estimated_size()
    
    return {
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "total_bytes": total_bytes,
        "rows": len(df),
        "columns": len(df.columns),
        "bytes_per_row": round(total_bytes / len(df), 2) if len(df) > 0 else 0,
    }


def split_features_target(df: pl.DataFrame, target_col: str) -> tuple:
    """
    Split dataframe into features and target.
    
    Args:
        df: Polars DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(target_col)
    y = df[target_col]
    
    return X, y


def remove_low_variance_features(df: pl.DataFrame, threshold: float = 0.01) -> pl.DataFrame:
    """
    Remove features with low variance.
    
    Args:
        df: Polars DataFrame
        threshold: Variance threshold (default 0.01)
        
    Returns:
        DataFrame with low variance columns removed
    """
    numeric_cols = get_numeric_columns(df)
    
    cols_to_keep = []
    for col in numeric_cols:
        variance = df[col].var()
        if variance is not None and variance > threshold:
            cols_to_keep.append(col)
    
    # Keep non-numeric columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    
    return df.select(cols_to_keep + non_numeric_cols)
