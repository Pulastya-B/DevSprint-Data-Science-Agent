"""
Data Profiling Tools
Tools for analyzing and understanding dataset characteristics.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import (
    load_dataframe,
    get_numeric_columns,
    get_categorical_columns,
    get_datetime_columns,
    get_column_info,
    calculate_memory_usage,
    detect_id_columns,
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
)


def profile_dataset(file_path: str) -> Dict[str, Any]:
    """
    Get comprehensive statistics about a dataset.
    
    Args:
        file_path: Path to CSV or Parquet file
        
    Returns:
        Dictionary with dataset profile including:
        - shape (rows, columns)
        - column types
        - memory usage
        - null counts
        - unique values
        - missing value percentage per column (NEW)
        - unique value counts per column (NEW)
        - basic statistics for each column
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Basic info
    profile = {
        "file_path": file_path,
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "memory_usage": calculate_memory_usage(df),
        "column_types": {
            "numeric": get_numeric_columns(df),
            "categorical": get_categorical_columns(df),
            "datetime": get_datetime_columns(df),
            "id_columns": detect_id_columns(df),
        },
        "columns": {},
        "missing_values_per_column": {},  # NEW: Per-column missing %
        "unique_counts_per_column": {}   # NEW: Per-column unique counts
    }
    
    # Per-column statistics with enhanced missing % and unique counts
    for col in df.columns:
        # Get existing column info
        profile["columns"][col] = get_column_info(df, col)
        
        # NEW: Calculate missing value percentage for this column
        null_count = df[col].null_count()
        missing_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0
        profile["missing_values_per_column"][col] = {
            "count": int(null_count),
            "percentage": missing_pct
        }
        
        # NEW: Calculate unique value counts (with dict handling)
        try:
            # Try to get unique count directly
            unique_count = df[col].n_unique()
            profile["unique_counts_per_column"][col] = int(unique_count)
        except Exception as e:
            # If column contains unhashable types (dicts, lists), handle gracefully
            try:
                # Convert to string and then count unique
                unique_count = df[col].cast(pl.Utf8).n_unique()
                profile["unique_counts_per_column"][col] = int(unique_count)
            except:
                profile["unique_counts_per_column"][col] = "N/A (unhashable type)"
    
    # Overall statistics
    total_nulls = sum(df[col].null_count() for col in df.columns)
    total_cells = len(df) * len(df.columns)
    
    profile["overall_stats"] = {
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "null_percentage": round(total_nulls / total_cells * 100, 2) if total_cells > 0 else 0,
        "duplicate_rows": df.is_duplicated().sum(),
        "duplicate_percentage": round(df.is_duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
    }
    
    return profile


def get_smart_summary(file_path: str, n_samples: int = 30) -> Dict[str, Any]:
    """
    Enhanced data summary with missing %, unique counts, and safe dict handling.
    
    This function provides a smarter, more LLM-friendly summary compared to profile_dataset().
    It includes per-column missing percentages, unique value counts, and handles
    dictionary columns gracefully (converts to strings to avoid hashing errors).
    
    Args:
        file_path: Path to CSV or Parquet file
        n_samples: Number of sample rows to include (default: 30)
    
    Returns:
        Dictionary with comprehensive smart summary including:
        - Basic shape info
        - Column data types
        - Missing value percentage by column (sorted by % descending)
        - Unique value counts by column
        - First N sample rows
        - Descriptive statistics for numeric columns
        - Safe handling of dictionary/unhashable columns
    
    Example:
        >>> summary = get_smart_summary("data.csv")
        >>> print(summary["missing_summary"])
        >>> # Output: [("col_A", 45.2), ("col_B", 12.3), ...]
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Convert dictionary-type columns to strings (prevents unhashable dict errors)
    for col in df.columns:
        try:
            # Try to detect if column might contain dicts/lists
            sample = df[col].drop_nulls().head(5)
            if len(sample) > 0:
                first_val = sample[0]
                # Check if it's a complex type
                if isinstance(first_val, (dict, list)):
                    df = df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
        except:
            # If any error, just continue
            pass
    
    # Calculate missing value statistics (sorted by % descending)
    missing_stats = []
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0
        missing_stats.append({
            "column": col,
            "count": int(null_count),
            "percentage": null_pct
        })
    
    # Sort by percentage descending
    missing_stats.sort(key=lambda x: x["percentage"], reverse=True)
    
    # Calculate unique value counts
    unique_counts = {}
    for col in df.columns:
        try:
            unique_count = df[col].n_unique()
            unique_counts[col] = int(unique_count)
        except:
            # Fallback for unhashable types
            try:
                unique_count = df[col].cast(pl.Utf8).n_unique()
                unique_counts[col] = int(unique_count)
            except:
                unique_counts[col] = "N/A"
    
    # Get column data types
    column_types = {col: str(df[col].dtype) for col in df.columns}
    
    # Get sample rows (first n_samples)
    sample_data = df.head(n_samples).to_dicts()
    
    # Get descriptive statistics for numeric columns
    numeric_cols = get_numeric_columns(df)
    numeric_stats = {}
    
    if numeric_cols:
        df_numeric = df.select(numeric_cols)
        # Convert to pandas for describe() functionality
        df_pd = df_numeric.to_pandas()
        stats_df = df_pd.describe()
        numeric_stats = stats_df.to_dict()
    
    # Build comprehensive summary
    summary = {
        "file_path": file_path,
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "column_types": column_types,
        "missing_summary": missing_stats,  # Sorted by % descending
        "unique_counts": unique_counts,
        "sample_data": sample_data,
        "numeric_statistics": numeric_stats,
        "memory_usage_mb": calculate_memory_usage(df),
        "summary_notes": []
    }
    
    # Add helpful notes for LLM
    high_missing_cols = [item for item in missing_stats if item["percentage"] > 40]
    if high_missing_cols:
        summary["summary_notes"].append(
            f"{len(high_missing_cols)} column(s) have >40% missing values (consider dropping)"
        )
    
    high_cardinality_cols = [col for col, count in unique_counts.items() 
                            if isinstance(count, int) and count > len(df) * 0.5]
    if high_cardinality_cols:
        summary["summary_notes"].append(
            f"{len(high_cardinality_cols)} column(s) have very high cardinality (>50% unique values)"
        )
    
    return summary


def detect_data_quality_issues(file_path: str) -> Dict[str, Any]:
    """
    Detect data quality issues in the dataset.
    
    Args:
        file_path: Path to CSV or Parquet file
        
    Returns:
        Dictionary with detected issues organized by severity:
        - critical: Issues that will break model training
        - warning: Issues that may affect model performance
        - info: Observations that may be relevant
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    issues = {
        "critical": [],
        "warning": [],
        "info": []
    }
    
    # Check for completely null columns
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = (null_count / len(df)) * 100
        
        if null_count == len(df):
            issues["critical"].append({
                "type": "all_null_column",
                "column": col,
                "message": f"Column '{col}' has all null values"
            })
        elif null_pct > 50:
            issues["warning"].append({
                "type": "high_null_percentage",
                "column": col,
                "null_percentage": round(null_pct, 2),
                "message": f"Column '{col}' has {round(null_pct, 2)}% null values"
            })
        elif null_pct > 10:
            issues["info"].append({
                "type": "moderate_null_percentage",
                "column": col,
                "null_percentage": round(null_pct, 2),
                "message": f"Column '{col}' has {round(null_pct, 2)}% null values"
            })
    
    # Check for duplicate rows
    dup_count = df.is_duplicated().sum()
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        severity = "warning" if dup_pct > 10 else "info"
        issues[severity].append({
            "type": "duplicate_rows",
            "count": int(dup_count),
            "percentage": round(dup_pct, 2),
            "message": f"Dataset has {dup_count} duplicate rows ({round(dup_pct, 2)}%)"
        })
    
    # Check for outliers in numeric columns using IQR method
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        col_data = df[col].drop_nulls()
        if len(col_data) == 0:
            continue
        
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
        
        if outliers > 0:
            outlier_pct = (outliers / len(col_data)) * 100
            if outlier_pct > 10:
                issues["warning"].append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "percentage": round(outlier_pct, 2),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                    "message": f"Column '{col}' has {outliers} outliers ({round(outlier_pct, 2)}%)"
                })
            elif outlier_pct > 1:
                issues["info"].append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "percentage": round(outlier_pct, 2),
                    "bounds": {"lower": float(lower_bound), "upper": float(upper_bound)},
                    "message": f"Column '{col}' has {outliers} outliers ({round(outlier_pct, 2)}%)"
                })
    
    # Check for high cardinality in categorical columns
    categorical_cols = get_categorical_columns(df)
    for col in categorical_cols:
        n_unique = df[col].n_unique()
        cardinality_pct = (n_unique / len(df)) * 100
        
        if n_unique > 100 and cardinality_pct > 50:
            issues["warning"].append({
                "type": "high_cardinality",
                "column": col,
                "unique_values": int(n_unique),
                "percentage": round(cardinality_pct, 2),
                "message": f"Column '{col}' has very high cardinality ({n_unique} unique values, {round(cardinality_pct, 2)}%)"
            })
    
    # Check for constant columns (single unique value)
    for col in df.columns:
        n_unique = df[col].n_unique()
        if n_unique == 1:
            issues["warning"].append({
                "type": "constant_column",
                "column": col,
                "message": f"Column '{col}' has only one unique value (constant)"
            })
    
    # Check for imbalanced datasets (for potential target columns)
    for col in df.columns:
        col_data = df[col]
        n_unique = col_data.n_unique()
        
        # Check if this could be a target column (2-20 unique values)
        if 2 <= n_unique <= 20:
            value_counts = col_data.value_counts()
            if len(value_counts) >= 2:
                max_count = value_counts[value_counts.columns[1]][0]
                max_pct = (max_count / len(df)) * 100
                
                if max_pct > 90:
                    issues["warning"].append({
                        "type": "class_imbalance",
                        "column": col,
                        "dominant_class_percentage": round(max_pct, 2),
                        "message": f"Column '{col}' may be imbalanced (dominant class: {round(max_pct, 2)}%)"
                    })
    
    # Summary
    issues["summary"] = {
        "total_issues": len(issues["critical"]) + len(issues["warning"]) + len(issues["info"]),
        "critical_count": len(issues["critical"]),
        "warning_count": len(issues["warning"]),
        "info_count": len(issues["info"])
    }
    
    return issues


def analyze_correlations(file_path: str, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze correlations between features.
    
    Args:
        file_path: Path to CSV or Parquet file
        target: Optional target column to analyze correlations with
        
    Returns:
        Dictionary with correlation analysis including:
        - correlation matrix (for numeric columns)
        - top correlations with target (if specified)
        - highly correlated feature pairs
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        return {
            "error": "Dataset must have at least 2 numeric columns for correlation analysis",
            "numeric_columns_found": len(numeric_cols)
        }
    
    # Select only numeric columns for correlation
    df_numeric = df.select(numeric_cols)
    
    # Calculate correlation matrix using pandas (Polars doesn't have native corr yet)
    df_pd = df_numeric.to_pandas()
    corr_matrix = df_pd.corr()
    
    result = {
        "numeric_columns": numeric_cols,
        "correlation_matrix": corr_matrix.to_dict()
    }
    
    # Find highly correlated pairs (excluding diagonal)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if abs(corr_value) > 0.7:  # High correlation threshold
                high_corr_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": round(float(corr_value), 4)
                })
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    result["high_correlations"] = high_corr_pairs
    
    # If target specified, show top correlations with target
    if target:
        if target not in df.columns:
            result["target_correlations_error"] = f"Target column '{target}' not found"
        elif target not in numeric_cols:
            result["target_correlations_error"] = f"Target column '{target}' is not numeric"
        else:
            target_corrs = []
            for col in numeric_cols:
                if col != target:
                    corr_value = corr_matrix.loc[target, col]
                    target_corrs.append({
                        "feature": col,
                        "correlation": round(float(corr_value), 4)
                    })
            
            # Sort by absolute correlation
            target_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            result["target_correlations"] = {
                "target": target,
                "top_features": target_corrs[:20]  # Top 20
            }
    
    return result
