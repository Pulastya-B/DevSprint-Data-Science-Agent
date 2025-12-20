"""
Feature Engineering Tools
Tools for creating new features from existing data.
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
    save_dataframe,
    get_numeric_columns,
    get_categorical_columns,
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
    validate_column_exists,
    validate_datetime_column,
)


def create_time_features(file_path: str, date_col: str, 
                        output_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive time-based features from datetime column.
    
    Args:
        file_path: Path to CSV or Parquet file
        date_col: Name of datetime column
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with feature engineering report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, date_col)
    
    # Try to parse datetime if it's a string
    if df[date_col].dtype == pl.Utf8:
        try:
            df = df.with_columns(
                pl.col(date_col).str.strptime(pl.Datetime, strict=False).alias(date_col)
            )
        except:
            return {
                "status": "error",
                "message": f"Could not parse column '{date_col}' as datetime"
            }
    
    # Validate it's now a datetime
    if df[date_col].dtype not in [pl.Date, pl.Datetime]:
        return {
            "status": "error",
            "message": f"Column '{date_col}' is not a datetime type (dtype: {df[date_col].dtype})"
        }
    
    features_created = []
    
    # Extract basic time features
    df = df.with_columns([
        pl.col(date_col).dt.year().alias(f"{date_col}_year"),
        pl.col(date_col).dt.month().alias(f"{date_col}_month"),
        pl.col(date_col).dt.day().alias(f"{date_col}_day"),
        pl.col(date_col).dt.weekday().alias(f"{date_col}_dayofweek"),
        pl.col(date_col).dt.quarter().alias(f"{date_col}_quarter"),
    ])
    
    features_created.extend([
        f"{date_col}_year",
        f"{date_col}_month",
        f"{date_col}_day",
        f"{date_col}_dayofweek",
        f"{date_col}_quarter"
    ])
    
    # Create is_weekend feature
    df = df.with_columns(
        (pl.col(f"{date_col}_dayofweek") >= 5).cast(pl.Int8).alias(f"{date_col}_is_weekend")
    )
    features_created.append(f"{date_col}_is_weekend")
    
    # Cyclical encoding for month (sin/cos)
    df = df.with_columns([
        (2 * np.pi * pl.col(f"{date_col}_month") / 12).sin().alias(f"{date_col}_month_sin"),
        (2 * np.pi * pl.col(f"{date_col}_month") / 12).cos().alias(f"{date_col}_month_cos"),
    ])
    features_created.extend([
        f"{date_col}_month_sin",
        f"{date_col}_month_cos"
    ])
    
    # If datetime has time component, extract hour
    if df[date_col].dtype == pl.Datetime:
        try:
            df = df.with_columns([
                pl.col(date_col).dt.hour().alias(f"{date_col}_hour"),
            ])
            features_created.append(f"{date_col}_hour")
            
            # Cyclical encoding for hour
            df = df.with_columns([
                (2 * np.pi * pl.col(f"{date_col}_hour") / 24).sin().alias(f"{date_col}_hour_sin"),
                (2 * np.pi * pl.col(f"{date_col}_hour") / 24).cos().alias(f"{date_col}_hour_cos"),
            ])
            features_created.extend([
                f"{date_col}_hour_sin",
                f"{date_col}_hour_cos"
            ])
        except:
            pass  # Hour extraction failed, skip
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, output_path)
    
    return {
        "status": "success",
        "features_created": features_created,
        "num_features": len(features_created),
        "output_path": output_path
    }


def encode_categorical(file_path: str, method: str = "auto", columns: Optional[List[str]] = None,
                      target_col: Optional[str] = None, 
                      output_path: str = None) -> Dict[str, Any]:
    """
    Encode categorical variables.
    
    Args:
        file_path: Path to CSV or Parquet file
        method: Encoding method ('one_hot', 'target', 'frequency', 'auto')
        columns: List of columns to encode, or ['all'] for all categorical. If None, defaults to all categorical columns
        target_col: Required for target encoding - name of target column
        output_path: Path to save dataset with encoded features
        
    Returns:
        Dictionary with encoding report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Determine which columns to process
    categorical_cols = get_categorical_columns(df)
    
    # Default to all categorical columns if not specified
    if columns is None or columns == ["all"]:
        target_cols = categorical_cols
    else:
        # Validate columns exist
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found")
        target_cols = columns
    
    # Auto-detect method if 'auto'
    if method == "auto":
        # Use frequency encoding for high-cardinality, one-hot for low
        method = "frequency"  # Default safe choice
    
    # For target encoding, validate target column
    if method == "target":
        if target_col is None:
            return {
                "status": "error",
                "message": "target_col is required for target encoding"
            }
        validate_column_exists(df, target_col)
    
    report = {
        "method": method,
        "columns_processed": {},
        "features_created": []
    }
    
    # Process each column
    for col in target_cols:
        if col not in df.columns:
            report["columns_processed"][col] = {
                "status": "error",
                "message": "Column not found"
            }
            continue
        
        n_unique = df[col].n_unique()
        
        try:
            if method == "one_hot":
                # One-hot encoding
                # Limit to top categories if too many
                if n_unique > 50:
                    report["columns_processed"][col] = {
                        "status": "warning",
                        "message": f"Column has {n_unique} unique values. Consider using frequency or target encoding instead."
                    }
                    continue
                
                # Get dummies
                encoded = df.select(pl.col(col)).to_dummies(columns=[col])
                
                # Add encoded columns to dataframe
                for enc_col in encoded.columns:
                    df = df.with_columns(encoded[enc_col])
                    report["features_created"].append(enc_col)
                
                # Drop original column
                df = df.drop(col)
                
                report["columns_processed"][col] = {
                    "status": "success",
                    "num_features_created": len(encoded.columns)
                }
            
            elif method == "frequency":
                # Frequency encoding
                value_counts = df[col].value_counts()
                freq_map = {
                    row[0]: row[1] / len(df)
                    for row in value_counts.iter_rows()
                }
                
                # Create new column with frequencies
                new_col_name = f"{col}_freq"
                df = df.with_columns(
                    pl.col(col).map_dict(freq_map, default=0.0).alias(new_col_name)
                )
                
                # Drop original column
                df = df.drop(col)
                
                report["features_created"].append(new_col_name)
                report["columns_processed"][col] = {
                    "status": "success",
                    "num_features_created": 1
                }
            
            elif method == "target":
                # Target encoding (mean encoding)
                # Calculate mean target value for each category
                target_means = (
                    df.group_by(col)
                    .agg(pl.col(target_col).mean().alias("target_mean"))
                )
                
                # Create dictionary for mapping
                target_map = {
                    row[0]: row[1]
                    for row in target_means.iter_rows()
                }
                
                # Global mean for unseen categories
                global_mean = df[target_col].mean()
                
                # Create new column with target encoding
                new_col_name = f"{col}_target_enc"
                df = df.with_columns(
                    pl.col(col).map_dict(target_map, default=global_mean).alias(new_col_name)
                )
                
                # Drop original column
                df = df.drop(col)
                
                report["features_created"].append(new_col_name)
                report["columns_processed"][col] = {
                    "status": "success",
                    "num_features_created": 1
                }
        
        except Exception as e:
            report["columns_processed"][col] = {
                "status": "error",
                "message": str(e)
            }
    
    report["total_features_created"] = len(report["features_created"])
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe(df, output_path)
    report["output_path"] = output_path
    
    return report
