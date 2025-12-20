"""
Enhanced Feature Engineering - Additional robust features
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import load_dataframe, save_dataframe, get_numeric_columns
from utils.validation import validate_file_exists, validate_dataframe


def create_ratio_features(file_path: str, 
                          columns: Optional[List[str]] = None,
                          max_ratios: int = 20,
                          output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create ratio features (a/b) for all numeric column pairs.
    ROBUST: Handles division by zero, infinity, and NaN values.
    
    Args:
        file_path: Path to dataset
        columns: Columns to use (None = all numeric)
        max_ratios: Maximum number of ratio features
        output_path: Output file path
        
    Returns:
        Dictionary with results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if columns is None:
        columns = get_numeric_columns(df)
    
    print(f"🔢 Creating ratio features from {len(columns)} columns...")
    
    ratio_exprs = []
    feature_names = []
    
    for i, col1 in enumerate(columns[:15]):
        for col2 in columns[i+1:16]:
            if len(ratio_exprs) >= max_ratios:
                break
            
            # Safe division (avoid div by zero, replace inf/nan)
            ratio_name = f"ratio_{col1}_div_{col2}"
            ratio_expr = (
                pl.when(pl.col(col2).abs() < 1e-10)
                .then(0)
                .otherwise(pl.col(col1) / pl.col(col2))
                .clip(-1e6, 1e6)  # Clip extreme values
                .fill_nan(0)
                .fill_null(0)
                .alias(ratio_name)
            )
            ratio_exprs.append(ratio_expr)
            feature_names.append(ratio_name)
    
    df = df.with_columns(ratio_exprs)
    
    if output_path:
        save_dataframe(df, output_path)
    
    return {
        'success': True,
        'tool': 'create_ratio_features',
        'result': {
            'new_features': len(ratio_exprs),
            'feature_names': feature_names,
            'output_path': output_path
        }
    }


def create_statistical_features(file_path: str,
                                columns: Optional[List[str]] = None,
                                output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create row-wise statistical features (mean, std, min, max, range).
    ROBUST: Handles missing values and edge cases.
    
    Args:
        file_path: Path to dataset
        columns: Columns to use (None = all numeric)
        output_path: Output file path
        
    Returns:
        Dictionary with results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if columns is None:
        columns = get_numeric_columns(df)
    
    print(f"📊 Creating statistical features across {len(columns)} columns...")
    
    # Row-wise statistics
    stat_features = [
        pl.concat_list([pl.col(c) for c in columns]).list.mean().fill_null(0).alias('row_mean'),
        pl.concat_list([pl.col(c) for c in columns]).list.std().fill_null(0).alias('row_std'),
        pl.concat_list([pl.col(c) for c in columns]).list.min().fill_null(0).alias('row_min'),
        pl.concat_list([pl.col(c) for c in columns]).list.max().fill_null(0).alias('row_max'),
        (pl.concat_list([pl.col(c) for c in columns]).list.max() - 
         pl.concat_list([pl.col(c) for c in columns]).list.min()).fill_null(0).alias('row_range'),
        pl.concat_list([pl.col(c) for c in columns]).list.sum().fill_null(0).alias('row_sum'),
    ]
    
    df = df.with_columns(stat_features)
    
    if output_path:
        save_dataframe(df, output_path)
    
    return {
        'success': True,
        'tool': 'create_statistical_features',
        'result': {
            'new_features': 6,
            'feature_names': ['row_mean', 'row_std', 'row_min', 'row_max', 'row_range', 'row_sum'],
            'output_path': output_path
        }
    }


def create_log_features(file_path: str,
                       columns: Optional[List[str]] = None,
                       output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create log-transformed features for skewed distributions.
    ROBUST: Handles negative values and zeros.
    
    Args:
        file_path: Path to dataset
        columns: Columns to use (None = all numeric with positive values)
        output_path: Output file path
        
    Returns:
        Dictionary with results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if columns is None:
        columns = get_numeric_columns(df)
    
    print(f"📈 Creating log-transformed features for {len(columns)} columns...")
    
    log_exprs = []
    feature_names = []
    
    for col in columns:
        # Check if column has positive values
        min_val = df[col].min()
        if min_val is not None and min_val > 0:
            # log(x)
            log_exprs.append(pl.col(col).log().fill_nan(0).alias(f"log_{col}"))
            feature_names.append(f"log_{col}")
        elif min_val is not None and min_val >= 0:
            # log(x+1) for non-negative values
            log_exprs.append((pl.col(col) + 1).log().fill_nan(0).alias(f"log1p_{col}"))
            feature_names.append(f"log1p_{col}")
    
    if log_exprs:
        df = df.with_columns(log_exprs)
    
    if output_path:
        save_dataframe(df, output_path)
    
    return {
        'success': True,
        'tool': 'create_log_features',
        'result': {
            'new_features': len(log_exprs),
            'feature_names': feature_names,
            'output_path': output_path
        }
    }


def create_binned_features(file_path: str,
                          columns: Optional[List[str]] = None,
                          n_bins: int = 5,
                          output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create binned (discretized) features from continuous variables.
    ROBUST: Uses quantile-based binning to handle outliers.
    
    Args:
        file_path: Path to dataset
        columns: Columns to use (None = all numeric)
        n_bins: Number of bins
        output_path: Output file path
        
    Returns:
        Dictionary with results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if columns is None:
        columns = get_numeric_columns(df)[:10]  # Limit to 10 columns
    
    print(f"🗂️  Creating binned features for {len(columns)} columns with {n_bins} bins...")
    
    binned_exprs = []
    feature_names = []
    
    for col in columns:
        # Quantile-based binning
        bin_name = f"{col}_binned"
        binned_exprs.append(
            pl.col(col).qcut(n_bins, labels=[f"bin_{i}" for i in range(n_bins)], 
                            allow_duplicates=True).fill_null("bin_0").alias(bin_name)
        )
        feature_names.append(bin_name)
    
    df = df.with_columns(binned_exprs)
    
    if output_path:
        save_dataframe(df, output_path)
    
    return {
        'success': True,
        'tool': 'create_binned_features',
        'result': {
            'new_features': len(binned_exprs),
            'feature_names': feature_names,
            'n_bins': n_bins,
            'output_path': output_path
        }
    }
