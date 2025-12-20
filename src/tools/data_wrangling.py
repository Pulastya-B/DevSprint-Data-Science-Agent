"""
Data Wrangling Tools
Tools for merging, concatenating, and manipulating multiple datasets.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import (
    load_dataframe,
    save_dataframe,
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
)


def merge_datasets(
    left_path: str,
    right_path: str,
    output_path: str,
    how: Literal["inner", "left", "right", "outer", "cross"] = "inner",
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    suffix: str = "_right"
) -> Dict[str, Any]:
    """
    Merge two datasets using various join strategies (SQL-like join operations).
    
    This function performs database-style joins on two datasets, similar to SQL JOIN operations.
    Supports inner, left, right, outer, and cross joins.
    
    Args:
        left_path: Path to left dataset (CSV or Parquet)
        right_path: Path to right dataset (CSV or Parquet)
        output_path: Path to save merged dataset
        how: Join type - "inner", "left", "right", "outer", or "cross"
            - "inner": Only rows with matching keys in both datasets
            - "left": All rows from left, matching rows from right (nulls if no match)
            - "right": All rows from right, matching rows from left (nulls if no match)
            - "outer": All rows from both (nulls where no match)
            - "cross": Cartesian product (all combinations)
        on: Column name to join on (if same in both datasets)
        left_on: Column name in left dataset (if different from right)
        right_on: Column name in right dataset (if different from left)
        suffix: Suffix to add to duplicate column names from right dataset (default: "_right")
    
    Returns:
        Dictionary with merge report including:
        - success: bool
        - output_path: str
        - left_rows: int
        - right_rows: int
        - result_rows: int
        - merge_type: str
        - join_columns: dict
        - duplicate_columns: list (columns that got suffixed)
    
    Examples:
        >>> # Simple join on same column name
        >>> merge_datasets(
        ...     "customers.csv", 
        ...     "orders.csv",
        ...     "merged.csv",
        ...     how="left",
        ...     on="customer_id"
        ... )
        
        >>> # Join on different column names
        >>> merge_datasets(
        ...     "products.csv",
        ...     "sales.csv",
        ...     "product_sales.csv",
        ...     how="inner",
        ...     left_on="product_id",
        ...     right_on="prod_id"
        ... )
    """
    try:
        # Validation
        validate_file_exists(left_path)
        validate_file_exists(right_path)
        validate_file_format(left_path)
        validate_file_format(right_path)
        
        # Load datasets
        left_df = load_dataframe(left_path)
        right_df = load_dataframe(right_path)
        
        validate_dataframe(left_df)
        validate_dataframe(right_df)
        
        left_rows = len(left_df)
        right_rows = len(right_df)
        
        # Determine join columns
        if on:
            # Same column name in both datasets
            join_left_on = on
            join_right_on = on
            
            # Validate column exists
            if on not in left_df.columns:
                return {
                    "success": False,
                    "error": f"Column '{on}' not found in left dataset. Available: {left_df.columns}"
                }
            if on not in right_df.columns:
                return {
                    "success": False,
                    "error": f"Column '{on}' not found in right dataset. Available: {right_df.columns}"
                }
        elif left_on and right_on:
            # Different column names
            join_left_on = left_on
            join_right_on = right_on
            
            # Validate columns exist
            if left_on not in left_df.columns:
                return {
                    "success": False,
                    "error": f"Column '{left_on}' not found in left dataset. Available: {left_df.columns}"
                }
            if right_on not in right_df.columns:
                return {
                    "success": False,
                    "error": f"Column '{right_on}' not found in right dataset. Available: {right_df.columns}"
                }
        else:
            return {
                "success": False,
                "error": "Must specify either 'on' (same column name) or both 'left_on' and 'right_on' (different names)"
            }
        
        # Check for duplicate column names (excluding join columns)
        left_cols = set(left_df.columns)
        right_cols = set(right_df.columns)
        duplicate_cols = list((left_cols & right_cols) - {join_left_on, join_right_on})
        
        # Perform merge
        merged_df = left_df.join(
            right_df,
            left_on=join_left_on,
            right_on=join_right_on,
            how=how,
            suffix=suffix
        )
        
        result_rows = len(merged_df)
        
        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(merged_df, output_path)
        
        # Build report
        report = {
            "success": True,
            "output_path": output_path,
            "left_file": Path(left_path).name,
            "right_file": Path(right_path).name,
            "left_rows": left_rows,
            "right_rows": right_rows,
            "result_rows": result_rows,
            "result_columns": len(merged_df.columns),
            "merge_type": how,
            "join_columns": {
                "left": join_left_on,
                "right": join_right_on
            },
            "duplicate_columns": duplicate_cols,
            "rows_added": result_rows - left_rows if how in ["left", "inner"] else None,
            "message": f"Successfully merged {left_rows:,} rows with {right_rows:,} rows using {how} join → {result_rows:,} rows"
        }
        
        # Add warnings
        if how == "inner" and result_rows < min(left_rows, right_rows):
            report["warning"] = f"Inner join reduced data: only {result_rows:,} of {min(left_rows, right_rows):,} rows had matches"
        elif how == "outer" and result_rows > left_rows + right_rows:
            report["warning"] = "Outer join created duplicate rows - check for many-to-many relationships"
        
        if duplicate_cols:
            report["note"] = f"{len(duplicate_cols)} column(s) were suffixed with '{suffix}': {', '.join(duplicate_cols)}"
        
        return report
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def concat_datasets(
    file_paths: List[str],
    output_path: str,
    axis: Literal["vertical", "horizontal"] = "vertical",
    ignore_index: bool = True
) -> Dict[str, Any]:
    """
    Concatenate multiple datasets vertically (stack rows) or horizontally (add columns).
    
    Args:
        file_paths: List of file paths to concatenate (CSV or Parquet)
        output_path: Path to save concatenated dataset
        axis: "vertical" to stack rows (union), "horizontal" to add columns side-by-side
        ignore_index: If True, reset index after concatenation (default: True)
    
    Returns:
        Dictionary with concatenation report including:
        - success: bool
        - output_path: str
        - input_files: int
        - result_rows: int
        - result_cols: int
        - axis: str
    
    Examples:
        >>> # Stack multiple CSV files (union)
        >>> concat_datasets(
        ...     ["jan_sales.csv", "feb_sales.csv", "mar_sales.csv"],
        ...     "q1_sales.csv",
        ...     axis="vertical"
        ... )
        
        >>> # Combine datasets side-by-side (add columns)
        >>> concat_datasets(
        ...     ["features.csv", "labels.csv"],
        ...     "full_dataset.csv",
        ...     axis="horizontal"
        ... )
    """
    try:
        # Validation
        if not file_paths or len(file_paths) < 2:
            return {
                "success": False,
                "error": "Must provide at least 2 files to concatenate"
            }
        
        for fp in file_paths:
            validate_file_exists(fp)
            validate_file_format(fp)
        
        # Load all datasets
        dfs = []
        file_info = []
        
        for fp in file_paths:
            df = load_dataframe(fp)
            validate_dataframe(df)
            dfs.append(df)
            file_info.append({
                "file": Path(fp).name,
                "rows": len(df),
                "columns": len(df.columns)
            })
        
        # Perform concatenation
        if axis == "vertical":
            # Stack rows (union) - requires same columns
            result = pl.concat(dfs, how="vertical")
        else:  # horizontal
            # Add columns side-by-side - requires same number of rows
            row_counts = [len(df) for df in dfs]
            if len(set(row_counts)) > 1:
                return {
                    "success": False,
                    "error": f"Horizontal concatenation requires same number of rows. Got: {row_counts}",
                    "file_info": file_info
                }
            result = pl.concat(dfs, how="horizontal")
        
        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(result, output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "input_files": len(file_paths),
            "file_info": file_info,
            "result_rows": len(result),
            "result_cols": len(result.columns),
            "axis": axis,
            "message": f"Successfully concatenated {len(file_paths)} files ({axis}) → {len(result):,} rows × {len(result.columns)} columns"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def reshape_dataset(
    file_path: str,
    output_path: str,
    operation: Literal["pivot", "melt", "transpose"],
    **kwargs
) -> Dict[str, Any]:
    """
    Reshape dataset using pivot, melt, or transpose operations.
    
    Args:
        file_path: Path to CSV or Parquet file
        output_path: Path to save reshaped dataset
        operation: "pivot" (wide format), "melt" (long format), or "transpose"
        **kwargs: Operation-specific parameters
            For pivot: index, columns, values, aggregate_function
            For melt: id_vars, value_vars, var_name, value_name
    
    Returns:
        Dictionary with reshape report
    
    Examples:
        >>> # Pivot: wide format
        >>> reshape_dataset(
        ...     "sales_long.csv",
        ...     "sales_wide.csv",
        ...     operation="pivot",
        ...     index="date",
        ...     columns="product",
        ...     values="sales"
        ... )
        
        >>> # Melt: long format
        >>> reshape_dataset(
        ...     "sales_wide.csv",
        ...     "sales_long.csv",
        ...     operation="melt",
        ...     id_vars=["date"],
        ...     value_vars=["product_a", "product_b"],
        ...     var_name="product",
        ...     value_name="sales"
        ... )
    """
    try:
        # Validation
        validate_file_exists(file_path)
        validate_file_format(file_path)
        
        # Load data
        df = load_dataframe(file_path)
        validate_dataframe(df)
        
        original_shape = (len(df), len(df.columns))
        
        # Perform operation
        if operation == "pivot":
            # Pivot to wide format
            index = kwargs.get("index")
            columns = kwargs.get("columns")
            values = kwargs.get("values")
            
            if not all([index, columns, values]):
                return {
                    "success": False,
                    "error": "Pivot requires: index, columns, values parameters"
                }
            
            result = df.pivot(
                index=index,
                columns=columns,
                values=values
            )
        
        elif operation == "melt":
            # Melt to long format
            id_vars = kwargs.get("id_vars")
            value_vars = kwargs.get("value_vars")
            var_name = kwargs.get("var_name", "variable")
            value_name = kwargs.get("value_name", "value")
            
            if not id_vars:
                return {
                    "success": False,
                    "error": "Melt requires: id_vars parameter"
                }
            
            result = df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                variable_name=var_name,
                value_name=value_name
            )
        
        elif operation == "transpose":
            # Transpose rows and columns
            result = df.transpose()
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Use 'pivot', 'melt', or 'transpose'"
            }
        
        # Save result
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_dataframe(result, output_path)
        
        return {
            "success": True,
            "output_path": output_path,
            "operation": operation,
            "original_shape": {
                "rows": original_shape[0],
                "columns": original_shape[1]
            },
            "result_shape": {
                "rows": len(result),
                "columns": len(result.columns)
            },
            "message": f"Successfully {operation}ed dataset: {original_shape[0]}×{original_shape[1]} → {len(result)}×{len(result.columns)}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
