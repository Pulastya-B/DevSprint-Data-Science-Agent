"""
Advanced data type conversion tools for handling tricky type issues.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import (
    load_dataframe,
    save_dataframe,
    get_numeric_columns,
    get_categorical_columns
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe
)


def force_numeric_conversion(
    file_path: str,
    columns: List[str],
    output_path: str,
    errors: str = "coerce"
) -> Dict[str, Any]:
    """
    Force convert columns to numeric type, even if they're detected as strings/objects.
    
    This is crucial for datasets where numeric columns are stored as strings with 
    formatting issues (commas, spaces, currency symbols, etc.).
    
    Args:
        file_path: Path to CSV or Parquet file
        columns: List of column names to force convert, or ["all"] for all non-ID columns
        output_path: Path to save converted dataset
        errors: How to handle conversion errors:
               - "coerce": Invalid values become null (default)
               - "raise": Raise error on invalid values
               
    Returns:
        Dictionary with conversion report and statistics
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    original_types = {col: str(df[col].dtype) for col in df.columns}
    
    # Determine which columns to convert
    if columns == ["all"]:
        # Auto-detect: skip ID columns, already-numeric columns, and actual text columns
        id_keywords = ['id', 'key', 'code', 'name', 'description', 'text', 'comment', 'notes']
        target_columns = []
        
        for col in df.columns:
            # Skip if already numeric
            if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                continue
            
            # Skip if looks like an ID or text column
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in id_keywords):
                continue
            
            # Only attempt conversion if column looks numeric
            # Sample first 100 non-null values to check if they're numeric-like
            sample_values = df[col].drop_nulls().head(100).to_list()
            if len(sample_values) == 0:
                continue
            
            numeric_like_count = 0
            for val in sample_values[:min(50, len(sample_values))]:  # Check first 50 samples
                val_str = str(val).replace(",", "").replace(" ", "").replace("$", "").replace("€", "").strip()
                
                # Check if it looks like a number (digits, decimal point, minus sign)
                if val_str.replace(".", "").replace("-", "").replace("+", "").replace("e", "").replace("E", "").isdigit():
                    numeric_like_count += 1
                # Also check for percentage-like values
                elif val_str.endswith("%") and val_str[:-1].replace(".", "").isdigit():
                    numeric_like_count += 1
            
            # Only include if >70% of samples look numeric
            if len(sample_values) > 0 and (numeric_like_count / min(50, len(sample_values))) > 0.7:
                target_columns.append(col)
                print(f"🔍 '{col}': Detected as numeric-like ({numeric_like_count}/{min(50, len(sample_values))} samples)")
            else:
                print(f"⏭️ '{col}': Skipping (appears to be text, not numeric)")
    else:
        target_columns = columns
    
    print(f"🔢 Force converting {len(target_columns)} columns to numeric...")
    
    # Track conversion results
    conversion_report = {
        "successful_conversions": [],
        "failed_conversions": [],
        "null_values_introduced": {}
    }
    
    # Convert each column
    for col in target_columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not found, skipping")
            conversion_report["failed_conversions"].append(col)
            continue
        
        try:
            # Get original null count
            original_nulls = df[col].null_count()
            
            # Try to convert to numeric
            # First, clean the column if it's a string (remove commas, spaces, etc.)
            if df[col].dtype == pl.Utf8:
                # Remove common non-numeric characters
                df = df.with_columns([
                    pl.col(col)
                    .str.replace_all(",", "")  # Remove commas
                    .str.replace_all(" ", "")  # Remove spaces
                    .str.replace_all("$", "")  # Remove dollar signs
                    .str.replace_all("€", "")  # Remove euro signs
                    .str.replace_all("%", "")  # Remove percent signs
                    .str.strip_chars()  # Strip whitespace
                    .alias(col)
                ])
            
            # Now convert to float
            if errors == "coerce":
                df = df.with_columns([
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                ])
            else:
                df = df.with_columns([
                    pl.col(col).cast(pl.Float64, strict=True).alias(col)
                ])
            
            # Check how many nulls were introduced
            new_nulls = df[col].null_count()
            nulls_introduced = new_nulls - original_nulls
            
            conversion_report["successful_conversions"].append(col)
            conversion_report["null_values_introduced"][col] = int(nulls_introduced)
            
            if nulls_introduced > 0:
                print(f"✅ '{col}': Converted to numeric ({nulls_introduced} values became null)")
            else:
                print(f"✅ '{col}': Converted to numeric (no data loss)")
                
        except Exception as e:
            print(f"❌ '{col}': Conversion failed - {str(e)}")
            conversion_report["failed_conversions"].append(col)
    
    # Save converted dataset
    save_dataframe(df, output_path)
    
    new_types = {col: str(df[col].dtype) for col in df.columns}
    
    return {
        "status": "success",
        "message": f"Force converted {len(conversion_report['successful_conversions'])} columns to numeric",
        "output_path": output_path,
        "conversion_report": conversion_report,
        "type_changes": {
            col: {"from": original_types[col], "to": new_types[col]}
            for col in conversion_report["successful_conversions"]
        },
        "total_successful": len(conversion_report["successful_conversions"]),
        "total_failed": len(conversion_report["failed_conversions"]),
        "total_nulls_introduced": sum(conversion_report["null_values_introduced"].values())
    }


def smart_type_inference(
    file_path: str,
    output_path: str,
    aggressive: bool = True
) -> Dict[str, Any]:
    """
    Intelligently infer and fix data types for all columns.
    
    This tool goes beyond basic type detection and tries to understand the
    semantic meaning of each column to assign the correct type.
    
    Args:
        file_path: Path to CSV or Parquet file
        output_path: Path to save dataset with fixed types
        aggressive: If True, attempts aggressive type conversion (force numeric on ambiguous columns)
        
    Returns:
        Dictionary with type inference report
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    original_types = {col: str(df[col].dtype) for col in df.columns}
    type_changes = {}
    
    print(f"🧠 Performing smart type inference on {len(df.columns)} columns...")
    
    for col in df.columns:
        current_type = df[col].dtype
        
        # Skip if already numeric
        if current_type in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            continue
        
        # If it's a string column, try to infer the correct type
        if current_type == pl.Utf8:
            sample_values = df[col].drop_nulls().head(100).to_list()
            
            if len(sample_values) == 0:
                continue
            
            # Try to detect if it's actually numeric
            numeric_count = 0
            for val in sample_values:
                # Clean and test
                cleaned = str(val).replace(",", "").replace(" ", "").replace("$", "").strip()
                try:
                    float(cleaned)
                    numeric_count += 1
                except:
                    pass
            
            # If >80% of values are numeric, convert to numeric
            if numeric_count / len(sample_values) > 0.8:
                print(f"🔢 '{col}': Detected as numeric ({numeric_count}/{len(sample_values)} samples)")
                
                # Clean and convert
                df = df.with_columns([
                    pl.col(col)
                    .str.replace_all(",", "")
                    .str.replace_all(" ", "")
                    .str.replace_all("$", "")
                    .str.replace_all("€", "")
                    .str.strip_chars()
                    .cast(pl.Float64, strict=False)
                    .alias(col)
                ])
                
                type_changes[col] = {"from": "Utf8", "to": "Float64", "reason": "numeric_pattern_detected"}
    
    # Save dataset
    save_dataframe(df, output_path)
    
    return {
        "status": "success",
        "message": f"Smart type inference completed, changed {len(type_changes)} columns",
        "output_path": output_path,
        "type_changes": type_changes,
        "original_types": original_types,
        "new_types": {col: str(df[col].dtype) for col in df.columns}
    }
