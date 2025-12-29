"""
Local Schema Extraction (No LLM)
Fast, cheap extraction of dataset metadata without sending to LLM.
"""

import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional


def extract_schema_local(file_path: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Extract dataset schema and basic stats locally without LLM.
    
    Returns:
        - column names and types
        - row/column counts
        - missing value counts
        - small sample for reference
        - memory usage
    """
    try:
        # Read with Polars (faster than pandas)
        if file_path.endswith('.csv'):
            df = pl.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pl.read_parquet(file_path)
        else:
            # Fallback to pandas
            import pandas as pd
            pdf = pd.read_csv(file_path)
            df = pl.from_pandas(pdf)
        
        # Basic metadata
        schema_info = {
            'file_path': file_path,
            'file_size_mb': round(Path(file_path).stat().st_size / (1024 * 1024), 2),
            'num_rows': df.shape[0],
            'num_columns': df.shape[1],
            'columns': {}
        }
        
        # Per-column metadata
        for col in df.columns:
            col_series = df[col]
            dtype_str = str(col_series.dtype)
            
            col_info = {
                'dtype': dtype_str,
                'missing_count': col_series.null_count(),
                'missing_pct': round(col_series.null_count() / len(col_series) * 100, 2),
                'unique_count': col_series.n_unique() if len(col_series) < 100000 else None  # Skip for huge datasets
            }
            
            # Type-specific stats (lightweight)
            if dtype_str in ['Int64', 'Float64', 'Int32', 'Float32']:
                try:
                    col_info['min'] = float(col_series.min())
                    col_info['max'] = float(col_series.max())
                    col_info['mean'] = float(col_series.mean())
                except:
                    pass
            
            schema_info['columns'][col] = col_info
        
        # Small sample for LLM context (only first few rows)
        sample_data = df.head(sample_rows).to_dicts()
        schema_info['sample_rows'] = sample_data
        
        # Categorize columns
        schema_info['numeric_columns'] = [
            col for col, info in schema_info['columns'].items()
            if 'Int' in info['dtype'] or 'Float' in info['dtype']
        ]
        schema_info['categorical_columns'] = [
            col for col, info in schema_info['columns'].items()
            if info['dtype'] in ['Utf8', 'String'] or (
                info.get('unique_count') is not None and 
                info.get('unique_count') < 50 and 
                col not in schema_info['numeric_columns']
            )
        ]
        schema_info['datetime_columns'] = [
            col for col, info in schema_info['columns'].items()
            if 'Date' in info['dtype'] or 'Time' in info['dtype']
        ]
        
        return schema_info
        
    except Exception as e:
        return {
            'error': f"Failed to extract schema: {str(e)}",
            'file_path': file_path
        }


def infer_task_type(target_column: str, schema_info: Dict[str, Any]) -> Optional[str]:
    """
    Infer ML task type from target column without LLM.
    """
    if not target_column or target_column not in schema_info.get('columns', {}):
        return None
    
    target_info = schema_info['columns'][target_column]
    
    # Numeric with many unique values → regression
    if target_info['dtype'] in ['Int64', 'Float64', 'Int32', 'Float32']:
        unique_count = target_info.get('unique_count')
        if unique_count and unique_count > 20:
            return 'regression'
        elif unique_count and unique_count <= 10:
            return 'classification'
    
    # Categorical or low cardinality → classification
    if target_info['dtype'] in ['Utf8', 'String'] or target_info.get('unique_count', 0) <= 20:
        return 'classification'
    
    return None
