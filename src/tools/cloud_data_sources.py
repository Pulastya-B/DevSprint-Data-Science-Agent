"""
Cloud Data Sources - BigQuery Integration
Tools for loading and writing data to/from Google BigQuery.
Compatible with existing DataScienceCopilot tool registry.
"""

import polars as pl
import pandas as pd
from typing import Dict, Any, Optional, Literal
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.validation import validate_dataframe

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    bigquery = None
    service_account = None


def _get_bigquery_client(project_id: str) -> 'bigquery.Client':
    """
    Initialize BigQuery client with credentials from environment.
    
    Credential sources (in order of priority):
    1. GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON path)
    2. Default application credentials (gcloud auth application-default login)
    
    Args:
        project_id: Google Cloud project ID
        
    Returns:
        BigQuery client instance
        
    Raises:
        ImportError: If google-cloud-bigquery not installed
        EnvironmentError: If credentials not found
    """
    if not BIGQUERY_AVAILABLE:
        raise ImportError(
            "google-cloud-bigquery is not installed. "
            "Install it with: pip install google-cloud-bigquery"
        )
    
    # Check for service account credentials
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if creds_path and Path(creds_path).exists():
        # Use service account JSON
        credentials = service_account.Credentials.from_service_account_file(creds_path)
        client = bigquery.Client(project=project_id, credentials=credentials)
    else:
        # Use default application credentials
        try:
            client = bigquery.Client(project=project_id)
        except Exception as e:
            raise EnvironmentError(
                "BigQuery credentials not found. Either:\n"
                "1. Set GOOGLE_APPLICATION_CREDENTIALS to service account JSON path\n"
                "2. Run: gcloud auth application-default login\n"
                f"Error: {str(e)}"
            )
    
    return client


def load_bigquery_table(
    project_id: str,
    dataset: str,
    table: str,
    limit: Optional[int] = None,
    columns: Optional[list] = None,
    where_clause: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load data from BigQuery table into a Polars DataFrame.
    
    This tool allows the agent to load data from BigQuery for analysis.
    Supports sampling via LIMIT and column selection for memory efficiency.
    
    Args:
        project_id: Google Cloud project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        limit: Optional row limit for sampling (e.g., 10000 for large tables)
        columns: Optional list of column names to load (default: all columns)
        where_clause: Optional SQL WHERE clause for filtering (without WHERE keyword)
            Example: "created_at > '2024-01-01'"
    
    Returns:
        Dictionary with:
        - success: bool
        - data_path: str (saved CSV path for downstream tools)
        - df_info: dict (shape, columns, memory_usage)
        - message: str
        - query_stats: dict (bytes processed, rows returned)
    
    Examples:
        >>> # Load full table
        >>> load_bigquery_table("my-project", "analytics", "users")
        
        >>> # Sample 10K rows for exploration
        >>> load_bigquery_table("my-project", "analytics", "events", limit=10000)
        
        >>> # Load specific columns with filter
        >>> load_bigquery_table(
        ...     "my-project", "sales", "transactions",
        ...     columns=["customer_id", "amount", "date"],
        ...     where_clause="date >= '2024-01-01'",
        ...     limit=50000
        ... )
    """
    try:
        # Initialize client
        client = _get_bigquery_client(project_id)
        
        # Build query
        table_ref = f"{project_id}.{dataset}.{table}"
        
        if columns:
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"
        
        query = f"SELECT {columns_str} FROM `{table_ref}`"
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute query
        query_job = client.query(query)
        
        # Load results into pandas (BigQuery SDK returns pandas)
        df_pandas = query_job.to_dataframe()
        
        # Convert to Polars for consistency with existing tools
        df = pl.from_pandas(df_pandas)
        
        # Validate
        validate_dataframe(df)
        
        # Save to outputs/data/ for downstream tool compatibility
        output_dir = Path("./outputs/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"bigquery_{dataset}_{table}.csv"
        df.write_csv(output_path)
        
        # Get query statistics
        bytes_processed = query_job.total_bytes_processed or 0
        bytes_billed = query_job.total_bytes_billed or 0
        
        return {
            "success": True,
            "data_path": str(output_path),
            "df_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns,
                "memory_mb": round(df.estimated_size("mb"), 2)
            },
            "query_stats": {
                "bytes_processed": bytes_processed,
                "bytes_processed_mb": round(bytes_processed / 1024 / 1024, 2),
                "bytes_billed": bytes_billed,
                "bytes_billed_mb": round(bytes_billed / 1024 / 1024, 2),
                "rows_returned": len(df)
            },
            "message": f"✅ Loaded {len(df):,} rows from {table_ref}. Saved to {output_path}",
            "table_reference": table_ref,
            "query": query
        }
    
    except ImportError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "ImportError",
            "message": "BigQuery library not installed. Run: pip install google-cloud-bigquery"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": f"Failed to load BigQuery table: {str(e)}"
        }


def write_bigquery_table(
    file_path: str,
    project_id: str,
    dataset: str,
    table: str,
    mode: Literal["append", "overwrite", "fail"] = "append"
) -> Dict[str, Any]:
    """
    Write DataFrame to BigQuery table from CSV/Parquet file.
    
    This tool allows the agent to save predictions, metrics, or processed data
    back to BigQuery for downstream consumption.
    
    Args:
        file_path: Path to CSV or Parquet file containing data to write
        project_id: Google Cloud project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
        mode: Write mode
            - "append": Add rows to existing table
            - "overwrite": Replace table contents
            - "fail": Raise error if table exists
    
    Returns:
        Dictionary with:
        - success: bool
        - table_reference: str
        - rows_written: int
        - message: str
    
    Examples:
        >>> # Write predictions to BigQuery
        >>> write_bigquery_table(
        ...     "./outputs/data/predictions.csv",
        ...     "my-project",
        ...     "ml_results",
        ...     "churn_predictions",
        ...     mode="append"
        ... )
        
        >>> # Overwrite existing metrics table
        >>> write_bigquery_table(
        ...     "./outputs/data/metrics.csv",
        ...     "my-project",
        ...     "ml_results",
        ...     "model_metrics",
        ...     mode="overwrite"
        ... )
    """
    try:
        # Initialize client
        client = _get_bigquery_client(project_id)
        
        # Load data from file
        file_path = Path(file_path)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "error_type": "FileNotFoundError"
            }
        
        # Load based on extension
        if file_path.suffix.lower() == ".csv":
            df = pl.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            df = pl.read_parquet(file_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file format: {file_path.suffix}",
                "error_type": "ValueError"
            }
        
        # Convert to pandas (BigQuery SDK requires pandas)
        df_pandas = df.to_pandas()
        
        # Build table reference
        table_ref = f"{project_id}.{dataset}.{table}"
        
        # Configure write disposition
        if mode == "append":
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        elif mode == "overwrite":
            write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        elif mode == "fail":
            write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
        else:
            return {
                "success": False,
                "error": f"Invalid mode: {mode}. Use 'append', 'overwrite', or 'fail'",
                "error_type": "ValueError"
            }
        
        # Configure job
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            autodetect=True  # Auto-detect schema from DataFrame
        )
        
        # Execute write job
        job = client.load_table_from_dataframe(
            df_pandas,
            table_ref,
            job_config=job_config
        )
        
        # Wait for completion
        job.result()
        
        return {
            "success": True,
            "table_reference": table_ref,
            "rows_written": len(df_pandas),
            "mode": mode,
            "message": f"✅ Wrote {len(df_pandas):,} rows to {table_ref} (mode: {mode})",
            "table_info": {
                "project": project_id,
                "dataset": dataset,
                "table": table,
                "columns": df.columns,
                "rows": len(df)
            }
        }
    
    except ImportError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "ImportError",
            "message": "BigQuery library not installed. Run: pip install google-cloud-bigquery"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": f"Failed to write to BigQuery: {str(e)}"
        }


def profile_bigquery_table(
    project_id: str,
    dataset: str,
    table: str
) -> Dict[str, Any]:
    """
    Profile a BigQuery table without loading all data.
    
    Returns metadata including row count, column types, null counts,
    and table size. Useful for initial exploration before full load.
    
    Args:
        project_id: Google Cloud project ID
        dataset: BigQuery dataset name
        table: BigQuery table name
    
    Returns:
        Dictionary with:
        - success: bool
        - table_reference: str
        - row_count: int
        - columns: list of dicts with column info
        - table_size_mb: float
        - created: str (timestamp)
        - modified: str (timestamp)
        - message: str
    
    Examples:
        >>> # Quick profile before loading
        >>> profile_bigquery_table("my-project", "analytics", "events")
        {
            "success": True,
            "row_count": 1000000,
            "columns": [
                {"name": "user_id", "type": "STRING", "mode": "NULLABLE"},
                {"name": "event_time", "type": "TIMESTAMP", "mode": "REQUIRED"},
                ...
            ],
            "table_size_mb": 125.5
        }
    """
    try:
        # Initialize client
        client = _get_bigquery_client(project_id)
        
        # Get table metadata
        table_ref = f"{project_id}.{dataset}.{table}"
        table_obj = client.get_table(table_ref)
        
        # Extract schema information
        columns_info = []
        for field in table_obj.schema:
            columns_info.append({
                "name": field.name,
                "type": field.field_type,
                "mode": field.mode,  # NULLABLE, REQUIRED, REPEATED
                "description": field.description or ""
            })
        
        # Get null counts via query (sample for efficiency)
        null_counts = {}
        try:
            # Use TABLESAMPLE for large tables (1% sample)
            sample_query = f"""
            SELECT 
                {', '.join([f'COUNTIF({col["name"]} IS NULL) AS {col["name"]}_nulls' for col in columns_info])}
            FROM `{table_ref}`
            TABLESAMPLE SYSTEM (1 PERCENT)
            """
            
            query_job = client.query(sample_query)
            result = query_job.result()
            row = next(iter(result))
            
            for col in columns_info:
                null_count = row.get(f'{col["name"]}_nulls', 0)
                null_counts[col["name"]] = null_count
        except Exception as e:
            # If sampling fails, skip null counts
            null_counts = {col["name"]: "N/A" for col in columns_info}
        
        # Table size information
        table_size_bytes = table_obj.num_bytes or 0
        table_size_mb = round(table_size_bytes / 1024 / 1024, 2)
        
        return {
            "success": True,
            "table_reference": table_ref,
            "profile": {
                "row_count": table_obj.num_rows,
                "column_count": len(columns_info),
                "table_size_mb": table_size_mb,
                "table_size_gb": round(table_size_mb / 1024, 2)
            },
            "columns": columns_info,
            "null_counts_sample": null_counts,
            "metadata": {
                "created": table_obj.created.isoformat() if table_obj.created else None,
                "modified": table_obj.modified.isoformat() if table_obj.modified else None,
                "location": table_obj.location,
                "expiration": table_obj.expires.isoformat() if table_obj.expires else None
            },
            "message": f"✅ Profiled {table_ref}: {table_obj.num_rows:,} rows, {len(columns_info)} columns, {table_size_mb} MB",
            "recommendation": (
                f"Table has {table_obj.num_rows:,} rows. "
                f"Consider using limit={min(10000, table_obj.num_rows)} for initial exploration."
                if table_obj.num_rows > 10000 else
                f"Table is small ({table_obj.num_rows:,} rows), safe to load fully."
            )
        }
    
    except ImportError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "ImportError",
            "message": "BigQuery library not installed. Run: pip install google-cloud-bigquery"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": f"Failed to profile BigQuery table: {str(e)}"
        }


def query_bigquery(
    project_id: str,
    query: str,
    output_path: Optional[str] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Execute a custom BigQuery SQL query and return results as DataFrame.
    
    This tool allows the agent to run custom SQL queries for complex
    data transformations before analysis.
    
    Args:
        project_id: Google Cloud project ID
        query: SQL query to execute
        output_path: Optional path to save results (default: auto-generated)
        limit: Optional row limit to append to query
    
    Returns:
        Dictionary with:
        - success: bool
        - data_path: str
        - df_info: dict
        - query_stats: dict
        - message: str
    
    Examples:
        >>> # Custom aggregation query
        >>> query_bigquery(
        ...     "my-project",
        ...     '''
        ...     SELECT 
        ...         customer_id,
        ...         SUM(amount) as total_spent,
        ...         COUNT(*) as num_orders
        ...     FROM `my-project.sales.orders`
        ...     WHERE date >= '2024-01-01'
        ...     GROUP BY customer_id
        ...     '''
        ... )
    """
    try:
        # Initialize client
        client = _get_bigquery_client(project_id)
        
        # Add limit if specified
        if limit:
            query = f"{query.rstrip(';')} LIMIT {limit}"
        
        # Execute query
        query_job = client.query(query)
        df_pandas = query_job.to_dataframe()
        
        # Convert to Polars
        df = pl.from_pandas(df_pandas)
        
        # Determine output path
        if output_path is None:
            output_dir = Path("./outputs/data")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / "bigquery_query_result.csv")
        
        # Save results
        df.write_csv(output_path)
        
        # Get query statistics
        bytes_processed = query_job.total_bytes_processed or 0
        
        return {
            "success": True,
            "data_path": output_path,
            "df_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": df.columns,
                "memory_mb": round(df.estimated_size("mb"), 2)
            },
            "query_stats": {
                "bytes_processed": bytes_processed,
                "bytes_processed_mb": round(bytes_processed / 1024 / 1024, 2),
                "rows_returned": len(df)
            },
            "message": f"✅ Query returned {len(df):,} rows. Saved to {output_path}",
            "query": query
        }
    
    except ImportError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "ImportError",
            "message": "BigQuery library not installed. Run: pip install google-cloud-bigquery"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "message": f"Failed to execute BigQuery query: {str(e)}"
        }


# Export functions for tool registry
__all__ = [
    'load_bigquery_table',
    'write_bigquery_table',
    'profile_bigquery_table',
    'query_bigquery'
]
