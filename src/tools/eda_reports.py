"""
EDA Report Generation Tools
Generates comprehensive HTML reports using Sweetviz and ydata-profiling.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import polars as pl


def generate_sweetviz_report(
    file_path: str,
    output_path: str = "./outputs/reports/sweetviz_report.html",
    target_column: Optional[str] = None,
    compare_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a beautiful HTML report using Sweetviz.
    
    Sweetviz creates stunning visualizations for EDA with:
    - Target analysis (associations with target variable)
    - Feature distributions and statistics
    - Correlations and relationships
    - Missing value analysis
    - Comparison between datasets (train vs test)
    
    Args:
        file_path: Path to the dataset CSV file
        output_path: Where to save the HTML report
        target_column: Optional target variable for analysis
        compare_file_path: Optional second dataset to compare against
        
    Returns:
        Dict with success status, report path, and summary
    """
    try:
        import sweetviz as sv
        import pandas as pd
        
        # Read dataset (Sweetviz requires pandas)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or "./outputs/reports", exist_ok=True)
        
        # Generate report based on configuration
        if compare_file_path:
            # Comparison report (e.g., train vs test)
            if compare_file_path.endswith('.csv'):
                df_compare = pd.read_csv(compare_file_path)
            elif compare_file_path.endswith('.parquet'):
                df_compare = pd.read_parquet(compare_file_path)
            else:
                raise ValueError(f"Unsupported compare file format: {compare_file_path}")
            
            report = sv.compare([df, "Dataset 1"], [df_compare, "Dataset 2"], target_column)
        elif target_column:
            # Analysis with target variable
            if target_column not in df.columns:
                available = list(df.columns)
                return {
                    "success": False,
                    "error": f"Column '{target_column}' not found. Available columns: {', '.join(available)}",
                    "suggestion": f"Did you mean one of: {', '.join(available[:5])}?"
                }
            report = sv.analyze([df, "Dataset"], target_feat=target_column)
        else:
            # Basic analysis without target
            report = sv.analyze(df)
        
        # Generate HTML report
        report.show_html(filepath=output_path, open_browser=False, layout='vertical', scale=1.0)
        
        # Get summary statistics
        num_features = len(df.columns)
        num_rows = len(df)
        num_numeric = df.select_dtypes(include=['number']).shape[1]
        num_categorical = df.select_dtypes(include=['object', 'category']).shape[1]
        missing_pct = (df.isnull().sum().sum() / (num_rows * num_features)) * 100
        
        return {
            "success": True,
            "report_path": output_path,
            "message": f"✅ Sweetviz report generated successfully at: {output_path}",
            "summary": {
                "features": num_features,
                "rows": num_rows,
                "numeric_features": num_numeric,
                "categorical_features": num_categorical,
                "missing_percentage": round(missing_pct, 2),
                "target_column": target_column,
                "has_comparison": compare_file_path is not None
            }
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "Sweetviz not installed. Install with: pip install sweetviz",
            "error_type": "MissingDependency"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate Sweetviz report: {str(e)}",
            "error_type": type(e).__name__
        }


def generate_ydata_profiling_report(
    file_path: str,
    output_path: str = "./outputs/reports/ydata_profile.html",
    minimal: bool = False,
    title: str = "Data Profiling Report"
) -> Dict[str, Any]:
    """
    Generate a comprehensive HTML report using ydata-profiling (formerly pandas-profiling).
    
    ydata-profiling provides extensive analysis including:
    - Overview: dataset statistics, warnings, reproduction
    - Variables: type inference, statistics, histograms, common values, missing values
    - Interactions: scatter plots, correlations (Pearson, Spearman, Kendall, Cramér's V)
    - Correlations: detailed correlation matrices and heatmaps
    - Missing values: matrix, heatmap, and dendrogram
    - Sample: first/last rows of the dataset
    - Duplicate rows: analysis and examples
    
    Args:
        file_path: Path to the dataset CSV file
        output_path: Where to save the HTML report
        minimal: If True, generates faster minimal report (useful for large datasets)
        title: Title for the report
        
    Returns:
        Dict with success status, report path, and statistics
    """
    try:
        from ydata_profiling import ProfileReport
        import pandas as pd
        
        # Read dataset (ydata-profiling requires pandas)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) or "./outputs/reports", exist_ok=True)
        
        # Configure profile based on minimal flag
        if minimal:
            # Minimal mode: faster for large datasets
            profile = ProfileReport(
                df,
                title=title,
                minimal=True,
                explorative=False
            )
        else:
            # Full mode: comprehensive analysis
            profile = ProfileReport(
                df,
                title=title,
                explorative=True,
                correlations={
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                    "kendall": {"calculate": False},  # Slow for large datasets
                    "phi_k": {"calculate": True},
                    "cramers": {"calculate": True},
                }
            )
        
        # Generate HTML report
        profile.to_file(output_path)
        
        # Extract key statistics
        num_features = len(df.columns)
        num_rows = len(df)
        num_numeric = df.select_dtypes(include=['number']).shape[1]
        num_categorical = df.select_dtypes(include=['object', 'category']).shape[1]
        num_boolean = df.select_dtypes(include=['bool']).shape[1]
        missing_cells = df.isnull().sum().sum()
        total_cells = num_rows * num_features
        missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum()
        
        return {
            "success": True,
            "report_path": output_path,
            "message": f"✅ ydata-profiling report generated successfully at: {output_path}",
            "statistics": {
                "dataset_size": {
                    "rows": num_rows,
                    "columns": num_features,
                    "cells": total_cells
                },
                "variable_types": {
                    "numeric": num_numeric,
                    "categorical": num_categorical,
                    "boolean": num_boolean
                },
                "data_quality": {
                    "missing_cells": missing_cells,
                    "missing_percentage": round(missing_pct, 2),
                    "duplicate_rows": int(duplicate_rows)
                },
                "report_config": {
                    "minimal_mode": minimal,
                    "title": title
                }
            }
        }
        
    except ImportError:
        return {
            "success": False,
            "error": "ydata-profiling not installed. Install with: pip install ydata-profiling",
            "error_type": "MissingDependency"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate ydata-profiling report: {str(e)}",
            "error_type": type(e).__name__
        }


def generate_combined_eda_report(
    file_path: str,
    output_dir: str = "./outputs/reports",
    target_column: Optional[str] = None,
    minimal: bool = False
) -> Dict[str, Any]:
    """
    Generate both Sweetviz and ydata-profiling reports in one call.
    
    This convenience function creates comprehensive EDA reports using both tools,
    giving you the best of both worlds:
    - Sweetviz: Beautiful, fast, focused visualizations
    - ydata-profiling: Comprehensive, detailed analysis
    
    Args:
        file_path: Path to the dataset CSV file
        output_dir: Directory to save both reports
        target_column: Optional target variable for Sweetviz analysis
        minimal: If True, uses minimal mode for ydata-profiling
        
    Returns:
        Dict with success status and paths to both reports
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate Sweetviz report
        sweetviz_path = os.path.join(output_dir, "sweetviz_report.html")
        sweetviz_result = generate_sweetviz_report(
            file_path=file_path,
            output_path=sweetviz_path,
            target_column=target_column
        )
        
        # Generate ydata-profiling report
        ydata_path = os.path.join(output_dir, "ydata_profile.html")
        ydata_result = generate_ydata_profiling_report(
            file_path=file_path,
            output_path=ydata_path,
            minimal=minimal
        )
        
        # Check if both succeeded
        both_success = sweetviz_result["success"] and ydata_result["success"]
        
        if both_success:
            return {
                "success": True,
                "message": f"✅ Generated both EDA reports successfully in: {output_dir}",
                "reports": {
                    "sweetviz": {
                        "path": sweetviz_path,
                        "summary": sweetviz_result.get("summary", {})
                    },
                    "ydata_profiling": {
                        "path": ydata_path,
                        "statistics": ydata_result.get("statistics", {})
                    }
                },
                "recommendation": "Open both reports in your browser to get comprehensive insights!"
            }
        else:
            # At least one failed
            errors = []
            if not sweetviz_result["success"]:
                errors.append(f"Sweetviz: {sweetviz_result['error']}")
            if not ydata_result["success"]:
                errors.append(f"ydata-profiling: {ydata_result['error']}")
            
            return {
                "success": False,
                "error": " | ".join(errors),
                "partial_results": {
                    "sweetviz": sweetviz_result,
                    "ydata_profiling": ydata_result
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate combined reports: {str(e)}",
            "error_type": type(e).__name__
        }
