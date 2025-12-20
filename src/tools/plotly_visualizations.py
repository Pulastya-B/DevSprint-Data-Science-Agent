"""
Plotly Interactive Visualization Tools
Create interactive, web-based visualizations that can be explored in browsers.
"""

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
    validate_column_exists,
)


def generate_interactive_scatter(
    file_path: str,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    output_path: str = "./outputs/plots/interactive/scatter.html"
) -> Dict[str, Any]:
    """
    Create interactive scatter plot with Plotly.
    
    Args:
        file_path: Path to dataset
        x_col: Column for X-axis
        y_col: Column for Y-axis
        color_col: Optional column for color coding
        size_col: Optional column for bubble size
        output_path: Path to save HTML file
        
    Returns:
        Dictionary with plot info and path
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, x_col)
    validate_column_exists(df, y_col)
    
    if color_col:
        validate_column_exists(df, color_col)
    if size_col:
        validate_column_exists(df, size_col)
    
    # Convert to pandas for plotly
    df_pd = df.to_pandas()
    
    # Create figure
    fig = px.scatter(
        df_pd,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        hover_data=df_pd.columns.tolist(),
        title=f"Interactive Scatter: {y_col} vs {x_col}",
        template="plotly_white"
    )
    
    # Update layout for better interactivity
    fig.update_layout(
        hovermode='closest',
        height=600,
        font=dict(size=12)
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    
    return {
        "status": "success",
        "plot_type": "interactive_scatter",
        "output_path": output_path,
        "x_col": x_col,
        "y_col": y_col,
        "color_col": color_col,
        "size_col": size_col,
        "num_points": len(df)
    }


def generate_interactive_histogram(
    file_path: str,
    column: str,
    bins: int = 30,
    color_col: Optional[str] = None,
    output_path: str = "./outputs/plots/interactive/histogram.html"
) -> Dict[str, Any]:
    """
    Create interactive histogram with Plotly.
    
    Args:
        file_path: Path to dataset
        column: Column to plot
        bins: Number of bins
        color_col: Optional column for grouped histograms
        output_path: Path to save HTML file
        
    Returns:
        Dictionary with plot info
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, column)
    
    if color_col:
        validate_column_exists(df, color_col)
    
    df_pd = df.to_pandas()
    
    # Create histogram
    fig = px.histogram(
        df_pd,
        x=column,
        nbins=bins,
        color=color_col,
        title=f"Distribution of {column}",
        template="plotly_white",
        marginal="box"  # Add box plot on top
    )
    
    fig.update_layout(
        bargap=0.1,
        height=600,
        showlegend=True if color_col else False
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    
    return {
        "status": "success",
        "plot_type": "interactive_histogram",
        "output_path": output_path,
        "column": column,
        "bins": bins,
        "color_col": color_col
    }


def generate_interactive_correlation_heatmap(
    file_path: str,
    output_path: str = "./outputs/plots/interactive/correlation_heatmap.html"
) -> Dict[str, Any]:
    """
    Create interactive correlation heatmap with Plotly.
    
    Args:
        file_path: Path to dataset
        output_path: Path to save HTML file
        
    Returns:
        Dictionary with plot info
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Get numeric columns
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        return {
            "status": "error",
            "message": "Need at least 2 numeric columns for correlation"
        }
    
    # Calculate correlation matrix
    df_numeric = df.select(numeric_cols)
    corr_matrix = df_numeric.to_pandas().corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Interactive Correlation Heatmap",
        template="plotly_white",
        height=max(600, len(numeric_cols) * 30),
        width=max(600, len(numeric_cols) * 30),
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    
    return {
        "status": "success",
        "plot_type": "interactive_correlation_heatmap",
        "output_path": output_path,
        "num_features": len(numeric_cols)
    }


def generate_interactive_box_plots(
    file_path: str,
    columns: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    output_path: str = "./outputs/plots/interactive/box_plots.html"
) -> Dict[str, Any]:
    """
    Create interactive box plots for outlier detection.
    
    Args:
        file_path: Path to dataset
        columns: Columns to plot (all numeric if None)
        group_by: Optional categorical column for grouping
        output_path: Path to save HTML file
        
    Returns:
        Dictionary with plot info
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Determine columns to plot
    if columns is None:
        columns = get_numeric_columns(df)
    else:
        for col in columns:
            validate_column_exists(df, col)
    
    if len(columns) == 0:
        return {
            "status": "error",
            "message": "No numeric columns to plot"
        }
    
    if group_by:
        validate_column_exists(df, group_by)
    
    df_pd = df.to_pandas()
    
    # Create subplots
    rows = (len(columns) + 2) // 3  # 3 plots per row
    cols = min(3, len(columns))
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=columns,
        vertical_spacing=0.1
    )
    
    for idx, col in enumerate(columns):
        row = idx // 3 + 1
        col_idx = idx % 3 + 1
        
        if group_by:
            for group in df_pd[group_by].unique():
                group_data = df_pd[df_pd[group_by] == group][col]
                fig.add_trace(
                    go.Box(y=group_data, name=str(group), showlegend=(idx == 0)),
                    row=row,
                    col=col_idx
                )
        else:
            fig.add_trace(
                go.Box(y=df_pd[col], name=col, showlegend=False),
                row=row,
                col=col_idx
            )
    
    fig.update_layout(
        title="Interactive Box Plots - Outlier Detection",
        template="plotly_white",
        height=400 * rows,
        showlegend=bool(group_by)
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    
    return {
        "status": "success",
        "plot_type": "interactive_box_plots",
        "output_path": output_path,
        "columns_plotted": columns,
        "group_by": group_by
    }


def generate_interactive_time_series(
    file_path: str,
    time_col: str,
    value_cols: List[str],
    output_path: str = "./outputs/plots/interactive/time_series.html"
) -> Dict[str, Any]:
    """
    Create interactive time series plot with Plotly.
    
    Args:
        file_path: Path to dataset
        time_col: Column with datetime values
        value_cols: Columns to plot over time
        output_path: Path to save HTML file
        
    Returns:
        Dictionary with plot info
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, time_col)
    
    for col in value_cols:
        validate_column_exists(df, col)
    
    # Parse datetime if needed
    if df[time_col].dtype == pl.Utf8:
        df = df.with_columns(
            pl.col(time_col).str.strptime(pl.Datetime, strict=False).alias(time_col)
        )
    
    df_pd = df.to_pandas()
    
    # Create figure
    fig = go.Figure()
    
    for col in value_cols:
        fig.add_trace(go.Scatter(
            x=df_pd[time_col],
            y=df_pd[col],
            mode='lines+markers',
            name=col,
            hovertemplate=f'<b>{col}</b><br>Time: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Interactive Time Series",
        xaxis_title=time_col,
        yaxis_title="Value",
        template="plotly_white",
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    
    return {
        "status": "success",
        "plot_type": "interactive_time_series",
        "output_path": output_path,
        "time_col": time_col,
        "value_cols": value_cols
    }


def generate_plotly_dashboard(
    file_path: str,
    target_col: Optional[str] = None,
    output_dir: str = "./outputs/plots/interactive"
) -> Dict[str, Any]:
    """
    Generate a complete dashboard with multiple interactive plots.
    
    Args:
        file_path: Path to dataset
        target_col: Optional target column for supervised analysis
        output_dir: Directory to save all plots
        
    Returns:
        Dictionary with paths to all generated plots
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if target_col:
        validate_column_exists(df, target_col)
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    plots_generated = []
    
    # 1. Correlation heatmap
    if len(numeric_cols) >= 2:
        result = generate_interactive_correlation_heatmap(
            file_path,
            output_path=f"{output_dir}/correlation_heatmap.html"
        )
        if result["status"] == "success":
            plots_generated.append(result)
    
    # 2. Box plots for outliers
    if len(numeric_cols) > 0:
        result = generate_interactive_box_plots(
            file_path,
            columns=numeric_cols[:10],  # Limit to 10 for performance
            output_path=f"{output_dir}/box_plots.html"
        )
        if result["status"] == "success":
            plots_generated.append(result)
    
    # 3. Target variable analysis if provided
    if target_col and target_col in numeric_cols:
        # Scatter plots against target
        for col in numeric_cols[:5]:  # Top 5 features
            if col != target_col:
                result = generate_interactive_scatter(
                    file_path,
                    x_col=col,
                    y_col=target_col,
                    output_path=f"{output_dir}/scatter_{col}_vs_{target_col}.html"
                )
                if result["status"] == "success":
                    plots_generated.append(result)
    
    # 4. Distribution plots for numeric features
    for col in numeric_cols[:5]:  # Top 5 features
        result = generate_interactive_histogram(
            file_path,
            column=col,
            output_path=f"{output_dir}/histogram_{col}.html"
        )
        if result["status"] == "success":
            plots_generated.append(result)
    
    return {
        "status": "success",
        "plots_generated": len(plots_generated),
        "plots": plots_generated,
        "output_dir": output_dir
    }
