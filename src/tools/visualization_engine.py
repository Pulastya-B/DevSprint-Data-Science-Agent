"""
Comprehensive Visualization Engine (Matplotlib + Seaborn)
Automatically generate all relevant plots for data analysis and model evaluation.

All functions now return matplotlib Figure objects for Gradio compatibility.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Gradio

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import load_dataframe
from utils.validation import validate_file_exists

# Import matplotlib visualization functions
try:
    from .matplotlib_visualizations import (
        create_scatter_plot,
        create_bar_chart,
        create_histogram,
        create_boxplot,
        create_correlation_heatmap,
        create_distribution_plot,
        create_roc_curve,
        create_confusion_matrix,
        create_feature_importance,
        create_residual_plot,
        create_missing_values_heatmap,
        create_missing_values_bar,
        create_outlier_detection_boxplot,
        save_figure,
        close_figure
    )
except ImportError:
    # Fallback for direct execution
    from matplotlib_visualizations import (
        create_scatter_plot,
        create_bar_chart,
        create_histogram,
        create_boxplot,
        create_correlation_heatmap,
        create_distribution_plot,
        create_roc_curve,
        create_confusion_matrix,
        create_feature_importance,
        create_residual_plot,
        create_missing_values_heatmap,
        create_missing_values_bar,
        create_outlier_detection_boxplot,
        save_figure,
        close_figure
    )

# Set global style
sns.set_style('whitegrid')


def generate_all_plots(file_path: str,
                       target_col: Optional[str] = None,
                       output_dir: str = "./outputs/plots") -> Dict[str, Any]:
    """
    Generate ALL plots for a dataset automatically.
    
    Generates:
    - Data quality plots
    - EDA plots
    - Distribution plots
    - Correlation plots
    
    Args:
        file_path: Path to dataset
        target_col: Optional target column
        output_dir: Directory to save plots
        
    Returns:
        Dictionary with Figure objects and saved file paths
    """
    validate_file_exists(file_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {
        "output_directory": output_dir,
        "plots_generated": [],
        "figure_objects": [],  # Store Figure objects
        "plot_categories": {}
    }
    
    print(f"🎨 Generating comprehensive visualizations...")
    
    # 1. Data Quality Plots
    quality_plots = generate_data_quality_plots(file_path, output_dir)
    results["plot_categories"]["data_quality"] = quality_plots
    results["plots_generated"].extend(quality_plots.get("plot_paths", []))
    results["figure_objects"].extend(quality_plots.get("figures", []))
    
    # 2. EDA Plots
    eda_plots = generate_eda_plots(file_path, target_col, output_dir)
    results["plot_categories"]["eda"] = eda_plots
    results["plots_generated"].extend(eda_plots.get("plot_paths", []))
    results["figure_objects"].extend(eda_plots.get("figures", []))
    
    # 3. Distribution Plots
    dist_plots = generate_distribution_plots(file_path, output_dir)
    results["plot_categories"]["distributions"] = dist_plots
    results["plots_generated"].extend(dist_plots.get("plot_paths", []))
    results["figure_objects"].extend(dist_plots.get("figures", []))
    
    results["total_plots"] = len(results["plots_generated"])
    print(f"✅ Generated {results['total_plots']} plots in {output_dir}")
    
    return results


def generate_data_quality_plots(file_path: str, output_dir: str) -> Dict[str, Any]:
    """Generate plots related to data quality using Matplotlib."""
    df = load_dataframe(file_path).to_pandas()
    plots = []
    figures = []
    
    # 1. Missing values bar chart
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig = create_missing_values_bar(
            df=df,
            title="Missing Values by Column",
            figsize=(10, 6)
        )
        if fig is not None:
            path = f"{output_dir}/missing_values.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Missing values plot")
    
    # 2. Data types distribution (pie chart alternative - bar chart)
    dtype_counts = df.dtypes.astype(str).value_counts()
    fig = create_bar_chart(
        categories=dtype_counts.index.tolist(),
        values=dtype_counts.values,
        title="Data Types Distribution",
        xlabel="Data Type",
        ylabel="Count",
        figsize=(8, 6),
        color='steelblue'
    )
    if fig is not None:
        path = f"{output_dir}/data_types.png"
        save_figure(fig, path)
        plots.append(path)
        figures.append(fig)
        print(f"   ✓ Data types plot")
    
    # 3. Outlier detection (box plots)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
    if len(numeric_cols) > 0:
        fig = create_boxplot(
            data=df[numeric_cols],
            title="Outlier Detection (Box Plots)",
            figsize=(12, 6)
        )
        if fig is not None:
            path = f"{output_dir}/outliers_boxplot.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Outlier detection plot")
    
    return {"plot_paths": plots, "figures": figures, "n_plots": len(plots)}


def generate_eda_plots(file_path: str, target_col: Optional[str], output_dir: str) -> Dict[str, Any]:
    """Generate exploratory data analysis plots using Matplotlib."""
    df = load_dataframe(file_path).to_pandas()
    plots = []
    figures = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # 1. Correlation heatmap
    if len(numeric_cols) > 1:
        fig = create_correlation_heatmap(
            data=df[numeric_cols[:15]],  # Limit to 15 features
            title="Feature Correlation Matrix",
            figsize=(12, 10),
            annot=True,
            cmap='RdBu_r'
        )
        if fig is not None:
            path = f"{output_dir}/correlation_heatmap.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Correlation heatmap")
    
    # 2. Feature relationships with target (scatter plots)
    if target_col and target_col in df.columns and len(numeric_cols) > 0:
        top_features = numeric_cols[:4]  # Top 4 features
        
        # Create multiple scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(top_features):
            ax = axes[i]
            ax.scatter(df[col], df[target_col], alpha=0.5, s=30, 
                      c='steelblue', edgecolors='black', linewidth=0.5)
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel(target_col, fontsize=11)
            ax.set_title(f"{col} vs {target_col}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle(f"Top Features vs {target_col}", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        path = f"{output_dir}/feature_relationships.png"
        save_figure(fig, path)
        plots.append(path)
        figures.append(fig)
        print(f"   ✓ Feature relationships plot")
    
    # 3. Pairplot for top features (sample data for performance)
    if len(numeric_cols) >= 3:
        sample_size = min(1000, len(df))
        sample_df = df[numeric_cols[:3]].sample(sample_size)
        
        # Create pairplot using seaborn
        pair_grid = sns.pairplot(sample_df, corner=True, diag_kind='kde', 
                                plot_kws={'alpha': 0.6, 's': 20})
        fig = pair_grid.fig
        fig.suptitle("Feature Pairplot (Top 3 Features)", fontsize=14, 
                    fontweight='bold', y=1.01)
        
        path = f"{output_dir}/pairplot.png"
        save_figure(fig, path)
        plots.append(path)
        figures.append(fig)
        print(f"   ✓ Pairplot")
    
    return {"plot_paths": plots, "figures": figures, "n_plots": len(plots)}


def generate_distribution_plots(file_path: str, output_dir: str) -> Dict[str, Any]:
    """Generate distribution analysis plots using Matplotlib."""
    df = load_dataframe(file_path).to_pandas()
    plots = []
    figures = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
    
    if len(numeric_cols) > 0:
        # Histograms for numeric features in a grid
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            data = df[col].dropna()
            
            # Create histogram with KDE
            ax.hist(data, bins=30, color='steelblue', edgecolor='black', 
                   alpha=0.7, density=True)
            
            # Add KDE
            try:
                sns.kdeplot(data, ax=ax, color='darkred', linewidth=2)
            except:
                pass  # Skip KDE if it fails
            
            ax.set_title(col[:25], fontsize=11, fontweight='bold')
            ax.set_xlabel('Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle("Feature Distributions", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        path = f"{output_dir}/distributions_histogram.png"
        save_figure(fig, path)
        plots.append(path)
        figures.append(fig)
        print(f"   ✓ Distribution histograms")
    
    return {"plot_paths": plots, "figures": figures, "n_plots": len(plots)}


def generate_model_performance_plots(y_true, y_pred, y_pred_proba=None,
                                     task_type="regression",
                                     model_name="Model",
                                     output_dir="./outputs/plots") -> Dict[str, Any]:
    """
    Generate model performance plots using Matplotlib.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for classification)
        task_type: 'classification' or 'regression'
        model_name: Name of the model
        output_dir: Output directory
        
    Returns:
        Dictionary with plot paths and figure objects
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plots = []
    figures = []
    
    if task_type == "classification":
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = [f"Class {i}" for i in range(len(cm))]
        
        fig = create_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=f"Confusion Matrix - {model_name}",
            show_percentages=True,
            figsize=(10, 8)
        )
        if fig is not None:
            path = f"{output_dir}/confusion_matrix_{model_name}.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Confusion matrix")
        
        # 2. ROC Curve (if probabilities provided)
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            y_proba = y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            models_data = {model_name: (fpr, tpr, roc_auc)}
            fig = create_roc_curve(
                models_data=models_data,
                title=f"ROC Curve - {model_name}",
                figsize=(10, 8)
            )
            if fig is not None:
                path = f"{output_dir}/roc_curve_{model_name}.png"
                save_figure(fig, path)
                plots.append(path)
                figures.append(fig)
                print(f"   ✓ ROC curve")
    
    else:  # Regression
        # 1. Residual plot (Predicted vs Actual + Residuals)
        fig = create_residual_plot(
            y_true=y_true,
            y_pred=y_pred,
            title=f"Residual Analysis - {model_name}",
            figsize=(10, 6)
        )
        if fig is not None:
            path = f"{output_dir}/residuals_{model_name}.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Residual plot")
        
        # 2. Residuals distribution
        residuals = y_true - y_pred
        fig = create_histogram(
            data=residuals,
            title=f"Residuals Distribution - {model_name}",
            xlabel="Residuals",
            ylabel="Frequency",
            bins=30,
            kde=True,
            figsize=(10, 6)
        )
        if fig is not None:
            path = f"{output_dir}/residuals_dist_{model_name}.png"
            save_figure(fig, path)
            plots.append(path)
            figures.append(fig)
            print(f"   ✓ Residuals distribution")
    
    return {"plot_paths": plots, "figures": figures, "n_plots": len(plots)}


def generate_feature_importance_plot(feature_importances: Dict[str, float],
                                     output_path: str = "./outputs/plots/feature_importance.png",
                                     top_n: int = 20) -> str:
    """
    Generate feature importance plot using Matplotlib.
    
    Args:
        feature_importances: Dictionary of feature: importance
        output_path: Where to save the plot
        top_n: Number of top features to show
        
    Returns:
        Path to saved plot
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dict to lists
    features = list(feature_importances.keys())
    importances = np.array(list(feature_importances.values()))
    
    # Create plot
    fig = create_feature_importance(
        feature_names=features,
        importances=importances,
        title=f"Top {top_n} Feature Importances",
        top_n=top_n,
        figsize=(10, max(8, top_n * 0.4))
    )
    
    if fig is not None:
        save_figure(fig, output_path)
        print(f"   ✓ Feature importance plot")
        close_figure(fig)
        return output_path
    
    return None


def generate_learning_curve(train_sizes, train_scores, val_scores,
                           model_name="Model",
                           output_path="./outputs/plots/learning_curve.png") -> str:
    """Generate learning curve plot using Matplotlib."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate mean and std
    if isinstance(train_scores, list):
        train_scores = np.array(train_scores)
        val_scores = np.array(val_scores)
    
    if train_scores.ndim > 1:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)
    else:
        train_scores_mean = train_scores
        train_scores_std = np.zeros_like(train_scores)
        val_scores_mean = val_scores
        val_scores_std = np.zeros_like(val_scores)
    
    # Create plot
    from .matplotlib_visualizations import create_learning_curve as mlp_learning_curve
    
    fig = mlp_learning_curve(
        train_sizes=train_sizes,
        train_scores_mean=train_scores_mean,
        train_scores_std=train_scores_std,
        val_scores_mean=val_scores_mean,
        val_scores_std=val_scores_std,
        title=f"Learning Curve - {model_name}",
        figsize=(10, 6)
    )
    
    if fig is not None:
        save_figure(fig, output_path)
        close_figure(fig)
        return output_path
    
    return None


def create_plot_gallery_html(plot_paths: List[str], output_path: str = "./outputs/plots/gallery.html") -> str:
    """
    Create an HTML gallery page showing all plots (now as PNG images).
    
    Args:
        plot_paths: List of paths to plot files (now PNG instead of HTML)
        output_path: Where to save the gallery
        
    Returns:
        Path to gallery HTML
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Plot Gallery</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .plot-container {
                background: white;
                margin: 20px 0;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .plot-image {
                width: 100%;
                max-width: 1200px;
                height: auto;
                display: block;
                margin: 0 auto;
            }
            .plot-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>📊 Data Analysis Visualization Gallery</h1>
        <p style="text-align: center; color: #666;">Total Plots: {}</p>
    """.format(len(plot_paths))
    
    for i, plot_path in enumerate(plot_paths, 1):
        plot_name = Path(plot_path).stem.replace('_', ' ').title()
        rel_path = os.path.relpath(plot_path, os.path.dirname(output_path))
        
        html_content += f"""
        <div class="plot-container">
            <div class="plot-title">{i}. {plot_name}</div>
            <img src="{rel_path}" alt="{plot_name}" class="plot-image">
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Created plot gallery: {output_path}")
    return output_path
