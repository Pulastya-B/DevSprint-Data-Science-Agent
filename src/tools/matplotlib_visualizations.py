"""
Matplotlib + Seaborn Visualization Engine
Production-quality visualizations that work reliably with Gradio UI.

All functions return matplotlib Figure objects (not file paths).
Designed for publication-quality plots with professional styling.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio compatibility

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set global style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


# ============================================================================
# BASIC PLOTS
# ============================================================================

def create_scatter_plot(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[np.ndarray, pd.Series, list],
    hue: Optional[Union[np.ndarray, pd.Series, list]] = None,
    size: Optional[Union[np.ndarray, pd.Series, list]] = None,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.6,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a professional scatter plot with optional color coding and size variation.
    
    Args:
        x: X-axis data
        y: Y-axis data
        hue: Optional categorical data for color coding
        size: Optional numeric data for size variation
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        alpha: Point transparency (0-1)
        save_path: Optional path to save PNG file
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = create_scatter_plot(df['feature1'], df['target'], 
        ...                           hue=df['category'], title='Feature vs Target')
        >>> # Display in Gradio: gr.Plot(value=fig)
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Convert inputs to arrays
        x = np.array(x)
        y = np.array(y)
        
        if hue is not None:
            hue = np.array(hue)
            unique_hues = np.unique(hue)
            colors = sns.color_palette('Set2', n_colors=len(unique_hues))
            
            for i, hue_val in enumerate(unique_hues):
                mask = hue == hue_val
                scatter_size = 50 if size is None else np.array(size)[mask]
                ax.scatter(x[mask], y[mask], 
                          c=[colors[i]], 
                          s=scatter_size,
                          alpha=alpha, 
                          label=str(hue_val),
                          edgecolors='black',
                          linewidth=0.5)
            ax.legend(title='Category', loc='best', framealpha=0.9)
        else:
            scatter_size = 50 if size is None else size
            ax.scatter(x, y, 
                      c='steelblue', 
                      s=scatter_size,
                      alpha=alpha,
                      edgecolors='black',
                      linewidth=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved scatter plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating scatter plot: {str(e)}")
        return None


def create_line_plot(
    x: Union[np.ndarray, pd.Series, list],
    y: Union[Dict[str, np.ndarray], np.ndarray, pd.Series, list],
    title: str = "Line Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    figsize: Tuple[int, int] = (10, 6),
    markers: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a line plot (supports multiple lines via dict).
    
    Args:
        x: X-axis data
        y: Y-axis data (dict for multiple lines: {'label1': y1, 'label2': y2})
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        markers: Show markers on lines
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.array(x)
        marker_style = 'o' if markers else None
        
        if isinstance(y, dict):
            colors = sns.color_palette('husl', n_colors=len(y))
            for i, (label, y_data) in enumerate(y.items()):
                ax.plot(x, np.array(y_data), 
                       marker=marker_style, 
                       label=label,
                       linewidth=2,
                       markersize=6,
                       color=colors[i])
            ax.legend(loc='best', framealpha=0.9)
        else:
            ax.plot(x, np.array(y), 
                   marker=marker_style,
                   linewidth=2,
                   markersize=6,
                   color='steelblue')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved line plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating line plot: {str(e)}")
        return None


def create_bar_chart(
    categories: Union[list, np.ndarray],
    values: Union[np.ndarray, pd.Series, list],
    title: str = "Bar Chart",
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    horizontal: bool = False,
    color: str = 'steelblue',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart (vertical or horizontal).
    
    Args:
        categories: Category names
        values: Values for each category
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        horizontal: If True, create horizontal bars
        color: Bar color
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        categories = list(categories)
        values = np.array(values)
        
        if horizontal:
            ax.barh(categories, values, color=color, edgecolor='black', linewidth=0.7)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel(xlabel, fontsize=12)
        else:
            ax.bar(categories, values, color=color, edgecolor='black', linewidth=0.7)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            
            # Rotate labels if many categories
            if len(categories) > 10:
                plt.xticks(rotation=45, ha='right')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y' if not horizontal else 'x')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved bar chart to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating bar chart: {str(e)}")
        return None


def create_histogram(
    data: Union[np.ndarray, pd.Series, list],
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    kde: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a histogram with optional KDE overlay.
    
    Args:
        data: Data to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        figsize: Figure size
        kde: Show kernel density estimate
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        data = np.array(data)
        data = data[~np.isnan(data)]  # Remove NaN values
        
        if len(data) == 0:
            print("   ✗ No valid data for histogram")
            return None
        
        # Create histogram
        ax.hist(data, bins=bins, color='steelblue', 
               edgecolor='black', alpha=0.7, density=kde)
        
        # Add KDE if requested
        if kde:
            ax2 = ax.twinx()
            sns.kdeplot(data, ax=ax2, color='darkred', linewidth=2, label='KDE')
            ax2.set_ylabel('Density', fontsize=12)
            ax2.legend(loc='upper right')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved histogram to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating histogram: {str(e)}")
        return None


def create_boxplot(
    data: Union[Dict[str, np.ndarray], pd.DataFrame],
    title: str = "Box Plot",
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    horizontal: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create box plots for multiple columns/categories.
    
    Args:
        data: Dictionary of {column_name: values} or DataFrame
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        horizontal: If True, create horizontal boxplots
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(data, pd.DataFrame):
            data_to_plot = [data[col].dropna() for col in data.columns]
            labels = data.columns
        elif isinstance(data, dict):
            data_to_plot = [np.array(v)[~np.isnan(np.array(v))] for v in data.values()]
            labels = list(data.keys())
        else:
            raise ValueError("Data must be DataFrame or dict")
        
        bp = ax.boxplot(data_to_plot, 
                       labels=labels,
                       vert=not horizontal,
                       patch_artist=True,
                       notch=True,
                       showmeans=True)
        
        # Styling
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5, color='gray')
        
        for cap in bp['caps']:
            cap.set(linewidth=1.5, color='gray')
        
        for median in bp['medians']:
            median.set(linewidth=2, color='darkred')
        
        for mean in bp['means']:
            mean.set(marker='D', markerfacecolor='green', markersize=6)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        if horizontal:
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel(xlabel, fontsize=12)
        else:
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            if len(labels) > 8:
                plt.xticks(rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y' if not horizontal else 'x')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved boxplot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating boxplot: {str(e)}")
        return None


# ============================================================================
# STATISTICAL PLOTS
# ============================================================================

def create_correlation_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[List[str]] = None,
    title: str = "Correlation Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    cmap: str = 'RdBu_r',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a correlation heatmap with annotations.
    
    Args:
        data: DataFrame or correlation matrix
        columns: Column names (if data is np.ndarray)
        title: Plot title
        figsize: Figure size
        annot: Show correlation values as annotations
        cmap: Colormap (diverging, centered at 0)
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = create_correlation_heatmap(df[numeric_cols])
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate correlation if DataFrame
        if isinstance(data, pd.DataFrame):
            corr_matrix = data.corr()
        else:
            corr_matrix = pd.DataFrame(data, columns=columns, index=columns)
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(corr_matrix,
                   mask=mask,
                   annot=annot,
                   fmt='.2f',
                   cmap=cmap,
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                   ax=ax,
                   vmin=-1,
                   vmax=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved correlation heatmap to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating correlation heatmap: {str(e)}")
        return None


def create_distribution_plot(
    data: Union[np.ndarray, pd.Series, list],
    title: str = "Distribution Plot",
    xlabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    show_rug: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a distribution plot with histogram and KDE.
    
    Args:
        data: Data to plot
        title: Plot title
        xlabel: X-axis label
        figsize: Figure size
        show_rug: Show rug plot (data points on x-axis)
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            print("   ✗ No valid data for distribution plot")
            return None
        
        # Create distribution plot
        sns.histplot(data, kde=True, ax=ax, color='steelblue', 
                    edgecolor='black', alpha=0.6, bins=30)
        
        if show_rug:
            sns.rugplot(data, ax=ax, color='darkred', alpha=0.5, height=0.05)
        
        # Add statistics text
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        
        stats_text = f'Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontsize=10)
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label='Median')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency / Density', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved distribution plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating distribution plot: {str(e)}")
        return None


def create_violin_plot(
    data: Union[Dict[str, np.ndarray], pd.DataFrame],
    title: str = "Violin Plot",
    xlabel: str = "Category",
    ylabel: str = "Value",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create violin plots showing distribution for multiple categories.
    
    Args:
        data: Dictionary or DataFrame with categories
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(data, dict):
            # Convert dict to DataFrame for seaborn
            df_list = []
            for key, values in data.items():
                df_list.append(pd.DataFrame({
                    'Category': [key] * len(values),
                    'Value': values
                }))
            plot_df = pd.concat(df_list, ignore_index=True)
        else:
            plot_df = data
        
        # Create violin plot
        sns.violinplot(data=plot_df, x='Category', y='Value', ax=ax,
                      palette='Set2', inner='box')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if len(plot_df['Category'].unique()) > 8:
            plt.xticks(rotation=45, ha='right')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved violin plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating violin plot: {str(e)}")
        return None


def create_pairplot(
    data: pd.DataFrame,
    hue: Optional[str] = None,
    title: str = "Pair Plot",
    figsize: Tuple[int, int] = (12, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a pairplot (scatterplot matrix) for multiple features.
    
    Args:
        data: DataFrame with features to plot
        hue: Column name for color coding
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Seaborn pairplot returns a PairGrid, we need to extract the figure
        if hue and hue in data.columns:
            pair_grid = sns.pairplot(data, hue=hue, palette='Set2',
                                    diag_kind='kde', corner=True)
        else:
            pair_grid = sns.pairplot(data, palette='Set2',
                                    diag_kind='kde', corner=True)
        
        fig = pair_grid.fig
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved pairplot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating pairplot: {str(e)}")
        return None


# ============================================================================
# MACHINE LEARNING PLOTS
# ============================================================================

def create_roc_curve(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curve Comparison",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create ROC curves for multiple models on the same plot.
    
    Args:
        models_data: Dict of {model_name: (fpr, tpr, auc_score)}
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from sklearn.metrics import roc_curve, auc
        >>> fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        >>> auc_score = auc(fpr, tpr)
        >>> models = {'Random Forest': (fpr, tpr, auc_score)}
        >>> fig = create_roc_curve(models)
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette('husl', n_colors=len(models_data))
        
        for i, (model_name, (fpr, tpr, auc_score)) in enumerate(models_data.items()):
            ax.plot(fpr, tpr, 
                   linewidth=2.5,
                   label=f'{model_name} (AUC = {auc_score:.3f})',
                   color=colors[i])
        
        # Add diagonal reference line (random classifier)
        ax.plot([0, 1], [0, 1], 
               linestyle='--', 
               linewidth=2,
               color='gray',
               label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved ROC curve to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating ROC curve: {str(e)}")
        return None


def create_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    show_percentages: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a confusion matrix heatmap with annotations.
    
    Args:
        cm: Confusion matrix (from sklearn.metrics.confusion_matrix)
        class_names: Names of classes (optional)
        title: Plot title
        figsize: Figure size
        show_percentages: Show percentages in addition to counts
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> fig = create_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Normalize for percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations
        if show_percentages:
            annotations = np.array([[f'{count}\n({percent:.1f}%)' 
                                   for count, percent in zip(row_counts, row_percents)]
                                  for row_counts, row_percents in zip(cm, cm_percent)])
        else:
            annotations = cm
        
        # Create heatmap
        sns.heatmap(cm,
                   annot=annotations,
                   fmt='',
                   cmap='Blues',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={'label': 'Count'},
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved confusion matrix to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating confusion matrix: {str(e)}")
        return None


def create_precision_recall_curve(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "Precision-Recall Curve",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create precision-recall curves for multiple models.
    
    Args:
        models_data: Dict of {model_name: (precision, recall, avg_precision)}
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = sns.color_palette('husl', n_colors=len(models_data))
        
        for i, (model_name, (precision, recall, avg_precision)) in enumerate(models_data.items()):
            ax.plot(recall, precision,
                   linewidth=2.5,
                   label=f'{model_name} (AP = {avg_precision:.3f})',
                   color=colors[i])
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved precision-recall curve to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating precision-recall curve: {str(e)}")
        return None


def create_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importances.
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        title: Plot title
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> importances = model.feature_importances_
        >>> fig = create_feature_importance(feature_names, importances, top_n=15)
    """
    try:
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Create figure with appropriate height
        height = max(8, top_n * 0.4)
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        
        # Color bars by positive/negative (if any negative values)
        colors = ['green' if x >= 0 else 'red' for x in sorted_importances]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importances, color=colors, edgecolor='black', linewidth=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.invert_yaxis()  # Top features at top
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved feature importance to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating feature importance plot: {str(e)}")
        return None


def create_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a residual plot (Predicted vs Actual) for regression models.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        
        residuals = y_true - y_pred
        
        # Plot 1: Predicted vs Actual
        ax1.scatter(y_true, y_pred, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title('Predicted vs Actual', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Residuals vs Predicted
        ax2.scatter(y_pred, residuals, alpha=0.5, s=50, 
                   color='steelblue', edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        ax2.set_xlabel('Predicted Values', fontsize=12)
        ax2.set_ylabel('Residuals', fontsize=12)
        ax2.set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved residual plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating residual plot: {str(e)}")
        return None


def create_learning_curve(
    train_sizes: np.ndarray,
    train_scores_mean: np.ndarray,
    train_scores_std: np.ndarray,
    val_scores_mean: np.ndarray,
    val_scores_std: np.ndarray,
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a learning curve showing training and validation scores.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores_mean: Mean training scores
        train_scores_std: Std of training scores
        val_scores_mean: Mean validation scores
        val_scores_std: Std of validation scores
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color='blue',
               linewidth=2, markersize=8, label='Training Score')
        ax.fill_between(train_sizes,
                       train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std,
                       alpha=0.2, color='blue')
        
        # Plot validation scores
        ax.plot(train_sizes, val_scores_mean, 'o-', color='orange',
               linewidth=2, markersize=8, label='Validation Score')
        ax.fill_between(train_sizes,
                       val_scores_mean - val_scores_std,
                       val_scores_mean + val_scores_std,
                       alpha=0.2, color='orange')
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved learning curve to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating learning curve: {str(e)}")
        return None


# ============================================================================
# DATA QUALITY PLOTS
# ============================================================================

def create_missing_values_heatmap(
    df: pd.DataFrame,
    title: str = "Missing Values Heatmap",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a heatmap showing missing values pattern.
    
    Args:
        df: DataFrame to analyze
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create binary matrix (1 = missing, 0 = present)
        missing_matrix = df.isnull().astype(int)
        
        # Plot heatmap
        sns.heatmap(missing_matrix.T,
                   cbar=False,
                   cmap='RdYlGn_r',
                   ax=ax,
                   yticklabels=df.columns)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved missing values heatmap to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating missing values heatmap: {str(e)}")
        return None


def create_missing_values_bar(
    df: pd.DataFrame,
    title: str = "Missing Values by Column",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart showing percentage of missing values per column.
    
    Args:
        df: DataFrame to analyze
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Calculate missing percentages
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]  # Only columns with missing values
        
        if len(missing_pct) == 0:
            print("   ℹ No missing values found")
            return None
        
        height = max(6, len(missing_pct) * 0.3)
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        
        # Create horizontal bar chart
        colors = plt.cm.Reds(missing_pct / 100)
        ax.barh(range(len(missing_pct)), missing_pct.values,
               color=colors, edgecolor='black', linewidth=0.7)
        
        ax.set_yticks(range(len(missing_pct)))
        ax.set_yticklabels(missing_pct.index)
        ax.set_xlabel('Missing Values (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Add percentage labels
        for i, v in enumerate(missing_pct.values):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved missing values bar chart to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating missing values bar chart: {str(e)}")
        return None


def create_outlier_detection_boxplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Outlier Detection",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create box plots for outlier detection across multiple columns.
    
    Args:
        df: DataFrame with numeric columns
        columns: Columns to plot (None = all numeric)
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        return create_boxplot(df[columns], title=title, figsize=figsize, save_path=save_path)
    
    except Exception as e:
        print(f"   ✗ Error creating outlier detection plot: {str(e)}")
        return None


def create_skewness_plot(
    df: pd.DataFrame,
    title: str = "Feature Skewness Distribution",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a bar chart showing skewness of numeric features.
    
    Args:
        df: DataFrame with numeric columns
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Calculate skewness for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skewness = df[numeric_cols].skew().sort_values(ascending=False)
        
        if len(skewness) == 0:
            print("   ℹ No numeric columns to analyze")
            return None
        
        height = max(6, len(skewness) * 0.3)
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        
        # Color by skewness level
        colors = ['green' if abs(x) < 0.5 else 'orange' if abs(x) < 1 else 'red' 
                 for x in skewness.values]
        
        ax.barh(range(len(skewness)), skewness.values,
               color=colors, edgecolor='black', linewidth=0.7)
        
        ax.set_yticks(range(len(skewness)))
        ax.set_yticklabels(skewness.index)
        ax.set_xlabel('Skewness', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=-0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Low (|skew| < 0.5)'),
            Patch(facecolor='orange', label='Moderate (0.5 ≤ |skew| < 1)'),
            Patch(facecolor='red', label='High (|skew| ≥ 1)')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved skewness plot to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating skewness plot: {str(e)}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_figure(fig: plt.Figure, path: str, dpi: int = 300) -> None:
    """
    Save a matplotlib figure to file.
    
    Args:
        fig: Matplotlib Figure object
        path: Output file path (supports .png, .jpg, .pdf, .svg)
        dpi: Resolution (dots per inch)
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved figure to {path}")
    except Exception as e:
        print(f"   ✗ Error saving figure: {str(e)}")


def close_figure(fig: plt.Figure) -> None:
    """
    Close a matplotlib figure to free memory.
    
    Args:
        fig: Matplotlib Figure object
    """
    if fig is not None:
        plt.close(fig)


def create_subplots_grid(
    plot_data: List[Dict[str, Any]],
    rows: int,
    cols: int,
    figsize: Tuple[int, int] = (15, 12),
    title: str = "Plot Grid",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid of subplots.
    
    Args:
        plot_data: List of dicts with plot specifications
        rows: Number of rows
        cols: Number of columns
        figsize: Figure size
        title: Overall title
        save_path: Optional save path
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> plots = [
        ...     {'type': 'scatter', 'x': x1, 'y': y1, 'title': 'Plot 1'},
        ...     {'type': 'hist', 'data': data1, 'title': 'Plot 2'}
        ... ]
        >>> fig = create_subplots_grid(plots, 2, 2)
    """
    try:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if rows * cols > 1 else [axes]
        
        for i, (ax, plot_spec) in enumerate(zip(axes, plot_data)):
            plot_type = plot_spec.get('type', 'scatter')
            
            if plot_type == 'scatter':
                ax.scatter(plot_spec['x'], plot_spec['y'], alpha=0.6)
            elif plot_type == 'hist':
                ax.hist(plot_spec['data'], bins=30, edgecolor='black')
            elif plot_type == 'line':
                ax.plot(plot_spec['x'], plot_spec['y'])
            
            ax.set_title(plot_spec.get('title', f'Subplot {i+1}'), fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(plot_data), len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   ✓ Saved subplot grid to {save_path}")
        
        return fig
    
    except Exception as e:
        print(f"   ✗ Error creating subplot grid: {str(e)}")
        return None
