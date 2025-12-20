"""
Advanced Analysis Tools
Tools for EDA, model diagnostics, anomaly detection, multicollinearity, and statistical tests.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
import warnings
import json

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import learning_curve
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind, pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from utils.polars_helpers import (
    load_dataframe, get_numeric_columns, get_categorical_columns
)
from utils.validation import (
    validate_file_exists, validate_file_format, validate_dataframe,
    validate_column_exists
)


def perform_eda_analysis(
    file_path: str,
    target_col: Optional[str] = None,
    output_html: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform comprehensive automated Exploratory Data Analysis with interactive visualizations.
    
    Args:
        file_path: Path to dataset
        target_col: Target column for supervised analysis
        output_html: Path to save HTML report
        
    Returns:
        Dictionary with EDA insights and statistics
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if target_col:
        validate_column_exists(df, target_col)
    
    print("📊 Performing comprehensive EDA...")
    
    # Basic statistics
    n_rows, n_cols = df.shape
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Missing values analysis
    missing_stats = {}
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            missing_stats[col] = {
                'count': null_count,
                'percentage': float(null_count / n_rows * 100)
            }
    
    # Univariate analysis for numeric columns
    numeric_stats = {}
    for col in numeric_cols[:20]:  # Limit to 20 columns
        col_data = df[col].drop_nulls().to_numpy()
        numeric_stats[col] = {
            'mean': float(np.mean(col_data)),
            'median': float(np.median(col_data)),
            'std': float(np.std(col_data)),
            'min': float(np.min(col_data)),
            'max': float(np.max(col_data)),
            'q25': float(np.percentile(col_data, 25)),
            'q75': float(np.percentile(col_data, 75)),
            'skewness': float(stats.skew(col_data)),
            'kurtosis': float(stats.kurtosis(col_data))
        }
    
    # Categorical analysis
    categorical_stats = {}
    for col in categorical_cols[:10]:  # Limit to 10 columns
        value_counts = df[col].value_counts().head(10)
        categorical_stats[col] = {
            'unique_values': df[col].n_unique(),
            'mode': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
            'top_10_values': {str(row[col]): row['count'] for row in value_counts.to_dicts()}
        }
    
    # Correlation analysis (numeric only)
    correlations = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols[:20]].to_pandas().corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        correlations['high_correlations'] = high_corr_pairs
        correlations['correlation_matrix_shape'] = corr_matrix.shape
    
    # Target analysis
    target_insights = {}
    if target_col:
        if target_col in numeric_cols:
            # Numeric target - regression
            target_data = df[target_col].drop_nulls().to_numpy()
            target_insights = {
                'type': 'regression',
                'mean': float(np.mean(target_data)),
                'std': float(np.std(target_data)),
                'min': float(np.min(target_data)),
                'max': float(np.max(target_data))
            }
            
            # Feature-target correlations
            target_corr = {}
            for col in numeric_cols:
                if col != target_col:
                    try:
                        corr, pval = pearsonr(
                            df[col].drop_nulls().to_numpy(),
                            df[target_col].drop_nulls().to_numpy()
                        )
                        if abs(corr) > 0.3:
                            target_corr[col] = {
                                'correlation': float(corr),
                                'p_value': float(pval)
                            }
                    except:
                        pass
            target_insights['correlated_features'] = target_corr
            
        else:
            # Categorical target - classification
            value_counts = df[target_col].value_counts()
            target_insights = {
                'type': 'classification',
                'classes': len(value_counts),
                'distribution': {str(row[target_col]): row['count'] for row in value_counts.to_dicts()},
                'imbalance_ratio': float(value_counts['count'].max() / value_counts['count'].min())
            }
    
    # Create visualizations if output_html requested
    if output_html:
        print("📈 Generating interactive visualizations...")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Distribution of Numeric Features', 'Missing Values',
                          'Correlation Heatmap', 'Target Distribution',
                          'Outliers Detection', 'Feature Importance')
        )
        
        # Distribution plot (first numeric column)
        if numeric_cols:
            col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=df[col].to_list(), name=col),
                row=1, col=1
            )
        
        # Missing values plot
        if missing_stats:
            missing_cols = list(missing_stats.keys())[:10]
            missing_pcts = [missing_stats[col]['percentage'] for col in missing_cols]
            fig.add_trace(
                go.Bar(x=missing_cols, y=missing_pcts, name='Missing %'),
                row=1, col=2
            )
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix_np = corr_matrix.values
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix_np,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.columns.tolist(),
                    colorscale='RdBu'
                ),
                row=2, col=1
            )
        
        # Target distribution
        if target_col and target_col in categorical_cols:
            target_counts = df[target_col].value_counts()
            fig.add_trace(
                go.Bar(
                    x=[str(row[target_col]) for row in target_counts.to_dicts()],
                    y=[row['count'] for row in target_counts.to_dicts()],
                    name='Target'
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Automated EDA Report")
        
        # Save HTML
        os.makedirs(os.path.dirname(output_html) if os.path.dirname(output_html) else '.', exist_ok=True)
        fig.write_html(output_html)
        print(f"💾 EDA report saved to: {output_html}")
    
    return {
        'status': 'success',
        'dataset_shape': {'rows': n_rows, 'columns': n_cols},
        'column_types': {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols)
        },
        'missing_values': missing_stats,
        'numeric_statistics': numeric_stats,
        'categorical_statistics': categorical_stats,
        'correlations': correlations,
        'target_insights': target_insights,
        'output_html': output_html
    }


def detect_model_issues(
    model_path: str,
    train_data_path: str,
    test_data_path: str,
    target_col: str
) -> Dict[str, Any]:
    """
    Detect overfitting, underfitting, and other model issues using learning curves and diagnostics.
    
    Args:
        model_path: Path to trained model (.pkl)
        train_data_path: Path to training dataset
        test_data_path: Path to test dataset
        target_col: Target column name
        
    Returns:
        Dictionary with model diagnostics
    """
    import joblib
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    
    # Validation
    validate_file_exists(model_path)
    validate_file_exists(train_data_path)
    validate_file_exists(test_data_path)
    
    # Load model
    model = joblib.load(model_path)
    
    # Load data
    train_df = load_dataframe(train_data_path)
    test_df = load_dataframe(test_data_path)
    
    validate_column_exists(train_df, target_col)
    validate_column_exists(test_df, target_col)
    
    # Prepare data
    from utils.polars_helpers import split_features_target
    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)
    
    print("🔍 Analyzing model performance...")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Detect task type
    unique_values = len(np.unique(y_train))
    task_type = "classification" if unique_values < 20 else "regression"
    
    # Calculate metrics
    if task_type == "classification":
        train_score = accuracy_score(y_train, y_train_pred)
        test_score = accuracy_score(y_test, y_test_pred)
        metric_name = "accuracy"
    else:
        train_score = r2_score(y_train, y_train_pred)
        test_score = r2_score(y_test, y_test_pred)
        metric_name = "r2"
    
    # Diagnose issues
    score_gap = train_score - test_score
    
    diagnosis = []
    if score_gap > 0.15:
        diagnosis.append({
            'issue': 'overfitting',
            'severity': 'high' if score_gap > 0.25 else 'medium',
            'description': f'Training {metric_name} ({train_score:.3f}) is much higher than test {metric_name} ({test_score:.3f})',
            'recommendations': [
                'Add regularization (L1/L2)',
                'Reduce model complexity',
                'Increase training data',
                'Use cross-validation',
                'Add dropout (for neural networks)'
            ]
        })
    
    if test_score < 0.6 and task_type == "classification":
        diagnosis.append({
            'issue': 'underfitting',
            'severity': 'high',
            'description': f'Test accuracy ({test_score:.3f}) is too low',
            'recommendations': [
                'Increase model complexity',
                'Engineer better features',
                'Try ensemble methods',
                'Tune hyperparameters',
                'Check for data quality issues'
            ]
        })
    
    if test_score < 0.3 and task_type == "regression":
        diagnosis.append({
            'issue': 'underfitting',
            'severity': 'high',
            'description': f'Test R² ({test_score:.3f}) is too low',
            'recommendations': [
                'Increase model complexity',
                'Engineer better features',
                'Try non-linear models',
                'Check for data scaling issues'
            ]
        })
    
    # Bias-variance analysis
    if abs(score_gap) < 0.05:
        bias_variance = 'balanced'
    elif score_gap > 0.15:
        bias_variance = 'high_variance'  # Overfitting
    else:
        bias_variance = 'high_bias'  # Underfitting
    
    # Generate learning curve data
    print("📊 Generating learning curve...")
    try:
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy' if task_type == "classification" else 'r2',
            n_jobs=-1
        )
        
        learning_curve_data = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist()
        }
    except Exception as e:
        learning_curve_data = {'error': str(e)}
    
    return {
        'status': 'success',
        'task_type': task_type,
        'train_score': float(train_score),
        'test_score': float(test_score),
        'score_gap': float(score_gap),
        'bias_variance_assessment': bias_variance,
        'diagnosis': diagnosis,
        'learning_curve': learning_curve_data,
        'summary': f"Model shows {bias_variance} with {len(diagnosis)} issues detected"
    }


def detect_anomalies(
    file_path: str,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    columns: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect anomalies/outliers using various methods.
    
    Args:
        file_path: Path to dataset
        method: Anomaly detection method:
            - 'isolation_forest': Isolation Forest (good for high-dim data)
            - 'lof': Local Outlier Factor
            - 'zscore': Z-score method (univariate)
            - 'iqr': Interquartile Range method (univariate)
        contamination: Expected proportion of outliers (0.01 to 0.5)
        columns: Columns to analyze (None = all numeric)
        output_path: Path to save dataset with anomaly labels
        
    Returns:
        Dictionary with anomaly detection results
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Get numeric columns if not specified
    if columns is None:
        columns = get_numeric_columns(df)
        print(f"🔢 Auto-detected {len(columns)} numeric columns")
    else:
        for col in columns:
            validate_column_exists(df, col)
    
    if not columns:
        return {
            'status': 'skipped',
            'message': 'No numeric columns found for anomaly detection'
        }
    
    X = df[columns].fill_null(0).to_numpy()
    
    print(f"🔍 Detecting anomalies using {method}...")
    
    # Detect anomalies based on method
    if method == "isolation_forest":
        detector = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = detector.fit_predict(X)
        anomaly_scores = detector.score_samples(X)
        anomalies = predictions == -1
        
    elif method == "lof":
        detector = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
        predictions = detector.fit_predict(X)
        anomaly_scores = detector.negative_outlier_factor_
        anomalies = predictions == -1
        
    elif method == "zscore":
        # Z-score for each column
        z_scores = np.abs(stats.zscore(X, axis=0))
        anomalies = (z_scores > 3).any(axis=1)
        anomaly_scores = z_scores.max(axis=1)
        
    elif method == "iqr":
        # IQR for each column
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomalies = ((X < lower_bound) | (X > upper_bound)).any(axis=1)
        # Calculate how many IQRs away from bounds
        dist_from_bounds = np.maximum(
            (lower_bound - X) / IQR,
            (X - upper_bound) / IQR
        ).max(axis=1)
        anomaly_scores = dist_from_bounds
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Count anomalies
    n_anomalies = int(anomalies.sum())
    anomaly_percentage = float(n_anomalies / len(df) * 100)
    
    print(f"🚨 Found {n_anomalies} anomalies ({anomaly_percentage:.2f}%)")
    
    # Add anomaly labels to dataframe
    df_with_anomalies = df.with_columns([
        pl.Series('is_anomaly', anomalies.astype(int)),
        pl.Series('anomaly_score', anomaly_scores)
    ])
    
    # Get indices of anomalies
    anomaly_indices = np.where(anomalies)[0].tolist()
    
    # Analyze anomalies by column
    column_anomaly_stats = {}
    for col in columns:
        col_data = df[col].to_numpy()
        anomaly_values = col_data[anomalies]
        
        if len(anomaly_values) > 0:
            column_anomaly_stats[col] = {
                'mean_normal': float(np.mean(col_data[~anomalies])),
                'mean_anomaly': float(np.mean(anomaly_values)),
                'std_normal': float(np.std(col_data[~anomalies])),
                'std_anomaly': float(np.std(anomaly_values))
            }
    
    # Save if output path provided
    if output_path:
        from utils.polars_helpers import save_dataframe
        save_dataframe(df_with_anomalies, output_path)
        print(f"💾 Dataset with anomaly labels saved to: {output_path}")
    
    return {
        'status': 'success',
        'method': method,
        'n_anomalies': n_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'anomaly_indices': anomaly_indices[:100],  # First 100
        'column_statistics': column_anomaly_stats,
        'contamination': contamination,
        'output_path': output_path
    }


def detect_and_handle_multicollinearity(
    file_path: str,
    threshold: float = 10.0,
    action: str = "report",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect and optionally handle multicollinearity using VIF (Variance Inflation Factor).
    
    Args:
        file_path: Path to dataset
        threshold: VIF threshold (10 = high multicollinearity, 5 = moderate)
        action: Action to take:
            - 'report': Only report VIF values
            - 'remove': Remove features with VIF > threshold
            - 'recommend': Provide regularization recommendations
        output_path: Path to save dataset with reduced features
        
    Returns:
        Dictionary with VIF values and recommendations
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Get numeric columns
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        return {
            'status': 'skipped',
            'message': 'Need at least 2 numeric columns for multicollinearity analysis'
        }
    
    print(f"🔍 Calculating VIF for {len(numeric_cols)} features...")
    
    # Prepare data
    X = df[numeric_cols].fill_null(0).to_numpy()
    
    # Calculate VIF for each feature
    vif_data = {}
    problematic_features = []
    
    for i, col in enumerate(numeric_cols):
        try:
            vif = variance_inflation_factor(X, i)
            vif_data[col] = float(vif)
            
            if vif > threshold:
                problematic_features.append({
                    'feature': col,
                    'vif': float(vif),
                    'severity': 'high' if vif > 20 else 'moderate'
                })
        except Exception as e:
            vif_data[col] = None
            print(f"⚠️ Could not calculate VIF for {col}: {str(e)}")
    
    # Sort by VIF
    sorted_vif = dict(sorted(vif_data.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True))
    
    # Generate recommendations
    recommendations = []
    
    if len(problematic_features) > 0:
        recommendations.append({
            'type': 'regularization',
            'description': 'Use Ridge (L2) or Elastic Net regularization to handle multicollinearity',
            'reason': f'{len(problematic_features)} features have VIF > {threshold}'
        })
        
        recommendations.append({
            'type': 'pca',
            'description': 'Apply PCA to reduce dimensionality and eliminate correlations',
            'reason': 'PCA creates orthogonal features'
        })
        
        if action == "remove":
            # Remove features with highest VIF iteratively
            features_to_remove = [f['feature'] for f in problematic_features]
            recommendations.append({
                'type': 'feature_removal',
                'description': f'Remove {len(features_to_remove)} features with high VIF',
                'features': features_to_remove
            })
    
    # Handle action
    if action == "remove" and len(problematic_features) > 0:
        # Remove features with VIF > threshold
        features_to_keep = [col for col in numeric_cols if col not in [f['feature'] for f in problematic_features]]
        categorical_cols = get_categorical_columns(df)
        
        df_reduced = df.select(features_to_keep + categorical_cols)
        
        if output_path:
            from utils.polars_helpers import save_dataframe
            save_dataframe(df_reduced, output_path)
            print(f"💾 Dataset with reduced features saved to: {output_path}")
        
        return {
            'status': 'success',
            'action': 'removed',
            'vif_values': sorted_vif,
            'problematic_features': problematic_features,
            'features_removed': len(problematic_features),
            'features_remaining': len(features_to_keep),
            'recommendations': recommendations,
            'output_path': output_path
        }
    
    return {
        'status': 'success',
        'action': action,
        'vif_values': sorted_vif,
        'problematic_features': problematic_features,
        'threshold': threshold,
        'recommendations': recommendations
    }


def perform_statistical_tests(
    file_path: str,
    target_col: str,
    test_type: str = "auto",
    features: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical hypothesis tests to validate feature relationships.
    
    Args:
        file_path: Path to dataset
        target_col: Target column name
        test_type: Type of test:
            - 'auto': Automatically select based on data types
            - 'chi2': Chi-square test (categorical vs categorical)
            - 'ttest': T-test (binary categorical vs numeric)
            - 'anova': ANOVA (multi-class categorical vs numeric)
            - 'pearson': Pearson correlation test (numeric vs numeric)
        features: Features to test (None = all)
        alpha: Significance level (default 0.05)
        
    Returns:
        Dictionary with test results and p-values
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Get column types
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Determine target type
    target_is_numeric = target_col in numeric_cols
    target_is_categorical = target_col in categorical_cols
    
    # Get features to test
    if features is None:
        features = [col for col in df.columns if col != target_col]
    
    print(f"📊 Performing statistical tests for {len(features)} features...")
    
    test_results = []
    
    for feature in features:
        feature_is_numeric = feature in numeric_cols
        feature_is_categorical = feature in categorical_cols
        
        # Skip if feature is target
        if feature == target_col:
            continue
        
        # Select appropriate test
        if test_type == "auto":
            if target_is_numeric and feature_is_numeric:
                selected_test = "pearson"
            elif target_is_categorical and feature_is_numeric:
                target_unique = df[target_col].n_unique()
                selected_test = "ttest" if target_unique == 2 else "anova"
            elif target_is_categorical and feature_is_categorical:
                selected_test = "chi2"
            elif target_is_numeric and feature_is_categorical:
                selected_test = "anova"
            else:
                continue
        else:
            selected_test = test_type
        
        # Perform test
        try:
            if selected_test == "pearson":
                # Pearson correlation
                feature_data = df[feature].drop_nulls().to_numpy()
                target_data = df[target_col].drop_nulls().to_numpy()
                
                # Align lengths
                min_len = min(len(feature_data), len(target_data))
                corr, pval = pearsonr(feature_data[:min_len], target_data[:min_len])
                
                test_results.append({
                    'feature': feature,
                    'test': 'pearson',
                    'statistic': float(corr),
                    'p_value': float(pval),
                    'significant': pval < alpha,
                    'interpretation': f"Correlation: {corr:.3f}"
                })
                
            elif selected_test == "chi2":
                # Chi-square test
                contingency_table = pd.crosstab(
                    df[feature].to_pandas(),
                    df[target_col].to_pandas()
                )
                chi2, pval, dof, expected = chi2_contingency(contingency_table)
                
                test_results.append({
                    'feature': feature,
                    'test': 'chi2',
                    'statistic': float(chi2),
                    'p_value': float(pval),
                    'dof': int(dof),
                    'significant': pval < alpha
                })
                
            elif selected_test == "ttest":
                # T-test
                target_values = df[target_col].unique().to_list()
                if len(target_values) != 2:
                    continue
                
                group1 = df.filter(pl.col(target_col) == target_values[0])[feature].drop_nulls().to_numpy()
                group2 = df.filter(pl.col(target_col) == target_values[1])[feature].drop_nulls().to_numpy()
                
                t_stat, pval = ttest_ind(group1, group2)
                
                test_results.append({
                    'feature': feature,
                    'test': 'ttest',
                    'statistic': float(t_stat),
                    'p_value': float(pval),
                    'significant': pval < alpha,
                    'mean_diff': float(np.mean(group1) - np.mean(group2))
                })
                
            elif selected_test == "anova":
                # ANOVA
                groups = []
                target_values = df[target_col].unique().to_list()
                
                for val in target_values:
                    group_data = df.filter(pl.col(target_col) == val)[feature].drop_nulls().to_numpy()
                    if len(group_data) > 0:
                        groups.append(group_data)
                
                if len(groups) > 1:
                    f_stat, pval = f_oneway(*groups)
                    
                    test_results.append({
                        'feature': feature,
                        'test': 'anova',
                        'statistic': float(f_stat),
                        'p_value': float(pval),
                        'significant': pval < alpha,
                        'n_groups': len(groups)
                    })
        
        except Exception as e:
            print(f"⚠️ Test failed for {feature}: {str(e)}")
    
    # Summary
    significant_features = [r for r in test_results if r['significant']]
    
    print(f"✅ {len(significant_features)}/{len(test_results)} features are statistically significant (α={alpha})")
    
    return {
        'status': 'success',
        'target_column': target_col,
        'alpha': alpha,
        'total_tests': len(test_results),
        'significant_features': len(significant_features),
        'test_results': test_results,
        'significant_features_list': [r['feature'] for r in significant_features]
    }
