"""
Production & MLOps Tools
Tools for model monitoring, explainability, governance, and production readiness.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
import json
import warnings
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats
from scipy.stats import ks_2samp, pearsonr
import shap
from lime import lime_tabular
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.polars_helpers import load_dataframe, get_numeric_columns, split_features_target
from utils.validation import validate_file_exists, validate_file_format, validate_dataframe, validate_column_exists


def monitor_model_drift(
    reference_data_path: str,
    current_data_path: str,
    target_col: Optional[str] = None,
    threshold_psi: float = 0.2,
    threshold_ks: float = 0.05,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect data drift and concept drift in production models.
    
    Args:
        reference_data_path: Path to training/reference dataset
        current_data_path: Path to production/current dataset
        target_col: Target column (for concept drift detection)
        threshold_psi: PSI threshold (>0.2 = significant drift)
        threshold_ks: KS test p-value threshold (<0.05 = significant drift)
        output_path: Path to save drift report
        
    Returns:
        Dictionary with drift metrics and alerts
    """
    # Validation
    validate_file_exists(reference_data_path)
    validate_file_exists(current_data_path)
    
    # Load data
    ref_df = load_dataframe(reference_data_path)
    curr_df = load_dataframe(current_data_path)
    
    validate_dataframe(ref_df)
    validate_dataframe(curr_df)
    
    print("🔍 Analyzing data drift...")
    
    # Get common columns
    common_cols = list(set(ref_df.columns) & set(curr_df.columns))
    numeric_cols = [col for col in get_numeric_columns(ref_df) if col in common_cols and col != target_col]
    
    # Calculate PSI (Population Stability Index) for each feature
    drift_results = {}
    alerts = []
    
    for col in numeric_cols:
        try:
            ref_data = ref_df[col].drop_nulls().to_numpy()
            curr_data = curr_df[col].drop_nulls().to_numpy()
            
            # PSI calculation
            # Create bins based on reference data
            bins = np.percentile(ref_data, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            bins = np.unique(bins)  # Remove duplicates
            
            ref_counts, _ = np.histogram(ref_data, bins=bins)
            curr_counts, _ = np.histogram(curr_data, bins=bins)
            
            # Add small constant to avoid division by zero
            ref_props = (ref_counts + 1e-6) / (len(ref_data) + len(bins) * 1e-6)
            curr_props = (curr_counts + 1e-6) / (len(curr_data) + len(bins) * 1e-6)
            
            psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
            
            # KS test (Kolmogorov-Smirnov)
            ks_stat, ks_pval = ks_2samp(ref_data, curr_data)
            
            # Distribution statistics
            ref_mean = float(np.mean(ref_data))
            curr_mean = float(np.mean(curr_data))
            mean_shift = float(abs(curr_mean - ref_mean) / (ref_mean + 1e-10))
            
            drift_results[col] = {
                'psi': float(psi),
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'ref_mean': ref_mean,
                'curr_mean': curr_mean,
                'mean_shift_pct': mean_shift * 100,
                'drift_detected': psi > threshold_psi or ks_pval < threshold_ks
            }
            
            # Generate alerts
            if psi > threshold_psi:
                alerts.append({
                    'feature': col,
                    'type': 'data_drift',
                    'severity': 'high' if psi > 0.5 else 'medium',
                    'metric': 'PSI',
                    'value': float(psi),
                    'message': f"PSI = {psi:.3f} exceeds threshold {threshold_psi}"
                })
            
            if ks_pval < threshold_ks:
                alerts.append({
                    'feature': col,
                    'type': 'data_drift',
                    'severity': 'high',
                    'metric': 'KS_test',
                    'value': float(ks_pval),
                    'message': f"KS test p-value = {ks_pval:.4f} < {threshold_ks}"
                })
                
        except Exception as e:
            print(f"⚠️ Could not calculate drift for {col}: {str(e)}")
    
    # Concept drift (target distribution change)
    concept_drift_result = None
    if target_col and target_col in common_cols:
        try:
            ref_target = ref_df[target_col].drop_nulls().to_numpy()
            curr_target = curr_df[target_col].drop_nulls().to_numpy()
            
            # Check if categorical
            if len(np.unique(ref_target)) < 20:
                # Categorical target - compare distributions
                ref_dist = {str(val): np.sum(ref_target == val) / len(ref_target) for val in np.unique(ref_target)}
                curr_dist = {str(val): np.sum(curr_target == val) / len(curr_target) for val in np.unique(curr_target)}
                
                concept_drift_result = {
                    'ref_distribution': ref_dist,
                    'curr_distribution': curr_dist,
                    'drift_detected': True if len(set(ref_dist.keys()) - set(curr_dist.keys())) > 0 else False
                }
            else:
                # Numeric target
                ks_stat, ks_pval = ks_2samp(ref_target, curr_target)
                concept_drift_result = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pval),
                    'drift_detected': ks_pval < threshold_ks
                }
                
            if concept_drift_result['drift_detected']:
                alerts.append({
                    'feature': target_col,
                    'type': 'concept_drift',
                    'severity': 'critical',
                    'message': 'Target distribution has changed - model may need retraining'
                })
        except Exception as e:
            print(f"⚠️ Could not detect concept drift: {str(e)}")
    
    # Summary
    drifted_features = [col for col, result in drift_results.items() if result['drift_detected']]
    
    print(f"🚨 {len(alerts)} drift alerts | {len(drifted_features)} features with significant drift")
    
    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'reference_samples': len(ref_df),
        'current_samples': len(curr_df),
        'features_analyzed': len(numeric_cols),
        'drift_results': drift_results,
        'concept_drift': concept_drift_result,
        'alerts': alerts,
        'drifted_features': drifted_features
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"💾 Drift report saved to: {output_path}")
    
    return {
        'status': 'success',
        'features_analyzed': len(numeric_cols),
        'drifted_features': drifted_features,
        'n_alerts': len(alerts),
        'alerts': alerts,
        'concept_drift_detected': concept_drift_result['drift_detected'] if concept_drift_result else False,
        'recommendation': 'Retrain model' if len(alerts) > 0 else 'No action needed',
        'report_path': output_path
    }


def explain_predictions(
    model_path: str,
    data_path: str,
    instance_indices: List[int],
    method: str = "shap",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate explainability reports for individual predictions using SHAP or LIME.
    
    Args:
        model_path: Path to trained model (.pkl)
        data_path: Path to dataset
        instance_indices: List of row indices to explain
        method: Explanation method ('shap', 'lime', or 'both')
        output_dir: Directory to save explanation plots
        
    Returns:
        Dictionary with explanations and feature importance
    """
    # Validation
    validate_file_exists(model_path)
    validate_file_exists(data_path)
    
    # Load model and data
    model = joblib.load(model_path)
    df = load_dataframe(data_path)
    validate_dataframe(df)
    
    print(f"🔍 Generating {method} explanations for {len(instance_indices)} instances...")
    
    X = df.to_numpy()
    feature_names = df.columns
    
    explanations = []
    
    # SHAP explanations
    if method in ["shap", "both"]:
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X[instance_indices])
            
            for idx, instance_idx in enumerate(instance_indices):
                shap_exp = {
                    'instance_idx': instance_idx,
                    'method': 'shap',
                    'prediction': model.predict(X[instance_idx:instance_idx+1])[0],
                    'feature_contributions': {
                        feature_names[i]: float(shap_values.values[idx, i])
                        for i in range(len(feature_names))
                    },
                    'top_5_positive': sorted(
                        [(feature_names[i], float(shap_values.values[idx, i])) 
                         for i in range(len(feature_names))],
                        key=lambda x: x[1], reverse=True
                    )[:5],
                    'top_5_negative': sorted(
                        [(feature_names[i], float(shap_values.values[idx, i])) 
                         for i in range(len(feature_names))],
                        key=lambda x: x[1]
                    )[:5]
                }
                explanations.append(shap_exp)
                
            # Save force plot if output_dir provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                for idx, instance_idx in enumerate(instance_indices):
                    plot_path = os.path.join(output_dir, f"shap_force_plot_instance_{instance_idx}.html")
                    shap.save_html(plot_path, shap.force_plot(
                        explainer.expected_value,
                        shap_values.values[idx],
                        X[instance_idx],
                        feature_names=feature_names
                    ))
                print(f"💾 SHAP plots saved to: {output_dir}")
                
        except Exception as e:
            print(f"⚠️ SHAP failed: {str(e)}")
    
    # LIME explanations
    if method in ["lime", "both"]:
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X,
                feature_names=feature_names,
                mode='classification' if hasattr(model, 'predict_proba') else 'regression'
            )
            
            for instance_idx in instance_indices:
                exp = explainer.explain_instance(
                    X[instance_idx],
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=len(feature_names)
                )
                
                lime_exp = {
                    'instance_idx': instance_idx,
                    'method': 'lime',
                    'prediction': model.predict(X[instance_idx:instance_idx+1])[0],
                    'feature_contributions': dict(exp.as_list()),
                    'top_features': exp.as_list()[:10]
                }
                explanations.append(lime_exp)
                
                # Save HTML if output_dir provided
                if output_dir:
                    plot_path = os.path.join(output_dir, f"lime_explanation_instance_{instance_idx}.html")
                    exp.save_to_file(plot_path)
                    
        except Exception as e:
            print(f"⚠️ LIME failed: {str(e)}")
    
    print(f"✅ Generated {len(explanations)} explanations")
    
    return {
        'status': 'success',
        'method': method,
        'n_explanations': len(explanations),
        'explanations': explanations,
        'output_dir': output_dir
    }


def generate_model_card(
    model_path: str,
    train_data_path: str,
    test_data_path: str,
    target_col: str,
    model_name: str,
    model_description: str,
    intended_use: str,
    sensitive_attributes: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive model card for governance and compliance.
    
    Args:
        model_path: Path to trained model
        train_data_path: Path to training data
        test_data_path: Path to test data
        target_col: Target column name
        model_name: Name of the model
        model_description: Description of model architecture
        intended_use: Intended use case
        sensitive_attributes: List of sensitive columns for fairness analysis
        output_path: Path to save model card (JSON/HTML)
        
    Returns:
        Dictionary with model card information
    """
    # Load model and data
    model = joblib.load(model_path)
    train_df = load_dataframe(train_data_path)
    test_df = load_dataframe(test_data_path)
    
    X_train, y_train = split_features_target(train_df, target_col)
    X_test, y_test = split_features_target(test_df, target_col)
    
    print("📋 Generating model card...")
    
    # Model performance
    y_pred = model.predict(X_test)
    
    task_type = "classification" if len(np.unique(y_test)) < 20 else "regression"
    
    if task_type == "classification":
        performance = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        performance = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred))
        }
    
    # Fairness metrics
    fairness_metrics = {}
    if sensitive_attributes:
        for attr in sensitive_attributes:
            if attr in test_df.columns:
                try:
                    groups = test_df[attr].unique().to_list()
                    group_metrics = {}
                    
                    for group in groups:
                        mask = test_df[attr].to_numpy() == group
                        group_pred = y_pred[mask]
                        group_true = y_test[mask]
                        
                        if task_type == "classification":
                            group_metrics[str(group)] = {
                                'accuracy': float(accuracy_score(group_true, group_pred)),
                                'sample_size': int(np.sum(mask))
                            }
                        else:
                            group_metrics[str(group)] = {
                                'rmse': float(np.sqrt(mean_squared_error(group_true, group_pred))),
                                'sample_size': int(np.sum(mask))
                            }
                    
                    fairness_metrics[attr] = group_metrics
                except Exception as e:
                    print(f"⚠️ Could not compute fairness for {attr}: {str(e)}")
    
    # Model card
    model_card = {
        'model_details': {
            'name': model_name,
            'description': model_description,
            'version': '1.0',
            'type': str(type(model).__name__),
            'created_date': datetime.now().isoformat(),
            'intended_use': intended_use
        },
        'training_data': {
            'n_samples': len(train_df),
            'n_features': len(train_df.columns) - 1,
            'target_column': target_col
        },
        'performance': performance,
        'fairness_metrics': fairness_metrics,
        'limitations': [
            f"Trained on {len(train_df)} samples",
            "Performance may degrade on out-of-distribution data",
            "Regular monitoring recommended"
        ],
        'ethical_considerations': [
            "Model should not be used for discriminatory purposes",
            "Predictions should be reviewed by domain experts",
            "Consider societal impact before deployment"
        ]
    }
    
    # Save model card
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        print(f"💾 Model card saved to: {output_path}")
    
    return {
        'status': 'success',
        'model_card': model_card,
        'output_path': output_path
    }


def perform_ab_test_analysis(
    control_data_path: str,
    treatment_data_path: str,
    metric_col: str,
    alpha: float = 0.05,
    power: float = 0.8
) -> Dict[str, Any]:
    """
    Perform A/B test statistical analysis with confidence intervals.
    
    Args:
        control_data_path: Path to control group data
        treatment_data_path: Path to treatment group data
        metric_col: Metric column to compare
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.8)
        
    Returns:
        Dictionary with A/B test results
    """
    # Load data
    control_df = load_dataframe(control_data_path)
    treatment_df = load_dataframe(treatment_data_path)
    
    validate_column_exists(control_df, metric_col)
    validate_column_exists(treatment_df, metric_col)
    
    control = control_df[metric_col].drop_nulls().to_numpy()
    treatment = treatment_df[metric_col].drop_nulls().to_numpy()
    
    print("📊 Performing A/B test analysis...")
    
    # Calculate statistics
    control_mean = float(np.mean(control))
    treatment_mean = float(np.mean(treatment))
    
    control_std = float(np.std(control, ddof=1))
    treatment_std = float(np.std(treatment, ddof=1))
    
    # T-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(treatment, control)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(control)-1)*control_std**2 + (len(treatment)-1)*treatment_std**2) / (len(control)+len(treatment)-2))
    cohens_d = (treatment_mean - control_mean) / pooled_std
    
    # Confidence intervals
    from scipy import stats as scipy_stats
    control_ci = scipy_stats.t.interval(1-alpha, len(control)-1, loc=control_mean, scale=control_std/np.sqrt(len(control)))
    treatment_ci = scipy_stats.t.interval(1-alpha, len(treatment)-1, loc=treatment_mean, scale=treatment_std/np.sqrt(len(treatment)))
    
    # Relative uplift
    relative_uplift = ((treatment_mean - control_mean) / control_mean) * 100
    
    # Sample size recommendation
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    required_n = 2 * ((z_alpha + z_beta) * pooled_std / (treatment_mean - control_mean + 1e-10))**2
    
    # Statistical significance
    is_significant = p_value < alpha
    
    result = {
        'control_group': {
            'n_samples': len(control),
            'mean': control_mean,
            'std': control_std,
            'ci_95': [float(control_ci[0]), float(control_ci[1])]
        },
        'treatment_group': {
            'n_samples': len(treatment),
            'mean': treatment_mean,
            'std': treatment_std,
            'ci_95': [float(treatment_ci[0]), float(treatment_ci[1])]
        },
        'test_results': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': is_significant,
            'alpha': alpha
        },
        'effect_size': {
            'cohens_d': float(cohens_d),
            'interpretation': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        },
        'business_impact': {
            'absolute_lift': float(treatment_mean - control_mean),
            'relative_lift_pct': float(relative_uplift)
        },
        'sample_size_recommendation': {
            'current_total': len(control) + len(treatment),
            'recommended_per_group': int(required_n),
            'is_sufficient': len(control) >= required_n and len(treatment) >= required_n
        },
        'conclusion': f"Treatment {'significantly' if is_significant else 'does not significantly'} outperform control (p={p_value:.4f})"
    }
    
    print(f"{'✅' if is_significant else '❌'} {result['conclusion']}")
    print(f"📈 Relative lift: {relative_uplift:+.2f}%")
    
    return {
        'status': 'success',
        **result
    }


def detect_feature_leakage(
    data_path: str,
    target_col: str,
    time_col: Optional[str] = None,
    correlation_threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Detect potential feature leakage (target leakage and temporal leakage).
    
    Args:
        data_path: Path to dataset
        target_col: Target column name
        time_col: Time column for temporal leakage detection
        correlation_threshold: Correlation threshold for leakage detection
        
    Returns:
        Dictionary with potential leakage issues
    """
    # Load data
    df = load_dataframe(data_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    print("🔍 Detecting feature leakage...")
    
    # Get numeric columns
    numeric_cols = [col for col in get_numeric_columns(df) if col != target_col]
    
    # Target leakage detection (high correlation with target)
    target_leakage = []
    target_data = df[target_col].drop_nulls().to_numpy()
    
    for col in numeric_cols:
        try:
            col_data = df[col].drop_nulls().to_numpy()
            
            # Align lengths
            min_len = min(len(target_data), len(col_data))
            corr, pval = pearsonr(target_data[:min_len], col_data[:min_len])
            
            if abs(corr) > correlation_threshold:
                target_leakage.append({
                    'feature': col,
                    'correlation': float(corr),
                    'p_value': float(pval),
                    'severity': 'critical' if abs(corr) > 0.99 else 'high',
                    'recommendation': f'Remove or investigate {col} - suspiciously high correlation with target'
                })
        except Exception as e:
            pass
    
    # Temporal leakage detection
    temporal_leakage = []
    if time_col and time_col in df.columns:
        # Check for future information
        # Features that shouldn't be available at prediction time
        potential_future_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['future', 'next', 'after', 'later'])]
        
        if potential_future_cols:
            temporal_leakage.append({
                'features': potential_future_cols,
                'issue': 'potential_future_information',
                'recommendation': 'Verify these features are available at prediction time'
            })
    
    # Check for perfect predictors (100% correlation or zero variance when grouped by target)
    perfect_predictors = []
    for col in numeric_cols:
        try:
            grouped_variance = df.group_by(target_col).agg(pl.col(col).var())
            if (grouped_variance[col].drop_nulls() < 1e-10).all():
                perfect_predictors.append({
                    'feature': col,
                    'issue': 'zero_variance_per_class',
                    'recommendation': f'{col} has zero variance within each target class - likely leakage'
                })
        except:
            pass
    
    # Summary
    total_issues = len(target_leakage) + len(temporal_leakage) + len(perfect_predictors)
    
    print(f"🚨 Found {total_issues} potential leakage issues")
    
    return {
        'status': 'success',
        'target_leakage': target_leakage,
        'temporal_leakage': temporal_leakage,
        'perfect_predictors': perfect_predictors,
        'total_issues': total_issues,
        'recommendation': 'Review and remove suspicious features before training' if total_issues > 0 else 'No obvious leakage detected'
    }
