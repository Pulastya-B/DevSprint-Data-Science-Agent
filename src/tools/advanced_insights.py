"""
Advanced Insights Tools
Tools for root cause analysis, trend detection, anomaly detection, and statistical testing.
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
from scipy import stats
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import load_dataframe, get_numeric_columns
from utils.validation import validate_file_exists, validate_file_format


def analyze_root_cause(file_path: str, 
                       target_col: str,
                       time_col: Optional[str] = None,
                       threshold_drop: float = 0.15) -> Dict[str, Any]:
    """
    Perform root cause analysis to identify why a metric dropped.
    
    Args:
        file_path: Path to dataset
        target_col: Column to analyze (e.g., 'sales')
        time_col: Optional time column for trend analysis
        threshold_drop: Percentage drop to flag as significant (default 15%)
        
    Returns:
        Dictionary with root cause insights
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    
    # Convert to pandas for easier analysis
    df_pd = df.to_pandas()
    
    results = {
        "target_column": target_col,
        "analysis_type": "root_cause",
        "insights": [],
        "correlations": {},
        "top_factors": []
    }
    
    # Check if target exists
    if target_col not in df_pd.columns:
        return {"status": "error", "message": f"Column '{target_col}' not found"}
    
    # Analyze overall trend
    target_mean = df_pd[target_col].mean()
    target_std = df_pd[target_col].std()
    
    # If time column exists, analyze temporal patterns
    if time_col and time_col in df_pd.columns:
        try:
            df_pd[time_col] = pd.to_datetime(df_pd[time_col])
            df_sorted = df_pd.sort_values(time_col)
            
            # Calculate period-over-period changes
            if len(df_sorted) > 10:
                mid_point = len(df_sorted) // 2
                first_half_mean = df_sorted[target_col].iloc[:mid_point].mean()
                second_half_mean = df_sorted[target_col].iloc[mid_point:].mean()
                
                change_pct = ((second_half_mean - first_half_mean) / first_half_mean) * 100
                
                if abs(change_pct) > threshold_drop * 100:
                    insight = f"📉 Significant change detected: {change_pct:+.1f}% between periods"
                    results["insights"].append(insight)
                    results["period_change"] = {
                        "first_period_avg": float(first_half_mean),
                        "second_period_avg": float(second_half_mean),
                        "change_percentage": float(change_pct)
                    }
        except Exception as e:
            results["insights"].append(f"⚠️ Could not analyze time series: {str(e)}")
    
    # Find correlations with target
    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if numeric_cols:
        correlations = {}
        for col in numeric_cols[:20]:  # Limit to top 20 for performance
            try:
                corr = df_pd[target_col].corr(df_pd[col])
                if not np.isnan(corr):
                    correlations[col] = float(corr)
            except:
                pass
        
        # Sort by absolute correlation
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        results["correlations"] = dict(sorted_corrs[:10])
        
        # Identify top factors
        top_factors = []
        for col, corr in sorted_corrs[:5]:
            if abs(corr) > 0.3:
                direction = "positively" if corr > 0 else "negatively"
                top_factors.append({
                    "factor": col,
                    "correlation": float(corr),
                    "description": f"{col} is {direction} correlated ({corr:.3f}) with {target_col}"
                })
        
        results["top_factors"] = top_factors
        
        if top_factors:
            results["insights"].append(f"🔍 Found {len(top_factors)} significant factors influencing {target_col}")
    
    # Outlier detection in target
    Q1 = df_pd[target_col].quantile(0.25)
    Q3 = df_pd[target_col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_pd[(df_pd[target_col] < Q1 - 1.5 * IQR) | (df_pd[target_col] > Q3 + 1.5 * IQR)]
    
    if len(outliers) > 0:
        outlier_pct = (len(outliers) / len(df_pd)) * 100
        results["insights"].append(f"⚠️ {len(outliers)} outliers detected ({outlier_pct:.1f}% of data)")
        results["outlier_count"] = len(outliers)
    
    return results


def detect_trends_and_seasonality(file_path: str,
                                  value_col: str,
                                  time_col: str,
                                  seasonal_period: Optional[int] = None) -> Dict[str, Any]:
    """
    Detect trends and seasonal patterns in time series data.
    
    Args:
        file_path: Path to dataset
        value_col: Column with values to analyze
        time_col: Column with timestamps
        seasonal_period: Expected seasonal period (auto-detected if None)
        
    Returns:
        Dictionary with trend and seasonality insights
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path).to_pandas()
    
    results = {
        "value_column": value_col,
        "time_column": time_col,
        "trend_detected": False,
        "seasonality_detected": False,
        "insights": []
    }
    
    # Validate columns
    if value_col not in df.columns or time_col not in df.columns:
        return {"status": "error", "message": "Columns not found"}
    
    # Convert to datetime and sort
    try:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
    except:
        return {"status": "error", "message": f"Could not parse {time_col} as datetime"}
    
    values = df[value_col].values
    
    # Trend detection using linear regression
    X = np.arange(len(values)).reshape(-1, 1)
    y = values
    
    # Simple linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    if p_value < 0.05:  # Significant trend
        results["trend_detected"] = True
        results["trend_slope"] = float(slope)
        results["trend_r_squared"] = float(r_value ** 2)
        
        direction = "upward" if slope > 0 else "downward"
        results["insights"].append(f"📈 {direction.capitalize()} trend detected (slope: {slope:.4f}, R²: {r_value**2:.3f})")
        results["trend_direction"] = direction
    else:
        results["insights"].append("📊 No significant trend detected")
    
    # Seasonality detection using autocorrelation
    if len(values) > 20:
        from statsmodels.tsa.stattools import acf
        
        try:
            autocorr = acf(values, nlags=min(len(values)//2, 50), fft=True)
            
            # Find peaks in autocorrelation (excluding lag 0)
            peaks, properties = find_peaks(autocorr[1:], height=0.3)
            
            if len(peaks) > 0:
                # Most prominent peak indicates seasonal period
                peak_lag = peaks[np.argmax(properties['peak_heights'])] + 1
                results["seasonality_detected"] = True
                results["seasonal_period"] = int(peak_lag)
                results["insights"].append(f"🔄 Seasonality detected with period of {peak_lag} observations")
            else:
                results["insights"].append("📊 No strong seasonality pattern detected")
        except Exception as e:
            results["insights"].append(f"⚠️ Could not analyze seasonality: {str(e)}")
    
    # Calculate summary statistics
    results["statistics"] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values))
    }
    
    return results


def detect_anomalies_advanced(file_path: str,
                              columns: Optional[List[str]] = None,
                              contamination: float = 0.1,
                              method: str = "isolation_forest") -> Dict[str, Any]:
    """
    Detect anomalies with confidence scores using advanced methods.
    
    Args:
        file_path: Path to dataset
        columns: Columns to analyze (all numeric if None)
        contamination: Expected proportion of outliers
        method: 'isolation_forest' or 'statistical'
        
    Returns:
        Dictionary with anomaly detection results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path)
    df_pd = df.to_pandas()
    
    # Select numeric columns
    if columns is None:
        numeric_cols = df_pd.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c for c in columns if c in df_pd.columns]
    
    if not numeric_cols:
        return {"status": "error", "message": "No numeric columns found"}
    
    X = df_pd[numeric_cols].fillna(df_pd[numeric_cols].mean())
    
    results = {
        "method": method,
        "columns_analyzed": numeric_cols,
        "total_rows": len(X),
        "anomaly_indices": [],
        "anomaly_scores": []
    }
    
    if method == "isolation_forest":
        # Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)
        scores = clf.score_samples(X)
        
        anomaly_mask = predictions == -1
        results["anomalies_detected"] = int(anomaly_mask.sum())
        results["anomaly_percentage"] = float((anomaly_mask.sum() / len(X)) * 100)
        results["anomaly_indices"] = np.where(anomaly_mask)[0].tolist()
        results["anomaly_scores"] = scores[anomaly_mask].tolist()
        
        results["insights"] = [
            f"🔍 Detected {results['anomalies_detected']} anomalies ({results['anomaly_percentage']:.2f}% of data)",
            f"📊 Using Isolation Forest with contamination={contamination}"
        ]
    
    else:  # Statistical method
        # Z-score method
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
        anomaly_mask = (z_scores > 3).any(axis=1)
        
        results["anomalies_detected"] = int(anomaly_mask.sum())
        results["anomaly_percentage"] = float((anomaly_mask.sum() / len(X)) * 100)
        results["anomaly_indices"] = np.where(anomaly_mask)[0].tolist()
        
        results["insights"] = [
            f"🔍 Detected {results['anomalies_detected']} anomalies ({results['anomaly_percentage']:.2f}% of data)",
            f"📊 Using statistical method (Z-score > 3)"
        ]
    
    return results


def perform_hypothesis_testing(file_path: str,
                               group_col: str,
                               value_col: str,
                               test_type: str = "auto") -> Dict[str, Any]:
    """
    Perform statistical hypothesis testing.
    
    Args:
        file_path: Path to dataset
        group_col: Column defining groups
        value_col: Column with values to compare
        test_type: 't-test', 'chi-square', 'anova', or 'auto'
        
    Returns:
        Dictionary with test results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path).to_pandas()
    
    if group_col not in df.columns or value_col not in df.columns:
        return {"status": "error", "message": "Columns not found"}
    
    results = {
        "group_column": group_col,
        "value_column": value_col,
        "test_type": test_type
    }
    
    # Get groups
    groups = df.groupby(group_col)[value_col].apply(list).to_dict()
    group_names = list(groups.keys())
    
    if len(group_names) < 2:
        return {"status": "error", "message": "Need at least 2 groups for comparison"}
    
    # Auto-detect test type
    if test_type == "auto":
        if len(group_names) == 2:
            test_type = "t-test"
        else:
            test_type = "anova"
    
    # Perform test
    if test_type == "t-test" and len(group_names) >= 2:
        group1_data = groups[group_names[0]]
        group2_data = groups[group_names[1]]
        
        statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        
        results["test_statistic"] = float(statistic)
        results["p_value"] = float(p_value)
        results["significant"] = p_value < 0.05
        results["groups_compared"] = [group_names[0], group_names[1]]
        
        results["interpretation"] = (
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference "
            f"between {group_names[0]} and {group_names[1]} (p={p_value:.4f})"
        )
        
        # Effect size (Cohen's d)
        mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
        std1, std2 = np.std(group1_data), np.std(group2_data)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        results["effect_size"] = float(cohens_d)
        results["group_means"] = {group_names[0]: float(mean1), group_names[1]: float(mean2)}
    
    elif test_type == "anova":
        group_data = [groups[g] for g in group_names]
        statistic, p_value = stats.f_oneway(*group_data)
        
        results["test_statistic"] = float(statistic)
        results["p_value"] = float(p_value)
        results["significant"] = p_value < 0.05
        results["groups_compared"] = group_names
        
        results["interpretation"] = (
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference "
            f"among {len(group_names)} groups (p={p_value:.4f})"
        )
        
        # Group means
        results["group_means"] = {g: float(np.mean(groups[g])) for g in group_names}
    
    return results


def analyze_distribution(file_path: str,
                        column: str,
                        tests: List[str] = ["normality", "skewness"]) -> Dict[str, Any]:
    """
    Analyze distribution of a column.
    
    Args:
        file_path: Path to dataset
        column: Column to analyze
        tests: List of tests to perform
        
    Returns:
        Dictionary with distribution analysis results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path).to_pandas()
    
    if column not in df.columns:
        return {"status": "error", "message": f"Column '{column}' not found"}
    
    data = df[column].dropna()
    
    results = {
        "column": column,
        "n_values": len(data),
        "n_missing": int(df[column].isna().sum()),
        "tests_performed": tests,
        "insights": []
    }
    
    # Basic statistics
    results["statistics"] = {
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75))
    }
    
    # Normality test
    if "normality" in tests:
        statistic, p_value = stats.shapiro(data.sample(min(5000, len(data))))  # Limit for performance
        results["normality_test"] = {
            "test": "Shapiro-Wilk",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05
        }
        
        if p_value > 0.05:
            results["insights"].append(f"✅ Data appears normally distributed (p={p_value:.4f})")
        else:
            results["insights"].append(f"⚠️ Data is NOT normally distributed (p={p_value:.4f})")
    
    # Skewness
    if "skewness" in tests:
        skewness = float(stats.skew(data))
        kurtosis = float(stats.kurtosis(data))
        
        results["skewness"] = skewness
        results["kurtosis"] = kurtosis
        
        if abs(skewness) < 0.5:
            skew_desc = "approximately symmetric"
        elif skewness > 0:
            skew_desc = "right-skewed (positive skew)"
        else:
            skew_desc = "left-skewed (negative skew)"
        
        results["insights"].append(f"📊 Distribution is {skew_desc} (skewness={skewness:.3f})")
    
    return results


def perform_segment_analysis(file_path: str,
                             n_segments: int = 5,
                             features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Perform cluster-based segment analysis.
    
    Args:
        file_path: Path to dataset
        n_segments: Number of segments to create
        features: Features to use for clustering (all numeric if None)
        
    Returns:
        Dictionary with segment analysis results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path).to_pandas()
    
    # Select features
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        features = [f for f in features if f in df.columns]
    
    if not features:
        return {"status": "error", "message": "No numeric features found for clustering"}
    
    # Prepare data
    X = df[features].fillna(df[features].mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['segment'] = labels
    
    # Analyze segments
    segment_profiles = []
    for i in range(n_segments):
        segment_data = df[df['segment'] == i]
        profile = {
            "segment_id": i,
            "size": len(segment_data),
            "percentage": float((len(segment_data) / len(df)) * 100),
            "characteristics": {}
        }
        
        # Calculate mean for each feature
        for feat in features:
            profile["characteristics"][feat] = {
                "mean": float(segment_data[feat].mean()),
                "std": float(segment_data[feat].std())
            }
        
        segment_profiles.append(profile)
    
    results = {
        "n_segments": n_segments,
        "features_used": features,
        "total_samples": len(df),
        "segments": segment_profiles,
        "insights": [
            f"🎯 Created {n_segments} segments from {len(df)} samples",
            f"📊 Used {len(features)} features for segmentation"
        ]
    }
    
    # Find most distinctive features for each segment
    for i, profile in enumerate(segment_profiles):
        results["insights"].append(
            f"Segment {i}: {profile['size']} samples ({profile['percentage']:.1f}%)"
        )
    
    return results
