"""
Advanced Preprocessing Tools
Tools for handling imbalanced data, feature scaling, and strategic data splitting.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
import joblib
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter

from utils.polars_helpers import (
    load_dataframe, save_dataframe, get_numeric_columns,
    get_categorical_columns, split_features_target
)
from utils.validation import (
    validate_file_exists, validate_file_format, validate_dataframe,
    validate_column_exists
)


def handle_imbalanced_data(
    file_path: str,
    target_col: str,
    strategy: str = "smote",
    sampling_ratio: float = 1.0,
    output_path: str = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Handle imbalanced datasets using various resampling techniques.
    
    Args:
        file_path: Path to dataset
        target_col: Target column name
        strategy: Resampling strategy:
            - 'smote': Synthetic Minority Over-sampling (SMOTE)
            - 'adasyn': Adaptive Synthetic Sampling
            - 'borderline_smote': Borderline SMOTE variant
            - 'random_undersample': Random undersampling
            - 'tomek': Tomek Links undersampling
            - 'smote_tomek': Combined SMOTE + Tomek Links
            - 'smote_enn': Combined SMOTE + Edited Nearest Neighbours
            - 'class_weights': Return class weights (no resampling)
        sampling_ratio: Ratio of minority to majority class (0.5 = 50%, 1.0 = 100%)
        output_path: Path to save balanced dataset
        random_state: Random seed
        
    Returns:
        Dictionary with balancing results and class distributions
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Get original class distribution
    original_dist = df[target_col].value_counts().to_dict()
    original_counts = dict(sorted(original_dist.items()))
    
    print(f"📊 Original class distribution: {original_counts}")
    
    # Calculate imbalance ratio
    class_counts = list(original_counts.values())
    imbalance_ratio = max(class_counts) / min(class_counts)
    
    if imbalance_ratio < 1.5:
        return {
            'status': 'skipped',
            'message': 'Dataset is already balanced (ratio < 1.5)',
            'original_distribution': original_counts,
            'imbalance_ratio': float(imbalance_ratio)
        }
    
    # Prepare data
    X, y = split_features_target(df, target_col)
    
    # Handle class weights strategy (no resampling)
    if strategy == "class_weights":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        return {
            'status': 'success',
            'strategy': 'class_weights',
            'class_weights': {str(k): float(v) for k, v in class_weights.items()},
            'original_distribution': original_counts,
            'imbalance_ratio': float(imbalance_ratio),
            'recommendation': 'Use class_weight parameter in your model training'
        }
    
    # Create resampler based on strategy
    sampling_strategy = sampling_ratio if sampling_ratio < 1.0 else 'auto'
    
    if strategy == "smote":
        resampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif strategy == "adasyn":
        resampler = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif strategy == "borderline_smote":
        resampler = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    elif strategy == "random_undersample":
        resampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    elif strategy == "tomek":
        resampler = TomekLinks(sampling_strategy='auto')
    elif strategy == "smote_tomek":
        resampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    elif strategy == "smote_enn":
        resampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    
    # Perform resampling
    print(f"⚖️ Applying {strategy} resampling...")
    X_resampled, y_resampled = resampler.fit_resample(X, y)
    
    # Get new class distribution
    new_counts = dict(Counter(y_resampled))
    new_counts = dict(sorted(new_counts.items()))
    
    print(f"✅ New class distribution: {new_counts}")
    
    # Calculate changes
    total_original = sum(original_counts.values())
    total_new = sum(new_counts.values())
    
    changes = {
        str(cls): {
            'original': original_counts.get(cls, 0),
            'new': new_counts.get(cls, 0),
            'change': new_counts.get(cls, 0) - original_counts.get(cls, 0)
        }
        for cls in set(list(original_counts.keys()) + list(new_counts.keys()))
    }
    
    # Create balanced dataframe
    feature_cols = [col for col in df.columns if col != target_col]
    balanced_data = {col: X_resampled[:, i] for i, col in enumerate(feature_cols)}
    balanced_data[target_col] = y_resampled
    
    balanced_df = pl.DataFrame(balanced_data)
    
    # Save if output path provided
    if output_path:
        save_dataframe(balanced_df, output_path)
        print(f"💾 Balanced dataset saved to: {output_path}")
    
    return {
        'status': 'success',
        'strategy': strategy,
        'original_distribution': original_counts,
        'new_distribution': new_counts,
        'changes_by_class': changes,
        'total_samples_before': total_original,
        'total_samples_after': total_new,
        'sample_change': f"{'+' if total_new > total_original else ''}{total_new - total_original}",
        'new_imbalance_ratio': float(max(new_counts.values()) / min(new_counts.values())),
        'output_path': output_path
    }


def perform_feature_scaling(
    file_path: str,
    scaler_type: str = "standard",
    columns: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    scaler_save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scale features using various normalization techniques.
    
    Args:
        file_path: Path to dataset
        scaler_type: Scaling method:
            - 'standard': StandardScaler (mean=0, std=1)
            - 'minmax': MinMaxScaler (range 0-1)
            - 'robust': RobustScaler (median, IQR - robust to outliers)
        columns: List of columns to scale (None = all numeric columns)
        output_path: Path to save scaled dataset
        scaler_save_path: Path to save fitted scaler for future use
        
    Returns:
        Dictionary with scaling statistics
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
        print(f"🔢 Auto-detected {len(columns)} numeric columns for scaling")
    else:
        for col in columns:
            validate_column_exists(df, col)
    
    if not columns:
        return {
            'status': 'skipped',
            'message': 'No numeric columns found to scale'
        }
    
    # Create scaler
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")
    
    # Get original statistics
    original_stats = {}
    for col in columns:
        col_data = df[col].to_numpy()
        original_stats[col] = {
            'mean': float(np.mean(col_data)),
            'std': float(np.std(col_data)),
            'min': float(np.min(col_data)),
            'max': float(np.max(col_data)),
            'median': float(np.median(col_data))
        }
    
    # Fit and transform
    print(f"📏 Applying {scaler_type} scaling to {len(columns)} columns...")
    scaled_data = scaler.fit_transform(df[columns].to_numpy())
    
    # Create scaled dataframe
    df_scaled = df.clone()
    for i, col in enumerate(columns):
        df_scaled = df_scaled.with_columns(
            pl.Series(col, scaled_data[:, i])
        )
    
    # Get new statistics
    new_stats = {}
    for i, col in enumerate(columns):
        new_stats[col] = {
            'mean': float(np.mean(scaled_data[:, i])),
            'std': float(np.std(scaled_data[:, i])),
            'min': float(np.min(scaled_data[:, i])),
            'max': float(np.max(scaled_data[:, i])),
            'median': float(np.median(scaled_data[:, i]))
        }
    
    # Save scaled data
    if output_path:
        save_dataframe(df_scaled, output_path)
        print(f"💾 Scaled dataset saved to: {output_path}")
    
    # Save scaler
    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"💾 Scaler saved to: {scaler_save_path}")
    
    return {
        'status': 'success',
        'scaler_type': scaler_type,
        'columns_scaled': columns,
        'n_columns': len(columns),
        'original_stats': original_stats,
        'scaled_stats': new_stats,
        'output_path': output_path,
        'scaler_path': scaler_save_path
    }


def split_data_strategically(
    file_path: str,
    target_col: Optional[str] = None,
    split_type: str = "train_test",
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    time_col: Optional[str] = None,
    group_col: Optional[str] = None,
    random_state: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform strategic data splitting with multiple options.
    
    Args:
        file_path: Path to dataset
        target_col: Target column (for stratification)
        split_type: Split strategy:
            - 'train_test': Train/test split
            - 'train_val_test': Train/validation/test split
            - 'time_based': Time-based split (requires time_col)
            - 'group_based': Group-based split (requires group_col, prevents leakage)
        test_size: Test set proportion
        val_size: Validation set proportion (for train_val_test)
        stratify: Whether to stratify by target
        time_col: Column to use for time-based splitting
        group_col: Column to use for group-based splitting
        random_state: Random seed
        output_dir: Directory to save split datasets
        
    Returns:
        Dictionary with split information and file paths
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    if target_col:
        validate_column_exists(df, target_col)
    
    n_samples = len(df)
    
    # Time-based split
    if split_type == "time_based":
        if not time_col:
            raise ValueError("time_col is required for time_based split")
        validate_column_exists(df, time_col)
        
        # Sort by time
        df = df.sort(time_col)
        
        # Calculate split points
        test_idx = int(n_samples * (1 - test_size))
        
        if output_dir:
            train_df = df[:test_idx]
            test_df = df[test_idx:]
            
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, "train.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            save_dataframe(train_df, train_path)
            save_dataframe(test_df, test_path)
            
            print(f"✅ Time-based split: train={len(train_df)}, test={len(test_df)}")
            
            return {
                'status': 'success',
                'split_type': 'time_based',
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_path': train_path,
                'test_path': test_path,
                'time_column': time_col
            }
    
    # Group-based split
    elif split_type == "group_based":
        if not group_col:
            raise ValueError("group_col is required for group_based split")
        validate_column_exists(df, group_col)
        
        # Get unique groups
        unique_groups = df[group_col].unique().to_list()
        n_groups = len(unique_groups)
        
        # Split groups
        np.random.seed(random_state)
        np.random.shuffle(unique_groups)
        
        test_n_groups = max(1, int(n_groups * test_size))
        test_groups = unique_groups[:test_n_groups]
        train_groups = unique_groups[test_n_groups:]
        
        train_df = df.filter(pl.col(group_col).is_in(train_groups))
        test_df = df.filter(pl.col(group_col).is_in(test_groups))
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, "train.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            save_dataframe(train_df, train_path)
            save_dataframe(test_df, test_path)
            
            print(f"✅ Group-based split: train={len(train_df)}, test={len(test_df)}")
            
            return {
                'status': 'success',
                'split_type': 'group_based',
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_groups': len(train_groups),
                'test_groups': len(test_groups),
                'train_path': train_path,
                'test_path': test_path,
                'group_column': group_col
            }
    
    # Standard train/test split
    elif split_type == "train_test":
        X, y = split_features_target(df, target_col) if target_col else (df.to_numpy(), None)
        
        stratify_y = y if (stratify and target_col and len(np.unique(y)) < 20) else None
        
        if target_col:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
            )
            
            # Reconstruct dataframes
            feature_cols = [col for col in df.columns if col != target_col]
            train_data = {col: X_train[:, i] for i, col in enumerate(feature_cols)}
            train_data[target_col] = y_train
            train_df = pl.DataFrame(train_data)
            
            test_data = {col: X_test[:, i] for i, col in enumerate(feature_cols)}
            test_data[target_col] = y_test
            test_df = pl.DataFrame(test_data)
        else:
            indices = np.arange(len(df))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            train_df = df[train_idx]
            test_df = df[test_idx]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, "train.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            save_dataframe(train_df, train_path)
            save_dataframe(test_df, test_path)
            
            print(f"✅ Train/test split: train={len(train_df)}, test={len(test_df)}")
            
            return {
                'status': 'success',
                'split_type': 'train_test',
                'train_size': len(train_df),
                'test_size': len(test_df),
                'stratified': bool(stratify_y is not None),
                'train_path': train_path,
                'test_path': test_path
            }
    
    # Train/val/test split
    elif split_type == "train_val_test":
        X, y = split_features_target(df, target_col) if target_col else (df.to_numpy(), None)
        
        stratify_y = y if (stratify and target_col and len(np.unique(y)) < 20) else None
        
        # First split: train+val vs test
        if target_col:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
            )
            
            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            stratify_temp = y_temp if stratify_y is not None else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=stratify_temp
            )
            
            # Reconstruct dataframes
            feature_cols = [col for col in df.columns if col != target_col]
            
            train_data = {col: X_train[:, i] for i, col in enumerate(feature_cols)}
            train_data[target_col] = y_train
            train_df = pl.DataFrame(train_data)
            
            val_data = {col: X_val[:, i] for i, col in enumerate(feature_cols)}
            val_data[target_col] = y_val
            val_df = pl.DataFrame(val_data)
            
            test_data = {col: X_test[:, i] for i, col in enumerate(feature_cols)}
            test_data[target_col] = y_test
            test_df = pl.DataFrame(test_data)
        else:
            indices = np.arange(len(df))
            temp_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            val_ratio = val_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                temp_idx, test_size=val_ratio, random_state=random_state
            )
            
            train_df = df[train_idx]
            val_df = df[val_idx]
            test_df = df[test_idx]
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            train_path = os.path.join(output_dir, "train.csv")
            val_path = os.path.join(output_dir, "val.csv")
            test_path = os.path.join(output_dir, "test.csv")
            
            save_dataframe(train_df, train_path)
            save_dataframe(val_df, val_path)
            save_dataframe(test_df, test_path)
            
            print(f"✅ Train/val/test split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
            
            return {
                'status': 'success',
                'split_type': 'train_val_test',
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'stratified': bool(stratify_y is not None),
                'train_path': train_path,
                'val_path': val_path,
                'test_path': test_path
            }
    
    else:
        raise ValueError(f"Unsupported split_type: {split_type}")
