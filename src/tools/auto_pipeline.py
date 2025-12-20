"""
Automated ML Pipeline
Zero-configuration automatic data processing: Clean → Encode → Engineer → Select
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.polars_helpers import load_dataframe, get_numeric_columns
from utils.validation import validate_file_exists
from tools.data_cleaning import clean_missing_values, handle_outliers
from tools.data_type_conversion import force_numeric_conversion, smart_type_inference
from tools.feature_engineering import encode_categorical, create_time_features
from tools.advanced_feature_engineering import create_interaction_features


def auto_ml_pipeline(file_path: str,
                     target_col: str,
                     task_type: str = "auto",
                     output_path: Optional[str] = None,
                     feature_engineering_level: str = "basic") -> Dict[str, Any]:
    """
    Fully automated ML pipeline with zero manual intervention.
    
    Pipeline stages:
    1. Auto-detect column types
    2. Clean missing values intelligently
    3. Handle outliers
    4. Encode categorical variables
    5. Engineer time features (if datetime detected)
    6. Create interaction features (if requested)
    7. Select best features
    
    Args:
        file_path: Path to input dataset
        target_col: Target column name
        task_type: 'classification', 'regression', or 'auto'
        output_path: Where to save processed data
        feature_engineering_level: 'basic', 'intermediate', 'advanced'
        
    Returns:
        Dictionary with pipeline results and explanations
    """
    validate_file_exists(file_path)
    
    if output_path is None:
        output_path = "./outputs/data/auto_pipeline_output.csv"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "stages_completed": [],
        "transformations_applied": [],
        "warnings": [],
        "final_features": [],
        "output_path": output_path
    }
    
    # Load data
    df = load_dataframe(file_path)
    original_shape = df.shape
    results["original_shape"] = {"rows": original_shape[0], "columns": original_shape[1]}
    
    print(f"🚀 Starting Auto ML Pipeline")
    print(f"📊 Original shape: {original_shape[0]:,} rows × {original_shape[1]} columns")
    
    # STAGE 1: Auto-detect column types
    print("\n🔍 Stage 1: Auto-detecting column types...")
    type_detection = smart_type_inference(file_path, output_path="./outputs/data/stage1_types.csv")
    results["stages_completed"].append("type_detection")
    results["transformations_applied"].append({
        "stage": "Type Detection",
        "description": f"Detected {len(type_detection.get('conversions_made', []))} type conversions"
    })
    current_file = "./outputs/data/stage1_types.csv"
    
    # STAGE 2: Clean missing values
    print("\n🧹 Stage 2: Cleaning missing values...")
    cleaning_result = clean_missing_values(
        current_file,
        strategy="auto",
        output_path="./outputs/data/stage2_cleaned.csv"
    )
    results["stages_completed"].append("missing_value_cleaning")
    results["transformations_applied"].append({
        "stage": "Missing Value Cleaning",
        "description": f"Cleaned {cleaning_result.get('total_nulls_before', 0)} missing values using auto-detected strategies"
    })
    current_file = "./outputs/data/stage2_cleaned.csv"
    
    # STAGE 3: Handle outliers
    print("\n📊 Stage 3: Handling outliers...")
    outlier_result = handle_outliers(
        current_file,
        columns=["all"],
        method="clip",
        output_path="./outputs/data/stage3_no_outliers.csv"
    )
    results["stages_completed"].append("outlier_handling")
    results["transformations_applied"].append({
        "stage": "Outlier Handling",
        "description": f"Clipped outliers in {outlier_result.get('columns_processed', 0)} columns"
    })
    current_file = "./outputs/data/stage3_no_outliers.csv"
    
    # STAGE 4: Force numeric conversion (for any remaining string numbers)
    print("\n🔢 Stage 4: Converting to numeric...")
    numeric_result = force_numeric_conversion(
        current_file,
        columns=["all"],
        errors="coerce",
        output_path="./outputs/data/stage4_numeric.csv"
    )
    results["stages_completed"].append("numeric_conversion")
    current_file = "./outputs/data/stage4_numeric.csv"
    
    # STAGE 5: Encode categorical variables
    print("\n🏷️  Stage 5: Encoding categorical variables...")
    encoding_result = encode_categorical(
        current_file,
        method="auto",
        output_path="./outputs/data/stage5_encoded.csv"
    )
    results["stages_completed"].append("categorical_encoding")
    results["transformations_applied"].append({
        "stage": "Categorical Encoding",
        "description": f"Encoded {len(encoding_result.get('encoded_columns', []))} categorical columns"
    })
    current_file = "./outputs/data/stage5_encoded.csv"
    
    # STAGE 6: Feature engineering (if requested)
    if feature_engineering_level in ["intermediate", "advanced"]:
        print("\n⚙️  Stage 6: Engineering features...")
        
        # Check for datetime columns and create time features
        df_current = load_dataframe(current_file).to_pandas()
        datetime_cols = df_current.select_dtypes(include=['datetime64']).columns.tolist()
        
        if datetime_cols:
            print(f"   Creating time features from {len(datetime_cols)} datetime columns...")
            for dt_col in datetime_cols:
                try:
                    time_result = create_time_features(
                        current_file,
                        date_column=dt_col,
                        output_path=current_file  # Overwrite
                    )
                    results["transformations_applied"].append({
                        "stage": "Time Feature Engineering",
                        "description": f"Created time features from {dt_col}"
                    })
                except Exception as e:
                    results["warnings"].append(f"Could not create time features from {dt_col}: {str(e)}")
        
        # Create interaction features for advanced mode
        if feature_engineering_level == "advanced":
            print("   Creating interaction features...")
            try:
                interaction_result = create_interaction_features(
                    current_file,
                    method="polynomial",
                    degree=2,
                    max_features=10,
                    output_path="./outputs/data/stage6_engineered.csv"
                )
                results["stages_completed"].append("interaction_features")
                results["transformations_applied"].append({
                    "stage": "Interaction Features",
                    "description": f"Created {len(interaction_result.get('new_features', []))} interaction features"
                })
                current_file = "./outputs/data/stage6_engineered.csv"
            except Exception as e:
                results["warnings"].append(f"Could not create interaction features: {str(e)}")
    
    # STAGE 7: Feature selection
    print("\n🎯 Stage 7: Selecting best features...")
    try:
        selection_result = auto_feature_selection(
            current_file,
            target_col=target_col,
            task_type=task_type,
            max_features=50,
            output_path=output_path
        )
        results["stages_completed"].append("feature_selection")
        results["transformations_applied"].append({
            "stage": "Feature Selection",
            "description": f"Selected {selection_result['n_features_selected']} best features from {selection_result['n_features_original']}"
        })
        results["selected_features"] = selection_result["selected_features"]
        results["feature_importance"] = selection_result.get("feature_scores", {})
    except Exception as e:
        results["warnings"].append(f"Feature selection failed: {str(e)}")
        # Just copy the file
        import shutil
        shutil.copy(current_file, output_path)
    
    # Final shape
    df_final = load_dataframe(output_path)
    final_shape = df_final.shape
    results["final_shape"] = {"rows": final_shape[0], "columns": final_shape[1]}
    results["final_features"] = df_final.columns
    
    print(f"\n✅ Pipeline completed!")
    print(f"📊 Final shape: {final_shape[0]:,} rows × {final_shape[1]} columns")
    print(f"💾 Saved to: {output_path}")
    
    # Generate summary
    results["summary"] = _generate_pipeline_summary(results)
    
    return results


def auto_feature_selection(file_path: str,
                           target_col: str,
                           task_type: str = "auto",
                           max_features: int = 50,
                           method: str = "auto",
                           output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Automatically select the best features for modeling.
    
    Args:
        file_path: Path to dataset
        target_col: Target column
        task_type: 'classification', 'regression', or 'auto'
        max_features: Maximum number of features to keep
        method: 'mutual_info', 'f_test', or 'auto'
        output_path: Where to save selected features
        
    Returns:
        Dictionary with selection results
    """
    validate_file_exists(file_path)
    df = load_dataframe(file_path).to_pandas()
    
    if target_col not in df.columns:
        return {"status": "error", "message": f"Target column '{target_col}' not found"}
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Get only numeric features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_features]
    
    if len(numeric_features) == 0:
        return {"status": "error", "message": "No numeric features found"}
    
    # Auto-detect task type
    if task_type == "auto":
        if y.dtype == 'object' or y.nunique() < 20:
            task_type = "classification"
        else:
            task_type = "regression"
    
    # Select method
    if method == "auto":
        method = "mutual_info" if task_type == "classification" else "f_test"
    
    # Perform selection
    n_features_to_select = min(max_features, len(numeric_features))
    
    if method == "mutual_info":
        if task_type == "classification":
            selector = SelectKBest(mutual_info_classif, k=n_features_to_select)
        else:
            from sklearn.feature_selection import mutual_info_regression
            selector = SelectKBest(mutual_info_regression, k=n_features_to_select)
    else:  # f_test
        if task_type == "classification":
            selector = SelectKBest(f_classif, k=n_features_to_select)
        else:
            selector = SelectKBest(f_regression, k=n_features_to_select)
    
    # Fit selector
    X_selected = selector.fit_transform(X_numeric.fillna(0), y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = np.array(numeric_features)[selected_mask].tolist()
    
    # Get feature scores
    feature_scores = dict(zip(numeric_features, selector.scores_))
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = {
        "n_features_original": len(numeric_features),
        "n_features_selected": len(selected_features),
        "selected_features": selected_features,
        "feature_scores": dict(sorted_features[:n_features_to_select]),
        "selection_method": method,
        "task_type": task_type
    }
    
    # Save selected features + target
    if output_path:
        df_selected = df[selected_features + [target_col]]
        df_selected.to_csv(output_path, index=False)
        results["output_path"] = output_path
    
    return results


def _generate_pipeline_summary(results: Dict[str, Any]) -> str:
    """Generate human-readable summary of pipeline execution."""
    summary = []
    
    summary.append("🔄 **Auto ML Pipeline Summary**\n")
    summary.append(f"Original shape: {results['original_shape']['rows']:,} rows × {results['original_shape']['columns']} columns")
    summary.append(f"Final shape: {results['final_shape']['rows']:,} rows × {results['final_shape']['columns']} columns\n")
    
    summary.append("**Stages Completed:**")
    for i, stage in enumerate(results['stages_completed'], 1):
        summary.append(f"{i}. {stage.replace('_', ' ').title()}")
    
    summary.append("\n**Transformations Applied:**")
    for transform in results['transformations_applied']:
        summary.append(f"• {transform['stage']}: {transform['description']}")
    
    if results.get('warnings'):
        summary.append("\n⚠️  **Warnings:**")
        for warning in results['warnings']:
            summary.append(f"• {warning}")
    
    if results.get('selected_features'):
        summary.append(f"\n🎯 **Selected {len(results['selected_features'])} best features**")
    
    summary.append(f"\n💾 Output saved to: {results['output_path']}")
    
    return "\n".join(summary)


def explain_pipeline_decision(stage: str, decision: str, reason: str) -> Dict[str, str]:
    """
    Explain a pipeline decision in human-readable format.
    
    Args:
        stage: Pipeline stage name
        decision: What decision was made
        reason: Why this decision was made
        
    Returns:
        Dictionary with explanation
    """
    return {
        "stage": stage,
        "decision": decision,
        "reason": reason,
        "explanation": f"In the {stage} stage, I decided to {decision} because {reason}"
    }
