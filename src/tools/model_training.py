"""
Model Training Tools
Tools for training machine learning models and generating reports.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os
import joblib
import json
import tempfile

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import artifact store
try:
    from storage.helpers import save_model_with_store
    ARTIFACT_STORE_AVAILABLE = True
except ImportError:
    ARTIFACT_STORE_AVAILABLE = False
    print("⚠️  Artifact store not available, using local paths")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import shap

try:
    from .visualization_engine import (
        generate_model_performance_plots,
        generate_feature_importance_plot
    )
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"⚠️  Visualization engine not available: {e}")

from utils.polars_helpers import (
    load_dataframe,
    get_numeric_columns,
    split_features_target,
)
from utils.validation import (
    validate_file_exists,
    validate_file_format,
    validate_dataframe,
    validate_column_exists,
    validate_target_column,
)


def train_baseline_models(file_path: str, target_col: str, 
                         task_type: str = "auto",
                         test_size: float = 0.2,
                         random_state: int = 42) -> Dict[str, Any]:
    """
    Train multiple baseline models and compare performance.
    
    Args:
        file_path: Path to prepared dataset
        target_col: Name of target column
        task_type: 'classification', 'regression', or 'auto'
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Dictionary with training results and best model
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Infer task type if auto
    if task_type == "auto":
        task_type = validate_target_column(df, target_col)
    
    # Split features and target
    X, y = split_features_target(df, target_col)
    
    # Convert to numpy for sklearn
    # Only keep numeric columns for X
    numeric_cols = get_numeric_columns(X)
    if len(numeric_cols) == 0:
        return {
            "status": "error",
            "message": "No numeric features found. Please encode categorical variables first."
        }
    
    X_numeric = X.select(numeric_cols)
    X_np = X_numeric.to_numpy()
    y_np = y.to_numpy()
    
    # Handle missing values (simple imputation with mean)
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_np = imputer.fit_transform(X_np)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=random_state
    )
    
    results = {
        "task_type": task_type,
        "n_features": X_np.shape[1],
        "n_samples": len(X_np),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_names": numeric_cols,
        "models": {}
    }
    
    # Train models based on task type
    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            "xgboost": XGBClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            "lightgbm": LGBMClassifier(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1),
            "catboost": CatBoostClassifier(iterations=100, random_state=random_state, verbose=0, allow_writing_files=False)
        }
        
        for model_name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                results["models"][model_name] = {
                    "train_metrics": {
                        "accuracy": float(accuracy_score(y_train, y_pred_train)),
                        "precision": float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                        "recall": float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
                        "f1": float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))
                    },
                    "test_metrics": {
                        "accuracy": float(accuracy_score(y_test, y_pred_test)),
                        "precision": float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                        "recall": float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
                        "f1": float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0))
                    }
                }
                
                # Save model using artifact store
                if ARTIFACT_STORE_AVAILABLE:
                    model_path = save_model_with_store(
                        model_data={
                            "model": model,
                            "imputer": imputer,
                            "feature_names": numeric_cols
                        },
                        filename=f"{model_name}.pkl",
                        metadata={
                            "model_name": model_name,
                            "task_type": "classification",
                            "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
                            "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
                            "features": numeric_cols
                        }
                    )
                else:
                    model_path = f"./outputs/models/{model_name}.pkl"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({
                        "model": model,
                        "imputer": imputer,
                        "feature_names": numeric_cols
                    }, model_path)
                
                results["models"][model_name]["model_path"] = model_path
                
            except Exception as e:
                results["models"][model_name] = {
                    "status": "error",
                    "message": str(e)
                }
    
    else:  # regression
        models = {
            "ridge": Ridge(random_state=random_state),
            "lasso": Lasso(random_state=random_state),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "xgboost": XGBRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            "lightgbm": LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1, verbose=-1),
            "catboost": CatBoostRegressor(iterations=100, random_state=random_state, verbose=0, allow_writing_files=False)
        }
        
        for model_name, model in models.items():
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                results["models"][model_name] = {
                    "train_metrics": {
                        "mse": float(mean_squared_error(y_train, y_pred_train)),
                        "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
                        "mae": float(mean_absolute_error(y_train, y_pred_train)),
                        "r2": float(r2_score(y_train, y_pred_train))
                    },
                    "test_metrics": {
                        "mse": float(mean_squared_error(y_test, y_pred_test)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                        "mae": float(mean_absolute_error(y_test, y_pred_test)),
                        "r2": float(r2_score(y_test, y_pred_test))
                    }
                }
                
                # Save model using artifact store
                if ARTIFACT_STORE_AVAILABLE:
                    model_path = save_model_with_store(
                        model_data={
                            "model": model,
                            "imputer": imputer,
                            "feature_names": numeric_cols
                        },
                        filename=f"{model_name}.pkl",
                        metadata={
                            "model_name": model_name,
                            "task_type": "regression",
                            "train_r2": float(r2_score(y_train, y_pred_train)),
                            "test_r2": float(r2_score(y_test, y_pred_test)),
                            "features": numeric_cols
                        }
                    )
                else:
                    model_path = f"./outputs/models/{model_name}.pkl"
                    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump({
                        "model": model,
                        "imputer": imputer,
                        "feature_names": numeric_cols
                    }, model_path)
                
                results["models"][model_name]["model_path"] = model_path
                
            except Exception as e:
                results["models"][model_name] = {
                    "status": "error",
                    "message": str(e)
                }
    
    # Determine best model
    best_model_name = None
    best_score = -float('inf')
    
    for model_name, model_results in results["models"].items():
        if "test_metrics" in model_results:
            if task_type == "classification":
                score = model_results["test_metrics"]["f1"]
            else:
                score = model_results["test_metrics"]["r2"]
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
    
    results["best_model"] = {
        "name": best_model_name,
        "score": best_score,
        "model_path": results["models"][best_model_name]["model_path"] if best_model_name else None
    }
    
    # Generate visualizations for best model
    if VISUALIZATION_AVAILABLE and best_model_name:
        try:
            print(f"\n🎨 Generating visualizations for {best_model_name}...")
            
            # Load best model
            model_data = joblib.dump({
                "model": models[best_model_name],
                "imputer": imputer,
                "feature_names": numeric_cols
            }, f"./outputs/models/{best_model_name}_temp.pkl")
            
            # Get predictions for visualization
            best_model = models[best_model_name]
            y_pred_test = best_model.predict(X_test)
            y_pred_proba = None
            if hasattr(best_model, "predict_proba") and task_type == "classification":
                y_pred_proba = best_model.predict_proba(X_test)
            
            # Generate model performance plots
            plot_dir = "./outputs/plots/model_performance"
            perf_plots = generate_model_performance_plots(
                y_true=y_test,
                y_pred=y_pred_test,
                y_pred_proba=y_pred_proba,
                task_type=task_type,
                model_name=best_model_name,
                output_dir=plot_dir
            )
            results["performance_plots"] = perf_plots["plot_paths"]
            
            # Generate feature importance plot if available
            if hasattr(best_model, "feature_importances_"):
                feature_importance = dict(zip(numeric_cols, best_model.feature_importances_))
                importance_plot = generate_feature_importance_plot(
                    feature_importances=feature_importance,
                    output_path=f"{plot_dir}/feature_importance_{best_model_name}.png"
                )
                results["feature_importance_plot"] = importance_plot
            
            print(f"   ✓ Generated {len(perf_plots.get('plot_paths', []))} performance plots")
            results["visualization_generated"] = True
            
        except Exception as e:
            print(f"   ⚠️ Could not generate visualizations: {str(e)}")
            results["visualization_generated"] = False
    else:
        results["visualization_generated"] = False
    
    return results


def generate_model_report(model_path: str, test_data_path: str, 
                         target_col: str, output_path: str) -> Dict[str, Any]:
    """
    Generate comprehensive model evaluation report.
    
    Args:
        model_path: Path to saved model file
        test_data_path: Path to test dataset
        target_col: Name of target column
        output_path: Path to save report JSON
        
    Returns:
        Dictionary with model report
    """
    # Validation
    validate_file_exists(model_path)
    validate_file_exists(test_data_path)
    
    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    imputer = model_data["imputer"]
    feature_names = model_data["feature_names"]
    
    # Load test data
    df = load_dataframe(test_data_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Prepare features
    X = df.select(feature_names)
    y = df[target_col].to_numpy()
    X_np = imputer.transform(X.to_numpy())
    
    # Predict
    y_pred = model.predict(X_np)
    
    # Determine task type
    if hasattr(model, "predict_proba"):
        task_type = "classification"
    else:
        task_type = "regression"
    
    report = {
        "model_path": model_path,
        "task_type": task_type,
        "n_features": len(feature_names),
        "n_samples": len(X_np)
    }
    
    # Calculate metrics
    if task_type == "classification":
        report["metrics"] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        report["confusion_matrix"] = cm.tolist()
        
        # Classification report
        class_report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        report["classification_report"] = class_report
    
    else:  # regression
        report["metrics"] = {
            "mse": float(mean_squared_error(y, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred))
        }
    
    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_importance = [
            {"feature": name, "importance": float(imp)}
            for name, imp in zip(feature_names, importances)
        ]
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        report["feature_importance"] = feature_importance[:20]  # Top 20
    
    # SHAP values (for top 10 features)
    try:
        # Use TreeExplainer for tree-based models
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models (sample for speed)
            sample_size = min(100, len(X_np))
            explainer = shap.KernelExplainer(
                model.predict, 
                X_np[:sample_size]
            )
        
        shap_values = explainer.shap_values(X_np[:100])  # First 100 samples
        
        # Calculate mean absolute SHAP values
        if isinstance(shap_values, list):  # Multi-class
            shap_values = shap_values[0]
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = [
            {"feature": name, "shap_value": float(val)}
            for name, val in zip(feature_names, mean_shap)
        ]
        shap_importance.sort(key=lambda x: x["shap_value"], reverse=True)
        report["shap_feature_importance"] = shap_importance[:10]  # Top 10
    
    except Exception as e:
        report["shap_error"] = f"Could not compute SHAP values: {str(e)}"
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    report["output_path"] = output_path
    
    return report
