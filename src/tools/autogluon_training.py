"""
AutoGluon-Powered Training Tools
Replaces manual model training with AutoGluon's automated ML for better accuracy,
automatic ensembling, and built-in handling of raw data (no pre-encoding needed).

Supports:
- Classification (binary + multiclass)
- Regression
- Time Series Forecasting (NEW capability)

Scalability safeguards:
- time_limit prevents runaway training
- presets control compute budget
- num_cpus capped to avoid hogging shared resources
- Memory-aware: excludes heavy models on limited RAM
"""

import os
import json
import time
import shutil
import warnings
from typing import Dict, Any, Optional, List
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Lazy import AutoGluon to avoid slow startup
AUTOGLUON_TABULAR_AVAILABLE = False
AUTOGLUON_TIMESERIES_AVAILABLE = False

def _ensure_autogluon_tabular():
    global AUTOGLUON_TABULAR_AVAILABLE
    try:
        from autogluon.tabular import TabularPredictor, TabularDataset
        AUTOGLUON_TABULAR_AVAILABLE = True
        return TabularPredictor, TabularDataset
    except ImportError:
        raise ImportError(
            "AutoGluon tabular not installed. Run: pip install autogluon.tabular"
        )

def _ensure_autogluon_timeseries():
    global AUTOGLUON_TIMESERIES_AVAILABLE
    try:
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
        AUTOGLUON_TIMESERIES_AVAILABLE = True
        return TimeSeriesPredictor, TimeSeriesDataFrame
    except ImportError:
        raise ImportError(
            "AutoGluon timeseries not installed. Run: pip install autogluon.timeseries"
        )


# ============================================================
# RESOURCE CONFIGURATION
# Adapt to deployment environment (HF Spaces, local, cloud)
# ============================================================

def _get_resource_config() -> Dict[str, Any]:
    """
    Detect available resources and return safe training config.
    Prevents AutoGluon from consuming too much memory/CPU on shared infra.
    """
    import psutil
    
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_count = os.cpu_count() or 2
    
    # Conservative defaults for shared environments (HF Spaces = 16GB, 2-8 vCPU)
    config = {
        "num_cpus": min(cpu_count, 4),  # Cap at 4 to leave room for other users
        "num_gpus": 0,  # No GPU on free HF Spaces
    }
    
    if total_ram_gb < 8:
        config["presets"] = "medium_quality"
        config["excluded_model_types"] = ["NN_TORCH", "FASTAI", "KNN"]
        config["time_limit"] = 60
    elif total_ram_gb < 16:
        config["presets"] = "medium_quality"
        config["excluded_model_types"] = ["NN_TORCH", "FASTAI"]
        config["time_limit"] = 120
    else:
        config["presets"] = "best_quality"
        config["excluded_model_types"] = ["NN_TORCH"]  # Still skip neural nets for speed
        config["time_limit"] = 180
    
    return config


# ============================================================
# TABULAR: Classification + Regression
# ============================================================

def train_with_autogluon(
    file_path: str,
    target_col: str,
    task_type: str = "auto",
    time_limit: int = 120,
    presets: str = "medium_quality",
    eval_metric: Optional[str] = None,
    output_dir: Optional[str] = None,
    infer_limit: Optional[float] = None
) -> Dict[str, Any]:
    """
    Train ML models using AutoGluon's automated approach.
    
    Handles raw data directly — no need to pre-encode categoricals or impute missing values.
    Automatically trains multiple models, performs stacking, and returns the best ensemble.
    
    Supports: classification (binary/multiclass), regression.
    
    Args:
        file_path: Path to CSV/Parquet dataset
        target_col: Column to predict
        task_type: 'classification', 'regression', or 'auto' (auto-detected)
        time_limit: Max training time in seconds (default 120 = 2 minutes)
        presets: Quality preset - 'medium_quality' (fast), 'best_quality' (slower, better),
                 'good_quality' (balanced)
        eval_metric: Metric to optimize (auto-selected if None).
                     Classification: 'accuracy', 'f1', 'roc_auc', 'log_loss'
                     Regression: 'rmse', 'mae', 'r2', 'mape'
        output_dir: Where to save trained model (default: ./outputs/autogluon_model)
    
    Returns:
        Dictionary with training results, leaderboard, best model info, and feature importance
    """
    TabularPredictor, TabularDataset = _ensure_autogluon_tabular()
    
    start_time = time.time()
    output_dir = output_dir or "./outputs/autogluon_model"
    
    # ── Validate input ──
    if not Path(file_path).exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    # ── Load data ──
    print(f"\n🚀 AutoGluon Training Starting...")
    print(f"   📁 Dataset: {file_path}")
    print(f"   🎯 Target: {target_col}")
    print(f"   ⏱️  Time limit: {time_limit}s")
    print(f"   📊 Presets: {presets}")
    
    try:
        train_data = TabularDataset(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    if target_col not in train_data.columns:
        return {
            "status": "error",
            "message": f"Target column '{target_col}' not found. Available: {list(train_data.columns)}"
        }
    
    n_rows, n_cols = train_data.shape
    print(f"   📐 Shape: {n_rows:,} rows × {n_cols} columns")
    
    # ── Get resource-aware config ──
    resource_config = _get_resource_config()
    
    # User overrides take priority
    effective_time_limit = min(time_limit, resource_config["time_limit"])
    effective_presets = presets
    
    # ── Auto-detect task type ──
    if task_type == "auto":
        n_unique = train_data[target_col].nunique()
        if n_unique <= 20 or train_data[target_col].dtype == 'object':
            task_type = "classification"
            if n_unique == 2:
                task_type_detail = "binary"
            else:
                task_type_detail = "multiclass"
        else:
            task_type = "regression"
            task_type_detail = "regression"
    else:
        task_type_detail = task_type
    
    # ── Select eval metric ──
    if eval_metric is None:
        if task_type == "classification":
            eval_metric = "f1_weighted" if task_type_detail == "multiclass" else "f1"
        else:
            eval_metric = "root_mean_squared_error"
    
    print(f"   🔍 Task type: {task_type_detail}")
    print(f"   📏 Eval metric: {eval_metric}")
    print(f"   🔧 Excluded models: {resource_config.get('excluded_model_types', [])}")
    
    # ── Clean output directory (AutoGluon needs fresh dir) ──
    if Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # ── Train ──
    try:
        predictor = TabularPredictor(
            label=target_col,
            eval_metric=eval_metric,
            path=output_dir,
            problem_type=task_type if task_type != "auto" else None
        )
        
        fit_kwargs = dict(
            train_data=train_data,
            time_limit=effective_time_limit,
            presets=effective_presets,
            excluded_model_types=resource_config.get("excluded_model_types", []),
            num_cpus=resource_config["num_cpus"],
            num_gpus=resource_config["num_gpus"],
            verbosity=1
        )
        if infer_limit is not None:
            fit_kwargs["infer_limit"] = infer_limit
        
        predictor.fit(**fit_kwargs)
    except Exception as e:
        return {"status": "error", "message": f"Training failed: {str(e)}"}
    
    elapsed = time.time() - start_time
    
    # ── Extract results ──
    leaderboard = predictor.leaderboard(silent=True)
    
    # Convert leaderboard to serializable format
    leaderboard_data = []
    for _, row in leaderboard.head(10).iterrows():
        entry = {
            "model": str(row.get("model", "")),
            "score_val": round(float(row.get("score_val", 0)), 4),
            "fit_time": round(float(row.get("fit_time", 0)), 1),
            "pred_time_val": round(float(row.get("pred_time_val", 0)), 3),
        }
        if "stack_level" in row:
            entry["stack_level"] = int(row["stack_level"])
        leaderboard_data.append(entry)
    
    # Best model info
    best_model = predictor.model_best
    best_score = float(leaderboard.iloc[0]["score_val"]) if len(leaderboard) > 0 else None
    
    # Feature importance (top 20)
    feature_importance_data = []
    try:
        fi = predictor.feature_importance(train_data, silent=True)
        for feat, row in fi.head(20).iterrows():
            feature_importance_data.append({
                "feature": str(feat),
                "importance": round(float(row.get("importance", 0)), 4),
                "p_value": round(float(row.get("p_value", 1)), 4) if "p_value" in row else None
            })
    except Exception:
        # feature_importance can fail on some model types
        pass
    
    # Model count
    n_models = len(leaderboard)
    
    # Summary
    results = {
        "status": "success",
        "task_type": task_type_detail,
        "eval_metric": eval_metric,
        "best_model": best_model,
        "best_score": best_score,
        "n_models_trained": n_models,
        "n_rows": n_rows,
        "n_features": n_cols - 1,
        "training_time_seconds": round(elapsed, 1),
        "time_limit_used": effective_time_limit,
        "presets": effective_presets,
        "leaderboard": leaderboard_data,
        "feature_importance": feature_importance_data,
        "model_path": output_dir,
        "output_path": output_dir,
    }
    
    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"✅ AUTOGLUON TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Models trained: {n_models}")
    print(f"🏆 Best model: {best_model}")
    print(f"📈 Best {eval_metric}: {best_score:.4f}" if best_score else "")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print(f"💾 Model saved: {output_dir}")
    if leaderboard_data:
        print(f"\n📋 Top 5 Leaderboard:")
        for i, entry in enumerate(leaderboard_data[:5], 1):
            print(f"   {i}. {entry['model']}: {entry['score_val']:.4f} (fit: {entry['fit_time']:.1f}s)")
    if feature_importance_data:
        print(f"\n🔑 Top 5 Features:")
        for fi_entry in feature_importance_data[:5]:
            print(f"   • {fi_entry['feature']}: {fi_entry['importance']:.4f}")
    print(f"{'='*60}\n")
    
    return results


def predict_with_autogluon(
    model_path: str,
    data_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Make predictions using a trained AutoGluon model.
    
    Args:
        model_path: Path to saved AutoGluon model directory
        data_path: Path to new data for prediction
        output_path: Path to save predictions CSV (optional)
    
    Returns:
        Dictionary with predictions and metadata
    """
    TabularPredictor, TabularDataset = _ensure_autogluon_tabular()
    
    if not Path(model_path).exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}
    if not Path(data_path).exists():
        return {"status": "error", "message": f"Data not found: {data_path}"}
    
    try:
        predictor = TabularPredictor.load(model_path)
        test_data = TabularDataset(data_path)
        
        predictions = predictor.predict(test_data)
        
        output_path = output_path or "./outputs/autogluon_predictions.csv"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        result_df = test_data.copy()
        result_df["prediction"] = predictions.values
        result_df.to_csv(output_path, index=False)
        
        # Prediction probabilities for classification
        probabilities = None
        try:
            proba = predictor.predict_proba(test_data)
            probabilities = {
                "columns": list(proba.columns),
                "sample": proba.head(5).to_dict()
            }
        except Exception:
            pass
        
        return {
            "status": "success",
            "n_predictions": len(predictions),
            "prediction_sample": predictions.head(10).tolist(),
            "output_path": output_path,
            "model_used": predictor.model_best,
            "probabilities": probabilities
        }
    except Exception as e:
        return {"status": "error", "message": f"Prediction failed: {str(e)}"}


# ============================================================
# TIME SERIES FORECASTING
# ============================================================

def forecast_with_autogluon(
    file_path: str,
    target_col: str,
    time_col: str,
    forecast_horizon: int = 30,
    id_col: Optional[str] = None,
    freq: Optional[str] = None,
    time_limit: int = 120,
    presets: str = "medium_quality",
    output_path: Optional[str] = None,
    static_features_path: Optional[str] = None,
    known_covariates_cols: Optional[List[str]] = None,
    holiday_country: Optional[str] = None,
    fill_missing: bool = True,
    models: Optional[List[str]] = None,
    quantile_levels: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Forecast time series using AutoGluon's TimeSeriesPredictor.
    
    Supports multiple forecasting models automatically: DeepAR, ETS, ARIMA, Theta,
    Chronos (foundation model), and statistical ensembles.
    Enhanced with covariates, holiday features, model selection, and quantile forecasting.
    
    Args:
        file_path: Path to time series CSV/Parquet
        target_col: Column with values to forecast
        time_col: Column with timestamps/dates
        forecast_horizon: Number of future periods to predict
        id_col: Column identifying different series (for multi-series)
        freq: Frequency string ('D'=daily, 'h'=hourly, 'MS'=monthly, 'W'=weekly)
        time_limit: Max training time in seconds
        presets: 'fast_training', 'medium_quality', 'best_quality', or 'chronos_tiny'
        output_path: Path to save forecast CSV
        static_features_path: CSV with per-series metadata (one row per series)
        known_covariates_cols: Columns with future-known values (holidays, promotions)
        holiday_country: Country code for auto holiday features (e.g. 'US', 'UK', 'IN')
        fill_missing: Whether to auto-fill missing values in time series
        models: Specific models to train (e.g. ['ETS', 'DeepAR', 'AutoARIMA'])
        quantile_levels: Quantile levels for probabilistic forecasts (e.g. [0.1, 0.5, 0.9])
    
    Returns:
        Dictionary with forecasts, model performance, and leaderboard
    """
    TimeSeriesPredictor, TimeSeriesDataFrame = _ensure_autogluon_timeseries()
    
    start_time = time.time()
    output_dir = "./outputs/autogluon_ts_model"
    output_path = output_path or "./outputs/autogluon_forecast.csv"
    
    # ── Validate ──
    if not Path(file_path).exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    print(f"\n🚀 AutoGluon Time Series Forecasting...")
    print(f"   📁 Dataset: {file_path}")
    print(f"   🎯 Target: {target_col}")
    print(f"   📅 Time column: {time_col}")
    print(f"   🔮 Forecast horizon: {forecast_horizon} periods")
    
    # ── Load and prepare data ──
    try:
        df = pd.read_csv(file_path)
    except Exception:
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    if target_col not in df.columns:
        return {
            "status": "error",
            "message": f"Target column '{target_col}' not found. Available: {list(df.columns)}"
        }
    if time_col not in df.columns:
        return {
            "status": "error",
            "message": f"Time column '{time_col}' not found. Available: {list(df.columns)}"
        }
    
    # Parse datetime
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    
    # If no id_col, create a dummy one (single series)
    if id_col is None or id_col not in df.columns:
        id_col = "__series_id"
        df[id_col] = "series_0"
    
    # Auto-detect frequency if not provided
    if freq is None:
        time_diffs = df[time_col].diff().dropna()
        median_diff = time_diffs.median()
        if median_diff <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_diff <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_diff <= pd.Timedelta(days=8):
            freq = "W"
        elif median_diff <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "D"  # Default
    
    print(f"   📊 Frequency: {freq}")
    print(f"   📐 Shape: {df.shape[0]:,} rows")
    
    # ── Add holiday features (#29) ──
    if holiday_country:
        try:
            import holidays as holidays_lib
            country_holidays = holidays_lib.country_holidays(holiday_country)
            df['is_holiday'] = df[time_col].dt.date.apply(
                lambda d: 1 if d in country_holidays else 0
            ).astype(float)
            if known_covariates_cols is None:
                known_covariates_cols = []
            if 'is_holiday' not in known_covariates_cols:
                known_covariates_cols.append('is_holiday')
            print(f"   🎄 Holiday features added for: {holiday_country}")
        except ImportError:
            print(f"   ⚠️ 'holidays' package not installed. Skipping holiday features.")
        except Exception as e:
            print(f"   ⚠️ Could not add holiday features: {e}")
    
    # ── Convert to TimeSeriesDataFrame ──
    try:
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column=id_col,
            timestamp_column=time_col
        )
    except Exception as e:
        return {"status": "error", "message": f"Failed to create time series: {str(e)}"}
    
    # ── Attach static features (#26) ──
    if static_features_path and Path(static_features_path).exists():
        try:
            static_df = pd.read_csv(static_features_path)
            ts_df.static_features = static_df
            print(f"   📌 Static features loaded: {list(static_df.columns)}")
        except Exception as e:
            print(f"   ⚠️ Could not load static features: {e}")
    
    # ── Fill missing values (#36) ──
    if fill_missing:
        try:
            ts_df = ts_df.fill_missing_values()
            print(f"   🔧 Missing values filled")
        except Exception:
            pass
    
    # ── Clean output dir ──
    if Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    
    # ── Get resource config ──
    resource_config = _get_resource_config()
    effective_time_limit = min(time_limit, resource_config["time_limit"])
    
    # ── Train forecasting models ──
    try:
        predictor_kwargs = dict(
            target=target_col,
            prediction_length=forecast_horizon,
            path=output_dir,
            freq=freq
        )
        if known_covariates_cols:
            predictor_kwargs["known_covariates_names"] = known_covariates_cols
        if quantile_levels:
            predictor_kwargs["quantile_levels"] = quantile_levels
        
        predictor = TimeSeriesPredictor(**predictor_kwargs)
        
        ts_fit_kwargs = dict(
            train_data=ts_df,
            time_limit=effective_time_limit,
            presets=presets,
        )
        if models:
            ts_fit_kwargs["hyperparameters"] = {m: {} for m in models}
        
        predictor.fit(**ts_fit_kwargs)
    except Exception as e:
        return {"status": "error", "message": f"Time series training failed: {str(e)}"}
    
    elapsed = time.time() - start_time
    
    # ── Generate forecasts ──
    try:
        predict_kwargs = {}
        if known_covariates_cols:
            try:
                future_known = predictor.make_future_data_frame(ts_df)
                if holiday_country:
                    import holidays as holidays_lib
                    country_holidays = holidays_lib.country_holidays(holiday_country)
                    dates = future_known.index.get_level_values('timestamp')
                    future_known['is_holiday'] = [
                        1.0 if d.date() in country_holidays else 0.0 for d in dates
                    ]
                predict_kwargs["known_covariates"] = future_known
            except Exception:
                pass
        forecasts = predictor.predict(ts_df, **predict_kwargs)
    except Exception as e:
        return {"status": "error", "message": f"Forecasting failed: {str(e)}"}
    
    # ── Leaderboard ──
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard_data = []
    for _, row in leaderboard.head(10).iterrows():
        leaderboard_data.append({
            "model": str(row.get("model", "")),
            "score_val": round(float(row.get("score_val", 0)), 4),
            "fit_time": round(float(row.get("fit_time", 0)), 1),
        })
    
    best_model = predictor.model_best if hasattr(predictor, 'model_best') else leaderboard_data[0]["model"] if leaderboard_data else "unknown"
    best_score = leaderboard_data[0]["score_val"] if leaderboard_data else None
    
    # ── Save forecasts ──
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        forecast_df = forecasts.reset_index()
        forecast_df.to_csv(output_path, index=False)
    except Exception:
        output_path = output_path  # Keep path but note it may not have saved
    
    # ── Forecast summary ──
    forecast_summary = {}
    try:
        mean_col = "mean" if "mean" in forecasts.columns else forecasts.columns[0]
        forecast_values = forecasts[mean_col].values
        forecast_summary = {
            "mean_forecast": round(float(np.mean(forecast_values)), 2),
            "min_forecast": round(float(np.min(forecast_values)), 2),
            "max_forecast": round(float(np.max(forecast_values)), 2),
            "forecast_std": round(float(np.std(forecast_values)), 2),
        }
    except Exception:
        pass
    
    results = {
        "status": "success",
        "task_type": "time_series_forecasting",
        "target_col": target_col,
        "time_col": time_col,
        "forecast_horizon": forecast_horizon,
        "frequency": freq,
        "n_series": df[id_col].nunique() if id_col != "__series_id" else 1,
        "n_data_points": len(df),
        "best_model": best_model,
        "best_score": best_score,
        "n_models_trained": len(leaderboard),
        "training_time_seconds": round(elapsed, 1),
        "leaderboard": leaderboard_data,
        "forecast_summary": forecast_summary,
        "output_path": output_path,
        "model_path": output_dir,
    }
    
    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"✅ TIME SERIES FORECASTING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Models trained: {len(leaderboard)}")
    print(f"🏆 Best model: {best_model}")
    print(f"📈 Best score: {best_score}")
    print(f"🔮 Forecast: {forecast_horizon} periods ahead")
    if forecast_summary:
        print(f"📉 Forecast range: {forecast_summary.get('min_forecast')} to {forecast_summary.get('max_forecast')}")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    print(f"💾 Forecasts saved: {output_path}")
    if leaderboard_data:
        print(f"\n📋 Leaderboard:")
        for i, entry in enumerate(leaderboard_data[:5], 1):
            print(f"   {i}. {entry['model']}: {entry['score_val']:.4f}")
    print(f"{'='*60}\n")
    
    return results


# ============================================================
# POST-TRAINING OPTIMIZATION (#1, #2, #6, #8, #9, #24)
# ============================================================

def optimize_autogluon_model(
    model_path: str,
    operation: str,
    data_path: Optional[str] = None,
    metric: Optional[str] = None,
    models_to_delete: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Post-training optimization on a trained AutoGluon model.
    
    Operations:
      - refit_full: Re-train best models on 100% data (no held-out fold) for deployment
      - distill: Compress ensemble into a single lighter model via knowledge distillation
      - calibrate_threshold: Optimize binary classification threshold for best F1/precision/recall
      - deploy_optimize: Strip training artifacts for minimal deployment footprint
      - delete_models: Remove specific models to free resources
    
    Args:
        model_path: Path to saved AutoGluon model directory
        operation: One of 'refit_full', 'distill', 'calibrate_threshold', 'deploy_optimize', 'delete_models'
        data_path: Path to dataset (required for distill, calibrate_threshold)
        metric: Metric to optimize for calibrate_threshold: 'f1', 'balanced_accuracy', 'precision', 'recall'
        models_to_delete: List of model names to delete (for delete_models operation)
        output_dir: Directory for optimized model output (for deploy_optimize)
    
    Returns:
        Dictionary with optimization results
    """
    TabularPredictor, TabularDataset = _ensure_autogluon_tabular()
    
    if not Path(model_path).exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}
    
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}
    
    print(f"\n🔧 AutoGluon Model Optimization: {operation}")
    print(f"   📁 Model: {model_path}")
    
    try:
        if operation == "refit_full":
            refit_map = predictor.refit_full()
            refit_models = list(refit_map.values())
            new_leaderboard = predictor.leaderboard(silent=True)
            
            leaderboard_data = []
            for _, row in new_leaderboard.head(10).iterrows():
                leaderboard_data.append({
                    "model": str(row.get("model", "")),
                    "score_val": round(float(row.get("score_val", 0)), 4),
                })
            
            print(f"   ✅ Models refit on 100% data: {refit_models}")
            return {
                "status": "success",
                "operation": "refit_full",
                "message": "Models re-trained on 100% data (no held-out folds) for deployment",
                "refit_models": refit_models,
                "original_best": predictor.model_best,
                "leaderboard": leaderboard_data,
                "model_path": model_path
            }
        
        elif operation == "distill":
            if not data_path or not Path(data_path).exists():
                return {"status": "error", "message": "data_path required for distillation"}
            
            train_data = TabularDataset(data_path)
            resource_config = _get_resource_config()
            
            distilled = predictor.distill(
                train_data=train_data,
                time_limit=resource_config["time_limit"],
                augment_method='spunge'
            )
            
            new_leaderboard = predictor.leaderboard(silent=True)
            leaderboard_data = []
            for _, row in new_leaderboard.head(10).iterrows():
                leaderboard_data.append({
                    "model": str(row.get("model", "")),
                    "score_val": round(float(row.get("score_val", 0)), 4),
                })
            
            print(f"   ✅ Ensemble distilled into: {distilled}")
            return {
                "status": "success",
                "operation": "distill",
                "message": "Ensemble distilled into lighter model(s) via knowledge distillation",
                "distilled_models": distilled,
                "best_model": predictor.model_best,
                "leaderboard": leaderboard_data,
                "model_path": model_path
            }
        
        elif operation == "calibrate_threshold":
            if not data_path or not Path(data_path).exists():
                return {"status": "error", "message": "data_path required for threshold calibration"}
            
            if predictor.problem_type != 'binary':
                return {"status": "error", "message": "Threshold calibration only works for binary classification"}
            
            test_data = TabularDataset(data_path)
            metric = metric or "f1"
            
            threshold, score = predictor.calibrate_decision_threshold(
                data=test_data,
                metric=metric
            )
            
            print(f"   ✅ Optimal threshold: {threshold:.4f} ({metric}={score:.4f})")
            return {
                "status": "success",
                "operation": "calibrate_threshold",
                "optimal_threshold": round(float(threshold), 4),
                "score_at_threshold": round(float(score), 4),
                "metric": metric,
                "message": f"Optimal threshold: {threshold:.4f} (default was 0.5), {metric}={score:.4f}",
                "model_path": model_path
            }
        
        elif operation == "deploy_optimize":
            output_dir = output_dir or model_path + "_deploy"
            
            size_before = sum(
                f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file()
            ) / (1024 * 1024)
            
            deploy_path = predictor.clone_for_deployment(output_dir)
            
            deploy_predictor = TabularPredictor.load(deploy_path)
            deploy_predictor.save_space()
            
            size_after = sum(
                f.stat().st_size for f in Path(deploy_path).rglob('*') if f.is_file()
            ) / (1024 * 1024)
            
            print(f"   ✅ Optimized: {size_before:.1f}MB → {size_after:.1f}MB")
            return {
                "status": "success",
                "operation": "deploy_optimize",
                "message": f"Model optimized for deployment: {size_before:.1f}MB → {size_after:.1f}MB ({(1-size_after/max(size_before,0.01))*100:.0f}% reduction)",
                "size_before_mb": round(size_before, 1),
                "size_after_mb": round(size_after, 1),
                "deploy_path": str(deploy_path),
                "best_model": deploy_predictor.model_best
            }
        
        elif operation == "delete_models":
            if not models_to_delete:
                return {"status": "error", "message": "models_to_delete list required"}
            
            before_count = len(predictor.model_names())
            predictor.delete_models(models_to_delete=models_to_delete, dry_run=False)
            after_count = len(predictor.model_names())
            
            print(f"   ✅ Deleted {before_count - after_count} models")
            return {
                "status": "success",
                "operation": "delete_models",
                "message": f"Deleted {before_count - after_count} models ({before_count} → {after_count})",
                "remaining_models": predictor.model_names(),
                "best_model": predictor.model_best,
                "model_path": model_path
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown operation '{operation}'. Choose: refit_full, distill, calibrate_threshold, deploy_optimize, delete_models"
            }
    
    except Exception as e:
        return {"status": "error", "message": f"Optimization failed: {str(e)}"}


# ============================================================
# MODEL ANALYSIS & INSPECTION (#19 + extended leaderboard)
# ============================================================

def analyze_autogluon_model(
    model_path: str,
    data_path: Optional[str] = None,
    operation: str = "summary"
) -> Dict[str, Any]:
    """
    Inspect and analyze a trained AutoGluon model.
    
    Operations:
      - summary: Extended leaderboard with detailed model info (stack levels, memory, etc.)
      - transform_features: Returns the internally transformed feature matrix
      - info: Comprehensive model metadata and training summary
    
    Args:
        model_path: Path to saved AutoGluon model directory
        data_path: Path to dataset (required for transform_features)
        operation: One of 'summary', 'transform_features', 'info'
    
    Returns:
        Dictionary with analysis results
    """
    TabularPredictor, TabularDataset = _ensure_autogluon_tabular()
    
    if not Path(model_path).exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}
    
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}
    
    try:
        if operation == "summary":
            leaderboard = predictor.leaderboard(extra_info=True, silent=True)
            
            leaderboard_data = []
            for _, row in leaderboard.iterrows():
                entry = {"model": str(row.get("model", ""))}
                for col in leaderboard.columns:
                    if col != "model":
                        val = row[col]
                        try:
                            entry[str(col)] = round(float(val), 4) if isinstance(val, (int, float, np.floating)) else str(val)
                        except (ValueError, TypeError):
                            entry[str(col)] = str(val)
                leaderboard_data.append(entry)
            
            return {
                "status": "success",
                "operation": "summary",
                "best_model": predictor.model_best,
                "problem_type": predictor.problem_type,
                "eval_metric": str(predictor.eval_metric),
                "n_models": len(leaderboard),
                "model_names": predictor.model_names(),
                "leaderboard": leaderboard_data
            }
        
        elif operation == "transform_features":
            if not data_path or not Path(data_path).exists():
                return {"status": "error", "message": "data_path required for transform_features"}
            
            data = TabularDataset(data_path)
            transformed = predictor.transform_features(data)
            
            output_path = "./outputs/autogluon_transformed_features.csv"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            transformed.to_csv(output_path, index=False)
            
            return {
                "status": "success",
                "operation": "transform_features",
                "original_shape": list(data.shape),
                "transformed_shape": list(transformed.shape),
                "original_columns": list(data.columns[:20]),
                "transformed_columns": list(transformed.columns[:30]),
                "output_path": output_path,
                "message": f"Features transformed: {data.shape[1]} original → {transformed.shape[1]} engineered"
            }
        
        elif operation == "info":
            info = predictor.info()
            
            safe_info = {}
            for key, val in info.items():
                try:
                    json.dumps(val)
                    safe_info[key] = val
                except (TypeError, ValueError):
                    safe_info[key] = str(val)
            
            return {
                "status": "success",
                "operation": "info",
                "model_info": safe_info
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown operation '{operation}'. Choose: summary, transform_features, info"
            }
    
    except Exception as e:
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}


# ============================================================
# INCREMENTAL TRAINING (#3, #5)
# ============================================================

def extend_autogluon_training(
    model_path: str,
    operation: str = "fit_extra",
    data_path: Optional[str] = None,
    time_limit: int = 60,
    hyperparameters: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Add models or re-fit ensemble on an existing AutoGluon predictor.
    
    Operations:
      - fit_extra: Train additional models/hyperparameters without retraining from scratch
      - fit_weighted_ensemble: Re-fit the weighted ensemble layer on existing base models
    
    Args:
        model_path: Path to saved AutoGluon model directory
        operation: 'fit_extra' or 'fit_weighted_ensemble'
        data_path: Path to training data (required for fit_extra)
        time_limit: Additional training time in seconds
        hyperparameters: Model hyperparameters dict for fit_extra.
            e.g. {"GBM": {"num_boost_round": 500}, "RF": {}}
    
    Returns:
        Dictionary with updated model info
    """
    TabularPredictor, TabularDataset = _ensure_autogluon_tabular()
    
    if not Path(model_path).exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}
    
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}
    
    before_models = predictor.model_names()
    print(f"\n🔧 Extending AutoGluon Model: {operation}")
    print(f"   📁 Model: {model_path}")
    print(f"   📊 Current models: {len(before_models)}")
    
    try:
        if operation == "fit_extra":
            if not data_path or not Path(data_path).exists():
                return {"status": "error", "message": "data_path required for fit_extra"}
            
            resource_config = _get_resource_config()
            
            hp = hyperparameters or {
                "GBM": [
                    {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
                    {"num_boost_round": 500},
                ],
                "RF": [
                    {"criterion": "gini", "ag_args": {"name_suffix": "Gini"}},
                    {"criterion": "entropy", "ag_args": {"name_suffix": "Entr"}},
                ],
            }
            
            predictor.fit_extra(
                hyperparameters=hp,
                time_limit=min(time_limit, resource_config["time_limit"]),
                num_cpus=resource_config["num_cpus"],
                num_gpus=0
            )
        
        elif operation == "fit_weighted_ensemble":
            predictor.fit_weighted_ensemble()
        
        else:
            return {
                "status": "error",
                "message": f"Unknown operation '{operation}'. Choose: fit_extra, fit_weighted_ensemble"
            }
        
        after_models = predictor.model_names()
        leaderboard = predictor.leaderboard(silent=True)
        
        leaderboard_data = []
        for _, row in leaderboard.head(10).iterrows():
            leaderboard_data.append({
                "model": str(row.get("model", "")),
                "score_val": round(float(row.get("score_val", 0)), 4),
                "fit_time": round(float(row.get("fit_time", 0)), 1),
            })
        
        new_models = [m for m in after_models if m not in before_models]
        
        print(f"   ✅ New models added: {len(new_models)}")
        print(f"   🏆 Best model: {predictor.model_best}")
        
        return {
            "status": "success",
            "operation": operation,
            "models_before": len(before_models),
            "models_after": len(after_models),
            "new_models": new_models,
            "best_model": predictor.model_best,
            "leaderboard": leaderboard_data,
            "model_path": model_path
        }
    
    except Exception as e:
        return {"status": "error", "message": f"Extension failed: {str(e)}"}


# ============================================================
# MULTI-LABEL PREDICTION (#14)
# ============================================================

def train_multilabel_autogluon(
    file_path: str,
    target_cols: List[str],
    time_limit: int = 120,
    presets: str = "medium_quality",
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train multi-label prediction using AutoGluon's MultilabelPredictor.
    Predicts multiple target columns simultaneously by training separate
    TabularPredictors per label with shared feature engineering.
    
    Args:
        file_path: Path to CSV/Parquet dataset
        target_cols: List of columns to predict (e.g. ['label1', 'label2', 'label3'])
        time_limit: Max training time per label in seconds
        presets: Quality preset
        output_dir: Where to save trained model
    
    Returns:
        Dictionary with per-label results and overall performance
    """
    try:
        from autogluon.tabular import TabularDataset, MultilabelPredictor
    except ImportError:
        return {
            "status": "error",
            "message": "MultilabelPredictor not available. Ensure autogluon.tabular>=1.2 is installed."
        }
    
    start_time = time.time()
    output_dir = output_dir or "./outputs/autogluon_multilabel"
    
    if not Path(file_path).exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    try:
        data = TabularDataset(file_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    missing_cols = [c for c in target_cols if c not in data.columns]
    if missing_cols:
        return {
            "status": "error",
            "message": f"Target columns not found: {missing_cols}. Available: {list(data.columns)}"
        }
    
    print(f"\n🚀 AutoGluon Multi-Label Training...")
    print(f"   📁 Dataset: {file_path}")
    print(f"   🎯 Targets: {target_cols}")
    print(f"   📐 Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
    
    resource_config = _get_resource_config()
    effective_time_limit = min(time_limit, resource_config["time_limit"])
    
    if Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    
    try:
        multi_predictor = MultilabelPredictor(
            labels=target_cols,
            path=output_dir
        )
        
        multi_predictor.fit(
            train_data=data,
            time_limit=effective_time_limit,
            presets=presets
        )
    except Exception as e:
        return {"status": "error", "message": f"Multi-label training failed: {str(e)}"}
    
    elapsed = time.time() - start_time
    
    per_label_results = {}
    for label in target_cols:
        try:
            label_predictor = multi_predictor.get_predictor(label)
            lb = label_predictor.leaderboard(silent=True)
            per_label_results[label] = {
                "best_model": label_predictor.model_best,
                "best_score": round(float(lb.iloc[0]["score_val"]), 4) if len(lb) > 0 else None,
                "n_models": len(lb),
                "problem_type": label_predictor.problem_type
            }
        except Exception:
            per_label_results[label] = {"error": "Could not retrieve results"}
    
    print(f"\n{'='*60}")
    print(f"✅ MULTI-LABEL TRAINING COMPLETE")
    print(f"{'='*60}")
    for label, result in per_label_results.items():
        score = result.get('best_score', 'N/A')
        model = result.get('best_model', 'N/A')
        print(f"   🎯 {label}: {model} (score: {score})")
    print(f"   ⏱️  Total time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    
    return {
        "status": "success",
        "task_type": "multilabel",
        "n_labels": len(target_cols),
        "labels": target_cols,
        "per_label_results": per_label_results,
        "training_time_seconds": round(elapsed, 1),
        "model_path": output_dir,
        "output_path": output_dir
    }


# ============================================================
# TIME SERIES BACKTESTING (#33)
# ============================================================

def backtest_timeseries(
    file_path: str,
    target_col: str,
    time_col: str,
    forecast_horizon: int = 30,
    id_col: Optional[str] = None,
    freq: Optional[str] = None,
    num_val_windows: int = 3,
    time_limit: int = 120,
    presets: str = "medium_quality",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Backtest time series models using multiple validation windows.
    
    Trains models with multi-window cross-validation for robust performance
    estimates. More reliable than a single train/test split.
    
    Args:
        file_path: Path to time series CSV/Parquet
        target_col: Column with values to forecast
        time_col: Column with timestamps/dates
        forecast_horizon: Periods to predict per window
        id_col: Column identifying different series
        freq: Frequency string ('D', 'h', 'W', 'MS')
        num_val_windows: Number of backtesting windows (default: 3)
        time_limit: Max training time in seconds
        presets: Quality preset
        output_path: Path to save backtest predictions CSV
    
    Returns:
        Dictionary with per-window evaluation and aggregate metrics
    """
    TimeSeriesPredictor, TimeSeriesDataFrame = _ensure_autogluon_timeseries()
    
    start_time = time.time()
    output_dir = "./outputs/autogluon_ts_backtest"
    output_path = output_path or "./outputs/autogluon_backtest.csv"
    
    if not Path(file_path).exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    print(f"\n📊 Time Series Backtesting ({num_val_windows} windows)...")
    print(f"   📁 Dataset: {file_path}")
    print(f"   🎯 Target: {target_col}")
    print(f"   🔮 Horizon: {forecast_horizon} periods × {num_val_windows} windows")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
    except Exception:
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load data: {str(e)}"}
    
    if target_col not in df.columns or time_col not in df.columns:
        return {"status": "error", "message": f"Columns not found. Available: {list(df.columns)}"}
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    
    if id_col is None or id_col not in df.columns:
        id_col_name = "__series_id"
        df[id_col_name] = "series_0"
    else:
        id_col_name = id_col
    
    # Auto-detect frequency
    if freq is None:
        time_diffs = df[time_col].diff().dropna()
        median_diff = time_diffs.median()
        if median_diff <= pd.Timedelta(hours=2):
            freq = "h"
        elif median_diff <= pd.Timedelta(days=1.5):
            freq = "D"
        elif median_diff <= pd.Timedelta(days=8):
            freq = "W"
        elif median_diff <= pd.Timedelta(days=35):
            freq = "MS"
        else:
            freq = "D"
    
    try:
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df, id_column=id_col_name, timestamp_column=time_col
        )
    except Exception as e:
        return {"status": "error", "message": f"Failed to create time series: {str(e)}"}
    
    if Path(output_dir).exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    
    resource_config = _get_resource_config()
    
    try:
        predictor = TimeSeriesPredictor(
            target=target_col,
            prediction_length=forecast_horizon,
            path=output_dir,
            freq=freq
        )
        
        predictor.fit(
            train_data=ts_df,
            time_limit=min(time_limit, resource_config["time_limit"]),
            presets=presets,
            num_val_windows=num_val_windows
        )
    except Exception as e:
        return {"status": "error", "message": f"Backtest training failed: {str(e)}"}
    
    elapsed = time.time() - start_time
    
    # Get backtest predictions
    try:
        bt_preds = predictor.backtest_predictions()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        bt_df = bt_preds.reset_index()
        bt_df.to_csv(output_path, index=False)
    except Exception:
        bt_preds = None
    
    # Leaderboard
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard_data = []
    for _, row in leaderboard.head(10).iterrows():
        leaderboard_data.append({
            "model": str(row.get("model", "")),
            "score_val": round(float(row.get("score_val", 0)), 4),
            "fit_time": round(float(row.get("fit_time", 0)), 1),
        })
    
    best_model = predictor.model_best if hasattr(predictor, 'model_best') else "unknown"
    best_score = leaderboard_data[0]["score_val"] if leaderboard_data else None
    
    print(f"\n{'='*60}")
    print(f"✅ BACKTESTING COMPLETE ({num_val_windows} windows)")
    print(f"{'='*60}")
    print(f"🏆 Best: {best_model} (score: {best_score})")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"{'='*60}\n")
    
    return {
        "status": "success",
        "task_type": "backtesting",
        "num_val_windows": num_val_windows,
        "forecast_horizon": forecast_horizon,
        "best_model": best_model,
        "best_score": best_score,
        "n_models_trained": len(leaderboard),
        "training_time_seconds": round(elapsed, 1),
        "leaderboard": leaderboard_data,
        "output_path": output_path,
        "model_path": output_dir
    }


# ============================================================
# TIME SERIES ANALYSIS (#34, #35, #37)
# ============================================================

def analyze_timeseries_model(
    model_path: str,
    data_path: str,
    time_col: str,
    id_col: Optional[str] = None,
    operation: str = "feature_importance",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze a trained AutoGluon time series model.
    
    Operations:
      - feature_importance: Permutation importance of covariates
      - plot: Generate forecast vs actuals visualization
      - make_future_dataframe: Generate future timestamp skeleton for prediction
    
    Args:
        model_path: Path to saved AutoGluon TimeSeriesPredictor
        data_path: Path to time series data
        time_col: Column with timestamps/dates
        id_col: Column identifying different series
        operation: One of 'feature_importance', 'plot', 'make_future_dataframe'
        output_path: Path to save output
    
    Returns:
        Dictionary with analysis results
    """
    TimeSeriesPredictor, TimeSeriesDataFrame = _ensure_autogluon_timeseries()
    
    if not Path(model_path).exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}
    if not Path(data_path).exists():
        return {"status": "error", "message": f"Data not found: {data_path}"}
    
    try:
        predictor = TimeSeriesPredictor.load(model_path)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {str(e)}"}
    
    # Reconstruct TimeSeriesDataFrame
    try:
        df = pd.read_csv(data_path)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        if id_col is None or id_col not in df.columns:
            id_col_name = "__series_id"
            df[id_col_name] = "series_0"
        else:
            id_col_name = id_col
        
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df, id_column=id_col_name, timestamp_column=time_col
        )
    except Exception as e:
        return {"status": "error", "message": f"Failed to create time series data: {str(e)}"}
    
    try:
        if operation == "feature_importance":
            fi = predictor.feature_importance(ts_df)
            
            fi_data = []
            if isinstance(fi, pd.DataFrame):
                for feat in fi.index:
                    row_data = {"feature": str(feat)}
                    for col in fi.columns:
                        try:
                            row_data[str(col)] = round(float(fi.loc[feat, col]), 4)
                        except (TypeError, ValueError):
                            row_data[str(col)] = str(fi.loc[feat, col])
                    fi_data.append(row_data)
            
            return {
                "status": "success",
                "operation": "feature_importance",
                "features": fi_data,
                "model_path": model_path,
                "message": f"Feature importance computed for {len(fi_data)} features"
            }
        
        elif operation == "plot":
            output_path = output_path or "./outputs/plots/ts_forecast_plot.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            predictions = predictor.predict(ts_df)
            
            try:
                predictor.plot(ts_df, predictions, quantile_levels=[0.1, 0.9])
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            except Exception:
                # Fallback: manual plot
                fig, ax = plt.subplots(figsize=(12, 6))
                target = predictor.target
                
                for item_id in list(ts_df.item_ids)[:3]:
                    actual = ts_df.loc[item_id][target].tail(100)
                    ax.plot(actual.index, actual.values, label=f'Actual ({item_id})', linewidth=1.5)
                    
                    if item_id in predictions.item_ids:
                        pred = predictions.loc[item_id]
                        mean_col = "mean" if "mean" in pred.columns else pred.columns[0]
                        ax.plot(pred.index, pred[mean_col].values, '--', label=f'Forecast ({item_id})', linewidth=1.5)
                
                ax.set_title(f'Time Series Forecast - {predictor.model_best}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            return {
                "status": "success",
                "operation": "plot",
                "output_path": output_path,
                "message": f"Forecast plot saved to {output_path}"
            }
        
        elif operation == "make_future_dataframe":
            output_path = output_path or "./outputs/future_dataframe.csv"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            future_df = predictor.make_future_data_frame(ts_df)
            future_df.reset_index().to_csv(output_path, index=False)
            
            return {
                "status": "success",
                "operation": "make_future_dataframe",
                "shape": list(future_df.shape),
                "columns": list(future_df.columns) if hasattr(future_df, 'columns') else [],
                "output_path": output_path,
                "message": f"Future dataframe generated: {len(future_df)} rows"
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown operation '{operation}'. Choose: feature_importance, plot, make_future_dataframe"
            }
    
    except Exception as e:
        return {"status": "error", "message": f"Analysis failed: {str(e)}"}
