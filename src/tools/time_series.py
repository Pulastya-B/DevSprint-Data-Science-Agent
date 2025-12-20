"""
Time Series & Forecasting Tools
Tools for time series analysis, forecasting, seasonality detection, and feature engineering.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Lazy imports - only import when needed to avoid blocking app startup
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.seasonal import seasonal_decompose, STL
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from prophet import Prophet
import pandas as pd

from utils.polars_helpers import load_dataframe, save_dataframe
from utils.validation import validate_file_exists, validate_file_format, validate_dataframe, validate_column_exists


def forecast_time_series(
    file_path: str,
    time_col: str,
    target_col: str,
    forecast_horizon: int = 30,
    method: str = "prophet",
    seasonal_period: Optional[int] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Forecast time series using ARIMA, SARIMA, Prophet, or Exponential Smoothing.
    
    Args:
        file_path: Path to time series dataset
        time_col: Time/date column name
        target_col: Target variable to forecast
        forecast_horizon: Number of periods to forecast ahead
        method: Forecasting method ('arima', 'sarima', 'prophet', 'exponential_smoothing')
        seasonal_period: Seasonal period (e.g., 7 for weekly, 12 for monthly)
        output_path: Path to save forecast results
        
    Returns:
        Dictionary with forecast values and metrics
    """
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, time_col)
    validate_column_exists(df, target_col)
    
    # Sort by time
    df = df.sort(time_col)
    
    # Lazy import of time series libraries
    try:
        if method == "prophet":
            from prophet import Prophet
        elif method in ["arima", "sarima"]:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        elif method == "exponential_smoothing":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as e:
        return {
            'status': 'error',
            'message': f"Required library not installed for {method}: {str(e)}"
        }
    
    print(f"📈 Forecasting with {method} (horizon={forecast_horizon})...")
    
    # Convert to pandas for time series libraries
    df_pd = df.to_pandas()
    
    if method == "prophet":
        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df_pd[time_col]),
            'y': df_pd[target_col]
        })
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_horizon)
        
        result = {
            'method': 'prophet',
            'forecast': forecast_values.to_dict('records'),
            'model_components': {
                'trend': forecast['trend'].tail(forecast_horizon).tolist(),
                'weekly': forecast.get('weekly', pd.Series([0]*forecast_horizon)).tail(forecast_horizon).tolist()
            }
        }
        
    elif method == "arima":
        # ARIMA model
        ts_data = df_pd.set_index(time_col)[target_col]
        
        # Auto-determine order (p,d,q) - simplified version
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_horizon)
        forecast_index = pd.date_range(start=ts_data.index[-1], periods=forecast_horizon+1, freq='D')[1:]
        
        result = {
            'method': 'arima',
            'order': '(1,1,1)',
            'forecast': [{'date': str(date), 'value': float(val)} for date, val in zip(forecast_index, forecast)],
            'aic': float(fitted_model.aic),
            'bic': float(fitted_model.bic)
        }
        
    elif method == "sarima":
        if not seasonal_period:
            seasonal_period = 7  # Default weekly
        
        ts_data = df_pd.set_index(time_col)[target_col]
        
        # SARIMA model
        model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, seasonal_period))
        fitted_model = model.fit(disp=False)
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_horizon)
        forecast_index = pd.date_range(start=ts_data.index[-1], periods=forecast_horizon+1, freq='D')[1:]
        
        result = {
            'method': 'sarima',
            'order': '(1,1,1)',
            'seasonal_order': f'(1,1,1,{seasonal_period})',
            'forecast': [{'date': str(date), 'value': float(val)} for date, val in zip(forecast_index, forecast)],
            'aic': float(fitted_model.aic)
        }
        
    elif method == "exponential_smoothing":
        ts_data = df_pd.set_index(time_col)[target_col]
        
        # Exponential Smoothing
        model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=seasonal_period if seasonal_period else 12,
            trend='add',
            seasonal='add' if seasonal_period else None
        )
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=forecast_horizon)
        forecast_index = pd.date_range(start=ts_data.index[-1], periods=forecast_horizon+1, freq='D')[1:]
        
        result = {
            'method': 'exponential_smoothing',
            'forecast': [{'date': str(date), 'value': float(val)} for date, val in zip(forecast_index, forecast)]
        }
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Save forecast
    if output_path:
        forecast_df = pl.DataFrame(result['forecast'])
        save_dataframe(forecast_df, output_path)
        print(f"💾 Forecast saved to: {output_path}")
    
    result['status'] = 'success'
    result['forecast_horizon'] = forecast_horizon
    result['output_path'] = output_path
    
    return result


def detect_seasonality_trends(
    file_path: str,
    time_col: str,
    target_col: str,
    period: Optional[int] = None,
    method: str = "stl",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect seasonality and trends in time series using STL decomposition.
    
    Args:
        file_path: Path to time series dataset
        time_col: Time/date column
        target_col: Target variable
        period: Seasonal period (None = auto-detect)
        method: Decomposition method ('stl', 'classical')
        output_path: Path to save decomposition results
        
    Returns:
        Dictionary with trend, seasonal, and residual components
    """
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, time_col)
    validate_column_exists(df, target_col)
    
    # Sort by time
    df = df.sort(time_col)
    
    # Lazy import of time series libraries
    try:
        if method == "stl":
            from statsmodels.tsa.seasonal import STL
        else:
            from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError as e:
        return {
            'status': 'error',
            'message': f"Required library not installed: {str(e)}"
        }
    
    print(f"🔍 Detecting seasonality and trends using {method}...")
    
    # Convert to pandas
    df_pd = df.to_pandas()
    ts_data = df_pd.set_index(time_col)[target_col]
    
    # Auto-detect period using FFT if not provided
    if period is None:
        from scipy.fft import fft
        from scipy.signal import find_peaks
        
        # Remove trend
        detrended = ts_data - ts_data.rolling(window=min(len(ts_data)//10, 30), center=True).mean()
        detrended = detrended.fillna(method='bfill').fillna(method='ffill')
        
        # FFT
        fft_vals = np.abs(fft(detrended.values))
        freqs = np.fft.fftfreq(len(fft_vals))
        
        # Find peaks
        peaks, _ = find_peaks(fft_vals[:len(fft_vals)//2], height=np.max(fft_vals)*0.1)
        
        if len(peaks) > 0:
            # Get dominant frequency
            dominant_freq = freqs[peaks[0]]
            period = int(1 / abs(dominant_freq)) if dominant_freq != 0 else 7
        else:
            period = 7  # Default weekly
        
        print(f"📊 Auto-detected period: {period}")
    
    # Perform decomposition
    if method == "stl":
        # STL decomposition (more robust)
        stl = STL(ts_data, seasonal=period*2+1, trend=period*4+1)
        result_decomp = stl.fit()
        
        trend = result_decomp.trend
        seasonal = result_decomp.seasonal
        residual = result_decomp.resid
        
    else:
        # Classical decomposition
        result_decomp = seasonal_decompose(ts_data, model='additive', period=period)
        trend = result_decomp.trend
        seasonal = result_decomp.seasonal
        residual = result_decomp.resid
    
    # Calculate seasonality strength
    var_resid = np.var(residual.dropna())
    var_seasonal_resid = np.var((seasonal + residual).dropna())
    seasonality_strength = 1 - (var_resid / var_seasonal_resid) if var_seasonal_resid > 0 else 0
    
    # Calculate trend strength
    var_detrended = np.var((ts_data - trend).dropna())
    trend_strength = 1 - (var_resid / var_detrended) if var_detrended > 0 else 0
    
    # Autocorrelation analysis
    from statsmodels.tsa.stattools import acf
    acf_values = acf(ts_data.dropna(), nlags=min(40, len(ts_data)//2))
    
    # Create decomposition dataframe
    decomp_df = pl.DataFrame({
        'time': df[time_col].to_list(),
        'original': ts_data.values,
        'trend': trend.fillna(0).values,
        'seasonal': seasonal.fillna(0).values,
        'residual': residual.fillna(0).values
    })
    
    # Save if output path provided
    if output_path:
        save_dataframe(decomp_df, output_path)
        print(f"💾 Decomposition saved to: {output_path}")
    
    return {
        'status': 'success',
        'method': method,
        'detected_period': period,
        'seasonality_strength': float(seasonality_strength),
        'trend_strength': float(trend_strength),
        'interpretation': {
            'seasonality': 'strong' if seasonality_strength > 0.6 else 'moderate' if seasonality_strength > 0.3 else 'weak',
            'trend': 'strong' if trend_strength > 0.6 else 'moderate' if trend_strength > 0.3 else 'weak'
        },
        'autocorrelation': acf_values[:min(10, len(acf_values))].tolist(),
        'output_path': output_path
    }


def create_time_series_features(
    file_path: str,
    time_col: str,
    target_col: str,
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    add_holiday_features: bool = True,
    country: str = "US",
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create comprehensive time series features including lags, rolling stats, and calendar features.
    
    Args:
        file_path: Path to time series dataset
        time_col: Time/date column
        target_col: Target variable
        lag_periods: Lag periods to create (e.g., [1, 7, 30])
        rolling_windows: Rolling window sizes (e.g., [7, 14, 30])
        add_holiday_features: Add holiday indicators
        country: Country for holiday calendar
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with feature engineering results
    """
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, time_col)
    validate_column_exists(df, target_col)
    
    # Sort by time
    df = df.sort(time_col)
    
    print("⏰ Creating time series features...")
    
    # Convert to pandas for easier datetime handling
    df_pd = df.to_pandas()
    df_pd[time_col] = pd.to_datetime(df_pd[time_col])
    df_pd = df_pd.set_index(time_col)
    
    created_features = []
    
    # Lag features
    if lag_periods is None:
        lag_periods = [1, 7, 14, 30]
    
    for lag in lag_periods:
        df_pd[f'{target_col}_lag_{lag}'] = df_pd[target_col].shift(lag)
        created_features.append(f'{target_col}_lag_{lag}')
    
    # Rolling window features
    if rolling_windows is None:
        rolling_windows = [7, 14, 30]
    
    for window in rolling_windows:
        df_pd[f'{target_col}_rolling_mean_{window}'] = df_pd[target_col].rolling(window=window).mean()
        df_pd[f'{target_col}_rolling_std_{window}'] = df_pd[target_col].rolling(window=window).std()
        df_pd[f'{target_col}_rolling_min_{window}'] = df_pd[target_col].rolling(window=window).min()
        df_pd[f'{target_col}_rolling_max_{window}'] = df_pd[target_col].rolling(window=window).max()
        
        created_features.extend([
            f'{target_col}_rolling_mean_{window}',
            f'{target_col}_rolling_std_{window}',
            f'{target_col}_rolling_min_{window}',
            f'{target_col}_rolling_max_{window}'
        ])
    
    # Exponential moving average
    df_pd[f'{target_col}_ema_7'] = df_pd[target_col].ewm(span=7).mean()
    df_pd[f'{target_col}_ema_30'] = df_pd[target_col].ewm(span=30).mean()
    created_features.extend([f'{target_col}_ema_7', f'{target_col}_ema_30'])
    
    # Calendar features
    df_pd['year'] = df_pd.index.year
    df_pd['month'] = df_pd.index.month
    df_pd['day'] = df_pd.index.day
    df_pd['dayofweek'] = df_pd.index.dayofweek
    df_pd['dayofyear'] = df_pd.index.dayofyear
    df_pd['quarter'] = df_pd.index.quarter
    df_pd['is_weekend'] = (df_pd.index.dayofweek >= 5).astype(int)
    df_pd['is_month_start'] = df_pd.index.is_month_start.astype(int)
    df_pd['is_month_end'] = df_pd.index.is_month_end.astype(int)
    
    # Cyclical encoding for periodic features
    df_pd['month_sin'] = np.sin(2 * np.pi * df_pd['month'] / 12)
    df_pd['month_cos'] = np.cos(2 * np.pi * df_pd['month'] / 12)
    df_pd['day_sin'] = np.sin(2 * np.pi * df_pd['day'] / 31)
    df_pd['day_cos'] = np.cos(2 * np.pi * df_pd['day'] / 31)
    df_pd['dayofweek_sin'] = np.sin(2 * np.pi * df_pd['dayofweek'] / 7)
    df_pd['dayofweek_cos'] = np.cos(2 * np.pi * df_pd['dayofweek'] / 7)
    
    created_features.extend([
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
        'is_weekend', 'is_month_start', 'is_month_end',
        'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'dayofweek_sin', 'dayofweek_cos'
    ])
    
    # Holiday features
    if add_holiday_features:
        try:
            import holidays
            country_holidays = holidays.country_holidays(country)
            df_pd['is_holiday'] = df_pd.index.map(lambda x: 1 if x in country_holidays else 0)
            
            # Days until next holiday
            holiday_dates = sorted([date for date in country_holidays if date >= df_pd.index.min()])
            df_pd['days_to_next_holiday'] = df_pd.index.map(
                lambda x: min([abs((hol - x).days) for hol in holiday_dates if hol >= x], default=365)
            )
            
            created_features.extend(['is_holiday', 'days_to_next_holiday'])
        except Exception as e:
            print(f"⚠️ Could not add holiday features: {str(e)}")
    
    # Convert back to polars
    df_pd = df_pd.reset_index()
    df_result = pl.from_pandas(df_pd)
    
    # Save if output path provided
    if output_path:
        save_dataframe(df_result, output_path)
        print(f"💾 Dataset with time series features saved to: {output_path}")
    
    return {
        'status': 'success',
        'features_created': len(created_features),
        'feature_names': created_features,
        'lag_periods': lag_periods,
        'rolling_windows': rolling_windows,
        'holiday_features_added': add_holiday_features,
        'output_path': output_path
    }
