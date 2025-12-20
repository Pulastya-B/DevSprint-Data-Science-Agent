"""Tools module initialization - All 44 tools."""

# Basic Tools (10)
from .data_profiling import (
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations,
    get_smart_summary  # NEW: Enhanced data summary
)

from .data_cleaning import (
    clean_missing_values,
    handle_outliers,
    fix_data_types
)

from .data_type_conversion import (
    force_numeric_conversion,
    smart_type_inference
)

# Data Wrangling Tools (3) - NEW
from .data_wrangling import (
    merge_datasets,
    concat_datasets,
    reshape_dataset
)

from .feature_engineering import (
    create_time_features,
    encode_categorical
)

from .model_training import (
    train_baseline_models,
    generate_model_report
)

# Advanced Analysis Tools (5)
from .advanced_analysis import (
    perform_eda_analysis,
    detect_model_issues,
    detect_anomalies,
    detect_and_handle_multicollinearity,
    perform_statistical_tests
)

# Advanced Feature Engineering Tools (4)
from .advanced_feature_engineering import (
    create_interaction_features,
    create_aggregation_features,
    engineer_text_features,
    auto_feature_engineering
)

# Advanced Preprocessing Tools (3)
from .advanced_preprocessing import (
    handle_imbalanced_data,
    perform_feature_scaling,
    split_data_strategically
)

# Advanced Training Tools (3)
from .advanced_training import (
    hyperparameter_tuning,
    train_ensemble_models,
    perform_cross_validation
)

# Business Intelligence Tools (4)
from .business_intelligence import (
    perform_cohort_analysis,
    perform_rfm_analysis,
    detect_causal_relationships,
    generate_business_insights
)

# Computer Vision Tools (3)
from .computer_vision import (
    extract_image_features,
    perform_image_clustering,
    analyze_tabular_image_hybrid
)

# NLP/Text Analytics Tools (4)
from .nlp_text_analytics import (
    perform_topic_modeling,
    perform_named_entity_recognition,
    analyze_sentiment_advanced,
    perform_text_similarity
)

# Production/MLOps Tools (5)
from .production_mlops import (
    monitor_model_drift,
    explain_predictions,
    generate_model_card,
    perform_ab_test_analysis,
    detect_feature_leakage
)

# Time Series Tools (3)
from .time_series import (
    forecast_time_series,
    detect_seasonality_trends,
    create_time_series_features
)

# Advanced Insights Tools (6) - NEW
from .advanced_insights import (
    analyze_root_cause,
    detect_trends_and_seasonality,
    detect_anomalies_advanced,
    perform_hypothesis_testing,
    analyze_distribution,
    perform_segment_analysis
)

# Automated Pipeline Tools (2) - NEW
from .auto_pipeline import (
    auto_ml_pipeline,
    auto_feature_selection
)

# Visualization Tools (5) - NEW
from .visualization_engine import (
    generate_all_plots,
    generate_data_quality_plots,
    generate_eda_plots,
    generate_model_performance_plots,
    generate_feature_importance_plot
)

# Interactive Plotly Visualizations (6) - NEW PHASE 2
from .plotly_visualizations import (
    generate_interactive_scatter,
    generate_interactive_histogram,
    generate_interactive_correlation_heatmap,
    generate_interactive_box_plots,
    generate_interactive_time_series,
    generate_plotly_dashboard
)

# EDA Report Generation (3) - NEW PHASE 2
from .eda_reports import (
    generate_sweetviz_report,
    generate_ydata_profiling_report,
    generate_combined_eda_report
)

# Code Interpreter (2) - NEW PHASE 2 - CRITICAL for True AI Agent
from .code_interpreter import (
    execute_python_code,
    execute_code_from_file
)

# Cloud Data Sources (4) - NEW: BigQuery Integration
from .cloud_data_sources import (
    load_bigquery_table,
    write_bigquery_table,
    profile_bigquery_table,
    query_bigquery
)

from .tools_registry import TOOLS, get_tool_by_name, get_all_tool_names

from tools.enhanced_feature_engineering import (
    create_ratio_features,
    create_statistical_features,
    create_log_features,
    create_binned_features,
)

__all__ = [
    # Basic Data Profiling (4) - UPDATED
    "profile_dataset",
    "detect_data_quality_issues",
    "analyze_correlations",
    "get_smart_summary",  # NEW
    
    # Basic Data Cleaning (3)
    "clean_missing_values",
    "handle_outliers",
    "fix_data_types",
    
    # Data Type Conversion (2)
    "force_numeric_conversion",
    "smart_type_inference",
    
    # Data Wrangling (3) - NEW
    "merge_datasets",
    "concat_datasets",
    "reshape_dataset",
    
    # Basic Feature Engineering (2)
    "create_time_features",
    "encode_categorical",
    
    # Basic Model Training (2)
    "train_baseline_models",
    "generate_model_report",
    
    # Advanced Analysis (5)
    "perform_eda_analysis",
    "detect_model_issues",
    "detect_anomalies",
    "detect_and_handle_multicollinearity",
    "perform_statistical_tests",
    
    # Advanced Feature Engineering (4)
    "create_interaction_features",
    "create_aggregation_features",
    "engineer_text_features",
    "auto_feature_engineering",
    
    # Advanced Preprocessing (3)
    "handle_imbalanced_data",
    "perform_feature_scaling",
    "split_data_strategically",
    
    # Advanced Training (3)
    "hyperparameter_tuning",
    "train_ensemble_models",
    "perform_cross_validation",
    
    # Business Intelligence (4)
    "perform_cohort_analysis",
    "perform_rfm_analysis",
    "detect_causal_relationships",
    "generate_business_insights",
    
    # Computer Vision (3)
    "extract_image_features",
    "perform_image_clustering",
    "analyze_tabular_image_hybrid",
    
    # NLP/Text Analytics (4)
    "perform_topic_modeling",
    "perform_named_entity_recognition",
    "analyze_sentiment_advanced",
    "perform_text_similarity",
    
    # Production/MLOps (5)
    "monitor_model_drift",
    "explain_predictions",
    "generate_model_card",
    "perform_ab_test_analysis",
    "detect_feature_leakage",
    
    # Time Series (3)
    "forecast_time_series",
    "detect_seasonality_trends",
    "create_time_series_features",
    
    # Advanced Insights (6) - NEW
    "analyze_root_cause",
    "detect_trends_and_seasonality",
    "detect_anomalies_advanced",
    "perform_hypothesis_testing",
    "analyze_distribution",
    "perform_segment_analysis",
    
    # Automated Pipeline (2) - NEW
    "auto_ml_pipeline",
    "auto_feature_selection",
    
    # Visualization (5) - NEW
    "generate_all_plots",
    "generate_data_quality_plots",
    "generate_eda_plots",
    "generate_model_performance_plots",
    "generate_feature_importance_plot",
    
    # Interactive Plotly Visualizations (6) - NEW PHASE 2
    "generate_interactive_scatter",
    "generate_interactive_histogram",
    "generate_interactive_correlation_heatmap",
    "generate_interactive_box_plots",
    "generate_interactive_time_series",
    "generate_plotly_dashboard",
    
    # EDA Report Generation (3) - NEW PHASE 2
    "generate_sweetviz_report",
    "generate_ydata_profiling_report",
    "generate_combined_eda_report",
    
    # Code Interpreter (2) - NEW PHASE 2 - CRITICAL for True AI Agent
    "execute_python_code",
    "execute_code_from_file",
    
    # Cloud Data Sources (4) - NEW: BigQuery Integration
    "load_bigquery_table",
    "write_bigquery_table",
    "profile_bigquery_table",
    "query_bigquery",
    
    # Enhanced Feature Engineering (4) - NEW
    "create_ratio_features",
    "create_statistical_features",
    "create_log_features",
    "create_binned_features",
    
    # Registry
    "TOOLS",
    "get_tool_by_name",
    "get_all_tool_names",
]
