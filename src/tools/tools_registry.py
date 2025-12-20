"""
Complete Tools Registry for Groq Function Calling - All 67 Tools
Defines all available tools in Groq's function calling format.
"""

TOOLS = [
    # ============================================
    # BASIC TOOLS (16)
    # ============================================
    
    # Data Profiling Tools (3)
    {
        "type": "function",
        "function": {
            "name": "profile_dataset",
            "description": "Get comprehensive statistics about a dataset including shape, data types, memory usage, null counts, and unique values. Use this as the first step to understand any new dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute or relative path to the CSV or Parquet file"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_data_quality_issues",
            "description": "Detect data quality issues including outliers (using IQR method), duplicate rows, inconsistent formats, and data anomalies. Returns a prioritized list of issues with severity levels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_correlations",
            "description": "Compute correlation matrix and identify top correlations. If a target column is specified, shows features most correlated with the target. Useful for feature selection and understanding relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target column name to analyze correlations with"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    
    # Data Cleaning Tools (3)
    {
        "type": "function",
        "function": {
            "name": "clean_missing_values",
            "description": "Handle missing values using appropriate strategies based on column type. Strategies include median/mean for numeric, mode for categorical, forward_fill for time series, or drop. In 'auto' mode, first drops columns with >threshold missing (default 40%), then imputes remaining columns. Will not impute ID columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "strategy": {
                        "oneOf": [
                            {
                                "type": "string",
                                "enum": ["auto"],
                                "description": "Use 'auto' to automatically decide strategies for all columns based on data type. First drops columns with >threshold missing, then imputes remaining columns."
                            },
                            {
                                "type": "object",
                                "description": "Dictionary mapping column names to strategies ('median', 'mean', 'mode', 'forward_fill', 'drop')",
                                "additionalProperties": {"type": "string"}
                            }
                        ],
                        "description": "Either 'auto' (string) to automatically handle all missing values, or a dictionary mapping specific columns to strategies"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save cleaned dataset"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "For 'auto' mode: drop columns with missing percentage above this threshold (default: 0.4 = 40%). Range: 0.0 to 1.0. For example, 0.7 means drop columns with >70% missing values."
                    }
                },
                "required": ["file_path", "strategy", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_outliers",
            "description": "Detect and handle outliers in numeric columns using IQR method. Methods: 'clip' (cap at boundaries), 'winsorize' (cap at percentiles), or 'remove' (delete rows).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["clip", "winsorize", "remove"],
                        "description": "Method to handle outliers"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of column names to check for outliers. Use 'all' to check all numeric columns."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save cleaned dataset"
                    }
                },
                "required": ["file_path", "method", "columns", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_data_types",
            "description": "Auto-detect and fix incorrect data types. Handles dates, booleans, categoricals, and numeric columns. Fixes common issues like 'null' strings and mixed types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "type_mapping": {
                        "type": "object",
                        "description": "Optional dictionary mapping column names to target types ('int', 'float', 'string', 'date', 'bool', 'category'). Use 'auto' for automatic detection.",
                        "additionalProperties": {"type": "string"}
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save dataset with fixed types"
                    }
                },
                "required": ["file_path", "output_path"]
            }
        }
    },
    
    # Data Type Conversion Tools (2)
    {
        "type": "function",
        "function": {
            "name": "force_numeric_conversion",
            "description": "CRITICAL TOOL: Force convert columns to numeric type even if detected as strings/objects. Essential for datasets with numeric columns stored as strings (with commas, spaces, currency symbols). Use this BEFORE encoding when you see 'no numeric features' errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of column names to force convert to numeric. Use ['all'] to auto-detect and convert all non-ID columns that look numeric."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save dataset with converted types"
                    },
                    "errors": {
                        "type": "string",
                        "enum": ["coerce", "raise"],
                        "description": "How to handle conversion errors. 'coerce' makes invalid values null (recommended), 'raise' throws error."
                    }
                },
                "required": ["file_path", "columns", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "smart_type_inference",
            "description": "Intelligently infer and fix data types for all columns by analyzing patterns. Goes beyond basic type detection to understand semantic meaning. Use when dataset has widespread type issues.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save dataset with inferred types"
                    },
                    "aggressive": {
                        "type": "boolean",
                        "description": "If true, attempts aggressive conversion on ambiguous columns. Recommended for messy datasets."
                    }
                },
                "required": ["file_path", "output_path"]
            }
        }
    },
    
    # Feature Engineering Tools (2)
    {
        "type": "function",
        "function": {
            "name": "create_time_features",
            "description": "Extract comprehensive time-based features from datetime columns including year, month, day, day_of_week, quarter, is_weekend, and cyclical encodings (sin/cos for month and hour).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "date_col": {
                        "type": "string",
                        "description": "Name of the datetime column to extract features from"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save dataset with new features"
                    }
                },
                "required": ["file_path", "date_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "encode_categorical",
            "description": "Encode categorical variables using one-hot encoding, target encoding, or frequency encoding. Handles high-cardinality columns intelligently. Use method='auto' to automatically choose the best encoding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["one_hot", "target", "frequency", "auto"],
                        "description": "Encoding method to use. 'auto' automatically selects the best method."
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of categorical columns to encode. Use ['all'] to encode all categorical columns. If not specified, defaults to all categorical columns."
                    },
                    "target_col": {
                        "type": "string",
                        "description": "Required for target encoding: name of the target column"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save dataset with encoded features"
                    }
                },
                "required": ["file_path", "output_path"]
            }
        }
    },
    
    # Model Training Tools (2)
    {
        "type": "function",
        "function": {
            "name": "train_baseline_models",
            "description": "Train multiple baseline models (Logistic Regression, Random Forest, XGBoost) and compare their performance. Automatically detects task type (classification/regression) and returns the best model with metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the prepared dataset file"
                    },
                    "target_col": {
                        "type": "string",
                        "description": "Name of the target column to predict"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                        "description": "Type of ML task. Use 'auto' to detect automatically."
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Proportion of data to use for testing (default: 0.2)"
                    },
                    "random_state": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (default: 42)"
                    }
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_model_report",
            "description": "Generate comprehensive model evaluation report including metrics, confusion matrix (for classification), feature importance, and SHAP values for top features. Saves report as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to saved model file (.pkl or .joblib)"
                    },
                    "test_data_path": {
                        "type": "string",
                        "description": "Path to test dataset file"
                    },
                    "target_col": {
                        "type": "string",
                        "description": "Name of the target column"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the report JSON file"
                    }
                },
                "required": ["model_path", "test_data_path", "target_col", "output_path"]
            }
        }
    },
    
    # New Data Wrangling Tools (3)
    {
        "type": "function",
        "function": {
            "name": "get_smart_summary",
            "description": "Generate an LLM-friendly smart summary of a dataset with per-column missing value percentages (sorted by severity), unique value counts, sample data, and numeric statistics. Much more detailed than profile_dataset for decision-making.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV or Parquet file to summarize"
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of sample rows to include in the summary (default: 5)"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "merge_datasets",
            "description": "Merge two datasets using SQL-like join operations (inner, left, right, outer, cross). Supports joining on single or multiple columns with same or different names. Automatically handles duplicate columns with suffixes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "left_path": {
                        "type": "string",
                        "description": "Path to the left (first) dataset file"
                    },
                    "right_path": {
                        "type": "string",
                        "description": "Path to the right (second) dataset file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the merged dataset"
                    },
                    "how": {
                        "type": "string",
                        "enum": ["inner", "left", "right", "outer", "cross"],
                        "description": "Join type: 'inner' (only matching rows), 'left' (all left + matching right), 'right' (all right + matching left), 'outer' (all rows from both), 'cross' (cartesian product)"
                    },
                    "on": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Column name(s) to join on (must exist in both datasets). Can be a single column name or list of columns. Use this when join columns have the same name in both datasets."
                    },
                    "left_on": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Column name(s) in left dataset to join on. Use with right_on when join columns have different names."
                    },
                    "right_on": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "description": "Column name(s) in right dataset to join on. Use with left_on when join columns have different names."
                    }
                },
                "required": ["left_path", "right_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "concat_datasets",
            "description": "Concatenate multiple datasets either vertically (stacking rows, useful for monthly data) or horizontally (adding columns side-by-side). Validates schema compatibility for vertical concat.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of paths to dataset files to concatenate (minimum 2 files)"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the concatenated dataset"
                    },
                    "axis": {
                        "type": "string",
                        "enum": ["vertical", "horizontal"],
                        "description": "'vertical' to stack rows (union, for monthly data), 'horizontal' to add columns side-by-side (default: 'vertical')"
                    }
                },
                "required": ["file_paths", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reshape_dataset",
            "description": "Transform dataset structure using pivot (long→wide format), melt (wide→long format), or transpose (swap rows and columns) operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file to reshape"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save the reshaped dataset"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["pivot", "melt", "transpose"],
                        "description": "Reshape operation: 'pivot' (long→wide, requires index/columns/values), 'melt' (wide→long, requires id_vars/value_vars), 'transpose' (swap rows/columns)"
                    },
                    "index": {
                        "type": "string",
                        "description": "Column to use as row index (for pivot operation)"
                    },
                    "columns": {
                        "type": "string",
                        "description": "Column whose values become new column names (for pivot operation)"
                    },
                    "values": {
                        "type": "string",
                        "description": "Column whose values populate the pivoted table (for pivot operation)"
                    },
                    "id_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to keep as identifiers (for melt operation)"
                    },
                    "value_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Columns to unpivot (for melt operation). If not specified, uses all columns except id_vars."
                    }
                },
                "required": ["file_path", "output_path", "operation"]
            }
        }
    },
    
    # ============================================
    # ADVANCED ANALYSIS TOOLS (5)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "perform_eda_analysis",
            "description": "Comprehensive Exploratory Data Analysis with visualizations, distribution analysis, and automated insights. Generates HTML report with plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Optional target column for supervised analysis"},
                    "output_dir": {"type": "string", "description": "Directory to save EDA report and plots"}
                },
                "required": ["file_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_model_issues",
            "description": "Detect overfitting, underfitting, class imbalance, and other model performance issues. Provides diagnostic recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to trained model"},
                    "train_data_path": {"type": "string", "description": "Path to training data"},
                    "test_data_path": {"type": "string", "description": "Path to test data"},
                    "target_col": {"type": "string", "description": "Target column name"}
                },
                "required": ["model_path", "train_data_path", "test_data_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalies using Isolation Forest, LOF, or statistical methods. Returns anomaly scores and flags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "method": {"type": "string", "enum": ["isolation_forest", "lof", "statistical"], "description": "Anomaly detection method"},
                    "contamination": {"type": "number", "description": "Expected proportion of anomalies (default: 0.1)"},
                    "output_path": {"type": "string", "description": "Path to save dataset with anomaly scores"}
                },
                "required": ["file_path", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_and_handle_multicollinearity",
            "description": "Detect and handle multicollinearity using VIF (Variance Inflation Factor). Removes highly correlated features automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "threshold": {"type": "number", "description": "VIF threshold (default: 10)"},
                    "method": {"type": "string", "enum": ["drop", "combine"], "description": "How to handle correlated features"},
                    "output_path": {"type": "string", "description": "Path to save cleaned dataset"}
                },
                "required": ["file_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_statistical_tests",
            "description": "Perform statistical hypothesis tests (t-test, chi-square, ANOVA) to analyze relationships between features and target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "test_type": {"type": "string", "enum": ["auto", "ttest", "chi2", "anova"], "description": "Type of statistical test"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    
    # ============================================
    # ADVANCED FEATURE ENGINEERING (4)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "create_interaction_features",
            "description": "Create polynomial, PCA, or cross-product interaction features to capture non-linear relationships.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "method": {"type": "string", "enum": ["polynomial", "pca", "cross"], "description": "Interaction method"},
                    "degree": {"type": "integer", "description": "Polynomial degree (default: 2)"},
                    "max_features": {"type": "integer", "description": "Maximum new features to create (default: 50)"},
                    "output_path": {"type": "string", "description": "Path to save enhanced dataset"}
                },
                "required": ["file_path", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_aggregation_features",
            "description": "Create aggregation features (mean, sum, count, etc.) grouped by categorical columns. Useful for customer/transaction data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "group_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by"},
                    "agg_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to aggregate"},
                    "agg_functions": {"type": "array", "items": {"type": "string"}, "description": "Aggregation functions (mean, sum, count, etc.)"},
                    "output_path": {"type": "string", "description": "Path to save dataset with aggregations"}
                },
                "required": ["file_path", "group_cols", "agg_cols", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "engineer_text_features",
            "description": "Extract features from text columns: TF-IDF, word counts, sentiment, readability scores, and embeddings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "text_col": {"type": "string", "description": "Text column name"},
                    "methods": {"type": "array", "items": {"type": "string"}, "description": "Feature extraction methods (tfidf, count, sentiment, readability)"},
                    "max_features": {"type": "integer", "description": "Max TF-IDF features (default: 100)"},
                    "output_path": {"type": "string", "description": "Path to save dataset with text features"}
                },
                "required": ["file_path", "text_col", "methods", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_feature_engineering",
            "description": "Use LLM (Gemini/Groq) to automatically generate creative feature engineering ideas and implement them. Works without API key if environment variables are set.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "groq_api_key": {"type": "string", "description": "Groq API key (optional - uses environment variable if not provided)"},
                    "max_suggestions": {"type": "integer", "description": "Maximum feature suggestions to generate (default: 10)"},
                    "implement_top_k": {"type": "integer", "description": "Number of top suggestions to implement (default: 5)"},
                    "output_path": {"type": "string", "description": "Path to save dataset with new features"}
                },
                "required": ["file_path", "target_col", "output_path"]
            }
        }
    },
    
    # ============================================
    # ADVANCED PREPROCESSING (3)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "handle_imbalanced_data",
            "description": "Handle class imbalance using SMOTE, ADASYN, or class weights. Critical for classification tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "method": {"type": "string", "enum": ["smote", "adasyn", "random_oversample", "random_undersample"], "description": "Balancing method"},
                    "sampling_strategy": {"type": "string", "description": "Sampling ratio (auto, minority, majority)"},
                    "output_path": {"type": "string", "description": "Path to save balanced dataset"}
                },
                "required": ["file_path", "target_col", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_feature_scaling",
            "description": "Scale features using StandardScaler, MinMaxScaler, or RobustScaler. Essential for distance-based algorithms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "method": {"type": "string", "enum": ["standard", "minmax", "robust"], "description": "Scaling method"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to scale (None = all numeric)"},
                    "output_path": {"type": "string", "description": "Path to save scaled dataset"}
                },
                "required": ["file_path", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "split_data_strategically",
            "description": "Split data with stratification, time-based splitting, or group-based splitting for better validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column for stratification"},
                    "method": {"type": "string", "enum": ["stratified", "time_based", "group_based"], "description": "Split method"},
                    "test_size": {"type": "number", "description": "Test set proportion (default: 0.2)"},
                    "time_col": {"type": "string", "description": "Time column for time-based split"},
                    "group_col": {"type": "string", "description": "Group column for group-based split"}
                },
                "required": ["file_path", "method"]
            }
        }
    },
    
    # ============================================
    # ADVANCED TRAINING (3)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "hyperparameter_tuning",
            "description": "Optimize model hyperparameters using Optuna (Bayesian optimization). Finds best model configuration automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to prepared dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "model_type": {"type": "string", "enum": ["random_forest", "xgboost", "lightgbm"], "description": "Model to tune"},
                    "n_trials": {"type": "integer", "description": "Number of tuning trials (default: 100)"},
                    "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "ML task type"},
                    "output_path": {"type": "string", "description": "Path to save tuned model"}
                },
                "required": ["file_path", "target_col", "model_type", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "train_ensemble_models",
            "description": "Train ensemble models using stacking, voting, or blending. Combines multiple models for better performance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to prepared dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "ensemble_method": {"type": "string", "enum": ["stacking", "voting", "blending"], "description": "Ensemble technique"},
                    "base_models": {"type": "array", "items": {"type": "string"}, "description": "Base model types to ensemble"},
                    "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "ML task type"},
                    "output_path": {"type": "string", "description": "Path to save ensemble model"}
                },
                "required": ["file_path", "target_col", "ensemble_method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_cross_validation",
            "description": "Perform k-fold cross-validation to get robust model performance estimates. Returns mean and std of metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "model_type": {"type": "string", "description": "Model type (random_forest, xgboost, logistic, ridge)"},
                    "n_splits": {"type": "integer", "description": "Number of CV folds/splits (default: 5)"},
                    "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "ML task type"},
                    "cv_strategy": {"type": "string", "enum": ["kfold", "stratified", "timeseries"], "description": "Cross-validation strategy (default: kfold)"},
                    "save_oof": {"type": "boolean", "description": "Whether to save out-of-fold predictions (default: false)"}
                },
                "required": ["file_path", "target_col", "model_type"]
            }
        }
    },
    
    # ============================================
    # BUSINESS INTELLIGENCE (4)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "perform_cohort_analysis",
            "description": "Analyze user cohorts over time (retention, revenue, engagement). Essential for SaaS and e-commerce businesses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to transaction/event data"},
                    "user_col": {"type": "string", "description": "User ID column"},
                    "date_col": {"type": "string", "description": "Date/timestamp column"},
                    "metric_col": {"type": "string", "description": "Metric to analyze (revenue, events, etc.)"},
                    "cohort_period": {"type": "string", "enum": ["daily", "weekly", "monthly"], "description": "Cohort grouping period"},
                    "output_path": {"type": "string", "description": "Path to save cohort analysis results"}
                },
                "required": ["file_path", "user_col", "date_col", "metric_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_rfm_analysis",
            "description": "RFM (Recency, Frequency, Monetary) analysis for customer segmentation. Identifies best/worst customers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to transaction data"},
                    "customer_col": {"type": "string", "description": "Customer ID column"},
                    "date_col": {"type": "string", "description": "Transaction date column"},
                    "amount_col": {"type": "string", "description": "Transaction amount column"},
                    "output_path": {"type": "string", "description": "Path to save RFM segments"}
                },
                "required": ["file_path", "customer_col", "date_col", "amount_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_causal_relationships",
            "description": "Detect potential causal relationships between features using Granger causality and correlation analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target/effect column"},
                    "feature_cols": {"type": "array", "items": {"type": "string"}, "description": "Potential cause columns"},
                    "method": {"type": "string", "enum": ["granger", "correlation"], "description": "Causality detection method"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_business_insights",
            "description": "Generate automated business insights using descriptive statistics, trends, and anomaly detection. Creates executive summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to business data"},
                    "metric_cols": {"type": "array", "items": {"type": "string"}, "description": "Key business metrics to analyze"},
                    "date_col": {"type": "string", "description": "Date column for trend analysis"},
                    "output_path": {"type": "string", "description": "Path to save insights report"}
                },
                "required": ["file_path", "metric_cols", "output_path"]
            }
        }
    },
    
    # ============================================
    # COMPUTER VISION (3)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "extract_image_features",
            "description": "Extract features from images using pre-trained CNNs (ResNet, VGG). Converts images to feature vectors for ML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_dir": {"type": "string", "description": "Directory containing images"},
                    "model": {"type": "string", "enum": ["resnet", "vgg", "mobilenet"], "description": "Pre-trained model to use"},
                    "output_path": {"type": "string", "description": "Path to save feature vectors CSV"}
                },
                "required": ["image_dir", "model", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_image_clustering",
            "description": "Cluster images based on visual similarity using K-means or DBSCAN on extracted features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_dir": {"type": "string", "description": "Directory containing images"},
                    "n_clusters": {"type": "integer", "description": "Number of clusters (default: auto-detect)"},
                    "method": {"type": "string", "enum": ["kmeans", "dbscan"], "description": "Clustering method"},
                    "output_path": {"type": "string", "description": "Path to save clustering results"}
                },
                "required": ["image_dir", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_tabular_image_hybrid",
            "description": "Combine tabular data with image features for hybrid ML models. Useful for e-commerce/medical data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tabular_path": {"type": "string", "description": "Path to tabular data CSV"},
                    "image_dir": {"type": "string", "description": "Directory with images"},
                    "image_id_col": {"type": "string", "description": "Column linking tabular data to images"},
                    "output_path": {"type": "string", "description": "Path to save combined features"}
                },
                "required": ["tabular_path", "image_dir", "image_id_col", "output_path"]
            }
        }
    },
    
    # ============================================
    # NLP/TEXT ANALYTICS (4)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "perform_topic_modeling",
            "description": "Discover topics in text documents using LDA or NMF. Extract themes from customer reviews, articles, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset with text"},
                    "text_col": {"type": "string", "description": "Text column name"},
                    "n_topics": {"type": "integer", "description": "Number of topics to extract (default: 5)"},
                    "method": {"type": "string", "enum": ["lda", "nmf"], "description": "Topic modeling method"},
                    "output_path": {"type": "string", "description": "Path to save topics and document-topic matrix"}
                },
                "required": ["file_path", "text_col", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_named_entity_recognition",
            "description": "Extract named entities (person, organization, location) from text using NER models.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset with text"},
                    "text_col": {"type": "string", "description": "Text column name"},
                    "output_path": {"type": "string", "description": "Path to save dataset with extracted entities"}
                },
                "required": ["file_path", "text_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment_advanced",
            "description": "Perform advanced sentiment analysis with aspect-based sentiment (what features customers like/dislike).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset with text"},
                    "text_col": {"type": "string", "description": "Text column name"},
                    "aspects": {"type": "array", "items": {"type": "string"}, "description": "Aspects to analyze sentiment for (e.g., 'price', 'quality')"},
                    "output_path": {"type": "string", "description": "Path to save sentiment scores"}
                },
                "required": ["file_path", "text_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_text_similarity",
            "description": "Calculate text similarity using cosine similarity, Jaccard, or semantic embeddings. Find duplicate/similar documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset with text"},
                    "text_col": {"type": "string", "description": "Text column name"},
                    "method": {"type": "string", "enum": ["cosine", "jaccard", "semantic"], "description": "Similarity method"},
                    "threshold": {"type": "number", "description": "Similarity threshold (0-1)"},
                    "output_path": {"type": "string", "description": "Path to save similarity matrix"}
                },
                "required": ["file_path", "text_col", "method", "output_path"]
            }
        }
    },
    
    # ============================================
    # PRODUCTION/MLOPS (5)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "monitor_model_drift",
            "description": "Detect data drift and concept drift in production models. Compare training vs production data distributions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "train_data_path": {"type": "string", "description": "Path to original training data"},
                    "production_data_path": {"type": "string", "description": "Path to recent production data"},
                    "features": {"type": "array", "items": {"type": "string"}, "description": "Features to monitor for drift"},
                    "output_path": {"type": "string", "description": "Path to save drift report"}
                },
                "required": ["train_data_path", "production_data_path", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_predictions",
            "description": "Explain model predictions using SHAP or LIME. Generate feature importance explanations for individual predictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to trained model"},
                    "data_path": {"type": "string", "description": "Path to data to explain"},
                    "method": {"type": "string", "enum": ["shap", "lime"], "description": "Explanation method"},
                    "n_samples": {"type": "integer", "description": "Number of samples to explain (default: 10)"},
                    "output_path": {"type": "string", "description": "Path to save explanations"}
                },
                "required": ["model_path", "data_path", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_model_card",
            "description": "Generate model card documentation with model details, performance metrics, bias analysis, and usage guidelines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to trained model"},
                    "train_data_path": {"type": "string", "description": "Path to training data"},
                    "test_data_path": {"type": "string", "description": "Path to test data"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "output_path": {"type": "string", "description": "Path to save model card JSON"}
                },
                "required": ["model_path", "train_data_path", "test_data_path", "target_col", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_ab_test_analysis",
            "description": "Analyze A/B test results with statistical significance testing. Determine if variant B is better than control A.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to A/B test data"},
                    "variant_col": {"type": "string", "description": "Column indicating variant (A/B)"},
                    "metric_col": {"type": "string", "description": "Success metric column"},
                    "confidence_level": {"type": "number", "description": "Confidence level for significance (default: 0.95)"}
                },
                "required": ["file_path", "variant_col", "metric_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_feature_leakage",
            "description": "Detect potential feature leakage by analyzing feature importance and temporal relationships. Prevents data leakage bugs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "date_col": {"type": "string", "description": "Optional date column for temporal analysis"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    
    # ============================================
    # TIME SERIES (3)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "forecast_time_series",
            "description": "Forecast future values using ARIMA, Prophet, or LSTM models. Handles seasonal and trend components.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to time series data"},
                    "date_col": {"type": "string", "description": "Date/timestamp column"},
                    "value_col": {"type": "string", "description": "Value column to forecast"},
                    "forecast_periods": {"type": "integer", "description": "Number of periods to forecast"},
                    "method": {"type": "string", "enum": ["arima", "prophet", "lstm"], "description": "Forecasting method"},
                    "output_path": {"type": "string", "description": "Path to save forecast results"}
                },
                "required": ["file_path", "date_col", "value_col", "forecast_periods", "method", "output_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_seasonality_trends",
            "description": "Detect seasonality patterns and trends in time series data using STL decomposition and statistical tests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to time series data"},
                    "date_col": {"type": "string", "description": "Date/timestamp column"},
                    "value_col": {"type": "string", "description": "Value column to analyze"},
                    "period": {"type": "integer", "description": "Expected seasonal period (e.g., 12 for monthly)"}
                },
                "required": ["file_path", "date_col", "value_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_time_series_features",
            "description": "Create comprehensive time series features: lags, rolling stats, exponential moving averages, and Fourier features.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to time series data"},
                    "date_col": {"type": "string", "description": "Date/timestamp column"},
                    "value_col": {"type": "string", "description": "Value column"},
                    "lags": {"type": "array", "items": {"type": "integer"}, "description": "Lag periods to create (e.g., [1, 7, 30])"},
                    "windows": {"type": "array", "items": {"type": "integer"}, "description": "Rolling window sizes (e.g., [7, 30])"},
                    "output_path": {"type": "string", "description": "Path to save dataset with time series features"}
                },
                "required": ["file_path", "date_col", "value_col", "output_path"]
            }
        }
    },
    
    # ============================================
    # ADVANCED INSIGHTS TOOLS (6) - NEW
    # ============================================
    
    {
        "type": "function",
        "function": {
            "name": "analyze_root_cause",
            "description": "Perform root cause analysis to identify why a metric dropped or changed. Analyzes correlations, temporal patterns, and identifies top influencing factors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Column to analyze (e.g., 'sales')"},
                    "time_col": {"type": "string", "description": "Optional time column for trend analysis"},
                    "threshold_drop": {"type": "number", "description": "Percentage drop to flag as significant (default 0.15)"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_trends_and_seasonality",
            "description": "Detect trends and seasonal patterns in time series data using statistical methods and autocorrelation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "value_col": {"type": "string", "description": "Column with values to analyze"},
                    "time_col": {"type": "string", "description": "Column with timestamps"},
                    "seasonal_period": {"type": "integer", "description": "Expected seasonal period (auto-detected if None)"}
                },
                "required": ["file_path", "value_col", "time_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies_advanced",
            "description": "Detect anomalies with confidence scores using Isolation Forest or statistical methods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to analyze (all numeric if None)"},
                    "contamination": {"type": "number", "description": "Expected proportion of outliers (default 0.1)"},
                    "method": {"type": "string", "enum": ["isolation_forest", "statistical"], "description": "Detection method"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_hypothesis_testing",
            "description": "Perform statistical hypothesis testing (t-test, ANOVA, chi-square) to compare groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "group_col": {"type": "string", "description": "Column defining groups"},
                    "value_col": {"type": "string", "description": "Column with values to compare"},
                    "test_type": {"type": "string", "enum": ["t-test", "anova", "chi-square", "auto"], "description": "Test type (auto-detected if 'auto')"}
                },
                "required": ["file_path", "group_col", "value_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_distribution",
            "description": "Analyze distribution of a column including normality tests, skewness, and kurtosis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "column": {"type": "string", "description": "Column to analyze"},
                    "tests": {"type": "array", "items": {"type": "string"}, "description": "Tests to perform (normality, skewness)"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "perform_segment_analysis",
            "description": "Perform cluster-based customer/data segmentation using K-means and profile each segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "n_segments": {"type": "integer", "description": "Number of segments to create (default 5)"},
                    "features": {"type": "array", "items": {"type": "string"}, "description": "Features for clustering (all numeric if None)"}
                },
                "required": ["file_path"]
            }
        }
    },
    
    # ============================================
    # AUTOMATED PIPELINE TOOLS (2) - NEW
    # ============================================
    
    {
        "type": "function",
        "function": {
            "name": "auto_ml_pipeline",
            "description": "Fully automated ML pipeline: auto-detect types, clean missing values, handle outliers, encode categorical, engineer features, and select best features. Zero configuration required!",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to input dataset"},
                    "target_col": {"type": "string", "description": "Target column name"},
                    "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "Task type (auto-detected if 'auto')"},
                    "output_path": {"type": "string", "description": "Where to save processed data"},
                    "feature_engineering_level": {"type": "string", "enum": ["basic", "intermediate", "advanced"], "description": "Feature engineering depth"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "auto_feature_selection",
            "description": "Automatically select the best features for modeling using mutual information or F-statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Target column"},
                    "task_type": {"type": "string", "enum": ["classification", "regression", "auto"], "description": "Task type"},
                    "max_features": {"type": "integer", "description": "Maximum features to keep (default 50)"},
                    "method": {"type": "string", "enum": ["mutual_info", "f_test", "auto"], "description": "Selection method"},
                    "output_path": {"type": "string", "description": "Where to save selected features"}
                },
                "required": ["file_path", "target_col"]
            }
        }
    },
    
    # ============================================
    # VISUALIZATION TOOLS (3) - NEW
    # ============================================
    
    {
        "type": "function",
        "function": {
            "name": "generate_all_plots",
            "description": "Generate ALL plots for a dataset automatically: data quality, EDA, distributions, and correlations. Creates interactive HTML plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Optional target column"},
                    "output_dir": {"type": "string", "description": "Directory to save plots (default ./outputs/plots)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_data_quality_plots",
            "description": "Generate data quality visualizations: missing values, data types, and outlier detection plots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "output_dir": {"type": "string", "description": "Directory to save plots"}
                },
                "required": ["file_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_eda_plots",
            "description": "Generate exploratory data analysis plots: correlation heatmap, feature relationships, and pairplots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Optional target column"},
                    "output_dir": {"type": "string", "description": "Directory to save plots"}
                },
                "required": ["file_path", "output_dir"]
            }
        }
    },
    
    # ============================================
    # INTERACTIVE PLOTLY VISUALIZATIONS (6)
    # ============================================
    {
        "type": "function",
        "function": {
            "name": "generate_interactive_scatter",
            "description": "Create interactive scatter plot with zoom, pan, and hover capabilities. Great for exploring relationships between variables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "x_col": {"type": "string", "description": "Column for X-axis"},
                    "y_col": {"type": "string", "description": "Column for Y-axis"},
                    "color_col": {"type": "string", "description": "Optional column for color coding points"},
                    "size_col": {"type": "string", "description": "Optional column for bubble size"},
                    "output_path": {"type": "string", "description": "Path to save HTML file (default: ./outputs/plots/interactive/scatter.html)"}
                },
                "required": ["file_path", "x_col", "y_col"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interactive_histogram",
            "description": "Create interactive histogram with box plot overlay. Users can explore distribution interactively.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "column": {"type": "string", "description": "Column to plot distribution"},
                    "bins": {"type": "integer", "description": "Number of bins (default: 30)"},
                    "color_col": {"type": "string", "description": "Optional column for grouped histograms"},
                    "output_path": {"type": "string", "description": "Path to save HTML file"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interactive_correlation_heatmap",
            "description": "Create interactive correlation heatmap with hover values. Better than static matplotlib version.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "output_path": {"type": "string", "description": "Path to save HTML file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interactive_box_plots",
            "description": "Create interactive box plots for outlier detection. Supports grouping by categorical variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to plot (all numeric if not specified)"},
                    "group_by": {"type": "string", "description": "Optional categorical column for grouping"},
                    "output_path": {"type": "string", "description": "Path to save HTML file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interactive_time_series",
            "description": "Create interactive time series plot with range slider and zoom. Perfect for temporal data analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "time_col": {"type": "string", "description": "Column with datetime values"},
                    "value_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns to plot over time"},
                    "output_path": {"type": "string", "description": "Path to save HTML file"}
                },
                "required": ["file_path", "time_col", "value_cols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_plotly_dashboard",
            "description": "Generate complete interactive dashboard with multiple visualizations: correlation heatmap, box plots, scatter plots, histograms. One-stop visualization solution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to dataset"},
                    "target_col": {"type": "string", "description": "Optional target column for supervised analysis"},
                    "output_dir": {"type": "string", "description": "Directory to save all plots (default: ./outputs/plots/interactive)"}
                },
                "required": ["file_path"]
            }
        }
    },
    # EDA Report Generation (3) - NEW PHASE 2
    {
        "type": "function",
        "function": {
            "name": "generate_sweetviz_report",
            "description": "Generate beautiful HTML EDA report using Sweetviz. Creates stunning visualizations with target analysis, feature distributions, correlations, missing values. Fast and visually appealing. Supports dataset comparison (train vs test).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the dataset CSV/Parquet file"},
                    "output_path": {"type": "string", "description": "Where to save HTML report (default: ./outputs/reports/sweetviz_report.html)"},
                    "target_column": {"type": "string", "description": "Optional target variable for association analysis"},
                    "compare_file_path": {"type": "string", "description": "Optional second dataset to compare (e.g., train vs test)"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_ydata_profiling_report",
            "description": "Generate comprehensive HTML report using ydata-profiling (formerly pandas-profiling). Provides extensive analysis: overview, variable statistics, interactions, correlations (Pearson, Spearman, Cramér's V), missing values matrix, duplicate analysis, and more. Most detailed profiling tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the dataset CSV/Parquet file"},
                    "output_path": {"type": "string", "description": "Where to save HTML report (default: ./outputs/reports/ydata_profile.html)"},
                    "minimal": {"type": "boolean", "description": "If true, generates faster minimal report (useful for large datasets, default: false)"},
                    "title": {"type": "string", "description": "Report title (default: 'Data Profiling Report')"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_combined_eda_report",
            "description": "Generate BOTH Sweetviz and ydata-profiling reports in one call. Best of both worlds: Sweetviz for beautiful fast visualizations + ydata-profiling for comprehensive detailed analysis. Recommended for complete EDA.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the dataset CSV/Parquet file"},
                    "output_dir": {"type": "string", "description": "Directory to save both reports (default: ./outputs/reports)"},
                    "target_column": {"type": "string", "description": "Optional target variable for Sweetviz analysis"},
                    "minimal": {"type": "boolean", "description": "If true, uses minimal mode for ydata-profiling (default: false)"}
                },
                "required": ["file_path"]
            }
        }
    },
    # ========================================
    # CODE INTERPRETER - THE GAME CHANGER 🚀
    # ========================================
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "⭐ CRITICAL TOOL - Execute custom Python code for ANY data science task not covered by existing tools. This is what makes you a TRUE AI AGENT, not just a function-calling bot. Use this when user requests: 1) Custom visualizations (specific Plotly plots, interactive dashboards, unique chart types) 2) Domain-specific calculations 3) Custom data transformations 4) Specific export formats 5) Interactive widgets/filters. Code has access to pandas, polars, numpy, matplotlib, seaborn, plotly. ALWAYS save outputs to files and return file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Auto-imported: pandas as pd, polars as pl, numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px, plotly.graph_objects as go. Code should save outputs to files in working_directory. Example: fig.write_html('./outputs/code/plot.html')"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Directory to run code in (default: ./outputs/code). Code can read from ./temp/ and write to this directory."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 60)"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code_from_file",
            "description": "Execute Python code from an existing .py file. Useful when code is too long to pass as string, or when running pre-written scripts. Same capabilities as execute_python_code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to .py file to execute"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Directory to run code in (default: ./outputs/code)"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 60)"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    
    # ============================================
    # CLOUD DATA SOURCES (4) - NEW
    # ============================================
    
    {
        "type": "function",
        "function": {
            "name": "load_bigquery_table",
            "description": "Load data from Google BigQuery table into a Polars DataFrame. Supports sampling via LIMIT and column selection. Returns CSV path for downstream tools. Use profile_bigquery_table first for large tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Google Cloud project ID"
                    },
                    "dataset": {
                        "type": "string",
                        "description": "BigQuery dataset name"
                    },
                    "table": {
                        "type": "string",
                        "description": "BigQuery table name"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional row limit for sampling (e.g., 10000 for large tables)"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names to load"
                    },
                    "where_clause": {
                        "type": "string",
                        "description": "Optional SQL WHERE clause for filtering (without WHERE keyword)"
                    }
                },
                "required": ["project_id", "dataset", "table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_bigquery_table",
            "description": "Write predictions or processed data from CSV/Parquet file to BigQuery table. Supports append, overwrite, or fail modes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to CSV or Parquet file to write"
                    },
                    "project_id": {
                        "type": "string",
                        "description": "Google Cloud project ID"
                    },
                    "dataset": {
                        "type": "string",
                        "description": "BigQuery dataset name"
                    },
                    "table": {
                        "type": "string",
                        "description": "BigQuery table name"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["append", "overwrite", "fail"],
                        "description": "Write mode: append (add rows), overwrite (replace), fail (error if exists)"
                    }
                },
                "required": ["file_path", "project_id", "dataset", "table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "profile_bigquery_table",
            "description": "Profile a BigQuery table without loading all data. Returns row count, column types, null counts (sampled), table size, and load recommendations. Use this BEFORE load_bigquery_table for large tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Google Cloud project ID"
                    },
                    "dataset": {
                        "type": "string",
                        "description": "BigQuery dataset name"
                    },
                    "table": {
                        "type": "string",
                        "description": "BigQuery table name"
                    }
                },
                "required": ["project_id", "dataset", "table"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_bigquery",
            "description": "Execute custom BigQuery SQL query and return results as DataFrame. Useful for complex aggregations, joins, or transformations before analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {
                        "type": "string",
                        "description": "Google Cloud project ID"
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to save results (default: auto-generated)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional row limit to append to query"
                    }
                },
                "required": ["project_id", "query"]
            }
        }
    }
]


def get_tool_by_name(tool_name: str) -> dict:
    """Get tool definition by name."""
    for tool in TOOLS:
        if tool["function"]["name"] == tool_name:
            return tool
    raise ValueError(f"Tool '{tool_name}' not found in registry")


def get_all_tool_names() -> list:
    """Get list of all tool names."""
    return [tool["function"]["name"] for tool in TOOLS]


def get_tools_by_category() -> dict:
    """Get tools organized by category."""
    return {
        "basic": [t["function"]["name"] for t in TOOLS[:16]],
        "advanced_analysis": [t["function"]["name"] for t in TOOLS[16:21]],
        "advanced_feature_engineering": [t["function"]["name"] for t in TOOLS[21:25]],
        "advanced_preprocessing": [t["function"]["name"] for t in TOOLS[25:28]],
        "advanced_training": [t["function"]["name"] for t in TOOLS[28:31]],
        "business_intelligence": [t["function"]["name"] for t in TOOLS[31:35]],
        "computer_vision": [t["function"]["name"] for t in TOOLS[35:38]],
        "nlp_text_analytics": [t["function"]["name"] for t in TOOLS[38:42]],
        "production_mlops": [t["function"]["name"] for t in TOOLS[42:47]],
        "time_series": [t["function"]["name"] for t in TOOLS[47:50]],
        "cloud_data_sources": [t["function"]["name"] for t in TOOLS[50:54]]
    }
