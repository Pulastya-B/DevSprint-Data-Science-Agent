"""
Tools Registry for Groq Function Calling
Defines all available tools in Groq's function calling format.
"""

TOOLS = [
    # Data Profiling Tools
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
    
    # Data Cleaning Tools
    {
        "type": "function",
        "function": {
            "name": "clean_missing_values",
            "description": "Handle missing values using appropriate strategies based on column type. Strategies include median/mean for numeric, mode for categorical, forward_fill for time series, or drop. Will not impute ID columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "strategy": {
                        "type": "object",
                        "description": "Dictionary mapping column names to strategies ('median', 'mean', 'mode', 'forward_fill', 'drop'). Use 'auto' to let the tool decide based on data type.",
                        "additionalProperties": {
                            "type": "string"
                        }
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save cleaned dataset"
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
                        "additionalProperties": {
                            "type": "string"
                        }
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
    
    # Feature Engineering Tools
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
            "description": "Encode categorical variables using one-hot encoding, target encoding, or frequency encoding. Handles high-cardinality columns intelligently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the dataset file"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["one_hot", "target", "frequency"],
                        "description": "Encoding method to use"
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of categorical columns to encode. Use 'all' to encode all categorical columns."
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
                "required": ["file_path", "method", "columns", "output_path"]
            }
        }
    },
    
    # Model Training Tools
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
    }
]


def get_tool_by_name(tool_name: str) -> dict:
    """
    Get tool definition by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool definition dictionary
        
    Raises:
        ValueError: If tool not found
    """
    for tool in TOOLS:
        if tool["function"]["name"] == tool_name:
            return tool
    
    raise ValueError(f"Tool '{tool_name}' not found in registry")


def get_all_tool_names() -> list:
    """
    Get list of all tool names.
    
    Returns:
        List of tool names
    """
    return [tool["function"]["name"] for tool in TOOLS]
