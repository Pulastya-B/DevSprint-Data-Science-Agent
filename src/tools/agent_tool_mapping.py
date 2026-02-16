"""
Agent-Specific Tool Mapping
Maps specialist agents to their relevant tools for dynamic loading.
"""

# Define tool categories and their tools
TOOL_CATEGORIES = {
    "profiling": [
        "profile_dataset",
        "detect_data_quality_issues",
        "analyze_correlations",
        "get_smart_summary",
    ],
    "cleaning": [
        "clean_missing_values",
        "handle_outliers",
        "fix_data_types",
        "force_numeric_conversion",
        "smart_type_inference",
        "remove_duplicates",
    ],
    "feature_engineering": [
        "create_time_features",
        "encode_categorical",
        "create_interaction_features",
        "create_ratio_features",
        "create_statistical_features",
        "create_log_features",
        "create_binned_features",
        "create_aggregation_features",
        "auto_feature_engineering",
    ],
    "visualization": [
        "generate_interactive_scatter",
        "generate_interactive_histogram",
        "generate_interactive_box_plots",
        "generate_interactive_correlation_heatmap",
        "generate_interactive_time_series",
        "generate_plotly_dashboard",
        "generate_eda_plots",
        "generate_combined_eda_report",
    ],
    "modeling": [
        "train_baseline_models",
        "train_with_autogluon",
        "predict_with_autogluon",
        "optimize_autogluon_model",
        "analyze_autogluon_model",
        "extend_autogluon_training",
        "train_multilabel_autogluon",
        "hyperparameter_tuning",
        "perform_cross_validation",
        "train_ensemble_models",
        "auto_ml_pipeline",
        "evaluate_model_performance",
    ],
    "time_series": [
        "detect_seasonality",
        "decompose_time_series",
        "forecast_arima",
        "forecast_prophet",
        "forecast_with_autogluon",
        "backtest_timeseries",
        "analyze_timeseries_model",
        "detect_anomalies_time_series",
    ],
    "nlp": [
        "extract_entities",
        "sentiment_analysis",
        "topic_modeling",
        "text_classification",
        "text_preprocessing",
    ],
    "computer_vision": [
        "image_classification",
        "object_detection",
        "image_preprocessing",
    ],
    "business_intelligence": [
        "calculate_kpis",
        "trend_analysis",
        "cohort_analysis",
        "churn_prediction",
    ],
    "production": [
        "export_model_to_onnx",
        "generate_inference_code",
        "create_model_documentation",
        "validate_model_drift",
    ],
    "code_execution": [
        "execute_python_code",
        "debug_code",
    ]
}

# Map specialist agents to their relevant tool categories
AGENT_TOOL_MAPPING = {
    "data_quality_agent": {
        "categories": ["profiling", "cleaning"],
        "description": "Focuses on data profiling, quality assessment, and cleaning operations"
    },
    "preprocessing_agent": {
        "categories": ["cleaning", "feature_engineering", "profiling"],
        "description": "Handles data cleaning, transformation, and feature engineering"
    },
    "visualization_agent": {
        "categories": ["visualization", "profiling"],
        "description": "Creates charts, plots, and interactive dashboards"
    },
    "modeling_agent": {
        "categories": ["modeling", "feature_engineering", "profiling"],
        "description": "Trains, tunes, and evaluates machine learning models"
    },
    "time_series_agent": {
        "categories": ["time_series", "profiling", "visualization"],
        "description": "Specializes in time series analysis and forecasting"
    },
    "nlp_agent": {
        "categories": ["nlp", "profiling", "visualization"],
        "description": "Natural language processing and text analytics"
    },
    "computer_vision_agent": {
        "categories": ["computer_vision", "profiling"],
        "description": "Image processing and computer vision tasks"
    },
    "business_intelligence_agent": {
        "categories": ["business_intelligence", "visualization", "profiling"],
        "description": "Business metrics, KPIs, and strategic insights"
    },
    "production_agent": {
        "categories": ["production", "modeling"],
        "description": "Model deployment, monitoring, and production operations"
    },
    "general_agent": {
        "categories": ["profiling", "cleaning", "visualization", "code_execution"],
        "description": "General purpose agent for exploratory analysis"
    }
}

# Core tools that should always be available regardless of agent
CORE_TOOLS = [
    "profile_dataset",
    "get_smart_summary",
    "execute_python_code",
]


def get_tools_for_agent(agent_name: str) -> list:
    """
    Get list of tool names relevant to a specific agent.
    
    Args:
        agent_name: Name of the specialist agent
        
    Returns:
        List of tool names the agent can use
    """
    if agent_name not in AGENT_TOOL_MAPPING:
        # Default to general agent tools
        agent_name = "general_agent"
    
    agent_info = AGENT_TOOL_MAPPING[agent_name]
    categories = agent_info["categories"]
    
    # Collect all tools from relevant categories
    tools = set(CORE_TOOLS)  # Start with core tools
    
    for category in categories:
        if category in TOOL_CATEGORIES:
            tools.update(TOOL_CATEGORIES[category])
    
    return list(tools)


def get_tool_categories_for_agent(agent_name: str) -> list:
    """
    Get categories of tools relevant to a specific agent.
    
    Args:
        agent_name: Name of the specialist agent
        
    Returns:
        List of tool category names
    """
    if agent_name not in AGENT_TOOL_MAPPING:
        agent_name = "general_agent"
    
    return AGENT_TOOL_MAPPING[agent_name]["categories"]


def filter_tools_by_names(all_tools: list, tool_names: list) -> list:
    """
    Filter tool definitions to only include specified tool names.
    
    Args:
        all_tools: List of all tool definitions (from TOOLS registry)
        tool_names: List of tool names to include
        
    Returns:
        Filtered list of tool definitions
    """
    filtered = []
    tool_names_set = set(tool_names)
    
    for tool in all_tools:
        if tool.get("type") == "function":
            function_name = tool.get("function", {}).get("name")
            if function_name in tool_names_set:
                # Compress description to reduce token usage
                compressed_tool = compress_tool_definition(tool)
                filtered.append(compressed_tool)
    
    return filtered


def compress_tool_definition(tool: dict) -> dict:
    """
    Compress tool definition to reduce token usage.
    
    Removes verbose examples and shortens descriptions while keeping
    essential information for the LLM to use the tool correctly.
    
    Args:
        tool: Tool definition dict
        
    Returns:
        Compressed tool definition
    """
    if tool.get("type") != "function":
        return tool
    
    compressed = {
        "type": "function",
        "function": {
            "name": tool["function"]["name"],
            "description": compress_description(tool["function"]["description"]),
            "parameters": tool["function"]["parameters"]
        }
    }
    
    # Compress parameter descriptions
    if "properties" in compressed["function"]["parameters"]:
        for param_name, param_info in compressed["function"]["parameters"]["properties"].items():
            if "description" in param_info:
                param_info["description"] = compress_description(param_info["description"])
    
    return compressed


def compress_description(description: str) -> str:
    """
    Compress a tool or parameter description.
    
    Removes examples, extra whitespace, and verbose explanations
    while keeping core functionality description.
    
    Args:
        description: Original description
        
    Returns:
        Compressed description
    """
    # Remove everything after "Example:" or "Examples:"
    if "Example:" in description:
        description = description.split("Example:")[0]
    if "Examples:" in description:
        description = description.split("Examples:")[0]
    
    # Remove extra whitespace and newlines
    description = " ".join(description.split())
    
    # Truncate if still too long (keep first 150 chars for params, 250 for tools)
    max_length = 250 if "Use this" in description else 150
    if len(description) > max_length:
        description = description[:max_length].rsplit(' ', 1)[0] + "..."
    
    return description.strip()


def get_agent_description(agent_name: str) -> str:
    """
    Get description of what an agent specializes in.
    
    Args:
        agent_name: Name of the specialist agent
        
    Returns:
        Agent description string
    """
    if agent_name in AGENT_TOOL_MAPPING:
        return AGENT_TOOL_MAPPING[agent_name]["description"]
    return "General purpose data science agent"


def suggest_next_agent(current_agent: str, completed_tools: list) -> str:
    """
    Suggest the next agent to hand off to based on completed tools.
    
    Args:
        current_agent: Current agent name
        completed_tools: List of tool names already executed
        
    Returns:
        Suggested next agent name, or None if workflow complete
    """
    # Define typical workflow progressions
    workflows = {
        "data_quality_agent": "preprocessing_agent",  # After profiling → cleaning
        "preprocessing_agent": "visualization_agent",   # After cleaning → visualize
        "visualization_agent": "modeling_agent",        # After EDA → modeling
        "modeling_agent": "production_agent",           # After training → deploy
    }
    
    # Check if current agent has completed its primary tasks
    agent_tools = set(get_tools_for_agent(current_agent))
    completed_set = set(completed_tools)
    
    # If less than 30% of agent's tools used, stay with current agent
    if len(completed_set & agent_tools) / max(len(agent_tools), 1) < 0.3:
        return current_agent
    
    # Suggest next agent in typical workflow
    return workflows.get(current_agent, None)
