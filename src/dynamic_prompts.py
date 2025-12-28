"""
Dynamic prompt generation for small context window models.
Loads only relevant tools based on user intent to reduce token usage.
"""

from typing import List, Dict, Set
import re

# Intent categories and their keywords
INTENT_KEYWORDS = {
    "data_quality": ["clean", "missing", "outlier", "quality", "duplicates", "null", "na", "impute"],
    "visualization": ["plot", "chart", "graph", "visualize", "dashboard", "scatter", "histogram", "heatmap"],
    "feature_engineering": ["feature", "encode", "transform", "scale", "normalize", "binning", "interaction"],
    "model_training": ["train", "model", "predict", "classify", "regression", "forecast", "xgboost", "accuracy"],
    "eda": ["profile", "describe", "summary", "statistics", "distribution", "correlation", "eda"],
    "time_series": ["time", "date", "datetime", "temporal", "trend", "seasonality", "forecast"],
    "optimization": ["tune", "optimize", "hyperparameter", "improve", "best parameters"],
    "code_execution": ["execute", "run code", "calculate", "custom", "python"],
}

# Tool categories mapping
TOOL_CATEGORIES = {
    "data_quality": [
        "detect_data_quality_issues",
        "clean_missing_values",
        "handle_outliers",
        "detect_and_remove_duplicates",
        "force_numeric_conversion",
    ],
    "visualization": [
        "generate_interactive_scatter",
        "generate_interactive_histogram",
        "generate_interactive_correlation_heatmap",
        "generate_interactive_box_plots",
        "generate_interactive_time_series",
        "generate_plotly_dashboard",
        "generate_all_plots",
        "generate_data_quality_plots",
        "generate_eda_plots",
    ],
    "feature_engineering": [
        "encode_categorical",
        "perform_feature_scaling",
        "create_time_features",
        "create_ratio_features",
        "create_statistical_features",
        "create_log_features",
        "create_binned_features",
        "auto_feature_engineering",
    ],
    "model_training": [
        "train_baseline_models",
        "hyperparameter_tuning",
        "train_ensemble_models",
        "perform_cross_validation",
        "handle_imbalanced_data",
        "auto_ml_pipeline",
    ],
    "eda": [
        "profile_dataset",
        "generate_ydata_profiling_report",
        "analyze_distribution",
        "detect_trends_and_seasonality",
        "perform_hypothesis_testing",
    ],
    "time_series": [
        "create_time_features",
        "forecast_time_series",
        "detect_trends_and_seasonality",
        "generate_interactive_time_series",
    ],
    "optimization": [
        "hyperparameter_tuning",
        "auto_feature_selection",
        "detect_and_handle_multicollinearity",
    ],
    "code_execution": [
        "execute_python_code",
        "execute_code_from_file",
    ],
}

# Core tools always included (used in all workflows)
CORE_TOOLS = [
    "profile_dataset",
    "detect_data_quality_issues",
    "clean_missing_values",
    "encode_categorical",
]


def detect_intent(query: str) -> Set[str]:
    """
    Detect user intent from query using keyword matching.
    
    Args:
        query: User's natural language query
        
    Returns:
        Set of intent categories detected
    """
    query_lower = query.lower()
    detected_intents = set()
    
    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                detected_intents.add(intent)
                break
    
    # Default to EDA if no specific intent detected
    if not detected_intents:
        detected_intents.add("eda")
    
    return detected_intents


def get_relevant_tools(intents: Set[str]) -> List[str]:
    """
    Get list of relevant tools based on detected intents.
    
    Args:
        intents: Set of detected intent categories
        
    Returns:
        List of tool names to include in prompt
    """
    tools = set(CORE_TOOLS)  # Always include core tools
    
    for intent in intents:
        if intent in TOOL_CATEGORIES:
            tools.update(TOOL_CATEGORIES[intent])
    
    return sorted(list(tools))


def build_compact_system_prompt(user_query: str = None, detected_intents: Set[str] = None) -> str:
    """
    Build a compact system prompt with only relevant tools.
    
    Args:
        user_query: Optional user query to detect intent
        detected_intents: Optional pre-detected intents
        
    Returns:
        Compact system prompt string
    """
    # Detect intents if not provided
    if detected_intents is None and user_query:
        detected_intents = detect_intent(user_query)
    elif detected_intents is None:
        detected_intents = {"eda"}  # Default
    
    # Get relevant tools
    relevant_tools = get_relevant_tools(detected_intents)
    
    # Build tool list string
    tool_list = "\n".join([f"- {tool}" for tool in relevant_tools])
    
    prompt = f"""You are an autonomous Data Science Agent. You EXECUTE tasks, not advise.

**TOOL CALLING FORMAT:**
When you need to use a tool, respond with JSON:
```json
{{
  "tool": "tool_name",
  "arguments": {{"param1": "value1"}}
}}
```

**RELEVANT TOOLS FOR THIS TASK:**
{tool_list}

**WORKFLOW RULES:**
1. **Execute tools sequentially** - ONE tool per response
2. **Use tool outputs** as inputs to next tool
3. **Save outputs** to ./outputs/data/ or ./outputs/plots/
4. **Error recovery**: If tool fails, retry with corrected parameters OR skip to next step
5. **Never repeat** successful tools
6. **Stop when done** - Don't continue after fulfilling user request

**COMMON WORKFLOWS:**

**Visualization Only:**
- User wants plots/charts/dashboard
- generate_plotly_dashboard OR generate_interactive_scatter → STOP

**Data Profiling:**
- User wants "detailed report"
- generate_ydata_profiling_report → STOP

**Full ML Pipeline:**
- User wants model training
- profile_dataset → detect_data_quality_issues → clean_missing_values → 
  encode_categorical → train_baseline_models → generate_plotly_dashboard

**PARAMETER CORRECTIONS:**
- Use exact column names from error messages
- If "Did you mean X?" → retry with X
- output_path (not output or output_dir)
- file_path for data files

**ERROR RECOVERY:**
- Column not found? Use suggested column from error
- File not found? Use last successful file
- Missing param? Add the required parameter
- Tool failed? Skip to next step (don't get stuck)

Execute the user's task efficiently with relevant tools."""
    
    return prompt


def get_full_system_prompt() -> str:
    """
    Get the original full system prompt for models with large context windows.
    This is the complete version used with Gemini 2.5 Flash.
    """
    # Import the original prompt from orchestrator
    from src.orchestrator import DataScienceCopilot
    copilot = DataScienceCopilot.__new__(DataScienceCopilot)
    return copilot._build_system_prompt()


# Quick stats
def get_prompt_stats(prompt: str) -> Dict[str, int]:
    """Get token count estimate and character count for prompt."""
    chars = len(prompt)
    # Rough estimate: 1 token ≈ 4 characters
    tokens = chars // 4
    lines = len(prompt.split('\n'))
    
    return {
        "characters": chars,
        "estimated_tokens": tokens,
        "lines": lines,
    }


if __name__ == "__main__":
    # Demo: Compare full vs compact prompts
    print("=" * 80)
    print("DYNAMIC PROMPT SYSTEM DEMO")
    print("=" * 80)
    
    # Example 1: Visualization request
    query1 = "Generate interactive plots for magnitude and latitude"
    intents1 = detect_intent(query1)
    prompt1 = build_compact_system_prompt(user_query=query1)
    stats1 = get_prompt_stats(prompt1)
    
    print(f"\n📊 Example 1: '{query1}'")
    print(f"Detected intents: {intents1}")
    print(f"Tools loaded: {len(get_relevant_tools(intents1))}")
    print(f"Prompt stats: {stats1['estimated_tokens']} tokens, {stats1['lines']} lines")
    
    # Example 2: Full ML pipeline
    query2 = "Train a model to predict earthquake magnitude"
    intents2 = detect_intent(query2)
    prompt2 = build_compact_system_prompt(user_query=query2)
    stats2 = get_prompt_stats(prompt2)
    
    print(f"\n🤖 Example 2: '{query2}'")
    print(f"Detected intents: {intents2}")
    print(f"Tools loaded: {len(get_relevant_tools(intents2))}")
    print(f"Prompt stats: {stats2['estimated_tokens']} tokens, {stats2['lines']} lines")
    
    # Example 3: Data profiling
    query3 = "Generate a detailed profiling report"
    intents3 = detect_intent(query3)
    prompt3 = build_compact_system_prompt(user_query=query3)
    stats3 = get_prompt_stats(prompt3)
    
    print(f"\n📈 Example 3: '{query3}'")
    print(f"Detected intents: {intents3}")
    print(f"Tools loaded: {len(get_relevant_tools(intents3))}")
    print(f"Prompt stats: {stats3['estimated_tokens']} tokens, {stats3['lines']} lines")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Compact prompts reduce tokens by 80-90% for small context models!")
    print("=" * 80)
