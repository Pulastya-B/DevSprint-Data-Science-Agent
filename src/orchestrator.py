"""
Data Science Copilot Orchestrator
Main orchestration class that uses LLM function calling to execute data science workflows.
Supports multiple providers: Groq and Gemini.
"""

import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import httpx

from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

from .cache.cache_manager import CacheManager
from .tools.tools_registry import TOOLS, get_all_tool_names, get_tools_by_category
from .session_memory import SessionMemory
from .session_store import SessionStore
from .workflow_state import WorkflowState
from .utils.schema_extraction import extract_schema_local, infer_task_type
from .tools import (
    # Basic Tools (13) - UPDATED: Added get_smart_summary + 3 wrangling tools
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations,
    get_smart_summary,  # NEW
    clean_missing_values,
    handle_outliers,
    fix_data_types,
    force_numeric_conversion,
    smart_type_inference,
    create_time_features,
    encode_categorical,
    train_baseline_models,
    generate_model_report,
    # Data Wrangling Tools (3) - NEW
    merge_datasets,
    concat_datasets,
    reshape_dataset,
    # Advanced Analysis (5)
    perform_eda_analysis,
    detect_model_issues,
    detect_anomalies,
    detect_and_handle_multicollinearity,
    perform_statistical_tests,
    # Advanced Feature Engineering (4)
    create_interaction_features,
    create_aggregation_features,
    engineer_text_features,
    auto_feature_engineering,
    # Advanced Preprocessing (3)
    handle_imbalanced_data,
    perform_feature_scaling,
    split_data_strategically,
    # Advanced Training (3)
    hyperparameter_tuning,
    train_ensemble_models,
    perform_cross_validation,
    # Business Intelligence (4)
    perform_cohort_analysis,
    perform_rfm_analysis,
    detect_causal_relationships,
    generate_business_insights,
    # Computer Vision (3)
    extract_image_features,
    perform_image_clustering,
    analyze_tabular_image_hybrid,
    # NLP/Text Analytics (4)
    perform_topic_modeling,
    perform_named_entity_recognition,
    analyze_sentiment_advanced,
    perform_text_similarity,
    # Production/MLOps (5)
    monitor_model_drift,
    explain_predictions,
    generate_model_card,
    perform_ab_test_analysis,
    detect_feature_leakage,
    # Time Series (3)
    forecast_time_series,
    detect_seasonality_trends,
    create_time_series_features,
    # Advanced Insights (6)
    analyze_root_cause,
    detect_trends_and_seasonality,
    detect_anomalies_advanced,
    perform_hypothesis_testing,
    analyze_distribution,
    perform_segment_analysis,
    # Automated Pipeline (2)
    auto_ml_pipeline,
    auto_feature_selection,
    # Visualization (5)
    generate_all_plots,
    generate_data_quality_plots,
    generate_eda_plots,
    generate_model_performance_plots,
    generate_feature_importance_plot,
    # Interactive Plotly Visualizations (6) - NEW PHASE 2
    generate_interactive_scatter,
    generate_interactive_histogram,
    generate_interactive_correlation_heatmap,
    generate_interactive_box_plots,
    generate_interactive_time_series,
    generate_plotly_dashboard,
    # EDA Report Generation (1) - NEW PHASE 2
    generate_ydata_profiling_report,
    # Code Interpreter (2) - NEW PHASE 2 - TRUE AI AGENT CAPABILITY
    execute_python_code,
    execute_code_from_file,
    # Cloud Data Sources (4) - NEW: BigQuery Integration
    load_bigquery_table,
    write_bigquery_table,
    profile_bigquery_table,
    query_bigquery,
    # Enhanced Feature Engineering (4)
    create_ratio_features,
    create_statistical_features,
    create_log_features,
    create_binned_features,
)


class DataScienceCopilot:
    """
    Main orchestrator for data science workflows using LLM function calling.
    
    Supports multiple providers: Groq and Gemini.
    Uses function calling to intelligently route to data profiling, cleaning,
    feature engineering, and model training tools.
    """
    
    def __init__(self, groq_api_key: Optional[str] = None, 
                 google_api_key: Optional[str] = None,
                 mistral_api_key: Optional[str] = None,
                 cache_db_path: Optional[str] = None,
                 reasoning_effort: str = "medium",
                 provider: Optional[str] = None,
                 session_id: Optional[str] = None,
                 use_session_memory: bool = True,
                 use_compact_prompts: bool = False):
        """
        Initialize the Data Science Copilot.
        
        Args:
            groq_api_key: Groq API key (or set GROQ_API_KEY env var)
            google_api_key: Google API key (or set GOOGLE_API_KEY env var)
            mistral_api_key: Mistral API key (or set MISTRAL_API_KEY env var)
            cache_db_path: Path to cache database
            reasoning_effort: Reasoning effort for Groq ('low', 'medium', 'high')
            provider: LLM provider - 'groq' or 'gemini' (or set LLM_PROVIDER env var)
            session_id: Session ID to resume (None = auto-resume recent or create new)
            use_session_memory: Enable session-based memory for context across requests
            use_compact_prompts: Use compact prompts for small context window models (e.g., Groq)
        """
        # Load environment variables
        load_dotenv()
        
        # Determine provider
        self.provider = provider or os.getenv("LLM_PROVIDER", "mistral").lower()
        
        # Set compact prompts: Auto-enable for Groq/Mistral, manual for others
        self.use_compact_prompts = use_compact_prompts or (self.provider in ["groq", "mistral"])
        
        if self.provider == "mistral":
            # Initialize Mistral client (OpenAI-compatible)
            api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Mistral API key must be provided or set in MISTRAL_API_KEY env var")
            
            from mistralai.client import MistralClient
            self.mistral_client = MistralClient(api_key=api_key.strip())
            self.model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
            self.reasoning_effort = reasoning_effort
            self.gemini_model = None
            self.groq_client = None
            print(f"🤖 Initialized with Mistral provider - Model: {self.model}")
            
        elif self.provider == "groq":
            # Initialize Groq client
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key must be provided or set in GROQ_API_KEY env var")
            
            self.groq_client = Groq(api_key=api_key.strip())
            self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            self.reasoning_effort = reasoning_effort
            self.gemini_model = None
            self.mistral_client = None
            print(f"🤖 Initialized with Groq provider - Model: {self.model}")
            
        elif self.provider == "gemini":
            # Initialize Gemini client
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY or GEMINI_API_KEY env var")
            
            genai.configure(api_key=api_key.strip())
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            
            # Configure safety settings to be more permissive for data science content
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            self.gemini_model = genai.GenerativeModel(
                self.model,
                generation_config={"temperature": 0.1},
                safety_settings=safety_settings
            )
            self.groq_client = None
            self.mistral_client = None
            print(f"🤖 Initialized with Gemini provider - Model: {self.model}")
            
        else:
            raise ValueError(f"Invalid provider: {self.provider}. Must be 'mistral', 'groq', or 'gemini'")
            raise ValueError(f"Unsupported provider: {self.provider}. Choose 'groq' or 'gemini'")
        
        # Initialize cache
        cache_path = cache_db_path or os.getenv("CACHE_DB_PATH", "./cache_db/cache.db")
        self.cache = CacheManager(db_path=cache_path)
        
        # 🧠 Initialize session memory
        self.use_session_memory = use_session_memory
        if use_session_memory:
            self.session_store = SessionStore()
            
            # Try to load existing session or create new one
            if session_id:
                # Explicit session ID provided - load it
                self.session = self.session_store.load(session_id)
                if not self.session:
                    print(f"⚠️  Session {session_id} not found, creating new session")
                    self.session = SessionMemory(session_id=session_id)
                else:
                    print(f"✅ Loaded session: {session_id}")
            else:
                # Try to continue recent session (within 24 hours)
                self.session = self.session_store.get_recent_session(max_age_hours=24)
                if self.session:
                    print(f"✅ Resuming recent session: {self.session.session_id}")
                else:
                    # No recent session - create new one
                    self.session = SessionMemory()
                    print(f"✅ Created new session: {self.session.session_id}")
            
            # Show context if available
            if self.session.last_dataset or self.session.last_model:
                print(f"📝 Session Context:")
                if self.session.last_dataset:
                    print(f"   - Last dataset: {self.session.last_dataset}")
                if self.session.last_model:
                    print(f"   - Last model: {self.session.last_model} (score: {self.session.best_score:.4f})" if self.session.best_score else f"   - Last model: {self.session.last_model}")
        else:
            self.session = None
            print("⚠️  Session memory disabled")
        
        # Tools registry
        self.tools_registry = TOOLS
        self.tool_functions = self._build_tool_functions_map()
        
        # Token tracking and rate limiting
        self.total_tokens_used = 0
        self.tokens_this_minute = 0
        self.minute_start_time = time.time()
        self.api_calls_made = 0
        
        # Provider-specific limits
        if self.provider == "mistral":
            self.tpm_limit = 500000  # 500K tokens/minute (very generous)
            self.rpm_limit = 500     # 500 requests/minute
            self.min_api_call_interval = 0.1  # Minimal delay
        elif self.provider == "groq":
            self.tpm_limit = 12000  # Tokens per minute
            self.rpm_limit = 30     # Requests per minute
            self.min_api_call_interval = 0.5  # Wait between calls
        elif self.provider == "gemini":
            self.tpm_limit = 32000  # More generous
            self.rpm_limit = 15
            self.min_api_call_interval = 1.0  # Gemini free tier: safer spacing
        
        # Rate limiting for Gemini (10 RPM free tier)
        self.last_api_call_time = 0
        
        # Workflow state for context management (reduces token usage)
        self.workflow_state = WorkflowState()
        
        # Ensure output directories exist
        Path("./outputs").mkdir(exist_ok=True)
        Path("./outputs/models").mkdir(exist_ok=True)
        Path("./outputs/reports").mkdir(exist_ok=True)
        Path("./outputs/data").mkdir(exist_ok=True)
    
    def _build_tool_functions_map(self) -> Dict[str, callable]:
        """Build mapping of tool names to their functions - All 75 tools."""
        return {
            # Basic Tools (13) - UPDATED: Added 4 new tools
            "profile_dataset": profile_dataset,
            "detect_data_quality_issues": detect_data_quality_issues,
            "analyze_correlations": analyze_correlations,
            "get_smart_summary": get_smart_summary,  # NEW
            "clean_missing_values": clean_missing_values,
            "handle_outliers": handle_outliers,
            "fix_data_types": fix_data_types,
            "force_numeric_conversion": force_numeric_conversion,
            "smart_type_inference": smart_type_inference,
            "create_time_features": create_time_features,
            "encode_categorical": encode_categorical,
            "train_baseline_models": train_baseline_models,
            "generate_model_report": generate_model_report,
            # Data Wrangling Tools (3) - NEW
            "merge_datasets": merge_datasets,
            "concat_datasets": concat_datasets,
            "reshape_dataset": reshape_dataset,
            # Advanced Analysis (5)
            "perform_eda_analysis": perform_eda_analysis,
            "detect_model_issues": detect_model_issues,
            "detect_anomalies": detect_anomalies,
            "detect_and_handle_multicollinearity": detect_and_handle_multicollinearity,
            "perform_statistical_tests": perform_statistical_tests,
            # Advanced Feature Engineering (4)
            "create_interaction_features": create_interaction_features,
            "create_aggregation_features": create_aggregation_features,
            "engineer_text_features": engineer_text_features,
            "auto_feature_engineering": auto_feature_engineering,
            # Advanced Preprocessing (3)
            "handle_imbalanced_data": handle_imbalanced_data,
            "perform_feature_scaling": perform_feature_scaling,
            "split_data_strategically": split_data_strategically,
            # Advanced Training (3)
            "hyperparameter_tuning": hyperparameter_tuning,
            "train_ensemble_models": train_ensemble_models,
            "perform_cross_validation": perform_cross_validation,
            # Business Intelligence (4)
            "perform_cohort_analysis": perform_cohort_analysis,
            "perform_rfm_analysis": perform_rfm_analysis,
            "detect_causal_relationships": detect_causal_relationships,
            "generate_business_insights": generate_business_insights,
            # Computer Vision (3)
            "extract_image_features": extract_image_features,
            "perform_image_clustering": perform_image_clustering,
            "analyze_tabular_image_hybrid": analyze_tabular_image_hybrid,
            # NLP/Text Analytics (4)
            "perform_topic_modeling": perform_topic_modeling,
            "perform_named_entity_recognition": perform_named_entity_recognition,
            "analyze_sentiment_advanced": analyze_sentiment_advanced,
            "perform_text_similarity": perform_text_similarity,
            # Production/MLOps (5)
            "monitor_model_drift": monitor_model_drift,
            "explain_predictions": explain_predictions,
            "generate_model_card": generate_model_card,
            "perform_ab_test_analysis": perform_ab_test_analysis,
            "detect_feature_leakage": detect_feature_leakage,
            # Time Series (3)
            "forecast_time_series": forecast_time_series,
            "detect_seasonality_trends": detect_seasonality_trends,
            "create_time_series_features": create_time_series_features,
            # Advanced Insights (6)
            "analyze_root_cause": analyze_root_cause,
            "detect_trends_and_seasonality": detect_trends_and_seasonality,
            "detect_anomalies_advanced": detect_anomalies_advanced,
            "perform_hypothesis_testing": perform_hypothesis_testing,
            "analyze_distribution": analyze_distribution,
            "perform_segment_analysis": perform_segment_analysis,
            # Automated Pipeline (2)
            "auto_ml_pipeline": auto_ml_pipeline,
            "auto_feature_selection": auto_feature_selection,
            # Visualization (5)
            "generate_all_plots": generate_all_plots,
            "generate_data_quality_plots": generate_data_quality_plots,
            "generate_eda_plots": generate_eda_plots,
            "generate_model_performance_plots": generate_model_performance_plots,
            "generate_feature_importance_plot": generate_feature_importance_plot,
            # Interactive Plotly Visualizations (6) - NEW PHASE 2
            "generate_interactive_scatter": generate_interactive_scatter,
            "generate_interactive_histogram": generate_interactive_histogram,
            "generate_interactive_correlation_heatmap": generate_interactive_correlation_heatmap,
            "generate_interactive_box_plots": generate_interactive_box_plots,
            "generate_interactive_time_series": generate_interactive_time_series,
            "generate_plotly_dashboard": generate_plotly_dashboard,
            # EDA Report Generation (1) - NEW PHASE 2
            "generate_ydata_profiling_report": generate_ydata_profiling_report,
            # Code Interpreter (2) - NEW PHASE 2 - TRUE AI AGENT CAPABILITY
            "execute_python_code": execute_python_code,
            "execute_code_from_file": execute_code_from_file,
            # Cloud Data Sources (4) - NEW: BigQuery Integration
            "load_bigquery_table": load_bigquery_table,
            "write_bigquery_table": write_bigquery_table,
            "profile_bigquery_table": profile_bigquery_table,
            "query_bigquery": query_bigquery,
            # Enhanced Feature Engineering (4)
            "create_ratio_features": create_ratio_features,
            "create_statistical_features": create_statistical_features,
            "create_log_features": create_log_features,
            "create_binned_features": create_binned_features,
        }
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for the copilot."""
        return """You are an autonomous Data Science Agent. You EXECUTE tasks, not advise.

**CRITICAL: Tool Calling Format**
When you need to use a tool, respond with a JSON block like this:
```json
{
  "tool": "tool_name",
  "arguments": {
    "param1": "value1",
    "param2": 123
  }
}
```

**ONE TOOL PER RESPONSE**. After tool execution, I will send you the result and you can call the next tool.

**CRITICAL: Detect the user's intent and use the appropriate workflow.**

**🎯 INTENT DETECTION (ALWAYS DO THIS FIRST):**

**A. CODE-ONLY TASKS** - User wants to execute custom Python code:
- Keywords: "execute", "run code", "calculate", "generate data", "create plot", "custom visualization"
- No dataset file provided (file_path="dummy" or similar)
- Specific programming task (Fibonacci, custom charts, synthetic data, etc.)
- **ACTION**: Use execute_python_code tool ONCE and IMMEDIATELY return success. DO NOT run ML workflow!
- **CRITICAL**: After execute_python_code succeeds → STOP IMMEDIATELY, return summary, DO NOT call any other tools!
- **Example**: "Calculate Fibonacci" → execute_python_code → RETURN SUCCESS ✓ (NO other tools!)

**B. VISUALIZATION-ONLY REQUESTS** - User wants charts/graphs without ML:
- Keywords: "generate plots", "create dashboard", "visualize", "show graphs", "interactive charts"
- **NO keywords for ML**: No "train", "predict", "model", "classify", "forecast"
- Real dataset provided BUT only wants visualization
- **ACTION**: Generate visualizations directly, skip data cleaning/ML steps
- **Workflow**: 
  1. generate_interactive_scatter() OR generate_plotly_dashboard() 
  2. STOP - DO NOT clean data, encode, or train models!
- **Example**: "Generate interactive plots for Magnitude and latitude" → generate_interactive_scatter → DONE ✓

**C. DATA PROFILING REPORT** - User wants comprehensive data analysis report:
- Keywords: "detailed report", "comprehensive report", "data report", "profiling report", "full analysis"
- **NO specific visualization mentioned** (no "plot", "chart", "graph")
- Real dataset provided
- **ACTION**: Use generate_ydata_profiling_report tool
- **Workflow**:
  1. generate_ydata_profiling_report(file_path) 
  2. STOP - This generates a complete HTML report with all stats, correlations, distributions
- **Example**: "Generate a detailed report for this" → generate_ydata_profiling_report → DONE ✓

**D. DATA ANALYSIS WITH ML** - Full workflow with model training:
- Real dataset file path provided (CSV, Excel, etc. - NOT "dummy")
- Keywords: "train model", "predict", "classify", "build model", "forecast"
- User wants: cleaning + feature engineering + model training
- **ACTION**: Run full ML workflow (steps 1-15 below)
- **Example**: "Train a model to predict earthquake magnitude" → Full pipeline

**E. UNCLEAR/AMBIGUOUS REQUESTS** - Intent is not obvious:
- User says: "analyze", "look at", "check", "review" (without specifics)
- Could mean: visualization only OR full ML OR just exploration
- **ACTION**: ASK USER to clarify BEFORE starting work
- **Questions to ask**:
  - "Would you like me to: (1) Just create visualizations, (2) Train a predictive model, or (3) Both?"
  - "Do you need model training or just want to explore the data visually?"
- **DO NOT ASSUME** - Always ask when unclear!

**F. SIMPLE QUESTIONS** - User asks for explanation/advice:
- Keywords: "what is", "how to", "explain", "recommend"
- **ACTION**: Answer directly, no tools needed

---

**WORKFLOW FOR VISUALIZATION-ONLY (Type B above):**
- User wants: "generate plots", "create dashboard", "visualize X and Y"
- **DO NOT run full pipeline** - Skip cleaning, encoding, training!
- **Quick workflow**:
  1. If specific columns mentioned → generate_interactive_scatter(x_col, y_col)
  2. If "dashboard" mentioned → generate_plotly_dashboard(file_path, target_col)
  3. STOP - Return success
- **Example**: "Generate interactive plots for Magnitude and latitude"
  → generate_interactive_scatter(x_col="mag", y_col="latitude") → DONE ✓

**📊 COLUMN SELECTION FOR VAGUE REQUESTS:**
When user doesn't specify columns (e.g., "plot a scatter" without mentioning X/Y):

1. **Analyze the dataset structure and domain**:
   - Inspect column names, types, and value ranges
   - Identify patterns: spatial coordinates (lat/lon, x/y), temporal data (dates, timestamps), 
     categorical hierarchies, numerical measurements, identifiers
   - Infer domain from filename/columns (geographic, financial, health, retail, etc.)
   
2. **Apply intelligent selection strategies**:
   
   **For Scatter Plots** - Choose variables with meaningful relationships:
   - Geographic data: Pair coordinate columns (latitude+longitude, x+y coordinates)
   - Price/size relationships: Pair cost with quantity/area/volume metrics
   - Performance metrics: Pair effort/input with outcome/output variables
   - Temporal relationships: Pair time with trend variables
   - Categorical vs numeric: Use most important numeric split by key category
   
   **For Histograms** - Select the primary measure of interest:
   - Target variable (if identified): The variable being predicted/analyzed
   - Main metric: Revenue, score, magnitude, count, amount (key business/scientific measure)
   - Distribution of interest: Variable with expected patterns (age, income, frequency)
   - First numeric column with meaningful range (avoid IDs, binary flags)
   
   **For Box Plots** - Show distribution comparisons:
   - Numeric variable grouped by categorical (e.g., price by category, score by region)
   - Multiple related numeric variables side-by-side
   
   **For Time Series** - Identify temporal patterns:
   - Date/datetime column + primary metric to track over time
   - Multiple metrics over time if related (sales, costs, profit)
   
   **For Heatmaps** - No column choice needed (shows all numeric correlations)
   
3. **Selection principles** (no dataset-specific bias):
   - Avoid ID columns, constants, or binary flags for visualizations
   - Prefer columns with high variance and meaningful ranges
   - Choose natural pairs (coordinates, input-output, cause-effect)
   - Select variables that answer implicit questions about the data
   - When uncertain, pick columns that reveal the most information
   
4. **ALWAYS EXPLAIN YOUR REASONING** in the final summary:
   - State WHAT columns you chose
   - Explain WHY those columns (their relationship/significance)
   - Describe WHAT INSIGHTS the visualization reveals
   
   ✅ Good explanation:
   "I created a scatter plot of [Column A] vs [Column B] because they represent [relationship type].
   This visualization reveals [pattern/insight]. For the histogram, I chose [Column C] as it's 
   the [primary metric/target variable], showing [distribution pattern]."
   
   ❌ Bad explanation:
   "Scatter plot created" (no reasoning about column selection)

**TRANSPARENCY RULE**: Justify every column choice with domain-agnostic reasoning based on data 
structure, variable relationships, and expected insights - not hardcoded domain assumptions.

**WORKFLOW FOR FULL ML ANALYSIS (Type C above):**
- User wants: model training, prediction, classification
- Execute steps IN ORDER (1 → 2 → 3 → ... → 15)
- Each step runs ONCE (unless explicitly noted like "call for each datetime column")
- After step completes successfully (✓ Completed) → IMMEDIATELY move to NEXT step
- DO NOT repeat steps, DO NOT go backwards, DO NOT skip steps (unless optional)
- Track your progress: "Completed steps 1-8, now executing step 9..."

**FULL ML WORKFLOW (Execute ALL steps - DO NOT SKIP):**
1. profile_dataset(file_path) - ONCE ONLY
2. detect_data_quality_issues(file_path) - ONCE ONLY
3. generate_data_quality_plots(file_path, output_dir="./outputs/plots/quality") - Generate quality visualizations
4. clean_missing_values(file_path, strategy="auto", output="./outputs/data/cleaned.csv")
5. handle_outliers(cleaned, method="clip", columns=["all"], output="./outputs/data/no_outliers.csv")
6. force_numeric_conversion(latest, columns=["all"], output="./outputs/data/numeric.csv", errors="coerce")
7. **IF DATETIME COLUMNS EXIST**: create_time_features(latest, date_col="<column_name>", output="./outputs/data/time_features.csv") - Extract year/month/day/hour/weekday/timestamp from each datetime column
8. encode_categorical(latest, method="auto", output="./outputs/data/encoded.csv")
9. generate_eda_plots(encoded, target_col, output_dir="./outputs/plots/eda") - Generate EDA visualizations
10. **ONLY IF USER EXPLICITLY REQUESTED ML**: train_baseline_models(encoded, target_col, task_type="auto")
11. **HYPERPARAMETER TUNING (OPTIONAL - Smart Decision)**:
    - IF user says "optimize", "tune", "improve", "best model possible" → ALWAYS tune
    - IF best model score < 0.90 → Tune to improve (user expects good accuracy)
    - IF best model score > 0.95 → Skip tuning (already excellent)
    - **How**: hyperparameter_tuning(file_path=encoded, target_col=target_col, model_type="xgboost", n_trials=50)
    - **Only tune the WINNING model** (don't waste time on others)
    - **Map model names**: XGBoost→"xgboost", RandomForest→"random_forest", Ridge→"ridge", Lasso→use Ridge
    - **Note**: Time features should already be extracted in step 7 (create_time_features)
12. **CROSS-VALIDATION (OPTIONAL - Production Models)**:
    - IF user says "validate", "production", "robust", "deploy" → ALWAYS cross-validate
    - IF best model score > 0.85 → Cross-validate to confirm robustness
    - ELSE → Skip (focus on improving score first with tuning)
    - **How**: perform_cross_validation(file_path=encoded, target_col=target_col, model_type="xgboost", cv_strategy="kfold", n_splits=5)
    - **Use same model type as winner** (e.g., if XGBoost won, use model_type="xgboost")
    - **Provides**: Mean CV score ± std dev (shows if model is reliable)
    - **Note**: Time features should already be extracted in step 7 (create_time_features)
13. **AFTER TRAINING/TUNING**: generate_combined_eda_report(encoded, target_col, output_dir="./outputs/reports") - Generate comprehensive HTML reports
14. **INTERACTIVE DASHBOARD (OPTIONAL - Smart Detection)**:
    - **ALWAYS generate IF user mentions**: "dashboard", "interactive", "plotly", "visualize", "charts", "graphs", "plots"
    - **ALWAYS generate IF user wants exploration**: "explore", "show me", "visualize data"
    - **SKIP IF**: User only wants model training without visualization
    - **How**: generate_plotly_dashboard(encoded, target_col, output_dir="./outputs/plots/interactive")
    - **What it creates**: Correlation heatmap, box plots, scatter plots, histograms - all interactive with zoom/pan/hover
    - **Works with ANY dataset**: Automatically detects numeric/categorical columns and generates appropriate visualizations
15. STOP when the user's request is fulfilled

**CRITICAL RULES:**

🚨 **RULE #1 - NEVER REPEAT SUCCESSFUL TOOLS**:
  - If a tool returns "✓ Completed" → MOVE TO NEXT STEP IMMEDIATELY
  - DO NOT call the same tool again (even with different arguments)
  - DO NOT call a different tool for the same task
  - Examples:
    * encode_categorical succeeded → DO NOT call execute_python_code for encoding
    * create_time_features succeeded → DO NOT call execute_python_code for time features
    * clean_missing_values succeeded → DO NOT call execute_python_code for cleaning
  - **ONLY EXCEPTION**: Different columns require separate calls (e.g., create_time_features for 'time' AND 'updated')

🚨 **RULE #2 - ENCODING IS ONE-TIME ONLY**:
  - Categorical encoding happens ONCE in step 8
  - If encode_categorical succeeds → SKIP to step 9 (generate_eda_plots)
  - DO NOT call execute_python_code with pd.get_dummies() or one-hot encoding
  - DO NOT call encode_categorical again
  - The file ./outputs/data/encoded.csv exists? → Encoding is DONE, move forward!

🚨 **RULE #3 - PREFER SPECIALIZED TOOLS**:
  - For time features → USE create_time_features(), NOT execute_python_code
  - For encoding → USE encode_categorical(), NOT execute_python_code
  - For cleaning → USE clean_missing_values(), NOT execute_python_code
  - For outliers → USE handle_outliers(), NOT execute_python_code
  - ONLY use execute_python_code when NO specialized tool exists!

- DO NOT repeat profile_dataset or detect_data_quality_issues multiple times
- DO NOT call smart_type_inference after encoding - data is ready
- **⚠️ ERROR RECOVERY - If a Tool Fails**:
  - DO NOT get stuck retrying the same failed tool
  - MOVE FORWARD to the next step (reports, visualizations, etc.)
  - Example: If hyperparameter_tuning fails → generate_combined_eda_report
  - Example: If encode_categorical fails → try force_numeric_conversion OR move to EDA
  - **NEVER let one failure stop the entire workflow!**
- **⚠️ HYPERPARAMETER TUNING - When to Use**:
  - AFTER train_baseline_models completes successfully
  - ONLY tune the BEST performing model (highest score)
  - DO NOT tune all 6 models (waste of time!)
  - Tune IF: user wants "optimize"/"improve" OR best score < 0.90
  - Skip IF: best score > 0.95 (already excellent)
  - **How to call**: hyperparameter_tuning(file_path, target_col, model_type="xgboost", n_trials=50)
  - **Model types**: "xgboost", "random_forest", "ridge", "logistic"
  - **Example**: If XGBoost wins → hyperparameter_tuning(..., model_type="xgboost")
- **⚠️ CROSS-VALIDATION - When to Use**:
  - AFTER hyperparameter_tuning (or if user explicitly requests validation)
  - Use to confirm model robustness with confidence intervals
  - IF best score > 0.85 → Cross-validate to ensure consistency
  - IF user says "validate", "production", "deploy" → ALWAYS cross-validate
  - **How to call**: perform_cross_validation(file_path, target_col, model_type="xgboost", cv_strategy="kfold", n_splits=5)
  - **Use same model_type as winner** (e.g., XGBoost→"xgboost", RandomForest→"random_forest")
  - **Returns**: Mean score ± std dev across folds (e.g., "0.92 ± 0.03" means reliable)
- **ALWAYS generate EDA reports after training/tuning** using generate_combined_eda_report
- **⭐ INTERACTIVE DASHBOARD - When to Generate**:
  - **ALWAYS IF user says**: "dashboard", "interactive", "plotly", "visualize", "charts", "graphs", "show plots", "explore data"
  - **ALWAYS IF analysis/exploration request**: "analyze dataset", "show insights", "explore patterns"
  - **SKIP IF**: User ONLY wants model training (e.g., "just train model", "only predict")
  - **Tool**: generate_plotly_dashboard(encoded, target_col, output_dir="./outputs/plots/interactive")
  - **Works with ANY dataset**: Auto-detects columns and generates appropriate visualizations
- **ONLY train models when user explicitly asks with keywords**: "train", "predict", "model", "classification", "regression", "forecast", "build a model"
- **For analysis/exploration requests ONLY**: Stop after EDA plots/dashboard - DO NOT train models
- **Read user intent carefully**: "analyze" ≠ "train", "show insights" ≠ "predict"
- **When target column is unclear**: Ask user before training

**🎯 CRITICAL EXAMPLES - DETECT INTENT CORRECTLY:**

**Type B (Visualization-Only) - NO ML WORKFLOW:**
- ✅ "Generate interactive plots for Magnitude and latitude"
  → generate_interactive_scatter(x_col="mag", y_col="latitude") → STOP
- ✅ "Create a dashboard showing correlations"
  → generate_plotly_dashboard(file_path) → STOP
- ✅ "Visualize the distribution of sales"
  → generate_interactive_histogram(column="sales") → STOP
- ✅ "Show me graphs of temperature over time"
  → generate_interactive_time_series() → STOP

**Type C (Full ML) - RUN COMPLETE WORKFLOW:**
- ✅ "Train a model to predict earthquake magnitude"
  → Full pipeline (steps 1-15)
- ✅ "Build a classifier for fraud detection"
  → Full pipeline (steps 1-15)
- ✅ "Analyze data and train model to forecast sales"
  → Full pipeline (steps 1-15)

**Type D (Unclear) - ASK USER:**
- ❓ "Analyze this earthquake dataset"
  → ASK: "Would you like me to (1) Create visualizations, (2) Train a predictive model, or (3) Both?"
- ❓ "Look at this CSV file"
  → ASK: "What would you like me to do? Visualize data or build a model?"
- ❓ "Check out my sales data"
  → ASK: "Do you want to explore the data visually or train a forecasting model?"

**⚠️ COMMON MISTAKES - AVOID THESE:**
- ❌ User says "generate plots" → Agent runs full ML workflow (WRONG!)
- ❌ User says "visualize" → Agent cleans data, encodes, trains models (WRONG!)
- ❌ User says "analyze" → Agent assumes ML training (WRONG - ask first!)
- ✅ User says "generate plots" → Agent creates plots and STOPS (CORRECT!)
- ✅ User says "train model" → Agent runs full pipeline (CORRECT!)

⭐ **CODE INTERPRETER - HOW TO USE:**

**For CODE-ONLY Tasks (Type A):**
1. User asks to "execute code", "calculate", "generate data", "create custom plot"
2. Call execute_python_code with the full Python code
3. STOP after code executes - DO NOT run ML workflow!
4. Example:
   ```
   execute_python_code(
       code='''
import numpy as np
# Calculate fibonacci
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        print(a)
        a, b = b, a+b
fib(20)
       ''',
       working_directory="./outputs/code"
   )
   # Then STOP - task complete!
   ```

**For Data Analysis Workflow (Type B):**
Use specialized tools FIRST. Only use execute_python_code for:
1. **Custom Visualizations**: Specific plot types (dropdown filters, custom buttons, animated charts)
2. **Domain-Specific Calculations**: Custom business metrics, specialized formulas
3. **Custom Data Transformations**: Unique reshaping not covered by tools
4. **Interactive Widgets**: Plotly dropdowns, sliders, buttons

**⚠️ DO NOT USE execute_python_code FOR:**
- ❌ Time feature extraction → USE create_time_features() tool
- ❌ Categorical encoding → USE encode_categorical() tool
- ❌ Missing values → USE clean_missing_values() tool
- ❌ Outliers → USE handle_outliers() tool
- ❌ Standard EDA plots → USE generate_eda_plots() or generate_plotly_dashboard()
- ❌ Model training → USE train_baseline_models() or hyperparameter_tuning()
- ❌ Tasks with dedicated tools → USE THE TOOL, NOT custom code!

**Rule of Thumb:**
- CODE-ONLY task? → execute_python_code ONCE → STOP
- Data analysis task? → Use specialized tools, execute_python_code only for custom needs
- If a specialized tool exists → USE THE TOOL, not custom code

**KEY TOOLS (77 total available via function calling):**
- force_numeric_conversion: Converts string columns to numeric (auto-detects, skips text)
- clean_missing_values: "auto" mode supported
- encode_categorical: one-hot/target/frequency encoding
- train_baseline_models: Trains multiple models automatically
- **⭐ execute_python_code**: Write and run custom Python code for ANY task not covered by tools (TRUE AI AGENT capability)
- **execute_code_from_file**: Run existing Python scripts
- Advanced: hyperparameter_tuning, train_ensemble_models, perform_eda_analysis, handle_imbalanced_data, perform_feature_scaling, detect_anomalies, detect_and_handle_multicollinearity, auto_feature_engineering, forecast_time_series, explain_predictions, generate_business_insights, perform_topic_modeling, extract_image_features, monitor_model_drift
- NEW Advanced Insights: analyze_root_cause, detect_trends_and_seasonality, detect_anomalies_advanced, perform_hypothesis_testing, analyze_distribution, perform_segment_analysis
- NEW Automation: auto_ml_pipeline (zero-config full pipeline), auto_feature_selection
- NEW Visualization: generate_all_plots, generate_data_quality_plots, generate_eda_plots, generate_model_performance_plots, generate_feature_importance_plot
- NEW Interactive Plotly Visualizations: generate_interactive_scatter, generate_interactive_histogram, generate_interactive_correlation_heatmap, generate_interactive_box_plots, generate_interactive_time_series, generate_plotly_dashboard (interactive web-based plots with zoom/pan/hover)
- NEW EDA Report Generation: generate_ydata_profiling_report (comprehensive detailed analysis with full statistics, distributions, correlations, and data quality insights)
- NEW Enhanced Feature Engineering: create_ratio_features, create_statistical_features, create_log_features, create_binned_features

**RULES:**
✅ **DETECT INTENT FIRST**: Code-only (Type A), Visualization-only (Type B), Full ML (Type C), or Unclear (Type D)?
✅ **ASK BEFORE ACTING** if user intent is ambiguous (Type D)
✅ **VISUALIZATION-ONLY**: If user just wants plots → generate_interactive_scatter OR generate_plotly_dashboard → STOP
✅ **CODE-ONLY Tasks**: execute_python_code → STOP (no ML workflow!)
✅ **FULL ML ONLY**: If user wants model training → Run complete workflow (steps 1-15)
✅ Use OUTPUT of each tool as INPUT to next
✅ Save to ./outputs/data/
✅ **CRITICAL ERROR RECOVERY - HIGHEST PRIORITY:**
   - When you see "💡 HINT: Did you mean 'X'?" → IMMEDIATELY retry with 'X'
   - When tool returns {"suggestion": "Did you mean: X?"} → Extract X and retry
   - Example: train_baseline_models fails with hint "Did you mean 'mag'?" 
     → Your NEXT call MUST be: train_baseline_models(..., target_col="mag")
   - NO OTHER CALLS until you retry with corrected parameter
✅ **READ ERROR MESSAGES CAREFULLY** - Extract actual column names from errors
✅ **When training fails with "Column X not found"**: 
   - Look for "Available columns:" in error message
   - Look for suggestion in tool_result["suggestion"]
   - Use the EXACT suggested column name
   - Common mapping: 'magnitude' → 'mag', 'latitude' → 'lat'
   - Retry IMMEDIATELY with correct column name (NO OTHER TOOLS FIRST)
✅ **When file not found**: Check previous step - if it failed, don't continue with that file
✅ **ASK USER for target column if unclear** - Don't guess!
✅ **STOP cascading errors**: If a file creation step fails, don't try to use that file in next steps
✅ When tool fails → analyze error → fix the specific issue → RETRY THAT SAME TOOL (max 1 retry per step)
❌ NO recommendations without action
❌ NO stopping after detecting issues
❌ NO repeating failed file paths - if file wasn't created, use previous working file
❌ NO repeating the same error twice - learn from error messages
❌ NO calling different tools when one fails - RETRY the failed tool with corrections first
❌ NO training models when user only wants analysis/exploration
❌ NO assuming column names - read error messages for actual names
❌ NO XML-style function syntax like <function=name />

**ERROR RECOVERY PATTERNS - FOLLOW THESE EXACTLY:**

**Pattern 1: Column Not Found**
❌ Tool fails: train_baseline_models(file_path="data.csv", target_col="magnitude")
📋 Error: "Column 'magnitude' not found. 💡 HINT: Did you mean 'mag'?"
✅ Next call MUST be: train_baseline_models(file_path="data.csv", target_col="mag")
❌ WRONG: Calling analyze_distribution or any other tool first!

**Pattern 2: File Not Found (Previous Step Failed)**
❌ Tool fails: auto_feature_engineering(...) → creates engineered_features.csv FAILED
❌ Next tool fails: train_baseline_models(file_path="engineered_features.csv") → File not found!
✅ Correct action: Use LAST SUCCESSFUL file → train_baseline_models(file_path="encoded.csv")

**Pattern 3: Missing Argument**
❌ Tool fails: "missing 1 required positional argument: 'target_col'"
✅ Next call: Include ALL required arguments

**CRITICAL RULES:**
1. If tool_result contains "suggestion", extract the suggested value and retry IMMEDIATELY
2. If you see "💡 HINT:", use that exact value in your retry
3. RETRY THE SAME TOOL with corrections before moving to different tools
4. Max 1 retry per tool - if it fails twice, move on with last successful file

**CRITICAL: Call ONE function at a time. Wait for its result before calling the next.**

**USER INTENT DETECTION:**
- Keywords for ML training: "train", "model", "predict", "classification", "regression", "forecast"
- Keywords for analysis only: "analyze", "explore", "show", "visualize", "understand", "summary"
- If ambiguous → Complete data prep, then ASK user about next steps

File chain: original → cleaned.csv → no_outliers.csv → numeric.csv → encoded.csv → models (if requested)

You are a DOER. Complete workflows based on user intent."""
    
    def _generate_cache_key(self, file_path: str, task_description: str, 
                           target_col: Optional[str] = None) -> str:
        """Generate cache key for a workflow."""
        # Include file hash to invalidate cache when data changes
        try:
            file_hash = self.cache.generate_file_hash(file_path)
        except:
            file_hash = "no_file"
        
        # Create simple string key (no kwargs unpacking to avoid dict hashing issues)
        cache_key_str = f"{file_hash}_{task_description}_{target_col or 'no_target'}"
        return self.cache._generate_key(cache_key_str)
    
    def _get_last_successful_file(self, workflow_history: List[Dict]) -> str:
        """Find the last successfully created file from workflow history."""
        # Check in reverse order for file-creating tools
        for step in reversed(workflow_history):
            result = step.get("result", {})
            if result.get("success"):
                # Check for output_path in result
                if "output_path" in result:
                    return result["output_path"]
                # For nested results
                if "result" in result and isinstance(result["result"], dict):
                    if "output_path" in result["result"]:
                        return result["result"]["output_path"]
        
        # Default fallback
        return "./outputs/data/encoded.csv"
    
    def _determine_next_step(self, stuck_tool: str, completed_tools: List[str]) -> str:
        """Determine what the next workflow step should be based on what's stuck."""
        # Map of stuck tools to their next step
        next_steps = {
            "profile_dataset": "detect_data_quality_issues",
            "detect_data_quality_issues": "generate_data_quality_plots",
            "generate_data_quality_plots": "clean_missing_values",
            "clean_missing_values": "handle_outliers",
            "handle_outliers": "force_numeric_conversion",
            "force_numeric_conversion": "create_time_features (for datetime columns)",
            "create_time_features": "encode_categorical",
            "encode_categorical": "generate_eda_plots",
            "execute_python_code": "move forward (stop writing custom code!)",
            "generate_eda_plots": "train_baseline_models",
            "train_baseline_models": "hyperparameter_tuning OR generate_combined_eda_report",
            "hyperparameter_tuning": "perform_cross_validation OR generate_combined_eda_report",
            "perform_cross_validation": "generate_combined_eda_report",
            "generate_combined_eda_report": "generate_plotly_dashboard",
            "generate_plotly_dashboard": "WORKFLOW COMPLETE"
        }
        
        return next_steps.get(stuck_tool, "generate_eda_plots OR train_baseline_models")
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool function.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tool_functions:
            return {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": get_all_tool_names()
            }
        
        try:
            tool_func = self.tool_functions[tool_name]
            
            # Fix common parameter mismatches from LLM hallucinations
            if tool_name == "generate_ydata_profiling_report":
                # LLM often calls with 'output_dir' instead of 'output_path'
                if "output_dir" in arguments and "output_path" not in arguments:
                    output_dir = arguments.pop("output_dir")
                    # Convert directory to full file path
                    arguments["output_path"] = f"{output_dir}/ydata_profile.html"
            
            # General parameter corrections for common LLM hallucinations
            if "output" in arguments and "output_path" not in arguments:
                # Many tools use 'output_path' but LLM uses 'output'
                arguments["output_path"] = arguments.pop("output")
            
            # Fix "None" string being passed as actual None
            for key, value in list(arguments.items()):
                if isinstance(value, str) and value.lower() in ["none", "null", "undefined"]:
                    arguments[key] = None
            
            result = tool_func(**arguments)
            
            # Check if tool itself returned an error (some tools return dict with 'status': 'error')
            if isinstance(result, dict) and result.get("status") == "error":
                tool_result = {
                    "success": False,
                    "tool": tool_name,
                    "arguments": arguments,
                    "error": result.get("message", result.get("error", "Tool returned error status")),
                    "error_type": "ToolError"
                }
            else:
                tool_result = {
                    "success": True,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                }
            
            # 🧠 Update session memory with tool execution
            if self.session:
                self.session.add_workflow_step(tool_name, tool_result)
            
            return tool_result
        
        except Exception as e:
            tool_result = {
                "success": False,
                "tool": tool_name,
                "arguments": arguments,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            # Still track failed tools in session
            if self.session:
                self.session.add_workflow_step(tool_name, tool_result)
            
            return tool_result
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.
        Handles matplotlib Figures, numpy arrays, and other non-serializable types.
        """
        try:
            import numpy as np
        except ImportError:
            np = None
        
        try:
            from matplotlib.figure import Figure
        except ImportError:
            Figure = None
        
        # Handle dictionaries recursively
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        
        # Handle lists recursively
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        
        # Handle matplotlib Figure objects
        elif Figure and isinstance(obj, Figure):
            return f"<Matplotlib Figure: {id(obj)}>"
        
        # Handle numpy arrays
        elif np and isinstance(obj, np.ndarray):
            return f"<NumPy array: shape={obj.shape}>"
        
        # Handle numpy scalar types
        elif hasattr(obj, 'item') and callable(obj.item):
            try:
                return obj.item()
            except:
                return str(obj)
        
        # Handle other non-serializable objects
        elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, type(None))):
            return f"<{obj.__class__.__name__} object>"
        
        # Already serializable
        return obj
    
    def _summarize_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """
        Summarize tool result for LLM consumption.
        Extracts only essential info to avoid token bloat from large dataset outputs.
        """
        if not tool_result.get("success"):
            # Always return errors in full
            return json.dumps({
                "error": tool_result.get("error"),
                "error_type": tool_result.get("error_type")
            }, indent=2)
        
        result = tool_result.get("result", {})
        tool_name = tool_result.get("tool", "")
        
        # Create concise summary based on tool type
        summary = {"status": "success"}
        
        # Profile dataset - extract key stats only
        if tool_name == "profile_dataset":
            summary.update({
                "rows": result.get("basic_info", {}).get("num_rows"),
                "cols": result.get("basic_info", {}).get("num_columns"),
                "numeric_cols": len(result.get("numeric_columns", [])),
                "categorical_cols": len(result.get("categorical_columns", [])),
                "datetime_cols": len(result.get("datetime_columns", [])),
                "memory_mb": result.get("basic_info", {}).get("memory_usage_mb"),
                "missing_values": result.get("basic_info", {}).get("missing_values", 0)
            })
        
        # Data quality - extract issue counts
        elif tool_name == "detect_data_quality_issues":
            issues = result.get("issues", {})
            summary.update({
                "missing_values": len(issues.get("missing_values", [])),
                "duplicate_rows": result.get("duplicate_count", 0),
                "high_cardinality": len(issues.get("high_cardinality", [])),
                "constant_cols": len(issues.get("constant_columns", [])),
                "outliers": len(issues.get("outliers", [])),
                "total_issues": sum([
                    len(issues.get("missing_values", [])),
                    result.get("duplicate_count", 0),
                    len(issues.get("high_cardinality", [])),
                    len(issues.get("constant_columns", [])),
                    len(issues.get("outliers", []))
                ])
            })
        
        # File operations - just confirm path
        elif tool_name in ["clean_missing_values", "handle_outliers", "fix_data_types", 
                           "force_numeric_conversion", "encode_categorical", "smart_type_inference"]:
            summary.update({
                "output_path": result.get("output_path"),
                "message": result.get("message", ""),
                "rows_affected": result.get("rows_removed", result.get("rows_affected", 0))
            })
        
        # Training - extract model performance only
        elif tool_name == "train_baseline_models":
            models = result.get("models", {})
            best = result.get("best_model", {})
            best_model_name = best.get("name") if isinstance(best, dict) else best
            summary.update({
                "best_model": best_model_name,
                "models_trained": list(models.keys()),
                "best_score": best.get("score") if isinstance(best, dict) else None,
                "task_type": result.get("task_type")
            })
        
        # Report generation
        elif tool_name == "generate_model_report":
            summary.update({
                "report_path": result.get("report_path"),
                "message": "Report generated successfully"
            })
        
        # Default: extract message and status
        else:
            summary.update({
                "message": result.get("message", str(result)[:200]),  # Max 200 chars
                "output_path": result.get("output_path")
            })
        
        return json.dumps(summary, indent=2)
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """Format tool result for LLM consumption (alias for summarize)."""
        return self._summarize_tool_result(tool_result)
    
    def _compress_tools_registry(self) -> List[Dict]:
        """
        Create compressed version of tools registry.
        Keeps ALL 46 tools but removes verbose parameter descriptions.
        """
        compressed = []
        
        for tool in self.tools_registry:
            # Compress parameters by removing descriptions
            params = tool["function"]["parameters"]
            compressed_params = {
                "type": params["type"],
                "properties": {},
                "required": list(params.get("required", []))  # Create new list, not reference
            }
            
            # Keep only type info for properties, remove descriptions
            for prop_name, prop_value in params.get("properties", {}).items():
                compressed_prop = {}
                
                # Handle oneOf (like clean_missing_values strategy parameter)
                if "oneOf" in prop_value:
                    # Deep copy to avoid reference issues
                    compressed_prop["oneOf"] = json.loads(json.dumps(prop_value["oneOf"]))
                else:
                    compressed_prop["type"] = prop_value.get("type", "string")
                
                # Keep enum if present (important for validation)
                if "enum" in prop_value:
                    compressed_prop["enum"] = list(prop_value["enum"])  # Create new list
                
                # Keep array items type - handle both "array" and ["string", "array"]
                prop_type = prop_value.get("type")
                is_array_type = False
                
                if isinstance(prop_type, list):
                    is_array_type = "array" in prop_type
                elif prop_type == "array":
                    is_array_type = True
                
                if is_array_type and "items" in prop_value:
                    compressed_prop["items"] = {"type": prop_value["items"].get("type", "string")}
                
                compressed_params["properties"][prop_name] = compressed_prop
            
            compressed_tool = {
                "type": tool["type"],
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"][:100],  # Short description
                    "parameters": compressed_params
                }
            }
            compressed.append(compressed_tool)
        
        return compressed
    
    def _compress_tool_result(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress tool results for small context models (production-grade approach).
        
        Keep only:
        - Status (success/failure)
        - Key metrics (5-10 most important numbers)
        - File paths created  
        - Next action hints
        
        Full results stored in workflow_history and session memory.
        LLM doesn't need verbose output - only decision-making info.
        
        Args:
            tool_name: Name of the tool executed
            result: Full tool result dict
            
        Returns:
            Compressed result dict (typically 100-500 tokens vs 5K-10K)
        """
        try:
            if not result.get("success", True):
                # Keep full error info (critical for debugging)
                return result
            
            compressed = {
                "success": True,
                "tool": tool_name
            }
            
            # Tool-specific compression rules
            if tool_name == "profile_dataset":
                # Original: ~5K tokens with full stats
                # Compressed: ~200 tokens with key metrics
                r = result.get("result", {})
                compressed["summary"] = {
                    "rows": r.get("num_rows"),
                    "cols": r.get("num_columns"),
                    "missing_pct": r.get("missing_percentage"),
                    "numeric_cols": len(r.get("numeric_columns", [])),
                    "categorical_cols": len(r.get("categorical_columns", [])),
                    "file_size_mb": round(r.get("memory_usage_mb", 0), 1),
                    "key_columns": list(r.get("columns", {}).keys())[:5]  # First 5 columns only
                }
                compressed["next_steps"] = ["clean_missing_values", "detect_data_quality_issues"]
                
            elif tool_name == "detect_data_quality_issues":
                r = result.get("result", {})
                compressed["summary"] = {
                    "total_issues": r.get("total_issues", 0),
                    "critical_issues": r.get("critical_issues", 0),
                    "missing_data": r.get("has_missing"),
                    "outliers": r.get("has_outliers"),
                    "duplicates": r.get("has_duplicates")
                }
                compressed["next_steps"] = ["clean_missing_values", "handle_outliers"]
                
            elif tool_name in ["clean_missing_values", "handle_outliers", "encode_categorical"]:
                r = result.get("result", {})
                compressed["summary"] = {
                    "output_file": r.get("output_file", r.get("output_path")),
                    "rows_processed": r.get("rows_after", r.get("num_rows")),
                    "changes_made": bool(r.get("changes", {}) or r.get("imputed_columns"))
                }
                compressed["next_steps"] = ["Use this file for next step"]
                
            elif tool_name == "train_baseline_models":
                r = result.get("result", {})
                models = r.get("models", [])
                if models and isinstance(models, list) and len(models) > 0:
                    # Filter to only dict entries (defensive)
                    valid_models = [m for m in models if isinstance(m, dict) and "test_score" in m]
                    if valid_models:
                        best = max(valid_models, key=lambda m: m.get("test_score", 0))
                        compressed["summary"] = {
                            "best_model": best.get("model"),
                            "test_score": round(best.get("test_score", 0), 4),
                            "train_score": round(best.get("train_score", 0), 4),
                            "task_type": r.get("task_type"),
                            "models_trained": len(valid_models)
                        }
                    else:
                        # Fallback if no valid models
                        compressed["summary"] = {
                            "task_type": r.get("task_type"),
                            "status": "No valid models trained"
                        }
                else:
                    compressed["summary"] = {"status": "No models found"}
                compressed["next_steps"] = ["hyperparameter_tuning", "generate_combined_eda_report"]
                
            elif tool_name in ["generate_plotly_dashboard", "generate_ydata_profiling_report", "generate_combined_eda_report"]:
                r = result.get("result", {})
                compressed["summary"] = {
                    "report_path": r.get("report_path", r.get("output_path")),
                    "report_type": tool_name,
                    "success": True
                }
                compressed["next_steps"] = ["Report ready for viewing"]
                
            elif tool_name == "hyperparameter_tuning":
                r = result.get("result", {})
                compressed["summary"] = {
                    "best_params": r.get("best_params", {}),
                    "best_score": round(r.get("best_score", 0), 4),
                    "model_type": r.get("model_type"),
                    "trials_completed": r.get("n_trials")
                }
                compressed["next_steps"] = ["perform_cross_validation", "generate_model_performance_plots"]
                
            else:
                # Generic compression: Keep only key fields
                r = result.get("result", {})
                if isinstance(r, dict):
                    # Extract key fields (common patterns)
                    key_fields = {}
                    for key in ["output_path", "output_file", "status", "message", "success"]:
                        if key in r:
                            key_fields[key] = r[key]
                    compressed["summary"] = key_fields or {"result": "completed"}
                else:
                    compressed["summary"] = {"result": str(r)[:200] if r else "completed"}
                compressed["next_steps"] = ["Continue workflow"]
            
            return compressed
            
        except Exception as e:
            # If compression fails, return minimal safe result
            print(f"⚠️  Compression failed for {tool_name}: {str(e)}")
            return {
                "success": result.get("success", True),
                "tool": tool_name,
                "summary": {"status": "completed (compression failed)"},
                "result": result.get("result", {}) if isinstance(result.get("result"), dict) else {}
            }


    def _parse_text_tool_calls(self, text_response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from text-based LLM response (ReAct pattern).
        Supports multiple formats:
        - JSON: {"tool": "tool_name", "arguments": {...}}
        - Function: tool_name(arg1="value", arg2="value")
        - Markdown: ```json {...} ```
        """
        import re
        tool_calls = []
        
        # Pattern 1: JSON blocks (most reliable)
        json_pattern = r'```(?:json)?\s*(\{[^\`]+\})\s*```'
        json_matches = re.findall(json_pattern, text_response, re.DOTALL)
        
        for match in json_matches:
            try:
                tool_data = json.loads(match)
                if "tool" in tool_data or "function" in tool_data:
                    tool_name = tool_data.get("tool") or tool_data.get("function")
                    arguments = tool_data.get("arguments") or tool_data.get("args") or {}
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    })
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: Function call format - tool_name(arg1="value", arg2=123)
        if not tool_calls:
            func_pattern = r'(\w+)\s*\((.*?)\)'
            for match in re.finditer(func_pattern, text_response):
                tool_name = match.group(1)
                args_str = match.group(2)
                
                # Check if this looks like a known tool
                if any(tool_name in tool["function"]["name"] for tool in self._compress_tools_registry()):
                    # Parse arguments
                    arguments = {}
                    arg_pattern = r'(\w+)\s*=\s*(["\']?)([^,\)]+)\2'
                    for arg_match in re.finditer(arg_pattern, args_str):
                        key = arg_match.group(1)
                        value = arg_match.group(3)
                        # Try to parse as number/bool
                        if value.lower() == "true":
                            arguments[key] = True
                        elif value.lower() == "false":
                            arguments[key] = False
                        elif value.isdigit():
                            arguments[key] = int(value)
                        else:
                            arguments[key] = value
                    
                    tool_calls.append({
                        "id": f"call_{len(tool_calls)}",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments)
                        }
                    })
        
        return tool_calls
    
    def _convert_to_gemini_tools(self, groq_tools: List[Dict]) -> List[Dict]:
        """
        Convert Groq/OpenAI format tools to Gemini format.
        
        Groq format: {"type": "function", "function": {...}}
        Gemini format: {"name": "...", "description": "...", "parameters": {...}}
        
        Gemini requires:
        - Property types as UPPERCASE (STRING, NUMBER, BOOLEAN, ARRAY, OBJECT)
        - No "type": "object" at root parameters level
        """
        gemini_tools = []
        
        def convert_type(json_type: str) -> str:
            """Convert JSON Schema type to Gemini type."""
            type_map = {
                "string": "STRING",
                "number": "NUMBER",
                "integer": "INTEGER",
                "boolean": "BOOLEAN",
                "array": "ARRAY",
                "object": "OBJECT"
            }
            
            # Handle list of types (e.g., ["string", "array"])
            if isinstance(json_type, list):
                # Use the first type in the list, or ARRAY if array is in the list
                if "array" in json_type:
                    return "ARRAY"
                elif len(json_type) > 0:
                    return type_map.get(json_type[0], "STRING")
                else:
                    return "STRING"
            
            return type_map.get(json_type, "STRING")
        
        def convert_properties(properties: Dict) -> Dict:
            """Convert property definitions to Gemini format."""
            converted = {}
            for prop_name, prop_def in properties.items():
                new_def = {}
                
                # Handle oneOf (like clean_missing_values strategy)
                if "oneOf" in prop_def:
                    # For oneOf, just pick the first option or simplify
                    if isinstance(prop_def["oneOf"], list) and len(prop_def["oneOf"]) > 0:
                        first_option = prop_def["oneOf"][0]
                        if "type" in first_option:
                            new_def["type"] = convert_type(first_option["type"])
                        if "enum" in first_option:
                            new_def["enum"] = first_option["enum"]
                    else:
                        new_def["type"] = "STRING"
                elif "type" in prop_def:
                    prop_type = prop_def["type"]
                    
                    # Handle list of types (e.g., ["string", "array"])
                    if isinstance(prop_type, list):
                        converted_type = convert_type(prop_type)
                        new_def["type"] = converted_type
                        
                        # If it's an array type, we MUST provide items for Gemini
                        if converted_type == "ARRAY":
                            if "items" in prop_def:
                                items_type = prop_def["items"].get("type", "string")
                                new_def["items"] = {"type": convert_type(items_type)}
                            else:
                                # Default to STRING items if not specified
                                new_def["items"] = {"type": "STRING"}
                    else:
                        new_def["type"] = convert_type(prop_type)
                        
                        # Handle arrays
                        if prop_type == "array" and "items" in prop_def:
                            items_type = prop_def["items"].get("type", "string")
                            new_def["items"] = {"type": convert_type(items_type)}
                        elif prop_type == "array":
                            # Array without items specification - default to STRING
                            new_def["items"] = {"type": "STRING"}
                    
                    # Keep enum
                    if "enum" in prop_def:
                        new_def["enum"] = prop_def["enum"]
                else:
                    new_def["type"] = "STRING"
                
                # Keep description if present
                if "description" in prop_def:
                    new_def["description"] = prop_def["description"]
                
                converted[prop_name] = new_def
            
            return converted
        
        for tool in groq_tools:
            func = tool["function"]
            params = func.get("parameters", {})
            
            # Convert parameters to Gemini format
            gemini_params = {
                "type": "OBJECT",  # Gemini uses UPPERCASE
                "properties": convert_properties(params.get("properties", {})),
                "required": params.get("required", [])
            }
            
            gemini_tool = {
                "name": func["name"],
                "description": func["description"],
                "parameters": gemini_params
            }
            gemini_tools.append(gemini_tool)
        
        return gemini_tools
    
    def _update_workflow_state(self, tool_name: str, tool_result: Dict[str, Any]):
        """
        Update workflow state based on tool execution.
        This reduces the need to keep full tool results in LLM context.
        """
        if not tool_result.get("success", True):
            return  # Don't update state on failures
        
        result_data = tool_result.get("result", {})
        
        # Profile dataset
        if tool_name == "profile_dataset":
            self.workflow_state.update_profiling({
                "num_rows": result_data.get("num_rows"),
                "num_columns": result_data.get("num_columns"),
                "missing_percentage": result_data.get("missing_percentage"),
                "numeric_columns": result_data.get("numeric_columns", []),
                "categorical_columns": result_data.get("categorical_columns", [])
            })
        
        # Quality check
        elif tool_name == "detect_data_quality_issues":
            self.workflow_state.update_quality({
                "total_issues": result_data.get("total_issues", 0),
                "has_missing": result_data.get("has_missing", False),
                "has_outliers": result_data.get("has_outliers", False),
                "has_duplicates": result_data.get("has_duplicates", False)
            })
        
        # Cleaning tools
        elif tool_name in ["clean_missing_values", "handle_outliers", "encode_categorical"]:
            self.workflow_state.update_cleaning({
                "output_file": result_data.get("output_file") or result_data.get("output_path"),
                "rows_processed": result_data.get("rows_after") or result_data.get("num_rows"),
                "tool": tool_name
            })
        
        # Feature engineering
        elif tool_name in ["create_time_features", "create_interaction_features", "auto_feature_engineering"]:
            self.workflow_state.update_features({
                "output_file": result_data.get("output_file") or result_data.get("output_path"),
                "new_features": result_data.get("new_columns", []),
                "tool": tool_name
            })
        
        # Model training
        elif tool_name == "train_baseline_models":
            models = result_data.get("models", [])
            best_model = None
            if models and isinstance(models, list):
                valid_models = [m for m in models if isinstance(m, dict) and "test_score" in m]
                if valid_models:
                    best_model = max(valid_models, key=lambda m: m.get("test_score", 0))
            
            self.workflow_state.update_modeling({
                "best_model": best_model.get("model") if best_model else None,
                "best_score": best_model.get("test_score") if best_model else None,
                "models_trained": len(valid_models) if best_model else 0,
                "task_type": result_data.get("task_type")
            })
    
    def analyze(self, file_path: str, task_description: str, 
               target_col: Optional[str] = None, 
               use_cache: bool = True,
               stream: bool = True,
               max_iterations: int = 20) -> Dict[str, Any]:
        """
        Main entry point for data science analysis.
        
        Args:
            file_path: Path to dataset file
            task_description: Natural language description of the task
            target_col: Optional target column name
            use_cache: Whether to use cached results
            stream: Whether to stream LLM responses
            max_iterations: Maximum number of tool execution iterations
            
        Returns:
            Analysis results including summary and tool outputs
        """
        start_time = time.time()
        
        # 🚀 LOCAL SCHEMA EXTRACTION (NO LLM) - Extract metadata before any LLM calls
        print("🔍 Extracting dataset schema locally (no LLM)...")
        schema_info = extract_schema_local(file_path, sample_rows=3)
        
        if 'error' not in schema_info:
            # Update workflow state with schema
            self.workflow_state.update_dataset_info(schema_info)
            print(f"✅ Schema extracted: {schema_info['num_rows']} rows × {schema_info['num_columns']} cols")
            print(f"   File size: {schema_info['file_size_mb']} MB")
            
            # Infer task type if target column provided
            if target_col and target_col in schema_info['columns']:
                inferred_task = infer_task_type(target_col, schema_info)
                if inferred_task:
                    self.workflow_state.task_type = inferred_task
                    self.workflow_state.target_column = target_col
                    print(f"   Task type inferred: {inferred_task}")
        else:
            print(f"⚠️  Schema extraction failed: {schema_info.get('error')}")
        
        # Check cache
        if use_cache:
            cache_key = self._generate_cache_key(file_path, task_description, target_col)
            cached = self.cache.get(cache_key)
            if cached:
                print("✓ Using cached results")
                return cached
        
        # Build initial messages
        # Use dynamic prompts for small context models
        if self.use_compact_prompts:
            from .dynamic_prompts import build_compact_system_prompt
            system_prompt = build_compact_system_prompt(user_query=task_description)
            print("🔧 Using compact prompt for small context window")
        else:
            system_prompt = self._build_system_prompt()
        
        # 🧠 RESOLVE AMBIGUITY USING SESSION MEMORY
        original_file_path = file_path
        original_target_col = target_col
        
        if self.session:
            # Check if request has ambiguous references
            resolved_params = self.session.resolve_ambiguity(task_description)
            
            # Use resolved params if user didn't specify
            if not file_path or file_path == "":
                if resolved_params.get("file_path"):
                    file_path = resolved_params["file_path"]
                    print(f"📝 Using dataset from session: {file_path}")
            
            if not target_col:
                if resolved_params.get("target_col"):
                    target_col = resolved_params["target_col"]
                    print(f"📝 Using target column from session: {target_col}")
            
            # Show session context if available
            if self.session.last_dataset or self.session.last_model:
                context_summary = self.session.get_context_summary()
                print(f"\n{context_summary}\n")
        
        # 🎯 PROACTIVE INTENT DETECTION - Tell LLM which tools to use BEFORE it tries wrong ones
        task_lower = task_description.lower()
        
        # Detect user intent
        wants_viz = any(kw in task_lower for kw in ["plot", "graph", "visualiz", "dashboard", "chart", "show", "display", "create", "generate"])
        wants_clean = any(kw in task_lower for kw in ["clean", "missing", "impute"])
        wants_features = any(kw in task_lower for kw in ["feature", "engineer", "time-based", "extract features"])
        wants_train = any(kw in task_lower for kw in ["train", "model", "predict", "best model"])
        
        # 📊 DETECT SPECIFIC PLOT TYPE - Match user's exact visualization request
        plot_type_guidance = ""
        if wants_viz:
            if "histogram" in task_lower or "distribution" in task_lower or "freq" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Histogram\n✅ Use: generate_interactive_histogram\n❌ Do NOT use: generate_interactive_scatter (that's for scatter plots!)"
            elif "scatter" in task_lower or "relationship" in task_lower or "correlation" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Scatter Plot\n✅ Use: generate_interactive_scatter\n❌ Do NOT use: generate_interactive_histogram (that's for distributions!)"
            elif "box plot" in task_lower or "boxplot" in task_lower or "outlier" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Box Plot\n✅ Use: generate_interactive_box_plots"
            elif "time series" in task_lower or "trend" in task_lower or "over time" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Time Series\n✅ Use: generate_interactive_time_series"
            elif "heatmap" in task_lower or "correlation" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Heatmap\n✅ Use: generate_interactive_correlation_heatmap"
            elif "dashboard" in task_lower or "all plot" in task_lower:
                plot_type_guidance = "\n\n📊 **PLOT TYPE DETECTED**: Dashboard/Multiple Plots\n✅ Use: generate_plotly_dashboard OR generate_all_plots"
            else:
                # Generic visualization - let LLM decide based on data
                plot_type_guidance = "\n\n📊 **PLOT TYPE**: Generic visualization\n✅ Choose appropriate tool based on:\n- Histogram: Single numeric variable distribution\n- Scatter: Relationship between 2 numeric variables\n- Box Plot: Compare distributions across categories\n- Time Series: Data with datetime column"
        
        # Build specific guidance based on intent
        workflow_guidance = ""
        
        if wants_train:
            # Full ML pipeline - ALWAYS run complete workflow for model training
            workflow_guidance = (
                "\n\n🎯 **WORKFLOW**: Full ML Pipeline (Training Requested)\n"
                "Execute ALL steps for best model performance:\n"
                "1. Profile dataset (understand data)\n"
                "2. Clean missing values (data quality)\n"
                "3. Handle outliers (prevent bias)\n"
                "4. Create features (time features, interactions)\n"
                "5. Encode categorical (prepare for ML)\n"
                "6. Train models (baseline + optimization)\n"
                "7. Generate visualizations (feature importance, residuals, performance)\n"
                "8. Create reports (comprehensive analysis)\n\n"
                "⚠️ ALL tools allowed - cleaning, feature engineering, visualization, and training!"
            )
        elif wants_clean and wants_viz and not wants_train:
            # Multi-intent: Clean + Visualize
            workflow_guidance = (
                "\n\n🎯 **WORKFLOW**: Multi-Intent (Clean + Visualize)\n"
                "Steps:\n"
                "1. clean_missing_values\n"
                "2. handle_outliers\n"
                "3. generate_interactive_scatter OR generate_plotly_dashboard\n"
                "4. STOP (no training!)"
            )
        elif wants_viz and not wants_train and not wants_clean:
            # Visualization only
            workflow_guidance = (
                f"\n\n🎯 **WORKFLOW**: Visualization ONLY{plot_type_guidance}\n"
                "⚠️ DO NOT run profiling or cleaning tools!\n"
                "✅ YOUR FIRST CALL: Use the EXACT plot type mentioned above\n"
                "✅ Then STOP immediately (no training, no cleaning needed!)"
            )
        elif wants_features and not wants_train:
            # Feature engineering only
            workflow_guidance = (
                "\n\n🎯 **WORKFLOW**: Feature Engineering ONLY\n"
                "Steps:\n"
                "1. (Optional) profile_dataset if you need column names\n"
                "2. create_time_features OR encode_categorical OR create_interaction_features\n"
                "3. STOP (no training!)"
            )
        elif wants_clean and not wants_train and not wants_viz:
            # Cleaning only
            workflow_guidance = (
                "\n\n🎯 **WORKFLOW**: Data Cleaning ONLY\n"
                "Steps:\n"
                "1. (Optional) profile_dataset to see issues\n"
                "2. clean_missing_values\n"
                "3. handle_outliers\n"
                "4. STOP (no training, no feature engineering!)"
            )
        else:
            # Default full workflow
            workflow_guidance = "\n\n🎯 **WORKFLOW**: Complete Analysis\nExecute: profile → clean → encode → train → report"
        
        # Build user message with workflow state context (minimal, not full history)
        state_context = ""
        if self.workflow_state.dataset_info:
            # Include schema summary instead of raw data
            info = self.workflow_state.dataset_info
            state_context = f"""
**Dataset Schema** (extracted locally):
- Rows: {info['num_rows']:,} | Columns: {info['num_columns']}
- Size: {info['file_size_mb']} MB
- Numeric columns: {len(info['numeric_columns'])}
- Categorical columns: {len(info['categorical_columns'])}
- Sample columns: {', '.join(list(info['columns'].keys())[:8])}{'...' if len(info['columns']) > 8 else ''}
"""
        
        user_message = f"""Please analyze the dataset and complete the following task:

**Dataset**: {file_path}
**Task**: {task_description}
**Target Column**: {target_col if target_col else 'Not specified - please infer from data'}{state_context}{workflow_guidance}"""

        #🧠 Store file path in session memory for follow-up requests
        if self.session and file_path:
            self.session.update(last_dataset=file_path)
            if target_col:
                self.session.update(last_target_col=target_col)
            print(f"💾 Saved to session: dataset={file_path}, target={target_col}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Track workflow
        workflow_history = []
        iteration = 0
        tool_call_counter = {}  # Track how many times each tool has been called
        
        # Prepare tools once
        tools_to_use = self._compress_tools_registry()
        
        # For Gemini, use the existing model without tools (text-only mode)
        # Gemini tool schema is incompatible with OpenAI/Groq format
        # Tool execution is handled by our orchestrator, not by Gemini itself
        gemini_chat = None
        if self.provider == "gemini":
            gemini_chat = self.gemini_model.start_chat(history=[])
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # 🚀 AGGRESSIVE CONVERSATION PRUNING (LangChain pattern)
                # Keep only: system + user + last 4 exchanges (8 messages)
                # This prevents token bloat while maintaining context
                if len(messages) > 10:
                    # Keep: system prompt [0], user query [1], last 4 exchanges
                    messages = [messages[0], messages[1]] + messages[-8:]
                    print(f"✂️  Pruned conversation (keeping last 4 exchanges, ~4K tokens saved)")
                
                # 🔍 Token estimation and warning
                estimated_tokens = sum(
                    len(str(m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', ''))) // 4 
                    for m in messages
                )
                if estimated_tokens > 8000:
                    # Emergency pruning - keep only last 2 exchanges
                    messages = [messages[0], messages[1]] + messages[-4:]
                    print(f"⚠️  Emergency pruning (conversation > 8K tokens)")
                
                # 💰 Token budget management (TPM limit)
                if self.provider in ["mistral", "groq"]:
                    # Reset minute counter if needed
                    elapsed = time.time() - self.minute_start_time
                    if elapsed > 60:
                        print(f"🔄 Token budget reset (was {self.tokens_this_minute}/{self.tpm_limit})")
                        self.tokens_this_minute = 0
                        self.minute_start_time = time.time()
                    
                    # Check if we're close to TPM limit (use 70% threshold to be safe)
                    if self.tokens_this_minute + estimated_tokens > self.tpm_limit * 0.7:
                        wait_time = 60 - elapsed
                        if wait_time > 0:
                            print(f"⏸️  Token budget: {self.tokens_this_minute}/{self.tpm_limit} used ({(self.tokens_this_minute/self.tpm_limit)*100:.0f}%)")
                            print(f"   Next request would use ~{estimated_tokens} tokens → exceeds safe limit")
                            print(f"   Waiting {wait_time:.0f}s for budget reset...")
                            time.sleep(wait_time)
                            self.tokens_this_minute = 0
                            self.minute_start_time = time.time()
                            print(f"✅ Token budget reset complete")
                    else:
                        print(f"💰 Token budget: {self.tokens_this_minute}/{self.tpm_limit} ({(self.tokens_this_minute/self.tpm_limit)*100:.0f}%)")

                
                # Rate limiting - wait if needed
                if self.min_api_call_interval > 0:
                    time_since_last_call = time.time() - self.last_api_call_time
                    if time_since_last_call < self.min_api_call_interval:
                        wait_time = self.min_api_call_interval - time_since_last_call
                        print(f"⏳ Rate limiting: waiting {wait_time:.1f}s...")
                        time.sleep(wait_time)
                
                # Initialize variables before try block to avoid UnboundLocalError
                tool_calls = None
                final_content = None
                response_message = None
                
                # Call LLM with function calling (provider-specific)
                if self.provider == "mistral":
                    try:
                        response = self.mistral_client.chat(
                            model=self.model,
                            messages=messages,
                            tools=tools_to_use,
                            tool_choice="auto",
                            temperature=0.1,
                            max_tokens=4096
                        )
                        
                        self.api_calls_made += 1
                        self.last_api_call_time = time.time()
                        
                        # Track tokens used (for TPM budget management)
                        if hasattr(response, 'usage') and response.usage:
                            tokens_used = response.usage.total_tokens
                            self.tokens_this_minute += tokens_used
                            print(f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute")
                        
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls
                        final_content = response_message.content
                        
                    except Exception as mistral_error:
                        error_str = str(mistral_error)
                        print(f"❌ MISTRAL ERROR: {error_str[:300]}")
                        raise
                
                elif self.provider == "groq":
                    try:
                        response = self.groq_client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools_to_use,
                            tool_choice="auto",
                            parallel_tool_calls=False,  # Disable parallel calls to prevent XML format errors
                            temperature=0.1,  # Low temperature for consistent outputs
                            max_tokens=4096
                        )
                        
                        self.api_calls_made += 1
                        self.last_api_call_time = time.time()
                        
                        # Track tokens used (for TPM budget management)
                        if hasattr(response, 'usage') and response.usage:
                            tokens_used = response.usage.total_tokens
                            self.tokens_this_minute += tokens_used
                            print(f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute")
                        
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls
                        final_content = response_message.content
                        
                    except Exception as groq_error:
                        # Check if it's a rate limit error (429)
                        error_str = str(groq_error)
                        if "rate_limit" in error_str.lower() or "429" in error_str:
                            # Parse retry delay from error message if available
                            retry_delay = 60  # Default to 60s for TPM limit
                            
                            # Try to extract retry delay from error
                            import re
                            delay_match = re.search(r'retry.*?(\d+).*?second', error_str, re.IGNORECASE)
                            if delay_match:
                                retry_delay = int(delay_match.group(1))
                            elif "tokens per minute" in error_str or "TPM" in error_str:
                                retry_delay = 60
                            elif "tokens per day" in error_str or "TPD" in error_str:
                                # Daily limit - give up immediately
                                print(f"❌ GROQ DAILY TOKEN LIMIT EXHAUSTED (100K tokens/day)")
                                print(f"   Your daily quota resets at UTC midnight")
                                print(f"   Error: {error_str[:400]}")
                                raise ValueError(f"Groq daily quota exhausted. Please wait for reset.\n{error_str[:500]}")
                            
                            # TPM limit - wait and retry
                            print(f"⚠️  GROQ TPM RATE LIMIT (rolling 60s window)")
                            print(f"   Groq uses account-wide rolling window - previous requests still count")
                            print(f"   Waiting {retry_delay}s and retrying...")
                            print(f"   Error: {error_str[:300]}")
                            
                            time.sleep(retry_delay)
                            
                            # Retry the request
                            print(f"🔄 Retrying after {retry_delay}s delay...")
                            response = self.groq_client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                tools=tools_to_use,
                                tool_choice="auto",
                                parallel_tool_calls=False,
                                temperature=0.1,
                                max_tokens=4096
                            )
                            
                            self.api_calls_made += 1
                            self.last_api_call_time = time.time()
                            
                            # Track tokens used
                            if hasattr(response, 'usage') and response.usage:
                                tokens_used = response.usage.total_tokens
                                self.tokens_this_minute += tokens_used
                                print(f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute")
                            
                            response_message = response.choices[0].message
                            tool_calls = response_message.tool_calls
                            final_content = response_message.content
                        else:
                            # Not a rate limit error, re-raise
                            raise
                
                # Check if done (no tool calls)
                if not tool_calls:
                    # Final response
                    final_summary = final_content or "Analysis completed"
                    
                    # 🧠 Save conversation to session memory
                    if self.session:
                        self.session.add_conversation(task_description, final_summary)
                        self.session_store.save(self.session)
                        print(f"\n✅ Session saved: {self.session.session_id}")
                    
                    result = {
                        "status": "success",
                        "summary": final_summary,
                        "workflow_history": workflow_history,
                        "iterations": iteration,
                        "api_calls": self.api_calls_made,
                        "execution_time": round(time.time() - start_time, 2)
                    }
                    
                    # Cache result
                    if use_cache:
                        self.cache.set(cache_key, result, metadata={
                            "file_path": file_path,
                            "task": task_description
                        })
                    
                    return result
                
                # Execute tool calls (provider-specific format)
                if self.provider == "groq":
                    messages.append(response_message)
                
                for tool_call in tool_calls:
                    # Extract tool name and args (provider-specific)
                    if self.provider == "groq":
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                    elif self.provider == "gemini":
                        tool_name = tool_call.name
                        # Convert protobuf args to Python dict
                        tool_args = {}
                        for key, value in tool_call.args.items():
                            # Handle different protobuf value types
                            if isinstance(value, (str, int, float, bool)):
                                tool_args[key] = value
                            elif hasattr(value, '__iter__') and not isinstance(value, str):
                                # Convert lists/repeated fields
                                tool_args[key] = list(value)
                            else:
                                # Fallback: try to convert to string
                                tool_args[key] = str(value)
                        tool_call_id = f"gemini_{iteration}_{tool_name}"
                    
                    # ⚠️ WORKFLOW STATE TRACKING: Block redundant operations
                    completed_tools = [step["tool"] for step in workflow_history]
                    
                    # 🎯 COMPREHENSIVE INTENT DETECTION SYSTEM
                    # Detect user's actual intent to prevent running full pipeline for partial tasks
                    
                    task_lower = task_description.lower()
                    
                    # Define intent keywords
                    visualization_keywords = ["plot", "graph", "visualiz", "dashboard", "chart", "show", "display", "create", "generate"]
                    cleaning_keywords = ["clean", "remove missing", "handle missing", "fill missing", "impute"]
                    feature_eng_keywords = ["feature", "engineer", "create features", "add features", "extract features", "time-based"]
                    profiling_keywords = ["profile", "explore", "understand", "summarize", "describe", "report", "analysis", "overview", "insights"]
                    ml_training_keywords = ["train", "model", "predict", "forecast", "classification", "regression", "tune", "optimize", "best model"]
                    
                    # Detect what user wants (can be multiple intents)
                    wants_visualization = any(kw in task_lower for kw in visualization_keywords)
                    wants_cleaning = any(kw in task_lower for kw in cleaning_keywords)
                    wants_feature_eng = any(kw in task_lower for kw in feature_eng_keywords)
                    wants_profiling = any(kw in task_lower for kw in profiling_keywords)
                    wants_ml_training = any(kw in task_lower for kw in ml_training_keywords)
                    
                    # Negation detection - "without", "no", "don't", "skip"
                    has_negation = any(neg in task_lower for neg in ["without", "no train", "don't train", "skip train", "no model"])
                    
                    # Count how many intents detected
                    intent_count = sum([wants_visualization, wants_cleaning, wants_feature_eng, wants_profiling, wants_ml_training])
                    
                    # Multi-intent detection: "Train model + feature engineering + graphs"
                    is_multi_intent = intent_count > 1
                    
                    # Determine intent type and allowed tools
                    # 🔥 CRITICAL: ML training ALWAYS needs full pipeline + visualization
                    if wants_ml_training and not has_negation:
                        # Full ML pipeline - training requires EVERYTHING
                        user_intent = "FULL_ML_PIPELINE"
                        allowed_tool_categories = ["all"]  # Allow all tools (cleaning, features, viz, training, reports)
                        
                    elif is_multi_intent and not wants_ml_training:
                        # Multi-intent WITHOUT training (e.g., "clean and visualize")
                        user_intent = "MULTI_INTENT"
                        allowed_tool_categories = []
                        
                        # Add categories based on detected intents
                        if wants_profiling:
                            allowed_tool_categories.append("profiling")
                        if wants_cleaning:
                            # Cleaning may need profiling to identify issues
                            allowed_tool_categories.extend(["profiling", "cleaning"])
                        if wants_feature_eng:
                            # Feature engineering may need profiling for column info
                            allowed_tool_categories.extend(["profiling", "cleaning", "feature_engineering"])
                        if wants_visualization:
                            allowed_tool_categories.append("visualization")
                        
                        # Remove duplicates
                        allowed_tool_categories = list(set(allowed_tool_categories))
                        
                    elif wants_visualization and not wants_ml_training:
                        # Visualization ONLY
                        user_intent = "VISUALIZATION_ONLY"
                        allowed_tool_categories = ["visualization"]
                        
                    elif wants_cleaning and not wants_ml_training:
                        # Data cleaning ONLY
                        user_intent = "CLEANING_ONLY"
                        allowed_tool_categories = ["profiling", "cleaning"]
                        
                    elif wants_feature_eng and not wants_ml_training:
                        # Feature engineering ONLY (may need cleaning first)
                        user_intent = "FEATURE_ENGINEERING_ONLY"
                        allowed_tool_categories = ["profiling", "cleaning", "feature_engineering"]
                        
                    elif wants_profiling and not wants_ml_training:
                        # Exploratory analysis ONLY
                        user_intent = "EXPLORATORY_ANALYSIS"
                        allowed_tool_categories = ["profiling", "visualization"]
                        
                    else:
                        # Default: Full pipeline if unclear
                        user_intent = "FULL_ML_PIPELINE"
                        allowed_tool_categories = ["all"]
                    
                    # Categorize tools
                    tool_categories = {
                        "profiling": ["profile_dataset", "detect_data_quality_issues", "analyze_correlations", "get_smart_summary"],
                        "cleaning": ["clean_missing_values", "handle_outliers", "fix_data_types", "force_numeric_conversion", "smart_type_inference"],
                        "feature_engineering": ["create_time_features", "encode_categorical", "create_interaction_features", 
                                               "create_aggregation_features", "auto_feature_engineering", "create_ratio_features",
                                               "create_statistical_features", "create_log_features", "create_binned_features"],
                        "ml_training": ["train_baseline_models", "hyperparameter_tuning", "perform_cross_validation", 
                                       "auto_ml_pipeline", "train_ensemble_models"],
                        "visualization": ["generate_interactive_scatter", "generate_interactive_histogram",
                                        "generate_interactive_correlation_heatmap", "generate_interactive_box_plots",
                                        "generate_interactive_time_series", "generate_plotly_dashboard",
                                        "generate_eda_plots", "generate_all_plots", "generate_data_quality_plots"]
                    }
                    
                    # Determine if tool should be blocked
                    should_block_tool = False
                    block_reason = ""
                    
                    if "all" not in allowed_tool_categories:
                        # Find which category this tool belongs to
                        tool_category = None
                        for category, tools in tool_categories.items():
                            if tool_name in tools:
                                tool_category = category
                                break
                        
                        # Block if tool category not in allowed categories
                        if tool_category and tool_category not in allowed_tool_categories:
                            should_block_tool = True
                            block_reason = f"User intent: {user_intent} (only allows: {', '.join(allowed_tool_categories)})"
                    
                    # 🚫 BLOCK tool if it doesn't match user intent
                    if should_block_tool:
                        print(f"\n🚫 BLOCKED: {tool_name}")
                        print(f"   Task: '{task_description}'")
                        print(f"   User Intent: {user_intent}")
                        print(f"   Reason: {block_reason}")
                        print(f"   Allowed categories: {', '.join(allowed_tool_categories)}")
                        
                        # Check if user's requested task is already complete
                        task_complete = False
                        completion_summary = ""
                        
                        if user_intent == "VISUALIZATION_ONLY":
                            viz_tools_used = [t for t in completed_tools if t in tool_categories["visualization"]]
                            if viz_tools_used:
                                task_complete = True
                                completion_summary = f"✅ Visualization completed: {', '.join(viz_tools_used)}"
                        
                        elif user_intent == "CLEANING_ONLY":
                            cleaning_tools_used = [t for t in completed_tools if t in tool_categories["cleaning"]]
                            if cleaning_tools_used:
                                task_complete = True
                                completion_summary = f"✅ Data cleaning completed: {', '.join(cleaning_tools_used)}"
                        
                        elif user_intent == "FEATURE_ENGINEERING_ONLY":
                            fe_tools_used = [t for t in completed_tools if t in tool_categories["feature_engineering"]]
                            if fe_tools_used:
                                task_complete = True
                                completion_summary = f"✅ Feature engineering completed: {', '.join(fe_tools_used)}"
                        
                        elif user_intent == "EXPLORATORY_ANALYSIS":
                            analysis_tools_used = [t for t in completed_tools if t in tool_categories["profiling"] or t in tool_categories["visualization"]]
                            if analysis_tools_used:
                                task_complete = True
                                completion_summary = f"✅ Exploratory analysis completed: {', '.join(analysis_tools_used)}"
                        
                        if task_complete:
                            print(f"   {completion_summary}")
                            
                            final_summary = (
                                f"{completion_summary}\n\n"
                                f"Task: {task_description}\n"
                                f"Intent: {user_intent}\n\n"
                                f"Tools executed:\n"
                                f"{chr(10).join(['- ' + tool for tool in completed_tools])}\n\n"
                                f"Check ./outputs/ for results."
                            )
                            
                            return {
                                "status": "completed",
                                "summary": final_summary,
                                "workflow_history": workflow_history,
                                "iterations": iteration,
                                "api_calls": self.api_calls_made,
                                "execution_time": round(time.time() - start_time, 2)
                            }
                        
                        # Build guidance for LLM based on intent
                        if user_intent == "VISUALIZATION_ONLY":
                            next_step_guidance = (
                                f"✅ YOUR NEXT CALL MUST BE a visualization tool:\n"
                                f"   - generate_interactive_scatter\n"
                                f"   - generate_plotly_dashboard\n"
                                f"   - generate_eda_plots\n"
                            )
                        elif user_intent == "CLEANING_ONLY":
                            next_step_guidance = (
                                f"✅ YOUR NEXT CALL should be a cleaning tool:\n"
                                f"   - clean_missing_values\n"
                                f"   - handle_outliers\n"
                                f"   - fix_data_types\n"
                                f"Then STOP (no training!)"
                            )
                        elif user_intent == "FEATURE_ENGINEERING_ONLY":
                            next_step_guidance = (
                                f"✅ YOUR NEXT CALL should be a feature engineering tool:\n"
                                f"   - create_time_features\n"
                                f"   - encode_categorical\n"
                                f"   - create_interaction_features\n"
                                f"Then STOP (no training!)"
                            )
                        elif user_intent == "EXPLORATORY_ANALYSIS":
                            next_step_guidance = (
                                f"✅ YOUR NEXT CALL should be profiling or visualization:\n"
                                f"   - profile_dataset\n"
                                f"   - generate_eda_plots\n"
                                f"   - analyze_correlations\n"
                                f"Then STOP (no training!)"
                            )
                        else:
                            next_step_guidance = "Continue with appropriate tools for the task."
                        
                        # Send blocking message to LLM
                        block_warning = {
                            "role": "user",
                            "content": (
                                f"🚫 BLOCKED: '{tool_name}' does not match user intent!\n\n"
                                f"Task: '{task_description}'\n"
                                f"Detected Intent: {user_intent}\n"
                                f"Allowed: {', '.join(allowed_tool_categories)}\n"
                                f"Blocked: {tool_name} (category: {tool_category if 'tool_category' in locals() else 'unknown'})\n\n"
                                f"{next_step_guidance}\n\n"
                                f"DO NOT call blocked tools. Proceed with allowed tools only!"
                            )
                        }
                        
                        # Track blocking
                        workflow_history.append({
                            "step": len(workflow_history) + 1,
                            "tool": "BLOCKED",
                            "blocked_tool": tool_name,
                            "reason": block_reason,
                            "user_intent": user_intent
                        })
                        
                        messages.append(block_warning)
                        continue
                    
                    # CRITICAL: Block execute_python_code if it's doing encoding/time features
                    if tool_name == "execute_python_code":
                        code = tool_args.get("code", "")
                        
                        # ✅ ALLOW: Data cleanup (dropping columns, fixing types, etc.)
                        is_cleanup = any(pattern in code.lower() for pattern in [
                            "drop(columns=", "drop_duplicates", "fillna", "dropna",
                            "select_dtypes", ".drop(", "errors='ignore'"
                        ])
                        
                        # Block if trying to do encoding (pd.get_dummies, one-hot, etc.) - UNLESS it's cleanup
                        if any(pattern in code.lower() for pattern in ["get_dummies", "onehot", "one-hot", "one_hot"]):
                            if "encode_categorical" in completed_tools and not is_cleanup:
                                print(f"\n🚫 BLOCKED: execute_python_code attempting to re-encode!")
                                print(f"   encode_categorical already completed. Skipping this call.")
                                print(f"   Using existing file: {self._get_last_successful_file(workflow_history)}")
                                
                                block_warning = {
                                    "role": "user",
                                    "content": (
                                        f"🚫 BLOCKED: You tried to use execute_python_code for encoding, but encode_categorical ALREADY completed!\n\n"
                                        f"Encoding is DONE. The file exists: {self._get_last_successful_file(workflow_history)}\n\n"
                                        f"MOVE TO NEXT STEP: generate_eda_plots OR train_baseline_models\n\n"
                                        f"DO NOT:\n"
                                        f"- Call execute_python_code for encoding\n"
                                        f"- Call encode_categorical again\n"
                                        f"- Repeat any completed step\n\n"
                                        f"PROCEED to the next workflow step immediately!"
                                    )
                                }
                                messages.append(block_warning)
                                continue
                        
                        # Block if trying to do time feature extraction - UNLESS it's cleanup
                        if any(pattern in code.lower() for pattern in ["dt.year", "dt.month", "dt.day", "dt.hour", "strptime", "to_datetime"]):
                            if "create_time_features" in completed_tools and not is_cleanup:
                                print(f"\n🚫 BLOCKED: execute_python_code attempting time feature extraction!")
                                print(f"   create_time_features already completed. Skipping this call.")
                                
                                block_warning = {
                                    "role": "user",
                                    "content": (
                                        f"🚫 BLOCKED: You tried to use execute_python_code for time features, but create_time_features ALREADY completed!\n\n"
                                        f"Time features are DONE. Use the existing file: {self._get_last_successful_file(workflow_history)}\n\n"
                                        f"MOVE TO NEXT STEP: encode_categorical\n\n"
                                        f"DO NOT call execute_python_code for time feature extraction!"
                                    )
                                }
                                messages.append(block_warning)
                                continue
                    
                    # CRITICAL: Block create_time_features if already called for both datetime columns
                    if tool_name == "create_time_features":
                        time_feature_calls = [step for step in workflow_history if step["tool"] == "create_time_features"]
                        if len(time_feature_calls) >= 2:  # Already called for 'time' and 'updated'
                            print(f"\n🚫 BLOCKED: create_time_features already called {len(time_feature_calls)} times!")
                            print(f"   Time features extracted for all datetime columns. Skipping.")
                            
                            block_warning = {
                                "role": "user",
                                "content": (
                                    f"🚫 BLOCKED: create_time_features already called {len(time_feature_calls)} times!\n\n"
                                    f"Time features extraction is COMPLETE for all datetime columns ('time' and 'updated').\n\n"
                                    f"MOVE TO NEXT STEP: encode_categorical\n\n"
                                    f"DO NOT call create_time_features again!"
                                )
                            }
                            messages.append(block_warning)
                            continue
                    
                    # CRITICAL: Block encode_categorical if already completed
                    if tool_name == "encode_categorical":
                        if "encode_categorical" in completed_tools:
                            print(f"\n🚫 BLOCKED: encode_categorical already completed!")
                            print(f"   Categorical encoding is DONE. Skipping.")
                            
                            block_warning = {
                                "role": "user",
                                "content": (
                                    f"🚫 BLOCKED: encode_categorical ALREADY completed!\n\n"
                                    f"Encoding is DONE. Use file: {self._get_last_successful_file(workflow_history)}\n\n"
                                    f"MOVE TO NEXT STEP: generate_eda_plots\n\n"
                                    f"DO NOT call encode_categorical again!"
                                )
                            }
                            messages.append(block_warning)
                            continue
                    
                    # CRITICAL: Block smart_type_inference after encoding (data is ready!)
                    if tool_name == "smart_type_inference":
                        if "encode_categorical" in completed_tools or "execute_python_code" in completed_tools:
                            print(f"\n🚫 BLOCKED: smart_type_inference after encoding!")
                            print(f"   Data is already encoded and ready. Skipping type inference.")
                            
                            block_warning = {
                                "role": "user",
                                "content": (
                                    f"🚫 BLOCKED: smart_type_inference is NOT needed after encoding!\n\n"
                                    f"The data is already encoded and ready for modeling.\n\n"
                                    f"MOVE TO NEXT STEP: generate_eda_plots OR train_baseline_models\n\n"
                                    f"DO NOT call smart_type_inference after encoding!"
                                )
                            }
                            messages.append(block_warning)
                            continue
                    
                    # ⚠️ LOOP DETECTION: Prevent calling the same tool multiple times in a row
                    # EXCEPTION: Don't apply loop detection for execute_python_code in code-only tasks
                    tool_call_counter[tool_name] = tool_call_counter.get(tool_name, 0) + 1
                    
                    # Detect if this is a code-only task (no ML workflow tools used)
                    ml_tools = ["profile_dataset", "detect_data_quality_issues", "clean_missing_values", 
                               "encode_categorical", "train_baseline_models"]
                    is_code_only_task = not any(tool in completed_tools for tool in ml_tools)
                    
                    # Skip loop detection for execute_python_code in code-only tasks
                    should_check_loops = not (is_code_only_task and tool_name == "execute_python_code")
                    
                    # Check for loops (same tool called 2+ times consecutively)
                    if should_check_loops and tool_call_counter[tool_name] >= 2:
                        # Check if the last call was also this tool (consecutive repetition)
                        if workflow_history and workflow_history[-1]["tool"] == tool_name:
                            print(f"\n⚠️  LOOP DETECTED: {tool_name} called {tool_call_counter[tool_name]} times consecutively!")
                            print(f"   This indicates the workflow is stuck. Skipping and forcing progression.")
                            print(f"   Last successful file: {self._get_last_successful_file(workflow_history)}")
                            
                            # Check if we've completed the main workflow (reports generated)
                            completed_tools = [step["tool"] for step in workflow_history]
                            reports_generated = any(tool in completed_tools for tool in [
                                "generate_combined_eda_report", 
                                "generate_plotly_dashboard",
                                "generate_ydata_profiling_report"
                            ])
                            training_done = "train_baseline_models" in completed_tools
                            
                            # If reports done and we're looping, mark as complete
                            if reports_generated and training_done:
                                print(f"   ✅ Main workflow complete. Marking as DONE.")
                                final_summary = (
                                    f"Analysis completed successfully! Main steps finished:\n"
                                    f"- Data profiling and cleaning\n"
                                    f"- Model training ({completed_tools.count('train_baseline_models')} models trained)\n"
                                    f"- {'Hyperparameter tuning' if 'hyperparameter_tuning' in completed_tools else 'Baseline models'}\n"
                                    f"- Comprehensive reports generated\n"
                                    f"- Interactive visualizations created\n\n"
                                    f"Check ./outputs/ for all results."
                                )
                                
                                return {
                                    "status": "completed",
                                    "summary": final_summary,
                                    "workflow_history": workflow_history,
                                    "iterations": iteration,
                                    "api_calls": self.api_calls_made,
                                    "execution_time": round(time.time() - start_time, 2)
                                }
                            
                            # Otherwise, force LLM to move on with VERY STRONG warning
                            next_step = self._determine_next_step(tool_name, completed_tools)
                            loop_warning = {
                                "role": "user",
                                "content": (
                                    f"🚨 CRITICAL ERROR: You are STUCK IN A LOOP! 🚨\n\n"
                                    f"You called '{tool_name}' {tool_call_counter[tool_name]} times consecutively.\n"
                                    f"This step is ALREADY COMPLETE (✓ Completed shown above).\n\n"
                                    f"**DO NOT call {tool_name} again!**\n"
                                    f"**DO NOT call execute_python_code for the same task!**\n\n"
                                    f"NEXT STEP: {next_step}\n\n"
                                    f"Last successful output file: {self._get_last_successful_file(workflow_history)}\n"
                                    f"Use this file and proceed to the NEXT step immediately.\n\n"
                                    f"Remember:\n"
                                    f"- If a tool succeeds (✓ Completed) → NEVER call it again\n"
                                    f"- Do NOT use execute_python_code for tasks that have dedicated tools\n"
                                    f"- Follow the workflow: Steps 1→2→3→...→15 (ONE TIME EACH)"
                                )
                            }
                            messages.append(loop_warning)
                            continue  # Skip this tool call
                    
                    print(f"\n🔧 Executing: {tool_name}")
                    try:
                        print(f"   Arguments: {json.dumps(tool_args, indent=2)}")
                    except:
                        print(f"   Arguments: {tool_args}")
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Check for errors and display them prominently
                    if not tool_result.get("success", True):
                        error_msg = tool_result.get("error", "Unknown error")
                        error_type = tool_result.get("error_type", "Error")
                        print(f"   ❌ FAILED: {tool_name}")
                        print(f"   ⚠️  Error Type: {error_type}")
                        print(f"   ⚠️  Error Message: {error_msg}")
                        
                        # Add recovery guidance with last successful file
                        last_successful_file = self._get_last_successful_file(workflow_history)
                        if last_successful_file:
                            tool_result["recovery_guidance"] = (
                                f"This tool failed. Use the last successful file for next steps: {last_successful_file}\n"
                                f"Do NOT try to use the failed tool's output file."
                            )
                            print(f"   🔄 Recovery: Use {last_successful_file} for next step")
                        
                        # Special handling for execute_python_code errors
                        if tool_name == "execute_python_code":
                            stderr = tool_result.get("stderr", "")
                            hints = tool_result.get("hints", [])
                            
                            if stderr:
                                print(f"   📄 Code Error Details:")
                                # Show last 10 lines of stderr (most relevant)
                                stderr_lines = stderr.split('\n')[-10:]
                                for line in stderr_lines:
                                    if line.strip():
                                        print(f"      {line}")
                            
                            if hints:
                                print(f"   💡 Suggestions:")
                                for hint in hints:
                                    print(f"      {hint}")
                            
                            # Add suggestion to use specialized tools instead
                            if error_type in ["PermissionError", "FileNotFoundError", "KeyError"]:
                                tool_result["suggestion"] = (
                                    f"Consider using specialized tools instead of execute_python_code:\n"
                                    f"- For file operations: use clean_missing_values(), encode_categorical(), etc.\n"
                                    f"- For data transformations: use create_ratio_features(), create_statistical_features(), etc.\n"
                                    f"- Specialized tools are more robust and handle edge cases better!"
                                )
                        
                        # Extract helpful info from common errors and add to result
                        if "Column" in error_msg and "not found" in error_msg and "Available columns:" in error_msg:
                            # Extract the column that was searched for and available columns
                            import re
                            searched = re.search(r"Column '([^']+)' not found", error_msg)
                            available = re.search(r"Available columns: (.+?)(?:\n|$)", error_msg)
                            if searched and available:
                                searched_col = searched.group(1)
                                available_cols = [c.strip() for c in available.group(1).split(',')]
                                
                                # Find similar column names (case-insensitive partial match)
                                suggestions = []
                                searched_lower = searched_col.lower()
                                for col in available_cols[:20]:  # Check first 20
                                    if searched_lower in col.lower() or col.lower() in searched_lower:
                                        suggestions.append(col)
                                
                                if suggestions:
                                    tool_result["suggestion"] = f"Did you mean: {suggestions[0]}? (Similar columns: {', '.join(suggestions[:3])})"
                                    print(f"   💡 HINT: Did you mean '{suggestions[0]}'?")
                        
                        # For critical tools, show detailed error to user
                        if tool_name in ["train_baseline_models", "auto_ml_pipeline"]:
                            print(f"\n🔴 CRITICAL ERROR in {tool_name}:")
                            print(f"   {error_msg}\n")
                    else:
                        print(f"   ✓ Completed: {tool_name}")
                    
                    # Track in workflow
                    workflow_history.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    # 🗂️ UPDATE WORKFLOW STATE (reduces need to send full history to LLM)
                    self._update_workflow_state(tool_name, tool_result)
                    
                    # ⚡ CRITICAL FIX: Add tool result back to messages so LLM sees it in next iteration!
                    if self.provider in ["mistral", "groq"]:
                        # For Mistral/Groq, add tool message with the result
                        # **COMPRESS RESULT** for small context models
                        clean_tool_result = self._make_json_serializable(tool_result)
                        
                        # Smart compression: Keep only what LLM needs for next decision
                        compressed_result = self._compress_tool_result(tool_name, clean_tool_result)
                        tool_response_content = json.dumps(compressed_result)
                        
                        # If tool failed, prepend ERROR indicator to make it obvious
                        if not tool_result.get("success", True):
                            error_msg = tool_result.get("error", "Unknown error")
                            suggestion = tool_result.get("suggestion", "")
                            
                            # Create VERY EXPLICIT error message
                            tool_response_content = json.dumps({
                                "❌ TOOL_FAILED": True,
                                "tool_name": tool_name,
                                "error": error_msg,
                                "suggestion": suggestion,
                                "⚠️ ACTION_REQUIRED": f"RETRY {tool_name} with corrected parameters. Do NOT call other tools first!",
                                "💡 HINT": suggestion if suggestion else "Check error message for details"
                            })
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_response_content
                        })
                    
                    elif self.provider == "gemini":
                        # For Gemini, add to messages for history tracking
                        # Gemini uses function responses differently but we still track
                        # Clean tool_result to make it JSON-serializable
                        clean_tool_result = self._make_json_serializable(tool_result)
                        tool_response_content = json.dumps(clean_tool_result)
                        
                        # If tool failed, make error VERY explicit
                        if not tool_result.get("success", True):
                            error_msg = tool_result.get("error", "Unknown error")
                            suggestion = tool_result.get("suggestion", "")
                            
                            tool_response_content = json.dumps({
                                "❌ TOOL_FAILED": True,
                                "tool_name": tool_name,
                                "error": error_msg,
                                "suggestion": suggestion,
                                "⚠️ ACTION_REQUIRED": f"RETRY {tool_name} with corrected parameters",
                                "💡 HINT": suggestion if suggestion else "Check error message"
                            })
                        
                        messages.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_response_content
                        })
                    
                    # Debug: Check if training completed
                    if tool_name == "train_baseline_models":
                        print(f"[DEBUG] train_baseline_models executed!")
                        print(f"[DEBUG]   tool_result keys: {list(tool_result.keys())}")
                        print(f"[DEBUG]   'best_model' in tool_result: {'best_model' in tool_result}")
                        if isinstance(tool_result, dict) and 'result' in tool_result:
                            print(f"[DEBUG]   Nested result keys: {list(tool_result['result'].keys()) if isinstance(tool_result['result'], dict) else 'Not a dict'}")
                            print(f"[DEBUG]   'best_model' in nested result: {'best_model' in tool_result['result'] if isinstance(tool_result['result'], dict) else False}")
                        if "best_model" in tool_result:
                            print(f"[DEBUG]   best_model value: {tool_result['best_model']}")
                    
                    # AUTO-FINISH DISABLED: Let agent complete full workflow including EDA reports
                    # Previously auto-finish would exit immediately after training, preventing
                    # report generation. Now the agent continues to generate visualizations and reports.
            
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                error_str = str(e)
                
                # Log the actual error for debugging
                print(f"❌ ERROR in analyze loop: {e}")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Full error: {error_str}")
                print(f"   Traceback:\n{error_traceback}")
                
                # Handle rate limit errors with retry (be more specific to avoid false positives)
                if ("429" in error_str or 
                    "Resource has been exhausted" in error_str or
                    "quota exceeded" in error_str.lower()):
                    
                    retry_delay = 10
                    if "retry after" in error_str.lower():
                        import re
                        match = re.search(r'retry after (\d+)', error_str.lower())
                        if match:
                            retry_delay = min(int(match.group(1)) + 2, 15)
                    
                    print(f"⏳ Rate limit detected (429/quota). Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)
                    iteration -= 1
                    continue
                
                # For other errors, don't retry - just report and continue
                print(f"   Traceback:\n{error_traceback}")
                
                # 🧠 Save session even on error
                if self.session:
                    self.session.add_conversation(task_description, f"Error: {str(e)}")
                    self.session_store.save(self.session)
                
                return {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": error_traceback,
                    "workflow_history": workflow_history,
                    "iterations": iteration
                }
        
        # Max iterations reached
        # 🧠 Save session
        if self.session:
            self.session.add_conversation(task_description, "Workflow incomplete - max iterations reached")
            self.session_store.save(self.session)
        
        return {
            "status": "incomplete",
            "message": f"Reached maximum iterations ({max_iterations})",
            "workflow_history": workflow_history,
            "iterations": iteration
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear_all()
    
    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        return self.session.session_id if self.session else None
    
    def clear_session(self) -> None:
        """Clear current session context (start fresh)."""
        if self.session:
            self.session.clear()
            print("✅ Session context cleared")
        else:
            print("⚠️  No active session")
    
    def get_session_context(self) -> str:
        """Get human-readable session context summary."""
        if self.session:
            return self.session.get_context_summary()
        else:
            return "No active session"

