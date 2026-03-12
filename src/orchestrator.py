"""
Data Science Copilot Orchestrator
Main orchestration class that uses LLM function calling to execute data science workflows.
Supports multiple providers: Groq and Gemini.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
import httpx

from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

from .cache.cache_manager import CacheManager
from .tools.tools_registry import TOOLS, get_all_tool_names, get_tools_by_category
from .tools.agent_tool_mapping import (get_tools_for_agent, filter_tools_by_names, 
                                        get_agent_description, suggest_next_agent)
from .reasoning.reasoning_trace import get_reasoning_trace, reset_reasoning_trace
from .reasoning.findings import FindingsAccumulator, Finding
from .reasoning.reasoner import Reasoner, ReasoningOutput
from .reasoning.evaluator import Evaluator, EvaluationOutput
from .reasoning.synthesizer import Synthesizer
from .routing.intent_classifier import IntentClassifier, IntentResult
from .session_memory import SessionMemory
from .session_store import SessionStore
from .workflow_state import WorkflowState
from .utils.schema_extraction import extract_schema_local, infer_task_type
from .progress_manager import progress_manager

# New systems for improvements
from .utils.semantic_layer import get_semantic_layer
from .utils.error_recovery import get_recovery_manager, retry_with_fallback
from .utils.token_budget import get_token_manager
from .utils.parallel_executor import get_parallel_executor, ToolExecution, TOOL_WEIGHTS, ToolWeight
import asyncio
from difflib import get_close_matches
from .tools import (
    # Basic Tools (13) - UPDATED: Added get_smart_summary + 3 wrangling tools
    profile_dataset,
    detect_data_quality_issues,
    analyze_correlations,
    detect_label_errors,  # NEW: cleanlab label error detection
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
    # AutoGluon Tools (9) - NEW: AutoML at Scale
    train_with_autogluon,
    predict_with_autogluon,
    forecast_with_autogluon,
    optimize_autogluon_model,
    analyze_autogluon_model,
    extend_autogluon_training,
    train_multilabel_autogluon,
    backtest_timeseries,
    analyze_timeseries_model,
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
    # Production/MLOps (5 + 2 new)
    monitor_model_drift,
    explain_predictions,
    generate_model_card,
    perform_ab_test_analysis,
    detect_feature_leakage,
    monitor_drift_evidently,
    explain_with_dtreeviz,
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
    # EDA Report Generation (2) - NEW PHASE 2
    generate_ydata_profiling_report,
    generate_sweetviz_report,
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
                 use_compact_prompts: bool = False,
                 progress_callback: Optional[callable] = None):
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
            progress_callback: Optional callback function to report progress (receives step_name, status)
        """
        # Load environment variables
        load_dotenv()
        
        # Store progress callback
        self.progress_callback = progress_callback
        
        # Store HTTP session key for SSE streaming (set by app.py)
        self.http_session_key = None
        
        # Determine provider
        self.provider = provider or os.getenv("LLM_PROVIDER", "mistral").lower()
        
        # Use compact prompts as specified (multi-agent has focused prompts per specialist)
        self.use_compact_prompts = use_compact_prompts
        
        if self.provider == "mistral":
            # Initialize Mistral client
            api_key = mistral_api_key or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Mistral API key must be provided or set in MISTRAL_API_KEY env var")
            
            # Try new SDK first (v1.x), fall back to old SDK (v0.x)
            try:
                from mistralai import Mistral  # New SDK (v1.x)
                self.mistral_client = Mistral(api_key=api_key.strip())
            except ImportError:
                # Fall back to old SDK (v0.x)
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
        
        # 🧠 Initialize semantic layer for column understanding and agent routing
        self.semantic_layer = get_semantic_layer()
        
        # 🛡️ Initialize error recovery manager
        self.recovery_manager = get_recovery_manager()
        
        # 📊 Initialize token budget manager
        # Calculate max tokens based on provider
        provider_max_tokens = {
            "mistral": 128000,  # Mistral Large
            "groq": 32768,     # Llama 3.3 70B
            "gemini": 1000000  # Gemini 2.5 Flash
        }
        max_context = provider_max_tokens.get(self.provider, 128000)
        self.token_manager = get_token_manager(model=self.model, max_tokens=max_context)
        
        # ⚡ Parallel executor DISABLED - running tools sequentially for stability
        # self.parallel_executor = get_parallel_executor()
        self.parallel_executor = None  # Disabled for scale optimization
        
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
        
        # 🔍 Initialize reasoning trace for decision tracking
        self.reasoning_trace = get_reasoning_trace()
        
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
        
        # Multi-Agent Architecture - Specialist Agents
        self.specialist_agents = self._initialize_specialist_agents()
        self.active_agent = "Orchestrator"  # Track which agent is working
        
        # Determine output directory based on environment
        # In production (HuggingFace/Cloud Run), use /tmp for ephemeral storage
        if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
            self.output_base = Path("/tmp/data_science_agent/outputs")
        else:
            self.output_base = Path("./outputs")
        
        # Set environment variable for tools to use
        os.environ["DS_AGENT_OUTPUT_DIR"] = str(self.output_base)
        
        # Ensure output directories exist
        self.output_base.mkdir(parents=True, exist_ok=True)
        (self.output_base / "models").mkdir(exist_ok=True)
        (self.output_base / "reports").mkdir(exist_ok=True)
        (self.output_base / "data").mkdir(exist_ok=True)
        (self.output_base / "plots").mkdir(exist_ok=True)
        (self.output_base / "plots" / "interactive").mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_base}")
    
    def _build_tool_functions_map(self) -> Dict[str, callable]:
        """Build mapping of tool names to their functions - All 75 tools."""
        return {
            # Basic Tools (13) - UPDATED: Added 4 new tools
            "profile_dataset": profile_dataset,
            "detect_data_quality_issues": detect_data_quality_issues,
            "analyze_correlations": analyze_correlations,
            "detect_label_errors": detect_label_errors,  # NEW: cleanlab
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
            # AutoGluon Tools (9) - NEW: AutoML at Scale
            "train_with_autogluon": train_with_autogluon,
            "predict_with_autogluon": predict_with_autogluon,
            "forecast_with_autogluon": forecast_with_autogluon,
            "optimize_autogluon_model": optimize_autogluon_model,
            "analyze_autogluon_model": analyze_autogluon_model,
            "extend_autogluon_training": extend_autogluon_training,
            "train_multilabel_autogluon": train_multilabel_autogluon,
            "backtest_timeseries": backtest_timeseries,
            "analyze_timeseries_model": analyze_timeseries_model,
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
            # "train_ensemble_models": train_ensemble_models,  # DISABLED - Too resource intensive for scale
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
            # Production/MLOps (5 + 2 new)
            "monitor_model_drift": monitor_model_drift,
            "explain_predictions": explain_predictions,
            "generate_model_card": generate_model_card,
            "perform_ab_test_analysis": perform_ab_test_analysis,
            "detect_feature_leakage": detect_feature_leakage,
            "monitor_drift_evidently": monitor_drift_evidently,
            "explain_with_dtreeviz": explain_with_dtreeviz,
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
            # EDA Report Generation (2) - NEW PHASE 2
            "generate_ydata_profiling_report": generate_ydata_profiling_report,
            "generate_sweetviz_report": generate_sweetviz_report,
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
    
    def _extract_content_text(self, content) -> str:
        """Extract text from message content (handles both string and list formats)"""
        if content is None:
            return None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Content is list of objects like [{'type': 'text', 'text': '...'}]
            text_parts = []
            for item in content:
                if isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
                elif isinstance(item, str):
                    text_parts.append(item)
            return ''.join(text_parts)
        return str(content)
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for the copilot."""
        return """You are an autonomous Data Science Agent. You EXECUTE tasks, not advise.

**CRITICAL: User Interface Integration & Response Formatting**
- The user interface automatically displays clickable buttons for all generated plots, reports, and outputs
- **ABSOLUTELY FORBIDDEN**: NEVER EVER mention file paths in your responses
  - ❌ NEVER write: "./outputs/...", "/outputs/...", "saved to", "output file:", "file path:"
  - ❌ NEVER use markdown code blocks for file paths (no backticks around paths)
  - ❌ NEVER say: "Output File:", "Saved to:", "File:", "Path:", "Location:"
- **WHAT TO SAY INSTEAD**:
  - ✅ "Generated an interactive correlation heatmap"
  - ✅ "Cleaned the dataset by handling missing values"
  - ✅ "Created visualizations showing the relationships"
  - ✅ "Trained multiple models and optimized the best performer"
- Users can click buttons to view outputs - you don't need to tell them where files are
- Use clean, aesthetic formatting with sections, bullets, and proper spacing

**🎨 MARKDOWN FORMATTING RULES (CRITICAL FOR CLEAN UI):**
- **INLINE CODE**: Keep inline code on the SAME LINE as surrounding text
  - ✅ CORRECT: "Extract features like `column_a`, `column_b`, and `column_c` from the dataset."
  - ❌ WRONG: "Extract features like\n`column_a`\n,\n`column_b`\n"
- **LISTS**: Write list items as complete sentences on single lines
  - ✅ CORRECT: "1. Extract `feature_1`, `feature_2`, `feature_3` from the datetime column"
  - ❌ WRONG: "1. Extract\n`feature_1`\n,\n`feature_2`\n"
- **TABLES**: Keep each cell's content on ONE line, no line breaks inside cells
  - ✅ CORRECT: "| `feature_name` | Numeric | Extracted from `source_column` |"
  - ❌ WRONG: "|\n`feature_name`\n| Numeric |\nExtracted from\n`source_column`\n|"
- **COMMAS/PUNCTUATION**: Keep punctuation attached to text, not on separate lines
  - ✅ CORRECT: "`col1`, `col2`, and `col3`"
  - ❌ WRONG: "`col1`\n,\n`col2`"
- **INLINE CODE IN SENTENCES**: Always embed column/feature names naturally in prose
  - ✅ CORRECT: "The `price` column shows correlation with `quantity` and `discount`."
  - ❌ WRONG: "The\n`price`\ncolumn shows correlation with\n`quantity`\n"
- **GENERAL**: Write flowing prose. Never put backticked terms on their own lines unless showing code blocks.

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
- **Example**: "Generate interactive scatter plot for price vs quantity" → generate_interactive_scatter → DONE ✓

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
- **🎯 IMPORTANT**: ALWAYS generate ydata_profiling_report at the END of workflow for comprehensive final analysis
- **Example**: "Train a model to predict sales/price/target" → Full pipeline + ydata_profiling_report at end

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
- **Example**: "Generate interactive plots for price and quantity"
  → generate_interactive_scatter(x_col="price", y_col="quantity") → DONE ✓

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
10. **ONLY IF USER EXPLICITLY REQUESTED ML**: train_with_autogluon(file_path=encoded, target_col=target_col, task_type="auto", time_limit=120, presets="medium_quality")
    - AutoGluon is the DEFAULT training tool. It trains 10+ models with auto ensembling.
    - It handles raw data directly (categoricals, missing values) but we clean first for best results.
    - Fallback: train_baseline_models(encoded, target_col, task_type="auto") if AutoGluon unavailable.
    - For multi-label prediction: train_multilabel_autogluon(file_path, target_cols=["col1","col2"])
    - Post-training: optimize_autogluon_model(model_path, operation="refit_full|distill|calibrate_threshold|deploy_optimize")
    - Model inspection: analyze_autogluon_model(model_path, operation="summary|transform_features|info")
    - Add more models: extend_autogluon_training(model_path, operation="fit_extra")
    - For time series: forecast_with_autogluon (supports covariates, holidays, model selection)
    - TS backtesting: backtest_timeseries(file_path, target_col, time_col, num_val_windows=3)
    - TS analysis: analyze_timeseries_model(model_path, data_path, time_col, operation="plot|feature_importance")
10b. **ALWAYS AFTER MODEL TRAINING**: generate_ydata_profiling_report(encoded, output_path="./outputs/reports/ydata_profile.html") - Comprehensive data analysis report
11. **HYPERPARAMETER TUNING (⚠️ ONLY WHEN EXPLICITLY REQUESTED)**:
    - ⚠️ **CRITICAL WARNING**: This is EXTREMELY expensive (5-10 minutes) and resource-intensive!
    - ⚠️ **DO NOT USE UNLESS USER EXPLICITLY ASKS FOR IT**
    - **ONLY use when user says**: "tune", "optimize", "hyperparameter", "improve model", "best parameters"
    - **NEVER auto-trigger** based on scores - user must explicitly request it
    - **How**: hyperparameter_tuning(file_path=encoded, target_col=target_col, model_type="xgboost", n_trials=50)
    - **Large datasets (>100K rows)**: n_trials automatically reduced to 20 to prevent timeout
    - **Only tune the WINNING model** (don't waste time on others)
    - **Map model names**: XGBoost→"xgboost", Ridge→"ridge", Lasso→use Ridge
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
- ✅ "Generate interactive plots for price and quantity"
  → generate_interactive_scatter(x_col="price", y_col="quantity") → STOP
- ✅ "Create a dashboard showing correlations"
  → generate_plotly_dashboard(file_path) → STOP
- ✅ "Visualize the distribution of revenue"
  → generate_interactive_histogram(column="revenue") → STOP
- ✅ "Show me graphs of sales over time"
  → generate_interactive_time_series() → STOP

**Type C (Full ML) - RUN COMPLETE WORKFLOW:**
- ✅ "Train a model to predict house prices"
  → Full pipeline (steps 1-15)
- ✅ "Build a classifier for customer churn"
  → Full pipeline (steps 1-15)
- ✅ "Analyze data and train model to forecast revenue"
  → Full pipeline (steps 1-15)

**Type D (Unclear) - ASK USER:**
- ❓ "Analyze this dataset"
  → ASK: "Would you like me to (1) Create visualizations, (2) Train a predictive model, or (3) Both?"
- ❓ "Look at this CSV file"
  → ASK: "What would you like me to do? Visualize data or build a model?"
- ❓ "Check out my data"
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
- ❌ Model training → USE train_with_autogluon() (preferred) or train_baseline_models()
- ❌ Model optimization → USE optimize_autogluon_model() (refit, distill, deploy)
- ❌ Time series forecasting → USE forecast_with_autogluon() (supports covariates, holidays)
- ❌ Time series backtesting → USE backtest_timeseries()
- ❌ Multi-label prediction → USE train_multilabel_autogluon()
- ❌ Tasks with dedicated tools → USE THE TOOL, NOT custom code!

**Rule of Thumb:**
- CODE-ONLY task? → execute_python_code ONCE → STOP
- Data analysis task? → Use specialized tools, execute_python_code only for custom needs
- If a specialized tool exists → USE THE TOOL, not custom code

**KEY TOOLS (77 total available via function calling):**
- force_numeric_conversion: Converts string columns to numeric (auto-detects, skips text)
- clean_missing_values: "auto" mode supported
- encode_categorical: one-hot/target/frequency encoding
- **⭐ train_with_autogluon**: AutoML - trains 10+ models with auto ensembling (PREFERRED)
- forecast_with_autogluon: Time series forecasting with AutoGluon (supports covariates, holidays, model selection)
- optimize_autogluon_model: Post-training optimization (refit_full, distill, calibrate_threshold, deploy_optimize, delete_models)
- analyze_autogluon_model: Model inspection (summary, transform_features, info)
- extend_autogluon_training: Add models incrementally (fit_extra, fit_weighted_ensemble)
- train_multilabel_autogluon: Multi-label prediction (multiple target columns)
- backtest_timeseries: Time series backtesting with multiple validation windows
- analyze_timeseries_model: TS model analysis (feature_importance, plot, make_future_dataframe)
- train_baseline_models: Fallback - trains 4 basic models
- **⭐ execute_python_code**: Write and run custom Python code for ANY task not covered by tools (TRUE AI AGENT capability)
- **execute_code_from_file**: Run existing Python scripts
- Advanced: hyperparameter_tuning, perform_eda_analysis, handle_imbalanced_data, perform_feature_scaling, detect_anomalies, detect_and_handle_multicollinearity, auto_feature_engineering, forecast_time_series, explain_predictions, generate_business_insights, perform_topic_modeling, extract_image_features, monitor_model_drift
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
   - Use the EXACT suggested column name from the error
   - Column names may be abbreviated or different from user input
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
❌ Tool fails: train_baseline_models(file_path="data.csv", target_col="target_column")
📋 Error: "Column 'target_column' not found. 💡 HINT: Did you mean 'target_col'?"
✅ Next call MUST be: train_baseline_models(file_path="data.csv", target_col="target_col")
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

**FINAL SUMMARY - WHEN WORKFLOW IS COMPLETE:**
When you've finished all tool executions and are ready to return the final response, provide a comprehensive summary that includes:

1. **What was accomplished**: List all major steps completed (data cleaning, feature engineering, model training, etc.)
2. **Key findings from the data**:
   - ONLY cite statistics and numbers that appeared in ACTUAL tool results — do NOT fabricate thresholds, anomalies, or percentages
   - If no data quality issues were reported by tools, state "No significant data quality issues detected"
   - BUT DO provide DEEP interpretation of actual values: explain what real column ranges, correlations, and distributions MEAN for the user's domain
   - Derive insights from actual data: compare feature distributions, explain what strong/weak correlations imply practically, identify which features vary most and why that matters
   - What correlations were found? (report EXACT values from tool results AND explain their practical significance)
   - What were the most important features? (based on actual scores, with domain interpretation)
3. **Model performance** (if trained) - **CRITICAL: YOU MUST INCLUDE THESE METRICS**:
   - **ALWAYS extract and display** the exact metrics from tool results:
   - R² Score, RMSE, MAE from the train_with_autogluon or train_baseline_models results
   - List ALL models trained (not just the best one)
   - Example: "Trained 6 models: XGBoost (R²=0.713, RMSE=0.207), Random Forest (R²=0.685, RMSE=0.218), etc."
   - If hyperparameter tuning was done, show before/after comparison
   - How accurate is the model? What does the score mean in practical terms?
   - Were there any challenges (imbalanced data, multicollinearity, etc.)?
4. **Recommendations** (grounded in data — recommend based on what the tools found, not hypothetical scenarios):
   - Is the model ready for use?
   - What could improve performance further?
   - Align recommendations with the user's stated goal (e.g., if the user said "energy optimization", recommend optimization-relevant next steps, NOT generic survival analysis)
5. **Generated artifacts**: Mention reports, plots, and visualizations (but DON'T include file paths - the UI shows buttons automatically)

Example final response:
"I've completed the full machine learning workflow for [TARGET] prediction:

**Data Preparation:**
- Cleaned [N] records from the dataset
- Removed [N] columns with >50% missing values
- Extracted time-based features (`year`, `month`, `day`, `hour`) from datetime columns
- Encoded categorical variables using appropriate methods

**Key Findings:**
- [Feature A] shows strong correlation with the target variable
- Identified [N] distinct patterns/clusters in the data
- Most records fall within [specific range or category]

**Model Performance:**
- Best model: [Model Name]
- R² Score: [X.XX] (explains [X]% of target variance) OR Accuracy: [X]% for classification
- RMSE/MAE: [X.XX] (prediction error margin)
- Cross-validation: [X.XX] ± [X.XX] (consistent performance across folds)

After hyperparameter tuning, improved [metric] from [X] to [Y].

**Recommendation:**
The model shows [good/moderate] predictive power. Consider:
- Adding more relevant features if available
- Trying ensemble methods to boost performance
- Collecting more data for underrepresented categories

All visualizations, reports, and the trained model are available via the buttons above."

You are a DOER. Complete workflows based on user intent."""
    
    def _initialize_specialist_agents(self) -> Dict[str, Dict]:
        """Initialize specialist agent configurations with focused system prompts."""
        return {
            "eda_agent": {
                "name": "EDA Specialist",
                "emoji": "🔬",
                "description": "Explore and understand data patterns, relationships, correlations, and distributions. Answer questions about how variables relate, change together, or affect each other. Analyze data quality, detect outliers and anomalies. Generate descriptive statistics, correlation matrices, scatter plots, histograms, box plots, and distribution visualizations to reveal insights.",
                "system_prompt": """You are the EDA Specialist Agent - an expert in exploratory data analysis.

**Your Expertise:**
- Data profiling and statistical summaries
- Data quality assessment and anomaly detection
- Correlation analysis and feature relationships
- Distribution analysis and outlier detection
- Missing data patterns and strategies

**Your Tools (13 EDA-focused):**
- profile_dataset, detect_data_quality_issues, analyze_correlations
- get_smart_summary, detect_anomalies, perform_statistical_tests
- perform_eda_analysis, generate_ydata_profiling_report
- profile_bigquery_table, query_bigquery

**Your Approach:**
1. Always start with comprehensive data profiling
2. Identify quality issues before recommending fixes
3. Generate visualizations to reveal patterns
4. Provide actionable insights about data characteristics
5. Recommend next steps for data preparation

You work collaboratively with other specialists and hand off cleaned data to preprocessing and modeling agents.""",
                "tool_keywords": ["profile", "eda", "quality", "correlat", "anomal", "statistic", "distribution", "explore", "understand", "detect", "outlier"]
            },
            
            "modeling_agent": {
                "name": "ML Modeling Specialist",
                "emoji": "🤖",
                "description": "Build and train predictive machine learning models to forecast outcomes, classify categories, predict future values, or forecast time series. Perform supervised learning tasks including regression, classification, and time series forecasting. Train models using AutoGluon AutoML (preferred) or baseline models, optimize hyperparameters, conduct cross-validation, and evaluate model performance.",
                "system_prompt": """You are the ML Modeling Specialist Agent - an expert in machine learning powered by AutoGluon AutoML.

**Your Expertise:**
- AutoML with AutoGluon (preferred for best results)
- Model selection and baseline training
- Hyperparameter tuning and optimization
- Ensemble methods and model stacking
- Time series forecasting
- Cross-validation strategies
- Model evaluation and performance metrics

**CRITICAL: Target Column Validation**
BEFORE calling any training tools, you MUST:
1. Use profile_dataset to see actual column names
2. Verify the target column exists in the dataset
3. NEVER hallucinate or guess column names
4. If target column was provided or inferred, proceed with modeling
5. Only if NO target is available: analyze correlations to find best candidate

**Your Tools (8 modeling-focused):**
- train_with_autogluon (PREFERRED - AutoML with 10+ models, auto ensembling, handles raw data)
- predict_with_autogluon (predictions with trained AutoGluon model)
- forecast_with_autogluon (time series forecasting with AutoGluon - better than Prophet/ARIMA)
- train_baseline_models (fallback - trains 4 basic models)
- hyperparameter_tuning, perform_cross_validation
- generate_model_report, detect_model_issues

**TOOL PRIORITY (use in this order):**
| Task | Use This Tool | NOT This |
|------|--------------|----------|
| Classification/Regression | train_with_autogluon | train_baseline_models |
| Time Series Forecasting | forecast_with_autogluon | forecast_time_series |
| Predictions on new data | predict_with_autogluon | execute_python_code |
| Quick baseline check | train_baseline_models | execute_python_code |

**AutoGluon Advantages (explain to user):**
- Trains 10+ models automatically (vs 4 in baseline)
- Auto ensembles with multi-layer stacking
- Handles categorical features directly (no manual encoding needed)
- Handles missing values automatically (no manual imputation needed)
- Time-bounded training (won't run forever)
- Better accuracy than manual model selection

**Your Approach:**
1. FIRST: Profile the dataset to see actual columns (if not done)
2. VALIDATE: Confirm target column exists
3. PREFERRED: Use train_with_autogluon for best results
4. For time series data: Use forecast_with_autogluon
5. Validate with proper cross-validation if needed
6. Generate comprehensive model reports with metrics
7. Detect and address model issues (overfitting, bias, etc.)

**Common Errors to Avoid:**
❌ Calling train tools with non-existent target column
❌ Guessing column names like "Occupation", "Target", "Label"
❌ Using execute_python_code when dedicated tools exist
❌ Using train_baseline_models when train_with_autogluon is available
✅ Always verify column names from profile_dataset first
✅ Use train_with_autogluon as the DEFAULT training tool

You receive preprocessed data from data engineering agents and collaborate with visualization agents for model performance plots.""",
                "tool_keywords": ["train", "model", "hyperparameter", "ensemble", "cross-validation", "predict", "classify", "regress", "autogluon", "automl", "forecast"]
            },
            
            "viz_agent": {
                "name": "Visualization Specialist",
                "emoji": "📊",
                "description": "Create visual representations, charts, graphs, and dashboards to display data patterns. Generate interactive plots including scatter plots, line charts, bar graphs, heatmaps, time series visualizations, and statistical plots. Design comprehensive dashboards and visual reports to communicate findings clearly.",
                "system_prompt": """You are the Visualization Specialist Agent - an expert in data visualization.

**Your Expertise:**
- Interactive Plotly visualizations
- Statistical matplotlib plots
- Business intelligence dashboards
- Model performance visualizations
- Time series and geospatial plots

**Your Tools (8 visualization-focused):**
- create_plotly_scatter, create_plotly_heatmap, create_plotly_line
- create_matplotlib_plots, create_combined_plots
- generate_data_quality_plots, create_shap_plots
- generate_ydata_profiling_report (visual report)

**Your Approach:**
1. Choose the right visualization type for the data
2. Create interactive plots when possible (Plotly)
3. Use appropriate color schemes and layouts
4. Generate comprehensive visual reports
5. Highlight key insights through visual storytelling

You collaborate with all agents to visualize their outputs - EDA results, model performance, feature importance, etc.""",
                "tool_keywords": ["plot", "visualiz", "chart", "graph", "heatmap", "scatter", "dashboard", "matplotlib", "plotly", "create", "generate", "show", "display"]
            },
            
            "insight_agent": {
                "name": "Business Insights Specialist",
                "emoji": "💡",
                "description": "Interpret trained machine learning model results and translate findings into actionable business recommendations. Explain why models make certain predictions, analyze feature importance from completed models, identify root causes in model outputs, generate what-if scenarios, and provide strategic business insights based on model performance and predictions.",
                "system_prompt": """You are the Business Insights Specialist Agent - an expert in translating data into action.

**Your Expertise:**
- Root cause analysis and causal inference
- What-if scenario analysis
- Feature contribution interpretation
- Business intelligence and cohort analysis
- Actionable recommendations from ML results

**Your Tools (10 insight-focused):**
- analyze_root_cause, detect_causal_relationships
- generate_business_insights, explain_predictions
- perform_cohort_analysis, perform_rfm_analysis
- perform_customer_segmentation, analyze_customer_churn
- detect_model_issues (interpret issues)

**Your Approach:**
1. Translate statistical findings into business language
2. Identify root causes of patterns in data
3. Run what-if scenarios for decision support
4. Generate specific, actionable recommendations
5. Explain model predictions in human terms

You synthesize outputs from all other agents and provide the final business narrative.""",
                "tool_keywords": ["insight", "recommend", "explain", "interpret", "why", "cause", "what-if", "business", "segment", "churn"]
            },
            
            "preprocessing_agent": {
                "name": "Data Engineering Specialist",
                "emoji": "⚙️",
                "description": "Clean and prepare raw data for analysis by handling missing values, removing or treating outliers, encoding categorical variables, scaling numerical features, and engineering new features. Transform messy data into analysis-ready datasets through imputation, normalization, one-hot encoding, and feature creation.",
                "system_prompt": """You are the Data Engineering Specialist Agent - an expert in data preparation.

**Your Expertise:**
- Missing value handling and outlier treatment
- Feature scaling and normalization
- Imbalanced data handling (SMOTE, etc.)
- Feature engineering and transformation
- Data type conversion and encoding

**Your Tools (15 preprocessing-focused):**
- clean_missing_values, handle_outliers, handle_imbalanced_data
- perform_feature_scaling, encode_categorical
- create_interaction_features, create_aggregation_features
- auto_feature_engineering, create_time_features
- force_numeric_conversion, smart_type_inference
- merge_datasets, concat_datasets, reshape_dataset

**Your Approach:**
1. Fix data quality issues identified by EDA agent
2. Handle missing values with appropriate strategies
3. Treat outliers based on domain context
4. Engineer features to boost model performance
5. Prepare clean, model-ready data

You receive quality reports from EDA agent and deliver clean data to modeling agent.""",
                "tool_keywords": ["clean", "preprocess", "feature", "encod", "scal", "outlier", "missing", "transform", "engineer"]
            }
        }
    
    def _select_specialist_agent(self, task_description: str) -> str:
        """
        Route task to appropriate specialist agent.
        
        Uses SBERT semantic similarity if available, falls back to keyword matching.
        """
        # Try semantic routing first (more accurate)
        if self.semantic_layer.enabled:
            try:
                # Build agent descriptions for semantic matching
                agent_descriptions = {
                    agent_key: f"{agent_config['name']}: {agent_config['description']}"
                    for agent_key, agent_config in self.specialist_agents.items()
                }
                
                best_agent, confidence = self.semantic_layer.route_to_agent(
                    task_description, 
                    agent_descriptions
                )
                
                agent_config = self.specialist_agents[best_agent]
                print(f"🧠 Semantic routing → {agent_config['emoji']} {agent_config['name']} (confidence: {confidence:.2f})")
                
                return best_agent
                
            except Exception as e:
                print(f"⚠️ Semantic routing failed: {e}, falling back to keyword matching")
        
        # Fallback: Keyword-based routing (original method)
        task_lower = task_description.lower()
        
        # Score each agent based on keyword matches
        scores = {}
        for agent_key, agent_config in self.specialist_agents.items():
            score = sum(1 for keyword in agent_config["tool_keywords"] if keyword in task_lower)
            scores[agent_key] = score
        
        # Get agent with highest score
        if max(scores.values()) > 0:
            best_agent = max(scores.items(), key=lambda x: x[1])[0]
            agent_config = self.specialist_agents[best_agent]
            print(f"🔑 Keyword routing → {agent_config['emoji']} {agent_config['name']} ({scores[best_agent]} matches)")
            return best_agent
        
        # Default to EDA agent for exploratory tasks
        print("📊 Default routing → 🔬 EDA Specialist")
        return "eda_agent"
    
    def _get_agent_system_prompt(self, agent_key: str) -> str:
        """Get system prompt for specialist agent, fallback to main prompt."""
        if agent_key in self.specialist_agents:
            return self.specialist_agents[agent_key]["system_prompt"]
        return self._build_system_prompt()  # Fallback to main orchestrator prompt
    
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
        """Find the last successfully created DATA file from workflow history.
        
        Only returns actual data files (CSV, parquet, etc.), NOT visualization
        artifacts (HTML, PNG, etc.) which would break downstream tools.
        """
        data_extensions = ('.csv', '.parquet', '.xlsx', '.xls', '.json', '.tsv')
        
        # Check in reverse order for file-creating tools
        for step in reversed(workflow_history):
            result = step.get("result", {})
            if result.get("success"):
                # Check for output_path in result
                if "output_path" in result:
                    if result["output_path"].lower().endswith(data_extensions):
                        return result["output_path"]
                # For nested results
                if "result" in result and isinstance(result["result"], dict):
                    nested = result["result"]
                    if "output_path" in nested:
                        if nested["output_path"].lower().endswith(data_extensions):
                            return nested["output_path"]
                    # Check output_dir for dashboard-type tools
                    if "output_dir" in nested:
                        return nested["output_dir"]
                    # Check generated_files from execute_python_code
                    if "generated_files" in nested and nested["generated_files"]:
                        for gen_file in nested["generated_files"]:
                            if gen_file.lower().endswith(data_extensions):
                                return gen_file
                # Check tool arguments for file_path as last resort
                args = step.get("arguments", step.get("result", {}).get("arguments", {}))
                if isinstance(args, dict) and "file_path" in args:
                    import os
                    if os.path.exists(args["file_path"]):
                        return args["file_path"]
        
        # 🔥 FIX: Return the original input file instead of a phantom path
        # Try to get from session or workflow state
        if hasattr(self, 'session') and self.session and self.session.last_dataset:
            return self.session.last_dataset
        if hasattr(self, 'workflow_state') and self.workflow_state.current_file:
            return self.workflow_state.current_file
        
        # Last resort: return empty string instead of phantom file
        return "(no file found - use the original uploaded dataset)"
    
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
    
    @staticmethod
    def _is_safe_path(path: Path, allowed_root: Path) -> bool:
        """Check if path is within an allowed root directory."""
        try:
            path.resolve().relative_to(allowed_root)
            return True
        except ValueError:
            return False
    
    # 🚀 PARALLEL EXECUTION: Helper methods for concurrent tool execution
    def _execute_tool_sync(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for _execute_tool to be used in async context.
        This allows the parallel executor to run tools concurrently.
        """
        return self._execute_tool(tool_name, tool_args)
    
    async def _async_progress_callback(self, tool_name: str, status: str):
        """
        Async progress callback for parallel execution.
        Emits SSE events for real-time progress tracking.
        """
        if hasattr(self, 'session') and self.session:
            session_id = self.session.session_id
            if status == "started":
                print(f"🚀 [Parallel] Started: {tool_name}")
                from .api.app import progress_manager
                progress_manager.emit(session_id, {
                    'type': 'tool_executing',
                    'tool': tool_name,
                    'message': f"🚀 [Parallel] Executing: {tool_name}",
                    'parallel': True
                })
            elif status == "completed":
                print(f"✓ [Parallel] Completed: {tool_name}")
                from .api.app import progress_manager
                progress_manager.emit(session_id, {
                    'type': 'tool_completed',
                    'tool': tool_name,
                    'message': f"✓ [Parallel] Completed: {tool_name}",
                    'parallel': True
                })
            elif status.startswith("error"):
                print(f"❌ [Parallel] Failed: {tool_name}")
    
    # 🤝 INTER-AGENT COMMUNICATION: Methods for agent hand-offs
    def _should_hand_off(self, current_agent: str, completed_tools: List[str], 
                        workflow_history: List[Dict]) -> Optional[str]:
        """
        Determine if workflow should hand off to a different specialist agent.
        
        Args:
            current_agent: Currently active agent
            completed_tools: List of tool names executed so far
            workflow_history: Full workflow history
            
        Returns:
            Name of agent to hand off to, or None to stay with current agent
        """
        # Suggest next agent based on completed work
        suggested_agent = suggest_next_agent(current_agent, completed_tools)
        
        # Hand off if different from current agent
        if suggested_agent and suggested_agent != current_agent:
            return suggested_agent
        
        return None
    
    def _hand_off_to_agent(self, target_agent: str, context: Dict[str, Any], 
                          iteration: int) -> Dict[str, Any]:
        """
        Hand off workflow to a different specialist agent.
        
        Args:
            target_agent: Agent to hand off to
            context: Shared context (dataset info, completed steps, etc.)
            iteration: Current iteration number
            
        Returns:
            Dictionary with hand-off details
        """
        if target_agent not in self.specialist_agents:
            # Silently skip invalid hand-off targets (common during workflow transitions)
            return {"success": False, "error": "Invalid target agent"}
        
        # Update active agent
        old_agent = self.active_agent
        self.active_agent = target_agent
        
        agent_config = self.specialist_agents[target_agent]
        
        print(f"\n🔄 AGENT HAND-OFF (iteration {iteration})")
        print(f"   From: {old_agent}")
        print(f"   To: {target_agent} {agent_config['emoji']}")
        print(f"   Reason: {context.get('reason', 'Workflow progression')}")
        
        # Reload tools for new agent
        new_tools = self._compress_tools_registry(agent_name=target_agent)
        print(f"   📦 Reloaded {len(new_tools)} tools for {target_agent}")
        
        # Emit hand-off event
        if self.progress_callback:
            self.progress_callback({
                "type": "agent_handoff",
                "from_agent": old_agent,
                "to_agent": target_agent,
                "agent_name": agent_config['name'],
                "emoji": agent_config['emoji'],
                "reason": context.get('reason', 'Workflow progression'),
                "tools_count": len(new_tools)
            })
        
        return {
            "success": True,
            "old_agent": old_agent,
            "new_agent": target_agent,
            "new_tools": new_tools,
            "system_prompt": agent_config["system_prompt"]
        }
    
    def _get_agent_chain_suggestions(self, task_description: str, 
                                     current_agent: str) -> List[str]:
        """
        Get suggested agent chain for complex workflows.
        
        Args:
            task_description: User's task description
            current_agent: Currently active agent
            
        Returns:
            List of agent names in suggested execution order
        """
        task_lower = task_description.lower()
        
        # Detect workflow type from task description
        if "full" in task_lower or "complete" in task_lower or "end-to-end" in task_lower:
            # Full ML pipeline
            return [
                "data_quality_agent",
                "preprocessing_agent",
                "visualization_agent",
                "modeling_agent",
                "production_agent"
            ]
        elif "train" in task_lower or "model" in task_lower:
            # ML-focused workflow
            return [
                "data_quality_agent",
                "preprocessing_agent",
                "modeling_agent"
            ]
        elif "visualiz" in task_lower or "plot" in task_lower or "chart" in task_lower:
            # Visualization-focused
            return [
                "data_quality_agent",
                "visualization_agent"
            ]
        elif "clean" in task_lower or "preprocess" in task_lower:
            # Data cleaning focused
            return [
                "data_quality_agent",
                "preprocessing_agent"
            ]
        else:
            # Default single agent
            return [current_agent]
    
    def _generate_enhanced_summary(
        self, 
        workflow_history: List[Dict], 
        llm_summary: str,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Generate an enhanced summary with extracted metrics, plots, and artifacts.
        
        Args:
            workflow_history: List of executed workflow steps
            llm_summary: Original summary from LLM
            task_description: User's original request
            
        Returns:
            Dictionary with enhanced summary text, metrics, and artifacts
        """
        metrics = {}
        artifacts = {
            "models": [],
            "reports": [],
            "data_files": []
        }
        plots = []
        
        # Extract information from workflow history
        for step in workflow_history:
            tool = step.get("tool", "")
            result = step.get("result", {})
            
            # Skip failed steps
            if not result.get("success", True):
                continue
            
            # Extract nested result if present
            # Tool results can be structured as:
            # 1. Direct: {"output_path": "...", "status": "success"}
            # 2. Nested: {"success": True, "result": {"output_path": "..."}}
            nested_result = result.get("result", result)
            
            # DEBUG: Log structure for visualization tools
            if "plot" in tool.lower() or "heatmap" in tool.lower() or "visualiz" in tool.lower():
                print(f"[DEBUG] Extracting plot from tool: {tool}")
                print(f"[DEBUG]   result keys: {list(result.keys())}")
                print(f"[DEBUG]   nested_result keys: {list(nested_result.keys()) if isinstance(nested_result, dict) else 'not a dict'}")
                print(f"[DEBUG]   output_path in nested_result: {'output_path' in nested_result if isinstance(nested_result, dict) else False}")
                if isinstance(nested_result, dict) and "output_path" in nested_result:
                    print(f"[DEBUG]   output_path value: {nested_result['output_path']}")
            
            # === EXTRACT MODEL METRICS ===
            if tool == "train_baseline_models":
                if "models" in nested_result:
                    models_data = nested_result["models"]
                    if models_data:
                        # Find best model (best_model is a dict with 'name', 'score', 'model_path')
                        best_model_info = nested_result.get("best_model", {})
                        if isinstance(best_model_info, dict):
                            best_model_name = best_model_info.get("name", "")
                        else:
                            best_model_name = str(best_model_info) if best_model_info else ""
                        
                        best_model_data = models_data.get(best_model_name, {})
                        # Metrics are nested inside test_metrics
                        test_metrics = best_model_data.get("test_metrics", {})
                        
                        metrics["best_model"] = {
                            "name": best_model_name,
                            "r2_score": test_metrics.get("r2", 0),
                            "rmse": test_metrics.get("rmse", 0),
                            "mae": test_metrics.get("mae", 0)
                        }
                        
                        # All models comparison - extract test_metrics for each
                        metrics["all_models"] = {}
                        for name, data in models_data.items():
                            if isinstance(data, dict) and "test_metrics" in data:
                                metrics["all_models"][name] = {
                                    "r2": data["test_metrics"].get("r2", 0),
                                    "rmse": data["test_metrics"].get("rmse", 0),
                                    "mae": data["test_metrics"].get("mae", 0)
                                }
                
                # Extract model artifacts
                if "model_path" in nested_result:
                    artifacts["models"].append({
                        "name": nested_result.get("best_model", "model"),
                        "path": nested_result["model_path"],
                        "url": f"/outputs/models/{nested_result['model_path'].split('/')[-1]}"
                    })
                
                # Extract performance plots
                if "performance_plots" in nested_result:
                    for plot_path in nested_result["performance_plots"]:
                        plots.append({
                            "title": plot_path.split("/")[-1].replace("_", " ").replace(".png", "").title(),
                            "path": plot_path,
                            "url": f"/outputs/{plot_path.replace('./outputs/', '')}"
                        })
                
                if "feature_importance_plot" in nested_result:
                    plot_path = nested_result["feature_importance_plot"]
                    plots.append({
                        "title": "Feature Importance",
                        "path": plot_path,
                        "url": f"/outputs/{plot_path.replace('./outputs/', '')}"
                    })
            
            # === HYPERPARAMETER TUNING METRICS ===
            elif tool == "hyperparameter_tuning":
                if "best_score" in nested_result:
                    metrics["tuned_model"] = {
                        "best_score": nested_result["best_score"],
                        "best_params": nested_result.get("best_params", {}),
                        "model_type": nested_result.get("model_type", "unknown")
                    }
                
                if "model_path" in nested_result:
                    artifacts["models"].append({
                        "name": f"{nested_result.get('model_type', 'model')}_tuned",
                        "path": nested_result["model_path"],
                        "url": f"/outputs/models/{nested_result['model_path'].split('/')[-1]}"
                    })
            
            # === CROSS-VALIDATION METRICS ===
            elif tool == "perform_cross_validation":
                if "mean_score" in nested_result:
                    metrics["cross_validation"] = {
                        "mean_score": nested_result["mean_score"],
                        "std_score": nested_result.get("std_score", 0),
                        "scores": nested_result.get("scores", [])
                    }
            
            # === COLLECT REPORT FILES ===
            elif "report" in tool.lower() or "dashboard" in tool.lower():
                print(f"[DEBUG] Report tool detected: {tool}")
                print(f"[DEBUG] nested_result keys: {list(nested_result.keys())}")
                # Check for both 'output_path' and 'report_path' keys
                report_path = nested_result.get("output_path") or nested_result.get("report_path")
                if report_path:
                    print(f"[DEBUG] Report path found: {report_path}")
                    # Clean path for URL — handle both ./outputs and /tmp paths
                    if report_path.startswith('./outputs/'):
                        url_path = report_path.replace('./outputs/', '')
                    elif report_path.startswith('/tmp/data_science_agent/outputs/'):
                        url_path = report_path.replace('/tmp/data_science_agent/outputs/', '')
                    elif report_path.startswith('/tmp/data_science_agent/'):
                        url_path = report_path.replace('/tmp/data_science_agent/', '')
                    else:
                        url_path = report_path.split('/')[-1]
                    artifacts["reports"].append({
                        "name": tool.replace("_", " ").title(),
                        "path": report_path,
                        "url": f"/outputs/{url_path}"
                    })
                    print(f"[DEBUG] Added to artifacts[reports], total reports: {len(artifacts['reports'])}")
                
                # 🔥 FIX: Extract individual plots from dashboard's 'plots' array
                # generate_plotly_dashboard returns {"plots": [{"output_path": ..., "status": "success"}, ...]}
                if "plots" in nested_result and isinstance(nested_result["plots"], list):
                    dashboard_output_dir = nested_result.get("output_dir", "./outputs/plots/interactive")
                    for sub_plot in nested_result["plots"]:
                        if isinstance(sub_plot, dict) and sub_plot.get("status") == "success":
                            sub_path = sub_plot.get("output_path", "")
                            if sub_path:
                                # Clean path for URL
                                if sub_path.startswith('./outputs/'):
                                    url_path = sub_path.replace('./outputs/', '')
                                elif sub_path.startswith('/tmp/data_science_agent/'):
                                    url_path = sub_path.replace('/tmp/data_science_agent/', '')
                                else:
                                    url_path = sub_path.split('/')[-1]
                                
                                plot_title = sub_path.split('/')[-1].replace('_', ' ').replace('.html', '').replace('.png', '').title()
                                plots.append({
                                    "title": plot_title,
                                    "path": sub_path,
                                    "url": f"/outputs/{url_path}",
                                    "type": "html" if sub_path.endswith(".html") else "image"
                                })
                                print(f"[DEBUG] Added dashboard sub-plot: {plot_title} -> /outputs/{url_path}")
                    
                    print(f"[DEBUG] Extracted {len(nested_result['plots'])} plots from dashboard")
                elif not report_path:
                    print(f"[DEBUG] No output_path, report_path, or plots array in nested_result for report tool")
            
            # === COLLECT VISUALIZATION FILES (interactive plots, charts, etc.) ===
            elif "plot" in tool.lower() or "visualiz" in tool.lower() or "chart" in tool.lower() or "heatmap" in tool.lower() or "scatter" in tool.lower() or "histogram" in tool.lower():
                if "output_path" in nested_result:
                    plot_path = nested_result["output_path"]
                    # Extract plot title from tool name or filename
                    plot_title = tool.replace("generate_", "").replace("interactive_", "").replace("_", " ").title()
                    if not plot_title or plot_title == "Output Path":
                        plot_title = plot_path.split("/")[-1].replace("_", " ").replace(".html", "").replace(".png", "").title()
                    
                    # Clean path for URL - handle both ./outputs and /tmp paths
                    if plot_path.startswith('./outputs/'):
                        url_path = plot_path.replace('./outputs/', '')
                    elif plot_path.startswith('/tmp/data_science_agent/outputs/'):
                        url_path = plot_path.replace('/tmp/data_science_agent/outputs/', '')
                    elif plot_path.startswith('/tmp/data_science_agent/'):
                        url_path = plot_path.replace('/tmp/data_science_agent/', '')
                    else:
                        # Just use filename for other paths
                        url_path = plot_path.split('/')[-1]
                    
                    plots.append({
                        "title": plot_title,
                        "path": plot_path,
                        "url": f"/outputs/{url_path}",
                        "type": "html" if plot_path.endswith(".html") else "image"
                    })
                    print(f"[DEBUG] Added plot to array:")
                    print(f"[DEBUG]   title: {plot_title}")
                    print(f"[DEBUG]   url: /outputs/{url_path}")
                    print(f"[DEBUG]   type: {'html' if plot_path.endswith('.html') else 'image'}")
            
            # === COLLECT PLOT FILES (from plot_paths key) ===
            if "plot_paths" in nested_result:
                for plot_path in nested_result["plot_paths"]:
                    # Clean path for URL
                    if plot_path.startswith('./outputs/'):
                        url_path = plot_path.replace('./outputs/', '')
                    elif plot_path.startswith('/tmp/data_science_agent/outputs/'):
                        url_path = plot_path.replace('/tmp/data_science_agent/outputs/', '')
                    elif plot_path.startswith('/tmp/data_science_agent/'):
                        url_path = plot_path.replace('/tmp/data_science_agent/', '')
                    else:
                        url_path = plot_path.split('/')[-1]
                    
                    plots.append({
                        "title": plot_path.split("/")[-1].replace("_", " ").replace(".png", "").replace(".html", "").title(),
                        "path": plot_path,
                        "url": f"/outputs/{url_path}",
                        "type": "html" if plot_path.endswith(".html") else "image"
                    })
            
            # === COLLECT DATA FILES ===
            if "output_path" in nested_result and nested_result["output_path"].endswith(".csv"):
                data_path = nested_result["output_path"]
                # Clean path for URL
                if data_path.startswith('./outputs/'):
                    url_path = data_path.replace('./outputs/', '')
                elif data_path.startswith('/tmp/data_science_agent/outputs/'):
                    url_path = data_path.replace('/tmp/data_science_agent/outputs/', '')
                elif data_path.startswith('/tmp/data_science_agent/'):
                    url_path = data_path.replace('/tmp/data_science_agent/', '')
                else:
                    url_path = data_path.split('/')[-1]
                
                artifacts["data_files"].append({
                    "name": data_path.split("/")[-1],
                    "path": data_path,
                    "url": f"/outputs/{url_path}"
                })
            
            # === SCAN execute_python_code OUTPUT FOR HTML FILES ===
            # When LLM uses execute_python_code to create visualizations, the HTML paths
            # are not in output_path - we need to scan the output/stdout for .html paths
            if tool == "execute_python_code":
                # Get raw output from code execution
                raw_output = str(nested_result.get("output", "")) + str(nested_result.get("stdout", "")) + str(result.get("output", ""))
                
                # Also scan the code itself for write_html() calls
                code_str = str(step.get("arguments", {}).get("code", ""))
                
                # Regex to find .html file paths in output or code
                html_paths = set()
                
                # Pattern 1: Paths in write_html() calls
                write_html_pattern = r"write_html\s*\(\s*['\"]([^'\"]+\.html)['\"]"
                html_paths.update(re.findall(write_html_pattern, code_str))
                
                # Pattern 2: Paths like /tmp/data_science_agent/*.html in output
                output_pattern = r"(/tmp/data_science_agent/[^\s'\"]+\.html)"
                html_paths.update(re.findall(output_pattern, raw_output))
                html_paths.update(re.findall(output_pattern, code_str))
                
                # Pattern 3: visualizations_created list in output (common pattern)
                viz_list_pattern = r"visualizations_created['\"]?\s*:\s*\[([^\]]+)\]"
                viz_match = re.search(viz_list_pattern, raw_output)
                if viz_match:
                    viz_paths = re.findall(r"['\"]([^'\"]+\.html)['\"]", viz_match.group(1))
                    html_paths.update(viz_paths)
                
                print(f"[DEBUG] execute_python_code artifact scanner found {len(html_paths)} HTML files: {html_paths}")
                
                # Register each found HTML as a plot
                for html_path in html_paths:
                    # Extract title from filename
                    filename = html_path.split("/")[-1]
                    plot_title = filename.replace("_", " ").replace(".html", "").title()
                    
                    # Clean path for URL
                    if html_path.startswith('/tmp/data_science_agent/'):
                        url_path = html_path.replace('/tmp/data_science_agent/', '')
                    else:
                        url_path = filename
                    
                    # Avoid duplicates
                    existing_urls = [p.get("url", "") for p in plots]
                    new_url = f"/outputs/{url_path}"
                    if new_url not in existing_urls:
                        plots.append({
                            "title": plot_title,
                            "path": html_path,
                            "url": new_url,
                            "type": "html"
                        })
                        print(f"[DEBUG] Registered plot from execute_python_code:")
                        print(f"[DEBUG]   title: {plot_title}")
                        print(f"[DEBUG]   url: {new_url}")
        
        # Build COMPREHENSIVE response template following user's format
        summary_lines = []
        
        # Start with the LLM's actual reasoning/summary
        if llm_summary and llm_summary.strip() and llm_summary != "Analysis completed":
            summary_lines.extend([
                llm_summary.strip(),
                "",
                "---",
                ""
            ])
        
        # Header
        summary_lines.extend([
            "## 📋 Workflow Summary:",
            ""
        ])
        
        # Extract task type and dataset info from workflow
        task_type = None
        n_features = 0
        n_samples = 0
        train_size = 0
        test_size = 0
        
        for step in workflow_history:
            if step.get("tool") == "train_baseline_models":
                result = step.get("result", {}).get("result", {})
                task_type = result.get("task_type", "regression")
                n_features = result.get("n_features", 0)
                train_size = result.get("train_size", 0)
                test_size = result.get("test_size", 0)
                n_samples = train_size + test_size
                break
        
        # SECTION 1: Dataset Profiling and Quality
        summary_lines.extend([
            "### 📊 Dataset Profiling and Quality:",
            ""
        ])
        
        if n_samples > 0:
            summary_lines.append(f"- The dataset contains **{n_samples:,} rows** and **{n_features} features**.")
        
        # Add workflow-specific insights
        profiling_done = any(s.get("tool") == "profile_dataset" for s in workflow_history)
        quality_checked = any(s.get("tool") == "detect_data_quality_issues" for s in workflow_history)
        
        if profiling_done:
            summary_lines.append("- Dataset profiling completed with comprehensive statistics.")
        if quality_checked:
            summary_lines.append("- Data quality issues were detected and analyzed.")
        
        summary_lines.extend(["", ""])
        
        # SECTION 2: Data Preprocessing
        summary_lines.extend([
            "### 🔧 Data Preprocessing:",
            ""
        ])
        
        preprocessing_steps = []
        for step in workflow_history:
            tool = step.get("tool", "")
            if tool == "clean_missing_values":
                preprocessing_steps.append("- Missing values were handled using automated strategies.")
            elif tool == "handle_outliers":
                preprocessing_steps.append("- Outliers were detected and handled appropriately.")
            elif tool == "encode_categorical":
                preprocessing_steps.append("- Categorical variables were encoded for ML compatibility.")
            elif tool == "feature_engineering" or tool == "enhanced_feature_engineering":
                preprocessing_steps.append("- Advanced feature engineering was performed to create predictive features.")
        
        if preprocessing_steps:
            summary_lines.extend(preprocessing_steps)
        else:
            summary_lines.append("- Data preprocessing steps were applied as needed.")
        
        summary_lines.extend(["", ""])
        
        # SECTION 3: Exploratory Data Analysis
        eda_done = any("eda" in s.get("tool", "").lower() or "plot" in s.get("tool", "").lower() 
                       for s in workflow_history)
        if eda_done:
            summary_lines.extend([
                "### 📈 Exploratory Data Analysis (EDA):",
                "",
                "- Comprehensive EDA visualizations were generated.",
                "- Correlation analysis, distribution plots, and feature relationships were examined.",
                f"- All visualizations are available in the **Visualization Gallery** below.",
                "",
                ""
            ])
        
        # SECTION 4: Model Training Results (ENHANCED - Following Template)
        if "all_models" in metrics and metrics["all_models"]:
            # Determine if classification or regression
            is_classification = task_type == "classification"
            metric_key = "f1" if is_classification else "r2"
            
            # Sort models by primary metric (descending)
            sorted_models = sorted(
                metrics["all_models"].items(),
                key=lambda x: x[1].get(metric_key, 0),
                reverse=True
            )
            
            best_model_name = sorted_models[0][0] if sorted_models else None
            best_model_score = sorted_models[0][1].get(metric_key, 0) if sorted_models else 0
            
            summary_lines.extend([
                "## 🎯 Model Training Results",
                "",
                f"**Task Type**: {task_type.title()}",
                f"**Features**: {n_features}",
                f"**Training Samples**: {train_size:,}",
                f"**Test Samples**: {test_size:,}",
                "",
                "### 📊 All Models Tested:",
                ""
            ])
            
            # Create detailed model performance table
            for model_name, model_metrics in sorted_models:
                is_best = (model_name == best_model_name)
                prefix = "🏆 " if is_best else "📊 "
                
                model_display_name = model_name.replace('_', ' ').title()
                
                if is_classification:
                    accuracy = model_metrics.get("accuracy", 0)
                    precision = model_metrics.get("precision", 0)
                    recall = model_metrics.get("recall", 0)
                    f1 = model_metrics.get("f1", 0)
                    
                    summary_lines.extend([
                        f"{prefix}**{model_display_name}**:",
                        "",
                        f"- Accuracy: **{accuracy:.4f}**",
                        f"- Precision: **{precision:.4f}**",
                        f"- Recall: **{recall:.4f}**",
                        f"- F1 Score: **{f1:.4f}**",
                        ""
                    ])
                else:  # regression
                    r2 = model_metrics.get("r2", 0)
                    rmse = model_metrics.get("rmse", 0)
                    mae = model_metrics.get("mae", 0)
                    
                    summary_lines.extend([
                        f"{prefix}**{model_display_name}**:",
                        "",
                        f"- R² Score: **{r2:.4f}**",
                        f"- RMSE: **{rmse:.4f}**",
                        f"- MAE: **{mae:.4f}**",
                        ""
                    ])
            
            # Best model highlight
            summary_lines.extend([
                f"### 🏆 Best Model: **{best_model_name.replace('_', ' ').title()}**",
                f"**Score**: {best_model_score:.4f}",
                "",
                ""
            ])
        
        # SECTION 5: Tuning Results (if hyperparameter tuning was done)
        if "tuned_model" in metrics:
            tuned = metrics["tuned_model"]
            summary_lines.extend([
                "### ⚙️ Hyperparameter Tuning:",
                "",
                f"- Model optimized: **{tuned.get('model_type', 'Unknown').replace('_', ' ').title()}**",
                f"- Best cross-validation score: **{tuned.get('best_score', 0):.4f}**",
                "- Hyperparameters were optimized using Bayesian optimization.",
                "",
                ""
            ])
        
        # SECTION 6: Cross-Validation (if performed)
        if "cross_validation" in metrics:
            cv = metrics["cross_validation"]
            summary_lines.extend([
                "### ✅ Cross-Validation:",
                "",
                f"- Mean Score: **{cv['mean_score']:.4f} ± {cv['std_score']:.4f}**",
                f"- Validated across multiple folds for robust performance estimation.",
                "",
                ""
            ])
        
        # SECTION 7: Workflow Steps Checklist
        summary_lines.extend([
            "## 🔧 Workflow Steps:",
            ""
        ])
        
        completed_steps = []
        for step in workflow_history:
            if step.get("result", {}).get("success", True):
                tool_name = step.get("tool", "")
                # Format tool name nicely
                display_name = tool_name.replace("_", " ").replace("generate ", "").title()
                completed_steps.append(f"✅ {display_name}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in completed_steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        summary_lines.extend(unique_steps)
        summary_lines.extend(["", ""])
        
        # SECTION 8: Generated Visualizations
        if plots:
            summary_lines.extend([
                f"## 📊 Generated Visualizations ({len(plots)} plots)",
                "",
                "✅ **Plots are displayed in the Visualization Gallery below!**",
                "",
                "Available visualizations include:",
                ""
            ])
            
            for plot in plots[:10]:  # Show up to 10 plots
                plot_title = plot.get('title', 'Visualization')
                summary_lines.append(f"- 📈 {plot_title}")
            
            if len(plots) > 10:
                summary_lines.append(f"- ... and {len(plots) - 10} more visualizations")
            
            summary_lines.extend(["", ""])
        
        # SECTION 9: Execution Summary
        total_time = sum(s.get("duration", 0) for s in workflow_history)
        summary_lines.extend([
            "## ⏱️ Execution Summary:",
            "",
            f"- **Tools Executed**: {len(completed_steps)}",
            f"- **Iterations**: {len(workflow_history)}",
            f"- **Time**: {total_time:.1f}s",
            ""
        ])
        
        # SECTION 10: Artifacts (if any)
        if artifacts["models"]:
            summary_lines.extend([
                "### 💾 Trained Models:",
                ""
            ])
            for model in artifacts["models"]:
                summary_lines.append(f"- {model['name']}")
            summary_lines.append("")
        
        if artifacts["reports"]:
            summary_lines.extend([
                "### 📄 Generated Reports:",
                ""
            ])
            for report in artifacts["reports"]:
                summary_lines.append(f"- {report['name']}")
            summary_lines.append("")
        
        # 🔥 MERGE REPORTS INTO PLOTS ARRAY FOR FRONTEND DISPLAY
        # Frontend expects everything viewable in result.plots array
        print(f"[DEBUG] Merging {len(artifacts['reports'])} reports into plots array")
        for report in artifacts["reports"]:
            plots.append({
                "title": report["name"],
                "url": report["url"],
                "type": "html"  # Reports are typically HTML
            })
            print(f"[DEBUG] Added report to plots array: title='{report['name']}', url='{report['url']}'")
        
        print(f"[DEBUG] Final plots array length: {len(plots)}")
        
        return {
            "text": "\n".join(summary_lines),
            "metrics": metrics,
            "artifacts": artifacts,
            "plots": plots
        }
    
    @retry_with_fallback(tool_name=None)  # 🛡️ ERROR RECOVERY: Auto-retry with fallback
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
        
        # Validate file_path arguments are within allowed directories
        ALLOWED_ROOTS = [
            Path("/tmp/data_science_agent").resolve(),
            Path("./outputs").resolve(),
            Path("./data").resolve(),
            Path("./cache_db").resolve(),
            Path("./checkpoints").resolve(),
        ]
        for key in ("file_path", "input_path", "train_data_path", "test_data_path"):
            if key in arguments and arguments[key]:
                try:
                    resolved = Path(arguments[key]).resolve()
                    if not any(self._is_safe_path(resolved, root) for root in ALLOWED_ROOTS):
                        return {
                            "success": False, 
                            "error": f"Path '{arguments[key]}' is outside allowed directories",
                            "error_type": "SecurityError"
                        }
                except (ValueError, OSError):
                    pass  # Let the tool handle invalid paths
        
        try:
            # Report progress before executing
            if self.progress_callback:
                self.progress_callback(tool_name, "running")
            
            tool_func = self.tool_functions[tool_name]
            
            # CRITICAL: Validate column names for modeling tools (prevent hallucinations)
            if tool_name in ["train_baseline_models", "hyperparameter_tuning", "train_ensemble_models"]:
                if "target_col" in arguments and arguments["target_col"]:
                    target_col = arguments["target_col"]
                    file_path = arguments.get("file_path", "")
                    
                    # Validate target column exists in dataset
                    try:
                        import polars as pl
                        df = pl.read_csv(file_path) if file_path.endswith('.csv') else pl.read_parquet(file_path)
                        actual_columns = df.columns
                        
                        if target_col not in actual_columns:
                            print(f"⚠️  HALLUCINATED TARGET COLUMN: '{target_col}'")
                            print(f"   Actual columns: {actual_columns}")
                            
                            # 🧠 Try semantic matching first (better than fuzzy)
                            corrected_col = None
                            if self.semantic_layer.enabled:
                                try:
                                    match = self.semantic_layer.semantic_column_match(target_col, actual_columns, threshold=0.6)
                                    if match:
                                        corrected_col, confidence = match
                                        print(f"   🧠 Semantic match: {corrected_col} (confidence: {confidence:.2f})")
                                except Exception as e:
                                    print(f"   ⚠️ Semantic matching failed: {e}")
                            
                            # Fallback to fuzzy matching if semantic didn't work
                            if not corrected_col:
                                close_matches = get_close_matches(target_col, actual_columns, n=1, cutoff=0.6)
                                if close_matches:
                                    corrected_col = close_matches[0]
                                    print(f"   ✓ Fuzzy match: {corrected_col}")
                            
                            if corrected_col:
                                arguments["target_col"] = corrected_col
                            else:
                                return {
                                    "success": False,
                                    "tool": tool_name,
                                    "arguments": arguments,
                                    "error": f"Target column '{target_col}' does not exist. Available columns: {actual_columns}",
                                    "error_type": "ColumnNotFoundError",
                                    "hint": "Please specify the correct target column name from the dataset."
                                }
                    except Exception as validation_error:
                        print(f"⚠️  Could not validate target column: {validation_error}")
            
            # Fix common parameter mismatches from LLM hallucinations
            if tool_name == "generate_ydata_profiling_report":
                # LLM often calls with 'output_dir' instead of 'output_path'
                if "output_dir" in arguments and "output_path" not in arguments:
                    output_dir = arguments.pop("output_dir")
                    # Convert directory to full file path
                    arguments["output_path"] = f"{output_dir}/ydata_profile.html"
            
            # Fix target_column → target_col (common LLM mistake)
            if "target_column" in arguments and "target_col" not in arguments:
                arguments["target_col"] = arguments.pop("target_column")
                print(f"   ✓ Parameter remapped: target_column → target_col")
            
            # Fix tool-specific parameter mismatches from LLM hallucinations
            if tool_name == "train_baseline_models":
                # LLM often adds 'models' parameter that doesn't exist
                if "models" in arguments:
                    models_val = arguments.pop("models")
                    print(f"   ✓ Stripped invalid parameter 'models': {models_val}")
                    print(f"   ℹ️ train_baseline_models trains all baseline models automatically")
                # LLM often adds 'feature_columns' parameter that doesn't exist
                if "feature_columns" in arguments:
                    feature_cols = arguments.pop("feature_columns")
                    print(f"   ✓ Stripped invalid parameter 'feature_columns': {feature_cols}")
                    print(f"   ℹ️ train_baseline_models uses all numeric columns automatically")
            
            if tool_name == "generate_model_report":
                # LLM uses 'file_path' instead of 'test_data_path'
                if "file_path" in arguments and "test_data_path" not in arguments:
                    arguments["test_data_path"] = arguments.pop("file_path")
                    print(f"   ✓ Parameter remapped: file_path → test_data_path")
            
            if tool_name == "detect_model_issues":
                # LLM adds invalid split parameters
                for invalid_param in ["train_target_path", "test_target_path"]:
                    if invalid_param in arguments:
                        val = arguments.pop(invalid_param)
                        print(f"   ✓ Stripped invalid parameter '{invalid_param}': {val}")
                # Ensure train_data_path is provided
                if "train_data_path" not in arguments:
                    print(f"   ⚠️ WARNING: detect_model_issues requires 'train_data_path' parameter")
            
            if tool_name == "create_statistical_features":
                # LLM confuses this with geospatial features and adds lat_col/lon_col
                for invalid_param in ["lat_col", "lon_col", "latitude", "longitude"]:
                    if invalid_param in arguments:
                        val = arguments.pop(invalid_param)
                        print(f"   ✓ Stripped invalid parameter '{invalid_param}': {val}")
                        print(f"   ℹ️ create_statistical_features creates row-wise stats (mean, std, min, max)")
            
            # 🔧 FIX: analyze_autogluon_model path resolution
            # The Reasoner hallucinates model paths — resolve to actual saved path
            if tool_name == "analyze_autogluon_model":
                model_path = arguments.get("model_path", "")
                if model_path and not Path(model_path).exists():
                    # Try the default AutoGluon output dir
                    fallback_paths = [
                        "./outputs/autogluon_model",
                        "outputs/autogluon_model",
                        "/tmp/data_science_agent/outputs/autogluon_model",
                    ]
                    for fallback in fallback_paths:
                        if Path(fallback).exists():
                            print(f"   ✓ Fixed model_path: '{model_path}' → '{fallback}'")
                            arguments["model_path"] = fallback
                            break
                    else:
                        print(f"   ⚠️ Model path '{model_path}' not found, no fallback available")
            
            # 🔧 FIX: predict_with_autogluon path resolution (same issue)
            if tool_name == "predict_with_autogluon":
                model_path = arguments.get("model_path", "")
                if model_path and not Path(model_path).exists():
                    fallback_paths = [
                        "./outputs/autogluon_model",
                        "outputs/autogluon_model",
                        "/tmp/data_science_agent/outputs/autogluon_model",
                    ]
                    for fallback in fallback_paths:
                        if Path(fallback).exists():
                            print(f"   ✓ Fixed model_path: '{model_path}' → '{fallback}'")
                            arguments["model_path"] = fallback
                            break
            
            # 🔥 FIX: Generic parameter sanitization - strip any unknown kwargs
            # This prevents "got an unexpected keyword argument" errors from LLM hallucinations
            import inspect
            try:
                sig = inspect.signature(tool_func)
                valid_params = set(sig.parameters.keys())
                invalid_args = [k for k in arguments.keys() if k not in valid_params]
                # Only strip if the function doesn't accept **kwargs
                has_var_keyword = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD 
                    for p in sig.parameters.values()
                )
                if invalid_args and not has_var_keyword:
                    for invalid_param in invalid_args:
                        val = arguments.pop(invalid_param)
                        print(f"   ✓ Stripped hallucinated parameter '{invalid_param}': {val}")
                    print(f"   ℹ️ Valid parameters for {tool_name}: {list(valid_params)}")
            except (ValueError, TypeError):
                pass  # Can't inspect, skip validation
            
            # General parameter corrections for common LLM hallucinations
            if "output" in arguments and "output_path" not in arguments:
                # Many tools use 'output_path' but LLM uses 'output'
                arguments["output_path"] = arguments.pop("output")
            
            # Fix "None" string being passed as actual None
            for key, value in list(arguments.items()):
                if isinstance(value, str) and value.lower() in ["none", "null", "undefined"]:
                    arguments[key] = None
            
            # Log final parameters before execution
            print(f"   📋 Final parameters: {list(arguments.keys())}")
            
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
                # Report failure
                if self.progress_callback:
                    self.progress_callback(tool_name, "failed")
            else:
                tool_result = {
                    "success": True,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result
                }
                # Report success
                if self.progress_callback:
                    self.progress_callback(tool_name, "completed")
            
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
        Handles matplotlib Figures, numpy arrays, infinity values, and other non-serializable types.
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
        
        # Handle infinity and NaN values (not JSON compliant)
        elif isinstance(obj, float):
            import math
            if math.isinf(obj):
                return "Infinity" if obj > 0 else "-Infinity"
            elif math.isnan(obj):
                return "NaN"
            return obj
        
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
    
    def _compress_tools_registry(self, agent_name: str = None) -> List[Dict]:
        """
        Create compressed version of tools registry.
        Optionally filter to only include tools relevant to a specific agent.
        
        Args:
            agent_name: If provided, only include tools relevant to this agent
        
        Returns:
            Compressed and optionally filtered tools list
        """
        # If agent specified, filter tools first
        if agent_name:
            tool_names = get_tools_for_agent(agent_name)
            tools_to_compress = filter_tools_by_names(self.tools_registry, tool_names)
            print(f"🎯 Agent-specific tools: {len(tools_to_compress)} tools for {agent_name}")
        else:
            tools_to_compress = self.tools_registry
        
        compressed = []
        
        for tool in tools_to_compress:
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
                # Compressed but preserves actual data values to prevent hallucination
                r = result.get("result", {})
                shape = r.get("shape", {})
                mem = r.get("memory_usage", {})
                col_types = r.get("column_types", {})
                columns_info = r.get("columns", {})
                
                # Build per-column stats summary (min/max/mean/median for numeric)
                column_stats = {}
                for col_name, col_info in columns_info.items():
                    stats = {"dtype": col_info.get("dtype", "unknown")}
                    if col_info.get("mean") is not None:
                        stats["min"] = col_info.get("min")
                        stats["max"] = col_info.get("max")
                        stats["mean"] = round(col_info["mean"], 4) if col_info["mean"] is not None else None
                        stats["median"] = round(col_info["median"], 4) if col_info.get("median") is not None else None
                    stats["null_pct"] = col_info.get("null_percentage", 0)
                    stats["unique"] = col_info.get("unique_count", 0)
                    if "top_values" in col_info:
                        stats["top_values"] = col_info["top_values"][:3]
                    column_stats[col_name] = stats
                
                compressed["summary"] = {
                    "rows": shape.get("rows"),
                    "cols": shape.get("columns"),
                    "missing_pct": r.get("overall_stats", {}).get("null_percentage", 0),
                    "duplicate_rows": r.get("overall_stats", {}).get("duplicate_rows", 0),
                    "numeric_cols": col_types.get("numeric", []),
                    "categorical_cols": col_types.get("categorical", []),
                    "file_size_mb": mem.get("total_mb", 0),
                    "column_stats": column_stats
                }
                compressed["next_steps"] = ["clean_missing_values", "detect_data_quality_issues"]
                
            elif tool_name == "detect_data_quality_issues":
                r = result.get("result", {})
                summary_data = r.get("summary", {})
                # Preserve actual issue details so LLM can cite real numbers
                critical_issues = r.get("critical", [])
                warning_issues = r.get("warning", [])[:10]  # Cap at 10
                info_issues = r.get("info", [])[:10]
                
                compressed["summary"] = {
                    "total_issues": summary_data.get("total_issues", 0),
                    "critical_count": summary_data.get("critical_count", 0),
                    "warning_count": summary_data.get("warning_count", 0),
                    "info_count": summary_data.get("info_count", 0),
                    "critical_issues": [{"type": i.get("type"), "column": i.get("column"), "message": i.get("message")} for i in critical_issues],
                    "warning_issues": [{"type": i.get("type"), "column": i.get("column"), "message": i.get("message"), "bounds": i.get("bounds")} for i in warning_issues],
                    "info_issues": [{"type": i.get("type"), "column": i.get("column"), "message": i.get("message")} for i in info_issues]
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
            
            # ── Feature importance / selection tools ──
            elif tool_name == "auto_feature_selection":
                r = result.get("result", {})
                # Preserve the actual feature scores — this IS the answer for "feature importance" queries
                feature_scores = r.get("feature_scores", r.get("feature_rankings", {}))
                # Keep top 15 features max
                if isinstance(feature_scores, dict):
                    sorted_feats = sorted(feature_scores.items(), key=lambda x: abs(float(x[1])) if x[1] is not None else 0, reverse=True)[:15]
                    feature_scores = {k: round(float(v), 4) if v is not None else 0 for k, v in sorted_feats}
                compressed["summary"] = {
                    "n_features_original": r.get("n_features_original"),
                    "n_features_selected": r.get("n_features_selected"),
                    "selected_features": r.get("selected_features", [])[:15],
                    "feature_scores": feature_scores,
                    "selection_method": r.get("selection_method"),
                    "task_type": r.get("task_type"),
                    "output_path": r.get("output_path")
                }
                compressed["next_steps"] = ["analyze_correlations", "generate_eda_plots"]
            
            elif tool_name == "analyze_correlations":
                r = result.get("result", {})
                # Preserve high correlations and target correlations — key analytical data
                high_corrs = r.get("high_correlations", [])[:10]  # Top 10 pairs
                target_corrs = r.get("target_correlations", {})
                if isinstance(target_corrs, dict) and "top_features" in target_corrs:
                    target_corrs = {
                        "target": target_corrs.get("target"),
                        "top_features": target_corrs["top_features"][:10]
                    }
                compressed["summary"] = {
                    "numeric_columns_count": len(r.get("numeric_columns", [])),
                    "high_correlations": high_corrs,
                    "target_correlations": target_corrs,
                }
                compressed["next_steps"] = ["auto_feature_selection", "generate_eda_plots"]
            
            elif tool_name in ["train_with_autogluon", "analyze_autogluon_model"]:
                r = result.get("result", {})
                # Preserve model metrics AND feature importance
                feature_importance = r.get("feature_importance", [])
                if isinstance(feature_importance, list):
                    feature_importance = feature_importance[:10]  # Top 10 features
                compressed["summary"] = {
                    "task_type": r.get("task_type"),
                    "best_model": r.get("best_model"),
                    "best_score": r.get("best_score"),
                    "eval_metric": r.get("eval_metric"),
                    "n_models_trained": r.get("n_models_trained"),
                    "feature_importance": feature_importance,
                    "model_path": r.get("model_path", r.get("output_path")),
                    "training_time_seconds": r.get("training_time_seconds")
                }
                compressed["next_steps"] = ["predict_with_autogluon", "generate_model_report"]
                
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
            shape = result_data.get("shape", {})
            col_types = result_data.get("column_types", {})
            overall = result_data.get("overall_stats", {})
            columns_info = result_data.get("columns", {})
            
            # Extract actual per-column stats for grounding
            column_ranges = {}
            for col_name, col_info in columns_info.items():
                if col_info.get("mean") is not None:
                    column_ranges[col_name] = {
                        "min": col_info.get("min"),
                        "max": col_info.get("max"),
                        "mean": round(col_info["mean"], 4) if col_info["mean"] is not None else None,
                        "median": round(col_info["median"], 4) if col_info.get("median") is not None else None,
                    }
            
            self.workflow_state.update_profiling({
                "num_rows": shape.get("rows"),
                "num_columns": shape.get("columns"),
                "missing_percentage": overall.get("null_percentage", 0),
                "duplicate_rows": overall.get("duplicate_rows", 0),
                "numeric_columns": col_types.get("numeric", []),
                "categorical_columns": col_types.get("categorical", []),
                "column_ranges": column_ranges
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
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REASONING LOOP INFRASTRUCTURE
    # Three new methods that power the hypothesis-driven analysis mode:
    #   _llm_text_call       → Provider-agnostic text LLM call (no tool schemas)
    #   _get_tools_description → Lightweight text description of available tools
    #   _run_reasoning_loop   → The core Reason → Act → Evaluate → Loop/Stop cycle
    # ═══════════════════════════════════════════════════════════════════════════

    def _llm_text_call(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
        """
        Simple text-only LLM call (no tool schemas).
        
        Used by Reasoner, Evaluator, and Synthesizer for lightweight
        reasoning calls. Much cheaper than full tool-calling API calls.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            max_tokens: Maximum response tokens
            
        Returns:
            Plain text response from the LLM
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Rate limiting
        if self.min_api_call_interval > 0:
            time_since_last_call = time.time() - self.last_api_call_time
            if time_since_last_call < self.min_api_call_interval:
                wait_time = self.min_api_call_interval - time_since_last_call
                time.sleep(wait_time)
        
        try:
            if self.provider == "mistral":
                if hasattr(self.mistral_client, 'chat') and hasattr(self.mistral_client.chat, 'complete'):
                    response = self.mistral_client.chat.complete(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens
                    )
                else:
                    response = self.mistral_client.chat(
                        model=self.model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=max_tokens
                    )
                self.api_calls_made += 1
                self.last_api_call_time = time.time()
                
                if hasattr(response, 'usage') and response.usage:
                    self.tokens_this_minute += response.usage.total_tokens
                
                return self._extract_content_text(response.choices[0].message.content)
            
            elif self.provider == "groq":
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                self.api_calls_made += 1
                self.last_api_call_time = time.time()
                
                if hasattr(response, 'usage') and response.usage:
                    self.tokens_this_minute += response.usage.total_tokens
                
                return self._extract_content_text(response.choices[0].message.content)
            
            elif self.provider == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.gemini_model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": max_tokens
                    }
                )
                self.api_calls_made += 1
                self.last_api_call_time = time.time()
                return response.text
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            error_str = str(e)
            # Handle rate limits
            if "429" in error_str or "rate_limit" in error_str.lower():
                print(f"⏳ Rate limit in reasoning call, waiting 10s...")
                time.sleep(10)
                return self._llm_text_call(system_prompt, user_prompt, max_tokens)
            raise

    def _get_tools_description(self, tool_names: Optional[List[str]] = None) -> str:
        """
        Build a lightweight text description of available tools.
        
        Used in Reasoner prompts instead of sending full JSON tool schemas.
        This is much more token-efficient than the OpenAI tools format.
        
        Args:
            tool_names: Optional list of tool names to include (None = all tools)
            
        Returns:
            Formatted text like:
                - profile_dataset(file_path): Profile a dataset to understand structure
                - analyze_correlations(file_path, target_col): Analyze column correlations
                ...
        """
        import inspect
        
        lines = []
        tool_map = self.tool_functions
        
        # Filter to specific tools if requested
        if tool_names:
            tool_map = {k: v for k, v in tool_map.items() if k in tool_names}
        
        for name, func in sorted(tool_map.items()):
            # Get function signature
            try:
                sig = inspect.signature(func)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name in ("kwargs", "args"):
                        continue
                    if param.default is inspect.Parameter.empty:
                        params.append(param_name)
                    else:
                        params.append(f"{param_name}=...")
                params_str = ", ".join(params[:5])  # Max 5 params shown
                if len(sig.parameters) > 5:
                    params_str += ", ..."
            except (ValueError, TypeError):
                params_str = "..."
            
            # Get first line of docstring
            doc = (func.__doc__ or "").strip().split("\n")[0][:100]
            
            lines.append(f"- {name}({params_str}): {doc}")
        
        return "\n".join(lines)

    def _get_relevant_tools_sbert(
        self,
        query: str,
        candidate_tools: Optional[set] = None,
        top_k: int = 20,
        threshold: float = 0.15
    ) -> set:
        """
        Use SBERT semantic similarity to rank tools by relevance to the query.
        
        Encodes the query and each tool's (name + docstring) into embeddings,
        then keeps only tools whose cosine similarity exceeds the threshold.
        Tool embeddings are lazily computed and cached for the lifetime of the
        orchestrator instance.
        
        Args:
            query: User's natural language question
            candidate_tools: Tools to score (default: all tool_functions)
            top_k: Max number of tools to return
            threshold: Minimum cosine similarity to include a tool (0.0-1.0)
            
        Returns:
            Set of tool names that are semantically relevant to the query.
            Falls back to candidate_tools unchanged if SBERT is unavailable.
        """
        if not self.semantic_layer.enabled:
            return candidate_tools or set(self.tool_functions.keys())
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            import numpy as np
        except ImportError:
            return candidate_tools or set(self.tool_functions.keys())
        
        candidates = candidate_tools or set(self.tool_functions.keys())
        
        # ── Lazily build & cache tool embeddings ──
        if not hasattr(self, '_tool_embeddings_cache'):
            self._tool_embeddings_cache = {}
        
        # Compute embeddings for any tools not yet cached
        tools_needing_embed = [t for t in candidates if t not in self._tool_embeddings_cache]
        if tools_needing_embed:
            texts = []
            for name in tools_needing_embed:
                func = self.tool_functions.get(name)
                doc = (func.__doc__ or "").strip().split("\n")[0][:150] if func else ""
                texts.append(f"{name}: {doc}")
            
            try:
                embeddings = self.semantic_layer.model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
                )
                for name, emb in zip(tools_needing_embed, embeddings):
                    self._tool_embeddings_cache[name] = emb
            except Exception as e:
                print(f"⚠️ SBERT tool encoding failed: {e}, returning all candidates")
                return candidates
        
        # ── Encode the query ──
        try:
            query_emb = self.semantic_layer.model.encode(
                query, convert_to_numpy=True, show_progress_bar=False
            ).reshape(1, -1)
        except Exception as e:
            print(f"⚠️ SBERT query encoding failed: {e}")
            return candidates
        
        # ── Score each candidate tool ──
        scored = []
        for name in candidates:
            emb = self._tool_embeddings_cache.get(name)
            if emb is None:
                continue
            sim = float(cos_sim(query_emb, emb.reshape(1, -1))[0][0])
            scored.append((name, sim))
        
        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Keep tools above threshold, up to top_k
        selected = {name for name, sim in scored[:top_k] if sim >= threshold}
        
        # ── Always include universally-useful core tools ──
        CORE_TOOLS = {
            "profile_dataset", "analyze_correlations", "auto_feature_selection",
            "generate_eda_plots", "clean_missing_values",
            "execute_python_code",
        }
        selected |= (CORE_TOOLS & candidates)
        
        if selected:
            # Log what SBERT chose
            top5 = scored[:5]
            print(f"   🧠 SBERT tool routing: {len(selected)}/{len(candidates)} tools selected")
            print(f"      Top-5 by similarity: {[(n, f'{s:.3f}') for n, s in top5]}")
        else:
            # Safety: if nothing passed threshold, return all candidates
            print(f"   ⚠️ SBERT: no tools above threshold {threshold}, using all {len(candidates)} candidates")
            selected = candidates
        
        return selected

    def _run_reasoning_loop(
        self,
        question: str,
        file_path: str,
        dataset_info: Dict[str, Any],
        target_col: Optional[str] = None,
        mode: str = "investigative",
        max_iterations: int = 7,
        tool_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the Reasoning Loop: Reason → Act → Evaluate → Loop/Stop → Synthesize.
        
        This is the core of the hypothesis-driven analysis mode.
        Instead of a pipeline, the agent:
        1. REASONS about what to investigate next
        2. ACTS (executes one tool)
        3. EVALUATES the result
        4. Decides to LOOP (investigate more) or STOP
        5. SYNTHESIZES all findings into a coherent answer
        
        Args:
            question: User's question or "Analyze this data"
            file_path: Path to the dataset
            dataset_info: Schema info from local extraction
            target_col: Optional target column
            mode: "investigative" or "exploratory"
            max_iterations: Max reasoning iterations (default 7)
            tool_names: Optional subset of tools to use
            
        Returns:
            Dict with status, summary, findings, workflow_history, etc.
        """
        start_time = time.time()
        
        # Initialize reasoning components (pass our LLM caller)
        reasoner = Reasoner(llm_caller=self._llm_text_call)
        evaluator = Evaluator(llm_caller=self._llm_text_call)
        synthesizer = Synthesizer(llm_caller=self._llm_text_call)
        findings = FindingsAccumulator(question=question, mode=mode)
        
        # ── Intelligent tool filtering for the reasoning loop ──
        # Step 1: Hard-exclude tools that can never work in the reasoning loop
        EXCLUDED_FROM_REASONING = {
            "generate_feature_importance_plot",  # needs Dict[str, float] — Reasoner can't supply
        }
        TRAINING_TOOLS = {
            "train_with_autogluon", "train_baseline_models", "train_model",
            "hyperparameter_tuning", "predict_with_autogluon",
            "analyze_autogluon_model", "advanced_model_training",
            "neural_architecture_search"
        }
        
        # Build initial candidate pool
        effective_tool_names = set(tool_names) if tool_names else set(self.tool_functions.keys())
        effective_tool_names -= EXCLUDED_FROM_REASONING
        
        # Step 2: SBERT semantic routing — score tools against the query
        # This replaces the old keyword-only approach with real semantic understanding
        if self.semantic_layer.enabled:
            print(f"   🧠 Using SBERT semantic routing for tool selection...")
            effective_tool_names = self._get_relevant_tools_sbert(
                query=question,
                candidate_tools=effective_tool_names,
                top_k=20,
                threshold=0.15
            )
        
        # Step 3: Hard safety rail — even if SBERT scores a training tool highly,
        # block it for pure EDA queries (training wastes 120-180s for no benefit)
        question_lower = question.lower()
        explicitly_wants_training = any(kw in question_lower for kw in [
            "train", "predict", "build a model", "classification", "regression",
            "classify", "forecast", "deploy model", "autogluon"
        ])
        if not explicitly_wants_training:
            EDA_KEYWORDS = [
                "feature importance", "important features", "most important",
                "correlations", "correlation", "explore", "explain",
                "understand", "patterns", "insights", "eda", "profiling",
                "distribution", "outliers", "summary", "describe", "overview",
                "what drives", "what affects", "key factors", "top features",
                "feature ranking", "data quality", "missing values"
            ]
            is_eda_query = any(kw in question_lower for kw in EDA_KEYWORDS)
            if is_eda_query:
                removed = effective_tool_names & TRAINING_TOOLS
                if removed:
                    print(f"   🚫 EDA safety rail — removing training tools: {removed}")
                effective_tool_names -= TRAINING_TOOLS
        
        # Get tools description for the reasoner (filtered)
        tools_desc = self._get_tools_description(list(effective_tool_names))
        print(f"   📋 Reasoning loop will see {len(effective_tool_names)} tools (of {len(self.tool_functions)})")
        
        # Track for API response
        workflow_history = []
        original_data_file = file_path  # NEVER changes — always the uploaded dataset
        current_file = file_path        # Tracks the latest DATA file (csv/parquet only)
        
        # Emit mode info for UI
        if hasattr(self, 'session') and self.session:
            progress_manager.emit(self.session.session_id, {
                'type': 'reasoning_mode',
                'mode': mode,
                'message': f"🧠 Reasoning Loop activated ({mode} mode)",
                'question': question
            })
        
        print(f"\n{'='*60}")
        print(f"🧠 REASONING LOOP ({mode.upper()} mode)")
        print(f"   Question: {question}")
        print(f"   Max iterations: {max_iterations}")
        print(f"{'='*60}")
        
        # ── EXPLORATORY MODE: Generate hypotheses first ──
        if mode == "exploratory":
            print(f"\n🔬 Generating hypotheses from data profile...")
            
            # Profile the dataset first if not already done
            profile_result = self._execute_tool("profile_dataset", {"file_path": file_path})
            profile_summary = ""
            if profile_result.get("success", True):
                profile_summary = json.dumps(
                    self._compress_tool_result("profile_dataset", 
                        self._make_json_serializable(profile_result)),
                    default=str
                )[:2000]
                
                workflow_history.append({
                    "iteration": 0,
                    "tool": "profile_dataset",
                    "arguments": {"file_path": file_path},
                    "result": profile_result
                })
                self._update_workflow_state("profile_dataset", profile_result)
            
            # Generate hypotheses
            hypotheses = reasoner.generate_hypotheses(
                dataset_info=dataset_info,
                file_path=file_path,
                target_col=target_col,
                profile_summary=profile_summary
            )
            
            print(f"   Generated {len(hypotheses)} hypotheses:")
            for i, h in enumerate(hypotheses):
                text = h.get("text", str(h))
                priority = h.get("priority", 0.5)
                findings.add_hypothesis(text, priority=priority, source_iteration=0)
                print(f"   {i+1}. [{priority:.1f}] {text}")
            
            # Emit hypothesis info
            if hasattr(self, 'session') and self.session:
                progress_manager.emit(self.session.session_id, {
                    'type': 'hypotheses_generated',
                    'hypotheses': [h.get("text", str(h)) for h in hypotheses],
                    'count': len(hypotheses)
                })
        
        # ── MAIN REASONING LOOP ──
        for iteration in range(1, max_iterations + 1):
            print(f"\n── Iteration {iteration}/{max_iterations} ──")
            
            # STEP 1: REASON - What should we investigate next?
            print(f"🤔 REASON: Deciding next action...")
            
            reasoning_output = reasoner.reason(
                question=question,
                dataset_info=dataset_info,
                findings=findings,
                available_tools=tools_desc,
                file_path=current_file,
                target_col=target_col
            )
            
            print(f"   Status: {reasoning_output.status}")
            print(f"   Reasoning: {reasoning_output.reasoning}")
            
            # Check if done
            if reasoning_output.status == "done":
                print(f"✅ Reasoner says: DONE (confidence: {reasoning_output.confidence:.0%})")
                print(f"   Reason: {reasoning_output.reasoning}")
                break
            
            tool_name = reasoning_output.tool_name
            tool_args = reasoning_output.arguments
            hypothesis = reasoning_output.hypothesis
            
            if not tool_name or tool_name not in self.tool_functions:
                print(f"⚠️  Invalid tool: {tool_name}, skipping iteration")
                continue
            
            print(f"   Tool: {tool_name}")
            print(f"   Hypothesis: {hypothesis}")
            
            # Emit reasoning step for UI
            if hasattr(self, 'session') and self.session:
                progress_manager.emit(self.session.session_id, {
                    'type': 'reasoning_step',
                    'iteration': iteration,
                    'tool': tool_name,
                    'hypothesis': hypothesis,
                    'reasoning': reasoning_output.reasoning
                })
            
            # STEP 2: ACT - Execute the tool
            print(f"⚡ ACT: Executing {tool_name}...")
            
            # Emit tool execution event
            if hasattr(self, 'session') and self.session:
                progress_manager.emit(self.session.session_id, {
                    'type': 'tool_executing',
                    'tool': tool_name,
                    'message': f"🔧 Executing: {tool_name}",
                    'arguments': tool_args
                })
            
            tool_result = self._execute_tool(tool_name, tool_args)
            
            # Determine success/failure
            tool_success = tool_result.get("success", True)
            tool_error = ""
            
            # Track output file for next iteration — ONLY update for data files
            if tool_success:
                result_data = tool_result.get("result", {})
                if isinstance(result_data, dict):
                    new_file = result_data.get("output_file") or result_data.get("output_path")
                    if new_file:
                        # Only update current_file for actual data files (CSV, parquet, etc.)
                        # NOT for visualizations (HTML, PNG, JPG) or reports
                        data_extensions = ('.csv', '.parquet', '.xlsx', '.xls', '.json', '.tsv')
                        if new_file.lower().endswith(data_extensions):
                            current_file = new_file
                            print(f"   📂 Updated current data file: {new_file}")
                        else:
                            print(f"   📊 Output artifact (not updating data file): {new_file}")
                
                # Emit success
                if hasattr(self, 'session') and self.session:
                    progress_manager.emit(self.session.session_id, {
                        'type': 'tool_completed',
                        'tool': tool_name,
                        'message': f"✓ Completed: {tool_name}"
                    })
                print(f"   ✓ Tool completed successfully")
            else:
                error_msg = tool_result.get("error", "Unknown error")
                tool_error = str(error_msg)[:300]
                print(f"   ❌ Tool failed: {error_msg}")
                # Record failure so Reasoner won't retry this tool
                findings.add_failed_tool(tool_name, tool_error)
                if hasattr(self, 'session') and self.session:
                    progress_manager.emit(self.session.session_id, {
                        'type': 'tool_failed',
                        'tool': tool_name,
                        'message': f"❌ FAILED: {tool_name}",
                        'error': error_msg
                    })
            
            # Track in workflow history
            workflow_history.append({
                "iteration": iteration,
                "tool": tool_name,
                "arguments": tool_args,
                "result": tool_result
            })
            
            # Update workflow state
            self._update_workflow_state(tool_name, tool_result)
            
            # Checkpoint
            if tool_success:
                session_id = self.http_session_key or "default"
                self.recovery_manager.checkpoint_manager.save_checkpoint(
                    session_id=session_id,
                    workflow_state={
                        'iteration': iteration,
                        'workflow_history': workflow_history,
                        'current_file': file_path,
                        'task_description': question,
                        'target_col': target_col
                    },
                    last_tool=tool_name,
                    iteration=iteration
                )
            
            # STEP 3: EVALUATE - What did we learn?
            print(f"📊 EVALUATE: Interpreting results...")
            
            evaluation = evaluator.evaluate(
                question=question,
                tool_name=tool_name,
                arguments=tool_args,
                result=tool_result,
                findings=findings,
                result_compressor=lambda tn, r: self._compress_tool_result(
                    tn, self._make_json_serializable(r)
                )
            )
            
            print(f"   Interpretation: {evaluation.interpretation}")
            print(f"   Answered: {evaluation.answered} (confidence: {evaluation.confidence:.0%})")
            print(f"   Should stop: {evaluation.should_stop}")
            if evaluation.next_questions:
                print(f"   Next questions: {evaluation.next_questions}")
            
            # Build finding and add to accumulator
            compressed_result = json.dumps(
                self._compress_tool_result(tool_name, self._make_json_serializable(tool_result)),
                default=str
            )
            
            finding = evaluator.build_finding(
                iteration=iteration,
                hypothesis=hypothesis,
                tool_name=tool_name,
                arguments=tool_args,
                result_summary=compressed_result,
                evaluation=evaluation,
                success=tool_success,
                error_message=tool_error
            )
            findings.add_finding(finding)
            
            # Update hypothesis status based on evaluation results
            if hypothesis:
                if tool_success and evaluation.confidence >= 0.6:
                    findings.update_hypothesis(
                        hypothesis, "supported", evaluation.interpretation, is_supporting=True
                    )
                elif tool_success and evaluation.confidence >= 0.3:
                    findings.update_hypothesis(
                        hypothesis, "inconclusive", evaluation.interpretation, is_supporting=True
                    )
                elif not tool_success:
                    findings.update_hypothesis(
                        hypothesis, "inconclusive", f"Tool failed: {tool_error}", is_supporting=False
                    )
            
            # Emit finding for UI
            if hasattr(self, 'session') and self.session:
                progress_manager.emit(self.session.session_id, {
                    'type': 'finding_discovered',
                    'iteration': iteration,
                    'interpretation': evaluation.interpretation,
                    'confidence': evaluation.confidence,
                    'answered': evaluation.answered
                })
            
            # Check if we should stop
            if evaluation.should_stop:
                print(f"\n✅ Evaluator says: STOP (confidence: {evaluation.confidence:.0%})")
                break
        
        # ── STEP 4: SYNTHESIZE - Build the final answer ──
        print(f"\n{'='*60}")
        print(f"📝 SYNTHESIZE: Building final answer from {len(findings.findings)} findings...")
        print(f"{'='*60}")
        
        # Guard: If ALL findings failed, return honest error instead of hallucinated synthesis
        successful_findings = findings.get_successful_findings()
        if findings.findings and not successful_findings:
            failed_tools = ", ".join(findings.failed_tools.keys()) if findings.failed_tools else "unknown"
            summary_text = (
                "## Analysis Could Not Be Completed\n\n"
                f"All {len(findings.findings)} investigation steps failed. "
                f"**Failed tools**: {failed_tools}\n\n"
                "**Possible causes:**\n"
                "- The dataset file may be corrupted or in an unsupported format\n"
                "- Column names in the query may not match the actual dataset\n"
                "- Required dependencies may be missing\n\n"
                "**Recommended next steps:**\n"
                "1. Re-upload the dataset and try again\n"
                "2. Check that column names are correct\n"
                "3. Try a simpler query first (e.g., 'profile this dataset')"
            )
            print(f"⚠️  All tools failed — returning honest error instead of synthesis")
        else:
            # Collect artifacts from workflow history
            artifacts = self._collect_artifacts(workflow_history)
        
            # Generate synthesis
            if mode == "exploratory":
                summary_text = synthesizer.synthesize_exploratory(
                    findings=findings,
                    artifacts=artifacts
                )
            else:
                summary_text = synthesizer.synthesize(
                    findings=findings,
                    artifacts=artifacts
                )
        
        # Also generate enhanced summary for plots/metrics extraction
        try:
            enhanced = self._generate_enhanced_summary(
                workflow_history, summary_text, question
            )
            plots_data = enhanced.get("plots", [])
            metrics_data = enhanced.get("metrics", {})
            artifacts_data = enhanced.get("artifacts", {})
        except Exception as e:
            print(f"⚠️  Enhanced summary generation failed: {e}")
            plots_data = []
            metrics_data = {}
            artifacts_data = {}
        
        # Save to session
        if self.session:
            self.session.add_conversation(question, summary_text)
            self.session_store.save(self.session)
        
        result = {
            "status": "success",
            "summary": summary_text,
            "metrics": metrics_data,
            "artifacts": artifacts_data,
            "plots": plots_data,
            "workflow_history": workflow_history,
            "findings": findings.to_dict(),
            "reasoning_trace": self.reasoning_trace.get_trace(),
            "reasoning_summary": self.reasoning_trace.get_trace_summary(),
            "execution_mode": mode,
            "iterations": findings.iteration_count,
            "api_calls": self.api_calls_made,
            "execution_time": round(time.time() - start_time, 2)
        }
        
        print(f"\n✅ Reasoning loop completed in {result['execution_time']}s")
        print(f"   Iterations: {findings.iteration_count}")
        print(f"   Tools used: {', '.join(findings.tools_used)}")
        print(f"   API calls: {self.api_calls_made}")
        
        return result

    def _collect_artifacts(self, workflow_history: List[Dict]) -> Dict[str, Any]:
        """Collect plots, files, and other artifacts from workflow history."""
        plots = []
        files = []
        
        for step in workflow_history:
            result = step.get("result", {})
            if not isinstance(result, dict):
                continue
            
            result_data = result.get("result", result)
            if isinstance(result_data, dict):
                # Collect output files
                for key in ["output_file", "output_path", "report_path"]:
                    if key in result_data and result_data[key]:
                        files.append(result_data[key])
                
                # Collect plots
                if "plots" in result_data:
                    for plot in result_data["plots"]:
                        if isinstance(plot, dict):
                            plots.append(plot)
                        elif isinstance(plot, str):
                            plots.append({"path": plot, "title": step.get("tool", "Plot")})
                
                # Check for HTML files (interactive plots)
                for key in ["html_path", "dashboard_path"]:
                    if key in result_data and result_data[key]:
                        plots.append({
                            "path": result_data[key],
                            "title": step.get("tool", "Interactive Plot"),
                            "type": "html"
                        })
        
        return {"plots": plots, "files": files}

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
        # 🛡️ SAFETY: Ensure max_iterations is never None (prevent NoneType comparison errors)
        if max_iterations is None:
            max_iterations = 20
            print(f"⚠️  max_iterations was None, defaulting to 20")
        
        start_time = time.time()
        
        # 🧹 CLEAR OLD CHECKPOINTS: Start fresh for each new workflow
        # This prevents stale checkpoint resumption when user starts a new query
        session_id = self.http_session_key or "default"
        if self.recovery_manager.checkpoint_manager.can_resume(session_id):
            print(f"🗑️  Clearing old checkpoint to start fresh workflow")
            self.recovery_manager.checkpoint_manager.clear_checkpoint(session_id)
        
        # 🧠 RESOLVE AMBIGUITY USING SESSION MEMORY (BEFORE SCHEMA EXTRACTION)
        # This ensures follow-up requests can find the file before we try to extract schema
        original_file_path = file_path
        original_target_col = target_col
        
        if self.session:
            # Check if request has ambiguous references
            resolved_params = self.session.resolve_ambiguity(task_description)
            print(f"[DEBUG] Orchestrator received resolved_params: {resolved_params}")
            print(f"[DEBUG] Current file_path: '{file_path}', target_col: '{target_col}'")
            
            # 🔥 FIX: Only use resolved file_path if user did NOT provide a new file
            # If file_path is already set (user uploaded a new file), DON'T override it
            if not file_path or file_path == "":
                if resolved_params.get("file_path"):
                    file_path = resolved_params["file_path"]
                    print(f"📝 Using dataset from session: {file_path}")
                else:
                    print(f"[DEBUG] No file_path in resolved_params")
            else:
                print(f"📝 User provided new file: {file_path} (ignoring session file: {resolved_params.get('file_path', 'none')})")
            
            if not target_col:
                if resolved_params.get("target_col"):
                    target_col = resolved_params["target_col"]
                    print(f"📝 Using target column from session: {target_col}")

            
            # Show session context if available (but show CURRENT file, not old one)
            if self.session.last_dataset or self.session.last_model:
                # 🔥 FIX: Update session's last_dataset to current file BEFORE showing context
                # This prevents stale session context from misleading the LLM
                if file_path and file_path != self.session.last_dataset:
                    print(f"📝 Updating session dataset: {self.session.last_dataset} → {file_path}")
                    self.session.last_dataset = file_path
                context_summary = self.session.get_context_summary()
                print(f"\n{context_summary}\n")
        
        # 🚀 LOCAL SCHEMA EXTRACTION (NO LLM) - Extract metadata before any LLM calls
        # Now that file_path is resolved from session if needed
        
        # 🛡️ VALIDATION: Ensure we have a valid file path
        if not file_path or file_path == "":
            error_msg = "No dataset file provided. Please upload a CSV, Excel, or Parquet file."
            print(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "summary": "Cannot proceed without a dataset file.",
                "workflow_history": [],
                "execution_time": 0.0
            }
        
        print("🔍 Extracting dataset schema locally (no LLM)...")
        schema_info = extract_schema_local(file_path, sample_rows=3)
        
        if 'error' not in schema_info:
            # Guard: Reject empty datasets immediately instead of wasting reasoning iterations
            if schema_info.get('num_rows', 0) == 0:
                return {
                    "status": "error",
                    "error": "Dataset is empty (0 rows)",
                    "summary": "The uploaded dataset contains no data rows. Please upload a dataset with at least one row of data.",
                    "workflow_history": [],
                    "execution_time": time.time() - start_time
                }
            
            # 🧠 SEMANTIC LAYER: Enrich dataset info with column embeddings
            if self.semantic_layer.enabled:
                try:
                    schema_info = self.semantic_layer.enrich_dataset_info(schema_info, file_path, sample_size=100)
                    print(f"🧠 Semantic layer enriched {len(schema_info.get('column_embeddings', {}))} columns")
                except Exception as e:
                    print(f"⚠️ Semantic enrichment failed: {e}")
            
            # Update workflow state with schema
            self.workflow_state.update_dataset_info(schema_info)
            print(f"✅ Schema extracted: {schema_info['num_rows']} rows × {schema_info['num_columns']} cols")
            print(f"   File size: {schema_info['file_size_mb']} MB")
            
            # 🧠 SEMANTIC LAYER: Infer target column if not provided
            if not target_col and self.semantic_layer.enabled:
                try:
                    inferred = self.semantic_layer.infer_target_column(
                        schema_info.get('column_embeddings', {}),
                        task_description
                    )
                    if inferred:
                        target_col, confidence = inferred
                        print(f"💡 Inferred target column: {target_col} (confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"⚠️ Target inference failed: {e}")
            
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🧠 INTENT CLASSIFICATION → MODE SELECTION
        # Classify the user's request into one of three execution modes:
        #   DIRECT:        "Make a scatter plot"      → existing pipeline
        #   INVESTIGATIVE: "Why are customers churning?" → reasoning loop  
        #   EXPLORATORY:   "Analyze this data"        → hypothesis-driven loop
        # ═══════════════════════════════════════════════════════════════════════
        intent_classifier = IntentClassifier(semantic_layer=self.semantic_layer)
        intent_result = intent_classifier.classify(
            query=task_description,
            dataset_info=schema_info if 'error' not in schema_info else None,
            has_target_col=bool(target_col)
        )
        
        print(f"\n🎯 Intent Classification:")
        print(f"   Mode: {intent_result.mode.upper()}")
        print(f"   Confidence: {intent_result.confidence:.0%}")
        print(f"   Reasoning: {intent_result.reasoning}")
        print(f"   Sub-intent: {intent_result.sub_intent}")
        
        # Emit intent info for UI
        if hasattr(self, 'session') and self.session:
            progress_manager.emit(self.session.session_id, {
                'type': 'intent_classified',
                'mode': intent_result.mode,
                'confidence': intent_result.confidence,
                'reasoning': intent_result.reasoning,
                'sub_intent': intent_result.sub_intent
            })
        
        # 📝 Record intent classification in reasoning trace
        self.reasoning_trace.trace_history.append({
            "type": "intent_classification",
            "query": task_description,
            "mode": intent_result.mode,
            "confidence": intent_result.confidence,
            "reasoning": intent_result.reasoning,
            "sub_intent": intent_result.sub_intent
        })
        
        # ═══════════════════════════════════════════════════════════════════════
        # 🧠 REASONING LOOP PATH (Investigative / Exploratory modes)
        # ═══════════════════════════════════════════════════════════════════════
        if intent_result.mode in ("investigative", "exploratory"):
            print(f"\n🧠 Routing to REASONING LOOP ({intent_result.mode} mode)")
            
            # Determine iteration count based on mode and reasoning effort
            if intent_result.mode == "exploratory":
                loop_max = min(max_iterations, 8)  # Exploratory gets more iterations
            else:
                loop_max = min(max_iterations, 6)  # Investigative is more focused
            
            reasoning_result = self._run_reasoning_loop(
                question=task_description,
                file_path=file_path,
                dataset_info=schema_info if 'error' not in schema_info else {},
                target_col=target_col,
                mode=intent_result.mode,
                max_iterations=loop_max
            )
            
            # Cache the result
            if use_cache and reasoning_result.get("status") == "success":
                self.cache.set(cache_key, reasoning_result, metadata={
                    "file_path": file_path,
                    "task": task_description,
                    "mode": intent_result.mode
                })
            
            return reasoning_result
        
        # ═══════════════════════════════════════════════════════════════════════
        # 📋 DIRECT MODE PATH (existing pipeline - below is unchanged)
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n📋 Routing to DIRECT pipeline mode")
        
        # Build initial messages
        # Use dynamic prompts for small context models
        if self.use_compact_prompts:
            from .dynamic_prompts import build_compact_system_prompt
            system_prompt = build_compact_system_prompt(user_query=task_description)
            print("🔧 Using compact prompt for small context window")
        else:
            # 🤖 MULTI-AGENT ARCHITECTURE: Route to specialist agent
            selected_agent = self._select_specialist_agent(task_description)
            self.active_agent = selected_agent
            current_agent = selected_agent  # Track for dynamic tool loading
            
            # 📝 Record agent selection in reasoning trace
            if self.semantic_layer.enabled:
                # Get confidence from semantic routing
                agent_descriptions = {name: config["description"] for name, config in self.specialist_agents.items()}
                _, confidence = self.semantic_layer.route_to_agent(task_description, agent_descriptions)
                self.reasoning_trace.record_agent_selection(
                    task=task_description,
                    selected_agent=selected_agent,
                    confidence=confidence,
                    alternatives=agent_descriptions
                )
            
            agent_config = self.specialist_agents[selected_agent]
            print(f"\n{agent_config['emoji']} Delegating to: {agent_config['name']}")
            print(f"   Specialization: {agent_config['description']}")
            
            # 🎯 DYNAMIC TOOL LOADING: Load only tools relevant to this agent
            tools_to_use = self._compress_tools_registry(agent_name=selected_agent)
            print(f"   📦 Loaded {len(tools_to_use)} agent-specific tools")
            
            # Use specialist's system prompt
            system_prompt = agent_config["system_prompt"]
            
            # Emit agent info for UI display
            if self.progress_callback:
                self.progress_callback({
                    "type": "agent_assigned",
                    "agent": agent_config['name'],
                    "emoji": agent_config['emoji'],
                    "description": agent_config['description'],
                    "tools_count": len(tools_to_use)
                })
        
        
        # 🎯 PROACTIVE INTENT DETECTION - Tell LLM which tools to use BEFORE it tries wrong ones
        task_lower = task_description.lower()
        
        # Detect user intent
        wants_viz = any(kw in task_lower for kw in ["plot", "graph", "visualiz", "dashboard", "chart", "show", "display", "create", "generate"])
        wants_clean = any(kw in task_lower for kw in ["clean", "missing", "impute"])
        wants_features = any(kw in task_lower for kw in ["feature", "engineer", "time-based", "extract features"])
        wants_train = any(kw in task_lower for kw in ["train", "model", "predict", "best model", "classify", "regression", "forecast", "build model"])
        
        # 🔍 CRITICAL: Detect exploratory/relationship questions (should NOT trigger ML training)
        wants_relationship = any(kw in task_lower for kw in [
            "how does", "how do", "relationship", "relate", "correlation", "correlate",
            "affect", "effect", "impact", "influence", "change with", "vary with",
            "compare", "difference between", "distribution", "pattern"
        ])
        
        # 🎯 AUTO-ENABLE TRAINING: Only if explicitly asking for predictions AND not asking about relationships
        # Don't auto-enable for exploratory questions even if target exists
        if target_col and not wants_viz and not wants_clean and not wants_relationship and self.workflow_state.task_type in ["regression", "classification"]:
            # Additional check: only auto-enable if question implies prediction
            if wants_train or any(kw in task_lower for kw in ["predict", "forecast", "estimate"]):
                print(f"   🎯 Auto-enabling ML training (detected {self.workflow_state.task_type} task with target='{target_col}')")
                wants_train = True
        elif wants_relationship:
            # Override: Relationship questions should NOT train models
            print(f"   🔍 Exploratory analysis detected - disabling auto-ML (question asks about relationships, not predictions)")
            wants_train = False
        
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
            target_info = f"\n🎯 **TARGET COLUMN**: '{target_col}' (Task: {self.workflow_state.task_type or 'auto'})\n" if target_col else "\n⚠️ **TARGET COLUMN**: Not specified - analyze correlations to find best candidate\n"
            workflow_guidance = (
                "\n\n🎯 **WORKFLOW**: Full ML Pipeline (Training Requested)"
                f"{target_info}"
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
            # Create explicit column list for validation
            all_columns = ', '.join([f"'{col}'" for col in list(info['columns'].keys())[:15]])
            if len(info['columns']) > 15:
                all_columns += f"... ({len(info['columns'])} total)"
            
            state_context = f"""
**Dataset Schema** (extracted locally):
- Rows: {info['num_rows']:,} | Columns: {info['num_columns']}
- Size: {info['file_size_mb']} MB
- Numeric columns ({len(info['numeric_columns'])}): {', '.join([f"'{c}'" for c in info['numeric_columns'][:10]])}{'...' if len(info['numeric_columns']) > 10 else ''}
- Categorical columns ({len(info['categorical_columns'])}): {', '.join([f"'{c}'" for c in info['categorical_columns'][:10]])}{'...' if len(info['categorical_columns']) > 10 else ''}

**IMPORTANT - Exact Column Names:**
{all_columns}

⚠️ When calling modeling tools, use EXACT column names from above.
⚠️ DO NOT hallucinate column names like "Target", "Label", "Occupation" unless they appear above.
⚠️ If unsure about target column, use profile_dataset first to inspect data.
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
        
        # current_agent and tools_to_use are set above in agent selection
        # If compact prompts used, prepare general tools here
        if self.use_compact_prompts:
            current_agent = None
            tools_to_use = self._compress_tools_registry(agent_name="general_agent")
        
        # For Gemini, use the existing model without tools (text-only mode)
        # Gemini tool schema is incompatible with OpenAI/Groq format
        # Tool execution is handled by our orchestrator, not by Gemini itself
        gemini_chat = None
        if self.provider == "gemini":
            gemini_chat = self.gemini_model.start_chat(history=[])
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # 🚀 SMART CONVERSATION PRUNING (Mistral-compatible)
                # Keep only: system + user + last 12 exchanges (24 messages) - INCREASED FOR BETTER CONTEXT
                # CRITICAL: Maintain valid message ordering for Mistral API
                
                # Helper function to get role from message (handles dict or ChatMessage object)
                def get_role(msg):
                    if isinstance(msg, dict):
                        return msg.get('role', '')
                    return getattr(msg, 'role', '')
                
                # Helper to check if message has tool_calls
                def has_tool_calls(msg):
                    if isinstance(msg, dict):
                        return bool(msg.get('tool_calls'))
                    return bool(getattr(msg, 'tool_calls', None))
                
                if len(messages) > 26:
                    # Keep: system prompt [0], user query [1], last valid exchanges
                    system_msg = messages[0]
                    user_msg = messages[1]
                    recent_msgs = messages[-8:]
                    
                    # CRITICAL: Keep complete tool call/response groups together
                    # Mistral requires: assistant (with tool_calls) → tool responses → assistant → user
                    cleaned_recent = []
                    i = 0
                    while i < len(recent_msgs):
                        msg = recent_msgs[i]
                        role = get_role(msg)
                        
                        if role == 'assistant' and has_tool_calls(msg):
                            # This assistant has tool calls - must keep it AND all following tool responses
                            cleaned_recent.append(msg)
                            i += 1
                            # Collect all consecutive tool responses
                            while i < len(recent_msgs) and get_role(recent_msgs[i]) == 'tool':
                                cleaned_recent.append(recent_msgs[i])
                                i += 1
                        elif role == 'tool':
                            # Orphaned tool message (no preceding assistant with tool_calls) - skip it
                            i += 1
                        else:
                            # Regular message (assistant without tool_calls, user, system)
                            cleaned_recent.append(msg)
                            i += 1
                    
                    # 🔥 CRITICAL FIX: Remove orphaned tool messages at the start of cleaned_recent
                    # Mistral NEVER allows 'tool' role immediately after 'user' role
                    while cleaned_recent and get_role(cleaned_recent[0]) == 'tool':
                        print(f"⚠️  Removed orphaned tool message at start of pruned history")
                        cleaned_recent.pop(0)
                    
                    messages = [system_msg, user_msg] + cleaned_recent
                    print(f"✂️  Pruned conversation (keeping last 12 exchanges for better context preservation)")
                    
                    # 🎯 INJECT CONTEXT REMINDER after pruning (prevent LLM from forgetting)
                    context_parts = []
                    if target_col and self.workflow_state.task_type:
                        context_parts.append(f"📌 Target column: '{target_col}' (Task: {self.workflow_state.task_type})")
                    
                    # Inject profiling/quality context that would have been pruned
                    if self.workflow_state.profiling_summary:
                        ps = self.workflow_state.profiling_summary
                        context_parts.append(f"📊 Dataset: {ps.get('num_rows', '?')} rows × {ps.get('num_columns', '?')} cols")
                        if ps.get('column_ranges'):
                            ranges = ps['column_ranges']
                            range_lines = [f"  {col}: min={v.get('min')}, max={v.get('max')}, mean={v.get('mean')}" 
                                          for col, v in list(ranges.items())[:8]]
                            context_parts.append("Column ranges:\n" + "\n".join(range_lines))
                    
                    if self.workflow_state.quality_issues:
                        qi = self.workflow_state.quality_issues
                        if qi.get('total_issues', 0) > 0:
                            context_parts.append(f"⚠️ Quality: {qi.get('total_issues', 0)} issues found")
                    
                    if context_parts:
                        reminder = {
                            "role": "user",
                            "content": "REMINDER (original profiling context — preserved after pruning):\n" + "\n".join(context_parts)
                        }
                        messages.insert(2, reminder)  # Insert after system + user query
                
                # 🔍 Token estimation and warning
                estimated_tokens = sum(
                    len(str(m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', ''))) // 4 
                    for m in messages
                )
                if estimated_tokens > 15000:
                    # Emergency pruning - keep only last 8 exchanges
                    system_msg = messages[0]
                    user_msg = messages[1]
                    recent_msgs = messages[-16:]
                    
                    # CRITICAL: Keep complete tool call/response groups together
                    cleaned_recent = []
                    i = 0
                    while i < len(recent_msgs):
                        msg = recent_msgs[i]
                        role = get_role(msg)
                        
                        if role == 'assistant' and has_tool_calls(msg):
                            # Keep assistant with tool calls AND all its tool responses
                            cleaned_recent.append(msg)
                            i += 1
                            while i < len(recent_msgs) and get_role(recent_msgs[i]) == 'tool':
                                cleaned_recent.append(recent_msgs[i])
                                i += 1
                        elif role == 'tool':
                            # Skip orphaned tool message
                            i += 1
                        else:
                            cleaned_recent.append(msg)
                            i += 1
                    
                    # 🔥 CRITICAL FIX: Remove orphaned tool messages at the start of cleaned_recent
                    # Mistral NEVER allows 'tool' role immediately after 'user' role
                    while cleaned_recent and get_role(cleaned_recent[0]) == 'tool':
                        print(f"⚠️  Removed orphaned tool message at start of emergency pruned history")
                        cleaned_recent.pop(0)
                    
                    messages = [system_msg, user_msg] + cleaned_recent
                    print(f"⚠️  Emergency pruning (conversation > 15K tokens, keeping last 8 exchanges)")
                
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
                
                # 💰 TOKEN BUDGET: Enforce context window limits before LLM call
                messages, token_count = self.token_manager.enforce_budget(
                    messages=messages,
                    system_prompt=system_prompt
                )
                print(f"💰 Token budget: {token_count}/{self.token_manager.max_tokens} ({(token_count/self.token_manager.max_tokens*100):.1f}%)")
                
                # 🔥 CRITICAL: Validate message order for Mistral API compliance
                # Mistral requires: system → user → assistant → tool (only after assistant with tool_calls) → assistant → user...
                # NEVER: user → tool (this causes "Unexpected role 'tool' after role 'user'" error)
                if self.provider in ["mistral", "groq"]:
                    validated_messages = []
                    for i, msg in enumerate(messages):
                        role = get_role(msg)
                        
                        # Check if this is a tool message after a user message
                        if role == 'tool' and validated_messages:
                            prev_role = get_role(validated_messages[-1])
                            if prev_role == 'user':
                                # Invalid! Skip this tool message
                                print(f"⚠️  WARNING: Skipped orphaned tool message at position {i} (after user message)")
                                continue
                        
                        validated_messages.append(msg)
                    
                    messages = validated_messages
                    print(f"✅ Message order validation complete: {len(messages)} messages")
                
                # Call LLM with function calling (provider-specific)
                if self.provider == "mistral":
                    try:
                        # Support both new SDK (v1.x) and old SDK (v0.x)
                        if hasattr(self.mistral_client, 'chat') and hasattr(self.mistral_client.chat, 'complete'):
                            # New SDK (v1.x)
                            response = self.mistral_client.chat.complete(
                                model=self.model,
                                messages=messages,
                                tools=tools_to_use,
                                tool_choice="auto",
                                temperature=0.1,
                                max_tokens=4096
                            )
                        else:
                            # Old SDK (v0.x)
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
                            
                            # Emit token update for SSE streaming using session UUID
                            if hasattr(self, 'session') and self.session:
                                progress_manager.emit(self.session.session_id, {
                                    'type': 'token_update',
                                    'message': f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute",
                                    'tokens_used': tokens_used,
                                    'tokens_this_minute': self.tokens_this_minute,
                                    'tpm_limit': self.tpm_limit
                                })
                        
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls
                        final_content = self._extract_content_text(response_message.content)
                        
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
                            
                            # Emit token update for SSE streaming using session UUID
                            if hasattr(self, 'session') and self.session:
                                progress_manager.emit(self.session.session_id, {
                                    'type': 'token_update',
                                    'message': f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute",
                                    'tokens_used': tokens_used,
                                    'tokens_this_minute': self.tokens_this_minute,
                                    'tpm_limit': self.tpm_limit
                                })
                        
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls
                        final_content = self._extract_content_text(response_message.content)
                        
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
                                
                                # Emit token update for SSE streaming using session UUID
                                if hasattr(self, 'session') and self.session:
                                    progress_manager.emit(self.session.session_id, {
                                        'type': 'token_update',
                                        'message': f"📊 Tokens: {tokens_used} this call | {self.tokens_this_minute}/{self.tpm_limit} this minute",
                                        'tokens_used': tokens_used,
                                        'tokens_this_minute': self.tokens_this_minute,
                                        'tpm_limit': self.tpm_limit
                                    })
                            
                            response_message = response.choices[0].message
                            tool_calls = response_message.tool_calls
                            final_content = self._extract_content_text(response_message.content)
                        else:
                            # Not a rate limit error, re-raise
                            raise
                
                # Check if done (no tool calls)
                if not tool_calls:
                    # Final response
                    final_summary = final_content or "Analysis completed"
                    
                    # 🎯 ENHANCED SUMMARY: Extract metrics and artifacts from workflow (with error handling)
                    try:
                        enhanced_summary = self._generate_enhanced_summary(
                            workflow_history, 
                            final_summary, 
                            task_description
                        )
                        summary_text = enhanced_summary["text"]
                        
                        # 🧹 POST-PROCESS: Light cleanup only
                        import re
                        
                        # Clean excessive whitespace only
                        summary_text = re.sub(r'\n\n\n+', '\n\n', summary_text)
                        summary_text = summary_text.strip()
                        
                        metrics_data = enhanced_summary.get("metrics", {})
                        artifacts_data = enhanced_summary.get("artifacts", {})
                        artifacts_data = enhanced_summary.get("artifacts", {})
                        plots_data = enhanced_summary.get("plots", [])
                        print(f"✅ Enhanced summary generated with {len(plots_data)} plots, {len(metrics_data)} metrics")
                        
                        # DEBUG: Log plots array details
                        if plots_data:
                            print(f"[DEBUG] Plots array contains {len(plots_data)} items:")
                            for idx, plot in enumerate(plots_data):
                                print(f"[DEBUG]   Plot {idx+1}: title='{plot.get('title')}', url='{plot.get('url')}', type='{plot.get('type')}'")
                    except Exception as e:
                        print(f"⚠️  Enhanced summary generation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: use basic summary
                        summary_text = final_summary
                        metrics_data = {}
                        artifacts_data = {}
                        plots_data = []
                    
                    # 🧠 Save conversation to session memory
                    if self.session:
                        self.session.add_conversation(task_description, summary_text)
                        self.session_store.save(self.session)
                        print(f"\n✅ Session saved: {self.session.session_id}")
                    
                    result = {
                        "status": "success",
                        "summary": summary_text,
                        "metrics": metrics_data,
                        "artifacts": artifacts_data,
                        "plots": plots_data,
                        "workflow_history": workflow_history,
                        "reasoning_trace": self.reasoning_trace.get_trace(),
                        "reasoning_summary": self.reasoning_trace.get_trace_summary(),
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
                if self.provider in ["groq", "mistral"]:
                    messages.append(response_message)
                
                # 🚀 PARALLEL EXECUTION: Detect multiple independent tool calls
                # ⚠️ DISABLED FOR STABILITY - Parallel execution causes race conditions and OOM errors
                # Re-enable only after implementing proper request isolation per user
                if len(tool_calls) > 1 and False:  # Disabled with "and False"
                    print(f"🚀 Detected {len(tool_calls)} tool calls - attempting parallel execution")
                    
                    # Extract tool executions with proper weight classification
                    tool_executions = []
                    heavy_tools = []
                    for idx, tc in enumerate(tool_calls):
                        if self.provider in ["groq", "mistral"]:
                            tool_name = tc.function.name
                            tool_args_raw = tc.function.arguments
                            # Sanitize tool name
                            import re
                            tool_name = re.sub(r'[^\x00-\x7F]+', '', str(tool_name))
                            match = re.search(r'([a-z_][a-z0-9_]*)', tool_name, re.IGNORECASE)
                            if match:
                                tool_name = match.group(1)
                            
                            if tool_name in self.tool_functions:
                                tool_args = json.loads(tool_args_raw)
                                weight = TOOL_WEIGHTS.get(tool_name, ToolWeight.MEDIUM)
                                
                                # Track heavy tools
                                if weight == ToolWeight.HEAVY:
                                    heavy_tools.append(tool_name)
                                
                                tool_executions.append(ToolExecution(
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    weight=weight,
                                    dependencies=set(),
                                    execution_id=f"{tool_name}_{idx}"
                                ))
                        elif self.provider == "gemini":
                            tool_name = tc.name
                            tool_args = {key: value for key, value in tc.args.items()}
                            if tool_name in self.tool_functions:
                                weight = TOOL_WEIGHTS.get(tool_name, ToolWeight.MEDIUM)
                                
                                # Track heavy tools
                                if weight == ToolWeight.HEAVY:
                                    heavy_tools.append(tool_name)
                                
                                tool_executions.append(ToolExecution(
                                    tool_name=tool_name,
                                    arguments=tool_args,
                                    weight=weight,
                                    dependencies=set(),
                                    execution_id=f"{tool_name}_{idx}"
                                ))
                    
                    # ⚠️ CRITICAL: Prevent multiple heavy tools from running in parallel
                    if len(heavy_tools) > 1:
                        print(f"⚠️ Multiple HEAVY tools detected: {heavy_tools}")
                        print(f"   These will run SEQUENTIALLY to prevent resource exhaustion")
                        print(f"   Heavy tools: {', '.join(heavy_tools)}")
                        # Fall through to sequential execution
                    elif len(tool_executions) > 1 and len(heavy_tools) <= 1 and self.parallel_executor is not None:
                        try:
                            results = asyncio.run(self.parallel_executor.execute_all(
                                tool_executions=tool_executions,
                                tool_executor=self._execute_tool_sync,
                                progress_callback=self._async_progress_callback
                            ))
                            
                            print(f"✓ Parallel execution completed: {len(results)} tools")
                            
                            # Add results to messages and workflow history
                            for tool_exec, tool_result in zip(tool_executions, results):
                                tool_name = tool_exec.tool_name
                                tool_args = tool_exec.arguments
                                tool_call_id = tool_exec.execution_id
                                
                                # Save checkpoint
                                if tool_result.get("success", True):
                                    session_id = self.http_session_key or "default"
                                    self.recovery_manager.checkpoint_manager.save_checkpoint(
                                        session_id=session_id,
                                        workflow_state={
                                            'iteration': iteration,
                                            'workflow_history': workflow_history,
                                            'current_file': file_path,
                                            'task_description': task_description,
                                            'target_col': target_col
                                        },
                                        last_tool=tool_name,
                                        iteration=iteration
                                    )
                                
                                # Track in workflow
                                workflow_history.append({
                                    "iteration": iteration,
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_result
                                })
                                
                                # Update workflow state
                                self._update_workflow_state(tool_name, tool_result)
                                
                                # Add to messages with compression
                                clean_tool_result = self._make_json_serializable(tool_result)
                                compressed_result = self._compress_tool_result(tool_name, clean_tool_result)
                                
                                if self.provider in ["mistral", "groq"]:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": json.dumps(compressed_result)
                                    })
                                elif self.provider == "gemini":
                                    messages.append({
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps(compressed_result)
                                    })
                            
                            # Skip sequential execution
                            continue
                            
                        except Exception as e:
                            print(f"⚠️ Parallel execution failed: {e}")
                            print("   Falling back to sequential execution")
                
                # Sequential execution (fallback or single tool)
                for tool_call in tool_calls:
                    # Extract tool name and args (provider-specific)
                    if self.provider in ["groq", "mistral"]:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_call_id = tool_call.id
                        
                        # CRITICAL FIX 1: Sanitize tool_name (remove any non-ASCII or prefix garbage)
                        import re
                        # Remove any non-ASCII characters and leading garbage
                        tool_name_cleaned = re.sub(r'[^\x00-\x7F]+', '', str(tool_name))
                        # Extract just the alphanumeric_underscore pattern
                        match = re.search(r'([a-z_][a-z0-9_]*)', tool_name_cleaned, re.IGNORECASE)
                        if match:
                            tool_name = match.group(1)
                        
                        # CRITICAL FIX 2: Validate tool exists before execution
                        if tool_name not in self.tool_functions:
                            print(f"⚠️  INVALID TOOL NAME: '{tool_name}' (original: {tool_call.function.name})")
                            print(f"   Available tools: {', '.join(list(self.tool_functions.keys())[:10])}...")
                            
                            # Explicit mappings for common LLM hallucinations
                            tool_name_mappings = {
                                "drop_columns": "execute_python_code",  # No drop_columns tool, use code
                                "select_columns": "execute_python_code",  # No select_columns tool, use code
                                "rename_columns": "execute_python_code",  # No rename_columns tool, use code
                                "create_geospatial_features": "create_interaction_features",  # No geospatial tool, use interaction features
                                "encode_categorical_variables": "encode_categorical",
                                "train_model": "train_baseline_models",
                                "train_models": "train_baseline_models",
                                "baseline_models": "train_baseline_models",
                                "tune_hyperparameters": "hyperparameter_tuning",
                                "hyperparameter_search": "hyperparameter_tuning",
                            }
                            
                            if tool_name in tool_name_mappings:
                                mapped_tool = tool_name_mappings[tool_name]
                                if mapped_tool == "execute_python_code":
                                    print(f"   ✓ Tool '{tool_name}' not available - LLM should use execute_python_code instead")
                                    # Skip and let LLM handle with code
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": json.dumps({
                                            "error": f"Tool '{tool_name}' does not exist",
                                            "hint": "Use execute_python_code with pandas to perform this operation. Example: df.drop(columns=['col1', 'col2'])"
                                        })
                                    })
                                    continue
                                else:
                                    tool_name = mapped_tool
                                    print(f"   ✓ Mapped to: {tool_name}")
                            else:
                                # Try fuzzy matching to recover
                                from difflib import get_close_matches
                                close_matches = get_close_matches(tool_name, self.tool_functions.keys(), n=1, cutoff=0.6)
                                if close_matches:
                                    tool_name = close_matches[0]
                                    print(f"   ✓ Recovered using fuzzy match: {tool_name}")
                                else:
                                    print(f"   ❌ Cannot recover tool name, skipping")
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": "invalid_tool",
                                        "content": json.dumps({
                                            "error": f"Invalid tool: {tool_call.function.name}",
                                            "message": "Tool does not exist in registry. Available tools can be found in the tools list.",
                                            "hint": "Check spelling and use exact tool names from the tools registry."
                                        })
                                    })
                                    continue
                        
                        # CRITICAL FIX 3: Check for corrupted tool names (length check)
                        if len(str(tool_call.function.name)) > 100:
                            print(f"⚠️  CORRUPTED TOOL NAME DETECTED: {str(tool_name)[:200]}")
                            # Try to extract actual tool name from garbage
                            import re
                            # Look for valid tool name pattern at the end
                            match = re.search(r'([a-z_]+)[\"\']?\s*$', str(tool_name), re.IGNORECASE)
                            if match:
                                recovered_name = match.group(1)
                                # Validate recovered tool name exists in registry
                                if recovered_name in self.tool_functions:
                                    tool_name = recovered_name
                                    print(f"✓ Recovered tool name: {tool_name}")
                                else:
                                    print(f"❌ Recovered '{recovered_name}' but it's not a valid tool")
                                    print(f"❌ Cannot recover tool name, skipping this tool call")
                                    # CRITICAL: Add tool response to maintain conversation state for Mistral API
                                    # Mistral requires messages to alternate: user -> assistant -> tool -> assistant
                                    # Skipping without adding response breaks this pattern
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": "invalid_tool",
                                        "content": json.dumps({
                                            "error": "Corrupted tool name detected",
                                            "message": "The LLM returned invalid text instead of a tool call. Please try again with a valid tool.",
                                            "hint": "Use the session context to continue from where you left off."
                                        })
                                    })
                                    continue
                            else:
                                print(f"❌ Cannot recover tool name, skipping this tool call")
                                # CRITICAL: Add tool response to maintain conversation state for Mistral API
                                # Mistral requires messages to alternate: user -> assistant -> tool -> assistant
                                # Skipping without adding response breaks this pattern
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": "invalid_tool",
                                    "content": json.dumps({
                                        "error": "Corrupted tool name detected",
                                        "message": "The LLM returned invalid text instead of a tool call. Please try again with a valid tool.",
                                        "hint": "Use the session context to continue from where you left off."
                                    })
                                })
                                continue
                        
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
                        
                        # CRITICAL: Add mock tool response to maintain message balance
                        if self.provider in ["mistral", "groq"]:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": json.dumps({"blocked": True, "reason": block_reason})
                            })
                        elif self.provider == "gemini":
                            messages.append({
                                "role": "tool",
                                "name": tool_name,
                                "content": json.dumps({"blocked": True, "reason": block_reason})
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
                                # CRITICAL: Add mock tool response
                                if self.provider in ["mistral", "groq"]:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": json.dumps({"blocked": True, "reason": "Encoding already done"})
                                    })
                                elif self.provider == "gemini":
                                    messages.append({
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps({"blocked": True, "reason": "Encoding already done"})
                                    })
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
                                # CRITICAL: Add mock tool response
                                if self.provider in ["mistral", "groq"]:
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "name": tool_name,
                                        "content": json.dumps({"blocked": True, "reason": "Time features already extracted"})
                                    })
                                elif self.provider == "gemini":
                                    messages.append({
                                        "role": "tool",
                                        "name": tool_name,
                                        "content": json.dumps({"blocked": True, "reason": "Time features already extracted"})
                                    })
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
                            # CRITICAL: Add mock tool response
                            if self.provider in ["mistral", "groq"]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Time features already extracted"})
                                })
                            elif self.provider == "gemini":
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Time features already extracted"})
                                })
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
                            # CRITICAL: Add mock tool response
                            if self.provider in ["mistral", "groq"]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Categorical encoding already done"})
                                })
                            elif self.provider == "gemini":
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Categorical encoding already done"})
                                })
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
                            # CRITICAL: Add mock tool response
                            if self.provider in ["mistral", "groq"]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Type inference not needed after encoding"})
                                })
                            elif self.provider == "gemini":
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps({"blocked": True, "reason": "Type inference not needed after encoding"})
                                })
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
                    
                    # AGGRESSIVE: For execute_python_code with same args, detect after 1 retry
                    loop_threshold = 2
                    if tool_name == "execute_python_code":
                        # Check if same code being executed repeatedly
                        if workflow_history:
                            last_exec_steps = [s for s in workflow_history if s["tool"] == "execute_python_code"]
                            if len(last_exec_steps) >= 1:
                                last_code = last_exec_steps[-1].get("arguments", {}).get("code", "")
                                current_code = tool_args.get("code", "")
                                # If same/similar code, be more aggressive
                                if last_code and current_code and len(set(last_code.split()) & set(current_code.split())) > len(current_code.split()) * 0.7:
                                    loop_threshold = 1  # Stop after first retry with similar code
                                    print(f"⚠️  Detected repeated similar code execution")
                    
                    # 🔥 FIX: Check if arguments are DIFFERENT from last call
                    # If the same tool is called with different arguments, it's NOT a loop
                    # (e.g., generating multiple different plots is legitimate)
                    is_same_args = False
                    if workflow_history and workflow_history[-1]["tool"] == tool_name:
                        last_args = workflow_history[-1].get("arguments", {})
                        # Compare key arguments (ignore output paths which may differ)
                        ignore_keys = {"output_path", "output_dir"}
                        last_key_args = {k: v for k, v in last_args.items() if k not in ignore_keys}
                        current_key_args = {k: v for k, v in tool_args.items() if k not in ignore_keys}
                        is_same_args = (last_key_args == current_key_args)
                    
                    # Check for loops (same tool called threshold+ times consecutively WITH SAME ARGS)
                    if should_check_loops and tool_call_counter[tool_name] >= loop_threshold:
                        # Only flag as loop if last call was same tool WITH same arguments
                        if workflow_history and workflow_history[-1]["tool"] == tool_name and is_same_args:
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
                            
                            # 🎯 If data prep is done but no training yet, push toward modeling
                            prep_done = any(t in completed_tools for t in ["encode_categorical", "create_time_features", "clean_missing_values"])
                            no_training = "train_baseline_models" not in completed_tools
                            if prep_done and no_training and target_col:
                                next_step = f"train_baseline_models (target_col='{target_col}') - Data preparation complete, proceed to modeling!"
                            
                            # CRITICAL: Add mock tool response to maintain message balance
                            # (Mistral API requires: every tool call must have a matching tool response)
                            if self.provider in ["mistral", "groq"]:
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": json.dumps({
                                        "blocked": True,
                                        "reason": f"Loop detected: {tool_name} called {tool_call_counter[tool_name]} times consecutively",
                                        "last_successful_file": self._get_last_successful_file(workflow_history)
                                    })
                                })
                            elif self.provider == "gemini":
                                messages.append({
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps({
                                        "blocked": True,
                                        "reason": f"Loop detected: {tool_name} called {tool_call_counter[tool_name]} times consecutively"
                                    })
                                })
                            
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
                    
                    # Emit progress event for SSE streaming using session UUID
                    if hasattr(self, 'session') and self.session:
                        session_id = self.session.session_id
                        print(f"[SSE] EMIT tool_executing: session={session_id}, tool={tool_name}")
                        progress_manager.emit(session_id, {
                            'type': 'tool_executing',
                            'tool': tool_name,
                            'message': f"🔧 Executing: {tool_name}",
                            'arguments': tool_args
                        })
                    
                    # Execute tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # 📂 CHECKPOINT: Save progress after successful tool execution
                    if tool_result.get("success", True):
                        session_id = self.http_session_key or "default"
                        self.recovery_manager.checkpoint_manager.save_checkpoint(
                            session_id=session_id,
                            workflow_state={
                                'iteration': iteration,
                                'workflow_history': workflow_history,
                                'current_file': file_path,
                                'task_description': task_description,
                                'target_col': target_col
                            },
                            last_tool=tool_name,
                            iteration=iteration
                        )
                    
                    # Check for errors and display them prominently
                    if not tool_result.get("success", True):
                        error_msg = tool_result.get("error", "Unknown error")
                        error_type = tool_result.get("error_type", "Error")
                        print(f"   ❌ FAILED: {tool_name}")
                        print(f"   ⚠️  Error Type: {error_type}")
                        print(f"   ⚠️  Error Message: {error_msg}")
                        
                        # Emit failure event for SSE streaming
                        if hasattr(self, 'session') and self.session:
                            progress_manager.emit(self.session.session_id, {
                                'type': 'tool_failed',
                                'tool': tool_name,
                                'message': f"❌ FAILED: {tool_name}",
                                'error': error_msg,
                                'error_type': error_type
                            })
                        
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
                        
                        # Emit completion event for SSE streaming
                        if hasattr(self, 'session') and self.session:
                            progress_manager.emit(self.session.session_id, {
                                'type': 'tool_completed',
                                'tool': tool_name,
                                'message': f"✓ Completed: {tool_name}"
                            })
                    
                    # Track in workflow
                    workflow_history.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": tool_result
                    })
                    
                    # 🤝 INTER-AGENT COMMUNICATION: Check if should hand off to specialist
                    if not self.use_compact_prompts:  # Only for multi-agent mode
                        completed_tool_names = [step["tool"] for step in workflow_history]
                        target_agent = self._should_hand_off(
                            current_agent=self.active_agent,
                            completed_tools=completed_tool_names,
                            workflow_history=workflow_history
                        )
                        
                        if target_agent:
                            hand_off_result = self._hand_off_to_agent(
                                target_agent=target_agent,
                                context={
                                    "completed_tools": completed_tool_names,
                                    "reason": "Workflow progression - ready for next phase"
                                },
                                iteration=iteration
                            )
                            
                            if hand_off_result["success"]:
                                # Update tools for new agent
                                tools_to_use = hand_off_result["new_tools"]
                                
                                # Update system prompt for new agent
                                messages[0] = {"role": "system", "content": hand_off_result["system_prompt"]}
                                
                                # 📝 Record hand-off in reasoning trace
                                self.reasoning_trace.record_agent_handoff(
                                    from_agent=hand_off_result["old_agent"],
                                    to_agent=hand_off_result["new_agent"],
                                    reason="Workflow progression - ready for next phase",
                                    iteration=iteration
                                )
                    
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

