"""
Production-grade tool result compression for small context window models.
Add this function to orchestrator.py before _parse_text_tool_calls method.
"""

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
        warning_issues = r.get("warning", [])[:10]
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
        if models:
            best = max(models, key=lambda m: m.get("test_score", 0))
            compressed["summary"] = {
                "best_model": best.get("model"),
                "test_score": round(best.get("test_score", 0), 4),
                "train_score": round(best.get("train_score", 0), 4),
                "task_type": r.get("task_type"),
                "models_trained": len(models)
            }
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
