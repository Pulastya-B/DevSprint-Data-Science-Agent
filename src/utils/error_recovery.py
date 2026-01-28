"""
Error Recovery and Graceful Degradation System

Provides retry mechanisms, fallback strategies, and workflow checkpointing
to make the agent resilient to tool failures and API errors.
"""

import functools
import time
import json
import traceback
from typing import Callable, Any, Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime


class RetryStrategy:
    """Configuration for retry behavior."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 exponential_backoff: bool = True, fallback_tools: Optional[List[str]] = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exponential_backoff = exponential_backoff
        self.fallback_tools = fallback_tools or []


# Tool-specific retry strategies
TOOL_RETRY_STRATEGIES = {
    # Data loading tools - retry with backoff
    "profile_dataset": RetryStrategy(max_retries=2, base_delay=1.0),
    "detect_data_quality_issues": RetryStrategy(max_retries=2, base_delay=1.0),
    
    # Expensive tools - don't retry, use fallback
    "train_baseline_models": RetryStrategy(max_retries=0, fallback_tools=["execute_python_code"]),
    "hyperparameter_tuning": RetryStrategy(max_retries=0),
    "train_ensemble_models": RetryStrategy(max_retries=0),
    
    # Visualization - retry once
    "generate_interactive_scatter": RetryStrategy(max_retries=1),
    "generate_plotly_dashboard": RetryStrategy(max_retries=1),
    
    # Code execution - retry with longer delay
    "execute_python_code": RetryStrategy(max_retries=1, base_delay=2.0),
    
    # Feature engineering - retry with alternative methods
    "encode_categorical": RetryStrategy(max_retries=1, fallback_tools=["force_numeric_conversion"]),
    "clean_missing_values": RetryStrategy(max_retries=1, fallback_tools=["handle_outliers"]),
}


def retry_with_fallback(tool_name: Optional[str] = None):
    """
    Decorator for automatic retry with exponential backoff and fallback strategies.
    
    Features:
    - Configurable retry attempts per tool
    - Exponential backoff between retries
    - Fallback to alternative tools on persistent failure
    - Detailed error logging
    
    Args:
        tool_name: Name of tool (for strategy lookup)
    
    Example:
        @retry_with_fallback(tool_name="train_baseline_models")
        def execute_tool(tool_name, arguments):
            # Tool execution logic
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get tool name from kwargs or args
            actual_tool_name = tool_name or kwargs.get('tool_name') or (args[0] if args else None)
            
            # Get retry strategy
            strategy = TOOL_RETRY_STRATEGIES.get(
                actual_tool_name,
                RetryStrategy(max_retries=1)  # Default strategy
            )
            
            last_error = None
            
            # Attempt execution with retries
            for attempt in range(strategy.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Success - check if result indicates error
                    if isinstance(result, dict):
                        if result.get("success") is False or "error" in result:
                            last_error = result.get("error", "Tool returned error")
                            # Don't retry if it's a validation error
                            if "does not exist" in str(last_error) or "not found" in str(last_error):
                                return result  # Validation errors shouldn't retry
                            raise Exception(last_error)
                    
                    # Success!
                    if attempt > 0:
                        print(f"✅ Retry successful on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    last_error = e
                    
                    if attempt < strategy.max_retries:
                        # Calculate delay with exponential backoff
                        delay = strategy.base_delay * (2 ** attempt) if strategy.exponential_backoff else strategy.base_delay
                        print(f"⚠️ {actual_tool_name} failed (attempt {attempt + 1}/{strategy.max_retries + 1}): {str(e)[:100]}")
                        print(f"   Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        # Max retries exhausted
                        print(f"❌ {actual_tool_name} failed after {strategy.max_retries + 1} attempts")
            
            # All retries failed - return error result with fallback info
            error_result = {
                "success": False,
                "error": str(last_error),
                "error_type": type(last_error).__name__,
                "traceback": traceback.format_exc(),
                "tool_name": actual_tool_name,
                "attempts": strategy.max_retries + 1,
                "fallback_suggestions": strategy.fallback_tools
            }
            
            print(f"💡 Suggested fallback tools: {strategy.fallback_tools}")
            
            return error_result
        
        return wrapper
    return decorator


class WorkflowCheckpointManager:
    """
    Manages workflow checkpoints for crash recovery.
    
    Saves workflow state after each successful tool execution,
    allowing resume from last successful step if process crashes.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, session_id: str, workflow_state: Any, 
                       last_tool: str, iteration: int) -> str:
        """
        Save workflow checkpoint.
        
        Args:
            session_id: Session identifier
            workflow_state: WorkflowState object
            last_tool: Last successfully executed tool
            iteration: Current iteration number
        
        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "last_tool": last_tool,
            "workflow_state": workflow_state.to_dict() if hasattr(workflow_state, 'to_dict') else {},
            "can_resume": True
        }
        
        checkpoint_path = self.checkpoint_dir / f"{session_id}_checkpoint.json"
        
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            
            print(f"💾 Checkpoint saved: iteration {iteration}, last tool: {last_tool}")
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
            return ""
    
    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{session_id}_checkpoint.json"
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            
            print(f"📂 Checkpoint loaded: iteration {checkpoint['iteration']}, last tool: {checkpoint['last_tool']}")
            return checkpoint
            
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            return None
    
    def can_resume(self, session_id: str) -> bool:
        """Check if session has resumable checkpoint."""
        checkpoint = self.load_checkpoint(session_id)
        return checkpoint is not None and checkpoint.get("can_resume", False)
    
    def clear_checkpoint(self, session_id: str):
        """Clear checkpoint after successful completion."""
        checkpoint_path = self.checkpoint_dir / f"{session_id}_checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"🗑️ Checkpoint cleared for session {session_id}")
            except Exception as e:
                print(f"⚠️ Failed to clear checkpoint: {e}")
    
    def list_checkpoints(self) -> List[Tuple[str, datetime]]:
        """List all available checkpoints with timestamps."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*_checkpoint.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                session_id = data['session_id']
                timestamp = datetime.fromisoformat(data['timestamp'])
                checkpoints.append((session_id, timestamp))
            except:
                continue
        
        return sorted(checkpoints, key=lambda x: x[1], reverse=True)


class ErrorRecoveryManager:
    """
    Centralized error recovery management.
    
    Combines retry logic, checkpointing, and error analysis.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_manager = WorkflowCheckpointManager(checkpoint_dir)
        self.error_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def log_error(self, session_id: str, tool_name: str, error: Exception,
                  context: Optional[Dict[str, Any]] = None):
        """Log error for analysis and pattern detection."""
        if session_id not in self.error_history:
            self.error_history[session_id] = []
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool_name": tool_name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.error_history[session_id].append(error_entry)
    
    def get_error_patterns(self, session_id: str) -> Dict[str, Any]:
        """Analyze error patterns for session."""
        if session_id not in self.error_history:
            return {}
        
        errors = self.error_history[session_id]
        
        # Count errors by tool
        tool_errors = {}
        for error in errors:
            tool = error['tool_name']
            tool_errors[tool] = tool_errors.get(tool, 0) + 1
        
        # Count errors by type
        error_types = {}
        for error in errors:
            err_type = error['error_type']
            error_types[err_type] = error_types.get(err_type, 0) + 1
        
        return {
            "total_errors": len(errors),
            "errors_by_tool": tool_errors,
            "errors_by_type": error_types,
            "most_recent": errors[-3:] if errors else []
        }
    
    def should_abort(self, session_id: str, max_errors: int = 10) -> bool:
        """Check if session should abort due to too many errors."""
        if session_id not in self.error_history:
            return False
        
        return len(self.error_history[session_id]) >= max_errors


# Global error recovery manager
_recovery_manager = None

def get_recovery_manager() -> ErrorRecoveryManager:
    """Get or create global error recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = ErrorRecoveryManager()
    return _recovery_manager
