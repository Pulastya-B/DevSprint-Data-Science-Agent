"""
Session Memory Manager
Maintains context across user interactions for intelligent follow-up handling.

This module enables the agent to remember previous interactions and resolve
ambiguous requests like "cross validate it" or "add features to that".
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


class SessionMemory:
    """
    Manages session-based memory for contextual AI interactions.
    
    Features:
    - Stores last dataset, model, target column
    - Tracks workflow history
    - Resolves ambiguous pronouns ("it", "that", "the model")
    - Maintains conversation context
    
    Example:
        User: "Train model on earthquake.csv predicting mag"
        Agent stores: last_model="XGBoost", last_dataset="earthquake.csv"
        
        User: "Cross validate it"
        Agent resolves: "it" → XGBoost, uses stored context
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize session memory.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        # Core context - what the agent last worked on
        self.last_dataset: Optional[str] = None
        self.last_target_col: Optional[str] = None
        self.last_model: Optional[str] = None
        self.last_task_type: Optional[str] = None  # regression, classification
        self.best_score: Optional[float] = None
        
        # Output tracking - where things were saved
        self.last_output_files: Dict[str, str] = {}
        
        # Workflow history - what steps were executed
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Conversation context - for pronoun resolution
        self.conversation_context: List[Dict[str, str]] = []
        
        # Tool results cache - detailed results from last tools
        self.last_tool_results: Dict[str, Any] = {}
    
    def update(self, **kwargs):
        """
        Update session context with new information.
        
        Args:
            last_dataset: Path to dataset
            last_target_col: Target column name
            last_model: Model name (XGBoost, RandomForest, etc.)
            last_task_type: Task type (regression, classification)
            best_score: Best model score
            last_output_files: Dict of output file paths
        
        Example:
            session.update(
                last_dataset="./data/sales.csv",
                last_model="XGBoost",
                best_score=0.92
            )
        """
        self.last_active = datetime.now()
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_workflow_step(self, tool_name: str, result: Dict[str, Any]):
        """
        Add a workflow step to history and extract context.
        
        Args:
            tool_name: Name of the tool executed
            result: Tool execution result
        """
        self.workflow_history.append({
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "result": result
        })
        
        # Update context based on tool results
        self._extract_context_from_tool(tool_name, result)
    
    def _extract_context_from_tool(self, tool_name: str, result: Dict[str, Any]):
        """
        Extract relevant context from tool execution.
        Automatically updates session state based on what tools did.
        
        Args:
            tool_name: Name of the tool
            result: Tool result dictionary
        """
        # Skip if tool failed
        if not result.get("success"):
            return
        
        tool_result = result.get("result", {})
        
        # Track dataset from profiling
        if tool_name == "profile_dataset":
            # Extract file path from arguments if available
            if "file_path" in result.get("arguments", {}):
                self.last_dataset = result["arguments"]["file_path"]
        
        # Track model training results
        if tool_name == "train_baseline_models":
            best_model = tool_result.get("best_model", {})
            if isinstance(best_model, dict):
                self.last_model = best_model.get("name")
                self.best_score = best_model.get("score")
            else:
                self.last_model = best_model
            
            self.last_task_type = tool_result.get("task_type")
            
            # Extract target column from arguments
            if "target_col" in result.get("arguments", {}):
                self.last_target_col = result["arguments"]["target_col"]
        
        # Track hyperparameter tuning results
        if tool_name == "hyperparameter_tuning":
            if "best_score" in tool_result:
                self.best_score = tool_result["best_score"]
            if "model_type" in result.get("arguments", {}):
                self.last_model = result["arguments"]["model_type"]
        
        # Track cross-validation results
        if tool_name == "perform_cross_validation":
            if "mean_score" in tool_result:
                # Store CV score separately (could add cv_score attribute)
                pass
        
        # Track output files from data processing
        if "output_path" in tool_result:
            tool_category = self._categorize_tool(tool_name)
            self.last_output_files[tool_category] = tool_result["output_path"]
            
            # Update last_dataset if this is a data transformation
            if tool_category in ["cleaned", "encoded", "engineered"]:
                self.last_dataset = tool_result["output_path"]
        
        # Store tool results for detailed access
        self.last_tool_results[tool_name] = tool_result
    
    def _categorize_tool(self, tool_name: str) -> str:
        """
        Categorize tool for output tracking.
        
        Args:
            tool_name: Name of the tool
        
        Returns:
            Category string (cleaned, encoded, model, etc.)
        """
        if "clean" in tool_name:
            return "cleaned"
        elif "encode" in tool_name:
            return "encoded"
        elif "feature" in tool_name and "engineer" in tool_name:
            return "engineered"
        elif "train" in tool_name or "model" in tool_name:
            return "model"
        elif "plot" in tool_name or "visual" in tool_name:
            return "visualization"
        elif "report" in tool_name:
            return "report"
        else:
            return "other"
    
    def add_conversation(self, user_message: str, agent_response: str):
        """
        Add conversation turn to context.
        
        Args:
            user_message: User's request
            agent_response: Agent's response/summary
        """
        self.conversation_context.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "agent": agent_response
        })
        
        # Keep only last 10 turns to avoid memory bloat
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def resolve_ambiguity(self, task_description: str) -> Dict[str, Any]:
        """
        Resolve ambiguous references in user request.
        
        Handles pronouns like "it", "that", "this" by mapping to session context.
        
        Args:
            task_description: User's request (may contain "it", "that", etc.)
        
        Returns:
            Dict with resolved parameters (file_path, target_col, model_type)
        
        Example:
            User: "Cross validate it"
            → Returns: {"file_path": "encoded.csv", "target_col": "mag", "model_type": "xgboost"}
        """
        task_lower = task_description.lower()
        resolved = {}
        
        # Pronouns that reference last model/dataset
        ambiguous_refs = ["it", "that", "this", "the model", "the dataset", "the data"]
        has_ambiguous_ref = any(ref in task_lower for ref in ambiguous_refs)
        
        # Cross-validation requests
        if "cross validat" in task_lower or "cv" in task_lower or "validate" in task_lower:
            if has_ambiguous_ref or not any(word in task_lower for word in ["file_path=", "target_col=", "model_type="]):
                # Use session context to fill in missing parameters
                if self.last_output_files.get("encoded"):
                    resolved["file_path"] = self.last_output_files.get("encoded")
                elif self.last_dataset:
                    resolved["file_path"] = self.last_dataset
                
                if self.last_target_col:
                    resolved["target_col"] = self.last_target_col
                
                if self.last_model:
                    resolved["model_type"] = self._normalize_model_name(self.last_model)
        
        # Hyperparameter tuning requests
        if "tun" in task_lower or "optim" in task_lower or "improve" in task_lower:
            if has_ambiguous_ref or "file_path" not in task_lower:
                if self.last_output_files.get("encoded"):
                    resolved["file_path"] = self.last_output_files.get("encoded")
                elif self.last_dataset:
                    resolved["file_path"] = self.last_dataset
                
                if self.last_target_col:
                    resolved["target_col"] = self.last_target_col
                
                if self.last_model:
                    resolved["model_type"] = self._normalize_model_name(self.last_model)
        
        # Visualization requests referencing "the results" or "it"
        if ("plot" in task_lower or "visualiz" in task_lower or "graph" in task_lower) and has_ambiguous_ref:
            if self.last_dataset:
                resolved["file_path"] = self.last_dataset
            
            if self.last_target_col:
                resolved["target_col"] = self.last_target_col
        
        # "Add feature" or "create feature" requests
        if ("add feature" in task_lower or "create feature" in task_lower or 
            "engineer feature" in task_lower or "extract feature" in task_lower):
            if has_ambiguous_ref or "file_path" not in task_lower:
                # Use most recent processed file
                if self.last_output_files.get("encoded"):
                    resolved["file_path"] = self.last_output_files.get("encoded")
                elif self.last_output_files.get("cleaned"):
                    resolved["file_path"] = self.last_output_files.get("cleaned")
                elif self.last_dataset:
                    resolved["file_path"] = self.last_dataset
        
        # Generic "use that" or "try it" commands
        if has_ambiguous_ref and not resolved:
            # Fallback: use last dataset and target
            if self.last_dataset:
                resolved["file_path"] = self.last_dataset
            if self.last_target_col:
                resolved["target_col"] = self.last_target_col
        
        return resolved
    
    def _normalize_model_name(self, model_name: Optional[str]) -> Optional[str]:
        """
        Normalize model name for tool compatibility.
        
        Different tools may use different naming conventions.
        This maps common variations to standard names.
        
        Args:
            model_name: Model name from session (e.g., "XGBoost Classifier")
        
        Returns:
            Normalized name (e.g., "xgboost")
        """
        if not model_name:
            return None
        
        name_lower = model_name.lower()
        
        if "xgb" in name_lower:
            return "xgboost"
        elif "random" in name_lower or "forest" in name_lower:
            return "random_forest"
        elif "ridge" in name_lower:
            return "ridge"
        elif "lasso" in name_lower:
            return "ridge"  # Use ridge for lasso (same tool)
        elif "logistic" in name_lower:
            return "logistic"
        elif "gradient boost" in name_lower and "xgb" not in name_lower:
            return "gradient_boosting"
        elif "svm" in name_lower or "support vector" in name_lower:
            return "svm"
        else:
            # Return as-is if unknown
            return model_name.lower().replace(" ", "_")
    
    def get_context_summary(self) -> str:
        """
        Generate human-readable context summary.
        
        Returns:
            Formatted string describing current session state
        
        Example:
            **Session Context:**
            - Dataset: ./data/earthquake.csv
            - Target Column: mag
            - Last Model: XGBoost
            - Best Score: 0.9234
            - Task Type: regression
        """
        if not self.last_dataset and not self.last_model:
            return "No previous context available."
        
        summary = "**Session Context:**\n"
        
        if self.last_dataset:
            summary += f"- Dataset: {self.last_dataset}\n"
        
        if self.last_target_col:
            summary += f"- Target Column: {self.last_target_col}\n"
        
        if self.last_model:
            summary += f"- Last Model: {self.last_model}\n"
        
        if self.best_score is not None:
            summary += f"- Best Score: {self.best_score:.4f}\n"
        
        if self.last_task_type:
            summary += f"- Task Type: {self.last_task_type}\n"
        
        if self.last_output_files:
            summary += "- Output Files:\n"
            for category, path in self.last_output_files.items():
                summary += f"  - {category}: {path}\n"
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize session to dictionary for storage.
        
        Returns:
            Dictionary with all session data
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "last_dataset": self.last_dataset,
            "last_target_col": self.last_target_col,
            "last_model": self.last_model,
            "last_task_type": self.last_task_type,
            "best_score": self.best_score,
            "last_output_files": self.last_output_files,
            "workflow_history": self.workflow_history,
            "conversation_context": self.conversation_context,
            "last_tool_results": self.last_tool_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionMemory':
        """
        Deserialize session from dictionary.
        
        Args:
            data: Dictionary with session data (from to_dict())
        
        Returns:
            SessionMemory instance
        """
        session = cls(session_id=data.get("session_id"))
        
        # Restore timestamps
        if data.get("created_at"):
            session.created_at = datetime.fromisoformat(data.get("created_at"))
        if data.get("last_active"):
            session.last_active = datetime.fromisoformat(data.get("last_active"))
        
        # Restore context
        session.last_dataset = data.get("last_dataset")
        session.last_target_col = data.get("last_target_col")
        session.last_model = data.get("last_model")
        session.last_task_type = data.get("last_task_type")
        session.best_score = data.get("best_score")
        session.last_output_files = data.get("last_output_files", {})
        session.workflow_history = data.get("workflow_history", [])
        session.conversation_context = data.get("conversation_context", [])
        session.last_tool_results = data.get("last_tool_results", {})
        
        return session
    
    def clear(self):
        """Clear all session context (start fresh)."""
        self.last_dataset = None
        self.last_target_col = None
        self.last_model = None
        self.last_task_type = None
        self.best_score = None
        self.last_output_files = {}
        self.workflow_history = []
        self.conversation_context = []
        self.last_tool_results = {}
        self.last_active = datetime.now()
