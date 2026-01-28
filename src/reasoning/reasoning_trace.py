"""
Reasoning Trace Module

Captures decision-making process for transparency and debugging.
Provides audit trail of why certain tools/agents were chosen.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class ReasoningTrace:
    """
    Records reasoning decisions made during workflow execution.
    
    Provides transparency into:
    - Why specific agents were selected
    - Why certain tools were chosen
    - What alternatives were considered
    - Decision confidence levels
    """
    
    def __init__(self):
        self.trace_history: List[Dict[str, Any]] = []
        self.current_context = {}
    
    def record_agent_selection(self, task: str, selected_agent: str, 
                              confidence: float, alternatives: Dict[str, float] = None):
        """
        Record why a specific agent was selected.
        
        Args:
            task: User's task description
            selected_agent: Agent that was selected
            confidence: Confidence score (0-1)
            alternatives: Other agents considered with their scores
        """
        decision = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_selection",
            "task": task,
            "decision": selected_agent,
            "confidence": confidence,
            "alternatives": alternatives or {},
            "reasoning": self._explain_agent_selection(task, selected_agent, confidence)
        }
        
        self.trace_history.append(decision)
        print(f"📝 Reasoning: Selected {selected_agent} (confidence: {confidence:.2f})")
    
    def record_tool_selection(self, tool_name: str, args: Dict[str, Any], 
                             reason: str, iteration: int):
        """
        Record why a specific tool was chosen.
        
        Args:
            tool_name: Tool that was selected
            args: Arguments passed to tool
            reason: Human-readable reason for selection
            iteration: Current workflow iteration
        """
        decision = {
            "timestamp": datetime.now().isoformat(),
            "type": "tool_selection",
            "iteration": iteration,
            "tool": tool_name,
            "arguments": self._sanitize_args(args),
            "reason": reason
        }
        
        self.trace_history.append(decision)
    
    def record_agent_handoff(self, from_agent: str, to_agent: str, 
                            reason: str, iteration: int):
        """
        Record agent hand-off decision.
        
        Args:
            from_agent: Previous agent
            to_agent: New agent
            reason: Why hand-off occurred
            iteration: Current workflow iteration
        """
        decision = {
            "timestamp": datetime.now().isoformat(),
            "type": "agent_handoff",
            "iteration": iteration,
            "from": from_agent,
            "to": to_agent,
            "reason": reason
        }
        
        self.trace_history.append(decision)
        print(f"📝 Reasoning: Hand-off {from_agent} → {to_agent} - {reason}")
    
    def record_decision_point(self, decision_type: str, options: List[str], 
                             chosen: str, reason: str):
        """
        Record a general decision point.
        
        Args:
            decision_type: Type of decision (e.g., "feature_selection", "model_type")
            options: Options that were available
            chosen: Option that was selected
            reason: Why this option was chosen
        """
        decision = {
            "timestamp": datetime.now().isoformat(),
            "type": decision_type,
            "options": options,
            "chosen": chosen,
            "reason": reason
        }
        
        self.trace_history.append(decision)
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get full reasoning trace."""
        return self.trace_history
    
    def get_trace_summary(self) -> str:
        """
        Get human-readable summary of reasoning trace.
        
        Returns:
            Formatted string summarizing all decisions
        """
        if not self.trace_history:
            return "No reasoning trace available."
        
        summary_parts = ["## Reasoning Trace\n"]
        
        for i, decision in enumerate(self.trace_history, 1):
            decision_type = decision.get("type", "unknown")
            timestamp = decision.get("timestamp", "")
            
            if decision_type == "agent_selection":
                summary_parts.append(
                    f"{i}. **Agent Selection** ({timestamp})\n"
                    f"   - Selected: {decision.get('decision')}\n"
                    f"   - Confidence: {decision.get('confidence', 0):.2f}\n"
                    f"   - Reasoning: {decision.get('reasoning', 'N/A')}\n"
                )
            
            elif decision_type == "tool_selection":
                summary_parts.append(
                    f"{i}. **Tool Execution** (Iteration {decision.get('iteration')})\n"
                    f"   - Tool: {decision.get('tool')}\n"
                    f"   - Reason: {decision.get('reason', 'N/A')}\n"
                )
            
            elif decision_type == "agent_handoff":
                summary_parts.append(
                    f"{i}. **Agent Hand-off** (Iteration {decision.get('iteration')})\n"
                    f"   - From: {decision.get('from')}\n"
                    f"   - To: {decision.get('to')}\n"
                    f"   - Reason: {decision.get('reason', 'N/A')}\n"
                )
            
            else:
                summary_parts.append(
                    f"{i}. **{decision_type}** ({timestamp})\n"
                    f"   - Chosen: {decision.get('chosen', 'N/A')}\n"
                    f"   - Reason: {decision.get('reason', 'N/A')}\n"
                )
        
        return "\n".join(summary_parts)
    
    def export_trace(self, file_path: str = "reasoning_trace.json"):
        """
        Export reasoning trace to JSON file.
        
        Args:
            file_path: Path to save trace file
        """
        with open(file_path, 'w') as f:
            json.dump(self.trace_history, f, indent=2)
        
        print(f"📄 Reasoning trace exported to {file_path}")
    
    def _explain_agent_selection(self, task: str, agent: str, confidence: float) -> str:
        """Generate explanation for agent selection."""
        if confidence > 0.9:
            certainty = "High confidence"
        elif confidence > 0.7:
            certainty = "Moderate confidence"
        else:
            certainty = "Low confidence"
        
        agent_explanations = {
            "data_quality_agent": "Task involves data profiling, quality assessment, or initial exploration",
            "preprocessing_agent": "Task requires data cleaning, transformation, or feature engineering",
            "visualization_agent": "Task focuses on creating visualizations, charts, or dashboards",
            "modeling_agent": "Task involves machine learning model training or evaluation",
            "time_series_agent": "Task involves time series analysis, forecasting, or temporal patterns",
            "nlp_agent": "Task involves text processing, sentiment analysis, or NLP operations",
            "business_intelligence_agent": "Task requires business metrics, KPIs, or strategic insights",
            "production_agent": "Task involves model deployment, monitoring, or production operations"
        }
        
        explanation = agent_explanations.get(
            agent, 
            "Selected based on task keywords and context"
        )
        
        return f"{certainty}: {explanation}"
    
    def _sanitize_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from arguments before logging."""
        sanitized = {}
        
        for key, value in args.items():
            if key in ["api_key", "password", "token", "secret"]:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, str) and len(value) > 100:
                sanitized[key] = value[:97] + "..."
            else:
                sanitized[key] = value
        
        return sanitized


# Global reasoning trace instance
_reasoning_trace = None


def get_reasoning_trace() -> ReasoningTrace:
    """Get or create global reasoning trace instance."""
    global _reasoning_trace
    if _reasoning_trace is None:
        _reasoning_trace = ReasoningTrace()
    return _reasoning_trace


def reset_reasoning_trace():
    """Reset reasoning trace for new workflow."""
    global _reasoning_trace
    _reasoning_trace = ReasoningTrace()
