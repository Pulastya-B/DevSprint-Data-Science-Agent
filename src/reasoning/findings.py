"""
Findings Accumulator - Core state for the Reasoning Loop.

Tracks everything discovered during investigation:
- Individual findings (action + result + interpretation)
- Hypotheses being tested
- Decision ledger (why each action was taken)
- Confidence tracking

This replaces the need for separate "step tracker" and "decision ledger" -
they're natural byproducts of the accumulated findings.

Architecture:
    ReasoningLoop iteration 1: Reason → Act → Evaluate → Finding #1
    ReasoningLoop iteration 2: Reason → Act → Evaluate → Finding #2
    ...
    Synthesizer reads all findings → produces final answer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class Finding:
    """A single finding from one reasoning loop iteration."""
    iteration: int
    hypothesis: str                # What we were testing
    action: str                    # Tool name executed
    arguments: Dict[str, Any]      # Tool arguments used
    result_summary: str            # Compressed result (what tool returned)
    interpretation: str            # What we learned from this result
    confidence: float              # 0.0-1.0 confidence in this finding
    answered_question: bool        # Did this iteration answer the user's question?
    next_questions: List[str]      # Follow-up questions generated
    success: bool = True           # Whether the tool execution succeeded
    error_message: str = ""        # Error message if tool failed
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "hypothesis": self.hypothesis,
            "action": self.action,
            "arguments": self.arguments,
            "result_summary": self.result_summary,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "answered": self.answered_question,
            "next_questions": self.next_questions,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


@dataclass
class Hypothesis:
    """A hypothesis being tested during exploration."""
    text: str
    status: str = "untested"      # untested, testing, supported, refuted, inconclusive
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    priority: float = 0.5         # 0.0-1.0, higher = investigate first
    source_iteration: int = 0     # Which iteration generated this hypothesis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "priority": self.priority,
            "source_iteration": self.source_iteration
        }


class FindingsAccumulator:
    """
    Accumulates findings across the reasoning loop.
    
    This is the central state object that the Reasoner reads from and
    the Evaluator writes to. It serves as:
    - Step tracker (each finding records what was done)
    - Decision ledger (each finding records WHY it was done)
    - Evidence accumulator (interpretations build the answer)
    - Hypothesis manager (for exploratory analysis)
    
    Usage:
        findings = FindingsAccumulator(question="Why are customers churning?")
        
        # After each iteration:
        findings.add_finding(Finding(
            iteration=1,
            hypothesis="High churn correlates with low engagement",
            action="analyze_correlations",
            arguments={"file_path": "data.csv", "target_col": "churn"},
            result_summary="Found 0.72 correlation between login_frequency and churn",
            interpretation="Strong evidence: infrequent logins predict churn",
            confidence=0.8,
            answered_question=False,
            next_questions=["Is there a threshold for login frequency?"]
        ))
        
        # For the Reasoner prompt:
        context = findings.get_context_for_reasoning()
        
        # For the Synthesizer:
        all_findings = findings.get_all_findings()
    """

    def __init__(self, question: str, mode: str = "investigative"):
        """
        Initialize findings accumulator.
        
        Args:
            question: The user's original question
            mode: "investigative" or "exploratory"
        """
        self.question = question
        self.mode = mode
        self.findings: List[Finding] = []
        self.hypotheses: List[Hypothesis] = []
        self.tools_used: List[str] = []
        self.tools_with_args: List[Dict[str, Any]] = []  # Track tool+args to detect repeats
        self.files_produced: List[str] = []
        self.failed_tools: Dict[str, str] = {}  # tool_name → error message
        self.is_answered = False
        self.answer_confidence = 0.0
        self.started_at = datetime.now().isoformat()

    @property
    def iteration_count(self) -> int:
        """Number of completed iterations."""
        return len(self.findings)

    def add_finding(self, finding: Finding):
        """Add a finding from a completed iteration."""
        self.findings.append(finding)
        
        if finding.action not in self.tools_used:
            self.tools_used.append(finding.action)
        
        # Track tool+args for duplicate detection
        self.tools_with_args.append({
            "tool": finding.action,
            "args_key": json.dumps(finding.arguments, sort_keys=True, default=str)
        })
        
        # Track answer progress
        if finding.answered_question:
            self.is_answered = True
            self.answer_confidence = max(self.answer_confidence, finding.confidence)
        
        # Add new hypotheses from next_questions
        for q in finding.next_questions:
            if not any(h.text == q for h in self.hypotheses):
                self.hypotheses.append(Hypothesis(
                    text=q,
                    status="untested",
                    priority=0.5,
                    source_iteration=finding.iteration
                ))

    def add_failed_tool(self, tool_name: str, error_message: str):
        """Record a tool that failed so the Reasoner avoids retrying it."""
        self.failed_tools[tool_name] = error_message

    def get_failed_tools_context(self) -> str:
        """Build context string listing tools that failed."""
        if not self.failed_tools:
            return ""
        parts = ["\n**FAILED TOOLS (do NOT retry these)**:"]
        for tool, error in self.failed_tools.items():
            parts.append(f"  - `{tool}`: {error[:150]}")
        return "\n".join(parts)

    def get_successful_findings(self) -> List[Finding]:
        """Return only findings from successful tool executions."""
        return [f for f in self.findings if f.success]

    def add_hypothesis(self, text: str, priority: float = 0.5, source_iteration: int = 0):
        """Add a hypothesis to test."""
        if not any(h.text == text for h in self.hypotheses):
            self.hypotheses.append(Hypothesis(
                text=text,
                status="untested",
                priority=priority,
                source_iteration=source_iteration
            ))

    def update_hypothesis(self, text: str, status: str, evidence: str, is_supporting: bool = True):
        """Update a hypothesis with new evidence."""
        for h in self.hypotheses:
            if h.text == text:
                h.status = status
                if is_supporting:
                    h.evidence_for.append(evidence)
                else:
                    h.evidence_against.append(evidence)
                return
    
    def get_untested_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that haven't been tested yet, sorted by priority."""
        untested = [h for h in self.hypotheses if h.status == "untested"]
        return sorted(untested, key=lambda h: h.priority, reverse=True)

    def get_last_output_file(self) -> Optional[str]:
        """Get the most recent output file from tool results."""
        for finding in reversed(self.findings):
            # Check if result mentions an output file
            result = finding.result_summary
            if "output_file" in result or "output_path" in result:
                try:
                    # Try to parse as JSON
                    result_dict = json.loads(result) if isinstance(result, str) else result
                    return result_dict.get("output_file") or result_dict.get("output_path")
                except (json.JSONDecodeError, TypeError):
                    pass
            # Check arguments for file paths
            for key in ["file_path", "input_path"]:
                if key in finding.arguments:
                    return finding.arguments[key]
        return None

    def get_context_for_reasoning(self, max_findings: int = 5) -> str:
        """
        Build context string for the Reasoner's prompt.
        
        Returns a concise summary of what's been discovered so far,
        formatted for LLM consumption.
        
        Args:
            max_findings: Maximum number of recent findings to include
        """
        if not self.findings:
            return "No investigations completed yet. This is the first step."

        parts = []
        
        # Summary of what's been done
        parts.append(f"**Investigations completed**: {len(self.findings)}")
        parts.append(f"**Tools used**: {', '.join(self.tools_used)}")
        
        # Warn about tools already called (with args) to prevent repeats
        if self.tools_with_args:
            seen = [f"`{t['tool']}`" for t in self.tools_with_args]
            parts.append(f"**Tools already called (DO NOT repeat with same args)**: {', '.join(seen)}")
        
        # Failed tools warning (critical for avoiding retries)
        failed_ctx = self.get_failed_tools_context()
        if failed_ctx:
            parts.append(failed_ctx)
        
        # Recent findings (most relevant for next decision)
        recent = self.findings[-max_findings:]
        parts.append("\n**Recent findings**:")
        for f in recent:
            status_tag = "" if f.success else " [FAILED]"
            parts.append(
                f"  Step {f.iteration}: Ran `{f.action}`{status_tag} to test: \"{f.hypothesis}\"\n"
                f"    → Result: {f.interpretation}\n"
                f"    → Confidence: {f.confidence:.0%}"
            )
        
        # Unanswered questions
        untested = self.get_untested_hypotheses()
        if untested:
            parts.append(f"\n**Open questions** ({len(untested)} remaining):")
            for h in untested[:3]:
                parts.append(f"  - {h.text} (priority: {h.priority:.1f})")
        
        # Overall progress
        if self.is_answered:
            parts.append(f"\n**Status**: Question partially answered (confidence: {self.answer_confidence:.0%})")
        else:
            parts.append(f"\n**Status**: Still investigating")
        
        return "\n".join(parts)

    def get_context_for_synthesis(self) -> str:
        """
        Build context string for the Synthesizer.
        
        Returns the complete investigative history with all findings
        and hypothesis statuses.
        """
        parts = []
        
        parts.append(f"**Original question**: {self.question}")
        parts.append(f"**Mode**: {self.mode}")
        parts.append(f"**Total iterations**: {len(self.findings)}")
        parts.append(f"**Tools used**: {', '.join(self.tools_used)}")
        
        # All findings in order
        parts.append("\n## Investigation Steps\n")
        for f in self.findings:
            status_label = "\u2705 SUCCESS" if f.success else "\u274c FAILED"
            parts.append(
                f"### Step {f.iteration}: {f.action} [{status_label}]\n"
                f"**Hypothesis**: {f.hypothesis}\n"
                f"**Arguments**: {json.dumps(f.arguments, default=str)}\n"
                f"**Result**: {f.result_summary}\n"
                f"**Interpretation**: {f.interpretation}\n"
                f"**Confidence**: {f.confidence:.0%}\n"
            )
        
        # Hypothesis outcomes (for exploratory mode)
        if self.hypotheses:
            parts.append("\n## Hypothesis Outcomes\n")
            for h in self.hypotheses:
                status_emoji = {
                    "supported": "✅",
                    "refuted": "❌",
                    "inconclusive": "❓",
                    "testing": "🔄",
                    "untested": "⬜"
                }.get(h.status, "⬜")
                
                parts.append(f"{status_emoji} **{h.text}** → {h.status}")
                if h.evidence_for:
                    parts.append(f"  Evidence for: {'; '.join(h.evidence_for)}")
                if h.evidence_against:
                    parts.append(f"  Evidence against: {'; '.join(h.evidence_against)}")
        
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response / session storage."""
        return {
            "question": self.question,
            "mode": self.mode,
            "iteration_count": self.iteration_count,
            "is_answered": self.is_answered,
            "answer_confidence": self.answer_confidence,
            "tools_used": self.tools_used,
            "files_produced": self.files_produced,
            "findings": [f.to_dict() for f in self.findings],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "started_at": self.started_at
        }
