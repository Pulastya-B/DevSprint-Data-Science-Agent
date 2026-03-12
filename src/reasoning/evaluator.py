"""
Evaluator Module - The EVALUATE step of the Reasoning Loop.

Interprets tool results and decides:
- What did we learn from this action?
- Does this answer the user's question?
- Should we continue investigating or stop?
- What follow-up questions emerged?

The Evaluator transforms raw tool output into understanding.

Architecture:
    Tool Result → Evaluator.evaluate() → EvaluationOutput
        - interpretation: natural language explanation
        - answered: did this answer the question?
        - confidence: how confident are we?
        - should_stop: should the loop stop?
        - next_questions: what to investigate next
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from .findings import Finding, FindingsAccumulator


@dataclass
class EvaluationOutput:
    """Output from one EVALUATE step."""
    interpretation: str            # What we learned from the tool result
    answered: bool                 # Does this answer the user's question?
    confidence: float              # 0.0-1.0 confidence
    should_stop: bool              # Should the reasoning loop stop?
    next_questions: List[str]      # Follow-up questions to investigate
    key_metric: Optional[str] = None  # Most important metric extracted


EVALUATOR_SYSTEM_PROMPT = """You are a senior data scientist interpreting analysis results.

Your job:
1. Interpret what the tool result MEANS (not just what it shows)
2. Decide if this answers the user's original question
3. Identify follow-up questions worth investigating
4. Assign confidence level to your interpretation

Be concise but insightful. Focus on:
- Statistical significance (not just numbers)
- Business implications (not just patterns)
- Confounders and caveats
- What's surprising vs expected

GROUNDING RULES:
- ONLY reference numbers and statistics that appear in the tool result below — do NOT invent values
- Do NOT claim data quality issues unless the tool explicitly reports them
- BUT DO deeply interpret the ACTUAL data: explain what the real min/max/mean/distributions MEAN
  • If you see column stats, analyze the spread, skewness, and domain implications of the real values
  • If you see correlations, explain their practical significance — not just "correlated" but what it implies
  • Derive rich insights FROM the actual data rather than fabricating data to support an insight
- If the tool result lacks data for a specific claim, say "insufficient data" — but DO thoroughly analyze what IS there

IMPORTANT CONFIDENCE RULES:
- If the tool returned feature_scores, feature_importance, or correlation values, and the user asked about features/importance/correlations → this IS the answer. Set answered=true, confidence ≥ 0.7.
- If the tool returned actual ranked data (top features, sorted scores, correlation pairs), set confidence ≥ 0.6.
- Do NOT keep saying "not answered" when the tool literally returned the requested information.
- Only say answered=false when the result is genuinely unrelated to the question or contains NO useful data.

CRITICAL: Output ONLY valid JSON, no other text."""

EVALUATOR_USER_TEMPLATE = """**User's original question**: {question}

**Action taken**: {tool_name}({arguments})

**Tool result** (compressed):
{result_summary}

**What we knew before this step**:
{prior_findings}

Evaluate this result. Respond with ONLY this JSON:
{{
  "interpretation": "1-3 sentences: What does this result MEAN for answering the question?",
  "answered": true/false,
  "confidence": 0.0-1.0,
  "should_stop": true/false,
  "next_questions": ["follow-up question 1", "follow-up question 2"],
  "key_metric": "most important number or finding (optional)"
}}

Guidelines for should_stop:
- true: Question is fully answered OR we've gathered enough evidence OR no more useful actions
- false: Important aspects remain uninvestigated

Guidelines for answered:
- true: The result contains data that directly addresses the user's question (e.g., feature scores for "which features are important?", correlations for "what correlates with X?")
- false: Result is unrelated to the question or contains only metadata without actual answers

Guidelines for confidence:
- 0.0-0.3: Weak evidence, need more investigation
- 0.3-0.6: Moderate evidence, some aspects unclear
- 0.6-0.8: Strong evidence, minor questions remain (e.g., got feature importance scores but could add more context)
- 0.8-1.0: Very strong evidence, question well answered (e.g., got ranked feature list with scores AND correlations)"""


class Evaluator:
    """
    The EVALUATE step of the Reasoning Loop.
    
    Takes a tool result and interprets it in the context of
    the user's question and prior findings.
    
    Usage:
        evaluator = Evaluator(llm_caller=orchestrator._llm_text_call)
        evaluation = evaluator.evaluate(
            question="Why are customers churning?",
            tool_name="analyze_correlations",
            arguments={"file_path": "data.csv", "target_col": "churn"},
            result=tool_result,
            findings=findings_accumulator
        )
        
        if evaluation.should_stop:
            # Move to synthesis
            ...
        else:
            # Continue reasoning loop
            ...
    """
    
    def __init__(self, llm_caller: Callable):
        """
        Args:
            llm_caller: Function (system_prompt, user_prompt, max_tokens) -> str
        """
        self.llm_caller = llm_caller

    def evaluate(
        self,
        question: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
        findings: FindingsAccumulator,
        result_compressor: Optional[Callable] = None
    ) -> EvaluationOutput:
        """
        Evaluate a tool result.
        
        Args:
            question: User's original question
            tool_name: Name of the tool that was executed
            arguments: Tool arguments used
            result: Raw tool result dict
            findings: Accumulated findings so far
            result_compressor: Optional function to compress tool results
            
        Returns:
            EvaluationOutput with interpretation and next steps
        """
        # Compress the result for LLM consumption
        if result_compressor:
            result_summary = json.dumps(result_compressor(tool_name, result), default=str)
        else:
            result_summary = self._default_compress(result)
        
        # Truncate if too long — use generous limit to preserve evidence
        if len(result_summary) > 6000:
            result_summary = result_summary[:6000] + "... [truncated]"
        
        # Build argument string
        args_str = json.dumps(arguments, default=str)
        if len(args_str) > 500:
            args_str = args_str[:500] + "..."
        
        user_prompt = EVALUATOR_USER_TEMPLATE.format(
            question=question,
            tool_name=tool_name,
            arguments=args_str,
            result_summary=result_summary,
            prior_findings=findings.get_context_for_reasoning(max_findings=3)
        )
        
        response_text = self.llm_caller(
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return self._parse_response(response_text, result_summary)

    def build_finding(
        self,
        iteration: int,
        hypothesis: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result_summary: str,
        evaluation: "EvaluationOutput",
        success: bool = True,
        error_message: str = ""
    ) -> Finding:
        """
        Build a Finding from a completed iteration.
        
        Convenience method that combines the action and evaluation
        into a single Finding for the accumulator.
        """
        return Finding(
            iteration=iteration,
            hypothesis=hypothesis,
            action=tool_name,
            arguments=arguments,
            result_summary=result_summary[:3000],  # Cap size — preserve more evidence for synthesizer
            interpretation=evaluation.interpretation,
            confidence=evaluation.confidence if success else 0.0,
            answered_question=evaluation.answered if success else False,
            next_questions=evaluation.next_questions,
            success=success,
            error_message=error_message
        )

    def _parse_response(self, response_text: str, result_summary: str) -> EvaluationOutput:
        """Parse LLM response into EvaluationOutput."""
        try:
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    return self._fallback_evaluation(response_text, result_summary)
            else:
                return self._fallback_evaluation(response_text, result_summary)
        
        return EvaluationOutput(
            interpretation=data.get("interpretation", "Result processed."),
            answered=data.get("answered", False),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.3)))),
            should_stop=data.get("should_stop", False),
            next_questions=data.get("next_questions", []),
            key_metric=data.get("key_metric")
        )

    def _fallback_evaluation(self, response_text: str, result_summary: str) -> EvaluationOutput:
        """Fallback when JSON parsing fails.
        
        Uses the raw tool result to generate a basic interpretation rather than
        returning a zombie low-confidence result that wastes iterations.
        """
        interpretation = response_text.strip()[:500] if response_text else "Analysis step completed."
        
        # Instead of low-confidence zombie (0.3 + should_stop=False which burns iterations),
        # use moderate confidence and stop=True so the loop doesn't waste cycles
        # on broken evaluation output. The raw result is still preserved in findings.
        return EvaluationOutput(
            interpretation=f"[Evaluation parsing failed — raw interpretation] {interpretation}",
            answered=False,
            confidence=0.4,
            should_stop=True,
            next_questions=[],
            key_metric=None
        )

    def _default_compress(self, result: Dict[str, Any]) -> str:
        """Default compression for tool results."""
        if not isinstance(result, dict):
            return str(result)[:2000]
        
        compressed = {}
        
        # Always include status
        if "success" in result:
            compressed["success"] = result["success"]
        if "error" in result:
            compressed["error"] = str(result["error"])[:300]
        
        # Include key result fields
        result_data = result.get("result", result)
        if isinstance(result_data, dict):
            for key in ["num_rows", "num_columns", "missing_percentage", "task_type",
                        "best_model", "best_score", "models", "correlations",
                        "output_file", "output_path", "plots", "summary",
                        "total_issues", "columns_affected", "features_created",
                        "accuracy", "r2_score", "rmse", "f1_score"]:
                if key in result_data:
                    value = result_data[key]
                    # Truncate long values
                    if isinstance(value, (list, dict)):
                        compressed[key] = str(value)[:500]
                    else:
                        compressed[key] = value
        
        return json.dumps(compressed, default=str)
