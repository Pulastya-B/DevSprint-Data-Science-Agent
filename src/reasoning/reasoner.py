"""
Reasoner Module - The REASON step of the Reasoning Loop.

Decides what to investigate next based on:
- The user's original question
- What we've discovered so far (findings)
- Available tools
- Dataset schema

The Reasoner does NOT execute anything. It only produces a structured
decision about what action to take next.

Architecture:
    Reasoner.reason() → ReasoningOutput
        - status: "investigating" | "done"  
        - reasoning: why this action (decision ledger entry)
        - tool_name: which tool to run
        - arguments: tool arguments
        - hypothesis: what we're testing

This replaces the old approach where a massive system prompt told the LLM
"follow steps 1-15." Instead, the Reasoner makes a strategic decision
each iteration based on what it's learned so far.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

from .findings import FindingsAccumulator


@dataclass
class ReasoningOutput:
    """Output from one REASON step."""
    status: str                    # "investigating" or "done"
    reasoning: str                 # Why this action was chosen
    tool_name: Optional[str]       # Tool to execute (None if done)
    arguments: Dict[str, Any]      # Tool arguments
    hypothesis: str                # What we're testing with this action
    confidence: float = 0.0        # How confident the reasoner is (0-1)
    
    @classmethod
    def done(cls, reasoning: str, confidence: float = 0.8) -> "ReasoningOutput":
        """Create a 'done' output (no more investigation needed)."""
        return cls(
            status="done",
            reasoning=reasoning,
            tool_name=None,
            arguments={},
            hypothesis="",
            confidence=confidence
        )


# System prompt for the Reasoner LLM call
REASONER_SYSTEM_PROMPT = """You are a senior data scientist. Your job is to decide the SINGLE MOST IMPORTANT next investigation step.

You are given:
1. The user's question
2. What has been discovered so far
3. The dataset schema
4. Available tools

Your task: Decide ONE action to take next. Be strategic:
- Start with understanding (profiling, correlations) before acting
- Test the most impactful hypothesis first
- Don't repeat actions that have already been done
- Stop when you have enough evidence to answer the question confidently

CRITICAL RULES:
- Output ONLY valid JSON, no other text
- Use EXACT tool names from the available tools list
- Use EXACT column names from the dataset schema
- For the file_path argument, ALWAYS use the ORIGINAL DATA FILE path (the CSV/parquet that was uploaded), NOT any output artifact paths (HTML reports, plots, etc.)
- If a previous tool produced a new data file (CSV/parquet), use THAT as file_path
- NEVER use an HTML, PNG, or report path as file_path for data-consuming tools
- For visualization, pick the chart type that best answers the question
- NEVER hallucinate column names - use only columns from the schema

TOOL FAILURE RULES:
- NEVER retry a tool that has already FAILED — try a DIFFERENT tool or approach instead
- If the "FAILED TOOLS" section lists a tool, that tool WILL fail again — do not call it
- If multiple tools have failed, consider stopping and synthesizing what you have

QUERY TYPE AWARENESS:
- For questions about "important features", "feature importance", "correlations", "patterns", or "explain the data":
  Use EDA tools (profile_dataset, analyze_correlations, auto_feature_selection, generate_eda_plots)
  Do NOT use model training tools (train_with_autogluon, train_model, etc.) — training is unnecessary for feature explanation
- Only use model training tools when the user explicitly asks to train, predict, build a model, or classify/regress"""

REASONER_USER_TEMPLATE = """**User's question**: {question}

**Dataset info**:
- Original data file (use this for file_path): {file_path}
- Rows: {num_rows:,} | Columns: {num_columns}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}
{target_info}

**Investigation so far**:
{findings_context}

**Available tools**:
{tools_description}

IMPORTANT:
- For ANY tool that needs a file_path argument, use "{file_path}" — the original data file. Do NOT use paths to HTML reports, plots, or other output artifacts.
- If a tool is listed under FAILED TOOLS above, do NOT call it again — it will fail. Choose a different tool or stop.

Decide the next action. Respond with ONLY this JSON:
{{
  "status": "investigating" or "done",
  "reasoning": "1-2 sentence explanation of why this action is needed",
  "tool_name": "exact_tool_name",
  "arguments": {{"arg1": "value1", "arg2": "value2"}},
  "hypothesis": "what we expect to learn from this action"
}}

If you have enough evidence to answer the user's question, respond:
{{
  "status": "done",
  "reasoning": "We have sufficient evidence because...",
  "tool_name": null,
  "arguments": {{}},
  "hypothesis": ""
}}"""


# System prompt for generating hypotheses (Exploratory mode)
HYPOTHESIS_SYSTEM_PROMPT = """You are a senior data scientist examining a dataset for the first time.
Given the dataset profile, generate 3-5 hypotheses worth investigating.

Focus on:
- Relationships between columns that could explain the target variable
- Which features might have the strongest predictive power
- Distribution patterns visible in the actual column stats (min/max/mean/median)
- Potential feature interactions worth exploring

Frame each hypothesis as a TESTABLE QUESTION (e.g., "Does Distance_to_Sink correlate with energy depletion?"), NOT as an assumed conclusion (e.g., "There is a distance penalty beyond 75m").
Base hypotheses on column names and any profile stats provided — do NOT assume anomalies or quality issues before seeing the data.

Output ONLY valid JSON array of hypotheses, ranked by priority (most interesting first)."""

HYPOTHESIS_USER_TEMPLATE = """**Dataset**: {file_path}
- Rows: {num_rows:,} | Columns: {num_columns}
- Numeric: {numeric_columns}
- Categorical: {categorical_columns}
{target_info}
{profile_summary}

Generate hypotheses as JSON:
[
  {{"text": "hypothesis description", "priority": 0.9, "suggested_tool": "tool_name"}},
  ...
]"""


class Reasoner:
    """
    The REASON step of the Reasoning Loop.
    
    Makes a strategic decision about what to investigate next,
    based on the user's question and accumulated findings.
    
    Usage:
        reasoner = Reasoner(llm_caller=orchestrator._llm_text_call)
        output = reasoner.reason(
            question="Why are customers churning?",
            dataset_info=schema_info,
            findings=findings_accumulator,
            available_tools=tools_description,
            file_path="data.csv"
        )
        
        if output.status == "investigating":
            result = execute_tool(output.tool_name, output.arguments)
        else:
            # Done investigating, synthesize answer
            ...
    """
    
    def __init__(self, llm_caller: Callable):
        """
        Args:
            llm_caller: Function (system_prompt, user_prompt, max_tokens) -> str
                        Wraps the orchestrator's provider-specific LLM call.
        """
        self.llm_caller = llm_caller

    def reason(
        self,
        question: str,
        dataset_info: Dict[str, Any],
        findings: FindingsAccumulator,
        available_tools: str,
        file_path: str,
        target_col: Optional[str] = None
    ) -> ReasoningOutput:
        """
        Decide the next investigation step.
        
        Args:
            question: User's original question
            dataset_info: Dataset schema (columns, types, stats)
            findings: Accumulated findings from previous iterations
            available_tools: Text description of available tools
            file_path: Current file path (latest output or original)
            target_col: Optional target column
            
        Returns:
            ReasoningOutput with the next action to take
        """
        # Build the user prompt
        numeric_cols = dataset_info.get("numeric_columns", [])
        categorical_cols = dataset_info.get("categorical_columns", [])
        
        target_info = ""
        if target_col:
            target_info = f"- Target column: '{target_col}'"
        
        user_prompt = REASONER_USER_TEMPLATE.format(
            question=question,
            file_path=file_path,
            num_rows=dataset_info.get("num_rows", 0),
            num_columns=dataset_info.get("num_columns", 0),
            numeric_columns=", ".join([f"'{c}'" for c in numeric_cols[:15]]),
            categorical_columns=", ".join([f"'{c}'" for c in categorical_cols[:15]]),
            target_info=target_info,
            findings_context=findings.get_context_for_reasoning(),
            tools_description=available_tools
        )
        
        # Call LLM
        response_text = self.llm_caller(
            system_prompt=REASONER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        # Parse response (pass findings so we can reject failed tools)
        return self._parse_response(response_text, file_path, findings)

    def generate_hypotheses(
        self,
        dataset_info: Dict[str, Any],
        file_path: str,
        target_col: Optional[str] = None,
        profile_summary: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Generate hypotheses for exploratory analysis.
        
        Called at the start of Exploratory mode to seed the
        reasoning loop with interesting questions to investigate.
        
        Args:
            dataset_info: Dataset schema
            file_path: Path to dataset
            target_col: Optional target column
            profile_summary: Optional profiling results summary
            
        Returns:
            List of hypothesis dicts with text, priority, suggested_tool
        """
        numeric_cols = dataset_info.get("numeric_columns", [])
        categorical_cols = dataset_info.get("categorical_columns", [])
        
        target_info = ""
        if target_col:
            target_info = f"- Target column: '{target_col}'"
        
        user_prompt = HYPOTHESIS_USER_TEMPLATE.format(
            file_path=file_path,
            num_rows=dataset_info.get("num_rows", 0),
            num_columns=dataset_info.get("num_columns", 0),
            numeric_columns=", ".join([f"'{c}'" for c in numeric_cols[:15]]),
            categorical_columns=", ".join([f"'{c}'" for c in categorical_cols[:15]]),
            target_info=target_info,
            profile_summary=profile_summary or "No profile available yet."
        )
        
        response_text = self.llm_caller(
            system_prompt=HYPOTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=1024
        )
        
        return self._parse_hypotheses(response_text)

    def _parse_response(self, response_text: str, file_path: str, findings: Optional[FindingsAccumulator] = None) -> ReasoningOutput:
        """Parse LLM response into ReasoningOutput."""
        try:
            # Try direct JSON parse
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from markdown/text
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    # Fallback: return a profiling action
                    return ReasoningOutput(
                        status="investigating",
                        reasoning="Could not parse LLM response, defaulting to profiling",
                        tool_name="profile_dataset",
                        arguments={"file_path": file_path},
                        hypothesis="Understanding the data structure first"
                    )
            else:
                return ReasoningOutput(
                    status="investigating",
                    reasoning="Could not parse LLM response, defaulting to profiling",
                    tool_name="profile_dataset",
                    arguments={"file_path": file_path},
                    hypothesis="Understanding the data structure first"
                )
        
        status = data.get("status", "investigating")
        tool_name = data.get("tool_name")
        arguments = data.get("arguments", {})
        
        # 🛡️ SAFETY: If LLM says "investigating" but provides no tool, treat as "done"
        # This prevents wasting iterations on empty responses (seen in logs: 5 consecutive skips)
        if status == "investigating" and not tool_name:
            print(f"   ⚠️  Reasoner returned 'investigating' with no tool — forcing done")
            return ReasoningOutput.done(
                reasoning=data.get("reasoning", "No further tool selected. Synthesizing available findings."),
                confidence=max(0.4, float(data.get("confidence", 0.4)))
            )
        
        # Ensure file_path is in arguments if tool needs it
        if tool_name and "file_path" not in arguments and tool_name not in [
            "execute_python_code", "get_smart_summary"
        ]:
            arguments["file_path"] = file_path
        
        # 🛡️ SAFETY: Override file_path if LLM picked a non-data file (HTML, PNG, etc.)
        if "file_path" in arguments:
            fp = arguments["file_path"]
            non_data_extensions = ('.html', '.png', '.jpg', '.jpeg', '.svg', '.gif', '.pdf')
            if fp.lower().endswith(non_data_extensions):
                arguments["file_path"] = file_path
        
        # 🛡️ SAFETY: Reject tools that already failed — force "done" to stop wasting iterations
        if tool_name and findings and tool_name in findings.failed_tools:
            print(f"   ⚠️  Reasoner picked failed tool '{tool_name}' — forcing done")
            return ReasoningOutput.done(
                reasoning=f"Tool '{tool_name}' previously failed. Stopping to synthesize available findings.",
                confidence=max(0.3, findings.answer_confidence)
            )
        
        return ReasoningOutput(
            status=status,
            reasoning=data.get("reasoning", ""),
            tool_name=tool_name if status == "investigating" else None,
            arguments=arguments,
            hypothesis=data.get("hypothesis", ""),
            confidence=data.get("confidence", 0.5)
        )

    def _parse_hypotheses(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse hypothesis generation response."""
        try:
            data = json.loads(response_text.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array
        array_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        # Fallback: generate basic hypotheses
        return [
            {"text": "What are the key statistical properties of this dataset?", "priority": 0.9, "suggested_tool": "profile_dataset"},
            {"text": "Are there any significant correlations between variables?", "priority": 0.8, "suggested_tool": "analyze_correlations"},
            {"text": "What does the distribution of key variables look like?", "priority": 0.7, "suggested_tool": "generate_eda_plots"}
        ]
