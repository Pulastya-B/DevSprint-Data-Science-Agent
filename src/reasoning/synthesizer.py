"""
Synthesizer Module - The SYNTHESIZE step of the Reasoning Loop.

Takes all accumulated findings and produces a coherent, narrative answer.

Unlike the old approach (where the LLM's last response WAS the summary),
the Synthesizer deliberately constructs the answer from evidence:
- Connects findings into a coherent story
- Cites evidence for each claim
- Highlights confidence levels
- Notes what wasn't investigated (limitations)
- Produces actionable insights, not just numbers

Architecture:
    FindingsAccumulator → Synthesizer.synthesize() → Markdown narrative
"""

import json
from typing import Dict, Any, List, Optional, Callable

from .findings import FindingsAccumulator


SYNTHESIS_SYSTEM_PROMPT = """You are a senior data scientist writing a concise analysis report.

Given the investigation findings, synthesize a clear, evidence-based answer to the user's question.

STRUCTURE (use markdown):
1. **Executive Summary** (2-3 sentences answering the question directly)
2. **Key Findings** (bullet points with evidence references)
3. **Supporting Evidence** (specific metrics, correlations, patterns)
4. **Visualizations** (mention any plots/charts generated, with file paths)
5. **Limitations & Caveats** (what we didn't investigate, caveats)
6. **Recommendations** (actionable next steps)

RULES:
- Lead with the answer, then show evidence
- ONLY cite numbers that appear VERBATIM in the findings below — do NOT round, invent thresholds, or fabricate statistics
- Do NOT claim data quality issues unless the tool results explicitly report them
- BUT DO deeply interpret what the ACTUAL data tells us:
  • Explain what real value ranges mean for the domain (e.g., "Distance_to_Sink spans 0.0–64.26m, indicating a moderately sized network")
  • Derive insights from actual distributions: compare min/median/mean/max to identify skewness, tight vs wide spreads
  • Explain the practical significance of correlation values (e.g., "r=0.825 between Energy and Alive suggests strongly coupled depletion")
  • Identify which features show the most variation and what that implies
  • Compare column ranges to draw cross-feature insights
- Mention generated files/plots so user can find them
- Be honest about confidence levels — if the data is insufficient for a conclusion, say so
- Keep it under 500 words unless complex analysis warrants more
- Use markdown formatting (headers, bullets, bold for emphasis)
- Do NOT wrap your response in code fences (``` or ```markdown) — output raw markdown directly
- ONLY report findings from SUCCESSFUL investigation steps
- If a step is marked [FAILED], ignore its results entirely — do not fabricate data from it
- If most steps failed, be transparent about limited evidence and recommend re-running"""

SYNTHESIS_USER_TEMPLATE = """**Original question**: {question}

**Investigation summary**:
{findings_context}

**Generated artifacts**:
{artifacts_summary}

Write the analysis report now. Focus on answering the question with evidence from the investigation."""


class Synthesizer:
    """
    The SYNTHESIZE step of the Reasoning Loop.
    
    Produces the final answer from accumulated evidence.
    
    Usage:
        synthesizer = Synthesizer(llm_caller=orchestrator._llm_text_call)
        report = synthesizer.synthesize(
            findings=findings_accumulator,
            artifacts={"plots": [...], "files": [...]}
        )
    """
    
    def __init__(self, llm_caller: Callable):
        """
        Args:
            llm_caller: Function (system_prompt, user_prompt, max_tokens) -> str
        """
        self.llm_caller = llm_caller

    def synthesize(
        self,
        findings: FindingsAccumulator,
        artifacts: Optional[Dict[str, Any]] = None,
        max_tokens: int = 3000
    ) -> str:
        """
        Synthesize all findings into a coherent answer.
        
        Args:
            findings: Accumulated findings from the reasoning loop
            artifacts: Optional dict of generated artifacts (plots, files, models)
            max_tokens: Max tokens for synthesis response
            
        Returns:
            Markdown-formatted analysis report
        """
        # Build artifacts summary
        artifacts_summary = self._format_artifacts(artifacts or {}, findings)
        
        # Build findings context — only successful findings get full detail
        findings_context = self._build_filtered_context(findings)
        
        user_prompt = SYNTHESIS_USER_TEMPLATE.format(
            question=findings.question,
            findings_context=findings_context,
            artifacts_summary=artifacts_summary
        )
        
        response = self.llm_caller(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )
        
        return self._strip_code_fences(response.strip())

    def synthesize_exploratory(
        self,
        findings: FindingsAccumulator,
        artifacts: Optional[Dict[str, Any]] = None,
        max_tokens: int = 3000
    ) -> str:
        """
        Synthesize findings from exploratory analysis (no specific question).
        
        Uses a different prompt that focuses on discovering patterns
        rather than answering a specific question.
        """
        exploratory_system = """You are a senior data scientist presenting exploratory analysis results.

The user asked for a general analysis. Present the most interesting discoveries.

STRUCTURE (use markdown):
1. **Dataset Overview** (size, structure, key characteristics)
2. **Most Interesting Discoveries** (ranked by insight value)
3. **Key Patterns & Relationships** (correlations, distributions, trends)
4. **Data Quality Notes** (missing data, outliers, issues found)
5. **Visualizations Generated** (list with descriptions)
6. **Recommended Next Steps** (what to investigate deeper)

RULES:
- Lead with the most surprising/important finding
- ONLY cite numbers that appear VERBATIM in the tool results — do NOT round, invent thresholds, or fabricate statistics
- Do NOT claim data quality issues or anomalies unless the tools explicitly reported them
- BUT DO provide RICH analytical depth using the ACTUAL data:
  • For each key column, interpret what its real min/max/mean/median tells us about the domain
  • Explain what the actual correlation values mean in practical terms — not just "correlated" but WHY it matters
  • Identify the most and least variable features and explain what that variability implies
  • Highlight interesting contrasts between columns (e.g., "while X spans a wide range, Y is tightly clustered")
  • Derive actionable insights from the real distributions — what do the actual values suggest the user should do?
- Mention all generated visualizations with file paths
- Suggest actionable next analysis steps grounded in the actual findings
- Keep it engaging, analytical, and data-driven — DEPTH comes from interpreting real data, not inventing data
- Do NOT wrap your response in code fences (``` or ```markdown) — output raw markdown directly
- ONLY report findings from SUCCESSFUL investigation steps
- If a step is marked [FAILED], ignore it entirely"""

        artifacts_summary = self._format_artifacts(artifacts or {}, findings)
        
        # Build filtered context — only successful findings
        findings_context = self._build_filtered_context(findings)
        
        user_prompt = f"""**Analysis request**: {findings.question}

**Investigation summary**:
{findings_context}

**Generated artifacts**:
{artifacts_summary}

Write the exploratory analysis report."""

        response = self.llm_caller(
            system_prompt=exploratory_system,
            user_prompt=user_prompt,
            max_tokens=max_tokens
        )
        
        return self._strip_code_fences(response.strip())

    def _strip_code_fences(self, text: str) -> str:
        """
        Remove wrapping code fences from LLM output.
        
        LLMs sometimes wrap markdown in ```markdown ... ``` which causes
        ReactMarkdown to render the entire response as a code block
        instead of parsing the markdown.
        """
        import re
        # Strip leading ```markdown or ``` and trailing ```
        stripped = re.sub(r'^\s*```(?:markdown|md|text)?\s*\n', '', text)
        stripped = re.sub(r'\n\s*```\s*$', '', stripped)
        return stripped.strip()

    def _format_artifacts(self, artifacts: Dict[str, Any], findings: FindingsAccumulator) -> str:
        """Format artifacts for the synthesis prompt."""
        parts = []
        
        # Extract plots from findings
        plots = artifacts.get("plots", [])
        if plots:
            parts.append("**Plots generated**:")
            for plot in plots:
                if isinstance(plot, dict):
                    parts.append(f"  - {plot.get('title', 'Plot')}: {plot.get('url', plot.get('path', 'N/A'))}")
                else:
                    parts.append(f"  - {plot}")
        
        # Extract files from findings
        files = artifacts.get("files", [])
        if files:
            parts.append("**Output files**:")
            for f in files:
                parts.append(f"  - {f}")
        
        # Extract from findings history — only from successful steps
        for finding in findings.findings:
            if not finding.success:
                continue
            result = finding.result_summary
            if "output_file" in result or "output_path" in result or ".html" in result or ".png" in result:
                parts.append(f"  - Step {finding.iteration} ({finding.action}): output in result")
        
        # Tools used summary
        if findings.tools_used:
            parts.append(f"\n**Tools used**: {', '.join(findings.tools_used)}")
        
        if not parts:
            return "No artifacts generated yet."
        
        return "\n".join(parts)

    def _build_filtered_context(self, findings: FindingsAccumulator) -> str:
        """
        Build synthesis context that only includes SUCCESSFUL findings in detail.
        Failed findings are listed as a brief summary so the LLM knows they happened
        but cannot hallucinate data from them.
        """
        import json
        
        parts = []
        parts.append(f"**Original question**: {findings.question}")
        parts.append(f"**Mode**: {findings.mode}")
        
        successful = findings.get_successful_findings()
        failed = [f for f in findings.findings if not f.success]
        
        parts.append(f"**Total iterations**: {len(findings.findings)} ({len(successful)} succeeded, {len(failed)} failed)")
        parts.append(f"**Tools used**: {', '.join(findings.tools_used)}")
        
        # Only successful findings get full detail
        if successful:
            parts.append("\n## Successful Investigation Steps\n")
            for f in successful:
                parts.append(
                    f"### Step {f.iteration}: {f.action}\n"
                    f"**Hypothesis**: {f.hypothesis}\n"
                    f"**Arguments**: {json.dumps(f.arguments, default=str)}\n"
                    f"**Result**: {f.result_summary}\n"
                    f"**Interpretation**: {f.interpretation}\n"
                    f"**Confidence**: {f.confidence:.0%}\n"
                )
        
        # Failed findings get just a one-line mention
        if failed:
            parts.append("\n## Failed Steps (no usable data — do NOT cite these)\n")
            for f in failed:
                parts.append(f"- Step {f.iteration}: `{f.action}` FAILED — {f.error_message or 'execution error'}")
        
        # Hypothesis outcomes
        if findings.hypotheses:
            parts.append("\n## Hypothesis Outcomes\n")
            for h in findings.hypotheses:
                status_emoji = {
                    "supported": "\u2705",
                    "refuted": "\u274c",
                    "inconclusive": "\u2753",
                    "testing": "\ud83d\udd04",
                    "untested": "\u2b1c"
                }.get(h.status, "\u2b1c")
                parts.append(f"{status_emoji} **{h.text}** \u2192 {h.status}")
                if h.evidence_for:
                    parts.append(f"  Evidence for: {'; '.join(h.evidence_for)}")
                if h.evidence_against:
                    parts.append(f"  Evidence against: {'; '.join(h.evidence_against)}")
        
        return "\n".join(parts)
