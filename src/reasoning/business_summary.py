"""
Business Summary Module

Translates technical ML results into business-friendly summaries.

KEY RULES:
- ✅ Accepts: Model results, metrics, insights
- ❌ NO: Raw technical details in output
- ✅ Returns: Executive summaries, ROI estimates, actionable recommendations
- ❌ NO: Code, statistical jargon, complex formulas

Use Cases:
1. Executive summaries of ML projects
2. ROI/impact estimation
3. Stakeholder-friendly reporting
4. Business recommendations from technical results

Example:
    from reasoning.business_summary import create_executive_summary
    
    results = {
        "model_accuracy": 0.95,
        "cost_savings": "$50K/year",
        "deployment_ready": True
    }
    
    summary = create_executive_summary(results, "churn_prediction")
    # Returns: "This churn prediction model can save $50K annually..."
"""

from typing import Dict, Any, List, Optional
from . import get_reasoner


def create_executive_summary(
    project_results: Dict[str, Any],
    project_name: str,
    business_objective: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create executive summary of ML project for non-technical stakeholders.
    
    Args:
        project_results: Technical results (metrics, insights, etc.)
        project_name: Name of the ML project
        business_objective: What business problem this solves
        
    Returns:
        {
            "executive_summary": str,          # 2-3 sentence overview
            "key_findings": List[str],         # 3-5 bullet points
            "business_impact": str,            # Expected impact
            "recommendations": List[str],      # What to do next
            "risks_and_limitations": List[str] # Important caveats
        }
    """
    reasoner = get_reasoner()
    
    objective = ""
    if business_objective:
        objective = f"\n**Business Objective:** {business_objective}"
    
    prompt = f"""Create an executive summary for this ML project:

**Project:** {project_name}{objective}

**Technical Results:**
{project_results}

Write for C-level executives who don't understand ML.

Include:
1. 2-3 sentence executive summary
2. 3-5 key findings (what we learned)
3. Business impact (quantified if possible)
4. Recommendations (what to do next)
5. Risks and limitations (important caveats)

Use business language, not technical jargon.
Focus on outcomes, not methods."""
    
    system_prompt = """You are translating technical ML results for business executives.
Avoid jargon like 'accuracy', 'recall', 'features' - use business terms.
Focus on ROI, impact, and actionable next steps."""
    
    schema = {
        "executive_summary": "string - 2-3 sentence overview",
        "key_findings": ["array of 3-5 key insights"],
        "business_impact": "string - Expected business impact",
        "recommendations": ["array of next steps"],
        "risks_and_limitations": ["array of important caveats"]
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def estimate_business_impact(
    model_performance: Dict[str, Any],
    business_metrics: Dict[str, Any],
    use_case: str
) -> Dict[str, Any]:
    """
    Estimate business impact of deploying the model.
    
    Args:
        model_performance: Model metrics (accuracy, recall, etc.)
        business_metrics: Business context
            Example: {
                "current_churn_rate": 0.25,
                "customer_lifetime_value": 1000,
                "customers": 10000
            }
        use_case: Description of use case
            Example: "churn prediction", "fraud detection", "demand forecasting"
        
    Returns:
        {
            "estimated_impact": str,           # Quantified impact
            "assumptions": List[str],          # Key assumptions made
            "sensitivity": str,                # How sensitive to assumptions
            "confidence_level": str,           # Confidence in estimates
            "impact_breakdown": Dict[str, str] # Detailed breakdown
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Estimate the business impact of deploying this model:

**Use Case:** {use_case}

**Model Performance:**
{model_performance}

**Business Context:**
{business_metrics}

Estimate:
1. Quantified business impact (revenue, cost savings, etc.)
2. Key assumptions in your calculation
3. Sensitivity to assumptions
4. Confidence level in estimates
5. Breakdown of impact by component

Be conservative in estimates. Show your reasoning."""
    
    system_prompt = """You are a business impact analyst.
Provide realistic, conservative estimates with clear assumptions.
Show how you calculated impact - don't just guess."""
    
    schema = {
        "estimated_impact": "string - Quantified impact estimate",
        "assumptions": ["array of key assumptions"],
        "sensitivity": "string - How sensitive to assumptions",
        "confidence_level": "string - low/medium/high",
        "impact_breakdown": {
            "component": "string - Impact value"
        }
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def create_stakeholder_report(
    audience: str,
    project_status: str,
    key_metrics: Dict[str, Any],
    timeline: Optional[Dict[str, str]] = None
) -> str:
    """
    Create customized report for specific stakeholder audience.
    
    Args:
        audience: 'executives', 'engineers', 'business_users', 'data_team'
        project_status: Current project status
        key_metrics: Relevant metrics for this audience
        timeline: Optional timeline information
        
    Returns:
        Natural language report customized for audience
    """
    reasoner = get_reasoner()
    
    timeline_section = ""
    if timeline:
        timeline_section = f"\n**Timeline:**\n{timeline}"
    
    # Audience-specific focus
    audience_focus = {
        "executives": "ROI, strategic alignment, high-level status",
        "engineers": "Technical implementation, architecture, performance",
        "business_users": "How to use, what it means for their work, training needs",
        "data_team": "Data quality, model performance, monitoring needs"
    }
    
    focus = audience_focus.get(audience, "General status")
    
    prompt = f"""Create a report for {audience}:

**Project Status:** {project_status}

**Key Metrics:**
{key_metrics}{timeline_section}

**Focus Areas:** {focus}

Tailor the report for this specific audience.
Use language and concepts they understand.
Highlight what matters most to them."""
    
    system_prompt = f"""You are writing a report for {audience}.
Use appropriate language and detail level for this audience.
Focus on what they care about most."""
    
    return reasoner.reason(prompt, system_prompt, temperature=0.2)


def translate_technical_to_business(
    technical_term: str,
    context: Optional[str] = None
) -> str:
    """
    Translate technical ML term to business-friendly language.
    
    Args:
        technical_term: ML term to translate
            Examples: "precision", "recall", "overfitting", "feature importance"
        context: Optional context for better translation
        
    Returns:
        Business-friendly explanation
    """
    reasoner = get_reasoner()
    
    context_section = ""
    if context:
        context_section = f"\n**Context:** {context}"
    
    prompt = f"""Translate this technical ML term to business language:

**Technical Term:** {technical_term}{context_section}

Explain:
1. What it means in plain English
2. Why it matters for business
3. Real-world analogy if helpful

Avoid technical jargon in your explanation."""
    
    system_prompt = """You are translating ML concepts for business audiences.
Use analogies and examples they can relate to.
Focus on 'why it matters', not 'how it works'."""
    
    return reasoner.reason(prompt, system_prompt, temperature=0.1)


def prioritize_next_steps(
    current_results: Dict[str, Any],
    available_resources: Dict[str, Any],
    business_constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prioritize next steps based on results, resources, and constraints.
    
    Args:
        current_results: Current project state and results
        available_resources: Available time, budget, team
        business_constraints: Deadlines, must-haves, etc.
        
    Returns:
        {
            "high_priority": List[Dict],    # Must-do items
            "medium_priority": List[Dict],  # Should-do items
            "low_priority": List[Dict],     # Nice-to-have items
            "rationale": str                # Prioritization reasoning
        }
    """
    reasoner = get_reasoner()
    
    constraints = ""
    if business_constraints:
        constraints = f"\n**Business Constraints:**\n{business_constraints}"
    
    prompt = f"""Prioritize next steps for this ML project:

**Current Results:**
{current_results}

**Available Resources:**
{available_resources}{constraints}

Categorize tasks into:
1. High Priority (must-do, high impact, blocking)
2. Medium Priority (should-do, good ROI)
3. Low Priority (nice-to-have, polish)

For each item, specify:
- What to do
- Why it's important
- Estimated effort
- Expected impact

Consider resource constraints and business deadlines."""
    
    system_prompt = """You are a product/project manager prioritizing ML work.
Use impact vs effort analysis.
Be realistic about what can be accomplished with available resources."""
    
    schema = {
        "high_priority": [
            {
                "task": "string",
                "reason": "string",
                "effort": "string",
                "impact": "string"
            }
        ],
        "medium_priority": [{"task": "string", "reason": "string", "effort": "string", "impact": "string"}],
        "low_priority": [{"task": "string", "reason": "string", "effort": "string", "impact": "string"}],
        "rationale": "string - Overall prioritization logic"
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def explain_to_customer(
    prediction: Any,
    explanation_level: str = "simple",
    allow_appeal: bool = False
) -> str:
    """
    Explain ML prediction to end customer (explainability for users).
    
    Args:
        prediction: What the model predicted
        explanation_level: 'simple', 'detailed', or 'technical'
        allow_appeal: Whether customer can appeal the decision
        
    Returns:
        Customer-facing explanation
    """
    reasoner = get_reasoner()
    
    appeal_text = ""
    if allow_appeal:
        appeal_text = "\n\nNote: Customer can appeal this decision, explain how."
    
    prompt = f"""Explain this ML prediction to an end customer:

**Prediction:** {prediction}

**Explanation Level:** {explanation_level}

**Requirements:**
- Be transparent but not technical
- Build trust, don't confuse
- Comply with explainability requirements (GDPR, fair lending, etc.)
- Don't expose proprietary model details{appeal_text}

Focus on:
- What was decided
- Key factors that influenced it
- What customer can do if they disagree"""
    
    system_prompt = """You are writing customer-facing explanations.
Be clear, honest, and empathetic.
Comply with regulatory explainability requirements.
Don't say 'the algorithm decided' - take ownership."""
    
    return reasoner.reason(prompt, system_prompt, temperature=0.2)


def assess_deployment_readiness(
    model_results: Dict[str, Any],
    production_requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Assess whether model is ready for production deployment.
    
    Args:
        model_results: Model performance and characteristics
        production_requirements: Production system requirements
            Example: {
                "min_accuracy": 0.90,
                "max_latency_ms": 100,
                "required_explainability": True
            }
        
    Returns:
        {
            "ready_for_deployment": bool,
            "readiness_score": float,          # 0-1 score
            "blockers": List[str],             # Must-fix issues
            "concerns": List[str],             # Should-fix issues
            "sign_offs_needed": List[str],     # Required approvals
            "deployment_recommendation": str    # Go/no-go reasoning
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Assess deployment readiness:

**Model Results:**
{model_results}

**Production Requirements:**
{production_requirements}

Determine:
1. Whether model is ready for deployment (yes/no)
2. Readiness score (0-1, where 1 = fully ready)
3. Blocking issues (must be fixed before deployment)
4. Concerns (should be addressed but not blockers)
5. Required sign-offs (legal, compliance, business, etc.)
6. Go/no-go recommendation with reasoning

Be thorough - production issues are costly."""
    
    system_prompt = """You are assessing production deployment readiness.
Be conservative - it's better to delay than deploy broken model.
Consider performance, reliability, explainability, fairness, and compliance."""
    
    schema = {
        "ready_for_deployment": "boolean",
        "readiness_score": "number between 0 and 1",
        "blockers": ["array of must-fix issues"],
        "concerns": ["array of should-fix issues"],
        "sign_offs_needed": ["array of required approvals"],
        "deployment_recommendation": "string - Go/no-go with reasoning"
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)
