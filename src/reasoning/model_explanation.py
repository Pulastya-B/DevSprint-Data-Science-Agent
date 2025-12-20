"""
Model Explanation Module

Provides reasoning about model behavior, performance, and interpretability.

KEY RULES:
- ✅ Accepts: Model metrics, predictions, feature importances
- ❌ NO: Raw model objects, training loops
- ✅ Returns: Explanations of WHY model behaves as it does
- ❌ NO: Model selection, hyperparameter choices

Use Cases:
1. Explain model performance metrics
2. Interpret feature importances
3. Diagnose model failures
4. Suggest model debugging steps

Example:
    from reasoning.model_explanation import explain_model_performance
    
    metrics = {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "confusion_matrix": [[800, 50], [100, 50]]
    }
    
    explanation = explain_model_performance(metrics, "classification")
    # Returns: "Your model has high accuracy but low recall..."
"""

from typing import Dict, Any, List, Optional
from . import get_reasoner


def explain_model_performance(
    metrics: Dict[str, Any],
    task_type: str,
    baseline_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Explain model performance metrics in plain English.
    
    Args:
        metrics: Performance metrics (accuracy, precision, recall, etc.)
        task_type: 'classification' or 'regression'
        baseline_metrics: Optional baseline to compare against
        
    Returns:
        {
            "summary": str,                    # Overall assessment
            "strengths": List[str],            # What model does well
            "weaknesses": List[str],           # What model struggles with
            "confusion_analysis": str,         # Confusion matrix interpretation
            "next_steps": List[str]            # Suggested improvements
        }
    """
    reasoner = get_reasoner()
    
    comparison = ""
    if baseline_metrics:
        comparison = f"\n**Baseline Metrics (for comparison):**\n{baseline_metrics}"
    
    prompt = f"""Analyze these model performance metrics:

**Task Type:** {task_type}

**Metrics:**
{metrics}{comparison}

Provide:
1. Overall performance summary (good/bad/acceptable)
2. Strengths (what model does well)
3. Weaknesses (where model struggles)
4. Confusion matrix analysis (if classification)
5. Next steps for improvement

Be specific and actionable. If performance is poor, suggest why."""
    
    system_prompt = """You are a model interpretation expert.
Explain performance metrics in terms business users understand.
Focus on actionable insights, not just numbers."""
    
    schema = {
        "summary": "string - Overall assessment",
        "strengths": ["array of strengths"],
        "weaknesses": ["array of weaknesses"],
        "confusion_analysis": "string - Confusion matrix explanation",
        "next_steps": ["array of improvement suggestions"]
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def interpret_feature_importance(
    feature_importances: Dict[str, float],
    top_n: int = 10,
    domain: Optional[str] = None
) -> Dict[str, Any]:
    """
    Interpret feature importance scores and explain what they mean.
    
    Args:
        feature_importances: {feature_name: importance_score}
        top_n: Number of top features to focus on
        domain: Optional domain context
        
    Returns:
        {
            "top_features": List[str],         # Most important features
            "interpretation": str,             # What importances mean
            "surprising_features": List[str],  # Unexpectedly important/unimportant
            "feature_relationships": str,      # How features might interact
            "recommendations": List[str]       # What to investigate further
        }
    """
    reasoner = get_reasoner()
    
    # Sort by importance
    sorted_features = sorted(
        feature_importances.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    domain_context = f"\nDomain: {domain}" if domain else ""
    
    prompt = f"""Interpret these feature importance scores:

**Top {top_n} Most Important Features:**
{dict(sorted_features)}

**All Features:**
{feature_importances}{domain_context}

Explain:
1. What these importances tell us about the model
2. Which features are surprisingly important/unimportant
3. Potential feature interactions or relationships
4. What to investigate further
5. Whether importances make intuitive sense

Be specific about WHY certain features might be important."""
    
    system_prompt = """You are a model interpretability expert.
Explain feature importances in domain terms, not just statistical terms.
Point out surprising or counterintuitive results."""
    
    schema = {
        "top_features": ["array of most important features"],
        "interpretation": "string - What importances mean overall",
        "surprising_features": ["array of unexpected results"],
        "feature_relationships": "string - How features might interact",
        "recommendations": ["array of investigation suggestions"]
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def diagnose_model_failure(
    failure_description: str,
    model_type: str,
    metrics: Dict[str, Any],
    sample_predictions: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Diagnose why a model is failing and suggest fixes.
    
    Args:
        failure_description: Description of the problem
            Example: "Model predicts all positives" or "Poor performance on test set"
        model_type: Model algorithm used
        metrics: Current performance metrics
        sample_predictions: Optional sample of predictions vs actuals
        
    Returns:
        {
            "diagnosis": str,              # What's likely wrong
            "root_causes": List[str],      # Possible root causes
            "debugging_steps": List[str],  # How to investigate
            "potential_fixes": List[str]   # Suggested solutions
        }
    """
    reasoner = get_reasoner()
    
    samples = ""
    if sample_predictions:
        samples = f"\n**Sample Predictions:**\n{sample_predictions[:10]}"
    
    prompt = f"""Diagnose this model failure:

**Problem:** {failure_description}

**Model Type:** {model_type}

**Current Metrics:**
{metrics}{samples}

Provide:
1. Diagnosis of what's likely wrong
2. Possible root causes
3. Debugging steps to take
4. Potential fixes to try

Be specific and prioritize most likely causes."""
    
    system_prompt = """You are a model debugging expert.
Provide systematic diagnostic steps, not just guesses.
Prioritize most common failure modes first."""
    
    schema = {
        "diagnosis": "string - What's likely wrong",
        "root_causes": ["array of possible causes"],
        "debugging_steps": ["array of investigation steps"],
        "potential_fixes": ["array of solutions to try"]
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def explain_prediction(
    prediction: Any,
    feature_values: Dict[str, Any],
    feature_contributions: Optional[Dict[str, float]] = None,
    model_type: str = "unknown"
) -> str:
    """
    Explain a single prediction in plain English.
    
    Args:
        prediction: Model's prediction
        feature_values: Feature values for this prediction
        feature_contributions: Optional SHAP values or contributions
        model_type: Type of model
        
    Returns:
        Natural language explanation of the prediction
    """
    reasoner = get_reasoner()
    
    contributions = ""
    if feature_contributions:
        contributions = f"\n**Feature Contributions:**\n{feature_contributions}"
    
    prompt = f"""Explain this model prediction in simple terms:

**Prediction:** {prediction}

**Input Features:**
{feature_values}{contributions}

**Model Type:** {model_type}

Explain:
- What the model predicted
- Which features most influenced the prediction
- Why this prediction makes sense (or doesn't)
- How confident we should be in this prediction

Make it understandable to non-technical users."""
    
    system_prompt = """You are explaining model predictions to business users.
Use plain English, avoid jargon, focus on the 'why' behind predictions."""
    
    return reasoner.reason(prompt, system_prompt, temperature=0.1)


def compare_models(
    model1_metrics: Dict[str, Any],
    model2_metrics: Dict[str, Any],
    model1_name: str = "Model A",
    model2_name: str = "Model B",
    business_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two models and recommend which to use.
    
    Args:
        model1_metrics: Metrics for first model
        model2_metrics: Metrics for second model
        model1_name: Name/description of first model
        model2_name: Name/description of second model
        business_context: Optional business requirements
            Example: "Need high recall, false negatives are costly"
        
    Returns:
        {
            "winner": str,                     # Which model is better
            "comparison": str,                 # Detailed comparison
            "tradeoffs": List[str],            # Key tradeoffs
            "recommendation": str,             # Final recommendation
            "context_considerations": str      # Business context factors
        }
    """
    reasoner = get_reasoner()
    
    context = ""
    if business_context:
        context = f"\n**Business Context:**\n{business_context}"
    
    prompt = f"""Compare these two models:

**{model1_name} Metrics:**
{model1_metrics}

**{model2_name} Metrics:**
{model2_metrics}{context}

Determine:
1. Which model is objectively better (if any)
2. Key differences and tradeoffs
3. Which model to choose given business context
4. When you might choose the "worse" model

Consider accuracy, precision, recall, training time, interpretability, etc."""
    
    system_prompt = """You are a model selection expert.
Don't just pick the highest accuracy - consider tradeoffs and business needs.
Sometimes a simpler or faster model is better."""
    
    schema = {
        "winner": "string - Which model is better overall",
        "comparison": "string - Detailed comparison",
        "tradeoffs": ["array of key tradeoffs"],
        "recommendation": "string - Final recommendation with reasoning",
        "context_considerations": "string - How business context affects choice"
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)


def explain_overfitting(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_complexity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect and explain overfitting (or underfitting).
    
    Args:
        train_metrics: Training set metrics
        test_metrics: Test set metrics
        model_complexity: Optional description of model complexity
        
    Returns:
        {
            "diagnosis": str,              # Overfitting/underfitting/good_fit
            "severity": str,               # Low/medium/high
            "explanation": str,            # Why this is happening
            "solutions": List[str]         # How to fix it
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Analyze these train vs test metrics for overfitting:

**Training Metrics:**
{train_metrics}

**Test Metrics:**
{test_metrics}

**Model Complexity:** {model_complexity or 'Unknown'}

Determine:
1. Whether model is overfitting, underfitting, or well-fitted
2. Severity of the problem
3. Why this is happening
4. Specific solutions to try

Be specific about the gap between train and test performance."""
    
    system_prompt = """You are a model diagnostics expert.
Explain overfitting in practical terms and provide actionable solutions."""
    
    schema = {
        "diagnosis": "string - overfitting/underfitting/good_fit",
        "severity": "string - low/medium/high",
        "explanation": "string - Why this is happening",
        "solutions": ["array of specific fixes to try"]
    }
    
    return reasoner.reason_structured(prompt, schema, system_prompt)
