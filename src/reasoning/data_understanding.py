"""
Data Understanding Module

Provides reasoning about data characteristics, patterns, and quality.

KEY RULES:
- ✅ Accepts: Statistical summaries, metadata, sample rows
- ❌ NO: Raw DataFrames, full datasets
- ✅ Returns: Natural language insights + structured recommendations
- ❌ NO: Training decisions, model selection

Use Cases:
1. Explain what data represents
2. Identify data quality issues
3. Suggest preprocessing steps
4. Highlight interesting patterns

Example:
    from reasoning.data_understanding import explain_dataset
    
    summary = {
        "rows": 10000,
        "columns": 20,
        "numeric": 15,
        "categorical": 5,
        "missing_values": {"age": 150, "income": 200},
        "target_distribution": {"yes": 7000, "no": 3000}
    }
    
    explanation = explain_dataset(summary)
    # Returns: {
    #     "overview": "This is an imbalanced classification dataset...",
    #     "quality_issues": ["Missing values in age and income"],
    #     "recommendations": ["Handle class imbalance", "Impute missing values"],
    #     "patterns": ["Target class imbalanced (70-30 split)"]
    # }
"""

from typing import Dict, Any, List, Optional
from . import get_reasoner


def explain_dataset(
    summary: Dict[str, Any],
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Explain dataset characteristics based on summary statistics.
    
    Args:
        summary: Statistical summary (NO raw data!)
            Must include: rows, columns, dtypes, missing_values
            Optional: target_distribution, correlations, outliers
        target_col: Target column name (if known)
        
    Returns:
        {
            "overview": str,              # High-level description
            "quality_issues": List[str],  # Data quality problems
            "recommendations": List[str], # Suggested preprocessing steps
            "patterns": List[str],        # Interesting patterns found
            "target_insights": str        # Target variable insights (if applicable)
        }
    """
    # Validate inputs FIRST (NO raw data allowed!)
    if "dataframe" in summary or "df" in summary:
        raise ValueError("Cannot pass raw DataFrames! Pass summary statistics only.")
    
    reasoner = get_reasoner()
    
    # Build reasoning prompt from summary
    prompt = f"""Analyze this dataset summary and provide insights:

**Dataset Summary:**
- Rows: {summary.get('rows', 'unknown')}
- Columns: {summary.get('columns', 'unknown')}
- Numeric columns: {summary.get('numeric_columns', [])}
- Categorical columns: {summary.get('categorical_columns', [])}
- Missing values: {summary.get('missing_values', {})}
- Target column: {target_col or 'Not specified'}

**Target Distribution (if available):**
{summary.get('target_distribution', 'Not provided')}

**Correlations (if available):**
{summary.get('top_correlations', 'Not provided')}

**Outliers (if available):**
{summary.get('outliers', 'Not provided')}

Provide:
1. Overview of what this data represents
2. Data quality issues identified
3. Preprocessing recommendations
4. Interesting patterns noticed
5. Target variable insights (if classification/regression)
"""
    
    system_prompt = """You are a data understanding expert. Your role is to:
- Explain what data means in plain English
- Identify data quality issues
- Suggest preprocessing steps
- Highlight patterns

You do NOT:
- Make training decisions
- Select models
- Access raw data
- Execute any code

You ONLY reason about summaries provided."""
    
    schema = {
        "overview": "string - High-level description of dataset",
        "quality_issues": ["array of strings - Data quality problems found"],
        "recommendations": ["array of strings - Preprocessing steps to take"],
        "patterns": ["array of strings - Interesting patterns noticed"],
        "target_insights": "string - Insights about target variable"
    }
    
    result = reasoner.reason_structured(prompt, schema, system_prompt)
    
    return result


def explain_data_profile(
    profile: Dict[str, Any]
) -> str:
    """
    Generate natural language explanation of data profiling results.
    
    Args:
        profile: Profiling output from tools (column stats, distributions, etc.)
            Example: {
                "column_stats": {...},
                "missing_summary": {...},
                "cardinality": {...}
            }
            
    Returns:
        Natural language explanation
    """
    reasoner = get_reasoner()
    
    prompt = f"""Explain these data profiling results in clear, actionable terms:

{profile}

Focus on:
- What the data looks like
- Any concerning patterns
- Next steps for data cleaning
"""
    
    system_prompt = """You are a data quality expert explaining profiling results.
Be concise, actionable, and highlight the most important findings."""
    
    return reasoner.reason(prompt, system_prompt, temperature=0.1)


def suggest_transformations(
    column_stats: Dict[str, Any],
    task_type: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Suggest transformations for each column based on statistics.
    
    Args:
        column_stats: Per-column statistics
            Example: {
                "age": {"min": 0, "max": 150, "outliers": 5},
                "income": {"skewness": 3.5, "distribution": "highly_skewed"}
            }
        task_type: 'classification' or 'regression' (if known)
        
    Returns:
        {
            "age": ["Remove outliers > 100", "Normalize to 0-1 range"],
            "income": ["Apply log transform (skewed)", "Remove negative values"]
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Based on these column statistics, suggest transformations:

**Column Statistics:**
{column_stats}

**Task Type:** {task_type or 'Unknown'}

For each column, suggest:
- Outlier handling
- Scaling/normalization
- Distribution transformations
- Encoding strategies (for categorical)

Be specific and actionable."""
    
    schema = {
        "column_name": ["array of transformation suggestions"]
    }
    
    return reasoner.reason_structured(prompt, schema)


def identify_feature_engineering_opportunities(
    summary: Dict[str, Any],
    domain: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Identify feature engineering opportunities based on data summary.
    
    Args:
        summary: Dataset summary with column names and types
        domain: Optional domain context (e.g., "healthcare", "finance")
        
    Returns:
        [
            {
                "opportunity": "Create age_bins feature",
                "reason": "Age is continuous but may benefit from binning",
                "suggested_code": "pd.cut(df['age'], bins=[0,18,35,50,65,100])"
            },
            ...
        ]
    """
    reasoner = get_reasoner()
    
    domain_context = f"\nDomain: {domain}" if domain else ""
    
    prompt = f"""Identify feature engineering opportunities from this data:

**Available Columns:**
{summary.get('columns', [])}

**Column Types:**
{summary.get('dtypes', {})}

**Sample Values:**
{summary.get('sample_values', 'Not provided')}{domain_context}

Suggest:
1. Interaction features (e.g., BMI from height/weight)
2. Binning/discretization opportunities
3. Time-based features (if datetime columns exist)
4. Encoding strategies
5. Domain-specific features

For each opportunity, explain WHY it would help."""
    
    system_prompt = """You are a feature engineering expert.
Suggest creative but practical feature transformations.
Focus on features that typically improve model performance."""
    
    schema = {
        "opportunities": [
            {
                "opportunity": "string - What to create",
                "reason": "string - Why it would help",
                "suggested_code": "string - Pseudo-code or actual code"
            }
        ]
    }
    
    result = reasoner.reason_structured(prompt, schema, system_prompt)
    return result.get("opportunities", [])


def explain_missing_values(
    missing_summary: Dict[str, Any]
) -> Dict[str, str]:
    """
    Explain missing value patterns and suggest strategies.
    
    Args:
        missing_summary: Summary of missing values
            Example: {
                "age": {"count": 150, "percentage": 1.5, "pattern": "random"},
                "income": {"count": 500, "percentage": 5.0, "pattern": "not_random"}
            }
            
    Returns:
        {
            "age": "1.5% missing (random) - Safe to impute with median",
            "income": "5% missing (non-random) - May indicate bias, consider separate category"
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Analyze these missing value patterns and suggest handling strategies:

{missing_summary}

For each column with missing values:
1. Assess the missing pattern (random vs systematic)
2. Suggest imputation strategy
3. Warn about any concerns (bias, data leakage, etc.)
"""
    
    schema = {
        "column_name": "string - Assessment and strategy"
    }
    
    return reasoner.reason_structured(prompt, schema)


def compare_datasets(
    dataset1_summary: Dict[str, Any],
    dataset2_summary: Dict[str, Any],
    comparison_purpose: str = "train_test_validation"
) -> Dict[str, Any]:
    """
    Compare two dataset summaries and identify differences.
    
    Args:
        dataset1_summary: Summary of first dataset
        dataset2_summary: Summary of second dataset
        comparison_purpose: 'train_test_validation', 'before_after', or 'a_b_test'
        
    Returns:
        {
            "differences": List[str],      # Key differences found
            "concerns": List[str],         # Potential issues
            "data_drift": bool,            # Whether distribution shift detected
            "recommendation": str          # What to do about differences
        }
    """
    reasoner = get_reasoner()
    
    prompt = f"""Compare these two datasets:

**Dataset 1:**
{dataset1_summary}

**Dataset 2:**
{dataset2_summary}

**Comparison Purpose:** {comparison_purpose}

Identify:
1. Distribution differences
2. Schema differences
3. Data quality differences
4. Potential data drift or leakage
5. Whether differences are concerning

Be specific about what changed and why it matters."""
    
    schema = {
        "differences": ["array of key differences"],
        "concerns": ["array of potential issues"],
        "data_drift": "boolean",
        "recommendation": "string - What to do"
    }
    
    return reasoner.reason_structured(prompt, schema)
