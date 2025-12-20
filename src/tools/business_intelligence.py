"""
Business Intelligence & Analytics Tools

Advanced business analytics tools for cohort analysis, RFM segmentation,
causal inference, and automated insight generation.
"""

import polars as pl
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json

# Statistical packages
try:
    from scipy import stats
    from scipy.stats import chi2_contingency, ttest_ind, f_oneway
except ImportError:
    pass

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.proportion import proportions_ztest
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Causal inference (optional)
try:
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

# Customer analytics (optional)
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    from lifetimes.utils import summary_data_from_transaction_data
    LIFETIMES_AVAILABLE = True
except ImportError:
    LIFETIMES_AVAILABLE = False

# For Groq API calls
import os
from groq import Groq


def perform_cohort_analysis(
    data: pl.DataFrame,
    customer_id_column: str,
    date_column: str,
    value_column: Optional[str] = None,
    cohort_period: str = "monthly",
    metric: str = "retention"
) -> Dict[str, Any]:
    """
    Perform cohort analysis for customer retention, CLV, and churn analysis.
    
    Args:
        data: Input DataFrame with transaction/event data
        customer_id_column: Column containing customer IDs
        date_column: Column containing dates
        value_column: Column containing transaction values (optional, for revenue cohorts)
        cohort_period: Period for cohorts ('daily', 'weekly', 'monthly', 'quarterly')
        metric: Metric to analyze ('retention', 'revenue', 'frequency', 'churn')
    
    Returns:
        Dictionary containing cohort analysis results, retention curves, and insights
    """
    print(f"🔍 Performing cohort analysis ({metric})...")
    
    # Validate input
    required_cols = [customer_id_column, date_column]
    if metric == "revenue" and value_column:
        required_cols.append(value_column)
    
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Convert to pandas for easier date manipulation
    df = data.to_pandas()
    
    # Parse dates
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create cohort based on first purchase date
    df['cohort'] = df.groupby(customer_id_column)[date_column].transform('min')
    
    # Extract period from dates
    period_map = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q'
    }
    
    if cohort_period not in period_map:
        raise ValueError(f"Unknown cohort_period '{cohort_period}'. Use: {list(period_map.keys())}")
    
    period_format = {
        'daily': '%Y-%m-%d',
        'weekly': '%Y-W%U',
        'monthly': '%Y-%m',
        'quarterly': '%Y-Q%q'
    }
    
    df['cohort_period'] = df['cohort'].dt.to_period(period_map[cohort_period])
    df['transaction_period'] = df[date_column].dt.to_period(period_map[cohort_period])
    
    # Calculate period number (periods since cohort start)
    df['period_number'] = (df['transaction_period'] - df['cohort_period']).apply(lambda x: x.n)
    
    result = {
        "metric": metric,
        "cohort_period": cohort_period,
        "total_customers": df[customer_id_column].nunique(),
        "cohorts": []
    }
    
    try:
        if metric == "retention":
            # Retention analysis
            cohort_data = df.groupby(['cohort_period', 'period_number']).agg({
                customer_id_column: 'nunique'
            }).reset_index()
            
            cohort_data.columns = ['cohort_period', 'period_number', 'customers']
            
            # Get cohort sizes (period 0)
            cohort_sizes = cohort_data[cohort_data['period_number'] == 0].set_index('cohort_period')['customers']
            
            # Calculate retention rates
            cohort_data['cohort_size'] = cohort_data['cohort_period'].map(cohort_sizes)
            cohort_data['retention_rate'] = cohort_data['customers'] / cohort_data['cohort_size']
            
            # Pivot for cohort matrix
            cohort_matrix = cohort_data.pivot(
                index='cohort_period',
                columns='period_number',
                values='retention_rate'
            )
            
            result["cohort_matrix"] = cohort_matrix.to_dict()
            result["avg_retention_by_period"] = cohort_matrix.mean().to_dict()
            
            # Calculate churn (1 - retention)
            result["avg_churn_by_period"] = (1 - cohort_matrix.mean()).to_dict()
            
            # Retention curve (average across all cohorts)
            retention_curve = cohort_matrix.mean().to_list()
            result["retention_curve"] = retention_curve
            
        elif metric == "revenue" and value_column:
            # Revenue cohort analysis
            cohort_data = df.groupby(['cohort_period', 'period_number']).agg({
                value_column: 'sum',
                customer_id_column: 'nunique'
            }).reset_index()
            
            cohort_data.columns = ['cohort_period', 'period_number', 'revenue', 'customers']
            
            # Revenue per customer
            cohort_data['revenue_per_customer'] = cohort_data['revenue'] / cohort_data['customers']
            
            # Pivot for cohort matrix
            cohort_matrix = cohort_data.pivot(
                index='cohort_period',
                columns='period_number',
                values='revenue_per_customer'
            )
            
            result["cohort_matrix"] = cohort_matrix.to_dict()
            result["avg_revenue_by_period"] = cohort_matrix.mean().to_dict()
            
            # Cumulative revenue
            cumulative_revenue = cohort_matrix.fillna(0).cumsum(axis=1)
            result["cumulative_revenue"] = cumulative_revenue.mean().to_dict()
            
            # Lifetime value estimate (sum of all periods)
            result["estimated_ltv"] = float(cohort_matrix.sum(axis=1).mean())
            
        elif metric == "frequency":
            # Frequency analysis (purchases per period)
            cohort_data = df.groupby(['cohort_period', 'period_number', customer_id_column]).size().reset_index(name='transactions')
            
            cohort_summary = cohort_data.groupby(['cohort_period', 'period_number']).agg({
                'transactions': 'mean',
                customer_id_column: 'count'
            }).reset_index()
            
            cohort_summary.columns = ['cohort_period', 'period_number', 'avg_transactions', 'active_customers']
            
            # Pivot
            cohort_matrix = cohort_summary.pivot(
                index='cohort_period',
                columns='period_number',
                values='avg_transactions'
            )
            
            result["cohort_matrix"] = cohort_matrix.to_dict()
            result["avg_frequency_by_period"] = cohort_matrix.mean().to_dict()
        
        # Cohort-level statistics
        cohort_stats = []
        for cohort in df['cohort_period'].unique():
            cohort_df = df[df['cohort_period'] == cohort]
            
            stats_dict = {
                "cohort": str(cohort),
                "size": int(cohort_df[customer_id_column].nunique()),
                "total_transactions": int(len(cohort_df)),
                "avg_transactions_per_customer": float(len(cohort_df) / cohort_df[customer_id_column].nunique())
            }
            
            if value_column:
                stats_dict["total_revenue"] = float(cohort_df[value_column].sum())
                stats_dict["avg_revenue_per_customer"] = float(cohort_df[value_column].sum() / cohort_df[customer_id_column].nunique())
            
            cohort_stats.append(stats_dict)
        
        result["cohort_statistics"] = cohort_stats
        
        # Calculate key insights
        result["insights"] = _generate_cohort_insights(result, metric)
        
        print(f"✅ Cohort analysis complete!")
        print(f"   Total customers: {result['total_customers']}")
        print(f"   Cohorts analyzed: {len(cohort_stats)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during cohort analysis: {str(e)}")
        raise


def _generate_cohort_insights(result: Dict[str, Any], metric: str) -> List[str]:
    """Generate insights from cohort analysis."""
    insights = []
    
    if metric == "retention" and "retention_curve" in result:
        retention = result["retention_curve"]
        if len(retention) > 1:
            initial_drop = (retention[0] - retention[1]) * 100
            insights.append(f"Initial retention drop: {initial_drop:.1f}% in first period")
            
            if len(retention) > 3:
                month_3_retention = retention[3] * 100
                insights.append(f"3-period retention: {month_3_retention:.1f}%")
    
    if metric == "revenue" and "estimated_ltv" in result:
        ltv = result["estimated_ltv"]
        insights.append(f"Estimated customer lifetime value: ${ltv:.2f}")
    
    return insights


def perform_rfm_analysis(
    data: pl.DataFrame,
    customer_id_column: str,
    date_column: str,
    value_column: str,
    reference_date: Optional[str] = None,
    rfm_bins: int = 5
) -> Dict[str, Any]:
    """
    Perform RFM (Recency, Frequency, Monetary) analysis for customer segmentation.
    
    Args:
        data: Input DataFrame with transaction data
        customer_id_column: Column containing customer IDs
        date_column: Column containing transaction dates
        value_column: Column containing transaction values
        reference_date: Reference date for recency calculation (default: max date in data)
        rfm_bins: Number of bins for RFM scoring (typically 3, 4, or 5)
    
    Returns:
        Dictionary containing RFM scores, segments, and customer profiles
    """
    print(f"🔍 Performing RFM analysis...")
    
    # Validate input
    required_cols = [customer_id_column, date_column, value_column]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Convert to pandas
    df = data.to_pandas()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set reference date
    if reference_date:
        ref_date = pd.to_datetime(reference_date)
    else:
        ref_date = df[date_column].max()
    
    print(f"  Reference date: {ref_date.strftime('%Y-%m-%d')}")
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_id_column).agg({
        date_column: lambda x: (ref_date - x.max()).days,  # Recency
        customer_id_column: 'count',  # Frequency
        value_column: 'sum'  # Monetary
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # RFM Scoring (1-5, where 5 is best)
    # Note: For recency, lower is better, so we reverse the scoring
    rfm['r_score'] = pd.qcut(rfm['recency'], rfm_bins, labels=range(rfm_bins, 0, -1), duplicates='drop')
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), rfm_bins, labels=range(1, rfm_bins+1), duplicates='drop')
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), rfm_bins, labels=range(1, rfm_bins+1), duplicates='drop')
    
    # Convert to int
    rfm['r_score'] = rfm['r_score'].astype(int)
    rfm['f_score'] = rfm['f_score'].astype(int)
    rfm['m_score'] = rfm['m_score'].astype(int)
    
    # RFM Score (concatenated)
    rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)
    
    # RFM Total Score (sum)
    rfm['rfm_total'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    # Segment customers based on RFM scores
    def segment_customer(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 4 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f < 3:
            return "Potential Loyalists"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Recent Customers"
        elif r >= 3 and m >= 4:
            return "Big Spenders"
        elif r < 3 and f >= 4:
            return "At Risk"
        elif r < 3 and f < 3 and m >= 4:
            return "Can't Lose Them"
        elif r < 2:
            return "Lost"
        else:
            return "Needs Attention"
    
    rfm['segment'] = rfm.apply(segment_customer, axis=1)
    
    # Results
    result = {
        "total_customers": len(rfm),
        "reference_date": ref_date.strftime('%Y-%m-%d'),
        "rfm_bins": rfm_bins,
        "rfm_data": rfm.reset_index().to_dict('records'),
        "segment_summary": {},
        "rfm_statistics": {}
    }
    
    # Segment summary
    segment_stats = rfm.groupby('segment').agg({
        'recency': ['mean', 'median'],
        'frequency': ['mean', 'median'],
        'monetary': ['mean', 'median', 'sum'],
        customer_id_column: 'count'
    }).round(2)
    
    for segment in rfm['segment'].unique():
        segment_data = rfm[rfm['segment'] == segment]
        result["segment_summary"][segment] = {
            "count": int(len(segment_data)),
            "percentage": float(len(segment_data) / len(rfm) * 100),
            "avg_recency": float(segment_data['recency'].mean()),
            "avg_frequency": float(segment_data['frequency'].mean()),
            "avg_monetary": float(segment_data['monetary'].mean()),
            "total_revenue": float(segment_data['monetary'].sum())
        }
    
    # Overall RFM statistics
    result["rfm_statistics"] = {
        "recency": {
            "mean": float(rfm['recency'].mean()),
            "median": float(rfm['recency'].median()),
            "min": int(rfm['recency'].min()),
            "max": int(rfm['recency'].max())
        },
        "frequency": {
            "mean": float(rfm['frequency'].mean()),
            "median": float(rfm['frequency'].median()),
            "min": int(rfm['frequency'].min()),
            "max": int(rfm['frequency'].max())
        },
        "monetary": {
            "mean": float(rfm['monetary'].mean()),
            "median": float(rfm['monetary'].median()),
            "min": float(rfm['monetary'].min()),
            "max": float(rfm['monetary'].max()),
            "total": float(rfm['monetary'].sum())
        }
    }
    
    # Top customers by RFM score
    result["top_customers"] = rfm.nlargest(20, 'rfm_total').reset_index().to_dict('records')
    
    # Actionable insights
    result["recommendations"] = _generate_rfm_recommendations(result)
    
    print(f"✅ RFM analysis complete!")
    print(f"   Total customers: {result['total_customers']}")
    print(f"   Segments: {len(result['segment_summary'])}")
    print(f"   Top segment: {max(result['segment_summary'].items(), key=lambda x: x[1]['count'])[0]}")
    
    return result


def _generate_rfm_recommendations(result: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate actionable recommendations based on RFM segments."""
    
    recommendations = {}
    
    segment_actions = {
        "Champions": [
            "Reward with exclusive perks and early access to new products",
            "Request reviews and referrals",
            "Engage for product development feedback"
        ],
        "Loyal Customers": [
            "Upsell higher value products",
            "Offer loyalty rewards",
            "Encourage referrals with incentives"
        ],
        "Potential Loyalists": [
            "Recommend related products",
            "Offer membership or loyalty program",
            "Engage with personalized communication"
        ],
        "Recent Customers": [
            "Provide onboarding support",
            "Build relationships with targeted content",
            "Offer starter discounts for repeat purchases"
        ],
        "Big Spenders": [
            "Target with premium products",
            "Increase engagement frequency",
            "Offer VIP treatment"
        ],
        "At Risk": [
            "Send win-back campaigns",
            "Offer special discounts or incentives",
            "Gather feedback on their experience"
        ],
        "Can't Lose Them": [
            "Aggressive win-back campaigns",
            "Personalized outreach",
            "Offer significant incentives"
        ],
        "Lost": [
            "Run re-engagement campaigns",
            "Survey for feedback",
            "Consider removing from active campaigns"
        ],
        "Needs Attention": [
            "Offer limited-time promotions",
            "Share valuable content",
            "Re-engage with surveys"
        ]
    }
    
    for segment, actions in segment_actions.items():
        if segment in result["segment_summary"]:
            recommendations[segment] = actions
    
    return recommendations


def detect_causal_relationships(
    data: pl.DataFrame,
    treatment_column: str,
    outcome_column: str,
    covariates: Optional[List[str]] = None,
    method: str = "granger",
    max_lag: int = 5,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Detect causal relationships using Granger causality, propensity matching, or uplift modeling.
    
    Args:
        data: Input DataFrame
        treatment_column: Column indicating treatment/intervention
        outcome_column: Column indicating outcome variable
        covariates: List of covariate columns for adjustment
        method: Method for causal inference ('granger', 'propensity', 'uplift')
        max_lag: Maximum lag for Granger causality test
        confidence_level: Confidence level for statistical tests
    
    Returns:
        Dictionary containing causal inference results and effect estimates
    """
    print(f"🔍 Detecting causal relationships using {method} method...")
    
    # Validate input
    required_cols = [treatment_column, outcome_column]
    if covariates:
        required_cols.extend(covariates)
    
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
    
    result = {
        "method": method,
        "treatment": treatment_column,
        "outcome": outcome_column,
        "covariates": covariates or [],
        "causal_effect": None,
        "statistical_significance": None
    }
    
    try:
        if method == "granger" and STATSMODELS_AVAILABLE:
            # Granger causality test for time series
            print(f"  Testing Granger causality with max lag = {max_lag}...")
            
            # Convert to pandas
            df = data.select([treatment_column, outcome_column]).to_pandas()
            
            # Ensure numeric
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            
            # Test both directions
            test_result = grangercausalitytests(
                df[[outcome_column, treatment_column]],
                max_lag,
                verbose=False
            )
            
            # Extract p-values for each lag
            granger_results = []
            for lag in range(1, max_lag + 1):
                ssr_ftest = test_result[lag][0]['ssr_ftest']
                granger_results.append({
                    "lag": lag,
                    "f_statistic": float(ssr_ftest[0]),
                    "p_value": float(ssr_ftest[1]),
                    "significant": ssr_ftest[1] < (1 - confidence_level)
                })
            
            result["granger_causality"] = granger_results
            result["causal_effect"] = any(r["significant"] for r in granger_results)
            result["statistical_significance"] = min(r["p_value"] for r in granger_results)
            
        elif method == "propensity":
            # Propensity score matching
            print("  Performing propensity score matching...")
            
            df = data.to_pandas()
            
            # Ensure treatment is binary
            treatment = df[treatment_column]
            if treatment.nunique() > 2:
                raise ValueError(f"Treatment column must be binary for propensity matching")
            
            outcome = df[outcome_column]
            
            # Simple comparison without covariates
            if not covariates:
                treated = outcome[treatment == 1]
                control = outcome[treatment == 0]
                
                # Calculate average treatment effect
                ate = treated.mean() - control.mean()
                
                # T-test for significance
                t_stat, p_value = ttest_ind(treated, control)
                
                result["average_treatment_effect"] = float(ate)
                result["t_statistic"] = float(t_stat)
                result["p_value"] = float(p_value)
                result["statistical_significance"] = float(p_value)
                result["causal_effect"] = float(ate)
                result["confidence_interval"] = [
                    float(ate - 1.96 * np.sqrt(treated.var()/len(treated) + control.var()/len(control))),
                    float(ate + 1.96 * np.sqrt(treated.var()/len(treated) + control.var()/len(control)))
                ]
            else:
                # With covariates (simplified - use logistic regression for propensity)
                from sklearn.linear_model import LogisticRegression
                from sklearn.neighbors import NearestNeighbors
                
                X = df[covariates].apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # Estimate propensity scores
                ps_model = LogisticRegression(max_iter=1000)
                ps_model.fit(X, treatment)
                propensity_scores = ps_model.predict_proba(X)[:, 1]
                
                df['propensity_score'] = propensity_scores
                
                # Matching (1:1 nearest neighbor)
                treated_df = df[treatment == 1]
                control_df = df[treatment == 0]
                
                # Simple matching on propensity scores
                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(control_df[['propensity_score']])
                
                distances, indices = nn.kneighbors(treated_df[['propensity_score']])
                matched_control = control_df.iloc[indices.flatten()]
                
                # Calculate ATE on matched sample
                ate = treated_df[outcome_column].mean() - matched_control[outcome_column].mean()
                
                result["average_treatment_effect"] = float(ate)
                result["n_matched_pairs"] = len(treated_df)
                result["causal_effect"] = float(ate)
        
        elif method == "uplift":
            # Uplift modeling (treatment effect heterogeneity)
            print("  Calculating uplift/treatment effect...")
            
            df = data.to_pandas()
            
            treatment = df[treatment_column]
            outcome = df[outcome_column]
            
            # Calculate uplift by treatment group
            treated_outcome = outcome[treatment == 1].mean()
            control_outcome = outcome[treatment == 0].mean()
            
            uplift = treated_outcome - control_outcome
            
            # Statistical significance
            t_stat, p_value = ttest_ind(
                outcome[treatment == 1],
                outcome[treatment == 0]
            )
            
            result["uplift"] = float(uplift)
            result["treated_mean"] = float(treated_outcome)
            result["control_mean"] = float(control_outcome)
            result["relative_uplift"] = float(uplift / control_outcome * 100) if control_outcome != 0 else 0
            result["t_statistic"] = float(t_stat)
            result["p_value"] = float(p_value)
            result["statistical_significance"] = float(p_value)
            result["causal_effect"] = float(uplift)
            
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'granger', 'propensity', or 'uplift'")
        
        print(f"✅ Causal analysis complete!")
        if result.get("causal_effect") is not None:
            print(f"   Estimated causal effect: {result['causal_effect']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during causal analysis: {str(e)}")
        raise


def generate_business_insights(
    data: pl.DataFrame,
    analysis_type: str,
    analysis_results: Dict[str, Any],
    additional_context: Optional[str] = None,
    groq_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate natural language business insights using Groq LLM.
    
    Args:
        data: Input DataFrame (for context)
        analysis_type: Type of analysis ('rfm', 'cohort', 'causal', 'general')
        analysis_results: Results from previous analysis (dict)
        additional_context: Additional business context
        groq_api_key: Groq API key (if not in environment)
    
    Returns:
        Dictionary containing natural language insights and recommendations
    """
    print(f"🔍 Generating business insights for {analysis_type} analysis...")
    
    # Get API key
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable or pass groq_api_key parameter")
    
    client = Groq(api_key=api_key)
    
    # Prepare data summary
    data_summary = {
        "shape": data.shape,
        "columns": data.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(data.columns, data.dtypes)},
        "sample_stats": {}
    }
    
    # Add numeric column stats
    for col in data.columns:
        if data[col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
            data_summary["sample_stats"][col] = {
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max())
            }
    
    # Create prompt based on analysis type
    prompt = f"""You are a senior business analyst. Analyze the following data and provide actionable business insights.

Analysis Type: {analysis_type.upper()}

Data Summary:
{json.dumps(data_summary, indent=2)}

Analysis Results:
{json.dumps(analysis_results, indent=2)}

Additional Context:
{additional_context or 'None provided'}

Please provide:
1. Key findings (3-5 bullet points)
2. Business implications
3. Actionable recommendations (3-5 specific actions)
4. Risk factors or caveats
5. Suggested next steps

Format your response as a structured business report."""
    
    try:
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior business analyst specializing in data-driven insights and strategic recommendations. Provide clear, actionable insights based on data analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        insights_text = response.choices[0].message.content
        
        # Parse insights (simple structure)
        result = {
            "analysis_type": analysis_type,
            "insights_summary": insights_text,
            "generated_at": datetime.now().isoformat(),
            "model": "llama-3.3-70b-versatile",
            "data_context": data_summary
        }
        
        # Try to extract structured sections
        sections = {}
        current_section = None
        
        for line in insights_text.split('\n'):
            line = line.strip()
            if line.startswith('1.') or line.lower().startswith('key findings'):
                current_section = 'key_findings'
                sections[current_section] = []
            elif line.startswith('2.') or line.lower().startswith('business implications'):
                current_section = 'implications'
                sections[current_section] = []
            elif line.startswith('3.') or line.lower().startswith('actionable recommendations'):
                current_section = 'recommendations'
                sections[current_section] = []
            elif line.startswith('4.') or line.lower().startswith('risk'):
                current_section = 'risks'
                sections[current_section] = []
            elif line.startswith('5.') or line.lower().startswith('next steps'):
                current_section = 'next_steps'
                sections[current_section] = []
            elif current_section and line:
                sections[current_section].append(line)
        
        result["structured_insights"] = sections
        
        print(f"✅ Business insights generated!")
        print(f"   Sections: {', '.join(sections.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating insights: {str(e)}")
        raise
