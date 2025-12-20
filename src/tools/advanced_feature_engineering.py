"""
Advanced Feature Engineering Tools
Tools for creating interaction features, aggregations, text features, and auto feature engineering.
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys
import os
import json
import warnings
from itertools import combinations

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest, f_classif, f_regression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import re

from utils.polars_helpers import (
    load_dataframe, save_dataframe, get_numeric_columns,
    get_categorical_columns, get_datetime_columns
)
from utils.validation import (
    validate_file_exists, validate_file_format, validate_dataframe,
    validate_column_exists
)


def create_interaction_features(
    file_path: str,
    method: str = "polynomial",
    degree: int = 2,
    n_components: Optional[int] = None,
    columns: Optional[List[str]] = None,
    max_features: int = 50,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create interaction features using polynomial features, PCA, or feature crossing.
    
    Args:
        file_path: Path to dataset
        method: Feature interaction method:
            - 'polynomial': Polynomial features (degree 2 or 3)
            - 'pca': Principal Component Analysis
            - 'cross': Manual feature crossing (multiply pairs)
            - 'mutual_info': Select best features by mutual information
        degree: Polynomial degree (for polynomial method)
        n_components: Number of components (for PCA, None = auto)
        columns: Columns to use (None = all numeric)
        max_features: Maximum number of new features to create
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with feature engineering results
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    
    # Get numeric columns if not specified
    if columns is None:
        columns = get_numeric_columns(df)
        print(f"🔢 Auto-detected {len(columns)} numeric columns")
    else:
        for col in columns:
            validate_column_exists(df, col)
    
    if not columns:
        return {
            'status': 'skipped',
            'message': 'No numeric columns found for interaction features'
        }
    
    # Limit columns if too many
    if len(columns) > 20:
        print(f"⚠️ Too many columns ({len(columns)}). Using top 20 by variance.")
        variances = df[columns].select([
            (pl.col(col).var().alias(col)) for col in columns
        ]).to_dicts()[0]
        columns = sorted(variances.keys(), key=lambda x: variances[x], reverse=True)[:20]
    
    X = df[columns].to_numpy()
    original_features = len(columns)
    
    # Create interaction features based on method
    if method == "polynomial":
        print(f"🔄 Creating polynomial features (degree={degree})...")
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Limit features
        if X_poly.shape[1] > max_features + original_features:
            # Keep original + top max_features new ones by variance
            variances = np.var(X_poly[:, original_features:], axis=0)
            top_indices = np.argsort(variances)[::-1][:max_features]
            X_new = np.hstack([X, X_poly[:, original_features:][:, top_indices]])
            new_feature_names = [feature_names[i + original_features] for i in top_indices]
        else:
            X_new = X_poly
            new_feature_names = feature_names[original_features:].tolist()
        
        # Create new dataframe
        df_new = df.clone()
        for i, name in enumerate(new_feature_names):
            clean_name = name.replace(' ', '_').replace('^', '_pow_')
            df_new = df_new.with_columns(
                pl.Series(f"poly_{clean_name}", X_new[:, original_features + i])
            )
        
        created_features = new_feature_names
        
    elif method == "pca":
        print(f"🔄 Creating PCA features...")
        if n_components is None:
            n_components = min(len(columns), max_features)
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create new dataframe
        df_new = df.clone()
        for i in range(n_components):
            df_new = df_new.with_columns(
                pl.Series(f"pca_{i+1}", X_pca[:, i])
            )
        
        created_features = [f"pca_{i+1}" for i in range(n_components)]
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
    elif method == "cross":
        print(f"🔄 Creating feature crosses...")
        # Create pairwise interactions
        pairs = list(combinations(columns, 2))
        
        # Limit number of pairs
        if len(pairs) > max_features:
            pairs = pairs[:max_features]
        
        df_new = df.clone()
        created_features = []
        
        for col1, col2 in pairs:
            new_name = f"{col1}_x_{col2}"
            df_new = df_new.with_columns(
                (pl.col(col1) * pl.col(col2)).alias(new_name)
            )
            created_features.append(new_name)
        
    elif method == "mutual_info":
        print(f"🔄 Selecting features by mutual information...")
        # This requires a target column - for now, create interaction features
        # and let the user select based on their target
        return {
            'status': 'error',
            'message': 'mutual_info method requires a target column. Use polynomial or cross instead, then use feature selection.'
        }
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Save if output path provided
    if output_path:
        save_dataframe(df_new, output_path)
        print(f"💾 Dataset with interaction features saved to: {output_path}")
    
    result = {
        'status': 'success',
        'method': method,
        'original_features': original_features,
        'new_features_created': len(created_features),
        'total_features': len(df_new.columns),
        'feature_names': created_features[:20],  # Show first 20
        'output_path': output_path
    }
    
    if method == "pca":
        result['explained_variance_ratio'] = explained_variance.tolist()
        result['cumulative_variance'] = cumulative_variance.tolist()
        result['variance_explained_by_top_5'] = float(cumulative_variance[min(4, len(cumulative_variance)-1)])
    
    return result


def create_aggregation_features(
    file_path: str,
    group_col: str,
    agg_columns: Optional[List[str]] = None,
    agg_functions: Optional[List[str]] = None,
    rolling_window: Optional[int] = None,
    time_col: Optional[str] = None,
    lag_periods: Optional[List[int]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create aggregation features including group-by aggregations, rolling windows, and lags.
    
    Args:
        file_path: Path to dataset
        group_col: Column to group by (e.g., 'customer_id', 'category')
        agg_columns: Columns to aggregate (None = all numeric)
        agg_functions: Aggregation functions ('mean', 'sum', 'std', 'min', 'max', 'count')
        rolling_window: Window size for rolling aggregations (requires sorted data)
        time_col: Time column for sorting (required for rolling/lag features)
        lag_periods: Lag periods to create (e.g., [1, 7, 30] for 1-day, 7-day, 30-day lags)
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with aggregation results
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, group_col)
    
    if time_col:
        validate_column_exists(df, time_col)
        df = df.sort(time_col)
    
    # Get numeric columns if not specified
    if agg_columns is None:
        agg_columns = [col for col in get_numeric_columns(df) if col != group_col]
        print(f"🔢 Auto-detected {len(agg_columns)} numeric columns for aggregation")
    else:
        for col in agg_columns:
            validate_column_exists(df, col)
    
    if not agg_columns:
        return {
            'status': 'skipped',
            'message': 'No numeric columns found for aggregation'
        }
    
    # Default aggregation functions
    if agg_functions is None:
        agg_functions = ['mean', 'sum', 'std', 'min', 'max', 'count']
    
    df_new = df.clone()
    created_features = []
    
    # Group-by aggregations
    print(f"📊 Creating group-by aggregations for {group_col}...")
    
    for agg_col in agg_columns:
        for agg_func in agg_functions:
            try:
                if agg_func == 'mean':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).mean().alias(f"{agg_col}_{group_col}_mean")
                    )
                elif agg_func == 'sum':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).sum().alias(f"{agg_col}_{group_col}_sum")
                    )
                elif agg_func == 'std':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).std().alias(f"{agg_col}_{group_col}_std")
                    )
                elif agg_func == 'min':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).min().alias(f"{agg_col}_{group_col}_min")
                    )
                elif agg_func == 'max':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).max().alias(f"{agg_col}_{group_col}_max")
                    )
                elif agg_func == 'count':
                    agg_df = df.group_by(group_col).agg(
                        pl.col(agg_col).count().alias(f"{agg_col}_{group_col}_count")
                    )
                else:
                    continue
                
                # Join back to original dataframe
                df_new = df_new.join(agg_df, on=group_col, how='left')
                created_features.append(f"{agg_col}_{group_col}_{agg_func}")
            except Exception as e:
                print(f"⚠️ Skipping {agg_col}_{agg_func}: {str(e)}")
    
    # Rolling window features
    if rolling_window and time_col:
        print(f"📈 Creating rolling window features (window={rolling_window})...")
        
        for agg_col in agg_columns[:5]:  # Limit to first 5 columns to avoid explosion
            try:
                # Rolling mean
                df_new = df_new.with_columns(
                    pl.col(agg_col).rolling_mean(window_size=rolling_window)
                    .over(group_col)
                    .alias(f"{agg_col}_rolling_{rolling_window}_mean")
                )
                created_features.append(f"{agg_col}_rolling_{rolling_window}_mean")
                
                # Rolling std
                df_new = df_new.with_columns(
                    pl.col(agg_col).rolling_std(window_size=rolling_window)
                    .over(group_col)
                    .alias(f"{agg_col}_rolling_{rolling_window}_std")
                )
                created_features.append(f"{agg_col}_rolling_{rolling_window}_std")
            except Exception as e:
                print(f"⚠️ Skipping rolling for {agg_col}: {str(e)}")
    
    # Lag features
    if lag_periods and time_col:
        print(f"⏰ Creating lag features (periods={lag_periods})...")
        
        for agg_col in agg_columns[:5]:  # Limit to avoid explosion
            for lag in lag_periods:
                try:
                    df_new = df_new.with_columns(
                        pl.col(agg_col).shift(lag)
                        .over(group_col)
                        .alias(f"{agg_col}_lag_{lag}")
                    )
                    created_features.append(f"{agg_col}_lag_{lag}")
                except Exception as e:
                    print(f"⚠️ Skipping lag {lag} for {agg_col}: {str(e)}")
    
    # Save if output path provided
    if output_path:
        save_dataframe(df_new, output_path)
        print(f"💾 Dataset with aggregation features saved to: {output_path}")
    
    return {
        'status': 'success',
        'group_column': group_col,
        'aggregated_columns': agg_columns,
        'aggregation_functions': agg_functions,
        'new_features_created': len(created_features),
        'total_features': len(df_new.columns),
        'feature_names': created_features[:30],  # Show first 30
        'rolling_window': rolling_window,
        'lag_periods': lag_periods,
        'output_path': output_path
    }


def engineer_text_features(
    file_path: str,
    text_column: str,
    methods: Optional[List[str]] = None,
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 2),
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract features from text columns using TF-IDF, n-grams, and text statistics.
    
    Args:
        file_path: Path to dataset
        text_column: Name of text column
        methods: List of methods to apply:
            - 'tfidf': TF-IDF vectorization
            - 'count': Count vectorization (bag of words)
            - 'sentiment': Sentiment analysis
            - 'stats': Text statistics (length, word count, etc.)
            - 'ngrams': N-gram features
        max_features: Maximum number of TF-IDF/count features
        ngram_range: N-gram range (e.g., (1, 2) for unigrams and bigrams)
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with text feature engineering results
    """
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, text_column)
    
    # Default methods
    if methods is None:
        methods = ['stats', 'sentiment', 'tfidf']
    
    df_new = df.clone()
    created_features = []
    
    # Get text data
    texts = df[text_column].fill_null("").to_list()
    
    # Text statistics
    if 'stats' in methods:
        print("📝 Extracting text statistics...")
        
        char_counts = [len(str(text)) for text in texts]
        word_counts = [len(str(text).split()) for text in texts]
        avg_word_lengths = [np.mean([len(word) for word in str(text).split()]) if text else 0 for text in texts]
        special_char_counts = [len(re.findall(r'[^a-zA-Z0-9\s]', str(text))) for text in texts]
        digit_counts = [len(re.findall(r'\d', str(text))) for text in texts]
        uppercase_counts = [len(re.findall(r'[A-Z]', str(text))) for text in texts]
        
        df_new = df_new.with_columns([
            pl.Series(f"{text_column}_char_count", char_counts),
            pl.Series(f"{text_column}_word_count", word_counts),
            pl.Series(f"{text_column}_avg_word_length", avg_word_lengths),
            pl.Series(f"{text_column}_special_char_count", special_char_counts),
            pl.Series(f"{text_column}_digit_count", digit_counts),
            pl.Series(f"{text_column}_uppercase_count", uppercase_counts)
        ])
        
        created_features.extend([
            f"{text_column}_char_count",
            f"{text_column}_word_count",
            f"{text_column}_avg_word_length",
            f"{text_column}_special_char_count",
            f"{text_column}_digit_count",
            f"{text_column}_uppercase_count"
        ])
    
    # Sentiment analysis
    if 'sentiment' in methods:
        print("💭 Performing sentiment analysis...")
        
        sentiments = []
        subjectivities = []
        
        for text in texts:
            try:
                blob = TextBlob(str(text))
                sentiments.append(blob.sentiment.polarity)
                subjectivities.append(blob.sentiment.subjectivity)
            except:
                sentiments.append(0.0)
                subjectivities.append(0.0)
        
        df_new = df_new.with_columns([
            pl.Series(f"{text_column}_sentiment", sentiments),
            pl.Series(f"{text_column}_subjectivity", subjectivities)
        ])
        
        created_features.extend([
            f"{text_column}_sentiment",
            f"{text_column}_subjectivity"
        ])
    
    # TF-IDF features
    if 'tfidf' in methods:
        print(f"🔤 Creating TF-IDF features (max_features={max_features})...")
        
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2
        )
        
        try:
            tfidf_matrix = tfidf.fit_transform([str(text) for text in texts])
            feature_names = tfidf.get_feature_names_out()
            
            # Add TF-IDF features to dataframe
            for i, feature_name in enumerate(feature_names):
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature_name)[:30]
                df_new = df_new.with_columns(
                    pl.Series(f"tfidf_{clean_name}", tfidf_matrix[:, i].toarray().flatten())
                )
                created_features.append(f"tfidf_{clean_name}")
        except Exception as e:
            print(f"⚠️ TF-IDF failed: {str(e)}")
    
    # Count vectorization
    if 'count' in methods:
        print(f"🔢 Creating count features (max_features={max_features})...")
        
        count_vec = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2
        )
        
        try:
            count_matrix = count_vec.fit_transform([str(text) for text in texts])
            feature_names = count_vec.get_feature_names_out()
            
            # Add count features to dataframe
            for i, feature_name in enumerate(feature_names[:50]):  # Limit to 50
                clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature_name)[:30]
                df_new = df_new.with_columns(
                    pl.Series(f"count_{clean_name}", count_matrix[:, i].toarray().flatten())
                )
                created_features.append(f"count_{clean_name}")
        except Exception as e:
            print(f"⚠️ Count vectorization failed: {str(e)}")
    
    # Save if output path provided
    if output_path:
        save_dataframe(df_new, output_path)
        print(f"💾 Dataset with text features saved to: {output_path}")
    
    return {
        'status': 'success',
        'text_column': text_column,
        'methods_applied': methods,
        'new_features_created': len(created_features),
        'total_features': len(df_new.columns),
        'feature_names': created_features[:30],  # Show first 30
        'output_path': output_path
    }


def auto_feature_engineering(
    file_path: str,
    target_col: str,
    groq_api_key: Optional[str] = None,
    max_suggestions: int = 10,
    implement_top_k: int = 5,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Use LLM (Groq or Gemini) to automatically generate and implement feature engineering ideas.
    
    Args:
        file_path: Path to dataset
        target_col: Target column name
        groq_api_key: Groq API key (optional - will try to use environment variable or Gemini)
        max_suggestions: Maximum number of feature suggestions to generate
        implement_top_k: Number of top suggestions to implement
        output_path: Path to save dataset with new features
        
    Returns:
        Dictionary with feature suggestions and implementation results
    """
    import os
    
    # Validation
    validate_file_exists(file_path)
    validate_file_format(file_path)
    
    # Load data
    df = load_dataframe(file_path)
    validate_dataframe(df)
    validate_column_exists(df, target_col)
    
    # Get dataset summary
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    # Sample data for analysis
    sample_df = df.head(5)
    
    # Create prompt for LLM
    prompt = f"""You are a data science expert. Analyze this dataset and suggest {max_suggestions} creative feature engineering ideas.

Dataset Overview:
- Target column: {target_col}
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:5])}
- Rows: {len(df)}

Sample data (first 5 rows):
{sample_df.head(5)}

Suggest {max_suggestions} feature engineering ideas that could improve model performance. For each idea:
1. Describe the feature clearly
2. Provide Python code using Polars to create it
3. Explain why it might be valuable

Format your response as JSON:
{{
  "suggestions": [
    {{
      "name": "feature_name",
      "description": "what it does",
      "code": "pl.col('a') * pl.col('b')",
      "reasoning": "why it helps"
    }}
  ]
}}
"""
    
    print("🤖 Asking LLM for feature engineering suggestions...")
    
    # Try multiple LLM providers in order of preference
    llm_response = None
    
    # Try Groq first if API key provided
    if groq_api_key or os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            api_key = groq_api_key or os.getenv("GROQ_API_KEY")
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            llm_response = response.choices[0].message.content
            print("   ✓ Using Groq LLM")
        except Exception as e:
            print(f"   ⚠️ Groq failed: {str(e)}, trying Gemini...")
    
    # Try Gemini if Groq failed or not available
    if not llm_response and os.getenv("GEMINI_API_KEY"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            llm_response = response.text
            print("   ✓ Using Gemini LLM")
        except Exception as e:
            print(f"   ⚠️ Gemini failed: {str(e)}")
    
    if not llm_response:
        return {
            "status": "error",
            "message": "No LLM API key available. Set GROQ_API_KEY or GEMINI_API_KEY environment variable."
        }
    
    try:
        # Parse JSON response
        import json
        # Extract JSON from response (might be wrapped in markdown code blocks)
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()
        
        suggestions = json.loads(llm_response)
        
        # Implement top K suggestions
        df_new = df.clone()
        implemented = []
        
        for i, suggestion in enumerate(suggestions['suggestions'][:implement_top_k]):
            try:
                # Execute feature creation code
                feature_name = suggestion['name']
                code = suggestion['code']
                
                # Create new column using eval (be careful in production!)
                df_new = df_new.with_columns(
                    eval(code).alias(feature_name)
                )
                
                implemented.append({
                    'name': feature_name,
                    'description': suggestion['description'],
                    'reasoning': suggestion['reasoning']
                })
                
                print(f"✅ Implemented: {feature_name}")
            except Exception as e:
                print(f"⚠️ Failed to implement {suggestion.get('name', 'unknown')}: {str(e)}")
        
        # Save if output path provided
        if output_path:
            save_dataframe(df_new, output_path)
            print(f"💾 Dataset with auto-generated features saved to: {output_path}")
        
        return {
            'status': 'success',
            'total_suggestions': len(suggestions['suggestions']),
            'suggestions': suggestions['suggestions'],
            'implemented': implemented,
            'new_features_created': len(implemented),
            'output_path': output_path
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Auto feature engineering failed: {str(e)}"
        }
