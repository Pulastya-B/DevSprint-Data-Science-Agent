"""
Intent Classifier - Determines execution mode for the Reasoning Loop.

Three execution modes:
1. DIRECT: "Make a scatter plot" → SBERT routing → tool → done
   - Clear, specific command with obvious tool mapping
   - No reasoning loop needed

2. INVESTIGATIVE: "Why are customers churning?" → reasoning loop
   - Analytical question requiring hypothesis testing
   - Reasoning loop drives tool selection

3. EXPLORATORY: "Analyze this data" → auto-hypothesis → reasoning loop
   - Open-ended request with no specific question
   - First profiles data, generates hypotheses, then investigates

Classification strategy (3-phase):
  Phase 1: Regex fast-path — catches obvious patterns instantly (0ms)
  Phase 2: SBERT semantic similarity — handles novel phrasings (~5ms)
  Phase 3: Keyword heuristic fallback — when SBERT unavailable
"""

import re
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Result of intent classification."""
    mode: str                      # "direct", "investigative", "exploratory"
    confidence: float              # 0.0-1.0
    reasoning: str                 # Why this mode was chosen
    sub_intent: Optional[str]      # More specific intent (e.g., "visualization", "cleaning")


# Patterns that indicate DIRECT mode (specific tool commands)
DIRECT_PATTERNS = [
    # Visualization commands
    (r"\b(make|create|generate|build|show|draw|plot)\b.*(scatter|histogram|heatmap|box\s*plot|bar\s*chart|pie\s*chart|line\s*chart|dashboard|time\s*series)", "visualization"),
    (r"\b(scatter|histogram|heatmap|boxplot|bar\s*chart)\b.*\b(of|for|between|showing)\b", "visualization"),
    
    # Data cleaning commands
    (r"\b(clean|remove|drop|fill|impute|handle)\b.*(missing|null|nan|outlier|duplicate)", "cleaning"),
    (r"\b(fix|convert|change)\b.*(data\s*type|dtype|column\s*type)", "cleaning"),
    
    # Feature engineering commands
    (r"\b(create|add|extract|generate)\b.*(feature|time\s*feature|interaction|encoding)", "feature_engineering"),
    (r"\b(encode|one-hot|label\s*encode|ordinal)\b.*\b(categorical|column)", "feature_engineering"),
    
    # Model training commands
    (r"\b(train|build|fit|run)\b.*(model|classifier|regressor|baseline|xgboost|random\s*forest)", "training"),
    (r"\b(tune|optimize)\b.*\b(hyperparameter|model|parameter)", "training"),
    (r"\b(cross[\s-]?valid)", "training"),
    
    # Profiling commands
    (r"\b(profile|describe|summarize)\b.*\b(dataset|data|table|file)", "profiling"),
    (r"\b(data\s*quality|quality\s*check|check\s*quality)", "profiling"),
    
    # Report generation
    (r"\b(generate|create|build)\b.*\b(report|eda\s*report|profiling\s*report)", "reporting"),
]

# Patterns that indicate INVESTIGATIVE mode (analytical questions)
INVESTIGATIVE_PATTERNS = [
    # Causal / explanatory questions
    (r"\bwhy\b.*(are|is|do|does|did)\b", "causal"),
    (r"\bwhat\b.*(cause|driv|factor|reason|explain|lead)", "causal"),
    (r"\bwhat\b.*(affect|impact|influence|determine)", "causal"),
    
    # Imperative analytical commands ("Explain X", "Identify Y", "Show me what drives Z")
    (r"\b(explain|describe|interpret|assess|evaluate|examine|investigate|understand)\b.*(feature|importance|correlation|distribution|relationship|data|model|pattern|variable|column|factor)", "analytical_imperative"),
    (r"\b(identify|find|determine|show|reveal)\b.*(important|key|significant|driving|top|main|critical|relevant)\b.*(feature|factor|variable|column|predictor|driver)", "feature_importance"),
    (r"\b(feature|variable)\b.*\b(importance|ranking|significance|selection|relevance)", "feature_importance"),
    (r"\b(important|key|significant|top|main|driving)\b.*\b(feature|factor|variable|column|predictor)", "feature_importance"),
    (r"\b(what|which)\b.*\b(feature|variable|column|factor)\b.*\b(important|matter|significant|relevant|impact)", "feature_importance"),
    
    # Relationship / correlation questions
    (r"\bhow\b.*(does|do|is|are)\b.*\b(relate|correlat|affect|impact|change|vary)", "relationship"),
    (r"\b(relationship|correlation|association)\b.*\bbetween\b", "relationship"),
    (r"\bcorrelat", "relationship"),
    
    # Comparison questions
    (r"\b(differ|compar|contrast)\b.*\bbetween\b", "comparison"),
    (r"\bwhich\b.*(better|worse|higher|lower|more|less|best|worst)", "comparison"),
    
    # Pattern / trend questions
    (r"\b(pattern|trend|anomal|outlier|unusual|interesting)\b", "pattern"),
    (r"\bis\s+there\b.*(pattern|trend|relationship|correlation|difference)", "pattern"),
    
    # Prediction-oriented questions (but NOT direct "train a model" commands)
    (r"\bcan\s+(we|i|you)\b.*(predict|forecast|estimate|determine)", "predictive"),
    (r"\bwhat\b.*(predict|forecast|expect|happen)", "predictive"),
    
    # Segmentation / grouping questions
    (r"\b(segment|group|cluster|categori)\b", "segmentation"),
    (r"\bwhat\b.*(type|kind|group|segment)\b.*\b(customer|user|product)", "segmentation"),
]

# Patterns that indicate EXPLORATORY mode (open-ended requests)
EXPLORATORY_PATTERNS = [
    (r"^analyze\b.*\b(this|the|my)\b.*\b(data|dataset|file|csv)", "general_analysis"),
    (r"^(tell|show)\b.*\b(me|us)\b.*\b(about|everything|what)", "general_analysis"),
    (r"^(explore|investigate|examine|look\s*(at|into))\b.*\b(this|the|my)\b", "general_analysis"),
    (r"^what\b.*\b(can|do)\b.*\b(you|we)\b.*\b(find|learn|discover|see)", "general_analysis"),
    (r"^(give|provide)\b.*\b(overview|summary|insight|analysis)", "general_analysis"),
    (r"^(run|do|perform)\b.*\b(full|complete|comprehensive|end.to.end)\b.*\b(analysis|pipeline|workflow)", "full_pipeline"),
    (r"^(find|discover|uncover)\b.*\b(insight|pattern|trend|interesting)", "general_analysis"),
]


# ──────────────────────────────────────────────────────────────────────────────
# SBERT EXEMPLAR QUERIES — one embedding per mode, computed once on first call.
# Add new examples here to improve semantic coverage without touching regex.
# ──────────────────────────────────────────────────────────────────────────────
SBERT_EXEMPLARS: Dict[str, List[str]] = {
    "direct": [
        "Make a scatter plot of age vs income",
        "Create a histogram for the salary column",
        "Generate an EDA report",
        "Build a bar chart showing revenue by region",
        "Clean missing values in the dataset",
        "Handle outliers in the price column",
        "Remove duplicate rows",
        "Encode categorical columns",
        "Fix data types",
        "Train a random forest classifier",
        "Train a model to predict churn",
        "Tune hyperparameters for the best model",
        "Run cross validation on the model",
        "Generate a profiling report",
        "Create a heatmap of correlations",
        "Build a dashboard for this data",
        "Split data into train and test sets",
        "Scale numeric features",
        "Create time-based features from the date column",
        "Export predictions to CSV",
    ],
    "investigative": [
        "Why are customers churning?",
        "What factors drive revenue?",
        "Explain feature importance in this dataset",
        "What is the relationship between price and demand?",
        "Which features are most important for predicting sales?",
        "How does age affect purchase behavior?",
        "What causes high employee attrition?",
        "Identify the key drivers of customer satisfaction",
        "Is there a correlation between marketing spend and conversions?",
        "Compare performance across different segments",
        "What patterns exist in the transaction data?",
        "Are there any anomalies or outliers worth investigating?",
        "Describe the distribution of income across groups",
        "Show me what impacts delivery time the most",
        "Break down the key factors behind loan defaults",
        "Determine which variables matter for this outcome",
        "Assess the statistical significance of these features",
        "Evaluate the relationship between temperature and energy usage",
        "Find what differentiates high-value and low-value customers",
        "Uncover hidden patterns in usage behavior",
    ],
    "exploratory": [
        "Analyze this dataset",
        "What can you find in this data?",
        "Explore the data and tell me what's interesting",
        "Give me an overview of this dataset",
        "Run a full analysis on this file",
        "Look at this data and find insights",
        "Tell me everything about this dataset",
        "Do a comprehensive analysis",
        "What does this data look like?",
        "Examine this CSV and summarize findings",
        "Discover insights from this data",
        "Perform end to end analysis on this dataset",
        "What's in this data?",
        "Summarize the key trends and patterns",
        "Provide a complete data exploration",
    ],
}


class IntentClassifier:
    """
    Classifies user intent into one of three execution modes.
    
    3-phase classification strategy:
      1. Regex fast-path — catches obvious patterns (0ms, ~70% of queries)
      2. SBERT semantic similarity — handles novel phrasings (~5ms)
      3. Keyword heuristic fallback — when SBERT unavailable
    
    When a SemanticLayer is provided (has a loaded SBERT model), exemplar
    queries for each mode are embedded once and cached. New queries are
    classified by cosine similarity to these exemplars — no regex needed.
    
    Usage:
        from src.utils.semantic_layer import get_semantic_layer
        classifier = IntentClassifier(semantic_layer=get_semantic_layer())
        result = classifier.classify("Why are customers churning?")
        # IntentResult(mode="investigative", confidence=0.9, ...)
        
        # Also works without SBERT (regex + heuristic only):
        classifier = IntentClassifier()
        result = classifier.classify("Make a scatter plot of age vs income")
    """
    
    def __init__(self, semantic_layer=None):
        """
        Args:
            semantic_layer: Optional SemanticLayer instance with loaded SBERT model.
                           If provided, enables semantic intent classification.
        """
        self.semantic_layer = semantic_layer
        self._exemplar_embeddings = None  # Lazy-computed: {mode: np.ndarray}
    
    def _ensure_exemplar_embeddings(self):
        """Lazily compute and cache SBERT embeddings for exemplar queries."""
        if self._exemplar_embeddings is not None:
            return
        if not self.semantic_layer or not self.semantic_layer.enabled:
            return
        
        try:
            self._exemplar_embeddings = {}
            for mode, exemplars in SBERT_EXEMPLARS.items():
                embeddings = self.semantic_layer.model.encode(
                    exemplars, convert_to_numpy=True, 
                    show_progress_bar=False, batch_size=32
                )
                self._exemplar_embeddings[mode] = embeddings  # shape: (N, dim)
            
            total = sum(len(v) for v in SBERT_EXEMPLARS.values())
            print(f"   🧠 IntentClassifier: Cached {total} exemplar embeddings across 3 modes")
        except Exception as e:
            print(f"   ⚠️ IntentClassifier: Failed to encode exemplars: {e}")
            self._exemplar_embeddings = None
    
    def _classify_sbert(self, query: str) -> Optional[IntentResult]:
        """
        Classify intent using SBERT semantic similarity to exemplar queries.
        
        For each mode, compute cosine similarity of the query to all exemplars
        in that mode, then take the max. The mode with the highest max-sim wins.
        
        Returns None if SBERT is unavailable or classification is ambiguous.
        """
        self._ensure_exemplar_embeddings()
        if self._exemplar_embeddings is None:
            return None
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity as cos_sim
            
            query_emb = self.semantic_layer.model.encode(
                query, convert_to_numpy=True, show_progress_bar=False
            ).reshape(1, -1)
            
            mode_scores = {}
            mode_best_exemplar = {}
            for mode, exemplar_embs in self._exemplar_embeddings.items():
                sims = cos_sim(query_emb, exemplar_embs)[0]  # shape: (N,)
                best_idx = int(np.argmax(sims))
                mode_scores[mode] = float(sims[best_idx])
                mode_best_exemplar[mode] = SBERT_EXEMPLARS[mode][best_idx]
            
            # Pick mode with highest score
            best_mode = max(mode_scores, key=mode_scores.get)
            best_score = mode_scores[best_mode]
            runner_up = sorted(mode_scores.values(), reverse=True)[1]
            margin = best_score - runner_up
            
            # Require minimum similarity AND reasonable margin
            if best_score < 0.35:
                # Too low similarity to any mode — fall through
                return None
            
            # Map raw cosine similarity (typically 0.4-0.9) to confidence (0.6-0.95)
            confidence = min(0.95, 0.55 + best_score * 0.45)
            
            # If margin is very thin, lower confidence
            if margin < 0.05:
                confidence = min(confidence, 0.60)
            
            best_match = mode_best_exemplar[best_mode]
            
            return IntentResult(
                mode=best_mode,
                confidence=round(confidence, 2),
                reasoning=f"SBERT semantic match (sim={best_score:.3f}, margin={margin:.3f}, closest: \"{best_match[:60]}\")",
                sub_intent="sbert_semantic"
            )
        except Exception as e:
            print(f"   ⚠️ SBERT classification failed: {e}")
            return None
    
    def classify(
        self, 
        query: str, 
        dataset_info: Optional[Dict[str, Any]] = None,
        has_target_col: bool = False
    ) -> IntentResult:
        """
        Classify user intent into execution mode.
        
        3-phase strategy:
          Phase 1: Regex fast-path (catches ~70% of queries, 0ms)
          Phase 2: SBERT semantic similarity (handles novel phrasings, ~5ms)
          Phase 3: Keyword heuristic fallback (when SBERT unavailable)
        
        Args:
            query: User's natural language query
            dataset_info: Optional dataset schema info
            has_target_col: Whether user provided a target column
            
        Returns:
            IntentResult with mode, confidence, and reasoning
        """
        query_lower = query.lower().strip()
        
        # ── Phase 1: Regex fast-path (strongest evidence, instant) ──
        direct_match = self._match_patterns(query_lower, DIRECT_PATTERNS)
        if direct_match:
            pattern, sub_intent = direct_match
            return IntentResult(
                mode="direct",
                confidence=0.90,
                reasoning=f"Direct command detected: {sub_intent} (pattern: {pattern[:50]})",
                sub_intent=sub_intent
            )
        
        invest_match = self._match_patterns(query_lower, INVESTIGATIVE_PATTERNS)
        if invest_match:
            pattern, sub_intent = invest_match
            return IntentResult(
                mode="investigative",
                confidence=0.85,
                reasoning=f"Analytical question detected: {sub_intent}",
                sub_intent=sub_intent
            )
        
        explore_match = self._match_patterns(query_lower, EXPLORATORY_PATTERNS)
        if explore_match:
            pattern, sub_intent = explore_match
            
            if sub_intent == "full_pipeline" and has_target_col:
                return IntentResult(
                    mode="direct",
                    confidence=0.85,
                    reasoning="Full ML pipeline requested with target column",
                    sub_intent="full_ml_pipeline"
                )
            
            return IntentResult(
                mode="exploratory",
                confidence=0.80,
                reasoning=f"Open-ended analysis request: {sub_intent}",
                sub_intent=sub_intent
            )
        
        # ── Phase 2: SBERT semantic classification (handles novel queries) ──
        sbert_result = self._classify_sbert(query)
        if sbert_result:
            # Apply special-case overrides
            if sbert_result.mode == "direct" and has_target_col:
                # If SBERT says direct but there's a target col + ML verbs, boost confidence
                if any(w in query_lower for w in ["predict", "train", "model", "classify"]):
                    sbert_result.confidence = max(sbert_result.confidence, 0.80)
            return sbert_result
        
        # ── Phase 3: Keyword heuristic fallback (no SBERT available) ──
        return self._heuristic_classify(query_lower, has_target_col)

    def _match_patterns(self, query: str, patterns: list) -> Optional[Tuple[str, str]]:
        """Try to match query against a list of (pattern, sub_intent) tuples."""
        for pattern, sub_intent in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return (pattern, sub_intent)
        return None

    def _heuristic_classify(self, query: str, has_target_col: bool) -> IntentResult:
        """Fallback classification using simple heuristics."""
        
        # Question words → investigative
        if query.startswith(("why", "how", "what", "which", "is there", "are there", "does", "do")):
            return IntentResult(
                mode="investigative",
                confidence=0.60,
                reasoning="Query starts with question word, likely analytical",
                sub_intent="general_question"
            )
        
        # Analytical imperative verbs → investigative
        if query.startswith(("explain", "describe", "interpret", "identify",
                            "assess", "evaluate", "examine", "investigate",
                            "determine", "understand", "show me", "tell me",
                            "find the", "reveal", "uncover")):
            return IntentResult(
                mode="investigative",
                confidence=0.70,
                reasoning="Analytical imperative verb detected, likely investigative",
                sub_intent="analytical_imperative"
            )
        
        # Very short queries → likely direct commands
        word_count = len(query.split())
        if word_count <= 5:
            return IntentResult(
                mode="direct",
                confidence=0.55,
                reasoning="Short query, likely a direct command",
                sub_intent="short_command"
            )
        
        # Has target column + action verbs → direct ML pipeline
        if has_target_col and any(w in query for w in ["predict", "train", "model", "classify", "regression"]):
            return IntentResult(
                mode="direct",
                confidence=0.75,
                reasoning="Target column provided with ML action verb",
                sub_intent="ml_pipeline"
            )
        
        # Default: exploratory (safest default for data science)
        return IntentResult(
            mode="exploratory",
            confidence=0.40,
            reasoning="No strong pattern match, defaulting to exploratory analysis",
            sub_intent="default"
        )

    @staticmethod
    def is_follow_up(query: str) -> bool:
        """
        Detect if this is a follow-up question (uses context from previous analysis).
        
        Follow-ups should generally be INVESTIGATIVE (they're asking about
        something specific in the context of previous results).
        """
        follow_up_patterns = [
            r"^(now|next|also|and|then)\b",
            r"\b(the same|that|this|those|these)\b.*\b(data|model|result|plot|chart)",
            r"\b(more|another|different)\b.*\b(plot|chart|analysis|model)",
            r"\b(what about|how about|can you also)\b",
            r"\b(using|with)\b.*\b(the same|that|this)\b",
        ]
        
        query_lower = query.lower().strip()
        return any(re.search(p, query_lower) for p in follow_up_patterns)
