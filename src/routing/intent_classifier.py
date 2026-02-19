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

The classifier uses keyword patterns + semantic features to decide.
This is a lightweight classification (no LLM call needed).
"""

import re
from typing import Optional, Dict, Any, Tuple
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
    
    # Relationship / correlation questions
    (r"\bhow\b.*(does|do|is|are)\b.*\b(relate|correlat|affect|impact|change|vary)", "relationship"),
    (r"\b(relationship|correlation|association)\b.*\bbetween\b", "relationship"),
    
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


class IntentClassifier:
    """
    Classifies user intent into one of three execution modes.
    
    Uses pattern matching (no LLM call needed) for fast classification.
    Falls back to heuristics when patterns don't match.
    
    Usage:
        classifier = IntentClassifier()
        result = classifier.classify("Why are customers churning?")
        # IntentResult(mode="investigative", confidence=0.9, ...)
        
        result = classifier.classify("Make a scatter plot of age vs income")
        # IntentResult(mode="direct", confidence=0.95, ...)
        
        result = classifier.classify("Analyze this dataset")
        # IntentResult(mode="exploratory", confidence=0.85, ...)
    """
    
    def classify(
        self, 
        query: str, 
        dataset_info: Optional[Dict[str, Any]] = None,
        has_target_col: bool = False
    ) -> IntentResult:
        """
        Classify user intent into execution mode.
        
        Args:
            query: User's natural language query
            dataset_info: Optional dataset schema info
            has_target_col: Whether user provided a target column
            
        Returns:
            IntentResult with mode, confidence, and reasoning
        """
        query_lower = query.lower().strip()
        
        # Phase 1: Check for DIRECT patterns (strongest evidence)
        direct_match = self._match_patterns(query_lower, DIRECT_PATTERNS)
        if direct_match:
            pattern, sub_intent = direct_match
            return IntentResult(
                mode="direct",
                confidence=0.90,
                reasoning=f"Direct command detected: {sub_intent} (pattern: {pattern[:50]})",
                sub_intent=sub_intent
            )
        
        # Phase 2: Check for INVESTIGATIVE patterns
        invest_match = self._match_patterns(query_lower, INVESTIGATIVE_PATTERNS)
        if invest_match:
            pattern, sub_intent = invest_match
            return IntentResult(
                mode="investigative",
                confidence=0.85,
                reasoning=f"Analytical question detected: {sub_intent}",
                sub_intent=sub_intent
            )
        
        # Phase 3: Check for EXPLORATORY patterns
        explore_match = self._match_patterns(query_lower, EXPLORATORY_PATTERNS)
        if explore_match:
            pattern, sub_intent = explore_match
            
            # Special case: "full pipeline" with target col → direct ML pipeline
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
        
        # Phase 4: Heuristic fallback
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
