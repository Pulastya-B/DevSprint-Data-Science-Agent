"""
Routing Module - Intent Classification and Request Routing.

Determines how the orchestrator should handle a user request:
- Direct: SBERT routing → tool execution (existing pipeline)
- Investigative: Reasoning loop with hypothesis testing
- Exploratory: Auto-hypothesis generation → reasoning loop
"""

from .intent_classifier import IntentClassifier, IntentResult

__all__ = ["IntentClassifier", "IntentResult"]
