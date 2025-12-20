"""
Reasoning Module - Core Abstraction

Provides clean separation between:
- Deterministic data processing (tools)
- Non-deterministic reasoning (LLM)

Design Principles:
- NO RAW DATA ACCESS - Only summaries/metadata
- NO TRAINING DECISIONS - Only explanations
- STRUCTURED I/O - JSON in, JSON + text out
- CACHEABLE - Deterministic enough to cache
- REASONING ONLY - No execution, no side effects

Architecture:
    Tool → Generates Summary → Reasoning Module → Returns Explanation
    
    Tool: "Here's what I found: {stats}"
    Reasoning: "Based on these stats, this means..."
    
Usage:
    from reasoning import get_reasoner
    
    reasoner = get_reasoner()
    result = reasoner.explain_data(
        summary={"rows": 1000, "columns": 20, "missing": 50}
    )
"""

import os
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod


class ReasoningBackend(ABC):
    """Abstract base class for reasoning backends."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """Generate reasoning response."""
        pass
    
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        pass


class GeminiBackend(ReasoningBackend):
    """Gemini reasoning backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-exp"):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model,
            generation_config={"temperature": 0.1}
        )
        self.model_name = model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """Generate reasoning response."""
        # Combine system and user prompts
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.model.generate_content(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        
        return response.text
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        import json
        
        # Add schema instruction
        schema_str = json.dumps(schema, indent=2)
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_str}

Your response must be valid JSON only, no other text."""
        
        response_text = self.generate(structured_prompt, system_prompt)
        
        # Extract JSON from response
        try:
            # Try direct parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to extract any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError(f"Failed to extract JSON from response: {response_text[:200]}...")


class GroqBackend(ReasoningBackend):
    """Groq reasoning backend."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq not installed. "
                "Install with: pip install groq"
            )
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key"
            )
        
        self.client = Groq(api_key=api_key)
        self.model_name = model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """Generate reasoning response."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        import json
        
        # Add schema instruction
        schema_str = json.dumps(schema, indent=2)
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_str}

Your response must be valid JSON only, no other text."""
        
        response_text = self.generate(structured_prompt, system_prompt)
        
        # Extract JSON from response
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to extract any JSON object
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            raise ValueError(f"Failed to extract JSON from response: {response_text[:200]}...")


class ReasoningEngine:
    """
    Main reasoning engine.
    
    Delegates to appropriate backend (Gemini, Groq, etc).
    Provides high-level reasoning capabilities.
    """
    
    def __init__(
        self,
        backend: Optional[ReasoningBackend] = None,
        provider: str = "gemini"
    ):
        """
        Initialize reasoning engine.
        
        Args:
            backend: Custom backend instance
            provider: 'gemini' or 'groq' (if backend not provided)
        """
        if backend:
            self.backend = backend
        else:
            provider = provider or os.getenv("LLM_PROVIDER", "gemini")
            
            if provider == "gemini":
                self.backend = GeminiBackend()
            elif provider == "groq":
                self.backend = GroqBackend()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        self.provider = provider
    
    def reason(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """
        General-purpose reasoning.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system context
            temperature: Creativity (0.0 = deterministic, 1.0 = creative)
            
        Returns:
            Natural language response
        """
        return self.backend.generate(prompt, system_prompt, temperature)
    
    def reason_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Structured reasoning with JSON output.
        
        Args:
            prompt: User prompt
            schema: Expected JSON schema
            system_prompt: Optional system context
            
        Returns:
            Parsed JSON response
        """
        return self.backend.generate_structured(prompt, schema, system_prompt)


# Singleton instance
_reasoning_engine: Optional[ReasoningEngine] = None


def get_reasoner(
    backend: Optional[ReasoningBackend] = None,
    provider: Optional[str] = None
) -> ReasoningEngine:
    """
    Get singleton reasoning engine.
    
    Args:
        backend: Custom backend instance
        provider: 'gemini' or 'groq'
        
    Returns:
        ReasoningEngine instance
    """
    global _reasoning_engine
    
    if _reasoning_engine is None or backend is not None:
        _reasoning_engine = ReasoningEngine(backend=backend, provider=provider)
    
    return _reasoning_engine


def reset_reasoner():
    """Reset singleton (for testing)."""
    global _reasoning_engine
    _reasoning_engine = None
