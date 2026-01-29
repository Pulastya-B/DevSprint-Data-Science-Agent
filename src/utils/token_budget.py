"""
Strict Token Budget Management

Implements sliding window conversation history, aggressive compression,
and emergency context truncation to prevent context window overflow.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import tiktoken
from pathlib import Path


class ConversationMessage:
    """Represents a message with priority for history management."""
    
    def __init__(self, role: str, content: str, message_type: str = "normal",
                 priority: int = 5, tokens: Optional[int] = None):
        self.role = role
        self.content = content
        self.message_type = message_type  # system, tool_result, assistant, user, normal
        self.priority = priority  # 1 (drop first) to 10 (keep last)
        self.tokens = tokens
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}


class TokenBudgetManager:
    """
    Manages conversation history with strict token budget enforcement.
    
    Features:
    - Accurate token counting using tiktoken
    - Priority-based message dropping
    - Sliding window with smart compression
    - Emergency context truncation
    - Keeps recent tool results, drops old assistant messages
    """
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 128000,
                 reserve_tokens: int = 8000):
        """
        Initialize token budget manager.
        
        Args:
            model: Model name for token counting
            max_tokens: Maximum context window size
            reserve_tokens: Tokens to reserve for response
        """
        self.model = model
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.available_tokens = max_tokens - reserve_tokens
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except:
            # Fallback to cl100k_base (GPT-4/GPT-3.5)
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        print(f"📊 Token Budget: {self.available_tokens:,} tokens available ({self.max_tokens:,} - {self.reserve_tokens:,} reserve)")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            return len(self.encoding.encode(text))
        except:
            # Fallback estimation: ~4 chars per token
            return len(text) // 4
    
    def count_message_tokens(self, message) -> int:
        """
        Count tokens in a message (includes role overhead).
        
        Args:
            message: Either a dict or a Pydantic ChatMessage object
        """
        # Format: <|role|>content<|endofmessage|>
        # Approximately 4 tokens overhead per message
        
        # Handle both dict and Pydantic object formats
        if isinstance(message, dict):
            content = message.get("content", "")
            role = message.get("role", "")
        else:
            # Pydantic object (like ChatMessage from Mistral SDK)
            content = getattr(message, "content", "")
            role = getattr(message, "role", "")
        
        content_tokens = self.count_tokens(str(content))
        role_tokens = self.count_tokens(str(role))
        return content_tokens + role_tokens + 4
    
    def count_messages_tokens(self, messages: List) -> int:
        """Count total tokens in message list."""
        return sum(self.count_message_tokens(msg) for msg in messages)
    
    def compress_tool_result(self, tool_result: str, max_tokens: int = 500) -> str:
        """
        Aggressively compress tool result while keeping key information.
        
        Keeps:
        - Success/failure status
        - Key metrics and numbers
        - Error messages
        
        Drops:
        - Verbose logs
        - Duplicate information
        - Large data structures
        """
        if self.count_tokens(tool_result) <= max_tokens:
            return tool_result
        
        try:
            # Try to parse as JSON
            result_dict = json.loads(tool_result)
            
            # Extract essential fields
            compressed = {
                "success": result_dict.get("success", True),
            }
            
            # Add error if present
            if "error" in result_dict:
                compressed["error"] = str(result_dict["error"])[:200]
            
            # Add key metrics (numbers, scores, paths)
            for key in ["score", "accuracy", "best_score", "n_rows", "n_cols", 
                       "output_path", "best_model", "result_summary"]:
                if key in result_dict:
                    compressed[key] = result_dict[key]
            
            # Add result if it's small
            if "result" in result_dict:
                result_str = str(result_dict["result"])
                if len(result_str) < 300:
                    compressed["result"] = result_str[:300]
            
            return json.dumps(compressed, indent=None)
            
        except json.JSONDecodeError:
            # Not JSON - truncate intelligently
            lines = tool_result.split('\n')
            
            # Keep first 5 and last 5 lines
            if len(lines) > 15:
                compressed_lines = lines[:5] + ["... (truncated) ..."] + lines[-5:]
                result = '\n'.join(compressed_lines)
            else:
                result = tool_result
            
            # Hard truncate if still too long
            token_count = self.count_tokens(result)
            if token_count > max_tokens:
                # Truncate to character limit (rough)
                char_limit = max_tokens * 4
                result = result[:char_limit] + "... (truncated)"
            
            return result
    
    def prioritize_messages(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """
        Assign priorities to messages based on type and importance.
        
        Priority levels:
        - 10: System prompt, recent user messages
        - 9: Recent tool results (last 3)
        - 8: Recent assistant responses (last 2)
        - 5: Normal messages
        - 3: Old tool results
        - 2: Old assistant responses
        - 1: Very old messages
        """
        # Find recent messages (last 5)
        recent_threshold = max(0, len(messages) - 5)
        
        for i, msg in enumerate(messages):
            if msg.message_type == "system":
                msg.priority = 10
            elif msg.role == "user":
                msg.priority = 10 if i >= recent_threshold else 7
            elif msg.message_type == "tool_result":
                msg.priority = 9 if i >= recent_threshold else 3
            elif msg.role == "assistant":
                msg.priority = 8 if i >= recent_threshold else 2
            else:
                msg.priority = 5 if i >= recent_threshold else 1
        
        return messages
    
    def apply_sliding_window(self, messages: List[ConversationMessage],
                            target_tokens: int) -> List[ConversationMessage]:
        """
        Apply sliding window to fit within token budget.
        
        Strategy:
        1. Always keep system prompt (first message)
        2. Keep recent messages (last N)
        3. Drop low-priority messages from middle
        4. Compress tool results if needed
        
        Args:
            messages: List of ConversationMessage objects
            target_tokens: Target token count
        
        Returns:
            Filtered message list within budget
        """
        if not messages:
            return []
        
        # Always keep system prompt
        system_msg = messages[0] if messages[0].message_type == "system" else None
        other_messages = messages[1:] if system_msg else messages
        
        # Prioritize messages
        other_messages = self.prioritize_messages(other_messages)
        
        # Sort by priority (high to low)
        sorted_messages = sorted(other_messages, key=lambda m: m.priority, reverse=True)
        
        # Calculate tokens for each message
        for msg in sorted_messages:
            if msg.tokens is None:
                msg.tokens = self.count_message_tokens(msg.to_dict())
        
        # Greedily add messages until budget exhausted
        kept_messages = []
        current_tokens = 0
        
        # Add system prompt first
        if system_msg:
            system_msg.tokens = self.count_message_tokens(system_msg.to_dict())
            kept_messages.append(system_msg)
            current_tokens += system_msg.tokens
        
        # Add other messages by priority
        for msg in sorted_messages:
            if current_tokens + msg.tokens <= target_tokens:
                kept_messages.append(msg)
                current_tokens += msg.tokens
            elif msg.message_type == "tool_result" and msg.priority >= 8:
                # Try compressing critical tool results
                compressed_content = self.compress_tool_result(msg.content, max_tokens=300)
                compressed_tokens = self.count_tokens(compressed_content)
                
                if current_tokens + compressed_tokens <= target_tokens:
                    msg.content = compressed_content
                    msg.tokens = compressed_tokens
                    kept_messages.append(msg)
                    current_tokens += compressed_tokens
        
        # Sort kept messages back to chronological order
        # System message stays first, rest in order they appeared
        if system_msg:
            non_system = [m for m in kept_messages if m != system_msg]
            # Sort by original index (approximate by content comparison)
            original_order = []
            for orig_msg in messages:
                for kept in non_system:
                    if kept.content == orig_msg.content:
                        original_order.append(kept)
                        break
            
            kept_messages = [system_msg] + original_order
        
        print(f"📊 Sliding window: {len(messages)} → {len(kept_messages)} messages ({current_tokens:,} tokens)")
        
        return kept_messages
    
    def emergency_truncate(self, messages: List[Dict[str, str]],
                          max_tokens: int) -> List[Dict[str, str]]:
        """
        Emergency truncation when context is about to overflow.
        
        Aggressive strategy:
        - Keep system prompt
        - Keep last user message
        - Keep last 2 messages
        - Truncate everything else
        
        Args:
            messages: Message list
            max_tokens: Hard token limit
        
        Returns:
            Truncated message list
        """
        if not messages:
            return []
        
        print("⚠️ EMERGENCY TRUNCATION: Context overflow imminent")
        
        # Always keep system, last user, and last 2 messages
        essential_messages = []
        
        # System prompt (first message)
        if messages:
            essential_messages.append(messages[0])
        
        # Last 2 messages
        if len(messages) > 2:
            essential_messages.extend(messages[-2:])
        else:
            essential_messages.extend(messages[1:])
        
        # Count tokens
        total_tokens = self.count_messages_tokens(essential_messages)
        
        if total_tokens <= max_tokens:
            return essential_messages
        
        # Still too large - truncate system prompt
        print("⚠️ Truncating system prompt to fit budget")
        system_msg = essential_messages[0]
        
        # Handle both dict and Pydantic object formats
        if isinstance(system_msg, dict):
            system_content = system_msg["content"]
        else:
            system_content = getattr(system_msg, "content", "")
        
        # Keep first 1000 chars of system prompt
        truncated_system = {
            "role": "system",
            "content": str(system_content)[:1000] + "\n\n... (truncated due to context limit) ..."
        }
        
        return [truncated_system] + essential_messages[1:]
    
    def enforce_budget(self, messages: List[Dict[str, str]],
                      system_prompt: Optional[str] = None) -> Tuple[List[Dict[str, str]], int]:
        """
        Main entry point: Enforce token budget on message list.
        
        Args:
            messages: List of messages
            system_prompt: Optional new system prompt to prepend
        
        Returns:
            (filtered_messages, total_tokens)
        """
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Count current tokens
        current_tokens = self.count_messages_tokens(messages)
        
        print(f"📊 Token Budget Check: {current_tokens:,} / {self.available_tokens:,} tokens")
        
        # If within budget, return as-is
        if current_tokens <= self.available_tokens:
            print("✅ Within budget")
            return messages, current_tokens
        
        print(f"⚠️ Over budget by {current_tokens - self.available_tokens:,} tokens")
        
        # Convert to ConversationMessage objects
        conv_messages = []
        for i, msg in enumerate(messages):
            # Handle both dict and Pydantic object formats
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "")
                content = getattr(msg, "content", "")
            
            msg_type = "system" if i == 0 and role == "system" else "normal"
            if "tool" in str(content).lower() or "function" in str(content).lower():
                msg_type = "tool_result"
            
            conv_msg = ConversationMessage(
                role=role,
                content=str(content),
                message_type=msg_type
            )
            conv_messages.append(conv_msg)
        
        # Apply sliding window
        filtered = self.apply_sliding_window(conv_messages, self.available_tokens)
        
        # Convert back to dict format
        result_messages = [msg.to_dict() for msg in filtered]
        final_tokens = self.count_messages_tokens(result_messages)
        
        # Emergency truncation if still over
        if final_tokens > self.available_tokens:
            result_messages = self.emergency_truncate(result_messages, self.available_tokens)
            final_tokens = self.count_messages_tokens(result_messages)
        
        print(f"✅ Budget enforced: {final_tokens:,} tokens ({len(result_messages)} messages)")
        
        return result_messages, final_tokens


# Global token budget manager instance
_token_manager = None

def get_token_manager(model: str = "gpt-4", max_tokens: int = 128000) -> TokenBudgetManager:
    """Get or create global token budget manager."""
    global _token_manager
    if _token_manager is None:
        _token_manager = TokenBudgetManager(model=model, max_tokens=max_tokens)
    return _token_manager
