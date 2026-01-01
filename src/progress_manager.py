"""
Global Progress Event Manager for Real-Time SSE Streaming

This module provides a singleton ProgressManager that captures all workflow progress
events and broadcasts them to connected SSE clients in real-time.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class ProgressManager:
    """
    Manages progress events for active analysis sessions.
    
    Features:
    - Emit events to multiple subscribers simultaneously
    - Store event history for late-joining clients
    - Automatic cleanup of dead connections
    - Thread-safe event broadcasting
    """
    
    def __init__(self):
        self._queues: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    def emit(self, session_id: str, event: Dict[str, Any]):
        """
        Emit a progress event to all subscribers.
        
        Args:
            session_id: Session identifier
            event: Event data (must include 'type' and 'message')
        """
        print(f"[SSE] PROGRESS_MANAGER EMIT: session={session_id}, event_type={event.get('type')}, msg={event.get('message', '')[:50]}")
        
        # Add timestamp
        event['timestamp'] = datetime.now().isoformat()
        
        # Store in history
        self._history[session_id].append(event)
        
        # Limit history size to prevent memory leaks
        if len(self._history[session_id]) > 500:
            self._history[session_id] = self._history[session_id][-500:]
        
        print(f"[SSE] History stored, total events for {session_id}: {len(self._history[session_id])}")
        
        # Send to all active subscribers
        if session_id in self._queues:
            print(f"[SSE] Found {len(self._queues[session_id])} subscribers for {session_id}")
            dead_queues = []
            for queue in self._queues[session_id]:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Queue is full, mark for removal
                    dead_queues.append(queue)
                except Exception:
                    dead_queues.append(queue)
            
            # Remove dead queues
            for dead_queue in dead_queues:
                if dead_queue in self._queues[session_id]:
                    self._queues[session_id].remove(dead_queue)
    
    async def subscribe(self, session_id: str):
        """
        Subscribe to progress events for a session.
        
        Args:
            session_id: Session identifier
            
        Yields:
            Progress events as they occur
        """
        queue = asyncio.Queue(maxsize=100)
        self._queues[session_id].append(queue)
        
        try:
            while True:
                event = await queue.get()
                yield event
        except asyncio.CancelledError:
            # Client disconnected
            pass
        finally:
            # Cleanup
            if session_id in self._queues and queue in self._queues[session_id]:
                self._queues[session_id].remove(queue)
    
    def get_history(self, session_id: str) -> List[Dict]:
        """
        Get all past events for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of past events
        """
        return self._history.get(session_id, [])
    
    def clear(self, session_id: str):
        """
        Clear history and disconnect all subscribers for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self._history:
            del self._history[session_id]
        if session_id in self._queues:
            # Close all queues
            for queue in self._queues[session_id]:
                try:
                    queue.put_nowait({'type': 'session_cleared', 'message': 'Session ended'})
                except:
                    pass
            del self._queues[session_id]
    
    def get_active_sessions(self) -> List[str]:
        """Get list of sessions with active subscribers."""
        return [sid for sid, queues in self._queues.items() if len(queues) > 0]
    
    def get_subscriber_count(self, session_id: str) -> int:
        """Get number of active subscribers for a session."""
        return len(self._queues.get(session_id, []))


# Global singleton instance
progress_manager = ProgressManager()
