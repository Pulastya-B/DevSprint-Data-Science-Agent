"""
FastAPI Application for Google Cloud Run
Thin HTTP wrapper around DataScienceCopilot - No logic changes, just API exposure.
"""

import os
import sys
import tempfile
import shutil
import time
import copy
import math
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import numpy as np

# Import from parent package
from src.orchestrator import DataScienceCopilot
from src.progress_manager import progress_manager
from src.session_memory import SessionMemory
from src.workflow_state import WorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JSON serializer that handles numpy types
def safe_json_dumps(obj):
    """Convert object to JSON string, handling numpy types, datetime, and all non-serializable objects."""
    from datetime import datetime, date, timedelta
    
    def convert(o):
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        elif isinstance(o, (np.floating, np.float64, np.float32)):
            val = float(o)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        elif isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
            return o
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, (datetime, date)):
            return o.isoformat()
        elif isinstance(o, timedelta):
            return str(o)
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            return [convert(item) for item in o]
        elif hasattr(o, '__dict__') and not isinstance(o, (str, int, float, bool, type(None))):
            # Non-serializable object (like DataScienceCopilot)
            return f"<{o.__class__.__name__} object>"
        elif hasattr(o, '__class__') and 'Figure' in o.__class__.__name__:
            return f"<{o.__class__.__name__} object>"
        return o
    
    return json.dumps(convert(obj))

# Initialize FastAPI
app = FastAPI(
    title="Data Science Agent API",
    description="Cloud Run wrapper for autonomous data science workflows",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SSE event queues for real-time streaming
class ProgressEventManager:
    """Manages SSE connections and progress events for real-time updates."""
    
    def __init__(self):
        self.active_streams: Dict[str, List[asyncio.Queue]] = {}
        self.session_status: Dict[str, Dict[str, Any]] = {}
    
    def create_stream(self, session_id: str) -> asyncio.Queue:
        """Create a new SSE stream for a session."""
        if session_id not in self.active_streams:
            self.active_streams[session_id] = []
        
        queue = asyncio.Queue()
        self.active_streams[session_id].append(queue)
        return queue
    
    def remove_stream(self, session_id: str, queue: asyncio.Queue):
        """Remove an SSE stream when client disconnects."""
        if session_id in self.active_streams:
            try:
                self.active_streams[session_id].remove(queue)
                if not self.active_streams[session_id]:
                    del self.active_streams[session_id]
            except (ValueError, KeyError):
                pass
    
    async def send_event(self, session_id: str, event_type: str, data: Dict[str, Any]):
        """Send an event to all connected clients for a session."""
        if session_id not in self.active_streams:
            return
        
        # Store current status
        self.session_status[session_id] = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        # Send to all connected streams
        dead_queues = []
        for queue in self.active_streams[session_id]:
            try:
                await asyncio.wait_for(queue.put((event_type, data)), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                dead_queues.append(queue)
        
        # Clean up dead queues
        for queue in dead_queues:
            self.remove_stream(session_id, queue)
    
    def get_current_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status for a session."""
        return self.session_status.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear all data for a session."""
        if session_id in self.active_streams:
            # Close all queues
            for queue in self.active_streams[session_id]:
                try:
                    queue.put_nowait(("complete", {}))
                except:
                    pass
            del self.active_streams[session_id]
        
        if session_id in self.session_status:
            del self.session_status[session_id]

# 👥 MULTI-USER SUPPORT: Session state isolation
# Heavy components (SBERT, tools, LLM client) are shared via global 'agent'
# Only session memory is isolated per user for fast initialization

from dataclasses import dataclass
from datetime import datetime, timedelta
import threading

@dataclass
class SessionState:
    """Wrapper for session with metadata for cleanup"""
    session: Any
    created_at: datetime
    last_accessed: datetime
    request_count: int = 0

session_states: Dict[str, SessionState] = {}  # session_id -> SessionState
agent_cache_lock = threading.Lock()  # threading.Lock for cross-event-loop safety
MAX_CACHED_SESSIONS = 50  # Increased limit for scale
SESSION_TTL_MINUTES = 60  # Sessions expire after 1 hour of inactivity
logger.info("👥 Multi-user session isolation initialized (fast mode)")

# Global agent - Heavy components loaded ONCE at startup
# SBERT model, tool functions, LLM client are shared across all users
# CRITICAL: We use threading.local() to ensure thread-safe session isolation
agent: Optional[DataScienceCopilot] = None
agent_thread_local = threading.local()  # Thread-local storage for session isolation
agent = None

# Session state isolation (lightweight - just session memory)
session_states: Dict[str, any] = {}  # session_id -> session memory only


async def get_agent_for_session(session_id: str) -> DataScienceCopilot:
    """
    Get agent with isolated session state.
    
    OPTIMIZATION: Heavy components (SBERT, tools, LLM client) are shared.
    Session state is isolated using thread-local storage to prevent race conditions.
    This reduces per-user initialization from 20s to <1s.
    
    THREAD SAFETY: Uses threading.Lock so this works from both the main event loop
    AND background thread-pool workers (avoiding asyncio event-loop binding issues).
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        DataScienceCopilot instance with isolated session for this user
    """
    global agent
    
    with agent_cache_lock:
        # Ensure base agent exists (heavy components loaded once at startup)
        if agent is None:
            logger.warning("Base agent not initialized - this shouldn't happen after startup")
            provider = os.getenv("LLM_PROVIDER", "mistral")
            agent = DataScienceCopilot(
                reasoning_effort="medium",
                provider=provider,
                use_compact_prompts=False
            )
        
        # Clean up expired sessions periodically (every 10th request)
        if len(session_states) > 0 and len(session_states) % 10 == 0:
            cleanup_expired_sessions()
        
        now = datetime.now()
        
        # Check if we have cached session memory for this session
        if session_id in session_states:
            state = session_states[session_id]
            state.last_accessed = now
            state.request_count += 1
            logger.info(f"[♻️] Reusing session {session_id[:8]}... (requests: {state.request_count})")
            
            # Create a lightweight copy so each request has its own session/state
            # Heavy components (SBERT, tool_functions, LLM client) are shared references
            request_agent = copy.copy(agent)
            request_agent.session = state.session
            request_agent.http_session_key = session_id
            request_agent.workflow_state = WorkflowState()
            return request_agent
        
        # 🚀 FAST PATH: Create new session memory only (no SBERT reload!)
        logger.info(f"[🆕] Creating lightweight session for {session_id[:8]}...")
        
        # Create isolated session memory for this user
        new_session = SessionMemory(session_id=session_id)
        
        # Cache management: Remove expired first, then LRU if still over limit
        if len(session_states) >= MAX_CACHED_SESSIONS:
            expired_count = cleanup_expired_sessions()
            
            # If still over limit after cleanup, remove least recently used
            if len(session_states) >= MAX_CACHED_SESSIONS:
                # Sort by last_accessed and remove oldest
                sorted_sessions = sorted(session_states.items(), key=lambda x: x[1].last_accessed)
                oldest_session_id = sorted_sessions[0][0]
                logger.info(f"[🗑️] Cache full, removing LRU session {oldest_session_id[:8]}...")
                del session_states[oldest_session_id]
        
        # Create session state wrapper with metadata
        session_state = SessionState(
            session=new_session,
            created_at=now,
            last_accessed=now,
            request_count=1
        )
        session_states[session_id] = session_state
        
        # Create a lightweight copy so each request has its own session/state
        request_agent = copy.copy(agent)
        request_agent.session = new_session
        request_agent.http_session_key = session_id
        request_agent.workflow_state = WorkflowState()
        
        logger.info(f"✅ Session created for {session_id[:8]} (cache: {len(session_states)}/{MAX_CACHED_SESSIONS}) - <1s init")
        
        return request_agent

def cleanup_expired_sessions():
    """Remove expired sessions based on TTL."""
    now = datetime.now()
    expired = []
    
    for session_id, state in session_states.items():
        # Check if session has been inactive for too long
        inactive_duration = now - state.last_accessed
        if inactive_duration > timedelta(minutes=SESSION_TTL_MINUTES):
            expired.append(session_id)
    
    for session_id in expired:
        logger.info(f"[🗑️] Removing expired session {session_id[:8]}... (inactive for {SESSION_TTL_MINUTES}min)")
        del session_states[session_id]
    
    return len(expired)

# 🔒 REQUEST QUEUING: Global lock to prevent concurrent workflows
# This ensures only one analysis runs at a time, preventing:
# - Race conditions on file writes
# - Memory exhaustion from parallel model training
# - Session state corruption
# NOTE: Uses threading.Lock (not asyncio.Lock) because run_analysis_background
# is executed in a Starlette thread pool worker, not the main event loop.
import threading
workflow_lock = threading.Lock()
logger.info("🔒 Workflow lock initialized for request queuing")

# Mount static files for React frontend
frontend_path = Path(__file__).parent.parent.parent / "FRRONTEEEND" / "dist"
if frontend_path.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_path / "assets")), name="assets")
    logger.info(f"✅ Frontend assets mounted from {frontend_path}")


@app.on_event("startup")
async def startup_event():
    """Initialize DataScienceCopilot on service startup."""
    global agent
    try:
        logger.info("Initializing legacy global agent for health checks...")
        provider = os.getenv("LLM_PROVIDER", "mistral")
        use_compact = False  # Always use multi-agent routing
        
        # Create one agent for health checks only
        # Real requests will use get_agent_for_session() for isolation
        agent = DataScienceCopilot(
            reasoning_effort="medium",
            provider=provider,
            use_compact_prompts=use_compact
        )
        logger.info(f"✅ Health check agent initialized with provider: {agent.provider}")
        logger.info("👥 Per-session agents enabled - each user gets isolated instance")
        logger.info("🤖 Multi-agent architecture enabled with 5 specialists")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {e}")
        raise


@app.get("/api/health")
async def root():
    """Health check endpoint."""
    return {
        "service": "Data Science Agent API",
        "status": "healthy",
        "provider": agent.provider if agent else "not initialized",
        "tools_available": len(agent.tool_functions) if agent else 0
    }


@app.get("/api/progress/{session_id}")
async def get_progress(session_id: str):
    """Get progress updates for a specific session (legacy polling endpoint)."""
    return {
        "session_id": session_id,
        "steps": progress_manager.get_history(session_id),
        "current": {"status": "active" if progress_manager.get_subscriber_count(session_id) > 0 else "idle"}
    }


@app.get("/api/progress/stream/{session_id}")
async def stream_progress(session_id: str):
    """Stream real-time progress updates using Server-Sent Events (SSE)."""
    print(f"[SSE] ENDPOINT: Client connected for session_id={session_id}")
    
    # CRITICAL: Create queue and register subscriber IMMEDIATELY
    queue = asyncio.Queue(maxsize=100)
    if session_id not in progress_manager._queues:
        progress_manager._queues[session_id] = []
    progress_manager._queues[session_id].append(queue)
    print(f"[SSE] Queue registered, total subscribers: {len(progress_manager._queues[session_id])}")
    
    async def event_generator():
        try:
            # 1. Send initial connection confirmation immediately
            try:
                connection_event = {
                    'type': 'connected',
                    'message': '🔗 Connected to progress stream',
                    'session_id': session_id
                }
                data_str = safe_json_dumps(connection_event)
                print(f"[SSE] SENDING connection event to client: {data_str[:100]}")
                yield f"data: {data_str}\n\n"
            except Exception as e:
                print(f"[SSE] ERROR sending connection event: {e}")
                logger.error(f"SSE connection event error: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': 'Connection failed'})}\n\n"
                return
            
            # 2. Replay history
            try:
                history = progress_manager.get_history(session_id)
                if history:
                    print(f"[SSE] Found {len(history)} events in history")
                    for past_event in history:
                        if past_event.get('type') != 'analysis_complete':
                            data_str = safe_json_dumps(past_event)
                            yield f"data: {data_str}\n\n"
                        else:
                            # Terminal event - send it with long retry, then close
                            data_str = safe_json_dumps(past_event)
                            yield f"retry: 86400000\ndata: {data_str}\n\n"
                            print(f"[SSE] Analysis already complete, closing stream")
                            await asyncio.sleep(1)
                            return
                else:
                    print(f"[SSE] No history to replay (fresh session)")
            except Exception as e:
                print(f"[SSE] ERROR replaying history: {e}")
                logger.error(f"SSE history replay error: {e}", exc_info=True)
            
            # 3. Stream new events
            print(f"[SSE] Starting event stream loop for session {session_id}")
            consecutive_empty_cycles = 0
            
            while True:
                try:
                    if not queue.empty():
                        consecutive_empty_cycles = 0
                        event = queue.get_nowait()
                        try:
                            data_str = safe_json_dumps(event)
                            print(f"[SSE] Sending {event.get('type')}: {data_str[:100]}")
                            yield f"data: {data_str}\n\n"
                            
                            if event.get('type') == 'analysis_complete':
                                print(f"[SSE] Analysis complete, closing stream")
                                await asyncio.sleep(1)
                                return
                        except Exception as e:
                            print(f"[SSE] ERROR serializing event: {e}")
                            logger.error(f"SSE event serialization error: {e}", exc_info=True)
                    else:
                        # Keep-alive ping every 500ms
                        consecutive_empty_cycles += 1
                        if consecutive_empty_cycles % 2 == 0:  # Every 1 second
                            yield ": keep-alive\n\n"
                        await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    print(f"[SSE] Stream cancelled for session {session_id}")
                    break
                except Exception as e:
                    print(f"[SSE] ERROR in event loop: {e}")
                    logger.error(f"SSE event loop error: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Stream error'})}\n\n"
                    break
                    
        except Exception as e:
            print(f"[SSE] CRITICAL ERROR in event_generator: {e}")
            logger.error(f"SSE generator error: {e}", exc_info=True)
        finally:
            # Cleanup
            try:
                if session_id in progress_manager._queues and queue in progress_manager._queues[session_id]:
                    progress_manager._queues[session_id].remove(queue)
                print(f"[SSE] Stream closed for session {session_id}, remaining subscribers: {len(progress_manager._queues.get(session_id, []))}")
            except Exception as e:
                print(f"[SSE] ERROR in cleanup: {e}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Max-Age": "3600"
        }
    )


@app.get("/health")
async def health_check():
    """
    Health check for Cloud Run.
    Returns 200 if service is ready to accept requests.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "status": "healthy",
        "agent_ready": True,
        "provider": agent.provider,
        "tools_count": len(agent.tool_functions)
    }


class AnalysisRequest(BaseModel):
    """Request model for analysis endpoint (JSON body)."""
    task_description: str
    target_col: Optional[str] = None
    use_cache: bool = True
    max_iterations: int = 20


def run_analysis_background(file_path: str, task_description: str, target_col: Optional[str], 
                            use_cache: bool, max_iterations: int, session_id: str):
    """Background task to run analysis and emit events.
    
    Runs in a Starlette thread-pool worker. Uses threading.Lock (not asyncio)
    to serialize concurrent analysis requests.
    """
    with workflow_lock:
        try:
            logger.info(f"[BACKGROUND] Starting analysis for session {session_id[:8]}...")
            
            # 🧹 Clear SSE history for fresh event stream (prevents duplicate results)
            print(f"[🧹] Clearing SSE history for {session_id[:8]}...")
            if session_id in progress_manager._history:
                progress_manager._history[session_id] = []
            
            # 👥 Get isolated agent for this session
            # get_agent_for_session is async but now uses threading.Lock internally,
            # so we need a small event loop just for the await
            import asyncio
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                session_agent = loop.run_until_complete(get_agent_for_session(session_id))
            finally:
                loop.close()
            
            result = session_agent.analyze(
                file_path=file_path,
                task_description=task_description,
                target_col=target_col,
                use_cache=use_cache,
                max_iterations=max_iterations
            )
            
            logger.info(f"[BACKGROUND] Analysis completed for session {session_id[:8]}...")
            
            # Send appropriate completion event based on status
            if result.get("status") == "error":
                progress_manager.emit(session_id, {
                    "type": "analysis_failed",
                    "status": "error",
                    "message": result.get("summary", "❌ Analysis failed"),
                    "error": result.get("error", "Analysis error"),
                    "result": result
                })
            else:
                progress_manager.emit(session_id, {
                    "type": "analysis_complete",
                    "status": result.get("status"),
                    "message": "✅ Analysis completed successfully!",
                    "result": result
                })
            
        except Exception as e:
            logger.error(f"[BACKGROUND] Analysis failed for session {session_id[:8]}...: {e}")
            import traceback
            traceback.print_exc()
            progress_manager.emit(session_id, {
                "type": "analysis_failed",
                "error": "Analysis failed. Please try again.",
                "message": "❌ Analysis failed. Please try again or upload a different file."
            })


@app.post("/run-async")
async def run_analysis_async(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    task_description: str = Form(...),
    target_col: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),  # Accept session_id from frontend for follow-ups
    use_cache: bool = Form(False),  # Disabled to show multi-agent in action
    max_iterations: int = Form(20)
) -> JSONResponse:
    """
    Start analysis in background and return session UUID immediately.
    Frontend can connect SSE with this UUID to receive real-time updates.
    
    For follow-up queries, frontend should send the same session_id to maintain context.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # 🆔 Session ID handling:
    # - If frontend sends a valid UUID, REUSE it (follow-up query)
    # - Otherwise generate a new one (first query)
    import uuid
    if session_id and '-' in session_id and len(session_id) > 20:
        # Valid UUID from frontend - this is a follow-up query
        logger.info(f"[ASYNC] Reusing session: {session_id[:8]}... (follow-up)")
    else:
        # Generate new session for first query
        session_id = str(uuid.uuid4())
        logger.info(f"[ASYNC] Created new session: {session_id[:8]}...")
    
    # Handle file upload
    temp_file_path = None
    if file:
        # File size guard: reject uploads > 500MB to prevent OOM
        MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset
        if file_size > MAX_UPLOAD_BYTES:
            return JSONResponse(
                content={"success": False, "error": f"File too large ({file_size / 1024 / 1024:.0f}MB). Maximum is 500MB."},
                status_code=413
            )
        
        temp_dir = Path("/tmp") / "data_science_agent"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize filename to prevent path traversal
        import secrets
        safe_name = secrets.token_hex(8) + Path(file.filename).suffix
        temp_file_path = temp_dir / safe_name
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[ASYNC] File saved: {file.filename}")
    else:
        # 🛡️ VALIDATION: Check if this session has dataset cached
        has_dataset = False
        with agent_cache_lock:
            # Check session_states cache for this specific session_id
            if session_id in session_states:
                state = session_states[session_id]
                cached_session = state.session  # Extract SessionMemory from wrapper
                if hasattr(cached_session, 'last_dataset') and cached_session.last_dataset:
                    has_dataset = True
                    logger.info(f"[ASYNC] Follow-up query for session {session_id[:8]}... - using cached dataset")
        
        if not has_dataset:
            logger.warning(f"[ASYNC] No file uploaded and no dataset for session {session_id[:8]}...")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No dataset available",
                    "message": "Please upload a CSV, Excel, or Parquet file first.",
                    "session_id": session_id
                },
                status_code=400
            )
    
    # Start background analysis
    background_tasks.add_task(
        run_analysis_background,
        file_path=str(temp_file_path) if temp_file_path else "",
        task_description=task_description,
        target_col=target_col,
        use_cache=use_cache,
        max_iterations=max_iterations,
        session_id=session_id
    )
    
    # Return UUID immediately so frontend can connect SSE
    return JSONResponse(content={
        "session_id": session_id,
        "status": "started",
        "message": "Analysis started in background"
    })


@app.post("/run")
async def run_analysis(
    file: Optional[UploadFile] = File(None, description="Dataset file (CSV or Parquet) - optional for follow-up requests"),
    task_description: str = Form(..., description="Natural language task description"),
    target_col: Optional[str] = Form(None, description="Target column name for prediction"),
    use_cache: bool = Form(False, description="Enable caching for expensive operations"),  # Disabled to show multi-agent
    max_iterations: int = Form(20, description="Maximum workflow iterations"),
    session_id: Optional[str] = Form(None, description="Session ID for follow-up requests")
) -> JSONResponse:
    """
    Run complete data science workflow on uploaded dataset.
    
    This is a thin wrapper - all logic lives in DataScienceCopilot.analyze().
    
    Args:
        file: CSV or Parquet file upload
        task_description: Natural language description of the task
        target_col: Optional target column for ML tasks
        use_cache: Whether to use cached results
        max_iterations: Maximum number of workflow steps
        
    Returns:
        JSON response with analysis results, workflow history, and execution stats
        
    Example:
        ```bash
        curl -X POST http://localhost:8080/run \
          -F "file=@data.csv" \
          -F "task_description=Analyze this dataset and predict house prices" \
          -F "target_col=price"
        ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # 🆔 Generate or use provided session ID
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        logger.info(f"[SYNC] Created new session: {session_id[:8]}...")
    else:
        logger.info(f"[SYNC] Using provided session: {session_id[:8]}...")
    
    # 👥 Get isolated agent for this session
    session_agent = await get_agent_for_session(session_id)
    
    # Handle follow-up requests (no file, using session memory)
    if file is None:
        logger.info(f"Follow-up request without file, using session memory")
        logger.info(f"Task: {task_description}")
        
        # 🛡️ VALIDATION: Check if session has a dataset
        if not (hasattr(session_agent, 'session') and session_agent.session and session_agent.session.last_dataset):
            logger.warning("No file uploaded and no session dataset available")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "No dataset available",
                    "message": "Please upload a CSV, Excel, or Parquet file first before asking questions."
                },
                status_code=400
            )
        
        # Get the agent's actual session UUID for SSE routing
        actual_session_id = session_agent.session.session_id if hasattr(session_agent, 'session') and session_agent.session else session_id
        print(f"[SSE] Follow-up using agent session UUID: {actual_session_id}")
        
        # NO progress_callback - orchestrator emits directly to UUID
        
        try:
            # Agent's session memory should resolve file_path from context
            result = session_agent.analyze(
                file_path="",  # Empty - will be resolved by session memory
                task_description=task_description,
                target_col=target_col,
                use_cache=use_cache,
                max_iterations=max_iterations
            )
            
            logger.info(f"Follow-up analysis completed: {result.get('status')}")
            
            # Send appropriate completion event based on status
            if result.get("status") == "error":
                progress_manager.emit(actual_session_id, {
                    "type": "analysis_failed",
                    "status": "error",
                    "message": result.get("summary", "❌ Analysis failed"),
                    "error": result.get("error", "No dataset available")
                })
            else:
                progress_manager.emit(actual_session_id, {
                    "type": "analysis_complete",
                    "status": result.get("status"),
                    "message": "✅ Analysis completed successfully!"
                })
            
            # Make result JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif hasattr(obj, '__class__') and obj.__class__.__name__ in ['Figure', 'Axes', 'Artist']:
                    return f"<{obj.__class__.__name__} object - see artifacts>"
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    try:
                        return str(obj)
                    except:
                        return f"<{type(obj).__name__}>"
            
            serializable_result = make_json_serializable(result)
            
            return JSONResponse(
                content={
                    "success": result.get("status") == "success",
                    "result": serializable_result,
                    "metadata": {
                        "filename": "session_context",
                        "task": task_description,
                        "target": target_col,
                        "provider": agent.provider,
                        "follow_up": True
                    }
                },
                status_code=200
            )
        
        except Exception as e:
            logger.error(f"Follow-up analysis failed: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Follow-up request failed. Make sure you've uploaded a file first.",
                    "error_type": "InternalError",
                    "message": "Follow-up request failed. Make sure you've uploaded a file first."
                }
            )
    
    # Validate file format for new uploads
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV and Parquet files are supported."
        )
    
    # File size guard: reject uploads > 500MB to prevent OOM
    MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large ({file_size / 1024 / 1024:.0f}MB). Maximum is 500MB.")
    
    # Use /tmp for Cloud Run (ephemeral storage)
    temp_dir = Path("/tmp") / "data_science_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = None
    
    try:
        # Sanitize filename to prevent path traversal
        import secrets
        safe_name = secrets.token_hex(8) + Path(file.filename).suffix
        temp_file_path = temp_dir / safe_name
        logger.info(f"Saving uploaded file to: {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File saved successfully: {file.filename} ({os.path.getsize(temp_file_path)} bytes)")
        
        # Get the agent's actual session UUID for SSE routing (BEFORE analyze())
        actual_session_id = session_agent.session.session_id if hasattr(session_agent, 'session') and session_agent.session else session_id
        print(f"[SSE] File upload using agent session UUID: {actual_session_id}")
        
        # NO progress_callback - orchestrator emits directly to UUID
        
        # Call existing agent logic
        logger.info(f"Starting analysis with task: {task_description}")
        result = session_agent.analyze(
            file_path=str(temp_file_path),
            task_description=task_description,
            target_col=target_col,
            use_cache=use_cache,
            max_iterations=max_iterations
        )
        
        logger.info(f"Analysis completed: {result.get('status')}")
        
        # Send appropriate completion event based on status
        if result.get("status") == "error":
            progress_manager.emit(actual_session_id, {
                "type": "analysis_failed",
                "status": "error",
                "message": result.get("summary", "❌ Analysis failed"),
                "error": result.get("error", "Analysis error")
            })
        else:
            progress_manager.emit(actual_session_id, {
                "type": "analysis_complete",
                "status": result.get("status"),
                "message": "✅ Analysis completed successfully!"
            })
        
        # Filter out non-JSON-serializable objects (like matplotlib/plotly Figures)
        def make_json_serializable(obj):
            """Recursively convert objects to JSON-serializable format."""
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__class__') and obj.__class__.__name__ in ['Figure', 'Axes', 'Artist']:
                # Skip matplotlib/plotly Figure objects
                return f"<{obj.__class__.__name__} object - see artifacts>"
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Try to convert to string for other types
                try:
                    return str(obj)
                except:
                    return f"<{type(obj).__name__}>"
        
        serializable_result = make_json_serializable(result)
        
        # Return result with ACTUAL session UUID for SSE
        return JSONResponse(
            content={
                "success": result.get("status") == "success",
                "result": serializable_result,
                "session_id": actual_session_id,  # Return UUID for SSE connection
                "metadata": {
                    "filename": file.filename,
                    "task": task_description,
                    "target": target_col,
                    "provider": agent.provider
                }
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Analysis workflow failed. Please try again.",
                "error_type": "InternalError",
                "message": "Analysis workflow failed. Check logs for details."
            }
        )
    
    finally:
        # Keep temporary file for session continuity (follow-up requests)
        # Files in /tmp are automatically cleaned up by the OS
        # For HuggingFace Spaces: space restart clears /tmp
        # For production: implement session-based cleanup after timeout
        pass


@app.post("/profile")
async def profile_dataset(
    file: UploadFile = File(..., description="Dataset file (CSV or Parquet)")
) -> JSONResponse:
    """
    Quick dataset profiling without full workflow.
    
    Returns basic statistics, data types, and quality issues.
    Useful for initial data exploration without running full analysis.
    
    Example:
        ```bash
        curl -X POST http://localhost:8080/profile \
          -F "file=@data.csv"
        ```
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    filename = file.filename.lower()
    if not (filename.endswith('.csv') or filename.endswith('.parquet')):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only CSV and Parquet files are supported."
        )
    
    temp_dir = Path("/tmp") / "data_science_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = None
    
    try:
        # Sanitize filename to prevent path traversal
        import secrets
        safe_name = secrets.token_hex(8) + Path(file.filename).suffix
        temp_file_path = temp_dir / safe_name
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Import profiling tool directly
        from tools.data_profiling import profile_dataset as profile_tool
        from tools.data_profiling import detect_data_quality_issues
        
        # Run profiling tools
        logger.info(f"Profiling dataset: {file.filename}")
        profile_result = profile_tool(str(temp_file_path))
        quality_result = detect_data_quality_issues(str(temp_file_path))
        
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "profile": profile_result,
                "quality_issues": quality_result
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Profiling failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Profiling failed. Please try again.",
                "error_type": "InternalError"
            }
        )
    
    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.get("/tools")
async def list_tools():
    """
    List all available tools in the agent.
    
    Returns tool names organized by category.
    Useful for understanding agent capabilities.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    from tools.tools_registry import get_tools_by_category
    
    return {
        "total_tools": len(agent.tool_functions),
        "tools_by_category": get_tools_by_category(),
        "all_tools": list(agent.tool_functions.keys())
    }


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""
    messages: List[ChatMessage]
    stream: bool = False


@app.post("/chat")
async def chat(request: ChatRequest) -> JSONResponse:
    """
    Chat endpoint for conversational interface.
    
    Processes chat messages and returns agent responses.
    Uses the same underlying agent as /run but in chat format.
    
    Args:
        request: Chat request with message history
        
    Returns:
        JSON response with agent's reply
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Extract the latest user message
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        latest_message = user_messages[-1].content
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GOOGLE_API_KEY or GEMINI_API_KEY not configured. Please set the environment variable."
            )
        
        # Use Google Gemini API
        import google.generativeai as genai
        
        logger.info(f"Configuring Gemini with API key (length: {len(api_key)})")
        genai.configure(api_key=api_key)
        
        # Safety settings for data science content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Initialize Gemini model (system_instruction not supported in this SDK version)
        model = genai.GenerativeModel(
            model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            generation_config={"temperature": 0.7},
            safety_settings=safety_settings
        )
        
        # System message will be prepended to first user message
        system_msg = "You are a Senior Data Science Autonomous Agent. You help users with end-to-end machine learning, data profiling, visualization, and strategic insights. Use a professional, technical yet accessible tone. Provide code snippets in Python if requested. You have access to tools for data analysis, ML training, visualization, and more.\\n\\n"
        
        # Convert messages to Gemini format (exclude system message, just conversation)
        chat_history = []
        first_user_msg = True
        for msg in request.messages[:-1]:  # Exclude the latest message
            content = msg.content
            # Prepend system instruction to first user message
            if first_user_msg and msg.role == "user":
                content = system_msg + content
                first_user_msg = False
            chat_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [content]
            })
        
        # Start chat with history
        chat = model.start_chat(history=chat_history)
        
        # Send the latest message
        response = chat.send_message(latest_message)
        
        assistant_message = response.text
        
        return JSONResponse(
            content={
                "success": True,
                "message": assistant_message,
                "model": "gemini-2.0-flash-exp",
                "provider": "gemini"
            },
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Chat request failed. Please try again.",
                "error_type": "InternalError"
            }
        )


# ==================== FILE STORAGE API ====================
# These endpoints handle persistent file storage with R2 + Supabase

class FileMetadataResponse(BaseModel):
    """Response model for file metadata."""
    id: str
    file_type: str
    file_name: str
    size_bytes: int
    created_at: str
    expires_at: str
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = {}

class UserFilesResponse(BaseModel):
    """Response model for user files list."""
    success: bool
    files: List[FileMetadataResponse]
    total_count: int
    total_size_mb: float

@app.get("/api/files")
async def get_user_files(
    user_id: str,
    file_type: Optional[str] = None,
    session_id: Optional[str] = None
):
    """
    Get all files for a user.
    
    Query params:
    - user_id: User ID (required)
    - file_type: Filter by type (plot, csv, report, model)
    - session_id: Filter by chat session
    """
    try:
        from src.storage.user_files_service import get_files_service, FileType
        from src.storage.r2_storage import get_r2_service
        
        files_service = get_files_service()
        r2_service = get_r2_service()
        
        # Convert file_type string to enum if provided
        file_type_enum = None
        if file_type:
            file_type_enum = FileType(file_type)
        
        files = files_service.get_user_files(
            user_id=user_id,
            file_type=file_type_enum,
            session_id=session_id
        )
        
        # Generate download URLs
        file_responses = []
        total_size = 0
        for f in files:
            download_url = None
            if f.file_type == FileType.CSV:
                download_url = r2_service.get_csv_download_url(f.r2_key)
            elif f.file_type in [FileType.REPORT, FileType.PLOT]:
                download_url = r2_service.get_report_url(f.r2_key)
            
            file_responses.append(FileMetadataResponse(
                id=f.id,
                file_type=f.file_type.value,
                file_name=f.file_name,
                size_bytes=f.size_bytes,
                created_at=f.created_at.isoformat(),
                expires_at=f.expires_at.isoformat(),
                download_url=download_url,
                metadata=f.metadata
            ))
            total_size += f.size_bytes
        
        return UserFilesResponse(
            success=True,
            files=file_responses,
            total_count=len(files),
            total_size_mb=round(total_size / (1024 * 1024), 2)
        )
        
    except ImportError:
        # Storage services not configured
        return UserFilesResponse(
            success=True,
            files=[],
            total_count=0,
            total_size_mb=0
        )
    except Exception as e:
        logger.error(f"Error fetching user files: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/api/files/{file_id}")
async def get_file(file_id: str):
    """Get a specific file by ID with download URL."""
    try:
        from src.storage.user_files_service import get_files_service, FileType
        from src.storage.r2_storage import get_r2_service
        
        files_service = get_files_service()
        r2_service = get_r2_service()
        
        file = files_service.get_file_by_id(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Generate appropriate URL
        download_url = None
        if file.file_type == FileType.CSV:
            download_url = r2_service.get_csv_download_url(file.r2_key)
        elif file.file_type == FileType.PLOT:
            # For plots, return the plot data directly
            plot_data = r2_service.get_plot_data(file.r2_key)
            return {
                "success": True,
                "file": {
                    "id": file.id,
                    "file_type": file.file_type.value,
                    "file_name": file.file_name,
                    "metadata": file.metadata
                },
                "plot_data": plot_data
            }
        else:
            download_url = r2_service.get_report_url(file.r2_key)
        
        return {
            "success": True,
            "file": FileMetadataResponse(
                id=file.id,
                file_type=file.file_type.value,
                file_name=file.file_name,
                size_bytes=file.size_bytes,
                created_at=file.created_at.isoformat(),
                expires_at=file.expires_at.isoformat(),
                download_url=download_url,
                metadata=file.metadata
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching file: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.delete("/api/files/{file_id}")
async def delete_file(file_id: str, user_id: str):
    """Delete a file (both from R2 and Supabase)."""
    try:
        from src.storage.user_files_service import get_files_service
        from src.storage.r2_storage import get_r2_service
        
        files_service = get_files_service()
        r2_service = get_r2_service()
        
        file = files_service.get_file_by_id(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Verify ownership
        if file.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        # Delete from R2
        r2_service.delete_file(file.r2_key)
        
        # Delete from Supabase
        files_service.hard_delete_file(file_id)
        
        return {"success": True, "message": "File deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

@app.get("/api/files/stats/{user_id}")
async def get_storage_stats(user_id: str):
    """Get storage statistics for a user."""
    try:
        from src.storage.user_files_service import get_files_service
        
        files_service = get_files_service()
        stats = files_service.get_user_storage_stats(user_id)
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "success": True,
            "stats": {
                "total_files": 0,
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "by_type": {}
            }
        }

@app.post("/api/files/extend/{file_id}")
async def extend_file_expiration(file_id: str, user_id: str, days: int = 7):
    """Extend a file's expiration date."""
    try:
        from src.storage.user_files_service import get_files_service
        
        files_service = get_files_service()
        
        file = files_service.get_file_by_id(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        if file.user_id != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        
        success = files_service.extend_expiration(file_id, days)
        
        return {"success": success}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extending expiration: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all error handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
            "error_type": "InternalError"
        }
    )


@app.get("/outputs/{file_path:path}")
async def serve_output_files(file_path: str):
    """
    Serve generated output files (reports, plots, models, etc.).
    Checks multiple locations: ./outputs, /tmp/data_science_agent/outputs, and /tmp/data_science_agent.
    """
    # Locations to check (in order of priority)
    search_paths = [
        Path("./outputs") / file_path,  # Local development
        Path("/tmp/data_science_agent/outputs") / file_path,  # Production with subdirs
        Path("/tmp/data_science_agent") / file_path,  # Production flat OR relative paths like plots/xxx.html
        Path("/tmp/data_science_agent/outputs") / Path(file_path).name,  # Production filename only
        Path("/tmp/data_science_agent") / Path(file_path).name,  # Production root filename only
        Path("./outputs") / Path(file_path).name,  # Local development filename only
    ]
    
    output_path = None
    for path in search_paths:
        logger.debug(f"Checking path: {path}")
        if path.exists() and path.is_file():
            output_path = path
            logger.info(f"Found file at: {path}")
            break
    
    if output_path is None:
        logger.error(f"File not found in any location: {file_path}")
        logger.error(f"Searched paths: {[str(p) for p in search_paths]}")
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    # Security: prevent directory traversal
    resolved_path = output_path.resolve()
    allowed_bases = [
        Path("./outputs").resolve(),
        Path("/tmp/data_science_agent").resolve()
    ]
    
    # Check if path is within allowed directories
    is_allowed = False
    for base in allowed_bases:
        try:
            resolved_path.relative_to(base)
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Determine media type based on file extension
    media_type = None
    if file_path.endswith('.html'):
        media_type = "text/html"
    elif file_path.endswith('.csv'):
        media_type = "text/csv"
    elif file_path.endswith('.json'):
        media_type = "application/json"
    elif file_path.endswith('.png'):
        media_type = "image/png"
    elif file_path.endswith('.jpg') or file_path.endswith('.jpeg'):
        media_type = "image/jpeg"
    
    return FileResponse(output_path, media_type=media_type)


# ============== HUGGINGFACE EXPORT ENDPOINT ==============

class HuggingFaceExportRequest(BaseModel):
    """Request model for HuggingFace export."""
    user_id: str
    session_id: str

@app.post("/api/export/huggingface")
async def export_to_huggingface(request: HuggingFaceExportRequest):
    """
    Export session assets (datasets, models, plots) to user's HuggingFace account.
    
    Requires user to have connected their HuggingFace token in settings.
    """
    import glob
    
    logger.info(f"[HF Export] Starting export for user {request.user_id[:8]}... session {request.session_id[:8]}...")
    
    try:
        # Try to import supabase - may not be installed
        try:
            from supabase import create_client, Client
        except ImportError as e:
            logger.error(f"[HF Export] Supabase package not installed: {e}")
            raise HTTPException(status_code=500, detail="Server error: supabase package not installed")
        
        # Get user's HuggingFace credentials from Supabase
        supabase_url = os.getenv("VITE_SUPABASE_URL") or os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
        
        logger.info(f"[HF Export] Supabase URL configured: {bool(supabase_url)}, Key configured: {bool(supabase_key)}")
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Supabase configuration missing")
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Fetch user's HuggingFace token from hf_tokens table (not user_profiles)
        logger.info(f"[HF Export] Fetching HF token from hf_tokens table...")
        try:
            result = supabase.table("hf_tokens").select(
                "huggingface_token, huggingface_username"
            ).eq("user_id", request.user_id).execute()
            
            logger.info(f"[HF Export] Query result: {result.data}")
            
            if not result.data or len(result.data) == 0:
                raise HTTPException(status_code=404, detail="HuggingFace not connected. Please connect in Settings first.")
            
            row = result.data[0]
            hf_token = row.get("huggingface_token")
            hf_username = row.get("huggingface_username")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[HF Export] Supabase query error: {e}")
            raise HTTPException(status_code=500, detail="Database error. Please try again.")
        
        if not hf_token:
            raise HTTPException(
                status_code=400, 
                detail="HuggingFace token not found. Please connect in Settings."
            )
        
        # Import HuggingFace storage service
        try:
            from src.storage.huggingface_storage import HuggingFaceStorage
            logger.info(f"[HF Export] HuggingFaceStorage imported successfully")
        except ImportError as e:
            logger.error(f"[HF Export] Failed to import HuggingFaceStorage: {e}")
            raise HTTPException(status_code=500, detail="Server error: required component not available")
        
        try:
            hf_service = HuggingFaceStorage(hf_token=hf_token)
            logger.info(f"[HF Export] HuggingFaceStorage initialized for user: {hf_username}")
        except Exception as e:
            logger.error(f"[HF Export] Failed to initialize HuggingFaceStorage: {e}")
            raise HTTPException(status_code=500, detail="HuggingFace connection error. Please check your token.")
        
        # Collect all session assets
        uploaded_files = []
        errors = []
        
        # Session-specific output directory - check /tmp/data_science_agent for HF Spaces
        session_outputs_dir = Path(f"./outputs/{request.session_id}")
        global_outputs_dir = Path("./outputs")
        tmp_outputs_dir = Path("/tmp/data_science_agent")
        
        logger.info(f"[HF Export] Looking for files in: {session_outputs_dir}, {global_outputs_dir}, {tmp_outputs_dir}")
        
        # Upload datasets (CSVs)
        csv_patterns = [
            session_outputs_dir / "*.csv",
            global_outputs_dir / "*.csv",
            tmp_outputs_dir / "*.csv"
        ]
        for pattern in csv_patterns:
            for csv_file in glob.glob(str(pattern)):
                try:
                    logger.info(f"[HF Export] Uploading dataset: {csv_file}")
                    result = hf_service.upload_dataset(
                        file_path=csv_file,
                        session_id=request.session_id,
                        file_name=Path(csv_file).name,
                        compress=True
                    )
                    if result.get("success"):
                        uploaded_files.append({"type": "dataset", "name": Path(csv_file).name, "url": result.get("url")})
                    else:
                        errors.append(f"Dataset {Path(csv_file).name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[HF Export] Dataset upload error: {e}")
                    errors.append(f"Dataset {Path(csv_file).name}: {str(e)}")
        
        # Upload models (PKL files)
        model_patterns = [
            session_outputs_dir / "models" / "*.pkl",
            global_outputs_dir / "models" / "*.pkl",
            tmp_outputs_dir / "models" / "*.pkl"
        ]
        for pattern in model_patterns:
            for model_file in glob.glob(str(pattern)):
                try:
                    logger.info(f"[HF Export] Uploading model: {model_file}")
                    result = hf_service.upload_model(
                        model_path=model_file,
                        session_id=request.session_id,
                        model_name=Path(model_file).stem,
                        model_type="sklearn"
                    )
                    if result.get("success"):
                        uploaded_files.append({"type": "model", "name": Path(model_file).name, "url": result.get("url")})
                    else:
                        errors.append(f"Model {Path(model_file).name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[HF Export] Model upload error: {e}")
                    errors.append(f"Model {Path(model_file).name}: {str(e)}")
        
        # Upload visualizations (HTML plots) - use generic file upload
        plot_patterns = [
            session_outputs_dir / "*.html",
            global_outputs_dir / "*.html",
            session_outputs_dir / "plots" / "*.html",
            global_outputs_dir / "plots" / "*.html",
            tmp_outputs_dir / "*.html",
            tmp_outputs_dir / "plots" / "*.html"
        ]
        for pattern in plot_patterns:
            for plot_file in glob.glob(str(pattern)):
                # Skip index.html or other non-plot files
                if "index" in Path(plot_file).name.lower():
                    continue
                try:
                    logger.info(f"[HF Export] Uploading HTML plot: {plot_file}")
                    result = hf_service.upload_generic_file(
                        file_path=plot_file,
                        session_id=request.session_id,
                        subfolder="plots"
                    )
                    if result.get("success"):
                        uploaded_files.append({"type": "plot", "name": Path(plot_file).name, "url": result.get("url")})
                    else:
                        errors.append(f"Plot {Path(plot_file).name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[HF Export] Plot upload error: {e}")
                    errors.append(f"Plot {Path(plot_file).name}: {str(e)}")
        
        # Upload PNG images - use generic file upload
        image_patterns = [
            session_outputs_dir / "*.png",
            global_outputs_dir / "*.png",
            session_outputs_dir / "plots" / "*.png",
            global_outputs_dir / "plots" / "*.png",
            tmp_outputs_dir / "*.png",
            tmp_outputs_dir / "plots" / "*.png"
        ]
        for pattern in image_patterns:
            for image_file in glob.glob(str(pattern)):
                try:
                    logger.info(f"[HF Export] Uploading image: {image_file}")
                    result = hf_service.upload_generic_file(
                        file_path=image_file,
                        session_id=request.session_id,
                        subfolder="images"
                    )
                    if result.get("success"):
                        uploaded_files.append({"type": "image", "name": Path(image_file).name, "url": result.get("url")})
                    else:
                        errors.append(f"Image {Path(image_file).name}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"[HF Export] Image upload error: {e}")
                    errors.append(f"Image {Path(image_file).name}: {str(e)}")
        
        if not uploaded_files and errors:
            logger.error(f"[HF Export] All uploads failed: {errors}")
            raise HTTPException(
                status_code=500, 
                detail=f"Export failed: {len(errors)} file(s) could not be uploaded."
            )
        
        if not uploaded_files and not errors:
            logger.info(f"[HF Export] No files found to export")
            return JSONResponse({
                "success": True,
                "uploaded_files": [],
                "errors": None,
                "message": "No files found to export. Run some analysis first to generate outputs."
            })
        
        logger.info(f"[HF Export] Export completed: {len(uploaded_files)} files uploaded, {len(errors)} errors")
        return JSONResponse({
            "success": True,
            "uploaded_files": uploaded_files,
            "errors": errors if errors else None,
            "message": f"Successfully exported {len(uploaded_files)} files to HuggingFace"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"HuggingFace export failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")


@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serve React frontend for all non-API routes.
    This should be the last route defined.
    """
    frontend_path = Path(__file__).parent.parent.parent / "FRRONTEEEND" / "dist"
    
    # Try to serve the requested file
    file_path = frontend_path / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    
    # Default to index.html for client-side routing
    index_path = frontend_path / "index.html"
    if index_path.exists():
        # Inject Supabase config at runtime for HuggingFace Spaces
        supabase_url = os.getenv("VITE_SUPABASE_URL", "")
        supabase_anon_key = os.getenv("VITE_SUPABASE_ANON_KEY", "")
        
        # Read the HTML file
        html_content = index_path.read_text()
        
        # Inject the config script before </head>
        config_script = f"""
    <script>
      window.__SUPABASE_CONFIG__ = {{
        url: "{supabase_url}",
        anonKey: "{supabase_anon_key}"
      }};
    </script>
  </head>"""
        
        # Replace </head> with our config + </head>
        html_content = html_content.replace("</head>", config_script)
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
    
    # Frontend not built
    raise HTTPException(
        status_code=404,
        detail="Frontend not found. Please build the frontend first: cd FRRONTEEEND && npm run build"
    )


# Cloud Run listens on PORT environment variable
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
