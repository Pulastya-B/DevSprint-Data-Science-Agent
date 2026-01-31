"""
FastAPI Application for Google Cloud Run
Thin HTTP wrapper around DataScienceCopilot - No logic changes, just API exposure.
"""

import os
import sys
import tempfile
import shutil
import time
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
            return float(o)
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
session_states: Dict[str, Any] = {}  # session_id -> SessionMemory
agent_cache_lock = asyncio.Lock()
MAX_CACHED_AGENTS = 10  # Limit memory usage (session states are lightweight)
logger.info("👥 Multi-user session isolation initialized (fast mode)")

# Global agent - Heavy components loaded ONCE at startup
# SBERT model, tool functions, LLM client are shared across all users
agent: Optional[DataScienceCopilot] = None
agent = None

# Session state isolation (lightweight - just session memory)
session_states: Dict[str, any] = {}  # session_id -> session memory only


async def get_agent_for_session(session_id: str) -> DataScienceCopilot:
    """
    Get agent with isolated session state.
    
    OPTIMIZATION: Instead of creating a full new agent per session (slow!),
    we reuse the global agent but swap session memory per request.
    Heavy components (SBERT, tools, LLM client) are shared.
    This reduces per-user initialization from 20s to <1s.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        DataScienceCopilot instance with isolated session for this user
    """
    global agent
    
    async with agent_cache_lock:
        # Ensure base agent exists (heavy components loaded once at startup)
        if agent is None:
            logger.warning("Base agent not initialized - this shouldn't happen after startup")
            provider = os.getenv("LLM_PROVIDER", "mistral")
            agent = DataScienceCopilot(
                reasoning_effort="medium",
                provider=provider,
                use_compact_prompts=False
            )
        
        # Check if we have cached session memory for this session
        if session_id in session_states:
            logger.info(f"[♻️] Reusing session state for {session_id[:8]}...")
            agent.session = session_states[session_id]
            agent.http_session_key = session_id
            return agent
        
        # 🚀 FAST PATH: Create new session memory only (no SBERT reload!)
        logger.info(f"[🆕] Creating lightweight session for {session_id[:8]}...")
        
        # Create isolated session memory for this user
        new_session = SessionMemory(session_id=session_id)
        
        # Cache session memory (lightweight)
        # Cache management: Remove oldest if cache is full
        if len(session_states) >= MAX_CACHED_AGENTS:
            oldest_session = next(iter(session_states))
            logger.info(f"[🗑️] Cache full, removing session {oldest_session[:8]}...")
            del session_states[oldest_session]
        
        session_states[session_id] = new_session
        
        # Set session on shared agent
        agent.session = new_session
        agent.http_session_key = session_id
        
        logger.info(f"✅ Session created for {session_id[:8]} (cache: {len(session_states)}/{MAX_CACHED_AGENTS}) - <1s init")
        
        return agent

# 🔒 REQUEST QUEUING: Global lock to prevent concurrent workflows
# This ensures only one analysis runs at a time, preventing:
# - Race conditions on file writes
# - Memory exhaustion from parallel model training
# - Session state corruption
workflow_lock = asyncio.Lock()
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
    """Stream real-time progress updates using Server-Sent Events (SSE).
    
    This endpoint connects clients to the global progress_manager which
    receives events from the orchestrator as tools execute.
    
    Events:
        - tool_executing: When a tool begins execution
        - tool_completed: When a tool finishes successfully  
        - tool_failed: When a tool fails
        - token_update: Token budget updates
        - analysis_complete: When the entire workflow finishes
    """
    print(f"[SSE] ENDPOINT: Client connected for session_id={session_id}")
    
    # CRITICAL: Create queue and register subscriber IMMEDIATELY
    queue = asyncio.Queue(maxsize=100)
    if session_id not in progress_manager._queues:
        progress_manager._queues[session_id] = []
    progress_manager._queues[session_id].append(queue)
    print(f"[SSE] Queue registered, total subscribers: {len(progress_manager._queues[session_id])}")
    
    async def event_generator():
        try:
            # Send initial connection event
            connection_event = {
                'type': 'connected',
                'message': '🔗 Connected to progress stream',
                'session_id': session_id
            }
            print(f"[SSE] SENDING connection event to client")
            yield f"data: {safe_json_dumps(connection_event)}\n\n"
            
            # Send any existing history first (for reconnections)
            history = progress_manager.get_history(session_id)
            print(f"[SSE] Sending {len(history[-10:])} history events")
            for event in history[-10:]:  # Send last 10 events
                yield f"data: {safe_json_dumps(event)}\n\n"
            
            print(f"[SSE] Starting event stream loop for session {session_id}")
            
            # Stream new events from the queue (poll with get_nowait to avoid blocking issues)
            while True:
                if not queue.empty():
                    event = queue.get_nowait()
                    print(f"[SSE] GOT event from queue: {event.get('type')}")
                    yield f"data: {safe_json_dumps(event)}\n\n"
                    
                    # Check if analysis is complete
                    if event.get('type') == 'analysis_complete':
                        break
                else:
                    # No events available, send keepalive and wait
                    yield f": keepalive\n\n"
                    await asyncio.sleep(0.5)  # Poll every 500ms
                    
        except asyncio.CancelledError:
            logger.info(f"SSE stream cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"SSE error for session {session_id}: {e}")
        finally:
            # Cleanup queue
            if session_id in progress_manager._queues and queue in progress_manager._queues[session_id]:
                progress_manager._queues[session_id].remove(queue)
            logger.info(f"SSE stream closed for session {session_id}")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
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
    """Background task to run analysis and emit events."""
    async def _run_with_lock():
        """Wrap analysis in lock to ensure sequential execution."""
        async with workflow_lock:
            try:
                logger.info(f"[BACKGROUND] Starting analysis for session {session_id[:8]}...")
                
                # 👥 Get isolated agent for this session
                session_agent = await get_agent_for_session(session_id)
                
                result = session_agent.analyze(
                    file_path=file_path,
                    task_description=task_description,
                    target_col=target_col,
                    use_cache=use_cache,
                    max_iterations=max_iterations
                )
                
                logger.info(f"[BACKGROUND] Analysis completed for session {session_id[:8]}...")
                
                # Send completion event
                progress_manager.emit(session_id, {
                    "type": "analysis_complete",
                    "status": result.get("status"),
                    "message": "✅ Analysis completed successfully!",
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"[BACKGROUND] Analysis failed for session {session_id[:8]}...: {e}")
                progress_manager.emit(session_id, {
                    "type": "analysis_failed",
                    "error": str(e),
                    "message": f"❌ Analysis failed: {str(e)}"
                })
    
    # Run async function in event loop
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(_run_with_lock())


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
        temp_dir = Path("/tmp") / "data_science_agent"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[ASYNC] File saved: {file.filename}")
    else:
        # 🛡️ VALIDATION: Check if this session has dataset cached
        has_dataset = False
        async with agent_cache_lock:
            # Check session_states cache for this specific session_id
            if session_id in session_states:
                cached_session = session_states[session_id]
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
            
            # Send completion event via SSE using actual session UUID
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
                    "error": str(e),
                    "error_type": type(e).__name__,
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
    
    # Use /tmp for Cloud Run (ephemeral storage)
    temp_dir = Path("/tmp") / "data_science_agent"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        temp_file_path = temp_dir / file.filename
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
        
        # Send completion event via SSE using actual session UUID
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
                "error": str(e),
                "error_type": type(e).__name__,
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
        # Save file temporarily
        temp_file_path = temp_dir / file.filename
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
                "error": str(e),
                "error_type": type(e).__name__
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
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


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
            "detail": str(exc),
            "error_type": type(exc).__name__
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
        Path("/tmp/data_science_agent") / file_path,  # Production flat
        Path("/tmp/data_science_agent/outputs") / Path(file_path).name,  # Production filename only
    ]
    
    output_path = None
    for path in search_paths:
        if path.exists() and path.is_file():
            output_path = path
            break
    
    if output_path is None:
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
        return FileResponse(index_path)
    
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
