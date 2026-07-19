"""
Real-time Conversation API Routes - Multi-tenant Voice Sessions

Endpoints for WebRTC-based voice conversations with AI Layer integration.
Supports multiple business cases (recruitment, education, etc.)

Modes:
- STT_ONLY: Speech-to-Text only, returns transcribed text
- TTS_ONLY: Text-to-Speech only, expects text input
- INTERACTIVE: Full conversation flow: STT → AI Layer → TTS
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from backend.models import (
    SessionStartRequest,
    SessionStartResponse,
    SessionConnectRequest,
    SessionConnectResponse,
    SessionStatusResponse,
    SessionStopResponse,
    InteractiveMessageRequest,
    InteractiveMessageResponse,
    SessionStats,
)
from backend.services.session_manager import (
    get_session_manager,
    SessionStatus,
    SessionMode,
)
from backend.services.ai_layer_client import (
    get_ai_layer_client,
    AILayerRequest,
    AILayerError,
)

router = APIRouter(prefix="/api/v1/voice", tags=["Voice Sessions"])


# =============================================================================
# Session Management Endpoints
# =============================================================================

@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    """
    Initialize a new voice session.
    
    **Multi-tenant Identifiers:**
    - `organisation_id`: Your organization ID
    - `agent_id`: The AI agent/bot ID
    - `user_id`: The end user ID
    
    **Session Modes:**
    - `stt_only`: Speech-to-Text only
    - `tts_only`: Text-to-Speech only
    - `interactive`: Full conversation (STT → AI → TTS)
    
    **Flow:**
    1. Call this endpoint to initialize session
    2. Call `/session/connect` with WebRTC SDP offer
    3. Audio streams via WebRTC
    4. Call `/session/{session_id}/stop` when done
    """
    try:
        session_manager = get_session_manager()
        
        session = session_manager.create_session(
            organisation_id=request.organisation_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
            mode=request.mode.value,
            language=request.language,
            voice=request.voice,
            client_metadata=request.metadata,
        )
        
        logger.info(
            f"🎙️ Session started: {session.session_id} | "
            f"org={request.organisation_id} | mode={request.mode.value}"
        )
        
        return SessionStartResponse(
            session_id=session.session_id,
            status="initialized",
            mode=session.context.mode.value,
            message="Session initialized. Ready for WebRTC connection.",
            next_step="POST /api/v1/voice/session/connect with SDP offer",
            organisation_id=request.organisation_id,
            agent_id=request.agent_id,
            user_id=request.user_id,
        )
        
    except Exception as e:
        logger.error(f"Session start failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/connect", response_model=SessionConnectResponse)
async def connect_session(request: SessionConnectRequest):
    """
    Establish WebRTC connection for a voice session.
    
    **Request:**
    - `session_id`: From `/start` endpoint (or auto-generated)
    - `sdp` or `sdp_offer`: WebRTC SDP offer from client
    
    **Response:**
    - `sdp_answer`: WebRTC SDP answer to establish connection
    
    If `session_id` is not provided, a new session is created automatically.
    """
    try:
        from backend.pipeline import create_and_run_pipeline
        from config.settings import get_settings
        
        session_manager = get_session_manager()
        settings = get_settings()
        
        sdp_offer = request.get_sdp()
        if not sdp_offer:
            raise HTTPException(
                status_code=400, 
                detail="Missing SDP offer (provide 'sdp' or 'sdp_offer')"
            )
        
        # Get or create session
        session = None
        if request.session_id:
            session = session_manager.get_session(request.session_id)
            if not session:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid session_id: {request.session_id}"
                )
        
        if not session:
            # Auto-create session with provided context
            session = session_manager.create_session(
                organisation_id=request.organisation_id,
                agent_id=request.agent_id,
                user_id=request.user_id,
                mode=request.mode.value,
                language=request.language,
                voice=request.voice,
                client_metadata=request.metadata,
            )
        
        # Update session status
        session_manager.update_session_status(
            session.session_id,
            SessionStatus.CONNECTING
        )

        # Create pipeline and get SDP answer. On failure the session must
        # not linger in CONNECTING — mark it errored and reap immediately.
        try:
            sdp_answer, task, runner, metrics_processor, transcript_logger = await create_and_run_pipeline(
                session=session,
                sdp_offer=sdp_offer,
                settings=settings,
            )
        except Exception:
            session_manager.update_session_status(
                session.session_id, SessionStatus.ERROR)
            await session_manager.close_session(session.session_id)
            raise
        
        # Store pipeline components in session
        session.task = task
        session.runner = runner
        session.metrics_processor = metrics_processor
        session.transcript_logger = transcript_logger
        
        # Update status
        session_manager.update_session_status(
            session.session_id, 
            SessionStatus.CONNECTED
        )
        
        logger.info(f"🔗 WebRTC connected: {session.session_id}")
        
        return SessionConnectResponse(
            sdp_answer=sdp_answer,
            type="answer",
            session_id=session.session_id,
            status="connected",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WebRTC connect failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/{session_id}/stop", response_model=SessionStopResponse)
async def stop_session(session_id: str):
    """
    Stop a voice session and get final metrics.
    
    Returns:
    - Session duration and message count
    - Performance metrics (latency, costs)
    - Transcript is saved automatically
    """
    try:
        session_manager = get_session_manager()
        
        summary = await session_manager.close_session(session_id)
        
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"🛑 Session stopped: {session_id}")
        
        return SessionStopResponse(
            session_id=session_id,
            status="stopped",
            duration_seconds=summary.get("duration_seconds", 0),
            message_count=summary.get("message_count", 0),
            metrics=summary.get("metrics"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """Get status of a voice session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatusResponse(
        session_id=session_id,
        status=session.status.value,
        is_active=session.is_active,
        duration_seconds=session.duration_seconds,
        message_count=session.message_count,
        organisation_id=session.context.organisation_id,
        agent_id=session.context.agent_id,
        user_id=session.context.user_id,
        mode=session.context.mode.value,
    )


@router.get("/session/{session_id}/transcript")
async def get_session_transcript(session_id: str):
    """
    Get live transcript for an active session.
    
    Returns recent conversation messages for UI updates.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.transcript_logger:
        return {"messages": [], "count": 0}
    
    transcript = session.transcript_logger.transcript
    messages = transcript.get("conversation", [])
    
    return {
        "session_id": session_id,
        "messages": messages[-20:],  # Last 20 messages
        "count": len(messages),
        "status": session.status.value,
    }


@router.get("/session/{session_id}/metrics")
async def get_session_metrics(session_id: str):
    """Get performance metrics for a session."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.metrics_processor:
        return {"message": "Metrics not available yet"}
    
    return session.metrics_processor.get_summary()


# =============================================================================
# Interactive Text Endpoints (Non-WebRTC)
# =============================================================================

@router.post("/session/{session_id}/message", response_model=InteractiveMessageResponse)
async def send_message(session_id: str, request: InteractiveMessageRequest):
    """
    Send a text message in an active session.
    
    Use this for:
    - Text-based chat fallback
    - Testing without WebRTC
    - Hybrid text+voice interactions
    
    The message is sent to AI Layer and optionally converted to speech.
    """
    import time
    
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    start_time = time.time()
    
    try:
        # Send to AI Layer
        ai_client = get_ai_layer_client()
        
        ai_request = AILayerRequest(
            organisation_id=session.context.organisation_id or "",
            agent_id=session.context.agent_id or "",
            user_id=session.context.user_id or "",
            session_id=session_id,
            message=request.message,
        )
        
        ai_start = time.time()
        ai_response = await ai_client.send_message(ai_request)
        ai_latency = (time.time() - ai_start) * 1000
        
        response_text = ai_response.text
        audio_base64 = None
        audio_format = None
        tts_latency = None
        
        # Convert to speech if requested
        if request.return_audio and response_text:
            from backend.routes.tts_routes import get_tts_service
            from backend.services.tts_service import TTSRequest as ServiceTTSRequest
            import uuid
            
            tts_service = get_tts_service()
            voice = request.voice or session.context.voice
            
            tts_start = time.time()
            tts_result = await tts_service.synthesize(ServiceTTSRequest(
                text=response_text,
                voice=voice,
                language=session.context.language,
                request_id=str(uuid.uuid4()),
            ))
            tts_latency = (time.time() - tts_start) * 1000
            
            audio_base64 = tts_result.audio_base64
            audio_format = tts_result.audio_format
        
        # Update session activity
        session.update_activity()
        
        # Log to transcript if available
        if session.transcript_logger:
            session.transcript_logger.add_message("User", request.message)
            session.transcript_logger.add_message("Assistant", response_text)
        
        total_time = (time.time() - start_time) * 1000
        
        return InteractiveMessageResponse(
            session_id=session_id,
            response_text=response_text,
            audio_base64=audio_base64,
            audio_format=audio_format,
            processing_time_ms=round(total_time, 2),
            ai_layer_latency_ms=round(ai_latency, 2),
            tts_latency_ms=round(tts_latency, 2) if tts_latency else None,
        )
        
    except AILayerError as e:
        logger.error(f"AI Layer error in message: {e}")
        raise HTTPException(status_code=502, detail=f"AI Layer error: {str(e)}")
    except Exception as e:
        logger.error(f"Message processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Session Listing & Stats
# =============================================================================

@router.get("/sessions")
async def list_sessions(
    organisation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    active_only: bool = False,
    limit: int = 100,
):
    """
    List voice sessions with optional filters.
    
    Filters:
    - `organisation_id`: Filter by organization
    - `user_id`: Filter by user
    - `active_only`: Only show active sessions
    """
    session_manager = get_session_manager()
    
    sessions = session_manager.list_sessions(
        organisation_id=organisation_id,
        user_id=user_id,
        status=SessionStatus.CONNECTED if active_only else None,
        limit=limit,
    )
    
    return {
        "count": len(sessions),
        "sessions": [s.to_dict() for s in sessions],
    }


@router.get("/stats", response_model=SessionStats)
async def get_session_stats():
    """Get voice session statistics."""
    session_manager = get_session_manager()
    return session_manager.get_stats()
