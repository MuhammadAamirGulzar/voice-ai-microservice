"""
Voice Microservice - Main FastAPI Application

Multi-tenant voice service providing:
- STT (Speech-to-Text) - /api/stt/*
- TTS (Text-to-Speech) - /api/tts/*
- Interactive Voice Sessions - /api/v1/voice/*

Supports multiple business cases:
- Recruitment interviews
- Education/tutoring
- Visa processing consultations
- General voice interactions
"""
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings

# =============================================================================
# Logging Configuration
# =============================================================================

logs_dir = Path(__file__).parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
(logs_dir / "transcripts").mkdir(parents=True, exist_ok=True)

logger.remove()

# Console - concise format with colors
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

# File - detailed format with rotation
logger.add(
    str(logs_dir / "app.log"),
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Error log - separate file for errors only
logger.add(
    str(logs_dir / "error.log"),
    rotation="1 day",
    retention="30 days",
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}"
)

# =============================================================================
# FastAPI Application
# =============================================================================

settings = get_settings()

app = FastAPI(
    title="Voice Microservice API",
    description="""
## Multi-tenant Voice Service

A scalable voice microservice supporting STT, TTS, and interactive voice sessions.

### Core Capabilities

#### 1. Speech-to-Text (STT)
Convert audio to text using AssemblyAI.
- `POST /api/stt` - Transcribe base64 audio
- `POST /api/stt/upload` - Transcribe uploaded file

#### 2. Text-to-Speech (TTS)
Convert text to natural speech using Google Cloud TTS.
- `POST /api/tts` - Get base64 audio
- `POST /api/tts/audio` - Get raw audio bytes
- `GET /api/tts/voices` - List available voices

#### 3. Interactive Voice Sessions (NEW)
WebRTC-based voice conversations with AI Layer integration.
- `POST /api/v1/voice/session/start` - Initialize session with multi-tenant context
- `POST /api/v1/voice/session/connect` - WebRTC SDP exchange
- `POST /api/v1/voice/session/{id}/stop` - End session
- `GET /api/v1/voice/session/{id}/transcript` - Get live transcript

### Session Modes
- **interactive**: Full conversation (STT → AI Layer → TTS)
- **stt_only**: Speech-to-Text only
- **tts_only**: Text-to-Speech only

### Business Cases
- Recruitment interviews
- Education/tutoring
- Visa processing
- General voice interactions
    """,
    version=settings.service_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# =============================================================================
# Middleware
# =============================================================================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    import time
    start_time = time.time()
    
    # Skip logging for static files and health checks
    if request.url.path.startswith("/static") or request.url.path == "/health":
        return await call_next(request)
    
    logger.debug(f"→ {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    duration = (time.time() - start_time) * 1000
    logger.debug(f"← {request.method} {request.url.path} | {response.status_code} | {duration:.0f}ms")
    
    return response


# =============================================================================
# Include Routers
# =============================================================================

from backend.routes.stt_routes import router as stt_router, init_stt_service
from backend.routes.tts_routes import router as tts_router, init_tts_service
from backend.routes.conversation_routes import router as voice_router

# Multi-tenant voice API
app.include_router(voice_router)

# STT/TTS standalone endpoints
app.include_router(stt_router)
app.include_router(tts_router)

# =============================================================================
# Core Endpoints
# =============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """API info endpoint."""
    return {
        "service": "Voice Microservice API",
        "version": settings.service_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and active session count.
    """
    from backend.services.session_manager import get_session_manager
    from backend.services.ai_layer_client import get_ai_layer_client
    
    session_manager = get_session_manager()
    stats = session_manager.get_stats()
    
    # Check AI Layer connectivity
    ai_layer_status = "unknown"
    try:
        ai_client = get_ai_layer_client()
        ai_layer_status = "configured" if ai_client else "not_configured"
    except Exception:
        ai_layer_status = "error"
    
    return {
        "status": "healthy",
        "service": settings.service_name,
        "version": settings.service_version,
        "services": {
            "stt": "assemblyai",
            "tts": "google-chirp3-hd",
            "ai_layer": ai_layer_status,
        },
        "sessions": {
            "active": stats.get("active_sessions", 0),
            "total": stats.get("total_sessions", 0),
        },
    }


@app.get("/api/config")
async def get_config():
    """Get client configuration."""
    from backend.services.tts_service import AVAILABLE_VOICES
    
    return {
        "service": settings.service_name,
        "version": settings.service_version,
        "max_session_duration_minutes": settings.max_session_duration_minutes,
        "default_session_mode": settings.default_session_mode,
        "tts": {
            "default_voice": settings.tts_voice_name,
            "available_voices": list(AVAILABLE_VOICES.keys()),
            "default_language": settings.tts_language_code,
        },
        "stt": {
            "sample_rate": settings.stt_sample_rate,
            "default_language": settings.stt_language,
        },
        "ai_layer": {
            "configured": bool(settings.ai_layer_base_url),
            "use_local_llm": settings.use_local_llm,
        },
    }


@app.get("/api/stats")
async def get_stats():
    """Get service statistics."""
    from backend.services.session_manager import get_session_manager
    from backend.processors.transcript_logger import get_transcript_stats
    
    session_manager = get_session_manager()
    session_stats = session_manager.get_stats()
    transcript_stats = get_transcript_stats(settings.transcript_dir)
    
    return {
        "sessions": session_stats,
        "transcripts": transcript_stats,
    }


@app.get("/api/metrics")
async def get_all_metrics():
    """Get aggregated metrics for all services."""
    from backend.routes.stt_routes import get_stt_service
    from backend.routes.tts_routes import get_tts_service
    from backend.services.session_manager import get_session_manager
    
    stt_metrics = {}
    tts_metrics = {}
    
    try:
        stt_metrics = get_stt_service().get_metrics()
    except Exception:
        pass
    
    try:
        tts_metrics = get_tts_service().get_metrics()
    except Exception:
        pass
    
    session_manager = get_session_manager()
    
    return {
        "stt": stt_metrics,
        "tts": tts_metrics,
        "sessions": session_manager.get_stats(),
    }


@app.get("/api/pricing")
async def get_pricing():
    """Get current API pricing information."""
    return {
        "stt": {
            "provider": "AssemblyAI",
            "price_per_second_usd": 0.00025,
            "price_per_minute_usd": 0.015,
        },
        "tts": {
            "provider": "Google Cloud TTS (Chirp3 HD)",
            "price_per_million_chars_usd": 16.0,
        },
        "ai_layer": {
            "note": "Pricing depends on AI Layer configuration"
        },
    }


# =============================================================================
# Transcript Endpoints
# =============================================================================

@app.get("/api/transcripts")
async def list_transcripts(
    organisation_id: str = None,
    limit: int = 100,
):
    """
    List saved session transcripts.
    
    Optionally filter by organization.
    """
    from backend.processors.transcript_logger import list_transcripts as get_transcripts
    
    transcripts = get_transcripts(
        transcripts_dir=settings.transcript_dir,
        organisation_id=organisation_id,
        limit=limit,
    )
    
    return {"count": len(transcripts), "transcripts": transcripts}


@app.get("/api/transcripts/{session_id}")
async def get_transcript(session_id: str):
    """Get transcript for a specific session."""
    from backend.processors.transcript_logger import load_transcript
    
    transcript = load_transcript(session_id, settings.transcript_dir)
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return transcript


# =============================================================================
# Lifecycle Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("=" * 70)
    logger.info(f"🚀 {settings.service_name} v{settings.service_version} starting...")
    logger.info("")
    
    # Initialize STT Service
    try:
        init_stt_service(settings.assemblyai_api_key)
        logger.info("   ✅ STT Service (AssemblyAI)")
    except Exception as e:
        logger.warning(f"   ⚠️ STT Service failed: {e}")
    
    # Initialize TTS Service
    try:
        creds_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            settings.google_cloud_tts_credentials
        )
        init_tts_service(creds_path)
        logger.info("   ✅ TTS Service (Google Chirp3 HD)")
    except Exception as e:
        logger.warning(f"   ⚠️ TTS Service failed: {e}")
    
    # Initialize AI Layer Client
    try:
        from backend.services.ai_layer_client import get_ai_layer_client
        ai_client = get_ai_layer_client()
        if settings.use_local_llm:
            logger.info(f"   ✅ Local LLM ({settings.llm_provider}/{settings.llm_model})")
        else:
            logger.info(f"   ✅ AI Layer Client ({settings.ai_layer_base_url})")
    except Exception as e:
        logger.warning(f"   ⚠️ AI Layer Client failed: {e}")
    
    logger.info("")
    logger.info("   Endpoints:")
    logger.info("   ├── Voice:  POST /api/v1/voice/session/start, /connect")
    logger.info("   ├── STT:    POST /api/stt, /api/stt/upload")
    logger.info("   ├── TTS:    POST /api/tts, /api/tts/audio")
    logger.info("   └── Legacy: POST /api/conversation/start, /connect")
    logger.info("")
    logger.info(f"   Mode: {'Local LLM' if settings.use_local_llm else 'AI Layer'}")
    logger.info(f"   Debug: {settings.debug}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Voice Microservice...")
    
    # Close all active sessions
    from backend.services.session_manager import get_session_manager
    
    session_manager = get_session_manager()
    active_sessions = session_manager.get_active_sessions()
    
    for session in active_sessions:
        try:
            await session_manager.close_session(session.session_id)
        except Exception as e:
            logger.warning(f"Failed to close session {session.session_id}: {e}")
    
    # Close service connections
    from backend.routes.stt_routes import _stt_service
    from backend.routes.tts_routes import _tts_service
    
    if _stt_service:
        await _stt_service.close()
    if _tts_service:
        await _tts_service.close()
    
    logger.info("Voice Microservice stopped")


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path),
        },
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
