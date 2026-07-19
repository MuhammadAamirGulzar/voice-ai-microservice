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
import asyncio
import hmac
import json
import os
import sys
import time
from collections import deque
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

# CORS — origins are configurable; wildcard cannot legally be combined with
# credentials (browsers reject it), so credentials turn on only when the
# deployment pins real origins.
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths that must stay open for load balancers, docs, and browsers.
_OPEN_PATHS = {"/", "/health", "/health/deep", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """
    Service-to-service auth: when VOICE_API_KEY is set, every /api/* request
    must carry it in X-API-Key. Without this, anyone who can reach the
    service can burn STT/TTS spend and read any tenant's transcripts.
    """
    if settings.api_key and request.url.path.startswith("/api"):
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, settings.api_key):
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing X-API-Key",
                         "status_code": 401,
                         "path": request.url.path},
            )
    return await call_next(request)


# Sliding-window rate limiter (per client IP). Opt-in: RATE_LIMIT_ENABLED=1.
# In-memory by design — this protects one instance from abuse; cross-instance
# fairness belongs in the gateway in front of a scaled deployment.
_rate_windows: dict = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if (settings.rate_limit_enabled
            and request.url.path.startswith("/api")
            and request.url.path not in _OPEN_PATHS):
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = _rate_windows.setdefault(client_ip, deque())
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= settings.rate_limit_requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded; retry shortly",
                         "status_code": 429,
                         "path": request.url.path},
                headers={"Retry-After": "10"},
            )
        window.append(now)
        # Drop idle IPs so the map can't grow unbounded.
        if len(_rate_windows) > 10000:
            for ip in [ip for ip, w in _rate_windows.items() if not w][:5000]:
                _rate_windows.pop(ip, None)
    return await call_next(request)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
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


@app.get("/health/deep")
async def deep_health_check():
    """
    Dependency-level health: actually pings the AI Layer. Kept separate
    from /health so load-balancer probes stay cheap and local.
    """
    from backend.services.ai_layer_client import get_ai_layer_client

    ai_layer = {"status": "unknown"}
    try:
        ai_layer = await get_ai_layer_client().health_check()
    except Exception as e:
        ai_layer = {"status": "error", "error": str(e)}

    healthy = ai_layer.get("status") == "healthy" or settings.use_local_llm
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "healthy" if healthy else "degraded",
            "ai_layer": ai_layer,
            "use_local_llm": settings.use_local_llm,
        },
    )


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
    # Scans and parses every transcript on disk — run it off the event loop
    # so a large transcript store can't stall live audio sessions.
    transcript_stats = await asyncio.to_thread(
        get_transcript_stats, settings.transcript_dir)

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

    transcripts = await asyncio.to_thread(
        get_transcripts,
        settings.transcript_dir,
        organisation_id,
        limit,
    )

    return {"count": len(transcripts), "transcripts": transcripts}


@app.get("/api/transcripts/{session_id}")
async def get_transcript(session_id: str):
    """Get transcript for a specific session."""
    from backend.processors.transcript_logger import load_transcript

    transcript = await asyncio.to_thread(
        load_transcript, session_id, settings.transcript_dir)

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
    # Session reaper: closes never-connected sessions and enforces the max
    # call duration. Without it both leak memory and provider spend forever.
    async def _session_reaper():
        from backend.services.session_manager import get_session_manager
        while True:
            await asyncio.sleep(settings.session_cleanup_interval_seconds)
            try:
                reaped = await get_session_manager().cleanup_stale_sessions(
                    max_pending_minutes=settings.max_pending_session_minutes,
                    max_duration_minutes=settings.max_session_duration_minutes,
                )
                if reaped:
                    logger.info(f"Session reaper closed {reaped} stale session(s)")
            except Exception as e:
                logger.error(f"Session reaper error: {e}")

    app.state.session_reaper = asyncio.create_task(_session_reaper())

    logger.info("")
    logger.info(f"   Mode: {'Local LLM' if settings.use_local_llm else 'AI Layer'}")
    logger.info(f"   Auth: {'X-API-Key required on /api/*' if settings.api_key else 'OPEN (set VOICE_API_KEY in production)'}")
    logger.info(f"   Debug: {settings.debug}")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Voice Microservice...")

    reaper = getattr(app.state, "session_reaper", None)
    if reaper is not None:
        reaper.cancel()

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
