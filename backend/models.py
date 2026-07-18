"""
API Models - Request/Response schemas for Voice Microservice.

Defines Pydantic models for:
- Multi-tenant session requests
- STT/TTS requests
- WebRTC connection payloads
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class SessionModeEnum(str, Enum):
    """Voice session modes."""
    STT_ONLY = "stt_only"
    TTS_ONLY = "tts_only"
    INTERACTIVE = "interactive"


# =============================================================================
# Session Models
# =============================================================================

class SessionStartRequest(BaseModel):
    """Request to start a new voice session."""
    
    # Multi-tenant identifiers (required for production)
    organisation_id: Optional[str] = Field(
        None, 
        description="Organization identifier from frontend"
    )
    agent_id: Optional[str] = Field(
        None, 
        description="Agent/bot identifier from frontend"
    )
    user_id: Optional[str] = Field(
        None, 
        description="User identifier from frontend"
    )
    
    # Session configuration
    mode: SessionModeEnum = Field(
        default=SessionModeEnum.INTERACTIVE,
        description="Session mode: stt_only, tts_only, or interactive"
    )
    
    # Voice configuration
    language: str = Field(
        default="en-US",
        description="Language code for STT/TTS"
    )
    voice: str = Field(
        default="Zephyr",
        description="TTS voice name"
    )
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional client metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "organisation_id": "org_123",
                "agent_id": "agent_456",
                "user_id": "user_789",
                "mode": "interactive",
                "language": "en-US",
                "voice": "Zephyr",
                "metadata": {"interview_type": "technical"}
            }
        }


class SessionStartResponse(BaseModel):
    """Response from session start."""
    session_id: str
    status: str
    mode: str
    message: str
    next_step: str
    
    # Context echo
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


class SessionConnectRequest(BaseModel):
    """WebRTC connection request."""
    session_id: Optional[str] = Field(
        None,
        description="Session ID from /start endpoint (or will be generated)"
    )
    
    # WebRTC SDP
    sdp: Optional[str] = Field(
        None,
        description="WebRTC SDP offer"
    )
    sdp_offer: Optional[str] = Field(
        None,
        description="Alternative field for SDP offer"
    )
    type: str = Field(
        default="offer",
        description="SDP type (always 'offer' from client)"
    )
    
    # Multi-tenant context (if not using /start first)
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    mode: SessionModeEnum = SessionModeEnum.INTERACTIVE
    language: str = "en-US"
    voice: str = "Zephyr"
    metadata: Optional[Dict[str, Any]] = None
    
    def get_sdp(self) -> Optional[str]:
        """Get SDP from either field."""
        return self.sdp_offer or self.sdp


class SessionConnectResponse(BaseModel):
    """WebRTC connection response."""
    sdp_answer: str
    type: str = "answer"
    session_id: str
    status: str = "connected"


class SessionStatusResponse(BaseModel):
    """Session status response."""
    session_id: str
    status: str
    is_active: bool
    duration_seconds: float = 0
    message_count: int = 0
    
    # Context
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    mode: str = "interactive"


class SessionStopResponse(BaseModel):
    """Response from stopping a session."""
    session_id: str
    status: str = "stopped"
    duration_seconds: float
    message_count: int
    metrics: Optional[Dict[str, Any]] = None


# =============================================================================
# STT Models
# =============================================================================

class STTRequest(BaseModel):
    """Speech-to-Text request."""
    audio_base64: Optional[str] = Field(
        None,
        description="Base64 encoded audio data"
    )
    audio_url: Optional[str] = Field(
        None,
        description="URL to audio file"
    )
    language: str = Field(
        default="en",
        description="Language code"
    )
    
    # Multi-tenant context (optional, for logging/billing)
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class STTResponse(BaseModel):
    """Speech-to-Text response."""
    text: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    processing_time_ms: float
    cost_usd: float
    
    # Echo session context
    session_id: Optional[str] = None


# =============================================================================
# TTS Models
# =============================================================================

class TTSRequest(BaseModel):
    """Text-to-Speech request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to convert to speech"
    )
    voice: str = Field(
        default="Zephyr",
        description="Voice name"
    )
    language: str = Field(
        default="en-US",
        description="Language code"
    )
    speaking_rate: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speaking speed"
    )
    pitch: float = Field(
        default=0.0,
        ge=-20.0,
        le=20.0,
        description="Voice pitch"
    )
    audio_format: str = Field(
        default="mp3",
        description="Output format"
    )
    
    # Multi-tenant context (optional)
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class TTSResponse(BaseModel):
    """Text-to-Speech response."""
    audio_base64: str
    audio_format: str
    character_count: int
    duration_seconds: Optional[float] = None
    processing_time_ms: float
    cost_usd: float
    from_cache: bool = False
    
    # Echo session context
    session_id: Optional[str] = None


# =============================================================================
# Interactive Session Models (for non-WebRTC text interaction)
# =============================================================================

class InteractiveMessageRequest(BaseModel):
    """
    Send a text message in an interactive session.
    Used for text-based interaction or when audio is handled separately.
    """
    session_id: str = Field(..., description="Active session ID")
    message: str = Field(..., min_length=1, description="User message text")
    
    # Optional: include audio for TTS response
    return_audio: bool = Field(
        default=True,
        description="Whether to return TTS audio in response"
    )
    voice: Optional[str] = Field(
        None,
        description="Override session voice for this response"
    )


class InteractiveMessageResponse(BaseModel):
    """Response from interactive message."""
    session_id: str
    response_text: str
    
    # Optional audio (if return_audio=True)
    audio_base64: Optional[str] = None
    audio_format: Optional[str] = None
    
    # Metrics
    processing_time_ms: float
    ai_layer_latency_ms: Optional[float] = None
    tts_latency_ms: Optional[float] = None


# =============================================================================
# Health & Stats Models
# =============================================================================

class ServiceHealth(BaseModel):
    """Service health status."""
    status: str
    version: str
    services: Dict[str, str]
    active_sessions: int
    ai_layer_status: Optional[str] = None


class SessionStats(BaseModel):
    """Session statistics."""
    total_sessions: int
    active_sessions: int
    organizations: int
    users: int
    sessions_by_status: Dict[str, int]
    sessions_by_mode: Dict[str, int]
