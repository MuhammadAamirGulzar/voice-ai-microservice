"""
Text-to-Speech API Routes

Endpoints for converting text to audio.
"""
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.services.tts_service import TTSService, TTSRequest as ServiceTTSRequest, AVAILABLE_VOICES

router = APIRouter(prefix="/api/tts", tags=["Text-to-Speech"])

# Service instance (initialized lazily)
_tts_service: Optional[TTSService] = None


def init_tts_service(google_credentials_path: str):
    """Initialize the TTS service with credentials."""
    global _tts_service
    _tts_service = TTSService(
        google_credentials_path=google_credentials_path,
        max_concurrent_requests=20,
        enable_cache=True,
        max_cache_size=500,
    )


def get_tts_service() -> TTSService:
    """Get the TTS service instance."""
    if _tts_service is None:
        raise HTTPException(status_code=500, detail="TTS service not initialized")
    return _tts_service


# Request/Response Models
class TTSRequest(BaseModel):
    """Request body for TTS endpoint."""
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(default="Kore", description="Voice name (Aoede, Charon, Fenrir, Kore, Puck, Zephyr)")
    language: str = Field(default="en-US", description="Language code")
    speaking_rate: float = Field(default=1.0, ge=0.25, le=4.0, description="Speaking speed")
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0, description="Voice pitch adjustment")
    audio_format: str = Field(default="mp3", description="Output format (mp3, wav, ogg)")


class TTSResponse(BaseModel):
    """Response from TTS endpoint."""
    audio_base64: str
    audio_format: str
    character_count: int
    duration_seconds: Optional[float] = None  # Estimated audio duration
    processing_time_ms: float
    cost_usd: float  # Renamed for frontend consistency
    from_cache: bool = False


class VoiceInfo(BaseModel):
    """Information about available voices."""
    voices: dict
    default_voice: str = "Kore"


# Endpoints
@router.post("", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio.
    
    Returns base64 encoded audio in the requested format.
    
    **Available Voices:**
    - Aoede: Warm, friendly female
    - Charon: Deep, authoritative male
    - Fenrir: Energetic male
    - Kore: Clear, neutral female (default)
    - Puck: Youthful, bright
    - Zephyr: Calm, professional
    
    **Pricing:** $16/million characters
    """
    try:
        service = get_tts_service()
        
        result = await service.synthesize(ServiceTTSRequest(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            audio_format=request.audio_format,
            request_id=str(uuid.uuid4()),
        ))
        
        return TTSResponse(
            audio_base64=result.audio_base64,
            audio_format=result.audio_format,
            character_count=result.character_count,
            duration_seconds=result.duration_seconds,
            processing_time_ms=round(result.processing_time_ms, 1),
            cost_usd=result.estimated_cost,
            from_cache=result.from_cache,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@router.post("/audio")
async def text_to_speech_audio(request: TTSRequest):
    """
    Convert text to speech and return raw audio bytes.
    
    Useful for direct audio playback without base64 decoding.
    Content-Type header indicates audio format.
    """
    try:
        service = get_tts_service()
        
        result = await service.synthesize(ServiceTTSRequest(
            text=request.text,
            voice=request.voice,
            language=request.language,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            audio_format=request.audio_format,
            request_id=str(uuid.uuid4()),
        ))
        
        media_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
        }.get(request.audio_format, "audio/mpeg")
        
        return Response(
            content=result.get_audio_bytes(),
            media_type=media_type,
            headers={
                "X-Processing-Time-Ms": str(round(result.processing_time_ms, 1)),
                "X-Character-Count": str(result.character_count),
                "X-Estimated-Cost-USD": str(result.estimated_cost),
                "X-From-Cache": str(result.from_cache).lower(),
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@router.get("/voices", response_model=VoiceInfo)
async def get_available_voices():
    """Get list of available TTS voices."""
    return VoiceInfo(
        voices=AVAILABLE_VOICES,
        default_voice="Kore"
    )


@router.get("/metrics")
async def get_tts_metrics():
    """Get TTS service metrics including cache statistics."""
    service = get_tts_service()
    return service.get_metrics()


@router.post("/cache/clear")
async def clear_tts_cache():
    """Clear the TTS response cache."""
    service = get_tts_service()
    service.clear_cache()
    return {"message": "Cache cleared", "cache_size": 0}
