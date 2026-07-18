"""
Speech-to-Text API Routes

Endpoints for converting audio to text.
"""
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.services.stt_service import STTService, STTRequest as ServiceSTTRequest

router = APIRouter(prefix="/api/stt", tags=["Speech-to-Text"])

# Service instance (initialized lazily)
_stt_service: Optional[STTService] = None


def init_stt_service(assemblyai_api_key: str):
    """Initialize the STT service with API key."""
    global _stt_service
    _stt_service = STTService(
        assemblyai_api_key=assemblyai_api_key,
        max_concurrent_requests=20,
    )


def get_stt_service() -> STTService:
    """Get the STT service instance."""
    if _stt_service is None:
        raise HTTPException(status_code=500, detail="STT service not initialized")
    return _stt_service


# Request/Response Models
class STTRequest(BaseModel):
    """Request body for STT endpoint."""
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    language: str = "en"


class STTResponse(BaseModel):
    """Response from STT endpoint."""
    text: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    processing_time_ms: float
    cost_usd: float  # Renamed for frontend consistency


# Endpoints
@router.post("", response_model=STTResponse)
async def speech_to_text(request: STTRequest):
    """
    Convert speech audio to text.
    
    Accepts either:
    - **audio_base64**: Base64 encoded audio data (mp3, wav, webm, etc.)
    - **audio_url**: Public URL to an audio file
    
    Returns transcribed text with confidence score and timing info.
    
    **Pricing:** $0.00025/second ($0.015/minute)
    """
    try:
        service = get_stt_service()
        
        if not request.audio_base64 and not request.audio_url:
            raise HTTPException(
                status_code=400,
                detail="Must provide either audio_base64 or audio_url"
            )
        
        result = await service.transcribe(ServiceSTTRequest(
            audio_base64=request.audio_base64,
            audio_url=request.audio_url,
            language=request.language,
            request_id=str(uuid.uuid4()),
        ))
        
        return STTResponse(
            text=result.text,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_ms=round(result.processing_time_ms, 1),
            cost_usd=result.estimated_cost,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")


@router.post("/upload", response_model=STTResponse)
async def speech_to_text_upload(
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Convert uploaded audio file to text.
    
    Accepts audio file upload (mp3, wav, webm, ogg, etc.)
    
    **Pricing:** $0.00025/second ($0.015/minute)
    
    **Note:** This is batch transcription. For real-time streaming,
    use the Live Interview feature which uses AssemblyAI's streaming API.
    """
    try:
        service = get_stt_service()
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        result = await service.transcribe(ServiceSTTRequest(
            audio_bytes=audio_bytes,
            language=language,
            request_id=str(uuid.uuid4()),
        ))
        
        return STTResponse(
            text=result.text,
            confidence=result.confidence,
            duration_seconds=result.duration_seconds,
            processing_time_ms=round(result.processing_time_ms, 1),
            cost_usd=result.estimated_cost,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {str(e)}")


@router.get("/metrics")
async def get_stt_metrics():
    """Get STT service metrics."""
    service = get_stt_service()
    return service.get_metrics()
