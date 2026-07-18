"""
Services Module

Provides standalone service classes for:
- STT (Speech-to-Text) via AssemblyAI
- TTS (Text-to-Speech) via Google Cloud TTS
- LLM integration via OpenAI/Groq
"""

from backend.services.stt_service import STTService, STTRequest, STTResult
from backend.services.tts_service import TTSService, TTSRequest, TTSResult, AVAILABLE_VOICES

__all__ = [
    # STT
    "STTService",
    "STTRequest", 
    "STTResult",
    # TTS
    "TTSService",
    "TTSRequest",
    "TTSResult",
    "AVAILABLE_VOICES",
]
