"""
Standalone Text-to-Speech Service

High-performance async TTS for one-shot synthesis requests.
- No WebRTC overhead
- Response caching (very effective for repeated text)
- Concurrent request limiting
- Multiple voice options
- Cost tracking per request

Provider: Google Cloud TTS (Chirp3 HD voices)
"""
import asyncio
import base64
import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

from loguru import logger


class TTSProvider(str, Enum):
    """Supported TTS providers."""
    GOOGLE = "google"


# Available Chirp3 HD voices
AVAILABLE_VOICES = {
    "Aoede": "Warm, friendly female voice",
    "Charon": "Deep, authoritative male voice",
    "Fenrir": "Energetic male voice",
    "Kore": "Clear, neutral female voice",
    "Puck": "Youthful, bright voice",
    "Zephyr": "Calm, professional voice",
}


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio_base64: str
    audio_format: str = "mp3"
    duration_seconds: Optional[float] = None
    character_count: int = 0
    processing_time_ms: float = 0
    provider: str = "google"
    estimated_cost: float = 0
    from_cache: bool = False
    
    def get_audio_bytes(self) -> bytes:
        """Decode base64 audio to bytes."""
        return base64.b64decode(self.audio_base64)


@dataclass
class TTSRequest:
    """Request for text-to-speech synthesis."""
    text: str
    voice: str = "Kore"
    language: str = "en-US"
    provider: TTSProvider = TTSProvider.GOOGLE
    sample_rate: int = 24000
    audio_format: str = "mp3"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    request_id: Optional[str] = None


class TTSService:
    """
    High-performance Text-to-Speech service.
    
    Features:
    - Async processing (non-blocking)
    - Response caching (highly effective for repeated text)
    - Concurrent request limiting
    - Multiple voice options
    - Cost tracking per request
    
    Usage:
        service = TTSService(credentials_path="...")
        result = await service.synthesize(TTSRequest(text="Hello!"))
        audio_bytes = result.get_audio_bytes()
    """
    
    # Pricing (USD per million characters)
    PRICING = {
        TTSProvider.GOOGLE: {
            "chirp3_hd": 16.0,
            "neural2": 16.0,
            "standard": 4.0,
        }
    }
    
    def __init__(
        self,
        google_credentials_path: Optional[str] = None,
        max_concurrent_requests: int = 20,
        timeout_seconds: float = 30.0,
        enable_cache: bool = True,
        max_cache_size: int = 500,
    ):
        self.google_credentials_path = google_credentials_path
        self.timeout = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._google_client = None
        self._enable_cache = enable_cache
        self._max_cache_size = max_cache_size
        self._cache: dict = {}
        
        # Set credentials env var if provided
        if google_credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_credentials_path
        
        # Metrics
        self.total_requests = 0
        self.total_characters = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        
        logger.info(
            f"TTSService initialized (cache={'enabled' if enable_cache else 'disabled'}, "
            f"max_concurrent={max_concurrent_requests})"
        )
    
    async def _get_google_client(self):
        """Get or create Google TTS client."""
        if self._google_client is None:
            from google.cloud import texttospeech_v1 as texttospeech
            self._google_client = texttospeech.TextToSpeechAsyncClient()
        return self._google_client
    
    async def synthesize(self, request: TTSRequest) -> TTSResult:
        """
        Synthesize text to speech.
        
        Args:
            request: TTSRequest with text and options
            
        Returns:
            TTSResult with audio data and metadata
        """
        start_time = time.perf_counter()
        
        if not request.text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache
        if self._enable_cache:
            cache_key = self._get_cache_key(request)
            if cache_key in self._cache:
                self.cache_hits += 1
                cached = self._cache[cache_key]
                cached.processing_time_ms = (time.perf_counter() - start_time) * 1000
                cached.from_cache = True
                logger.debug(f"🔊 TTS cache hit: {len(request.text)} chars")
                return cached
        
        # Synthesize
        async with self._semaphore:
            if request.provider == TTSProvider.GOOGLE:
                result = await self._synthesize_google(request)
            else:
                raise ValueError(f"Unsupported provider: {request.provider}")
        
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update metrics
        self.total_requests += 1
        self.total_characters += result.character_count
        self.total_cost += result.estimated_cost
        
        # Cache result
        if self._enable_cache:
            self._add_to_cache(cache_key, result)
        
        logger.info(
            f"🔊 TTS completed: {result.character_count} chars, "
            f"{result.processing_time_ms:.0f}ms, ${result.estimated_cost:.6f}"
        )
        
        return result
    
    async def synthesize_stream(
        self,
        request: TTSRequest,
        chunk_size: int = 4096
    ) -> AsyncIterator[bytes]:
        """
        Synthesize and stream audio in chunks.
        
        Yields:
            Audio chunks as bytes
        """
        result = await self.synthesize(request)
        audio_bytes = result.get_audio_bytes()
        
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]
    
    async def _synthesize_google(self, request: TTSRequest) -> TTSResult:
        """Synthesize using Google Cloud TTS API."""
        from google.cloud import texttospeech_v1 as texttospeech
        
        client = await self._get_google_client()
        
        # Build synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=request.text)
        
        # Voice selection - Chirp3 HD voices
        voice = texttospeech.VoiceSelectionParams(
            language_code=request.language,
            name=f"{request.language}-Chirp3-HD-{request.voice}"
        )
        
        # Audio config
        audio_encoding = {
            "mp3": texttospeech.AudioEncoding.MP3,
            "wav": texttospeech.AudioEncoding.LINEAR16,
            "ogg": texttospeech.AudioEncoding.OGG_OPUS,
        }.get(request.audio_format, texttospeech.AudioEncoding.MP3)
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding,
            sample_rate_hertz=request.sample_rate,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
        )
        
        # API call
        response = await client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        
        # Calculate cost
        char_count = len(request.text)
        cost = (char_count / 1_000_000) * self.PRICING[TTSProvider.GOOGLE]["chirp3_hd"]
        
        # Estimate duration (~15 chars/second for natural speech at 1.0 rate)
        # This accounts for pauses, pacing, and natural speech rhythm
        estimated_duration = (char_count / 15.0) / request.speaking_rate
        
        return TTSResult(
            audio_base64=base64.b64encode(response.audio_content).decode(),
            audio_format=request.audio_format,
            duration_seconds=round(estimated_duration, 1),
            character_count=char_count,
            provider="google",
            estimated_cost=cost,
        )
    
    def _get_cache_key(self, request: TTSRequest) -> str:
        """Generate cache key from request parameters."""
        key_parts = f"{request.text}|{request.voice}|{request.language}|{request.speaking_rate}|{request.pitch}"
        return hashlib.md5(key_parts.encode()).hexdigest()
    
    def _add_to_cache(self, key: str, result: TTSResult):
        """Add to cache with size limit (FIFO eviction)."""
        if len(self._cache) >= self._max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = result
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        logger.info("TTS cache cleared")
    
    async def close(self):
        """Close the client."""
        pass  # Google client doesn't need explicit close
    
    def get_metrics(self) -> dict:
        """Get service metrics."""
        total_with_cache = self.total_requests + self.cache_hits
        return {
            "total_requests": self.total_requests,
            "total_characters": self.total_characters,
            "total_cost_usd": round(self.total_cost, 6),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": round(self.cache_hits / max(1, total_with_cache), 3),
            "cache_size": len(self._cache),
        }
    
    @staticmethod
    def get_available_voices() -> dict:
        """Get list of available voices."""
        return AVAILABLE_VOICES.copy()
