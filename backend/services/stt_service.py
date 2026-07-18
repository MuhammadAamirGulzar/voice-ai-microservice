"""
Standalone Speech-to-Text Service

High-performance async STT for one-shot transcription requests.
- No WebRTC overhead
- Connection pooling for API efficiency
- Concurrent request limiting
- Cost tracking per request

Provider: AssemblyAI
"""
import asyncio
import base64
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import aiohttp
from loguru import logger


class STTProvider(str, Enum):
    """Supported STT providers."""
    ASSEMBLYAI = "assemblyai"


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""
    text: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
    language: Optional[str] = None
    processing_time_ms: float = 0
    provider: str = "assemblyai"
    estimated_cost: float = 0


@dataclass
class STTRequest:
    """Request for speech-to-text transcription."""
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    language: str = "en"
    provider: STTProvider = STTProvider.ASSEMBLYAI
    request_id: Optional[str] = None


class STTService:
    """
    High-performance Speech-to-Text service.
    
    Features:
    - Async processing (non-blocking)
    - Connection pooling for API calls
    - Semaphore for concurrent request limiting
    - Cost tracking per request
    
    Usage:
        service = STTService(api_key="...")
        result = await service.transcribe(STTRequest(audio_base64="..."))
        print(result.text)
    """
    
    # Pricing (USD per second of audio)
    PRICING = {
        STTProvider.ASSEMBLYAI: 0.00025,  # $0.015/min
    }
    
    def __init__(
        self,
        assemblyai_api_key: str,
        max_concurrent_requests: int = 20,
        timeout_seconds: float = 60.0,
    ):
        self.assemblyai_api_key = assemblyai_api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.total_requests = 0
        self.total_audio_seconds = 0.0
        self.total_cost = 0.0
        
        logger.info(f"STTService initialized (max_concurrent={max_concurrent_requests})")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=30,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout
            )
        return self._session
    
    async def transcribe(self, request: STTRequest) -> STTResult:
        """
        Transcribe audio to text.
        
        Args:
            request: STTRequest with audio data
            
        Returns:
            STTResult with transcription and metadata
        """
        start_time = time.perf_counter()
        
        if not any([request.audio_base64, request.audio_url, request.audio_bytes]):
            raise ValueError("Must provide audio_base64, audio_url, or audio_bytes")
        
        async with self._semaphore:
            if request.provider == STTProvider.ASSEMBLYAI:
                result = await self._transcribe_assemblyai(request)
            else:
                raise ValueError(f"Unsupported provider: {request.provider}")
        
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Update metrics
        self.total_requests += 1
        if result.duration_seconds:
            self.total_audio_seconds += result.duration_seconds
            self.total_cost += result.estimated_cost
        
        logger.info(
            f"🎤 STT completed: {len(result.text)} chars, "
            f"{result.processing_time_ms:.0f}ms, ${result.estimated_cost:.6f}"
        )
        
        return result
    
    async def _transcribe_assemblyai(self, request: STTRequest) -> STTResult:
        """Transcribe using AssemblyAI API."""
        session = await self._get_session()
        
        headers = {
            "authorization": self.assemblyai_api_key,
            "content-type": "application/json"
        }
        
        # Step 1: Upload audio if raw data provided
        if request.audio_base64 or request.audio_bytes:
            audio_bytes = (
                base64.b64decode(request.audio_base64)
                if request.audio_base64
                else request.audio_bytes
            )
            
            async with session.post(
                "https://api.assemblyai.com/v2/upload",
                headers={"authorization": self.assemblyai_api_key},
                data=audio_bytes
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise RuntimeError(f"AssemblyAI upload failed: {error}")
                upload_result = await resp.json()
                audio_url = upload_result["upload_url"]
        else:
            audio_url = request.audio_url
        
        # Step 2: Create transcription job
        async with session.post(
            "https://api.assemblyai.com/v2/transcript",
            headers=headers,
            json={
                "audio_url": audio_url,
                "language_code": request.language,
            }
        ) as resp:
            if resp.status != 200:
                error = await resp.text()
                raise RuntimeError(f"AssemblyAI transcription failed: {error}")
            transcript_data = await resp.json()
            transcript_id = transcript_data["id"]
        
        # Step 3: Poll for completion
        poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
        while True:
            async with session.get(poll_url, headers=headers) as resp:
                result = await resp.json()
                status = result["status"]
                
                if status == "completed":
                    duration = result.get("audio_duration", 0)
                    return STTResult(
                        text=result["text"] or "",
                        confidence=result.get("confidence"),
                        duration_seconds=duration,
                        language=result.get("language_code"),
                        provider="assemblyai",
                        estimated_cost=duration * self.PRICING[STTProvider.ASSEMBLYAI]
                    )
                elif status == "error":
                    raise RuntimeError(f"AssemblyAI error: {result.get('error')}")
                
                await asyncio.sleep(0.5)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def get_metrics(self) -> dict:
        """Get service metrics."""
        return {
            "total_requests": self.total_requests,
            "total_audio_seconds": round(self.total_audio_seconds, 1),
            "total_cost_usd": round(self.total_cost, 6),
        }
