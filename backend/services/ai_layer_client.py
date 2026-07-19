"""
AI Layer Client Service - Communicates with external AI Layer microservice.

Handles:
- HTTP requests to AI Layer endpoints
- Request/response transformation
- Retry logic and error handling
- Streaming responses (if supported)
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, AsyncGenerator
from enum import Enum

import httpx
from loguru import logger

from config.settings import get_settings


class AILayerError(Exception):
    """Base exception for AI Layer errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class AILayerConnectionError(AILayerError):
    """Connection to AI Layer failed."""
    pass


class AILayerTimeoutError(AILayerError):
    """AI Layer request timed out."""
    pass


class AILayerResponseError(AILayerError):
    """AI Layer returned an error response."""
    pass


@dataclass
class AILayerRequest:
    """Request to AI Layer."""
    # Multi-tenant identifiers
    organisation_id: str
    agent_id: str
    user_id: str
    session_id: str
    
    # Message content
    message: str
    
    # Optional context
    conversation_history: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "organisation_id": self.organisation_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "message": self.message,
            "conversation_history": self.conversation_history or [],
            "metadata": self.metadata or {},
        }


@dataclass
class AILayerResponse:
    """Response from AI Layer."""
    text: str
    session_id: str
    
    # Optional response metadata
    confidence: Optional[float] = None
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[list] = None
    
    # Performance metrics
    processing_time_ms: float = 0
    
    # Raw response for debugging
    raw_response: Optional[Dict] = None


class AILayerClient:
    """
    Async HTTP client for AI Layer microservice.
    
    Features:
    - Connection pooling
    - Automatic retries with exponential backoff
    - Request/response logging
    - Timeout handling
    """
    
    _instance: Optional["AILayerClient"] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        if self._initialized:
            return
        
        settings = get_settings()
        
        self.base_url = (base_url or settings.ai_layer_base_url).rstrip("/")
        self.chat_endpoint = settings.ai_layer_chat_endpoint
        self.timeout_seconds = timeout_seconds or settings.ai_layer_timeout_seconds
        self.max_retries = max_retries or settings.ai_layer_retry_attempts
        self.api_key = api_key or settings.ai_layer_api_key
        
        # HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0
        
        self._initialized = True
        
        logger.info(
            f"AILayerClient initialized | "
            f"base_url={self.base_url} | "
            f"timeout={self.timeout_seconds}s | "
            f"max_retries={self.max_retries}"
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Service": "voice-microservice",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def send_message(
        self,
        request: AILayerRequest,
        endpoint: Optional[str] = None,
        timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> AILayerResponse:
        """
        Send a message to AI Layer and get response.

        Args:
            request: AILayerRequest with message and context
            endpoint: Optional custom endpoint (defaults to chat_endpoint)
            timeout_s: Per-call timeout override. Live voice turns pass a
                short budget here; the client default suits batch/text calls.
            max_retries: Per-call retry override (voice turns use 1 — a
                caller won't wait through exponential backoff).

        Returns:
            AILayerResponse with AI-generated text

        Raises:
            AILayerConnectionError: If connection fails
            AILayerTimeoutError: If request times out
            AILayerResponseError: If AI Layer returns an error
        """
        endpoint = endpoint or self.chat_endpoint
        url = f"{self.base_url}{endpoint}"
        retries = max_retries or self.max_retries
        effective_timeout = timeout_s or self.timeout_seconds

        start_time = time.time()
        self._request_count += 1

        logger.debug(
            f"AI Layer request | session={request.session_id} | "
            f"org={request.organisation_id} | message_len={len(request.message)}"
        )

        last_error = None
        for attempt in range(retries):
            try:
                client = await self._get_client()

                response = await client.post(
                    endpoint,
                    json=request.to_dict(),
                    headers=self._get_headers(),
                    timeout=effective_timeout,
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                self._total_latency_ms += elapsed_ms
                
                # Check for errors
                if response.status_code >= 400:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("detail", error_detail)
                    except:
                        pass
                    
                    raise AILayerResponseError(
                        f"AI Layer error: {error_detail}",
                        status_code=response.status_code,
                        details={"response": response.text}
                    )
                
                # Parse response
                data = response.json()
                
                ai_response = AILayerResponse(
                    text=data.get("text", data.get("response", data.get("message", ""))),
                    session_id=request.session_id,
                    confidence=data.get("confidence"),
                    intent=data.get("intent"),
                    entities=data.get("entities"),
                    suggested_actions=data.get("suggested_actions"),
                    processing_time_ms=elapsed_ms,
                    raw_response=data,
                )
                
                logger.debug(
                    f"AI Layer response | session={request.session_id} | "
                    f"response_len={len(ai_response.text)} | latency={elapsed_ms:.0f}ms"
                )
                
                return ai_response
                
            except httpx.ConnectError as e:
                last_error = AILayerConnectionError(
                    f"Failed to connect to AI Layer: {e}",
                    details={"url": url, "attempt": attempt + 1}
                )
                logger.warning(f"AI Layer connection failed (attempt {attempt + 1}): {e}")
                
            except httpx.TimeoutException as e:
                last_error = AILayerTimeoutError(
                    f"AI Layer request timed out after {effective_timeout}s",
                    details={"url": url, "attempt": attempt + 1}
                )
                logger.warning(f"AI Layer timeout (attempt {attempt + 1}): {e}")

            except AILayerResponseError:
                raise

            except asyncio.CancelledError:
                raise

            except Exception as e:
                last_error = AILayerError(
                    f"Unexpected AI Layer error: {e}",
                    details={"url": url, "attempt": attempt + 1}
                )
                logger.error(f"AI Layer unexpected error (attempt {attempt + 1}): {e}")

            # Exponential backoff before retry
            if attempt < retries - 1:
                backoff = min(2 ** attempt, 10)  # Max 10 seconds
                await asyncio.sleep(backoff)
        
        # All retries failed
        self._error_count += 1
        raise last_error
    
    async def send_message_stream(
        self,
        request: AILayerRequest,
        endpoint: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        
        Yields text chunks as they arrive.
        Useful for real-time TTS where we want to start speaking
        before the full response is ready.
        """
        endpoint = endpoint or f"{self.chat_endpoint}/stream"
        
        logger.debug(f"AI Layer stream request | session={request.session_id}")
        
        try:
            client = await self._get_client()
            
            async with client.stream(
                "POST",
                endpoint,
                json=request.to_dict(),
                headers=self._get_headers(),
            ) as response:
                if response.status_code >= 400:
                    raise AILayerResponseError(
                        f"AI Layer stream error",
                        status_code=response.status_code,
                    )
                
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        yield chunk
                        
        except httpx.ConnectError as e:
            raise AILayerConnectionError(f"Stream connection failed: {e}")
        except httpx.TimeoutException as e:
            raise AILayerTimeoutError(f"Stream timed out: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check AI Layer health."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "url": self.base_url,
                    "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else {},
                }
            else:
                return {
                    "status": "unhealthy",
                    "url": self.base_url,
                    "status_code": response.status_code,
                }
        except Exception as e:
            return {
                "status": "unreachable",
                "url": self.base_url,
                "error": str(e),
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_latency = (
            self._total_latency_ms / self._request_count 
            if self._request_count > 0 else 0
        )
        
        return {
            "base_url": self.base_url,
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "error_rate": (
                self._error_count / self._request_count 
                if self._request_count > 0 else 0
            ),
            "average_latency_ms": round(avg_latency, 2),
        }
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("AILayerClient closed")


# Singleton accessor
_ai_layer_client: Optional[AILayerClient] = None


def get_ai_layer_client() -> AILayerClient:
    """Get the global AI Layer client instance."""
    global _ai_layer_client
    if _ai_layer_client is None:
        _ai_layer_client = AILayerClient()
    return _ai_layer_client


async def init_ai_layer_client(
    base_url: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    max_retries: Optional[int] = None,
    api_key: Optional[str] = None,
) -> AILayerClient:
    """Initialize the AI Layer client with custom settings."""
    global _ai_layer_client
    _ai_layer_client = AILayerClient.__new__(AILayerClient)
    _ai_layer_client._initialized = False
    _ai_layer_client.__init__(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        api_key=api_key,
    )
    return _ai_layer_client
