"""
Configuration settings for Voice Microservice.
Multi-tenant voice service supporting STT, TTS, and Interactive sessions.
Loads settings from environment variables and .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from typing import Optional
import os

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class SessionMode(str, Enum):
    """Voice session modes"""
    STT_ONLY = "stt_only"           # Speech-to-Text only
    TTS_ONLY = "tts_only"           # Text-to-Speech only
    INTERACTIVE = "interactive"      # Full conversation: STT → AI Layer → TTS


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # Server Settings
    # ==========================================================================
    host: str = Field(default="127.0.0.1", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # ==========================================================================
    # Service Identity
    # ==========================================================================
    service_name: str = Field(default="voice-microservice", alias="SERVICE_NAME")
    service_version: str = Field(default="3.0.0", alias="SERVICE_VERSION")
    
    # ==========================================================================
    # AI Layer Configuration (External Microservice)
    # ==========================================================================
    ai_layer_base_url: str = Field(
        default="http://localhost:8001", 
        alias="AI_LAYER_BASE_URL",
        description="Base URL for the AI Layer microservice"
    )
    ai_layer_chat_endpoint: str = Field(
        default="/api/v1/chat", 
        alias="AI_LAYER_CHAT_ENDPOINT",
        description="Endpoint for chat/conversation requests"
    )
    ai_layer_timeout_seconds: int = Field(
        default=30, 
        alias="AI_LAYER_TIMEOUT_SECONDS",
        description="Timeout for AI Layer requests"
    )
    ai_layer_retry_attempts: int = Field(
        default=3, 
        alias="AI_LAYER_RETRY_ATTEMPTS",
        description="Number of retry attempts for failed AI Layer requests"
    )
    ai_layer_api_key: str = Field(
        default="", 
        alias="AI_LAYER_API_KEY",
        description="API key for AI Layer authentication (if required)"
    )
    
    # ==========================================================================
    # API Keys for Voice Services
    # ==========================================================================
    assemblyai_api_key: str = Field(default="", alias="ASSEMBLYAI_API_KEY")
    google_cloud_tts_credentials: str = Field(default="google-credentials.json", alias="GOOGLE_CLOUD_TTS_CREDENTIALS")
    
    # Legacy LLM keys (for fallback/testing without AI layer)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    
    # ==========================================================================
    # LLM Configuration (Fallback - when AI Layer unavailable)
    # ==========================================================================
    use_local_llm: bool = Field(
        default=False, 
        alias="USE_LOCAL_LLM",
        description="Use local LLM instead of AI Layer (for testing)"
    )
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.7, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=300, alias="LLM_MAX_TOKENS")
    
    # ==========================================================================
    # TTS Configuration
    # ==========================================================================
    tts_voice_name: str = Field(default="en-US-Chirp3-HD-Zephyr", alias="TTS_VOICE_NAME")
    tts_language_code: str = Field(default="en-US", alias="TTS_LANGUAGE_CODE")
    tts_speaking_rate: float = Field(default=1.0, alias="TTS_SPEAKING_RATE")
    tts_pitch: float = Field(default=0.0, alias="TTS_PITCH")
    
    # ==========================================================================
    # STT Configuration
    # ==========================================================================
    stt_language: str = Field(default="en", alias="STT_LANGUAGE")
    stt_sample_rate: int = Field(default=16000, alias="STT_SAMPLE_RATE")
    
    # ==========================================================================
    # Session Configuration
    # ==========================================================================
    max_session_duration_minutes: int = Field(default=30, alias="MAX_SESSION_DURATION_MINUTES")
    default_session_mode: str = Field(default="interactive", alias="DEFAULT_SESSION_MODE")
    enable_session_recording: bool = Field(default=False, alias="ENABLE_SESSION_RECORDING")
    
    # ==========================================================================
    # Storage & Logging
    # ==========================================================================
    transcript_dir: str = Field(default="logs/transcripts", alias="TRANSCRIPT_DIR")
    recordings_dir: str = Field(default="logs/recordings", alias="RECORDINGS_DIR")
    enable_detailed_logging: bool = Field(default=True, alias="ENABLE_DETAILED_LOGGING")
    
    # ==========================================================================
    # Rate Limiting & Security
    # ==========================================================================
    rate_limit_requests_per_minute: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    enable_request_validation: bool = Field(default=True, alias="ENABLE_REQUEST_VALIDATION")

    class Config:
        env_file = ".env"
        extra = "allow"
        populate_by_name = True
    
    def get_ai_layer_url(self, endpoint: Optional[str] = None) -> str:
        """Get full AI Layer URL with optional custom endpoint."""
        base = self.ai_layer_base_url.rstrip("/")
        ep = endpoint or self.ai_layer_chat_endpoint
        return f"{base}{ep}"


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_session_mode(mode_str: str) -> SessionMode:
    """Convert string to SessionMode enum safely."""
    try:
        return SessionMode(mode_str.lower())
    except ValueError:
        return SessionMode.INTERACTIVE


# =============================================================================
# Legacy functions (for backward compatibility - will be removed)
# These functions are now deprecated. Use AI Layer for content.
# =============================================================================

def load_custom_instructions() -> str:
    """
    DEPRECATED: Custom instructions should come from AI Layer.
    This is a fallback for testing without AI Layer.
    """
    return """You are a professional AI assistant conducting a voice conversation. 
Be warm, professional, and conversational. Keep responses concise for voice.
IMPORTANT: Never prefix your responses with your name or role - just speak naturally.
"""


def load_questions() -> list:
    """
    DEPRECATED: Questions should come from AI Layer based on business case.
    This is a fallback for testing without AI Layer.
    """
    return [
        {"type": "greeting", "text": "Hello! How can I help you today?", "wait_for_response": True},
    ]


def load_sample_resume() -> dict:
    """
    DEPRECATED: Candidate data should come from AI Layer.
    This is a fallback for testing without AI Layer.
    """
    return {
        "name": "Test User",
        "email": "test@example.com",
        "skills": ["Communication"],
        "experience": []
    }
