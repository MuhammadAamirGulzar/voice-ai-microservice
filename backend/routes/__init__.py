"""
Routes Package - API endpoint modules

Organized by service:
- stt_routes: Speech-to-Text endpoints
- tts_routes: Text-to-Speech endpoints  
- conversation_routes: Real-time WebRTC conversation endpoints
"""

from backend.routes.stt_routes import router as stt_router
from backend.routes.tts_routes import router as tts_router
from backend.routes.conversation_routes import router as conversation_router

__all__ = ["stt_router", "tts_router", "conversation_router"]
