"""
Session Manager - Multi-tenant session management for voice microservice.

Handles:
- Session lifecycle (create, update, close)
- Multi-tenant context (org_id, agent_id, user_id)
- Transcript and recording storage
- Session state management
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List
import json

from loguru import logger


class SessionStatus(str, Enum):
    """Session lifecycle states"""
    INITIALIZED = "initialized"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


class SessionMode(str, Enum):
    """Voice session modes"""
    STT_ONLY = "stt_only"
    TTS_ONLY = "tts_only"
    INTERACTIVE = "interactive"


@dataclass
class SessionContext:
    """
    Multi-tenant session context.
    Contains all identifiers needed for AI Layer communication and transcript storage.
    """
    # Required identifiers
    session_id: str
    
    # Multi-tenant context (from frontend)
    organisation_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Session configuration
    mode: SessionMode = SessionMode.INTERACTIVE
    language: str = "en-US"
    voice: str = "Zephyr"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    client_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for API calls."""
        return {
            "session_id": self.session_id,
            "organisation_id": self.organisation_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "mode": self.mode.value if isinstance(self.mode, SessionMode) else self.mode,
            "language": self.language,
            "voice": self.voice,
            "created_at": self.created_at.isoformat(),
            "client_metadata": self.client_metadata,
        }
    
    def get_ai_layer_payload(self) -> Dict[str, Any]:
        """Get payload for AI Layer requests."""
        return {
            "organisation_id": self.organisation_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class Session:
    """
    Voice session with full state management.
    """
    context: SessionContext
    status: SessionStatus = SessionStatus.INITIALIZED
    
    # Pipeline components (set during connection)
    task: Optional[asyncio.Task] = None
    runner: Optional[Any] = None
    metrics_processor: Optional[Any] = None
    transcript_logger: Optional[Any] = None
    
    # Timing
    started_at: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Statistics
    message_count: int = 0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    
    @property
    def session_id(self) -> str:
        return self.context.session_id
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self.started_at:
            return 0.0
        end = self.closed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def is_active(self) -> bool:
        """Check if session is in an active state."""
        return self.status in (SessionStatus.CONNECTED, SessionStatus.ACTIVE)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
        self.message_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "context": self.context.to_dict(),
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "is_active": self.is_active,
        }


class SessionManager:
    """
    Manages all active voice sessions.
    Thread-safe singleton pattern for global session access.
    """
    
    _instance: Optional["SessionManager"] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._sessions: Dict[str, Session] = {}
        self._org_sessions: Dict[str, List[str]] = {}  # org_id -> [session_ids]
        self._user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
        self._initialized = True
        
        logger.info("SessionManager initialized")
    
    # =========================================================================
    # Session Lifecycle
    # =========================================================================
    
    def create_session(
        self,
        organisation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        mode: str = "interactive",
        language: str = "en-US",
        voice: str = "Zephyr",
        client_metadata: Optional[Dict[str, Any]] = None,
    ) -> Session:
        """
        Create a new voice session.
        
        Args:
            organisation_id: Organization identifier (from frontend)
            agent_id: Agent identifier (from frontend)
            user_id: User identifier (from frontend)
            mode: Session mode (stt_only, tts_only, interactive)
            language: Language code for STT/TTS
            voice: TTS voice name
            client_metadata: Additional client-provided metadata
            
        Returns:
            Created Session object
        """
        # Generate session ID
        session_id = self._generate_session_id()
        
        # Parse mode
        try:
            session_mode = SessionMode(mode.lower())
        except ValueError:
            session_mode = SessionMode.INTERACTIVE
        
        # Create context
        context = SessionContext(
            session_id=session_id,
            organisation_id=organisation_id,
            agent_id=agent_id,
            user_id=user_id,
            mode=session_mode,
            language=language,
            voice=voice,
            client_metadata=client_metadata or {},
        )
        
        # Create session
        session = Session(
            context=context,
            status=SessionStatus.INITIALIZED,
            started_at=datetime.now(),
        )
        
        # Store session
        self._sessions[session_id] = session
        
        # Track by org/user for easy lookup
        if organisation_id:
            if organisation_id not in self._org_sessions:
                self._org_sessions[organisation_id] = []
            self._org_sessions[organisation_id].append(session_id)
        
        if user_id:
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = []
            self._user_sessions[user_id].append(session_id)
        
        logger.info(
            f"Session created: {session_id} | "
            f"org={organisation_id} | agent={agent_id} | user={user_id} | "
            f"mode={session_mode.value}"
        )
        
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        old_status = session.status
        session.status = status
        
        # Update timestamps
        if status == SessionStatus.CONNECTED:
            session.connected_at = datetime.now()
        elif status in (SessionStatus.CLOSED, SessionStatus.ERROR):
            session.closed_at = datetime.now()
        
        logger.debug(f"Session {session_id}: {old_status.value} -> {status.value}")
        return True
    
    async def close_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Close a session and cleanup resources. Idempotent: the stop
        endpoint, the WebRTC disconnect handler, the reaper and shutdown
        can all race here — only the first caller does the teardown.

        Returns session summary with metrics.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        if session.status in (SessionStatus.CLOSING, SessionStatus.CLOSED):
            return session.to_dict()

        logger.info(f"Closing session: {session_id}")

        # Update status
        session.status = SessionStatus.CLOSING
        session.closed_at = datetime.now()
        
        # Cancel pipeline task
        if session.task and not session.task.done():
            session.task.cancel()
            try:
                await session.task
            except asyncio.CancelledError:
                pass
        
        # Get metrics
        metrics = None
        if session.metrics_processor:
            try:
                metrics = session.metrics_processor.get_summary()
            except Exception as e:
                logger.warning(f"Failed to get metrics: {e}")
        
        # Finalize transcript
        if session.transcript_logger:
            try:
                session.transcript_logger.finalize(
                    summary="Session completed",
                    metrics=metrics
                )
            except Exception as e:
                logger.warning(f"Failed to finalize transcript: {e}")
        
        # Build summary
        summary = session.to_dict()
        summary["metrics"] = metrics
        
        # Update status and cleanup
        session.status = SessionStatus.CLOSED

        # Remove from tracking (pop, not del — a racing closer may have won)
        self._cleanup_session_tracking(session)
        self._sessions.pop(session_id, None)
        
        logger.info(
            f"Session closed: {session_id} | "
            f"duration={summary['duration_seconds']}s | "
            f"messages={session.message_count}"
        )
        
        return summary
    
    def _cleanup_session_tracking(self, session: Session):
        """Remove session from org/user tracking."""
        session_id = session.session_id
        org_id = session.context.organisation_id
        user_id = session.context.user_id
        
        if org_id and org_id in self._org_sessions:
            if session_id in self._org_sessions[org_id]:
                self._org_sessions[org_id].remove(session_id)
            if not self._org_sessions[org_id]:
                del self._org_sessions[org_id]
        
        if user_id and user_id in self._user_sessions:
            if session_id in self._user_sessions[user_id]:
                self._user_sessions[user_id].remove(session_id)
            if not self._user_sessions[user_id]:
                del self._user_sessions[user_id]
    
    async def cleanup_stale_sessions(
        self,
        max_pending_minutes: float = 10.0,
        max_duration_minutes: float = 30.0,
    ) -> int:
        """
        Reap leaked sessions. Two failure modes both grow memory and cost
        money unbounded without this:
        - Sessions created via /start whose client never connected.
        - Live sessions past the max duration (a wedged WebRTC peer keeps
          STT/TTS billing running forever).

        Returns the number of sessions closed.
        """
        now = datetime.now()
        to_close: List[str] = []

        for session in list(self._sessions.values()):
            age_minutes = (now - session.context.created_at).total_seconds() / 60.0
            if session.status in (SessionStatus.INITIALIZED, SessionStatus.CONNECTING):
                if age_minutes > max_pending_minutes:
                    logger.warning(
                        f"Reaping never-connected session {session.session_id} "
                        f"(pending {age_minutes:.0f} min)")
                    to_close.append(session.session_id)
            elif session.is_active and age_minutes > max_duration_minutes:
                logger.warning(
                    f"Reaping over-duration session {session.session_id} "
                    f"({age_minutes:.0f} min > {max_duration_minutes:.0f} min limit)")
                to_close.append(session.session_id)

        for session_id in to_close:
            try:
                await self.close_session(session_id)
            except Exception as e:
                logger.error(f"Failed to reap session {session_id}: {e}")

        return len(to_close)

    # =========================================================================
    # Session Queries
    # =========================================================================
    
    def get_sessions_by_org(self, organisation_id: str) -> List[Session]:
        """Get all sessions for an organization."""
        session_ids = self._org_sessions.get(organisation_id, [])
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]
    
    def get_sessions_by_user(self, user_id: str) -> List[Session]:
        """Get all sessions for a user."""
        session_ids = self._user_sessions.get(user_id, [])
        return [self._sessions[sid] for sid in session_ids if sid in self._sessions]
    
    def get_active_sessions(self) -> List[Session]:
        """Get all active sessions."""
        return [s for s in self._sessions.values() if s.is_active]
    
    def list_sessions(
        self,
        organisation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100,
    ) -> List[Session]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        
        if organisation_id:
            sessions = [s for s in sessions if s.context.organisation_id == organisation_id]
        
        if user_id:
            sessions = [s for s in sessions if s.context.user_id == user_id]
        
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # Sort by created time (newest first)
        sessions.sort(key=lambda s: s.context.created_at, reverse=True)
        
        return sessions[:limit]
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        sessions = list(self._sessions.values())
        active_sessions = [s for s in sessions if s.is_active]
        
        return {
            "total_sessions": len(sessions),
            "active_sessions": len(active_sessions),
            "organizations": len(self._org_sessions),
            "users": len(self._user_sessions),
            "sessions_by_status": {
                status.value: len([s for s in sessions if s.status == status])
                for status in SessionStatus
            },
            "sessions_by_mode": {
                mode.value: len([s for s in sessions if s.context.mode == mode])
                for mode in SessionMode
            },
        }
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        unique_id = uuid.uuid4().hex[:12]
        return f"session_{unique_id}"


# Global session manager instance
def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    return SessionManager()
