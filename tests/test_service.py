"""
Unit tests for the voice microservice — no network, no WebRTC, no pipecat.

Run:  python -m pytest tests/ -q

Covers the non-pipeline surface: settings, session lifecycle (including the
stale-session reaper), transcript path safety, TTS cache bounds, and the
HTTP surface (auth middleware, rate limiting, session start/stop, error
status codes) via FastAPI's TestClient.
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings, get_session_mode, SessionMode as CfgSessionMode
from backend.processors.transcript_logger import TranscriptLogger, safe_id
from backend.services.session_manager import (
    SessionManager,
    SessionStatus,
    get_session_manager,
)
from backend.services.stt_service import STTService, STTRequest
from backend.services.tts_service import TTSService, TTSResult, TTSRequest


# ─────────────────────────────────────────────────────────────────────────
# settings
# ─────────────────────────────────────────────────────────────────────────
def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.ai_layer_voice_timeout_seconds < s.ai_layer_timeout_seconds
    assert s.rate_limit_enabled is False
    assert s.api_key == ""
    assert s.max_pending_session_minutes > 0


def test_session_mode_parse_is_safe():
    assert get_session_mode("interactive") == CfgSessionMode.INTERACTIVE
    assert get_session_mode("STT_ONLY") == CfgSessionMode.STT_ONLY
    assert get_session_mode("bogus") == CfgSessionMode.INTERACTIVE


# ─────────────────────────────────────────────────────────────────────────
# transcript path safety
# ─────────────────────────────────────────────────────────────────────────
def test_safe_id_sanitizes_traversal():
    assert safe_id("../../etc/passwd") == "etc_passwd"
    assert ".." not in safe_id("..\\..\\windows")
    assert "/" not in safe_id("a/b/c") and "\\" not in safe_id("a\\b")
    assert safe_id("") == "unknown"
    assert safe_id(None) == "unknown"
    assert safe_id("org_123-ok") == "org_123-ok"


def test_transcript_logger_confines_hostile_org_id(tmp_path):
    hostile = "../../outside"
    tlog = TranscriptLogger(
        session_id="../sess/../../x",
        output_dir=str(tmp_path),
        organisation_id=hostile,
    )
    tlog.add_message("User", "hello there")
    tlog.finalize(summary="done")

    written = list(tmp_path.rglob("*.json"))
    assert len(written) == 1
    # The file must be inside the transcript root, not above it.
    assert tmp_path in written[0].parents
    assert (tmp_path.parent / "outside").exists() is False


def test_transcript_logger_counts_messages(tmp_path):
    tlog = TranscriptLogger(session_id="s1", output_dir=str(tmp_path))
    tlog.add_message("User", "hi")
    tlog.add_message("Assistant", "hello")
    tlog.add_message("Assistant", "   ")   # blank: ignored
    meta = tlog.transcript["metadata"]
    assert meta["user_messages"] == 1
    assert meta["assistant_messages"] == 1
    assert meta["exchanges_count"] == 2


# ─────────────────────────────────────────────────────────────────────────
# session manager
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def manager():
    m = get_session_manager()
    # isolate: clear anything previous tests left behind
    m._sessions.clear()
    m._org_sessions.clear()
    m._user_sessions.clear()
    return m


def test_session_lifecycle_and_idempotent_close(manager):
    async def scenario():
        session = manager.create_session(organisation_id="org1", user_id="u1")
        sid = session.session_id
        assert manager.get_session(sid) is session
        assert manager.get_sessions_by_org("org1")

        first = await manager.close_session(sid)
        second = await manager.close_session(sid)   # racing closer: no KeyError
        return first, second, sid

    first, second, sid = asyncio.run(scenario())
    assert first is not None
    assert second is None or second.get("session_id") == sid
    assert manager.get_session(sid) is None
    assert manager.get_sessions_by_org("org1") == []


def test_reaper_closes_pending_and_overdue_sessions(manager):
    async def scenario():
        pending = manager.create_session(organisation_id="org1")
        pending.context.created_at = datetime.now() - timedelta(minutes=20)

        overdue = manager.create_session(organisation_id="org1")
        overdue.status = SessionStatus.CONNECTED
        overdue.context.created_at = datetime.now() - timedelta(minutes=45)

        fresh = manager.create_session(organisation_id="org1")

        reaped = await manager.cleanup_stale_sessions(
            max_pending_minutes=10, max_duration_minutes=30)
        return reaped, pending, overdue, fresh

    reaped, pending, overdue, fresh = asyncio.run(scenario())
    assert reaped == 2
    assert manager.get_session(pending.session_id) is None
    assert manager.get_session(overdue.session_id) is None
    assert manager.get_session(fresh.session_id) is not None


def test_session_stats_shape(manager):
    manager.create_session(organisation_id="orgA", user_id="u1")
    stats = manager.get_stats()
    assert stats["total_sessions"] == 1
    assert stats["organizations"] == 1
    assert "sessions_by_status" in stats and "sessions_by_mode" in stats


# ─────────────────────────────────────────────────────────────────────────
# TTS cache bounds
# ─────────────────────────────────────────────────────────────────────────
def _fake_result(size: int) -> TTSResult:
    return TTSResult(audio_base64="x" * size, character_count=size)


def test_tts_cache_respects_byte_cap():
    service = TTSService(max_cache_size=100, max_cache_bytes=1000)
    for i in range(5):
        service._add_to_cache(f"k{i}", _fake_result(300))
    # 5 x 300 = 1500 bytes > 1000 cap: oldest entries must have been evicted
    assert service._cache_bytes <= 1000
    assert "k0" not in service._cache
    assert "k4" in service._cache


def test_tts_cache_rejects_oversized_entry():
    service = TTSService(max_cache_bytes=100)
    service._add_to_cache("big", _fake_result(500))
    assert "big" not in service._cache
    assert service._cache_bytes == 0


def test_tts_cache_key_includes_voice_params():
    service = TTSService()
    a = service._get_cache_key(TTSRequest(text="hi", voice="Kore"))
    b = service._get_cache_key(TTSRequest(text="hi", voice="Puck"))
    c = service._get_cache_key(TTSRequest(text="hi", voice="Kore"))
    assert a != b and a == c


# ─────────────────────────────────────────────────────────────────────────
# STT service validation
# ─────────────────────────────────────────────────────────────────────────
def test_stt_requires_audio_input():
    service = STTService(assemblyai_api_key="test")
    with pytest.raises(ValueError):
        asyncio.run(service.transcribe(STTRequest()))


# ─────────────────────────────────────────────────────────────────────────
# HTTP surface (TestClient — boots the real app, no WebRTC)
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "sessions" in body


def test_api_key_middleware_blocks_and_allows(client):
    from backend.main import settings
    original = settings.api_key
    try:
        settings.api_key = "secret-key"
        assert client.get("/api/config").status_code == 401
        assert client.get("/api/config",
                          headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/api/config",
                          headers={"X-API-Key": "secret-key"}).status_code == 200
        # health stays open for load balancers
        assert client.get("/health").status_code == 200
    finally:
        settings.api_key = original


def test_rate_limiter_enforces_and_recovers(client):
    from backend.main import settings, _rate_windows
    original = (settings.rate_limit_enabled, settings.rate_limit_requests_per_minute)
    try:
        _rate_windows.clear()
        settings.rate_limit_enabled = True
        settings.rate_limit_requests_per_minute = 3
        codes = [client.get("/api/config").status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert 429 in codes[3:]
    finally:
        settings.rate_limit_enabled, settings.rate_limit_requests_per_minute = original
        _rate_windows.clear()


def test_session_start_status_stop_flow(client):
    r = client.post("/api/v1/voice/session/start", json={
        "organisation_id": "org_t", "agent_id": "agent_t",
        "user_id": "user_t", "mode": "interactive",
    })
    assert r.status_code == 200
    sid = r.json()["session_id"]

    r = client.get(f"/api/v1/voice/session/{sid}/status")
    assert r.status_code == 200
    assert r.json()["organisation_id"] == "org_t"

    r = client.get(f"/api/v1/voice/session/{sid}/transcript")
    assert r.status_code == 200 and r.json()["count"] == 0

    r = client.post(f"/api/v1/voice/session/{sid}/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"

    # stopped session is gone
    assert client.get(f"/api/v1/voice/session/{sid}/status").status_code == 404


def test_stt_route_503_when_service_unconfigured(client):
    import backend.routes.stt_routes as stt_routes
    original = stt_routes._stt_service
    try:
        stt_routes._stt_service = None
        r = client.post("/api/stt", json={"audio_base64": "aGVsbG8="})
        assert r.status_code == 503
    finally:
        stt_routes._stt_service = original


def test_stt_route_400_when_no_audio(client):
    r = client.post("/api/stt", json={"language": "en"})
    assert r.status_code == 400


def test_tts_route_503_when_service_unconfigured(client):
    import backend.routes.tts_routes as tts_routes
    original = tts_routes._tts_service
    try:
        tts_routes._tts_service = None
        r = client.post("/api/tts", json={"text": "hello"})
        assert r.status_code == 503
    finally:
        tts_routes._tts_service = original


def test_transcript_lookup_traversal_returns_404(client):
    r = client.get("/api/transcripts/..%2F..%2Fetc")
    assert r.status_code == 404


def test_connect_requires_sdp(client):
    r = client.post("/api/v1/voice/session/connect", json={})
    assert r.status_code == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
