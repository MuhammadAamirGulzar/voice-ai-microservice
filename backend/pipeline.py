"""
Pipecat Pipeline - Multi-tenant Voice Pipeline with AI Layer Integration

Supports three modes:
- STT_ONLY: Audio → STT → Text (returns transcribed text)
- TTS_ONLY: Text → TTS → Audio (returns synthesized speech)
- INTERACTIVE: Audio → STT → AI Layer → TTS → Audio (full conversation)

Features:
- External AI Layer integration for response generation
- Fallback to local LLM for testing
- Multi-tenant session context
- Performance metrics and transcript logging
"""
import asyncio
import os
from datetime import datetime
from typing import Optional, Tuple

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.base_transport import TransportParams

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

from config.settings import get_settings, load_custom_instructions
from backend.processors.transcript_logger import TranscriptLogger
from backend.processors.transcript_processor import UserTranscriptProcessor, AssistantTranscriptProcessor
from backend.processors.metrics_processor import MetricsProcessor
from backend.services.session_manager import Session, SessionMode


# ICE servers for WebRTC connectivity
ICE_SERVERS = [
    IceServer(urls="stun:stun.l.google.com:19302"),
    IceServer(urls="stun:stun1.l.google.com:19302"),
]


class AILayerProcessor(FrameProcessor):
    """
    Processor that forwards transcribed text to AI Layer and receives responses.

    Replaces the local LLM in the pipeline when AI Layer is configured.

    Concurrency contract (matters for voice):
    - process_frame() must NEVER block: the AI Layer request runs in its own
      task. Blocking here stalls every frame behind it — including the audio
      the transport is trying to play — for up to the full request timeout.
    - Interruptions cancel the in-flight request: when the user speaks over
      the assistant, a stale response arriving seconds later must not be
      spoken. Each request carries a turn number; only the current turn's
      response is pushed.
    - Responses are wrapped in LLMFullResponseStart/End frames so the
      assistant transcript processor and metrics see them exactly like a
      local LLM's output. (Bare TextFrames are invisible to both.)
    """

    def __init__(
        self,
        session: Session,
        transcript_logger: Optional[TranscriptLogger] = None,
    ):
        super().__init__()
        self.session = session
        self.transcript_logger = transcript_logger
        self._turn = 0
        self._request_task: Optional[asyncio.Task] = None

        logger.info(f"AILayerProcessor initialized for session: {session.session_id}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            # User barged in: whatever the AI Layer is computing is stale.
            self._cancel_inflight()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if text:
                logger.debug(f"AILayer received transcript: {text[:50]}...")
                self._turn += 1
                self._cancel_inflight()
                self._request_task = asyncio.create_task(
                    self._respond(text, self._turn))
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)

    def _cancel_inflight(self):
        if self._request_task and not self._request_task.done():
            self._request_task.cancel()

    async def _respond(self, user_message: str, turn: int):
        """Fetch the AI response off the frame path and push it when ready."""
        try:
            response = await self._get_ai_response(user_message)
        except asyncio.CancelledError:
            logger.debug(f"AI Layer request cancelled (turn {turn})")
            return
        if not response or turn != self._turn:
            # A newer utterance superseded this one while we waited.
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(TextFrame(text=response))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _get_ai_response(self, user_message: str) -> Optional[str]:
        """Get response from AI Layer (voice-tuned timeout, single retry)."""
        try:
            from backend.services.ai_layer_client import (
                get_ai_layer_client,
                AILayerRequest,
                AILayerError,
            )
            from config.settings import get_settings

            client = get_ai_layer_client()

            request = AILayerRequest(
                organisation_id=self.session.context.organisation_id or "",
                agent_id=self.session.context.agent_id or "",
                user_id=self.session.context.user_id or "",
                session_id=self.session.session_id,
                message=user_message,
            )

            response = await client.send_message(
                request,
                timeout_s=get_settings().ai_layer_voice_timeout_seconds,
                max_retries=1,
            )

            logger.debug(
                f"AI Layer response | session={self.session.session_id} | "
                f"latency={response.processing_time_ms:.0f}ms"
            )

            return response.text

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"AI Layer error: {e}")
            # Return a graceful fallback message
            return "I'm sorry, I'm having trouble processing your request. Could you please repeat that?"


async def create_and_run_pipeline(
    session: Session,
    sdp_offer: str,
    settings,
) -> Tuple[str, asyncio.Task, PipelineRunner, "MetricsProcessor", "TranscriptLogger"]:
    """
    Create and run the voice pipeline with SmallWebRTC transport.
    
    Supports multiple modes:
    - INTERACTIVE: Full STT → AI Layer → TTS pipeline
    - STT_ONLY: Audio → STT only
    - TTS_ONLY: Text → TTS only
    
    Args:
        session: Session object with context and configuration
        sdp_offer: WebRTC SDP offer from client
        settings: Application settings
    
    Returns:
        Tuple of (sdp_answer, task, runner, metrics_processor, transcript_logger)
    """
    session_id = session.session_id
    context = session.context
    
    logger.info(
        f"Creating pipeline | session={session_id} | "
        f"mode={context.mode.value} | org={context.organisation_id}"
    )
    
    # Initialize transcript logger with multi-tenant context
    transcript_logger = TranscriptLogger(
        session_id=session_id,
        output_dir=settings.transcript_dir,
        organisation_id=context.organisation_id,
        agent_id=context.agent_id,
        user_id=context.user_id,
    )
    
    # Create WebRTC connection and initialize with SDP offer
    webrtc_connection = SmallWebRTCConnection(ice_servers=ICE_SERVERS)
    await webrtc_connection.initialize(sdp=sdp_offer, type="offer")
    
    # Create SmallWebRTC transport with VAD
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    min_volume=0.5,
                    start_secs=0.15,
                    stop_secs=0.6,
                )
            ),
            vad_audio_passthrough=True,
        )
    )
    
    # Initialize STT (AssemblyAI)
    from pipecat.services.assemblyai.stt import AssemblyAISTTService
    stt = AssemblyAISTTService(api_key=settings.assemblyai_api_key)
    logger.info(f"⚡ STT initialized: AssemblyAI")
    
    # Initialize TTS (Google Cloud)
    from pipecat.services.google.tts import GoogleTTSService
    credentials_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        settings.google_cloud_tts_credentials
    )
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    # Use voice from session context
    voice_name = _get_voice_name(context.voice, context.language)
    tts = GoogleTTSService(voice_id=voice_name)
    logger.info(f"⚡ TTS initialized: Google Cloud ({voice_name})")
    
    # Create transcript processors
    user_transcript = UserTranscriptProcessor(transcript_logger=transcript_logger)
    assistant_transcript = AssistantTranscriptProcessor(transcript_logger=transcript_logger)
    
    # Create metrics processor
    metrics_processor = MetricsProcessor(
        session_id=session_id,
        llm_provider="ai_layer" if not settings.use_local_llm else settings.llm_provider,
        llm_model="external" if not settings.use_local_llm else settings.llm_model,
        log_interval_seconds=30.0,
    )
    
    # Build pipeline based on mode
    if context.mode == SessionMode.INTERACTIVE:
        pipeline = await _build_interactive_pipeline(
            session=session,
            settings=settings,
            transport=transport,
            stt=stt,
            tts=tts,
            user_transcript=user_transcript,
            assistant_transcript=assistant_transcript,
            metrics_processor=metrics_processor,
            transcript_logger=transcript_logger,
        )
    elif context.mode == SessionMode.STT_ONLY:
        pipeline = await _build_stt_only_pipeline(
            transport=transport,
            stt=stt,
            user_transcript=user_transcript,
            metrics_processor=metrics_processor,
        )
    else:  # TTS_ONLY
        pipeline = await _build_tts_only_pipeline(
            transport=transport,
            tts=tts,
            assistant_transcript=assistant_transcript,
            metrics_processor=metrics_processor,
        )
    
    # Create pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    
    # Set up event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected | session={session_id}")
        transcript_logger.add_message("System", "Session started")
        
        # Send initial greeting for interactive sessions
        if context.mode == SessionMode.INTERACTIVE:
            await _send_initial_greeting(task, session, settings)
    
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected | session={session_id}")
        transcript_logger.add_message("System", "Session ended")
        transcript_logger.finalize(
            summary="Session completed",
            metrics=metrics_processor.get_summary()
        )
        await task.cancel()
    
    # Create runner
    runner = PipelineRunner()
    
    # Get SDP answer
    answer = webrtc_connection.get_answer()
    sdp_answer = answer["sdp"] if answer else None
    
    if not sdp_answer:
        raise Exception("Failed to generate SDP answer")
    
    # Start pipeline in background task
    async def run_pipeline_with_cleanup():
        """Run pipeline and ensure transcript is saved."""
        try:
            await runner.run(task)
        finally:
            if not transcript_logger.transcript.get("end_time"):
                transcript_logger.finalize(
                    summary="Session terminated",
                    metrics=metrics_processor.get_summary()
                )
            logger.info(f"Pipeline completed | session={session_id}")
    
    pipeline_task = asyncio.create_task(
        run_pipeline_with_cleanup(),
        name=f"pipeline_{session_id}"
    )
    
    logger.info(f"Pipeline started | session={session_id} | mode={context.mode.value}")
    
    return sdp_answer, pipeline_task, runner, metrics_processor, transcript_logger


async def _build_interactive_pipeline(
    session: Session,
    settings,
    transport,
    stt,
    tts,
    user_transcript,
    assistant_transcript,
    metrics_processor,
    transcript_logger,
) -> Pipeline:
    """Build full interactive pipeline: STT → LLM/AI Layer → TTS"""
    
    if settings.use_local_llm:
        # Use local LLM for testing
        llm, context_aggregator = await _create_local_llm(settings, session)
        
        pipeline = Pipeline([
            transport.input(),
            stt,
            user_transcript,
            context_aggregator.user(),
            llm,
            assistant_transcript,
            tts,
            metrics_processor,
            transport.output(),
            context_aggregator.assistant(),
        ])
    else:
        # Use AI Layer
        ai_processor = AILayerProcessor(
            session=session,
            transcript_logger=transcript_logger,
        )
        
        pipeline = Pipeline([
            transport.input(),
            stt,
            user_transcript,
            ai_processor,
            assistant_transcript,
            tts,
            metrics_processor,
            transport.output(),
        ])
    
    return pipeline


async def _build_stt_only_pipeline(
    transport,
    stt,
    user_transcript,
    metrics_processor,
) -> Pipeline:
    """Build STT-only pipeline: Audio → STT → Text"""
    return Pipeline([
        transport.input(),
        stt,
        user_transcript,
        metrics_processor,
        transport.output(),
    ])


async def _build_tts_only_pipeline(
    transport,
    tts,
    assistant_transcript,
    metrics_processor,
) -> Pipeline:
    """Build TTS-only pipeline: Text → TTS → Audio"""
    return Pipeline([
        transport.input(),
        tts,
        assistant_transcript,
        metrics_processor,
        transport.output(),
    ])


async def _create_local_llm(settings, session: Session):
    """Create local LLM for testing/fallback."""
    # Build system prompt
    custom_instructions = load_custom_instructions()
    system_prompt = f"""You are a professional AI assistant in a voice conversation.
Organisation: {session.context.organisation_id or 'Not specified'}
Agent: {session.context.agent_id or 'Not specified'}

{custom_instructions}

Keep responses SHORT (1-2 sentences for voice).
Be conversational and natural.
NEVER prefix your messages with a name or role."""

    messages = [{"role": "system", "content": system_prompt}]
    
    if settings.llm_provider.lower() == "groq":
        from pipecat.services.groq.llm import GroqLLMService
        llm = GroqLLMService(
            api_key=settings.groq_api_key,
            model=settings.llm_model or "llama-3.1-70b-versatile",
        )
    else:
        from pipecat.services.openai.llm import OpenAILLMService
        llm = OpenAILLMService(
            api_key=settings.openai_api_key,
            model=settings.llm_model or "gpt-4o-mini",
        )
    
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    
    logger.info(f"⚡ Local LLM initialized: {settings.llm_provider} ({settings.llm_model})")
    
    return llm, context_aggregator


async def _send_initial_greeting(task, session: Session, settings):
    """Send initial greeting message."""
    if settings.use_local_llm:
        # For local LLM, trigger context to generate greeting
        messages = [
            {"role": "system", "content": "Start the conversation with a brief, friendly greeting."},
        ]
        await task.queue_frames([LLMMessagesFrame(messages)])
    else:
        # For AI Layer, send a start message to get initial greeting
        try:
            from backend.services.ai_layer_client import get_ai_layer_client, AILayerRequest
            
            client = get_ai_layer_client()
            request = AILayerRequest(
                organisation_id=session.context.organisation_id or "",
                agent_id=session.context.agent_id or "",
                user_id=session.context.user_id or "",
                session_id=session.session_id,
                message="__SESSION_START__",
                metadata={"event": "session_start"},
            )
            
            response = await client.send_message(request)
            if response.text:
                await task.queue_frames([TextFrame(text=response.text)])
                
        except Exception as e:
            logger.warning(f"Failed to get initial greeting from AI Layer: {e}")
            # Send a default greeting
            await task.queue_frames([
                TextFrame(text="Hello! I'm ready to assist you. How can I help?")
            ])


def _get_voice_name(voice: str, language: str) -> str:
    """Get full voice name from short name and language."""
    # Map short names to full Google TTS voice IDs
    voice_map = {
        "Aoede": f"{language}-Chirp3-HD-Aoede",
        "Charon": f"{language}-Chirp3-HD-Charon",
        "Fenrir": f"{language}-Chirp3-HD-Fenrir",
        "Kore": f"{language}-Chirp3-HD-Kore",
        "Puck": f"{language}-Chirp3-HD-Puck",
        "Zephyr": f"{language}-Chirp3-HD-Zephyr",
    }
    
    # If already a full voice ID, return as-is
    if "Chirp" in voice or "Neural" in voice:
        return voice
    
    return voice_map.get(voice, f"{language}-Chirp3-HD-Zephyr")
