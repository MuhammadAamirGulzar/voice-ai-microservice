"""
Transcript Processor - Captures all speech for transcript logging
Listens to STT transcriptions and TTS outputs to build a complete transcript.
"""
import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from backend.processors.transcript_logger import TranscriptLogger


class TranscriptProcessor(FrameProcessor):
    """
    Captures speech from user (STT) and assistant (LLM/TTS) for transcript logging.
    
    Listens for:
    - TranscriptionFrame: User speech from STT
    - TextFrame after LLM: Assistant responses
    """
    
    def __init__(
        self,
        *,
        transcript_logger: TranscriptLogger,
        **kwargs
    ):
        """
        Initialize the transcript processor.
        
        Args:
            transcript_logger: TranscriptLogger instance to log to
        """
        super().__init__(**kwargs)
        
        self._logger = transcript_logger
        self._current_assistant_text = ""
        self._in_llm_response = False
        
        logger.info(f"TranscriptProcessor initialized for session: {transcript_logger.session_id}")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and capture speech for transcript."""
        await super().process_frame(frame, direction)
        
        try:
            # Capture user speech from STT transcription
            if isinstance(frame, TranscriptionFrame):
                if frame.text and frame.text.strip():
                    self._logger.add_message("Candidate", frame.text)
                    logger.debug(f"Transcript - Candidate: {frame.text[:50]}...")
            
            # Track LLM response boundaries
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._in_llm_response = True
                self._current_assistant_text = ""
            
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._in_llm_response = False
                if self._current_assistant_text.strip():
                    self._logger.add_message("Interviewer", self._current_assistant_text)
                    logger.debug(f"Transcript - Interviewer: {self._current_assistant_text[:50]}...")
                self._current_assistant_text = ""
            
            # Capture assistant text during LLM response
            elif isinstance(frame, TextFrame) and self._in_llm_response:
                if frame.text:
                    self._current_assistant_text += frame.text
        
        except Exception as e:
            logger.error(f"Error processing transcript frame: {e}")
        
        # Always pass frame through
        await self.push_frame(frame, direction)
    
    def get_transcript(self):
        """Get current transcript data."""
        return self._logger.get_transcript()
    
    def finalize(self, summary: Optional[str] = None):
        """Finalize and save the transcript."""
        self._logger.finalize(summary)


class TranscriptCaptureProcessor(FrameProcessor):
    """
    Simplified transcript capture that logs all text frames.
    Place this after TTS to capture what gets spoken.
    """
    
    def __init__(
        self,
        *,
        transcript_logger: TranscriptLogger,
        speaker: str = "Interviewer",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._logger = transcript_logger
        self._speaker = speaker
        self._buffer = ""
        self._last_frame_time = datetime.now()
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Capture text frames for transcript."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame) and frame.text:
            self._buffer += frame.text
            self._last_frame_time = datetime.now()
        
        elif isinstance(frame, TTSStoppedFrame):
            # TTS finished, log the complete utterance
            if self._buffer.strip():
                self._logger.add_message(self._speaker, self._buffer.strip())
                logger.debug(f"Transcript - {self._speaker}: {self._buffer[:50]}...")
            self._buffer = ""
        
        await self.push_frame(frame, direction)


class UserTranscriptProcessor(FrameProcessor):
    """
    Captures user speech transcriptions with buffering.
    Accumulates speech segments and logs complete utterances when user stops speaking.
    Uses UserStoppedSpeakingFrame from VAD to detect end of utterance.
    """
    
    def __init__(
        self,
        *,
        transcript_logger: TranscriptLogger,
        buffer_timeout_seconds: float = 1.5,  # Fallback timeout if VAD frame not received
        **kwargs
    ):
        super().__init__(**kwargs)
        self._logger = transcript_logger
        self._buffer = ""
        self._buffer_timeout = buffer_timeout_seconds
        self._flush_task: Optional[asyncio.Task] = None
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Capture user transcription frames with buffering."""
        await super().process_frame(frame, direction)
        
        # Import VAD frame types
        from pipecat.frames.frames import UserStoppedSpeakingFrame, UserStartedSpeakingFrame
        
        if isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                # Add to buffer with space separator
                if self._buffer:
                    self._buffer += " " + frame.text.strip()
                else:
                    self._buffer = frame.text.strip()
                
                logger.debug(f"📝 Buffer updated: {self._buffer[:50]}...")
                
                # Cancel existing flush task and start new one (fallback timeout)
                if self._flush_task and not self._flush_task.done():
                    self._flush_task.cancel()
                self._flush_task = asyncio.create_task(self._delayed_flush())
        
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # VAD detected user stopped speaking - flush buffer
            await self._flush_buffer()
        
        await self.push_frame(frame, direction)
    
    async def _delayed_flush(self):
        """Flush buffer after timeout (fallback if VAD frame not received)."""
        try:
            await asyncio.sleep(self._buffer_timeout)
            await self._flush_buffer()
        except asyncio.CancelledError:
            pass  # Expected when new speech arrives
    
    async def _flush_buffer(self):
        """Flush accumulated buffer to transcript."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        
        if self._buffer.strip():
            self._logger.add_message("Candidate", self._buffer.strip())
            logger.info(f"📝 Candidate: {self._buffer}")
            self._buffer = ""


class AssistantTranscriptProcessor(FrameProcessor):
    """
    Captures assistant/interviewer speech.
    Place this to capture LLM outputs before TTS.
    """
    
    def __init__(
        self,
        *,
        transcript_logger: TranscriptLogger,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._logger = transcript_logger
        self._buffer = ""
        self._in_response = False
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Capture assistant text frames."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, LLMFullResponseStartFrame):
            self._in_response = True
            self._buffer = ""
        
        elif isinstance(frame, TextFrame) and self._in_response:
            if frame.text:
                self._buffer += frame.text
        
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._in_response = False
            if self._buffer.strip():
                self._logger.add_message("Interviewer", self._buffer.strip())
                logger.info(f"🎤 Interviewer: {self._buffer[:100]}...")
            self._buffer = ""
        
        await self.push_frame(frame, direction)
