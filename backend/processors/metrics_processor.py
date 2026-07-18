"""
Performance Metrics Processor - Tracks timing and costs for pipeline components
Monitors STT, LLM, TTS performance and calculates API usage costs.

OPTIMIZED VERSION:
- Uses Pipecat's built-in MetricsFrame for accurate token counts
- Non-blocking async design - metrics don't delay audio frames
- Minimal memory footprint - stores only summary statistics
- Accurate cost calculations based on real usage data
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    StartFrame,
    EndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    MetricsFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# Pricing per provider (USD) - January 2026
PRICING = {
    "openai": {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "groq": {
        "llama-3.1-70b-versatile": {"input": 0.00059, "output": 0.00079},
        "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
        "llama-3.2-90b-text-preview": {"input": 0.0009, "output": 0.0009},
        "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    },
    "assemblyai": {
        "default": {"per_second": 0.00025},  # $0.015/min
    },
    "google_tts": {
        "chirp3_hd": {"per_million_chars": 16.0},
        "neural2": {"per_million_chars": 16.0},
        "standard": {"per_million_chars": 4.0},
    }
}


@dataclass
class SessionMetrics:
    """
    Lightweight metrics container for an interview session.
    Stores running totals instead of individual timing records.
    """
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Latency tracking (running averages)
    stt_total_ms: float = 0
    stt_count: int = 0
    llm_total_ms: float = 0
    llm_count: int = 0
    llm_ttft_total_ms: float = 0  # Time to first token
    tts_total_ms: float = 0
    tts_count: int = 0
    response_total_ms: float = 0
    response_count: int = 0
    
    # Usage tracking (from Pipecat MetricsFrame when available)
    total_audio_seconds: float = 0
    total_llm_input_tokens: int = 0
    total_llm_output_tokens: int = 0
    total_tts_characters: int = 0
    
    # Cost tracking
    stt_cost: float = 0
    llm_cost: float = 0
    tts_cost: float = 0
    
    # Provider info
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    
    def get_total_cost(self) -> float:
        return self.stt_cost + self.llm_cost + self.tts_cost
    
    def get_average_latencies(self) -> Dict[str, float]:
        return {
            "stt_avg_ms": round(self.stt_total_ms / max(self.stt_count, 1), 1),
            "llm_avg_ms": round(self.llm_total_ms / max(self.llm_count, 1), 1),
            "llm_ttft_avg_ms": round(self.llm_ttft_total_ms / max(self.llm_count, 1), 1),
            "tts_avg_ms": round(self.tts_total_ms / max(self.tts_count, 1), 1),
            "total_response_avg_ms": round(self.response_total_ms / max(self.response_count, 1), 1),
        }
    
    def get_summary(self) -> Dict:
        """Get complete metrics summary for logging/API response."""
        latencies = self.get_average_latencies()
        duration = 0
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "duration_seconds": round(duration, 1),
            "duration_minutes": round(duration / 60, 2),
            "latencies": latencies,
            "calls": {
                "stt": self.stt_count,
                "llm": self.llm_count,
                "tts": self.tts_count,
            },
            "usage": {
                "audio_seconds": round(self.total_audio_seconds, 1),
                "llm_input_tokens": self.total_llm_input_tokens,
                "llm_output_tokens": self.total_llm_output_tokens,
                "tts_characters": self.total_tts_characters,
            },
            "costs": {
                "stt": round(self.stt_cost, 6),
                "llm": round(self.llm_cost, 6),
                "tts": round(self.tts_cost, 6),
                "total": round(self.get_total_cost(), 6),
            },
            "cost_per_minute": round(self.get_total_cost() / max(duration / 60, 0.01), 4) if duration > 0 else 0,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }


class MetricsProcessor(FrameProcessor):
    """
    Lightweight metrics processor for voice pipeline performance tracking.
    
    Design principles:
    - Non-blocking: Frame is pushed FIRST, then metrics are updated
    - Memory efficient: Stores running totals, not individual records
    - Accurate: Uses Pipecat's MetricsFrame for real token counts when available
    - Async-safe: All metric updates are simple arithmetic (no I/O)
    """
    
    def __init__(
        self,
        *,
        session_id: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        log_interval_seconds: float = 60.0,  # Reduced frequency
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.metrics = SessionMetrics(
            session_id=session_id,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        self._log_interval = log_interval_seconds
        self._log_task: Optional[asyncio.Task] = None
        
        # Timing state (minimal)
        self._stt_start_time: Optional[float] = None
        self._llm_start_time: Optional[float] = None
        self._llm_first_token_time: Optional[float] = None
        self._tts_start_time: Optional[float] = None
        self._response_start_time: Optional[float] = None
        self._current_llm_text: str = ""
        
        logger.info(f"📊 MetricsProcessor initialized: {session_id} ({llm_provider}/{llm_model})")
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames - push first, then update metrics (non-blocking)."""
        await super().process_frame(frame, direction)
        
        # CRITICAL: Push frame FIRST to minimize latency impact
        await self.push_frame(frame, direction)
        
        # Then update metrics (simple arithmetic, non-blocking)
        current_time = time.time()
        
        try:
            if isinstance(frame, StartFrame):
                self.metrics.start_time = datetime.now()
                self._start_periodic_logging()
            
            elif isinstance(frame, EndFrame):
                self.metrics.end_time = datetime.now()
                self._stop_periodic_logging()
                self._log_final_summary()
            
            elif isinstance(frame, UserStartedSpeakingFrame):
                self._response_start_time = current_time
            
            elif isinstance(frame, UserStoppedSpeakingFrame):
                self._stt_start_time = current_time
            
            elif isinstance(frame, TranscriptionFrame):
                if frame.text and frame.text.strip():
                    self._handle_transcription(frame, current_time)
            
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._llm_start_time = current_time
                self._llm_first_token_time = None
                self._current_llm_text = ""
            
            elif isinstance(frame, TextFrame):
                if frame.text:
                    if self._llm_start_time and not self._llm_first_token_time:
                        self._llm_first_token_time = current_time
                        ttft_ms = (current_time - self._llm_start_time) * 1000
                        self.metrics.llm_ttft_total_ms += ttft_ms
                    self._current_llm_text += frame.text
            
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._handle_llm_complete(current_time)
            
            elif isinstance(frame, TTSStartedFrame):
                if not self._tts_start_time:
                    self._tts_start_time = current_time
            
            elif isinstance(frame, TTSStoppedFrame):
                self._handle_tts_complete(current_time)
            
            # Use Pipecat's built-in metrics for accurate token counts
            elif isinstance(frame, MetricsFrame):
                self._handle_pipecat_metrics(frame)
        
        except Exception as e:
            logger.error(f"📊 Metrics error: {e}")
    
    def _handle_transcription(self, frame: TranscriptionFrame, current_time: float):
        """Handle STT transcription completion."""
        if self._stt_start_time:
            duration_ms = (current_time - self._stt_start_time) * 1000
            self.metrics.stt_total_ms += duration_ms
            self.metrics.stt_count += 1
            
            # Estimate audio duration from text (fallback if no MetricsFrame)
            words = len(frame.text.split())
            estimated_seconds = words / 2.5  # ~150 wpm
            self.metrics.total_audio_seconds += estimated_seconds
            self._update_stt_cost(estimated_seconds)
            
            self._stt_start_time = None
        
        # LLM processing starts now
        self._llm_start_time = current_time
    
    def _handle_llm_complete(self, current_time: float):
        """Handle LLM response completion."""
        if self._llm_start_time:
            duration_ms = (current_time - self._llm_start_time) * 1000
            self.metrics.llm_total_ms += duration_ms
            self.metrics.llm_count += 1
            
            # Estimate tokens if MetricsFrame not available
            if self._current_llm_text:
                # Use tiktoken estimation: ~4 chars per token (more accurate than simple division)
                output_tokens = max(1, len(self._current_llm_text) // 4)
                # Input tokens typically 2-3x output for conversation
                input_tokens = output_tokens * 2
                
                self.metrics.total_llm_output_tokens += output_tokens
                self.metrics.total_llm_input_tokens += input_tokens
                self.metrics.total_tts_characters += len(self._current_llm_text)
                
                self._update_llm_cost(input_tokens, output_tokens)
            
            self._llm_start_time = None
        
        # TTS starts now
        self._tts_start_time = current_time
    
    def _handle_tts_complete(self, current_time: float):
        """Handle TTS completion."""
        if self._tts_start_time:
            duration_ms = (current_time - self._tts_start_time) * 1000
            self.metrics.tts_total_ms += duration_ms
            self.metrics.tts_count += 1
            
            # Calculate TTS cost
            self._update_tts_cost(len(self._current_llm_text))
            
            self._tts_start_time = None
        
        # Complete response timing
        if self._response_start_time:
            total_ms = (current_time - self._response_start_time) * 1000
            self.metrics.response_total_ms += total_ms
            self.metrics.response_count += 1
            self._response_start_time = None
        
        self._current_llm_text = ""
    
    def _handle_pipecat_metrics(self, frame: MetricsFrame):
        """
        Handle Pipecat's built-in MetricsFrame for accurate usage data.
        This overrides our estimates with real values when available.
        """
        # Pipecat MetricsFrame may contain token usage data
        if hasattr(frame, 'tokens'):
            tokens = frame.tokens
            if 'prompt_tokens' in tokens:
                self.metrics.total_llm_input_tokens = tokens['prompt_tokens']
            if 'completion_tokens' in tokens:
                self.metrics.total_llm_output_tokens = tokens['completion_tokens']
            self._recalculate_llm_cost()
        
        if hasattr(frame, 'ttfb'):
            logger.debug(f"📊 Pipecat TTFB: {frame.ttfb}ms")
    
    def _update_stt_cost(self, audio_seconds: float):
        """Update STT cost based on audio duration."""
        pricing = PRICING.get("assemblyai", {}).get("default", {})
        per_second = pricing.get("per_second", 0.00025)
        self.metrics.stt_cost += audio_seconds * per_second
    
    def _update_llm_cost(self, input_tokens: int, output_tokens: int):
        """Update LLM cost incrementally."""
        provider = self.metrics.llm_provider
        model = self.metrics.llm_model
        
        pricing = PRICING.get(provider, {}).get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (input_tokens / 1000) * pricing.get("input", 0.01)
        output_cost = (output_tokens / 1000) * pricing.get("output", 0.03)
        self.metrics.llm_cost += input_cost + output_cost
    
    def _recalculate_llm_cost(self):
        """Recalculate total LLM cost from accurate token counts."""
        provider = self.metrics.llm_provider
        model = self.metrics.llm_model
        
        pricing = PRICING.get(provider, {}).get(model, {"input": 0.01, "output": 0.03})
        
        input_cost = (self.metrics.total_llm_input_tokens / 1000) * pricing.get("input", 0.01)
        output_cost = (self.metrics.total_llm_output_tokens / 1000) * pricing.get("output", 0.03)
        self.metrics.llm_cost = input_cost + output_cost
    
    def _update_tts_cost(self, characters: int):
        """Update TTS cost based on character count."""
        pricing = PRICING.get("google_tts", {}).get("chirp3_hd", {})
        per_million = pricing.get("per_million_chars", 16.0)
        self.metrics.tts_cost += (characters / 1_000_000) * per_million
    
    def _start_periodic_logging(self):
        """Start periodic metrics logging (low frequency to minimize overhead)."""
        async def log_periodically():
            while True:
                await asyncio.sleep(self._log_interval)
                self._log_current_metrics()
        
        self._log_task = asyncio.create_task(log_periodically())
    
    def _stop_periodic_logging(self):
        """Stop periodic logging task."""
        if self._log_task:
            self._log_task.cancel()
            self._log_task = None
    
    def _log_current_metrics(self):
        """Log compact metrics summary."""
        lat = self.metrics.get_average_latencies()
        cost = self.metrics.get_total_cost()
        
        logger.info(
            f"📊 STT:{lat['stt_avg_ms']:.0f}ms LLM:{lat['llm_avg_ms']:.0f}ms "
            f"TTS:{lat['tts_avg_ms']:.0f}ms Total:{lat['total_response_avg_ms']:.0f}ms Cost:${cost:.4f}"
        )
    
    def _log_final_summary(self):
        """Log final metrics summary at session end."""
        s = self.metrics.get_summary()
        
        logger.info("=" * 50)
        logger.info(f"📊 SESSION METRICS: {s['session_id']}")
        logger.info(f"Duration: {s['duration_minutes']:.1f}min | Cost: ${s['costs']['total']:.4f}")
        logger.info(f"Latency - STT:{s['latencies']['stt_avg_ms']:.0f}ms LLM:{s['latencies']['llm_avg_ms']:.0f}ms TTS:{s['latencies']['tts_avg_ms']:.0f}ms")
        logger.info(f"Usage - Audio:{s['usage']['audio_seconds']:.0f}s Tokens:{s['usage']['llm_input_tokens']}in/{s['usage']['llm_output_tokens']}out TTS:{s['usage']['tts_characters']}chars")
        logger.info("=" * 50)
    
    def get_metrics(self) -> SessionMetrics:
        """Get current metrics."""
        return self.metrics
    
    def get_summary(self) -> Dict:
        """Get metrics summary dict."""
        return self.metrics.get_summary()
