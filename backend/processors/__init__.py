"""
Processors module - Custom Pipecat frame processors

Includes:
- transcript_logger: Saves interview conversations to JSON files
- transcript_processor: Captures speech in pipeline for logging
- metrics_processor: Tracks performance metrics and costs (non-blocking)
- dispatcher: Interview state management utilities
"""

from backend.processors.transcript_logger import TranscriptLogger
from backend.processors.transcript_processor import (
    UserTranscriptProcessor,
    AssistantTranscriptProcessor,
)
from backend.processors.metrics_processor import MetricsProcessor, SessionMetrics, PRICING

__all__ = [
    "TranscriptLogger",
    "UserTranscriptProcessor",
    "AssistantTranscriptProcessor",
    "MetricsProcessor",
    "SessionMetrics",
    "PRICING",
]