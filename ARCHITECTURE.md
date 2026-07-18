# Voice Microservice - Architecture Documentation

## Overview

A **multi-tenant voice microservice** built on the **Pipecat** framework, providing real-time **STT**, **TTS**, and **interactive voice sessions** for multiple business cases including recruitment interviews, education tutoring, visa processing, and more.

The service uses WebRTC for low-latency bidirectional audio streaming and integrates with an external **AI Layer** microservice for intelligent response generation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Voice Microservice Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Frontend Client (Any Platform)                                             │
│        │                                                                     │
│        │  POST /api/v1/voice/session/start                                  │
│        │  {organisation_id, agent_id, user_id, mode}                        │
│        ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       Session Manager                                │   │
│   │   - Multi-tenant context (org_id, agent_id, user_id, session_id)    │   │
│   │   - Session lifecycle management                                     │   │
│   │   - Transcript storage by tenant                                     │   │
│   └──────────────────────────┬──────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     SmallWebRTC Transport                            │   │
│   │   (Audio In/Out, VAD - Voice Activity Detection)                    │   │
│   └──────────────────────────┬──────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    Pipeline Processing Chain                         │   │
│   │                                                                      │   │
│   │   INTERACTIVE: Audio → STT → AI Layer → TTS → Audio                 │   │
│   │   STT_ONLY:    Audio → STT → Text                                   │   │
│   │   TTS_ONLY:    Text → TTS → Audio                                   │   │
│   │                                   │                                  │   │
│   │                          MetricsProcessor                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    AI Layer (External Service)                       │   │
│   │   POST /api/v1/chat                                                  │   │
│   │   - Receives multi-tenant context                                    │   │
│   │   - Fetches agent instructions from backend                          │   │
│   │   - Returns intelligent responses                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Tenant Architecture

### Tenant Context

Every session carries full multi-tenant context:

```python
@dataclass
class SessionContext:
    organisation_id: str       # Client organization
    agent_id: str              # AI agent configuration
    user_id: str               # End user identifier
    session_id: str            # Unique session ID
    mode: SessionMode          # interactive, stt_only, tts_only
```

### Session Modes

| Mode | Pipeline | Use Case |
|------|----------|----------|
| `INTERACTIVE` | STT → AI Layer → TTS | Full voice conversations |
| `STT_ONLY` | Audio → STT → Text | Transcription services |
| `TTS_ONLY` | Text → TTS → Audio | Audio generation |

### Session Lifecycle

```
1. Client: POST /api/v1/voice/session/start
   └── Session created with multi-tenant context
   
2. Client: POST /api/v1/voice/session/connect
   └── WebRTC connection established
   └── Pipeline started based on mode
   
3. Voice interaction (mode-dependent)
   └── INTERACTIVE: Audio → STT → AI Layer → TTS → Audio
   └── STT_ONLY: Audio → STT → Text callback
   └── TTS_ONLY: Text API → TTS → Audio
   
4. Client: POST /api/v1/voice/session/{id}/stop
   └── Pipeline stopped
   └── Transcript saved
   └── Metrics returned
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Transport** | Pipecat SmallWebRTC | Real-time audio streaming via WebRTC |
| **STT** | AssemblyAI | Real-time speech-to-text transcription |
| **AI Layer** | External HTTP Service | Response generation (multi-tenant aware) |
| **Local LLM** | OpenAI GPT-4o-mini / Groq | Fallback for testing |
| **TTS** | Google Cloud TTS (Chirp 3 HD) | Natural voice synthesis |
| **VAD** | Silero VAD | Voice activity detection |
| **Framework** | Pipecat | Real-time AI voice pipeline framework |
| **Backend** | FastAPI | HTTP/WebSocket server |
| **HTTP Client** | httpx | Async AI Layer communication |

---

## Pipeline Architecture

### Frame-Based Processing

The pipeline uses a **frame-based architecture** where data flows through processors as typed frames:

```python
# INTERACTIVE MODE Pipeline
Pipeline([
    transport.input(),           # Audio frames from WebRTC
    stt,                         # TranscriptionFrame (user speech text)
    user_transcript,             # Logs user speech
    ai_layer_processor,          # Calls external AI Layer, returns TextFrame
    assistant_transcript,        # Logs AI speech  
    tts,                         # AudioFrame (synthesized speech)
    metrics_processor,           # Tracks performance (non-blocking)
    transport.output(),          # Audio frames to WebRTC
])

# STT_ONLY MODE Pipeline
Pipeline([
    transport.input(),           # Audio frames from WebRTC
    stt,                         # TranscriptionFrame
    transcript_callback,         # Sends transcription to client
    metrics_processor,
])
```

### AILayerProcessor

Replaces direct LLM integration with external AI Layer calls:

```python
class AILayerProcessor(FrameProcessor):
    """Sends transcribed text to AI Layer and returns response for TTS."""
    
    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame):
            # Forward to AI Layer with full context
            response = await self.ai_layer_client.chat(
                AILayerRequest(
                    organisation_id=self.session.context.organisation_id,
                    agent_id=self.session.context.agent_id,
                    user_id=self.session.context.user_id,
                    session_id=self.session.session_id,
                    message=frame.text,
                )
            )
            # Push response for TTS
            await self.push_frame(TextFrame(text=response.text))
        else:
            await self.push_frame(frame, direction)
```

### Key Frame Types

| Frame | Direction | Description |
|-------|-----------|-------------|
| `AudioRawFrame` | Input | Raw audio from user's microphone |
| `TranscriptionFrame` | Downstream | Final transcription from STT |
| `TextFrame` | Downstream | LLM response text (streamed token-by-token) |
| `TTSAudioRawFrame` | Downstream | Synthesized audio from TTS |
| `UserStartedSpeakingFrame` | Downstream | VAD detected speech start |
| `UserStoppedSpeakingFrame` | Downstream | VAD detected speech end |
| `LLMFullResponseStartFrame` | Downstream | LLM started generating response |
| `LLMFullResponseEndFrame` | Downstream | LLM finished generating response |
| `MetricsFrame` | Downstream | Pipecat internal metrics |

---

## Core Components

### 1. Transport Layer (`SmallWebRTCTransport`)

Handles WebRTC connection with built-in VAD:

```python
transport = SmallWebRTCTransport(
    webrtc_connection=webrtc_connection,
    params=TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                min_volume=0.5,     # Noise rejection threshold
                start_secs=0.15,    # How quickly to detect speech start
                stop_secs=0.6,      # How quickly to detect speech end
            )
        ),
        vad_audio_passthrough=True,
    )
)
```

**VAD Parameters Explained:**
- `min_volume`: Volume threshold below which audio is considered noise
- `start_secs`: Time of continuous speech before triggering `UserStartedSpeakingFrame`
- `stop_secs`: Time of silence before triggering `UserStoppedSpeakingFrame`

### 2. Speech-to-Text (`AssemblyAISTTService`)

Converts user audio to text in real-time:

```python
stt = AssemblyAISTTService(
    api_key=settings.assemblyai_api_key,
)
```

**Output:** `TranscriptionFrame` with final transcription text

### 3. AI Layer Client (`AILayerClient`)

Forwards transcribed text to external AI Layer microservice:

```python
@dataclass
class AILayerRequest:
    organisation_id: str
    agent_id: str
    user_id: str
    session_id: str
    message: str

class AILayerClient:
    async def chat(self, request: AILayerRequest) -> AILayerResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/chat",
                json=asdict(request),
                timeout=30.0
            )
            return AILayerResponse(**response.json())
```

**Fallback:** When `USE_LOCAL_LLM=true`, uses local OpenAI/Groq LLM instead.

**Streaming:** Response text is pushed as `TextFrame` to TTS for streaming synthesis.

### 4. Text-to-Speech (`GoogleTTSService`)

Converts AI text to natural speech:

```python
tts = GoogleTTSService(
    voice_id="en-US-Chirp3-HD-Zephyr",  # Chirp 3 HD for streaming
)
```

**Note:** Chirp 3 HD voices are required for streaming TTS.

### 5. Context Aggregator (`OpenAILLMContext`)

Manages conversation history:

```python
context = OpenAILLMContext(messages)
context_aggregator = llm.create_context_aggregator(context)
```

The aggregator has two parts:
- `context_aggregator.user()` - Captures user speech into context
- `context_aggregator.assistant()` - Captures AI response into context

---

## Processors

### TranscriptProcessor

Logs all conversation to JSON files for record-keeping:

```python
class UserTranscriptProcessor(FrameProcessor):
    async def process_frame(self, frame, direction):
        if isinstance(frame, TranscriptionFrame):
            self._logger.add_message("Candidate", frame.text)
        await self.push_frame(frame, direction)
```

### MetricsProcessor

Tracks performance with **zero latency impact**:

```python
async def process_frame(self, frame, direction):
    # CRITICAL: Push frame FIRST to minimize latency
    await self.push_frame(frame, direction)
    
    # Then update metrics (simple arithmetic, non-blocking)
    if isinstance(frame, TranscriptionFrame):
        self._handle_transcription(frame, current_time)
```

**Design Principles:**
1. **Non-blocking**: Frame is pushed downstream BEFORE metrics are updated
2. **Memory efficient**: Stores running totals, not individual records
3. **Accurate when possible**: Uses Pipecat's `MetricsFrame` for real token counts

---

## Data Flow Example

Here's what happens in **INTERACTIVE mode** when a user speaks:

```
1. User speaks into microphone
   │
   ▼
2. Browser captures audio via WebRTC
   │
   ▼
3. SmallWebRTCTransport receives audio
   ├── VAD analyzes audio
   ├── Emits UserStartedSpeakingFrame
   └── Passes AudioRawFrame downstream
   │
   ▼
4. AssemblyAI STT processes audio
   └── Emits TranscriptionFrame("Tell me about yourself")
   │
   ▼
5. UserTranscriptProcessor logs speech
   └── transcript.add_message("User", "Tell me about yourself")
   │
   ▼
6. AILayerProcessor sends to external AI Layer
   ├── POST /api/v1/chat
   │   {
   │     "organisation_id": "org_123",
   │     "agent_id": "agent_456", 
   │     "user_id": "user_789",
   │     "session_id": "session_abc123",
   │     "message": "Tell me about yourself"
   │   }
   └── Receives: {"text": "That's a great question..."}
   │
   ▼
7. AILayerProcessor emits TextFrame
   └── TextFrame("That's a great question...")
   │
   ▼
8. AssistantTranscriptProcessor logs response
   │
   ▼
9. Google TTS converts text to audio (streaming)
   ├── TTSStartedFrame
   ├── TTSAudioRawFrame (audio chunks)
   └── TTSStoppedFrame
   │
   ▼
10. MetricsProcessor records timing/costs
    │
    ▼
11. SmallWebRTCTransport sends audio to browser
    │
    ▼
12. User hears AI response through speakers
```

---

## Latency Optimization

### Current Latency Targets

| Stage | Target | Typical |
|-------|--------|---------|
| VAD Detection | <200ms | 150ms |
| STT Processing | <500ms | 300-500ms |
| LLM First Token | <500ms | 200-400ms |
| TTS First Audio | <300ms | 200-300ms |
| **Total Response** | <1500ms | 800-1200ms |

### Optimization Strategies

1. **Streaming Everything**: LLM tokens stream to TTS, TTS streams to WebRTC
2. **Fast LLM Model**: GPT-4o-mini or Groq Llama for speed
3. **Short Responses**: System prompt instructs "1-2 sentences for voice"
4. **VAD Tuning**: Lower `stop_secs` for faster turn-taking
5. **Non-blocking Metrics**: Metrics update after frame is pushed

---

## Configuration

### Environment Variables

```bash
# Server
HOST=127.0.0.1
PORT=8000
DEBUG=true

# AI Layer (External Microservice)
AI_LAYER_BASE_URL=http://localhost:8001
AI_LAYER_CHAT_ENDPOINT=/api/v1/chat
AI_LAYER_TIMEOUT_SECONDS=30

# Use local LLM instead of AI Layer (for testing)
USE_LOCAL_LLM=false

# API Keys
ASSEMBLYAI_API_KEY=your_key
OPENAI_API_KEY=your_key  # For local LLM fallback
GROQ_API_KEY=your_key    # Optional
GOOGLE_CLOUD_TTS_CREDENTIALS=google-credentials.json

# Local LLM Configuration (when USE_LOCAL_LLM=true)
LLM_PROVIDER=openai  # or "groq"
LLM_MODEL=gpt-4o-mini

# TTS Configuration  
TTS_VOICE_NAME=en-US-Chirp3-HD-Zephyr

# Session Management
SESSION_TIMEOUT_MINUTES=60
MAX_SESSIONS_PER_ORG=100
```

### Settings Class (`config/settings.py`)

```python
class Settings(BaseSettings):
    # AI Layer
    ai_layer_base_url: str = "http://localhost:8001"
    ai_layer_chat_endpoint: str = "/api/v1/chat"
    ai_layer_timeout_seconds: int = 30
    use_local_llm: bool = False
    
    # Voice services
    tts_voice_name: str = "en-US-Chirp3-HD-Zephyr"
    
    # Session management
    session_timeout_minutes: int = 60
    max_sessions_per_org: int = 100
    transcript_dir: str = "logs/transcripts"

class SessionMode(str, Enum):
    STT_ONLY = "stt_only"
    TTS_ONLY = "tts_only"
    INTERACTIVE = "interactive"
```

---

## Metrics & Cost Tracking

### What's Tracked

| Metric | How It's Measured |
|--------|-------------------|
| **STT Latency** | Time from `UserStoppedSpeakingFrame` to `TranscriptionFrame` |
| **LLM Latency** | Time from transcription to `LLMFullResponseEndFrame` |
| **LLM TTFT** | Time to first `TextFrame` (time-to-first-token) |
| **TTS Latency** | Time from `LLMFullResponseEndFrame` to `TTSStoppedFrame` |
| **Total Response** | Time from `UserStartedSpeakingFrame` to `TTSStoppedFrame` |

### Cost Calculation

**Note:** Token counts are estimated (~4 chars/token) unless Pipecat's `MetricsFrame` provides real values.

```python
# STT Cost (AssemblyAI)
stt_cost = audio_seconds * $0.00025/second

# LLM Cost (OpenAI GPT-4o-mini)
llm_cost = (input_tokens/1000 * $0.00015) + (output_tokens/1000 * $0.0006)

# TTS Cost (Google Chirp 3 HD)
tts_cost = (characters / 1_000_000) * $16.0
```

### Metrics Summary (End of Session)

```
==================================================
📊 SESSION METRICS: session_abc123
Duration: 5.2min | Cost: $0.0142
Latency - STT:320ms LLM:450ms TTS:280ms
Usage - Audio:45s Tokens:1250in/890out TTS:3420chars
==================================================
```

---

## Transcript Storage

Transcripts are saved to `logs/transcripts/interview_{session_id}.json`:

```json
{
  "session_id": "session_abc123",
  "start_time": "2026-01-09T10:30:00.000000",
  "end_time": "2026-01-09T10:35:12.000000",
  "metadata": {
    "candidate_name": "John Doe",
    "position": "Software Engineer",
    "interviewer": "SUSAN AI",
    "total_duration_seconds": 312.5,
    "exchanges_count": 24
  },
  "conversation": [
    {
      "timestamp": "2026-01-09T10:30:05.123456",
      "speaker": "System",
      "text": "Interview session started",
      "elapsed_seconds": 0.1
    },
    {
      "timestamp": "2026-01-09T10:30:08.456789",
      "speaker": "Interviewer",
      "text": "Hello! I'm SUSAN, your AI interviewer today...",
      "elapsed_seconds": 3.5
    },
    ...
  ],
  "metrics": {
    "session_id": "session_abc123",
    "duration_seconds": 312.5,
    "latencies": {
      "stt_avg_ms": 320.5,
      "llm_avg_ms": 450.2,
      "tts_avg_ms": 280.1,
      "total_response_avg_ms": 1050.8
    },
    "usage": {
      "audio_seconds": 45.2,
      "llm_input_tokens": 1250,
      "llm_output_tokens": 890,
      "tts_characters": 3420
    },
    "costs": {
      "stt": 0.0113,
      "llm": 0.0007,
      "tts": 0.0022,
      "total": 0.0142
    }
  },
  "summary": "Interview completed"
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve frontend HTML |
| `/health` | GET | Health check |
| `/api/config` | GET | Client configuration |
| `/api/interview/start` | POST | Create new session |
| `/api/rtc-connect` | POST | WebRTC SDP exchange |
| `/api/interview/{id}/stop` | POST | Stop interview |
| `/api/interview/{id}/status` | GET | Get session status |
| `/api/interview/{id}/metrics` | GET | Get session metrics |
| `/api/transcripts` | GET | List all transcripts |
| `/api/transcripts/{id}` | GET | Get specific transcript |
| `/api/pricing` | GET | Current API pricing |

---

## File Structure

```
susan-v2-voice/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, endpoints
│   ├── pipeline.py          # Pipecat pipeline setup
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── transcript_logger.py    # JSON transcript storage
│   │   ├── transcript_processor.py # Pipeline processors for logging
│   │   ├── metrics_processor.py    # Performance & cost tracking
│   │   └── dispatcher.py           # Interview state management
│   └── services/
│       ├── __init__.py
│       ├── llm_service.py   # LLM configuration utilities
│       ├── stt_service.py   # STT configuration utilities
│       └── tts_service.py   # TTS configuration utilities
├── config/
│   ├── __init__.py
│   ├── settings.py          # Pydantic settings
│   └── sample_resume.json   # Demo candidate data
├── frontend/
│   ├── index.html           # WebRTC client UI
│   └── app.js               # WebRTC JavaScript client
├── logs/
│   ├── app.log              # Application logs
│   └── transcripts/         # Interview JSON files
├── google-credentials.json  # GCP service account
├── requirements.txt
├── environment.yml          # Conda environment
└── README.md
```

---

## Extending the Pipeline

### Adding a New Processor

1. Create a new file in `backend/processors/`:

```python
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

class MyCustomProcessor(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Your custom logic here
        if isinstance(frame, TranscriptionFrame):
            # Do something with user speech
            pass
        
        # ALWAYS push frame downstream
        await self.push_frame(frame, direction)
```

2. Add to pipeline in `backend/pipeline.py`:

```python
from backend.processors.my_custom import MyCustomProcessor

my_processor = MyCustomProcessor()

pipeline = Pipeline([
    transport.input(),
    stt,
    my_processor,  # Add your processor
    user_transcript,
    ...
])
```

### Switching LLM Provider

Change in `.env`:

```bash
LLM_PROVIDER=groq
LLM_MODEL=llama-3.1-70b-versatile
```

### Changing TTS Voice

Available Chirp 3 HD voices:
- `en-US-Chirp3-HD-Zephyr` (default, neutral)
- `en-US-Chirp3-HD-Aoede` (warm, friendly)
- `en-US-Chirp3-HD-Charon` (deep, authoritative)

Change in `.env`:

```bash
TTS_VOICE_NAME=en-US-Chirp3-HD-Aoede
```

---

## Troubleshooting

### Common Issues

**1. High latency (>2s response time)**
- Check network latency to API providers
- Use faster LLM model (gpt-4o-mini instead of gpt-4)
- Reduce `stop_secs` in VAD for faster turn-taking

**2. Audio cutting off**
- Increase `stop_secs` in VAD parameters
- Check for network packet loss

**3. Transcription errors**
- Ensure good audio quality (no echo/background noise)
- Check AssemblyAI API status

**4. TTS not speaking**
- Verify Google Cloud credentials path
- Check TTS voice name is valid Chirp 3 HD voice

### Debug Mode

Enable debug logging:

```bash
LOG_LEVEL=DEBUG
DEBUG=true
```

Check logs at `logs/app.log` for detailed frame-by-frame processing.

---

## Performance Considerations

1. **Memory**: Running totals instead of storing individual records
2. **CPU**: Minimal processing in frame handlers
3. **Network**: WebRTC handles transport optimization
4. **Async**: All I/O is non-blocking

The system is designed to handle multiple concurrent interviews, with each session running in its own pipeline task.
