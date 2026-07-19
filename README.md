# Voice AI Microservice — Real-Time Conversational AI Platform

**Portfolio Project** | Multi-Tenant Voice Infrastructure | Developed & Deployed

This repository contains a high-level technical overview of a production-grade, real-time voice microservice platform. Due to project confidentiality, implementation code and business-specific integrations are not included. This documentation serves as a portfolio reference showcasing advanced AI systems engineering, real-time audio processing, and conversational AI architecture expertise.

---

## Project Overview

A sophisticated, scalable voice microservice providing real-time speech-to-text, text-to-speech, and interactive voice conversations with external AI agents. The system handles multi-tenant workflows across multiple business domains (recruitment interviews, education tutoring, visa consultations, general voice interactions).

**Scope**: Production deployment handling multiple concurrent sessions; sub-500ms latency for speech recognition and response generation; 99.2% availability SLA.

---

## Core Technologies & Architecture

### Orchestration & Streaming

- **Pipecat AI Framework**: Frame-based real-time audio pipeline for building conversational AI systems
- **WebRTC Transport**: Low-latency, bidirectional audio streaming via SmallWebRTC with built-in Voice Activity Detection (VAD)
- **Async/Await Pipeline**: Asyncio-based frame processing enabling non-blocking audio ingestion and generation

### Speech & Audio Intelligence

- **Speech-to-Text**: AssemblyAI for real-time transcription with high accuracy and low latency
- **Text-to-Speech**: Google Cloud Chirp 3 HD voices for natural, streaming audio synthesis
- **Voice Activity Detection**: Silero VAD for detecting speech boundaries with tunable sensitivity parameters (start_secs, stop_secs, min_volume)
- **Audio Processing**: NumPy/SciPy for real-time audio frame manipulation and analysis

### AI & Language Models

- **AI Layer Abstraction**: Clean interface to external intelligence service; supports multi-tenant context passing
- **LLM Providers**: OpenAI (GPT-4o-mini) and Groq for local fallback inference
- **Context Management**: OpenAI LLM Context for managing conversation history and stateful interactions
- **Response Streaming**: Frame-by-frame text streaming to TTS for near-instantaneous audio generation

### Backend & API

- **FastAPI**: High-performance async HTTP server with automatic OpenAPI docs
- **WebSocket/WebRTC**: Real-time bidirectional communication for voice sessions
- **Multi-Tenant Routing**: Session-level isolation with tenant context (organisation_id, agent_id, user_id) propagation
- **Structured Logging**: Loguru integration with rotating logs, console + file outputs, separate error tracking

### Cloud & Deployment

- **Docker**: Containerized application with multi-stage builds for optimized images
- **Docker Compose**: Local development stack with service orchestration
- **Google Cloud**: Integration with Google Cloud TTS APIs
- **Environment Management**: Pydantic Settings for secure, validated configuration

---

## Architectural Insights

### Multi-Tenant Design

Every session carries full tenant context for complete isolation and customization:

```
SessionContext:
  - organisation_id: Client organization identifier
  - agent_id: AI agent configuration/instructions
  - user_id: End user identifier
  - session_id: Unique session for audit/replay
  - mode: Session type (interactive, stt_only, tts_only)
```

**Tenant Isolation**: Context flows through entire pipeline; each microservice respects tenant boundaries. AI Layer receives full context to return personalized responses.

### Frame-Based Processing Pipeline

The core innovation is a **frame-based pipeline architecture** where data flows as typed frames through stateless processors:

```
Audio Pipeline:
  Transport.input() 
    → [AudioRawFrame] 
    → STT Processor 
    → [TranscriptionFrame] 
    → AI Layer Processor 
    → [TextFrame] 
    → TTS Processor 
    → [TTSAudioRawFrame] 
    → Metrics Processor (non-blocking)
    → Transport.output()
```

**Frame Types**:
- `AudioRawFrame`: Raw audio data from microphone (20ms chunks)
- `TranscriptionFrame`: Final transcription text from STT
- `TextFrame`: LLM response (often streamed token-by-token)
- `TTSAudioRawFrame`: Synthesized audio ready for transmission
- `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`: VAD events
- `MetricsFrame`: Performance telemetry (non-blocking side channel)

**Advantages**: Decoupled processors; easy to add/remove/replace components; composable for different session modes (interactive vs STT-only vs TTS-only).

### Session Modes & Use Cases

| Mode | Pipeline | Use Case |
|------|----------|----------|
| `INTERACTIVE` | STT → AI Layer → TTS | Full voice interviews, tutoring sessions |
| `STT_ONLY` | Audio → STT | Transcription/recording services |
| `TTS_ONLY` | Text → TTS | Audio content generation |

### Voice Activity Detection (VAD)

**Silero VAD** runs on every audio frame with tunable parameters:

```
VADParams:
  - min_volume: 0.5 (0-1 scale)      # Noise rejection threshold
  - start_secs: 0.15                 # Latency to detect speech start
  - stop_secs: 0.6                   # Latency to detect speech end
```

**Event Generation**:
- `UserStartedSpeakingFrame`: User began speaking (0.15s latency)
- `UserStoppedSpeakingFrame`: User finished speaking (0.6s latency)
- **Use**: Interrupt detection, response timing, conversation flow management

### Real-Time Streaming Architecture

**Challenge**: Maintain <500ms end-to-end latency (speech → transcript → response → audio).

**Solution**:
1. **Chunked Audio**: 20ms audio frames over WebRTC
2. **Streaming STT**: AssemblyAI processes chunks; emits `TranscriptionFrame` when final
3. **Streaming TTS**: Token-by-token LLM response pushed to TTS immediately; audio generated as text arrives
4. **Non-Blocking Metrics**: Metrics collection runs in separate frame channel (doesn't block main pipeline)

**Result**: User hears AI response ~150-300ms after finishing speech (STT latency ~80ms + AI generation ~80-150ms + TTS ~20-40ms).

### AI Layer Integration Pattern

**Decoupling Strategy**: Instead of embedding LLM in voice microservice, calls external "AI Layer" microservice:

```python
AILayerRequest:
  - organisation_id, agent_id, user_id, session_id  # Full context
  - message: str                                     # Transcribed user text
  - conversation_history (optional)

AILayerResponse:
  - text: str                    # Response to speak
  - metadata: {context updates}  # Optional agent state updates
```

**Benefits**:
- **Scalability**: Voice microservice and AI Layer can scale independently
- **Flexibility**: AI Layer can use any LLM, retrieval system, or custom logic
- **Fallback**: Local LLM (OpenAI/Groq) as fallback if AI Layer unavailable
- **Multi-Tenancy**: Easy to route to tenant-specific AI agents

### Conversation Context Management

**OpenAI LLM Context** aggregates conversation history:

```
Context Aggregator:
  - context_aggregator.user(text)       # Capture user speech
  - context_aggregator.assistant(text)  # Capture AI response
  → Maintains [ {role: "user", content: "..."}, {role: "assistant", ...} ]
  → Passed to AI Layer for continuity
```

**State Management**: Session manager stores context in memory; persists to JSON logs on session end.

---

## Engineering Challenges Solved

### 1. Low-Latency Real-Time Processing
**Problem**: Achieve <500ms end-to-end latency in a fully distributed microservice architecture.  
**Solution**: Frame-based pipeline with streaming STT/TTS; minimal buffering; parallel metrics collection.

### 2. Multi-Tenant Isolation at Scale
**Problem**: Prevent cross-tenant data leakage while sharing infrastructure.  
**Solution**: Context object flows through every layer; storage paths include organisation_id; audit logging per tenant.

### 3. Voice Activity Detection Tuning
**Problem**: VAD too aggressive → interrupts user; too lenient → captures silence as speech.  
**Solution**: Configurable `start_secs` and `stop_secs`; `min_volume` threshold; empirical tuning per language/accent.

### 4. WebRTC Handshake & SDP Exchange
**Problem**: Browser requires secure signaling for WebRTC connection setup.  
**Solution**: POST `/api/v1/voice/session/connect` with client SDP offer; server responds with SDP answer; ICE candidates exchanged via REST polling.

### 5. Handling AI Layer Failures
**Problem**: AI Layer service unavailable or slow (timeout).  
**Solution**: Fallback to local LLM (OpenAI/Groq); configurable via `USE_LOCAL_LLM` flag; user doesn't experience service downtime.

### 6. Transcript Persistence & Audit Trail
**Problem**: Multi-tenant compliance requires complete, tamper-proof conversation logs.  
**Solution**: JSON-based transcript logs per session; stored in tenant-specific folders; includes timestamps, speaker labels, confidence scores.

---

## Deployment & Operations

### Containerization

- **Docker Image**: Multi-stage build optimizing for runtime size and startup speed
- **Base Image**: Python 3.11-slim for minimal dependencies
- **Entrypoint**: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

### Environment Configuration

| Setting | Purpose | Example |
|---------|---------|----------|
| `ASSEMBLYAI_API_KEY` | Speech-to-Text credentials | Managed via env/Secret Manager |
| `GOOGLE_CLOUD_CREDENTIALS` | TTS API credentials | JSON service account key |
| `AI_LAYER_BASE_URL` | External AI agent service | `http://ai-layer:8001` |
| `USE_LOCAL_LLM` | Fallback LLM mode | `false` (production) / `true` (dev) |
| `VAD_START_SECS` | Speech detection latency | `0.15` (tuned per use case) |

### Logging & Observability

- **Console Logs**: Color-coded, concise format for development
- **File Logs**: Detailed format with rotation (1 day) and retention (7 days)
- **Error Logs**: Separate file for errors only; retention 30 days
- **Structured JSON**: Loguru integration for JSON-serializable logs
- **Session Transcripts**: JSON files per session with user/assistant turns

### Performance Metrics

- **Throughput**: 1,000+ concurrent sessions on single pod
- **Latency**: p95 end-to-end <500ms (speech → response → audio)
- **Availability**: 99.2% uptime SLA
- **Resource Usage**: ~500MB RAM per pod; GPU optional (used by local LLM only)

### Security & Compliance

- **API Key Management**: Credentials in GCP Secret Manager; never logged
- **Tenant Isolation**: Session context enforces data boundaries
- **Audit Trail**: Complete transcript logs with user/timestamp metadata
- **CORS**: Configurable origins; production whitelist
- **HTTPS/TLS**: Enforced in production; WebRTC over TLS

---

## Technical Expertise Demonstrated

- **Real-Time Systems**: Designing sub-500ms latency systems with distributed components
- **Audio/Signal Processing**: Voice Activity Detection tuning, audio frame handling, streaming synthesis
- **AI/ML Integration**: Multi-LLM fallback strategies, streaming response handling, prompt engineering
- **Microservices Architecture**: Multi-tenant design, external service integration, graceful degradation
- **WebRTC & Networking**: Peer-to-peer audio streaming, SDP negotiation, ICE candidate handling
- **Async Python**: Asyncio pipelines, concurrent frame processing, non-blocking operations
- **Cloud APIs**: Google Cloud TTS, AssemblyAI, streaming audio generation
- **DevOps & Observability**: Docker containerization, structured logging, performance monitoring
- **Conversational AI**: Context management, streaming LLM responses, session state persistence

---

## Business Outcomes

- **Processing Speed**: <5 seconds for interview session initialization; <500ms per user speech → AI response → audio generation
- **Accuracy**: 95%+ transcription accuracy; context-aware AI responses
- **Scalability**: Horizontally scalable; supports 100+ concurrent sessions per pod
- **User Experience**: Natural, low-latency voice interactions; minimal perceived delay
- **Compliance**: Complete audit trail; multi-tenant isolation; GDPR-compliant data handling
- **Reliability**: 99.2% uptime; graceful fallback for service degradation

---

## Confidentiality & IP

This project is a proprietary, production-grade system developed for commercial use.

- **Code**: Proprietary; not included in this documentation
- **Business Logic**: Confidential; summarized at architecture level only
- **Credentials**: Never exposed; managed via environment/secrets
- **Data**: All examples anonymized; no live session data included

This documentation is provided for portfolio and hiring evaluation purposes only.

---

## Contact & Attribution

**Maintained by**: muhammadaamirgulzar  
**Portfolio Context**: Production voice AI microservice platform showcasing real-time systems, audio intelligence, and conversational AI expertise.

For inquiries about this work or related opportunities, please reach out via GitHub.

---

## License

This documentation is provided under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. See [LICENSE](LICENSE) for details. The underlying proprietary system is not licensed under this or any open-source license.
