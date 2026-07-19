"""
Transcript Logger - Multi-tenant session transcript management.

Features:
- Stores complete session transcripts with multi-tenant context
- Organizes transcripts by organization/business case
- Includes metadata, timing, metrics, costs, and conversation history
- Supports future database integration
"""
from loguru import logger
from pathlib import Path
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any

_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def safe_id(value: Optional[str], fallback: str = "unknown") -> str:
    """
    Sanitize client-supplied identifiers before they touch the filesystem.
    organisation_id and session_id come from API requests; without this a
    value like "../../etc" writes or reads outside the transcript root.
    """
    if not value:
        return fallback
    cleaned = _SAFE_ID.sub("_", str(value))[:128].strip("._") or fallback
    return cleaned


class TranscriptLogger:
    """
    Logs session conversations to structured JSON files.
    Includes multi-tenant metadata, timing, metrics, and full transcript.
    
    Storage structure:
    logs/transcripts/
        {org_id}/
            session_{session_id}.json
    """
    
    def __init__(
        self,
        session_id: str,
        output_dir: str = "logs/transcripts",
        organisation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.session_id = session_id
        self.base_output_dir = Path(output_dir)
        
        # Multi-tenant context
        self.organisation_id = organisation_id
        self.agent_id = agent_id
        self.user_id = user_id
        
        # Build output path with organization structure
        self.output_dir = self._build_output_path()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize transcript structure
        self.transcript = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            
            # Multi-tenant context
            "context": {
                "organisation_id": organisation_id,
                "agent_id": agent_id,
                "user_id": user_id,
            },
            
            # Session metadata
            "metadata": {
                "participant_name": None,
                "session_type": None,
                "total_duration_seconds": 0,
                "exchanges_count": 0,
                "user_messages": 0,
                "assistant_messages": 0,
            },
            
            # Conversation history
            "conversation": [],
            
            # Performance metrics (populated at finalize)
            "metrics": None,
            
            # Summary (populated at finalize)
            "summary": None,
        }
        
        self.start_time = datetime.now()
        
        logger.debug(
            f"TranscriptLogger initialized | session={session_id} | "
            f"org={organisation_id}"
        )
    
    def _build_output_path(self) -> Path:
        """Build output path based on multi-tenant context."""
        path = self.base_output_dir

        # Add organization folder if available (sanitized — client input)
        if self.organisation_id:
            path = path / safe_id(self.organisation_id)

        return path
    
    def add_message(self, speaker: str, text: str, metadata: Optional[Dict] = None):
        """
        Add a message to the transcript.
        
        Args:
            speaker: Speaker identifier ("User", "Assistant", "System")
            text: Message text content
            metadata: Optional additional metadata for the message
        """
        if not text or not text.strip():
            return
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "text": text.strip(),
            "elapsed_seconds": round((datetime.now() - self.start_time).total_seconds(), 1),
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        self.transcript["conversation"].append(entry)
        self.transcript["metadata"]["exchanges_count"] += 1
        
        # Track message counts by speaker
        if speaker.lower() == "user":
            self.transcript["metadata"]["user_messages"] += 1
        elif speaker.lower() == "assistant":
            self.transcript["metadata"]["assistant_messages"] += 1
        
        logger.debug(f"Transcript [{self.session_id}] {speaker}: {text[:50]}...")
    
    def set_participant_name(self, name: str):
        """Set the participant/candidate name in metadata."""
        self.transcript["metadata"]["participant_name"] = name
    
    def set_session_type(self, session_type: str):
        """Set the session type in metadata."""
        self.transcript["metadata"]["session_type"] = session_type
    
    def set_metrics(self, metrics: Dict):
        """Set performance metrics from MetricsProcessor."""
        self.transcript["metrics"] = metrics
    
    def add_metadata(self, key: str, value: Any):
        """Add custom metadata field."""
        self.transcript["metadata"][key] = value
    
    def finalize(self, summary: Optional[str] = None, metrics: Optional[Dict] = None):
        """
        Finalize transcript and write to file.
        
        Args:
            summary: Optional session summary text
            metrics: Optional performance metrics dictionary
        """
        # Update metadata
        end_time = datetime.now()
        self.transcript["end_time"] = end_time.isoformat()
        self.transcript["metadata"]["total_duration_seconds"] = round(
            (end_time - self.start_time).total_seconds(), 1
        )
        
        if summary:
            self.transcript["summary"] = summary
        
        if metrics:
            self.transcript["metrics"] = metrics
        
        # Write to file
        self._write_to_file()
        
        logger.info(
            f"Transcript saved | session={self.session_id} | "
            f"exchanges={self.transcript['metadata']['exchanges_count']} | "
            f"duration={self.transcript['metadata']['total_duration_seconds']:.0f}s"
        )
    
    def _write_to_file(self):
        """Write transcript to JSON file."""
        try:
            filename = f"session_{safe_id(self.session_id)}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.transcript, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Transcript written to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to write transcript: {e}")
    
    def get_transcript(self) -> Dict:
        """Get current transcript data."""
        return self.transcript.copy()
    
    def get_conversation(self) -> List[Dict]:
        """Get conversation messages only."""
        return self.transcript["conversation"].copy()
    
    def get_last_messages(self, count: int = 10) -> List[Dict]:
        """Get last N messages from conversation."""
        return self.transcript["conversation"][-count:]


# =============================================================================
# Utility Functions
# =============================================================================

def load_transcript(
    session_id: str,
    transcripts_dir: str = "logs/transcripts",
    organisation_id: Optional[str] = None,
) -> Optional[Dict]:
    """
    Load a transcript from file.
    
    Args:
        session_id: Session identifier
        transcripts_dir: Base directory containing transcripts
        organisation_id: Optional org ID for path lookup
    
    Returns:
        Transcript dictionary or None if not found
    """
    try:
        # Sanitize before building a glob from client input — an id like
        # "../../x" must not escape the transcript root.
        session_id = safe_id(session_id)

        # Search for file
        matches = list(Path(transcripts_dir).glob(f"**/*{session_id}.json"))
        
        if not matches:
            logger.warning(f"Transcript not found: {session_id}")
            return None
        
        # Use first match
        with open(matches[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return None


def list_transcripts(
    transcripts_dir: str = "logs/transcripts",
    organisation_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """
    List available transcripts with filtering.
    
    Args:
        transcripts_dir: Base directory containing transcripts
        organisation_id: Filter by organization
        limit: Maximum number of transcripts to return
    
    Returns:
        List of transcript metadata dictionaries
    """
    try:
        base_path = Path(transcripts_dir)
        if not base_path.exists():
            return []
        
        # Build search pattern (org id is client input — sanitize)
        if organisation_id:
            search_path = base_path / safe_id(organisation_id)
            pattern = "*.json"
        else:
            search_path = base_path
            pattern = "**/*.json"
        
        if not search_path.exists():
            return []
        
        transcripts = []
        for file in sorted(search_path.glob(pattern), reverse=True)[:limit]:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcripts.append({
                        "filename": file.name,
                        "filepath": str(file.relative_to(base_path)),
                        "session_id": data.get("session_id"),
                        "start_time": data.get("start_time"),
                        "duration_seconds": data.get("metadata", {}).get("total_duration_seconds", 0),
                        "exchanges_count": data.get("metadata", {}).get("exchanges_count", 0),
                        "context": data.get("context", {}),
                    })
            except Exception as e:
                logger.warning(f"Error reading transcript {file}: {e}")
        
        return transcripts
    
    except Exception as e:
        logger.error(f"Failed to list transcripts: {e}")
        return []


def get_transcript_stats(
    transcripts_dir: str = "logs/transcripts",
) -> Dict[str, Any]:
    """
    Get aggregate statistics for all transcripts.
    
    Returns:
        Dictionary with transcript statistics
    """
    try:
        base_path = Path(transcripts_dir)
        if not base_path.exists():
            return {"total": 0, "by_organisation": {}}
        
        stats = {
            "total": 0,
            "by_organisation": {},
            "total_duration_seconds": 0,
            "total_exchanges": 0,
        }
        
        for file in base_path.glob("**/*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats["total"] += 1
                stats["total_duration_seconds"] += data.get("metadata", {}).get("total_duration_seconds", 0)
                stats["total_exchanges"] += data.get("metadata", {}).get("exchanges_count", 0)
                
                context = data.get("context", {})
                org_id = context.get("organisation_id", "unknown")
                
                stats["by_organisation"][org_id] = stats["by_organisation"].get(org_id, 0) + 1
                
            except Exception:
                continue
        
        return stats
    
    except Exception as e:
        logger.error(f"Failed to get transcript stats: {e}")
        return {"total": 0, "error": str(e)}
