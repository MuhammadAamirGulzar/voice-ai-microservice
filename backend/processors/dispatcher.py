"""
Interview Dispatcher - Manages interview flow, question progression, and state
This is a utility module for interview state management.
The actual conversation flow is handled by the LLM with a carefully crafted system prompt.
"""
from loguru import logger
from typing import Optional, List, Dict
from datetime import datetime
import json
from pathlib import Path


class InterviewState:
    """Tracks interview session state for logging and analytics."""
    
    def __init__(self, session_id: str, questions: List[str], resume: Dict, custom_instructions: str):
        self.session_id = session_id
        self.questions = questions
        self.resume = resume
        self.custom_instructions = custom_instructions
        
        # Question management
        self.current_question_index = -1
        self.asked_questions = []
        
        # Conversation tracking
        self.conversation_history = []
        self.start_time = datetime.now()
        self.last_activity_time = datetime.now()
        
        # Status flags
        self.is_greeting_done = False
        self.is_interview_complete = False
        
        logger.info(f"Interview state initialized: {session_id}")
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_activity_time = datetime.now()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in minutes."""
        return (datetime.now() - self.start_time).total_seconds() / 60
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "duration_minutes": round(self.get_elapsed_time(), 2),
            "questions_count": len(self.questions),
            "asked_questions": len(self.asked_questions),
            "conversation_history": self.conversation_history,
            "is_complete": self.is_interview_complete,
            "start_time": self.start_time.isoformat(),
            "last_activity": self.last_activity_time.isoformat()
        }


class InterviewManager:
    """
    Manages interview sessions and logging.
    Used for tracking and analytics, not for controlling conversation flow.
    """
    
    def __init__(self, output_dir: str = "logs/transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions: Dict[str, InterviewState] = {}
        
        logger.info(f"InterviewManager initialized, output_dir: {output_dir}")
    
    def create_session(
        self,
        session_id: str,
        questions: List[str],
        resume: Dict,
        custom_instructions: str
    ) -> InterviewState:
        """Create a new interview session."""
        state = InterviewState(
            session_id=session_id,
            questions=questions,
            resume=resume,
            custom_instructions=custom_instructions
        )
        self.active_sessions[session_id] = state
        logger.info(f"Created interview session: {session_id}")
        return state
    
    def get_session(self, session_id: str) -> Optional[InterviewState]:
        """Get an existing session."""
        return self.active_sessions.get(session_id)
    
    def end_session(self, session_id: str) -> Optional[dict]:
        """End a session and save transcript."""
        state = self.active_sessions.pop(session_id, None)
        if state:
            state.is_interview_complete = True
            self._save_transcript(state)
            logger.info(f"Ended interview session: {session_id}")
            return state.to_dict()
        return None
    
    def _save_transcript(self, state: InterviewState):
        """Save interview transcript to JSON file."""
        try:
            transcript = {
                "session_id": state.session_id,
                "start_time": state.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_minutes": round(state.get_elapsed_time(), 2),
                "metadata": {
                    "candidate_name": state.resume.get("name", "Unknown"),
                    "questions_asked": len(state.asked_questions),
                    "total_exchanges": len(state.conversation_history),
                },
                "conversation": state.conversation_history,
                "questions": state.questions,
            }
            
            filename = f"interview_{state.session_id}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transcript, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcript saved: {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")


def build_interview_system_prompt(
    custom_instructions: str,
    resume: dict,
    questions: list
) -> str:
    """
    Build the system prompt for the interviewer AI.
    This is the main way to control the interview flow.
    """
    
    # Format questions for the prompt
    questions_text = "\n".join([f"- {q}" for q in questions])
    
    # Format resume info
    candidate_name = resume.get("name", "the candidate")
    skills = ", ".join(resume.get("skills", []))
    
    experience_text = ""
    for exp in resume.get("experience", []):
        experience_text += f"- {exp.get('role', 'Role')} at {exp.get('company', 'Company')} ({exp.get('years', 0)} years)\n"
    
    system_prompt = f"""You are a professional AI recruiter conducting a voice interview. You should be warm, professional, and conversational.

{custom_instructions}

## CANDIDATE INFORMATION
Name: {candidate_name}
Skills: {skills}
Experience:
{experience_text}

## INTERVIEW QUESTIONS TO COVER
{questions_text}

## GUIDELINES
1. Start with a warm greeting and introduce yourself
2. Ask one question at a time and wait for the response
3. Listen carefully and ask relevant follow-up questions
4. Be conversational - don't sound robotic
5. Keep responses concise - this is a voice conversation
6. If the candidate seems stuck, offer gentle encouragement
7. After covering all questions, thank them and conclude

## IMPORTANT
- Speak naturally as this is a voice interview
- Keep responses brief (2-3 sentences typically)
- Use conversational language
- React naturally to what the candidate says
- Start by greeting the candidate and asking them to introduce themselves"""

    return system_prompt


# Global interview manager instance
_interview_manager: Optional[InterviewManager] = None


def get_interview_manager() -> InterviewManager:
    """Get or create the global interview manager."""
    global _interview_manager
    if _interview_manager is None:
        _interview_manager = InterviewManager()
    return _interview_manager