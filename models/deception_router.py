"""
Deception Router — Adaptive Deception Defense Framework (ADDF)
Risk-based routing: diverts medium/high-risk sessions to decoy environment.
Pure logic, no LLM; keeps security path quick and fast.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

# Align with fix_agent_data.py RISK_LEVEL logic
MEDIUM_RISK_THRESHOLD = 20.0   # COMBINED_RISK_SCORE > 20 → consider diversion
HIGH_RISK_THRESHOLD = 50.0     # HIGH when > 50


@dataclass
class DeceptionSession:
    """Tracks a session that has been diverted to the deception grid."""
    session_id: str
    risk_score: float
    first_trigger: str          # transaction | login | chat
    first_trigger_time: datetime
    source: str = "transaction"  # first source that triggered


class DeceptionRouter:
    """
    In-memory router. First medium-or-higher risk for a session triggers
    diversion; thereafter that session always gets decoy responses.
    """

    def __init__(
        self,
        medium_threshold: float = MEDIUM_RISK_THRESHOLD,
        high_threshold: float = HIGH_RISK_THRESHOLD,
    ):
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self._diverted_sessions: Dict[str, DeceptionSession] = {}

    def should_divert(
        self,
        session_id: Optional[str],
        risk_score: float,
        source: str = "transaction",
    ) -> bool:
        """
        Returns True if this request should be diverted to decoy.
        First time risk_score is medium+ for this session → divert and remember.
        """
        if not session_id:
            return False
        if self.is_diverted(session_id):
            return True
        if risk_score >= self.medium_threshold:
            self._diverted_sessions[session_id] = DeceptionSession(
                session_id=session_id,
                risk_score=risk_score,
                first_trigger=source,
                first_trigger_time=datetime.utcnow(),
                source=source,
            )
            return True
        return False

    def is_diverted(self, session_id: Optional[str]) -> bool:
        """True if session is already in the deception grid."""
        if not session_id:
            return False
        return session_id in self._diverted_sessions

    def get_session(self, session_id: Optional[str]) -> Optional[DeceptionSession]:
        """Return deception session if diverted."""
        if not session_id:
            return None
        return self._diverted_sessions.get(session_id)

    def status(self, session_id: Optional[str]) -> dict:
        """For API: is_diverted, risk_score, first_trigger_time."""
        if not session_id:
            return {"is_diverted": False, "risk_score": 0.0, "first_trigger_time": None}
        s = self._diverted_sessions.get(session_id)
        if not s:
            return {"is_diverted": False, "risk_score": 0.0, "first_trigger_time": None}
        return {
            "is_diverted": True,
            "risk_score": s.risk_score,
            "first_trigger_time": s.first_trigger_time.isoformat(),
            "first_trigger": s.first_trigger,
        }
