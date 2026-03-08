"""
Honeypot Agent — Adaptive Deception Defense Framework (ADDF)
Orchestrates synthetic environments and decoy data. Uses fast, separate AI model
for deception so security path stays quick and does not block main GuardAgent.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Optional: use FastDeceptionLLM for quick decoy text; no dependency on heavy LocalLLM
try:
    from models.local_llm import FastDeceptionLLM
except Exception:
    FastDeceptionLLM = None

# Tactic categories for threat intel (rule-based for speed)
TACTIC_RECON_CHAT = "RECONNAISSANCE_CHAT"
TACTIC_CREDENTIAL_PROBE = "CREDENTIAL_PROBE"
TACTIC_DATA_EXFIL = "DATA_EXFILTRATION_ATTEMPT"
TACTIC_AUDIT_EXPLORATION = "AUDIT_EXPLORATION"
TACTIC_DECOY_USER = "DECOY_USER_PROFILE_ACCESS"
TACTIC_DECOY_TXN = "DECOY_TXN_ACCESS"

# Fast template responses when LLM disabled — keeps security path instant
DECEPTION_TEMPLATES = [
    "Your request has been logged. For elevated access, our team may contact you after verifying audit logs.",
    "Temporary access tokens are issued via the internal audit portal. Please allow 24–48 hours for provisioning.",
    "We've noted your inquiry. Additional security checks are in progress; you'll receive an update shortly.",
]


def _classify_tactic(action: str, details: Any) -> str:
    """Rule-based tactic classification — no LLM, fast."""
    action_lower = (action or "").lower()
    details_str = str(details).lower() if details else ""
    combined = f"{action_lower} {details_str}"
    if "audit" in combined or "internal" in combined:
        return TACTIC_AUDIT_EXPLORATION
    if "user" in action_lower and "risk" in action_lower:
        return TACTIC_DECOY_USER
    if "high_risk" in action_lower or "transaction" in action_lower:
        return TACTIC_DECOY_TXN
    if any(x in combined for x in ["password", "credential", "token", "bypass", "root", "auth"]):
        return TACTIC_CREDENTIAL_PROBE
    if any(x in combined for x in ["export", "dump", "database", "exfiltr"]):
        return TACTIC_DATA_EXFIL
    if "chat" in action_lower or "security" in action_lower:
        return TACTIC_RECON_CHAT
    return TACTIC_RECON_CHAT


def _detect_faas_pattern(session_logs: List[Dict]) -> List[str]:
    """Detect scripted/FaaS behavior from recent logs — rule-based, fast."""
    indicators = []
    if len(session_logs) >= 5:
        indicators.append("rapid_fire_requests")
    if len(session_logs) >= 3:
        actions = [l.get("action", "") for l in session_logs[-10:]]
        if len(set(actions)) <= 2 and len(actions) >= 3:
            indicators.append("scripted_pattern")
    return indicators


class HoneypotAgent:
    """
    Generates decoy data and threat intel. Uses optional FastDeceptionLLM so
    security path is quick; falls back to templates when LLM disabled.
    """

    def __init__(self, fast_llm: Optional[Any] = None):
        self.fast_llm = fast_llm
        if fast_llm is None and FastDeceptionLLM is not None:
            try:
                self.fast_llm = FastDeceptionLLM()
            except Exception:
                self.fast_llm = None
        self.logs: List[Dict[str, Any]] = []
        self._log_by_session: Dict[str, List[Dict]] = {}

    def generate_decoy_filesystem(self, depth: int = 3, seed: Optional[int] = None) -> Dict[str, Any]:
        """Realistic decoy filesystem. Uses static fallback for speed; optional LLM if needed."""
        if seed is not None:
            random.seed(seed)
        if self.fast_llm and getattr(self.fast_llm, "model_id", None):
            prompt = f"JSON of a nested filesystem for a financial server. Depth {depth}. Only JSON."
            try:
                out = self.fast_llm.generate(prompt, max_tokens=400, temp=0.5)
                clean = out.strip("`").replace("json\n", "").strip()
                return json.loads(clean)
            except Exception:
                pass
        return {
            "root": {
                "etc": {"config.yaml": "[ENCRYPTED]", "passwd.bak": "root:*:0:0..."},
                "var": {"log": {"audit.log": "Access denied at 2026-03-07...", "secure": "..."}},
                "home": {"admin": {".ssh": {"id_rsa": "-----BEGIN RSA PRIVATE KEY-----"}}},
                "opt": {"db": {"users_v4.sqlite": "[BINARY_DATA]"}},
            }
        }

    def generate_synthetic_transactions(
        self, user_id: str, count: int = 5
    ) -> List[Dict[str, Any]]:
        """Fast schema-mirror decoy transactions (no LLM in path)."""
        categories = ["shopping_net", "travel", "entertainment", "health", "transfer"]
        txns = []
        now = datetime.now()
        for i in range(count):
            txns.append({
                "TRANSACTION_ID": f"DEC-TXN-{random.randint(100000, 999999)}",
                "USER_ID": user_id,
                "CATEGORY": random.choice(categories),
                "MERCHANT": f"Decoy_{random.choice(['AWS', 'Stripe', 'Oracle'])}",
                "AMOUNT": round(random.uniform(10.0, 5000.0), 2),
                "COMBINED_RISK_SCORE": 5.0,
                "RISK_LEVEL": "LOW",
                "timestamp": (now - timedelta(minutes=random.randint(1, 1440))).isoformat(),
                "status": "authorized_decoy",
            })
        return txns

    def generate_synthetic_account(self, user_id: str) -> Dict[str, Any]:
        """Decoy user risk profile matching tool_get_user_risk_profile schema — fast, no LLM."""
        return {
            "user_id": user_id,
            "found": True,
            "security_level": "Standard",
            "avg_risk": round(random.uniform(0.05, 0.15), 3),
            "high_risk_count": 0,
            "txn_count": random.randint(5, 30),
            "avg_combined_score": round(random.uniform(5.0, 18.0), 2),
            "risk_distribution": {"LOW": 8, "MEDIUM": 2, "HIGH": 0, "CRITICAL": 0},
        }

    def generate_honeytoken(self, token_type: str = "api_key") -> Dict[str, str]:
        """Fake credentials; session type uses non-static format."""
        if token_type == "session":
            return {
                "session_token": f"sess_decoy_{random.getrandbits(96):024x}",
                "expires": (datetime.utcnow() + timedelta(hours=1)).isoformat() + "Z",
            }
        if token_type == "api_key":
            return {"key": f"sk_live_{random.getrandbits(128):x}", "usage": "Production Vault Access"}
        if token_type == "db_creds":
            return {
                "host": "internal-db-01.prod.lan",
                "user": "readonly_audit",
                "pass": f"P@ss_{random.getrandbits(48):012x}!",
            }
        return {"token": "unknown"}

    def log_interaction(
        self,
        session_id: str,
        action: str,
        details: Any,
        risk_score: float = 0.0,
        honeypot_signal: str = "HIGH_CONFIDENCE_ATTACKER",
    ) -> None:
        """Structured threat intel log with tactic and FaaS fields."""
        tactic = _classify_tactic(action, details)
        session_logs = self._log_by_session.get(session_id, [])
        session_logs.append({"action": action, "details": details})
        self._log_by_session[session_id] = session_logs[-20:]  # keep last 20
        faas = _detect_faas_pattern(session_logs)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "action": action,
            "details": details,
            "honeypot_signal": honeypot_signal,
            "tactic_category": tactic,
            "faas_indicators": faas,
            "risk_score": risk_score,
        }
        self.logs.append(entry)
        print(f"🕵️ Honeypot [{tactic}] Session {session_id}: {action}")

    def get_deception_response(self, original_query: str) -> str:
        """Fast path: template first; if fast LLM loaded, use it for variety."""
        if self.fast_llm and getattr(self.fast_llm, "model_id", None):
            prompt = (
                f"User asked: '{original_query[:200]}'. Reply as support: sound legitimate, "
                "mention 'audit files' or 'temporary tokens'. One short paragraph."
            )
            out = self.fast_llm.generate(prompt, max_tokens=120, temp=0.7)
            if out and len(out.strip()) > 20:
                return out.strip()
        return random.choice(DECEPTION_TEMPLATES)

    def get_deception_advisor_reply(self, message: str, user_id: str) -> str:
        """Decoy reply for Financial Advisor when session is diverted (ADDF covers entire system)."""
        if self.fast_llm and getattr(self.fast_llm, "model_id", None):
            prompt = (
                f"User {user_id} asked about finances: '{message[:200]}'. "
                "Reply as a financial assistant: generic, reassuring, one short paragraph. No real data."
            )
            out = self.fast_llm.generate(prompt, max_tokens=150, temp=0.5)
            if out and len(out.strip()) > 20:
                return out.strip()
        return (
            "Your request has been received. Our team is reviewing your account. "
            "You will receive a detailed summary via secure message within 24–48 hours."
        )

    def generate_decoy_dna_profile(self, user_id: str) -> Dict[str, Any]:
        """Decoy Spending DNA profile for diverted sessions (ADDF)."""
        labels = ["Savings", "Travel", "Dining", "Shopping", "Subscriptions", "Utilities", "Health", "Other"]
        values = [round(random.uniform(0.3, 0.9), 3) for _ in range(8)]
        return {
            "user_id": user_id,
            "radar_labels": labels,
            "radar_values": values,
            "raw_axes": dict(zip([l.lower() for l in labels], values)),
            "avg_trust_score": round(random.uniform(0.6, 0.85), 3),
            "avg_deviation": round(random.uniform(0.05, 0.2), 3),
            "anomalous_count": random.randint(0, 3),
            "total_sessions": random.randint(10, 50),
            "trust_grade": random.choice(["B", "B+", "A-"]),
            "time_preference": random.choice(["morning", "afternoon", "evening"]),
        }

    def generate_decoy_dna_compare(self, user_id: str) -> Dict[str, Any]:
        """Decoy DNA compare response for diverted sessions (ADDF)."""
        labels = ["Savings", "Travel", "Dining", "Shopping", "Subscriptions", "Utilities", "Health", "Other"]
        baseline = [round(random.uniform(0.4, 0.9), 3) for _ in range(8)]
        session = [round(b + random.uniform(-0.1, 0.1), 3) for b in baseline]
        deviations = {labels[i]: round(session[i] - baseline[i], 3) for i in range(8)}
        return {
            "user_id": user_id,
            "baseline_radar": baseline,
            "session_radar": session,
            "radar_labels": labels,
            "axis_deviations": deviations,
            "composite_deviation": round(random.uniform(0.02, 0.15), 3),
            "session_trust_score": round(random.uniform(0.65, 0.88), 3),
            "verdict": random.choice(["Normal", "Slight variance", "Within range"]),
        }

    def get_threat_intel_export(self) -> List[Dict[str, Any]]:
        """Export logs with tactic_category and faas_indicators for SIEM."""
        return list(self.logs)
