"""
Veriscan — FastAPI Pydantic Schemas
Request/Response models for all REST endpoints.
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Fraud Prediction
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    """Input for single-transaction fraud prediction."""
    category: str = Field(..., example="shopping_net")
    amt: float = Field(..., example=499.99)
    gender: str = Field(..., example="M")
    state: str = Field(..., example="CA")
    merchant: str = Field(..., example="fraud_Kirlin and Sons")
    hour: int = Field(..., ge=0, le=23, example=14)
    day_of_week: int = Field(..., ge=0, le=6, example=3)


class FraudPredictionResponse(BaseModel):
    """Response from the fraud prediction endpoint."""
    risk_score: float
    risk_level: str
    feature_importances: Dict[str, float]


# ---------------------------------------------------------------------------
# High-Risk Transactions
# ---------------------------------------------------------------------------

class HighRiskTransaction(BaseModel):
    """A single high-risk transaction record."""
    data: Dict[str, Any]


class HighRiskTransactionsResponse(BaseModel):
    """Response from the high-risk transactions endpoint."""
    count: int
    transactions: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# User Risk Profile
# ---------------------------------------------------------------------------

class UserRiskResponse(BaseModel):
    """Response from the user risk profile endpoint."""
    user_id: str
    found: bool
    security_level: Optional[str] = None
    avg_risk: Optional[float] = None
    high_risk_count: Optional[int] = None
    txn_count: Optional[int] = None
    avg_combined_score: Optional[float] = None
    risk_distribution: Optional[Dict[str, int]] = None


# ---------------------------------------------------------------------------
# Shared Agent Models
# ---------------------------------------------------------------------------

class AgentActionStep(BaseModel):
    """A single step in the agent's reasoning trace."""
    step: int
    tool: str
    args: Dict[str, Any]
    result: Optional[str] = None


# ---------------------------------------------------------------------------
# RAG Query
# ---------------------------------------------------------------------------

class RAGQueryRequest(BaseModel):
    """Request for a RAG semantic search."""
    query: str = Field(..., example="identity theft")
    n_results: int = Field(default=5, ge=1, le=20)


class RAGResult(BaseModel):
    """A single RAG search result."""
    text: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


class RAGQueryResponse(BaseModel):
    """Response from the RAG query endpoint."""
    query: str
    count: int
    results: List[RAGResult]




# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str
    version: str
    services: Dict[str, str]


# ---------------------------------------------------------------------------
# Feature 1: AI Financial Advisor Chat
# ---------------------------------------------------------------------------

class AdvisorChatRequest(BaseModel):
    """Request for a natural-language financial question."""
    user_id: str = Field(..., example="USER_0001")
    message: str = Field(..., example="Am I spending more this month than last?")


class AdvisorChatResponse(BaseModel):
    """Response from the conversational financial advisor."""
    user_id: str
    message: str
    reply: str
    tool_results: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Feature 2: Spending DNA
# ---------------------------------------------------------------------------

class SpendingDNAResponse(BaseModel):
    """The 8-axis radar chart fingerprint for a user."""
    user_id: str
    radar_labels: List[str]
    radar_values: List[float]
    raw_axes: Dict[str, float]
    avg_trust_score: float
    avg_deviation: float
    anomalous_count: int
    total_sessions: int
    trust_grade: str
    time_preference: str


class DNACompareRequest(BaseModel):
    """Request to compare a session against a user's DNA baseline."""
    user_id: str = Field(..., example="USER_0001")
    session_overrides: Optional[Dict[str, float]] = None


class DNACompareResponse(BaseModel):
    """Comparison between the current session and the DNA baseline."""
    user_id: str
    baseline_radar: List[float]
    session_radar: List[float]
    radar_labels: List[str]
    axis_deviations: Dict[str, float]
    composite_deviation: float
    session_trust_score: float
    verdict: str




class SecurityChatRequest(BaseModel):
    """Request strictly for the Security Analyst."""
    message: str = Field(..., example="Scan my account for fraud anomalies.")
    session_id: Optional[str] = None


class SecurityChatResponse(BaseModel):
    """Response strictly from the Security Analyst."""
    reply: str
    actions: List[AgentActionStep]
    status: str
    session_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Feature: Proactive Deception Grid (ADDF)

class HoneypotLogEntry(BaseModel):
    """A log of attacker activity captured in the deception environment."""
    timestamp: str
    session_id: str
    action: str
    details: Any
    honeypot_signal: str


class DeceptionStatus(BaseModel):
    """Current state of the Deception Grid for a given session."""
    is_diverted: bool
    risk_score: float
    decoy_type: Optional[str] = None
    honeypot_logs: List[HoneypotLogEntry] = []


class TransactionValidateRequest(BaseModel):
    """Transaction validation with optional session for risk-based diversion."""
    session_id: Optional[str] = None
    category: str = Field(..., example="shopping_net")
    amt: float = Field(..., example=499.99)
    gender: str = Field(..., example="M")
    state: str = Field(..., example="CA")
    merchant: str = Field(..., example="Acme Corp")
    hour: int = Field(..., ge=0, le=23, example=14)
    day_of_week: int = Field(..., ge=0, le=6, example=3)


class TransactionValidateResponse(BaseModel):
    """Result of transaction validation; may be decoy when diverted."""
    authorized: bool
    risk_score: float
    diverted: bool = False
    decoy_txn_id: Optional[str] = None


class ThreatIntelLogEntry(BaseModel):
    """Structured threat intel log with tactic and FaaS indicators."""
    timestamp: str
    session_id: str
    action: str
    details: Any
    honeypot_signal: str
    tactic_category: Optional[str] = None
    faas_indicators: List[str] = []
    risk_score: float = 0.0


class DeceptionSessionStatus(BaseModel):
    """Session diversion status for API."""
    is_diverted: bool
    risk_score: float
    first_trigger_time: Optional[str] = None
    first_trigger: Optional[str] = None

