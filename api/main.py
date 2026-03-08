"""
Veriscan — FastAPI Microservices Backend
Decoupled REST API for Fraud Prediction, GuardAgent, and RAG Engine.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
# ---------------------------------------------------------------------------
# System Stability Guards (Fixes SIGABRT on macOS Sequoia)
# ---------------------------------------------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
import anyio

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Request

from api.schemas import (
    HighRiskTransactionsResponse,
    UserRiskResponse,
    AgentActionStep,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    HealthResponse,
    AdvisorChatRequest,
    AdvisorChatResponse,
    SpendingDNAResponse,
    DNACompareRequest,
    DNACompareResponse,
    SecurityChatRequest,
    SecurityChatResponse,
    TransactionValidateRequest,
    TransactionValidateResponse,
    ThreatIntelLogEntry,
    DeceptionSessionStatus,
)

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
_agent = None
_rag_engine = None
_advisor_agent = None
_dna_agent = None
_router = None
_honeypot = None

# Fast path: attack keywords for immediate diversion (no LLM)
DECEPTION_KEYWORDS = ("bypass", "root", "database", "dump", "password", "credential", "token", "audit file", "internal")


def _session_id(request: Request, body_session_id: Optional[str] = None) -> Optional[str]:
    """Session ID from body, X-Session-ID header, or session_id query — for ADDF routing."""
    return body_session_id or request.headers.get("X-Session-ID") or request.query_params.get("session_id")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    Agents are loaded as singletons to avoid redundant data reloading.
    """
    global _agent, _rag_engine, _advisor_agent, _dna_agent, _router, _honeypot
    print("🚀 Veriscan API — Loading resources...")

    # 1. RAG Engine
    try:
        from models.rag_engine_local import RAGEngineLocal
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
        print("✅ RAG Engine loaded.")
    except Exception as e:
        print(f"⚠️  RAG Engine failed: {e}")

    # 2. GuardAgent (MLX)
    try:
        from models.guard_agent_local import LocalGuardAgent
        _agent = LocalGuardAgent()
        print("✅ GuardAgent (Security Analyst) loaded.")
    except Exception as e:
        print(f"ℹ️  GuardAgent not loaded: {e}")

    # 3. Financial Advisor Agent (CSV-heavy)
    try:
        from agents.financial_advisor_agent import FinancialAdvisorAgent
        _advisor_agent = FinancialAdvisorAgent()
        # Trigger lazy data loading at startup
        _ = _advisor_agent.df
        print("✅ Financial Advisor Agent loaded.")
    except Exception as e:
        print(f"⚠️  Financial Advisor failed: {e}")

    # 4. Spending DNA Agent
    try:
        from agents.spending_dna_agent import SpendingDNAAgent
        _dna_agent = SpendingDNAAgent()
        print("✅ Spending DNA Agent loaded.")
    except Exception as e:
        print(f"⚠️  DNA Agent failed: {e}")

    # 5. ADDF: Deception Router + Honeypot (fast path; optional fast LLM)
    try:
        from models.deception_router import DeceptionRouter
        from agents.honeypot_agent import HoneypotAgent
        _router = DeceptionRouter()
        _honeypot = HoneypotAgent()  # uses FastDeceptionLLM if available, else templates
        print("✅ ADDF (Deception Grid) loaded.")
    except Exception as e:
        print(f"⚠️  ADDF failed: {e}")

    print("🟢 Veriscan API is ready.")
    yield
    print("🔴 Veriscan API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Veriscan — Fraud Intelligence API",
    description="Microservices backend for ML fraud prediction, agentic investigation, and RAG-powered knowledge retrieval.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit and other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 1. Health Check
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Return API health status and loaded services."""
    return HealthResponse(
        status="operational",
        version="2.0.0",
        services={
            "guard_agent": "loaded" if _agent else "unavailable",
            "rag_engine": "loaded" if _rag_engine else "unavailable",
            "advisor_agent": "loaded" if _advisor_agent else "unavailable",
            "dna_agent": "loaded" if _dna_agent else "unavailable",
            "addf": "loaded" if _router and _honeypot else "unavailable",
        },
    )




# ---------------------------------------------------------------------------
# 3. High-Risk Transactions (session-aware: diverted → decoy)
# ---------------------------------------------------------------------------
@app.get("/api/fraud/high-risk", response_model=HighRiskTransactionsResponse, tags=["Fraud ML"])
async def get_high_risk_transactions(request: Request, limit: int = Query(default=10, ge=1, le=100), session_id: Optional[str] = Query(None)):
    """Get the top N highest-risk transactions. Diverted sessions get decoy list."""
    sid = _session_id(request, session_id)
    if _router and _honeypot and sid and _router.is_diverted(sid):
        decoy = _honeypot.generate_synthetic_transactions("DECOY_USER", count=min(limit, 20))
        _honeypot.log_interaction(sid, "DECOY_TXN_ACCESS", {"limit": limit}, risk_score=0)
        return HighRiskTransactionsResponse(count=len(decoy), transactions=decoy)
    from models.agent_tools_data import tool_get_high_risk_transactions
    results = tool_get_high_risk_transactions(limit=limit)
    return HighRiskTransactionsResponse(count=len(results), transactions=results)


# ---------------------------------------------------------------------------
# 4. User Risk Profile (session-aware: diverted → decoy)
# ---------------------------------------------------------------------------
@app.get("/api/user/{user_id}/risk", response_model=UserRiskResponse, tags=["User Intelligence"])
async def get_user_risk(user_id: str, request: Request, session_id: Optional[str] = Query(None)):
    """Retrieve the risk profile for a specific user. Diverted sessions get decoy profile."""
    sid = _session_id(request, session_id)
    if _router and _honeypot and sid and _router.is_diverted(sid):
        decoy = _honeypot.generate_synthetic_account(user_id)
        _honeypot.log_interaction(sid, "DECOY_USER_PROFILE_ACCESS", {"user_id": user_id}, risk_score=0)
        return UserRiskResponse(**decoy)
    from models.agent_tools_data import tool_get_user_risk_profile
    result = tool_get_user_risk_profile(user_id)
    return UserRiskResponse(**result)




# ---------------------------------------------------------------------------
# 6. RAG Query
# ---------------------------------------------------------------------------
@app.post("/api/rag/query", response_model=RAGQueryResponse, tags=["Knowledge Base"])
async def rag_query(req: RAGQueryRequest):
    """Perform a semantic search over the local knowledge base."""
    if not _rag_engine:
        raise HTTPException(status_code=503, detail="RAG engine not loaded.")

    results = _rag_engine.query(req.query, n_results=req.n_results)

    parsed = [
        RAGResult(
            text=r["text"],
            confidence=r["confidence"],
            metadata=r.get("metadata"),
        )
        for r in results
    ]

    return RAGQueryResponse(query=req.query, count=len(parsed), results=parsed)


# ---------------------------------------------------------------------------
# ADDF: Transaction validate (risk-based diversion)
# ---------------------------------------------------------------------------
@app.post("/api/transactions/validate", response_model=TransactionValidateResponse, tags=["ADDF"])
async def validate_transaction(req: TransactionValidateRequest):
    """Validate transaction; medium+ risk triggers diversion for this session."""
    from models.agent_tools_data import score_transaction
    risk = score_transaction(req.category, req.amt, req.merchant, req.hour, req.day_of_week)
    diverted = False
    decoy_txn_id = None
    if _router and req.session_id:
        diverted = _router.should_divert(req.session_id, risk, "transaction")
        if diverted and _honeypot:
            import random
            decoy_txn_id = f"DEC-TXN-{random.randint(100000, 999999)}"
            _honeypot.log_interaction(req.session_id, "DECOY_TXN_ACCESS", {"decoy_txn_id": decoy_txn_id}, risk_score=risk)
    authorized = (risk < 50.0) if not diverted else True
    return TransactionValidateResponse(authorized=authorized, risk_score=risk, diverted=diverted, decoy_txn_id=decoy_txn_id)


@app.get("/api/honeypot/threat-intel", tags=["ADDF"])
async def get_threat_intel():
    if not _honeypot:
        return {"logs": []}
    return {"logs": _honeypot.get_threat_intel_export()}


@app.get("/api/decoy/internal/audit", tags=["ADDF"])
async def decoy_internal_audit(request: Request, session_id: Optional[str] = Query(None)):
    sid = _session_id(request, session_id)
    if _honeypot and sid:
        _honeypot.log_interaction(sid, "AUDIT_EXPLORATION", {"endpoint": "decoy/internal/audit"}, risk_score=0)
    return {"entries": [{"ts": "2026-03-07T10:00:00Z", "level": "INFO", "msg": "Access granted (staging)."}]}


@app.get("/api/honeypot/filesystem", tags=["ADDF"])
async def honeypot_filesystem(request: Request, depth: int = Query(2, ge=1, le=5), session_id: Optional[str] = Query(None)):
    sid = _session_id(request, session_id)
    if _honeypot:
        if sid:
            _honeypot.log_interaction(sid, "DECOY_FILESYSTEM", {"depth": depth}, risk_score=0)
        return _honeypot.generate_decoy_filesystem(depth=depth)
    return {"root": {}}


@app.get("/api/honeypot/transactions/{user_id}", tags=["ADDF"])
async def honeypot_transactions(user_id: str, request: Request, count: int = Query(5, ge=1, le=20), session_id: Optional[str] = Query(None)):
    sid = _session_id(request, session_id)
    if _honeypot:
        if sid:
            _honeypot.log_interaction(sid, "DECOY_TXN_ACCESS", {"user_id": user_id}, risk_score=0)
        return _honeypot.generate_synthetic_transactions(user_id, count=count)
    return []


@app.get("/api/honeypot/logs", tags=["ADDF"])
async def honeypot_logs():
    if not _honeypot:
        return {"logs": []}
    return {"logs": _honeypot.logs}


@app.get("/api/deception/status", response_model=DeceptionSessionStatus, tags=["ADDF"])
async def deception_status(request: Request, session_id: Optional[str] = Query(None)):
    sid = _session_id(request, session_id)
    if not _router:
        return DeceptionSessionStatus(is_diverted=False, risk_score=0.0)
    return DeceptionSessionStatus(**_router.status(sid))


# ===========================================================================
# NEW FEATURE ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature 1: AI Financial Advisor Chat
# ---------------------------------------------------------------------------
@app.post("/api/advisor/chat", response_model=AdvisorChatResponse, tags=["AI Financial Advisor"])
async def advisor_chat(req: AdvisorChatRequest):
    """Conversational financial advisor — answers natural-language questions about spending."""
    if not _advisor_agent:
        raise HTTPException(status_code=503, detail="Financial Advisor not loaded.")
    result = _advisor_agent.chat(req.message, req.user_id)
    return AdvisorChatResponse(
        user_id=req.user_id,
        message=req.message,
        reply=result.get("reply", ""),
        tool_results=result.get("tool_results", []),
    )


@app.get("/api/advisor/users", tags=["AI Financial Advisor"])
async def advisor_users():
    """Return all user IDs in the financial advisor dataset."""
    if not _advisor_agent:
        raise HTTPException(status_code=503, detail="Financial Advisor not loaded.")
    return {"users": _advisor_agent.get_all_users()}


# ---------------------------------------------------------------------------
# Feature 2: AI Security Analyst Chat (risk-based + keyword diversion → fast decoy)
# ---------------------------------------------------------------------------
@app.post("/api/security/chat", response_model=SecurityChatResponse, tags=["AI Security Analyst"])
async def security_chat(req: SecurityChatRequest):
    """Security analyst; diverted or attack-keyword sessions get fast decoy response (other AI model)."""
    msg_lower = (req.message or "").lower()
    use_deception = False
    if _router and _honeypot and req.session_id:
        if _router.is_diverted(req.session_id):
            use_deception = True
        elif any(k in msg_lower for k in DECEPTION_KEYWORDS):
            _router.should_divert(req.session_id, 25.0, "chat")
            _honeypot.log_interaction(req.session_id, "RECONNAISSANCE_CHAT", {"message": req.message[:200]}, risk_score=25.0)
            use_deception = True
    if use_deception and _honeypot:
        reply = await anyio.to_thread.run_sync(_honeypot.get_deception_response, req.message)
        return SecurityChatResponse(reply=reply, actions=[], status="deception", session_id=req.session_id)

    if not _agent:
        raise HTTPException(status_code=503, detail="GuardAgent not loaded.")
    try:
        system_prompt = (
            "You are an elite AI Security Analyst. Your job is strictly to analyze "
            "security data, explain fraud risks, and provide safety protocols. "
            "Never provide financial advice, budgets, or savings plans.\n\n"
            "Format your response professionally:\n"
            "1. Be extremely concise (under 200 words max).\n"
            "2. Use bullet points and bold text for key findings.\n"
            "3. Provide direct, actionable safety advice without unnecessary filler text."
        )
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{req.message}<|im_end|>\n<|im_start|>assistant\n"
        # Slightly lower max_tokens for faster replies; model size controlled by VERISCAN_FAST_MODE/VERISCAN_LLM_MODEL.
        reply = await _agent.llm.generate_async(prompt, max_tokens=140, temp=0.2)
        return SecurityChatResponse(reply=reply, actions=[], status="completed", session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Feature 3: Spending DNA
# ---------------------------------------------------------------------------
@app.get("/api/dna/profile/{user_id}", response_model=SpendingDNAResponse, tags=["Spending DNA"])
async def get_dna_profile(user_id: str):
    """Return the 8-axis Spending DNA radar chart fingerprint for a user."""
    if not _dna_agent:
        raise HTTPException(status_code=503, detail="DNA Agent not loaded.")
    result = _dna_agent.compute_dna(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return SpendingDNAResponse(**result)


@app.post("/api/dna/compare", response_model=DNACompareResponse, tags=["Spending DNA"])
async def compare_dna(req: DNACompareRequest):
    """Compare a new session against the user's DNA baseline."""
    if not _dna_agent:
        raise HTTPException(status_code=503, detail="DNA Agent not loaded.")
    result = _dna_agent.compare_session(req.user_id, session_overrides=req.session_overrides)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return DNACompareResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
