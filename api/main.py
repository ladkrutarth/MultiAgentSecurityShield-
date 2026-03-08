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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on the path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import (
    HighRiskTransactionsResponse,
    UserRiskResponse,
    AgentActionStep,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    HealthResponse,
    # Feature: Financial Advisor
    AdvisorChatRequest,
    AdvisorChatResponse,
    # Feature: Spending DNA
    SpendingDNAResponse,
    DNACompareRequest,
    DNACompareResponse,
    SecurityChatRequest,
    SecurityChatResponse,
)


# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
_agent = None
_rag_engine = None
_advisor_agent = None
_dna_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    Agents are loaded as singletons to avoid redundant data reloading.
    """
    global _agent, _rag_engine, _advisor_agent, _dna_agent
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
        },
    )




# ---------------------------------------------------------------------------
# 3. High-Risk Transactions
# ---------------------------------------------------------------------------
@app.get("/api/fraud/high-risk", response_model=HighRiskTransactionsResponse, tags=["Fraud ML"])
async def get_high_risk_transactions(limit: int = Query(default=10, ge=1, le=100)):
    """Get the top N highest-risk transactions across the system."""
    from models.agent_tools_data import tool_get_high_risk_transactions

    results = tool_get_high_risk_transactions(limit=limit)
    return HighRiskTransactionsResponse(count=len(results), transactions=results)


# ---------------------------------------------------------------------------
# 4. User Risk Profile
# ---------------------------------------------------------------------------
@app.get("/api/user/{user_id}/risk", response_model=UserRiskResponse, tags=["User Intelligence"])
async def get_user_risk(user_id: str):
    """Retrieve the risk profile for a specific user."""
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
# Feature 2: AI Security Analyst Chat
# ---------------------------------------------------------------------------
@app.post("/api/security/chat", response_model=SecurityChatResponse, tags=["AI Security Analyst"])
async def security_chat(req: SecurityChatRequest):
    """Conversational security analyst — answers questions about fraud, anomalies, and safety."""
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
        
        reply = await _agent.llm.generate_async(prompt, max_tokens=200, temp=0.2)
        
        return SecurityChatResponse(
            reply=reply,
            actions=[],
            status="completed",
            session_id=req.session_id
        )
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
