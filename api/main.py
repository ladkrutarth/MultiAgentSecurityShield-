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
    AgentInvestigationRequest,
    AgentInvestigationResponse,
    AgentActionStep,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    LLMGenerationRequest,
    LLMGenerationResponse,
    HealthResponse,
    # Feature: Financial Advisor
    AdvisorChatRequest,
    AdvisorChatResponse,
    # Feature: Spending DNA
    SpendingDNAResponse,
    DNACompareRequest,
    DNACompareResponse,
)


# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
_agent = None
_rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    GuardAgent (MLX) is skipped at startup to avoid Metal crash when GPU unavailable.
    """
    print("🚀 Veriscan API — Loading resources...")

    # 2. RAG Engine (no MLX — safe to load)
    try:
        from models.rag_engine_local import RAGEngineLocal
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
        print("✅ RAG Engine loaded and indexed.")
    except Exception as e:
        print(f"⚠️  RAG Engine failed to load: {e}")
        _rag_engine = None

    # 3. GuardAgent (uses MLX — try to load on Apple Silicon; skip if Metal unavailable)
    try:
        from models.guard_agent_local import LocalGuardAgent
        _agent = LocalGuardAgent()
        print("✅ GuardAgent loaded.")
    except Exception as e:
        _agent = None
        print(f"ℹ️  GuardAgent not loaded: {e}")
        print("   Use /api/rag/query for knowledge search, or run on Apple Silicon for full agent.")

    print("🟢 Veriscan API is ready.")
    yield  # App is running

    # Shutdown cleanup (if needed)
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
# 5. Agent Investigation
# ---------------------------------------------------------------------------
@app.post("/api/agent/investigate", response_model=AgentInvestigationResponse, tags=["Agentic AI"])
async def agent_investigate(req: AgentInvestigationRequest):
    """Run a full GuardAgent agentic investigation."""
    if not _agent:
        raise HTTPException(
            status_code=503,
            detail=(
                "GuardAgent not loaded. It requires Apple Silicon (M1/M2/M3) and MLX. "
                "Use the RAG Knowledge Search tab for semantic search, or run the API on a Mac with Apple Silicon to enable agent investigations."
            ),
        )

    raw = _agent.analyze(req.query, session_id=req.session_id)

    actions = []
    for a in raw.get("actions", []):
        actions.append(
            AgentActionStep(
                step=a.get("step", 0),
                tool=a.get("tool", "unknown"),
                args=a.get("args", {}),
                result=a.get("result"),
            )
        )

    return AgentInvestigationResponse(
        answer=raw.get("answer", "No answer produced."),
        actions=actions,
        status=raw.get("status", "unknown"),
        session_id=raw.get("session_id"),
        trace=raw.get("trace")
    )


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
# 7. Simple LLM Generation
# ---------------------------------------------------------------------------
@app.post("/api/llm/generate", response_model=LLMGenerationResponse, tags=["AI Core"])
async def generate_text(req: LLMGenerationRequest):
    """Perform raw text generation using the local MLX LLM."""
    if not _agent:
        raise HTTPException(status_code=503, detail="Local LLM / GuardAgent not loaded.")

    # Access the underlying LocalLLM from the agent
    try:
        response = _agent.llm.generate(req.prompt, max_tokens=req.max_tokens, temp=req.temp)
        return LLMGenerationResponse(prompt=req.prompt, response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


# ===========================================================================
# NEW FEATURE ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# Feature 1: AI Financial Advisor Chat
# ---------------------------------------------------------------------------
@app.post("/api/advisor/chat", response_model=AdvisorChatResponse, tags=["AI Financial Advisor"])
async def advisor_chat(req: AdvisorChatRequest):
    """Conversational financial advisor — answers natural-language questions about spending."""
    from agents.financial_advisor_agent import FinancialAdvisorAgent
    agent = FinancialAdvisorAgent()
    result = agent.chat(req.message, req.user_id)
    return AdvisorChatResponse(
        user_id=req.user_id,
        message=req.message,
        reply=result.get("reply", ""),
        tool_results=result.get("tool_results", []),
    )


@app.get("/api/advisor/users", tags=["AI Financial Advisor"])
async def advisor_users():
    """Return all user IDs in the financial advisor dataset."""
    from agents.financial_advisor_agent import FinancialAdvisorAgent
    return {"users": FinancialAdvisorAgent().get_all_users()}


# ---------------------------------------------------------------------------
# Feature 2: Spending DNA
# ---------------------------------------------------------------------------
@app.get("/api/dna/profile/{user_id}", response_model=SpendingDNAResponse, tags=["Spending DNA"])
async def get_dna_profile(user_id: str):
    """Return the 8-axis Spending DNA radar chart fingerprint for a user."""
    from agents.spending_dna_agent import SpendingDNAAgent
    agent = SpendingDNAAgent()
    result = agent.compute_dna(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return SpendingDNAResponse(**result)


@app.post("/api/dna/compare", response_model=DNACompareResponse, tags=["Spending DNA"])
async def compare_dna(req: DNACompareRequest):
    """Compare a new session against the user's DNA baseline."""
    from agents.spending_dna_agent import SpendingDNAAgent
    agent = SpendingDNAAgent()
    result = agent.compare_session(req.user_id, session_overrides=req.session_overrides)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return DNACompareResponse(**result)


@app.get("/api/dna/users", tags=["Spending DNA"])
async def dna_users():
    """Return all user IDs in the Spending DNA dataset."""
    from agents.spending_dna_agent import SpendingDNAAgent
    return {"users": SpendingDNAAgent().get_all_users()}
