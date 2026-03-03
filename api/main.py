"""
Veriscan — FastAPI Microservices Backend
Decoupled REST API for Fraud Prediction, GuardAgent, and RAG Engine.

Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

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
    TransactionInput,
    FraudPredictionResponse,
    HighRiskTransactionsResponse,
    UserRiskResponse,
    AgentInvestigationRequest,
    AgentInvestigationResponse,
    AgentActionStep,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGResult,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Global singletons — loaded once at startup
# ---------------------------------------------------------------------------
_model = None
_encoders = None
_agent = None
_rag_engine = None

FEATURES = ["category", "amt", "gender", "state", "merchant", "hour", "day_of_week"]
CATEGORICAL_COLS = ["category", "gender", "state", "merchant"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once when the server boots.
    GuardAgent (MLX) is skipped at startup to avoid Metal crash when GPU unavailable.
    """
    global _model, _encoders, _agent, _rag_engine

    import joblib

    model_path = PROJECT_ROOT / "models" / "fraud_model_rf.joblib"
    enc_path = PROJECT_ROOT / "models" / "encoders.joblib"

    print("🚀 Veriscan API — Loading resources...")

    # 1. ML Model
    if model_path.exists() and enc_path.exists():
        _model = joblib.load(model_path)
        _encoders = joblib.load(enc_path)
        print("✅ Fraud ML model loaded.")
    else:
        print("⚠️  ML model files not found — prediction endpoint will be unavailable.")

    # 2. RAG Engine (no MLX — safe to load)
    try:
        from models.rag_engine_local import RAGEngineLocal
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
        print("✅ RAG Engine loaded and indexed.")
    except Exception as e:
        print(f"⚠️  RAG Engine failed to load: {e}")
        _rag_engine = None

    # 3. GuardAgent uses MLX — skip at startup to avoid Metal crash
    _agent = None
    print("ℹ️  GuardAgent skipped at startup (requires Apple Silicon for /api/agent/investigate).")

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
            "ml_model": "loaded" if _model else "unavailable",
            "guard_agent": "loaded" if _agent else "unavailable",
            "rag_engine": "loaded" if _rag_engine else "unavailable",
        },
    )


# ---------------------------------------------------------------------------
# 2. Fraud Prediction (Single Transaction)
# ---------------------------------------------------------------------------
@app.post("/api/fraud/predict", response_model=FraudPredictionResponse, tags=["Fraud ML"])
async def predict_fraud(txn: TransactionInput):
    """Predict fraud risk for a single transaction using the Random Forest model."""
    if not _model or not _encoders:
        raise HTTPException(status_code=503, detail="ML model not loaded.")

    import pandas as pd

    input_dict = txn.model_dump()
    row = pd.DataFrame([input_dict])

    for col in CATEGORICAL_COLS:
        try:
            row[col] = _encoders[col].transform(row[col].astype(str))
        except ValueError:
            row[col] = 0  # Unknown category fallback

    prob = float(_model.predict_proba(row[FEATURES])[0][1])
    risk = "CRITICAL" if prob > 0.8 else "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.2 else "LOW"

    importances = dict(zip(FEATURES, [float(x) for x in _model.feature_importances_]))

    return FraudPredictionResponse(
        risk_score=round(prob * 100, 2),
        risk_level=risk,
        feature_importances=importances,
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
        raise HTTPException(status_code=503, detail="GuardAgent not loaded.")

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
