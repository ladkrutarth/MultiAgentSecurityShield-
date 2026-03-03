"""
Veriscan — Multi-Agent GuardAgent: Clean Architecture Facade
This file acts as the primary entry point for the agent system, coordinating
data loading, tool definitions, and routing to specialized modular agents.
"""

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from models.local_llm import LocalLLM
from models.rag_engine_local import RAGEngineLocal
from models.agent_tools_data import (
    tool_get_user_risk_profile,
    tool_get_high_risk_transactions,
)
from agents import (
    KnowledgeAgent,
    RiskScannerAgent,
    ProfileAgent,
    SynthesisAgent,
    AgentResult
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG singleton for tool_query_rag (only guard_agent uses this here)
_rag_engine = None
def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
    return _rag_engine

# ---------------------------------------------------------------------------
# TOOLS (Used by modular agents) — tool_get_user_risk_profile, tool_get_high_risk_transactions from agent_tools_data
# ---------------------------------------------------------------------------

def tool_query_rag(question: str) -> str:
    try:
        return _get_rag().get_context_for_query(question)
    except Exception as e:
        return f"RAG tool error: {e}"

# ---------------------------------------------------------------------------
# MASTER AGENT: Clean Architecture Facade
# ---------------------------------------------------------------------------

class LocalGuardAgent:
    def __init__(self):
        self.llm = LocalLLM()
        self.rag_agent = KnowledgeAgent(self.llm)
        self.risk_agent = RiskScannerAgent()
        self.user_agent = ProfileAgent()
        self.synthesis_agent = SynthesisAgent(self.llm)

    def _classify_query(self, query: str) -> str:
        q = query.upper()
        user_ids = re.findall(r'USER[_\s]?\d+', q)
        if len(user_ids) >= 2: return "synthesis"

        is_risk_scan = any(kw in q for kw in ["HIGH RISK", "TOP TRANSACTION", "MOST DANGEROUS", "SYSTEM SCAN", "STATE OF"])
        is_knowledge = any(kw in q for kw in ["WHAT IS", "EXPLAIN", "HOW TO", "DEFINE", "SEARCH", "CFPB"])

        if (int(len(user_ids) > 0) + int(is_risk_scan) + int(is_knowledge)) >= 2:
            return "synthesis"

        if len(user_ids) > 0: return "user_profile"
        if is_risk_scan: return "risk_scan"
        if is_knowledge: return "knowledge"
        return "synthesis"

    def analyze(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        agent_type = self._classify_query(question)
        print(f"⚡ GuardAgent Facade → {agent_type.upper()} agent (Session: {session_id})")

        try:
            if agent_type == "knowledge":
                res = self.rag_agent.run(question, session_id)
            elif agent_type == "risk_scan":
                res = self.risk_agent.run(question, session_id)
            elif agent_type == "user_profile":
                res = self.user_agent.run(question, session_id)
            else:
                res = self.synthesis_agent.run(question, session_id)

            return {
                "answer": res.answer,
                "actions": [a.dict() for a in res.actions],
                "status": res.status,
                "session_id": res.session_id,
                "trace": res.trace
            }
        except Exception as e:
            traceback.print_exc()
            return {"answer": f"Error in {agent_type} agent: {e}", "actions": [], "status": "error"}

if __name__ == "__main__":
    agent = LocalGuardAgent()
    print(agent.analyze("Investigate USER_1")["answer"])
