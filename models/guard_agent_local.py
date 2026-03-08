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
from agents import AgentResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RAG singleton for tool_query_rag
_rag_engine = None
def _get_rag():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngineLocal()
        _rag_engine.index_data()
    return _rag_engine

# ---------------------------------------------------------------------------
# MASTER AGENT: Clean Architecture Facade
# ---------------------------------------------------------------------------

class LocalGuardAgent:
    """
    Facade for Local AI services (LLM and RAG).
    Used by the Security Analyst and other core services.
    """
    def __init__(self):
        self.llm = LocalLLM()
        self._rag = None

    def analyze(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Streamlined analysis loop using direct LLM + RAG context.
        """
        print(f"⚡ GuardAgent Facade → Specialist Analysis (Session: {session_id})")
        
        try:
            # Get context from RAG
            context = _get_rag().get_context_for_query(question)
            
            prompt = f"""You are an AI Security Analyst. Use the following context to answer the user's question.

            Context:
            {context}

            Question: {question}

            Answer (keep it concise and to the point):"""
            
            # Lower max_tokens for faster responses; rely on VERISCAN_FAST_MODE for smaller model if desired.
            answer = self.llm.generate(prompt, max_tokens=140)
            
            return {
                "answer": answer,
                "actions": [],
                "status": "completed",
                "session_id": session_id,
                "trace": ["Direct LLM + RAG reasoning"]
            }
        except Exception as e:
            traceback.print_exc()
            return {"answer": f"Error in GuardAgent: {e}", "actions": [], "status": "error"}

if __name__ == "__main__":
    agent = LocalGuardAgent()
    print(agent.analyze("Investigate USER_1")["answer"])
