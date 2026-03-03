import re
import json
from typing import Dict, Any, Optional, List
from .base import BaseAgent, AgentResult, AgentAction
from .memory import get_memory
from models.local_llm import LocalLLM

class SynthesisAgent(BaseAgent):
    """Handles complex queries that need multiple tools + LLM reasoning + History."""

    def __init__(self, llm: Optional[LocalLLM] = None):
        from models.guard_agent_local import LocalLLM
        self.llm = llm or LocalLLM()
        self.memory = get_memory()

    def run(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        from models.guard_agent_local import (
            tool_get_user_risk_profile, 
            tool_get_high_risk_transactions, 
            tool_query_rag
        )
        
        actions: List[AgentAction] = []
        context_parts: List[str] = []
        trace: List[str] = []

        # 1. Retrieve Conversation History
        history = ""
        if session_id:
            history = self.memory.get_history(session_id)
            if history:
                trace.append(f"Loaded {len(history.splitlines())} lines of conversation history.")

        # 2. Tool Execution logic (simplified for move)
        # Check for user IDs
        user_matches = re.findall(r'USER[_\s]?(\d+)', query.upper())
        for uid in user_matches[:2]:
            user_id = f"USER_{uid}"
            res = tool_get_user_risk_profile(user_id)
            actions.append(AgentAction(step=len(actions), tool="get_user_risk_profile", args={"user_id": user_id}, result=str(res)[:200]))
            context_parts.append(f"Profile {user_id}: {res}")
            trace.append(f"Executed profile tool for {user_id}")

        # RAG query
        rag_res = tool_query_rag(query)
        actions.append(AgentAction(step=len(actions), tool="query_rag", args={"question": query}, result=rag_res[:200]))
        context_parts.append(f"RAG Intel: {rag_res}")
        trace.append("Executed RAG search")

        # 3. LLM Synthesis with History
        combined_context = "\n".join(context_parts)
        prompt = f"""You are a Veriscan Security Analyst. Using the history and evidence, answer the user.

Conversation History:
{history}

Gathered Evidence:
{combined_context[:2000]}

Current Question: {query}

Professional Report:"""

        trace.append("Generating LLM synthesis report...")
        answer = self.llm.generate(prompt, max_tokens=1024)
        answer = re.sub(r'<\|.*?\|>', '', answer).strip()
        
        # Save to memory
        if session_id:
            self.memory.add_message(session_id, "user", query)
            self.memory.add_message(session_id, "assistant", answer)

        return AgentResult(answer=answer, actions=actions, session_id=session_id, trace=trace)
