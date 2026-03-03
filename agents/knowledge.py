import re
from typing import Dict, Any, Optional
from .base import BaseAgent, AgentResult, AgentAction
from models.local_llm import LocalLLM
from models.rag_engine_local import RAGEngineLocal

class KnowledgeAgent(BaseAgent):
    """Handles policy/knowledge queries by routing through LLM synthesis for expert-level answers."""
    
    def __init__(self, llm: Optional[LocalLLM] = None, rag: Optional[RAGEngineLocal] = None):
        from models.guard_agent_local import LocalLLM, _get_rag  # Avoid circular imports
        self.llm = llm or LocalLLM()
        self.rag = rag or _get_rag()

    def run(self, query: str, session_id: Optional[str] = None) -> AgentResult:
        results = self.rag.query(query, n_results=5)
        raw_context = "\n".join([r["text"] for r in results]) if results else "No documents found."
        
        actions = [AgentAction(step=0, tool="query_rag", args={"question": query}, result=raw_context[:500])]
        
        prompt = f"""You are a Senior Fraud Intelligence Analyst. Answer this question in 150-200 words.

Question: {query}

Evidence from CFPB database and fraud intelligence:
{raw_context[:1500]}

Write a professional, detailed analysis. Include:
- Clear definitions for any fraud terms mentioned
- Specific references to data (companies, states, amounts from the evidence)
- How Veriscan's ML detection capabilities relate to this topic
- A concrete recommendation at the end

Analysis:"""

        answer = self.llm.generate(prompt, max_tokens=800)
        answer = re.sub(r'<\|.*?\|>', '', answer).strip()
        for artifact in ["assistant", "user", "Question:", "Context:", "Evidence:", "Analysis:"]:
            answer = answer.replace(artifact, "")
        answer = answer.strip()
        
        if not answer or len(answer) < 30:
            answer = "Default knowledge synthesis failed. Please refer to top evidence documents."
            
        return AgentResult(answer=answer, actions=actions, session_id=session_id)
