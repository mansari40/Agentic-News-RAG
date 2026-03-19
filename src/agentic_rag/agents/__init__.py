"""
Agentic RAG Agents.

Architecture:
- QueryPlanner:      single LLM call — intent + retrieval plan
- ResearchAgent: ReAct loop — LLM decides which tools to call
- SourceRanker:      lightweight pre-verifier triage
- FactVerifier:      LLM reads ranked shortlist, selects relevant sources
- AnswerSynthesizer: writes answer from verified sources
"""

from agentic_rag.agents.planner import QueryPlanner
from agentic_rag.agents.ranker import SourceRanker
from agentic_rag.agents.researcher import ResearchAgent
from agentic_rag.agents.synthesizer import AnswerSynthesizer
from agentic_rag.agents.verifier import FactVerifier

__all__ = [
    "QueryPlanner",
    "ResearchAgent",
    "SourceRanker",
    "FactVerifier",
    "AnswerSynthesizer",
]
