"""
Agentic RAG Pipeline — the main entry point for running a query end-to-end.
"""

from collections.abc import Callable
from typing import Any

import structlog
from agentic_rag.configuration import agentic_settings
from agentic_rag.graph import create_agentic_graph
from agentic_rag.memory import ConversationMemory
from agentic_rag.utils import is_pure_social, is_temporal_query
from langchain_openai import ChatOpenAI

logger = structlog.get_logger(__name__)


class ConversationalHandler:
    """Handles greetings and small-talk that don't need any search."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=agentic_settings.chat_model_name,
            temperature=agentic_settings.conversational_temperature,
        )

    def respond(self, query: str) -> dict[str, Any] | None:
        from agentic_rag.domain import is_timber_related
        from agentic_rag.prompts.prompts_template import CONVERSATIONAL_RESPONSE_PROMPT

        if is_timber_related(query.lower()):
            return None
        try:
            response = self.llm.invoke(CONVERSATIONAL_RESPONSE_PROMPT.format(query=query))
            return {
                "answer": response.content,
                "sources": [],
                "confidence": 1.0,
                "reasoning_steps": ["conversational"],
                "llm_calls": 1,
                "retrieval_iterations": 0,
                "cached": False,
                "query_type": "conversational",
                "intent": "conversational",
                "cost_usd": 0.0,
                "total_tokens": 0,
                "is_domain_relevant": True,
                "evidence_summary": "",
                "key_facts": [],
                "researcher_scratchpad": [],
            }
        except Exception:
            return {
                "answer": "Hello! I can help with German timber market questions.",
                "sources": [],
                "confidence": 1.0,
                "reasoning_steps": ["fallback"],
                "llm_calls": 0,
                "retrieval_iterations": 0,
                "cached": False,
                "query_type": "conversational",
                "intent": "conversational",
                "cost_usd": 0.0,
                "total_tokens": 0,
                "is_domain_relevant": True,
                "evidence_summary": "",
                "key_facts": [],
                "researcher_scratchpad": [],
            }


class AgenticRAGPipeline:
    def __init__(
        self, enable_memory: bool = True, event_callback: Callable[..., None] | None = None
    ) -> None:
        """
        event_callback: optional callable(dict) forwarded to ResearchAgent
        for live Thought/Action/Observation streaming.
        """
        self.memory: ConversationMemory | None = None
        self._event_callback = event_callback

        if enable_memory:
            try:
                self.memory = ConversationMemory(
                    max_history=agentic_settings.max_conversation_history,
                    cache_ttl_minutes=agentic_settings.cache_ttl_minutes,
                )
                logger.info("Memory enabled")
            except Exception as e:
                logger.warning(f"Memory init failed: {e}")

        self.conversational_handler = ConversationalHandler()

        try:
            self.workflow = create_agentic_graph(
                memory=self.memory,
                event_callback=event_callback,
            )
            logger.info("Agentic workflow compiled")
        except Exception as e:
            logger.error(f"Workflow creation failed: {e}")
            raise

    def set_event_callback(self, callback: Callable[..., None]) -> None:
        """
        Swap in a new event callback before each streaming request.
        This wires the researcher to the right SSE queue for that request.
        """
        self._event_callback = callback
        # Rebuild the graph so the new callback reaches the researcher node
        self.workflow = create_agentic_graph(
            memory=self.memory,
            event_callback=callback,
        )

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        try:
            # Catch greetings and small-talk using pattern matching — no LLM needed
            if is_pure_social(question):
                result = self.conversational_handler.respond(question)
                if result:
                    self._save_to_memory(question, result)
                    return result

            # Check cache — skip for "latest news" style questions that need fresh results
            if (
                self.memory
                and agentic_settings.enable_query_cache
                and not is_temporal_query(question)
            ):
                cached = self.memory.check_query_cache(question)
                if cached:
                    logger.info("Cache hit")
                    return {
                        "answer": cached["answer"],
                        "sources": [],
                        "confidence": 1.0,
                        "cached": True,
                        "reasoning_steps": ["cache_hit"],
                        "retrieval_iterations": 0,
                        "llm_calls": 0,
                        "query_type": "cached",
                        "intent": "cached",
                        "cost_usd": 0.0,
                        "total_tokens": 0,
                        "is_domain_relevant": True,
                        "evidence_summary": "",
                        "key_facts": [],
                        "researcher_scratchpad": [],
                    }

            # Run the full pipeline — the planner inside handles classification
            initial_state = self._build_initial_state(question)
            initial_state.update(kwargs)
            final_state = self.workflow.invoke(initial_state)
            result = self._extract_result(final_state)
            self._save_to_memory(question, result, final_state)

            logger.info(
                f"Done: {result['llm_calls']} LLM calls | "
                f"{result['retrieval_iterations']} retrievals | "
                f"confidence={result['confidence']:.2f} | "
                f"cost=${result.get('cost_usd', 0):.6f}"
            )
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            if self.memory:
                self.memory.add_failed_query(question)
            return {
                "answer": "I encountered an error processing your question. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "error": str(e),
                "reasoning_steps": ["error"],
                "retrieval_iterations": 0,
                "llm_calls": 0,
                "query_type": "error",
                "intent": "error",
                "cost_usd": 0.0,
                "total_tokens": 0,
                "is_domain_relevant": True,
                "evidence_summary": "",
                "key_facts": [],
                "researcher_scratchpad": [],
            }

    def _build_initial_state(self, question: str) -> dict[str, Any]:
        conversation_context: dict[str, Any] = {}
        if self.memory and self.memory.conversation_history:
            conversation_context = {
                "context_text": self.memory.get_conversation_context(context_window=3),
                "previous_entities": self.memory.get_previous_entities(),
            }
        return {
            "query": question,
            "intent": "domain",
            "query_plan": None,
            "query_type": "simple",
            "entities": [],
            "sub_queries": [question],
            "search_angles": [],
            "temporal_info": None,
            "complexity": "moderate",
            "complexity_score": 0.5,
            "is_follow_up": False,
            "retrieval_count": 0,
            "seen_urls": [],
            "all_sources": [],
            "ranked_sources": [],
            "researcher_scratchpad": [],
            "llm_calls": 0,
            "steps": [],
            "is_domain_relevant": True,
            "cost_usd": 0.0,
            "total_tokens": 0,
            "conversation_context": conversation_context,
            "verified_sources": [],
            "key_facts": [],
            "evidence_summary": "",
            "confidence_score": 0.0,
            "off_topic_indices": [],
            "needs_refinement": False,
            "refinement_hint": "",
            "refinement_count": 0,
        }

    def _extract_result(
        self, final_state: dict[str, Any], intent_value: str = "domain"
    ) -> dict[str, Any]:
        return {
            "answer": final_state.get(
                "answer", "I couldn't find relevant information for your question."
            ),
            "sources": final_state.get("citations", []),
            "confidence": final_state.get("confidence_score", 0.0),
            "reasoning_steps": list(final_state.get("steps", [])),
            "llm_calls": final_state.get("llm_calls", 0),
            "retrieval_iterations": final_state.get("retrieval_count", 0),
            "cached": False,
            "query_type": final_state.get("query_type", "unknown"),
            "intent": final_state.get("intent", intent_value),
            "is_domain_relevant": final_state.get("is_domain_relevant", True),
            "evidence_summary": final_state.get("evidence_summary", ""),
            "key_facts": final_state.get("key_facts", []),
            "cost_usd": round(final_state.get("cost_usd", 0.0), 6),
            "total_tokens": final_state.get("total_tokens", 0),
            "researcher_scratchpad": final_state.get("researcher_scratchpad", []),
        }

    def _save_to_memory(
        self, question: str, result: dict[str, Any], state: dict[str, Any] | None = None
    ) -> None:
        if not self.memory:
            return
        try:
            self.memory.add_interaction(
                query=question,
                answer=result["answer"],
                sources=result.get("sources", []),
                metadata={
                    "entities": state.get("entities", []) if state else [],
                    "confidence": result.get("confidence", 0),
                    "query_type": result.get("query_type", ""),
                    "intent": result.get("intent", ""),
                    "cost_usd": result.get("cost_usd", 0.0),
                    "total_tokens": result.get("total_tokens", 0),
                },
            )
        except Exception as e:
            logger.warning(f"Memory save failed: {e}")

    def get_conversation_summary(self) -> dict[str, Any]:
        if self.memory:
            result: dict[str, Any] = self.memory.get_summary()
            return result
        return {"total_interactions": 0, "memory_enabled": False}

    def get_session_cost(self) -> dict[str, Any]:
        if self.memory:
            result: dict[str, Any] = self.memory.get_session_cost()
            return result
        return {"total_cost_usd": 0.0, "total_tokens": 0, "total_queries": 0}

    def clear_memory(self) -> None:
        if self.memory:
            self.memory.clear()
            logger.info("Memory cleared")

    def get_coverage_info(self) -> dict[str, Any]:
        return {
            "scope": "German timber sector + EU policy affecting Germany",
            "topics": [
                "Timber and wood prices (Holzpreise)",
                "Forestry supply and log market",
                "Sawmill production and capacity",
                "Timber construction and housing demand",
                "Environmental factors (bark beetle, storm damage)",
                "EU/German policy (EUDR, regulations, tariffs)",
                "Timber trade (exports, imports)",
            ],
            "sources": {
                "baseline": "Vector DB with internal semantic search",
                "mediastack": "Live news API — German timber keywords",
                "tavily": "Web search — specialist timber domains + open web",
            },
            "approach": "ReAct Researcher — LLM decides tool calls; LLM verifier judges relevance",
        }
