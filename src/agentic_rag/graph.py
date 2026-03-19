"""
Agentic workflow graph — wires all the agents together into a LangGraph pipeline.

Flow:
  plan → [domain check] → research → rank → verify → synthesize
                        ↓ out of scope
                   synthesize_oos
"""

from typing import Any

from agentic_rag.agents.planner import QueryPlanner
from agentic_rag.agents.ranker import SourceRanker
from agentic_rag.agents.researcher import ResearchAgent
from agentic_rag.agents.synthesizer import AnswerSynthesizer
from agentic_rag.agents.verifier import FactVerifier
from agentic_rag.memory import ConversationMemory
from agentic_rag.prompts.prompts_template import OUT_OF_SCOPE_RESPONSE
from agentic_rag.state import AgenticState
from agentic_rag.tools.tool_registry import ToolRegistry
from langgraph.graph import END, StateGraph


def create_agentic_graph(
    memory: ConversationMemory | None = None,
    event_callback: Any | None = None,
) -> Any:
    """
    Build and compile the LangGraph workflow.

    Pass event_callback from the FastAPI streaming endpoint to get live
    Thought/Action/Observation events as the researcher runs.
    """
    planner = QueryPlanner(memory=memory)
    registry = ToolRegistry()
    researcher = ResearchAgent(
        tool_registry=registry,
        event_callback=event_callback,
    )
    ranker = SourceRanker()
    verifier = FactVerifier()
    synthesizer = AnswerSynthesizer()

    #  Node functions

    def plan_node(state: AgenticState) -> AgenticState:
        return planner.plan(state)

    def research_node(state: AgenticState) -> AgenticState:
        return researcher.research(state)

    def rank_node(state: AgenticState) -> AgenticState:
        return ranker.rank(state)

    def verify_node(state: AgenticState) -> AgenticState:
        # Feed ranked_sources to the Verifier instead of all_sources to keep tokens in check.
        # After verification all_sources are put back so the refinement pass can still use it.
        state_for_verify = dict(state)
        ranked = state.get("ranked_sources") or []
        if ranked:
            state_for_verify["all_sources"] = ranked
        result = verifier.verify(state_for_verify)
        # Don't let the verifier overwrite all_sources — the refinement researcher needs it
        result["all_sources"] = list(state.get("all_sources", []))
        return result

    def synthesize_node(state: AgenticState) -> AgenticState:
        return synthesizer.synthesize(state)

    def synthesize_oos_node(state: AgenticState) -> AgenticState:
        """Fast exit for out-of-scope questions — returns the decline message immediately."""
        return {
            **state,
            "answer": OUT_OF_SCOPE_RESPONSE,
            "citations": [],
            "steps": list(state.get("steps", [])) + ["out_of_scope"],
        }

    # Routing functions

    def after_plan(state: AgenticState) -> str:
        """After planning, skip research entirely if the question is out of scope."""
        if not state.get("is_domain_relevant", True):
            return "synthesize_oos"
        return "research"

    def after_synthesize(state: AgenticState) -> str:
        """
        After synthesis, decide whether to loop back for another research pass.

        If the synthesizer flagged the answer as weak and the refinement
        limit havn't been hit yet, send it back to research with the hint already in state.
        Otherwise we're done.
        """
        if state.get("needs_refinement", False):
            return "research"
        return "end"

    # Build graph

    workflow = StateGraph(AgenticState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("synthesize_oos", synthesize_oos_node)

    workflow.set_entry_point("plan")

    workflow.add_conditional_edges(
        "plan",
        after_plan,
        {"research": "research", "synthesize_oos": "synthesize_oos"},
    )

    workflow.add_edge("research", "rank")
    workflow.add_edge("rank", "verify")
    workflow.add_edge("verify", "synthesize")

    # If the answer needs improvement, loop back to research, otherwise finish
    workflow.add_conditional_edges(
        "synthesize",
        after_synthesize,
        {"research": "research", "end": END},
    )

    workflow.add_edge("synthesize_oos", END)

    return workflow.compile()
