"""Centralised prompt templates for all agentic agents."""

from agentic_rag.prompts.prompts_template import (
    CONVERSATIONAL_RESPONSE_PROMPT,
    FACT_VERIFIER_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
    QUERY_PLANNER_PROMPT,
    RANKER_PROMPT,
    REFINEMENT_EVAL_PROMPT,
    RESEARCHER_REACT_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT,
    format_key_facts,
    format_sources_for_prompt,
    format_verified_sources_for_synthesis,
)

__all__ = [
    "QUERY_PLANNER_PROMPT",
    "RANKER_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "RESEARCHER_REACT_PROMPT",
    "REFINEMENT_EVAL_PROMPT",
    "FACT_VERIFIER_PROMPT",
    "SYNTHESIS_SYSTEM_PROMPT",
    "SYNTHESIS_USER_PROMPT",
    "CONVERSATIONAL_RESPONSE_PROMPT",
    "OUT_OF_SCOPE_RESPONSE",
    "format_sources_for_prompt",
    "format_verified_sources_for_synthesis",
    "format_key_facts",
]
