"""
Data models used across the pipeline.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryPlan(BaseModel):
    """What the Planner figured out — used by every node that comes after it."""

    intent: Literal["conversational", "domain"] = "domain"
    conversational_response: str | None = None
    is_domain_relevant: bool = True
    domain_relevance_reason: str = ""
    query_type: Literal["simple", "comparison", "temporal", "multi_hop"] = "simple"
    entities: list[str] = []
    search_angles: list[str] = []
    research_mode: Literal["full_research", "skip_research"] = "full_research"
    complexity: Literal["simple", "moderate", "complex"] = "moderate"
    is_follow_up: bool = False
    temporal_info: dict[str, Any] | None = None


class QueryAnalysis(BaseModel):
    """Kept for backward compatibility — use QueryPlan for new code."""

    query_type: Literal["simple", "comparison", "temporal", "multi_hop"] = "simple"
    entities: list[str] = []
    temporal_info: dict[str, Any] | None = None
    sub_queries: list[str] = []
    complexity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_domain_relevant: bool = True
    domain_relevance_reason: str = ""
    retrieval_strategy: Literal[
        "baseline_only",
        "baseline_mediastack",
        "baseline_tavily",
        "all_sources",
    ] = "all_sources"
    use_hybrid: bool = True
    top_k: int = Field(default=5, ge=1, le=10)
    search_angles: list[str] = []


class RetrievedSource(BaseModel):
    chunk_id: str = ""
    article_id: str = ""
    content: str = ""
    score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: str = "unknown"
    source_name: str | None = None
    title: str | None = None
    url: str | None = None
    published_at: str | None = None
    language: str | None = None
    country: str | None = None
    keywords: list[str] = []
    # Written by the Verifier LLM — not computed from any score
    relevance_reason: str = ""


class VerificationResult(BaseModel):
    selected_sources: list[dict[str, Any]] = []
    rejected_sources: list[dict[str, Any]] = []
    key_facts: list[str] = []
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_summary: str = ""
    off_topic_indices: list[int] = []


class ResearchStep(BaseModel):
    """One step in the researcher's loop — can be a thought, tool call, or observation."""

    step: int
    type: Literal["thought", "action", "observation", "error", "reformulation"]
    tool: str | None = None
    args: dict[str, Any] = {}
    summary: str = ""
    count: int = 0
