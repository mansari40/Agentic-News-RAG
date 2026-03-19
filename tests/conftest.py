import os
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import pytest
from agentic_rag.memory import ConversationMemory
from agentic_rag.models import RetrievedSource


@pytest.fixture(scope="session", autouse=True)  # type: ignore[misc]
def set_test_api_keys() -> Generator[None, None, None]:
    """Set fake API keys for testing"""
    os.environ["OPENAI_API_KEY"] = "sk-test-fake-key-for-testing"
    os.environ["TAVILY_API_KEY"] = "tvly-test-fake-key"
    os.environ["LANGCHAIN_API_KEY"] = "ls-test-fake-key"
    os.environ["MEDIASTACK_API_KEY"] = "ms-test-fake-key"
    yield
    for key in ["OPENAI_API_KEY", "TAVILY_API_KEY", "LANGCHAIN_API_KEY", "MEDIASTACK_API_KEY"]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture  # type: ignore[misc]
def mock_llm() -> Mock:
    """Mock LLM for testing without API calls"""
    llm = Mock()
    llm.invoke.return_value = Mock(
        content=(
            '{"intent": "domain", "is_domain_relevant": true, '
            '"domain_relevance_reason": "timber price query", '
            '"query_type": "simple", "entities": ["Lumber"], '
            '"search_angles": ["lumber market prices"], '
            '"research_mode": "skip_research", "complexity": "simple", '
            '"is_follow_up": false, "temporal_info": null, '
            '"conversational_response": null}'
        ),
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
    )
    return llm


@pytest.fixture  # type: ignore[misc]
def mock_memory() -> Mock:
    """Mock conversation memory"""
    memory = Mock(spec=ConversationMemory)
    memory.is_follow_up_question.return_value = (False, None)
    memory.check_query_cache.return_value = None
    memory.get_previous_entities.return_value = []
    memory.conversation_history = []
    return memory


@pytest.fixture  # type: ignore[misc]
def sample_state() -> dict[str, Any]:
    """Sample agentic state for testing — matches AgenticState TypedDict"""
    return {
        "query": "What are recent lumber prices?",
        "intent": "domain",
        "query_plan": None,
        "query_type": "simple",
        "entities": ["Lumber"],
        "search_angles": [],
        "sub_queries": ["What are recent lumber prices?"],
        "temporal_info": None,
        "complexity": "simple",
        "complexity_score": 0.2,
        "is_follow_up": False,
        "is_domain_relevant": True,
        "domain_relevance_reason": "",
        "research_mode": "skip_research",
        "use_hybrid": True,
        "top_k": 5,
        "all_sources": [],
        "retrieval_count": 0,
        "seen_urls": [],
        "ranked_sources": [],
        "verified_sources": [],
        "key_facts": [],
        "evidence_summary": "",
        "confidence_score": 0.0,
        "off_topic_indices": [],
        "researcher_scratchpad": [],
        "answer": "",
        "citations": [],
        "cost_usd": 0.0,
        "total_tokens": 0,
        "steps": [],
        "llm_calls": 0,
        "conversation_context": {},
        "allowed_tools": None,
        "cutoff_date_override": None,
        "needs_refinement": False,
        "refinement_hint": "",
        "refinement_count": 0,
    }


@pytest.fixture  # type: ignore[misc]
def sample_sources() -> list[RetrievedSource]:
    """Sample retrieved sources as RetrievedSource objects"""
    return [
        RetrievedSource(
            chunk_id="chunk_1",
            article_id="article_1",
            content="Lumber prices increased 10% this month according to market reports.",
            score=0.95,
            source_type="baseline",
            source_name="Reuters",
            title="Lumber Market Update",
            url="https://reuters.com/lumber",
            published_at="2026-02-20",
            keywords=["lumber", "prices"],
        ),
        RetrievedSource(
            chunk_id="chunk_2",
            article_id="article_2",
            content="Softwood lumber demand remains strong in construction sector.",
            score=0.88,
            source_type="baseline",
            source_name="Bloomberg",
            title="Construction Demand Analysis",
            url="https://bloomberg.com/construction",
            published_at="2026-02-19",
            keywords=["lumber", "construction"],
        ),
    ]


@pytest.fixture  # type: ignore[misc]
def mock_baseline_tool() -> Mock:
    """Mock baseline retriever tool returning RetrievedSource objects"""
    tool = Mock()
    tool.search.return_value = [
        RetrievedSource(
            chunk_id="test_chunk",
            content="Test content",
            score=0.9,
            source_type="baseline",
            source_name="Test Source",
            title="Test Title",
            url="https://test.com",
        )
    ]
    return tool


@pytest.fixture  # type: ignore[misc]
def mock_tavily_tool() -> Mock:
    """Mock Tavily search tool returning RetrievedSource objects"""
    tool = Mock()
    tool.search.return_value = [
        RetrievedSource(
            chunk_id="",
            content="Latest timber market news",
            score=0.95,
            source_type="tavily",
            title="Breaking News",
            url="https://news.com/latest",
        )
    ]
    return tool
