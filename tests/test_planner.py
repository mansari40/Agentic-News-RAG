from typing import Any
from unittest.mock import Mock, patch

from agentic_rag.agents.planner import QueryPlanner


def _make_state(query: str = "What are lumber prices?") -> dict[str, Any]:
    return {
        "query": query,
        "steps": [],
        "llm_calls": 0,
        "cost_usd": 0.0,
        "total_tokens": 0,
    }


def _mock_llm_response(content: str) -> Mock:
    return Mock(
        content=content,
        usage_metadata={"input_tokens": 100, "output_tokens": 50},
    )


_SIMPLE_PLAN = (
    '{"intent": "domain", "is_domain_relevant": true, '
    '"domain_relevance_reason": "timber price query", '
    '"query_type": "simple", "entities": ["Lumber"], '
    '"search_angles": ["lumber market"], '
    '"research_mode": "skip_research", "complexity": "simple", '
    '"is_follow_up": false, "temporal_info": null, "conversational_response": null}'
)


class TestQueryPlanner:
    """Test suite for QueryPlanner"""

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_planner_initialization(self, mock_llm_class: Mock) -> None:
        """Test planner can be initialized"""
        planner = QueryPlanner()
        assert planner is not None

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_simple_query(self, mock_llm_class: Mock) -> None:
        """Test planning for simple domain query"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(_SIMPLE_PLAN)
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("What are lumber prices?"))

        assert result["query_type"] == "simple"
        assert result["research_mode"] == "skip_research"
        assert result["is_domain_relevant"] is True
        assert "Lumber" in result["entities"]
        assert "planned" in result["steps"]

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_temporal_query(self, mock_llm_class: Mock) -> None:
        """Test planning for temporal query routes to baseline_tavily"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"intent": "domain", "is_domain_relevant": true, '
            '"domain_relevance_reason": "recent timber news", '
            '"query_type": "temporal", "entities": ["timber"], '
            '"search_angles": ["recent timber market news 2026"], '
            '"research_mode": "full_research", "complexity": "moderate", '
            '"is_follow_up": false, "temporal_info": {"type": "recent"}, '
            '"conversational_response": null}'
        )
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("What is the latest timber news?"))

        assert result["query_type"] == "temporal"
        assert result["research_mode"] == "full_research"

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_out_of_scope_query(self, mock_llm_class: Mock) -> None:
        """Test planning marks out-of-scope queries with is_domain_relevant=False"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"intent": "domain", "is_domain_relevant": false, '
            '"domain_relevance_reason": "not related to timber", '
            '"query_type": "simple", "entities": [], '
            '"search_angles": [], "research_mode": "skip_research", '
            '"complexity": "simple", "is_follow_up": false, '
            '"temporal_info": null, "conversational_response": null}'
        )
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("What is the weather in Berlin?"))

        assert result["is_domain_relevant"] is False

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_normalises_invalid_strategy(self, mock_llm_class: Mock) -> None:
        """Test planner normalises an unrecognised retrieval strategy to all_sources"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"intent": "domain", "is_domain_relevant": true, '
            '"domain_relevance_reason": "", "query_type": "simple", '
            '"entities": [], "search_angles": [], '
            '"research_mode": "invalid_mode", "complexity": "simple", '
            '"is_follow_up": false, "temporal_info": null, '
            '"conversational_response": null}'
        )
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("Test query"))

        assert result["research_mode"] == "full_research"

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_llm_failure_uses_fallback(self, mock_llm_class: Mock) -> None:
        """Test planner falls back gracefully on LLM error"""
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM API error")
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("latest timber news today"))

        assert "research_mode" in result
        assert result["research_mode"] in {"full_research", "skip_research"}

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_increments_llm_calls(self, mock_llm_class: Mock) -> None:
        """Test plan() increments llm_calls counter by 1"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(_SIMPLE_PLAN)
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state())

        assert result["llm_calls"] == 1

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_with_memory_context(self, mock_llm_class: Mock) -> None:
        """Test planner works correctly when memory provides follow-up context"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"intent": "domain", "is_domain_relevant": true, '
            '"domain_relevance_reason": "", "query_type": "simple", '
            '"entities": [], "search_angles": [], '
            '"research_mode": "skip_research", "complexity": "simple", '
            '"is_follow_up": true, "temporal_info": null, '
            '"conversational_response": null}'
        )
        mock_llm_class.return_value = mock_llm

        mock_memory = Mock()
        mock_memory.is_follow_up_question.return_value = (
            True,
            {"query": "prev question", "answer": "prev answer"},
        )
        mock_memory.get_conversation_context.return_value = (
            "Previous Q: prev question\nPrevious A: prev answer..."
        )

        planner = QueryPlanner(memory=mock_memory)
        result = planner.plan(_make_state("What about it?"))

        assert result["is_follow_up"] is True

    @patch("agentic_rag.agents.planner.ChatOpenAI")
    def test_plan_complexity_score_mapping(self, mock_llm_class: Mock) -> None:
        """Test complexity strings are correctly mapped to float scores"""
        mock_llm = Mock()
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"intent": "domain", "is_domain_relevant": true, '
            '"domain_relevance_reason": "", "query_type": "multi_hop", '
            '"entities": [], "search_angles": [], '
            '"research_mode": "full_research", "complexity": "complex", '
            '"is_follow_up": false, "temporal_info": null, '
            '"conversational_response": null}'
        )
        mock_llm_class.return_value = mock_llm

        planner = QueryPlanner()
        result = planner.plan(_make_state("Compare German vs EU timber policy impacts"))

        assert result["complexity"] == "complex"
        assert result["complexity_score"] == 0.8
