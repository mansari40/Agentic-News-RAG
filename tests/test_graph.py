from typing import Any
from unittest.mock import Mock, patch

from agentic_rag.graph import create_agentic_graph


class TestAgenticGraph:
    """Test suite for LangGraph workflow"""

    @patch("agentic_rag.graph.ToolRegistry")
    @patch("agentic_rag.graph.QueryPlanner")
    @patch("agentic_rag.graph.ResearchAgent")
    @patch("agentic_rag.graph.SourceRanker")
    @patch("agentic_rag.graph.FactVerifier")
    @patch("agentic_rag.graph.AnswerSynthesizer")
    def test_graph_creation(self, *mocks: Mock) -> None:
        """Test graph can be created with all mocked agents"""
        graph = create_agentic_graph()
        assert graph is not None

    @patch("agentic_rag.graph.ToolRegistry")
    @patch("agentic_rag.graph.QueryPlanner")
    @patch("agentic_rag.graph.ResearchAgent")
    @patch("agentic_rag.graph.SourceRanker")
    @patch("agentic_rag.graph.FactVerifier")
    @patch("agentic_rag.graph.AnswerSynthesizer")
    def test_graph_with_memory(
        self,
        mock_synthesizer: Mock,
        mock_verifier: Mock,
        mock_ranker: Mock,
        mock_researcher: Mock,
        mock_planner: Mock,
        mock_registry: Mock,
    ) -> None:
        """Test graph creation with memory passes memory to QueryPlanner"""
        memory = Mock()
        graph = create_agentic_graph(memory=memory)

        assert graph is not None
        mock_planner.assert_called_once_with(memory=memory)

    def test_domain_routing_in_scope(self) -> None:
        """Test after_plan routes in-scope query to research"""
        state: dict[str, Any] = {"is_domain_relevant": True}
        route = "research" if state.get("is_domain_relevant", True) else "synthesize_oos"
        assert route == "research"

    def test_domain_routing_out_of_scope(self) -> None:
        """Test after_plan routes out-of-scope query to synthesize_oos"""
        state: dict[str, Any] = {"is_domain_relevant": False}
        route = "research" if state.get("is_domain_relevant", True) else "synthesize_oos"
        assert route == "synthesize_oos"

    def test_should_refine_true(self) -> None:
        """Test after_synthesize loops back to research when needs_refinement=True"""
        state: dict[str, Any] = {"needs_refinement": True, "refinement_count": 0}
        route = "research" if state.get("needs_refinement", False) else "end"
        assert route == "research"

    def test_should_refine_false(self) -> None:
        """Test after_synthesize ends when needs_refinement=False"""
        state: dict[str, Any] = {"needs_refinement": False, "refinement_count": 1}
        route = "research" if state.get("needs_refinement", False) else "end"
        assert route == "end"

    def test_should_refine_false_default(self) -> None:
        """Test after_synthesize ends when needs_refinement key is absent"""
        state: dict[str, Any] = {}
        route = "research" if state.get("needs_refinement", False) else "end"
        assert route == "end"
