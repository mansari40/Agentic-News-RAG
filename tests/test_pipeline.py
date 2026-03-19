from typing import Any
from unittest.mock import Mock, patch

from agentic_rag.pipeline import AgenticRAGPipeline


def _final_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "answer": "Generated answer",
        "citations": [],
        "confidence_score": 0.85,
        "steps": ["planned", "researched", "synthesized"],
        "llm_calls": 3,
        "retrieval_count": 1,
        "query_type": "simple",
        "entities": ["Lumber"],
        "intent": "domain",
        "is_domain_relevant": True,
        "evidence_summary": "",
        "key_facts": [],
        "cost_usd": 0.001,
        "total_tokens": 800,
        "researcher_scratchpad": [],
        "needs_refinement": False,
        "refinement_count": 0,
    }
    base.update(overrides)
    return base


class TestAgenticRAGPipeline:
    """Test suite for AgenticRAGPipeline"""

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_pipeline_initialization(self, mock_create_graph: Mock) -> None:
        """Test pipeline can be initialized without memory"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=False)

        assert pipeline is not None
        assert pipeline.memory is None

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_pipeline_with_memory(self, mock_create_graph: Mock) -> None:
        """Test pipeline with memory enabled creates ConversationMemory"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=True)

        assert pipeline.memory is not None

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_answer_with_cache_hit(self, mock_create_graph: Mock) -> None:
        """Test pipeline returns cached answer without running the workflow"""
        mock_workflow = Mock()
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=True)
        pipeline.memory.check_query_cache = Mock(
            return_value={"answer": "Cached answer", "timestamp": Mock()}
        )

        result = pipeline.answer("Test query")

        assert result["cached"] is True
        assert result["answer"] == "Cached answer"
        assert result["llm_calls"] == 0
        mock_workflow.invoke.assert_not_called()

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_answer_workflow_execution(self, mock_create_graph: Mock) -> None:
        """Test full workflow execution extracts result correctly"""
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = _final_state()
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=False)
        result = pipeline.answer("What are lumber prices?")

        assert result["answer"] == "Generated answer"
        assert result["confidence"] == 0.85
        assert result["llm_calls"] == 3
        assert result["cached"] is False

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_answer_with_workflow_error(self, mock_create_graph: Mock) -> None:
        """Test error handling — pipeline returns an error result, not an exception"""
        mock_workflow = Mock()
        mock_workflow.invoke.side_effect = Exception("Workflow error")
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=False)
        result = pipeline.answer("Test query")

        assert "error" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_get_conversation_summary(self, mock_create_graph: Mock) -> None:
        """Test get_conversation_summary returns dict with total_interactions"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=True)
        summary = pipeline.get_conversation_summary()

        assert "total_interactions" in summary

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_clear_memory(self, mock_create_graph: Mock) -> None:
        """Test clear_memory delegates to memory.clear()"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=True)
        pipeline.memory.clear = Mock()

        pipeline.clear_memory()

        pipeline.memory.clear.assert_called_once()

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_get_session_cost(self, mock_create_graph: Mock) -> None:
        """Test get_session_cost returns cost and token counts"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=True)
        cost = pipeline.get_session_cost()

        assert "total_cost_usd" in cost
        assert "total_tokens" in cost

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_get_coverage_info(self, mock_create_graph: Mock) -> None:
        """Test get_coverage_info returns scope and sources"""
        mock_create_graph.return_value = Mock()

        pipeline = AgenticRAGPipeline(enable_memory=False)
        info = pipeline.get_coverage_info()

        assert "scope" in info
        assert "sources" in info
