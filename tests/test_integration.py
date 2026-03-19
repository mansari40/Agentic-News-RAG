from typing import Any
from unittest.mock import Mock, patch


def _workflow_state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "answer": "Holzpreise liegen bei €500/m³",
        "citations": [],
        "confidence_score": 0.85,
        "steps": ["planned", "researched", "ranked", "verified", "synthesized"],
        "llm_calls": 4,
        "retrieval_count": 1,
        "query_type": "simple",
        "entities": ["Lumber"],
        "intent": "domain",
        "is_domain_relevant": True,
        "evidence_summary": "Strong evidence from 3 sources.",
        "key_facts": ["Price is €500/m³"],
        "cost_usd": 0.002,
        "total_tokens": 1500,
        "researcher_scratchpad": [],
        "needs_refinement": False,
        "refinement_count": 0,
    }
    base.update(overrides)
    return base


class TestIntegration:
    """End-to-end integration tests"""

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_simple_query_end_to_end(self, mock_create_graph: Mock) -> None:
        """Test complete workflow for a simple query produces a valid result"""
        from agentic_rag.pipeline import AgenticRAGPipeline

        mock_workflow = Mock()
        mock_workflow.invoke.return_value = _workflow_state()
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=False)
        result = pipeline.answer("What are lumber prices?")

        assert "answer" in result
        assert result["llm_calls"] > 0
        assert result["confidence"] > 0
        assert result["cached"] is False

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_workflow_with_refinement(self, mock_create_graph: Mock) -> None:
        """Test workflow that completed a refinement loop returns correct iteration count"""
        from agentic_rag.pipeline import AgenticRAGPipeline

        mock_workflow = Mock()
        mock_workflow.invoke.return_value = _workflow_state(
            answer="Refined answer after second research pass",
            confidence_score=0.90,
            steps=[
                "planned",
                "researched",
                "ranked",
                "verified",
                "synthesized",
                "researched",
                "ranked",
                "verified",
                "synthesized",
            ],
            llm_calls=8,
            retrieval_count=2,
            query_type="comparison",
            entities=["German Lumber", "EU Lumber"],
            cost_usd=0.005,
            total_tokens=3000,
            refinement_count=1,
        )
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=False)
        result = pipeline.answer("Compare German vs EU lumber prices")

        assert result["retrieval_iterations"] == 2
        assert len(result["reasoning_steps"]) > 6

    @patch("agentic_rag.pipeline.create_agentic_graph")
    def test_out_of_scope_query(self, mock_create_graph: Mock) -> None:
        """Test out-of-scope query returns answer with is_domain_relevant=False"""
        from agentic_rag.pipeline import AgenticRAGPipeline

        mock_workflow = Mock()
        mock_workflow.invoke.return_value = _workflow_state(
            answer="This question is outside my German timber market scope.",
            confidence_score=0.0,
            steps=["planned", "out_of_scope"],
            llm_calls=1,
            retrieval_count=0,
            is_domain_relevant=False,
        )
        mock_create_graph.return_value = mock_workflow

        pipeline = AgenticRAGPipeline(enable_memory=False)
        result = pipeline.answer("Who won the football match?")

        assert "answer" in result
        assert result["is_domain_relevant"] is False
