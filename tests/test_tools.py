from unittest.mock import Mock, patch

from agentic_rag.models import RetrievedSource
from agentic_rag.tools.baseline_tool import BaselineRetrieverTool
from agentic_rag.tools.tavily_tool import TavilySearchTool
from agentic_rag.tools.tool_registry import ToolRegistry


class TestBaselineRetrieverTool:
    """Test suite for BaselineRetrieverTool"""

    def test_tool_initialization(self) -> None:
        """Test tool can be initialized with mock retriever"""
        mock_retriever = Mock()
        tool = BaselineRetrieverTool(retriever=mock_retriever)

        assert tool.retriever == mock_retriever

    def test_search_returns_formatted_results(self) -> None:
        """Test search returns properly formatted RetrievedSource objects"""
        mock_retriever = Mock()
        mock_result = Mock()
        mock_result.chunk_id = "test_id"
        mock_result.article_id = "article_1"
        mock_result.content = "test content"
        mock_result.similarity_score = 0.95
        mock_result.source = "Test Source"
        mock_result.title = "Test Title"
        mock_result.url = "https://test.com"
        mock_result.published_at = None
        mock_result.keywords = ["test"]

        mock_retriever.retrieve.return_value = [mock_result]

        tool = BaselineRetrieverTool(retriever=mock_retriever)
        results = tool.search("test query", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "test_id"
        assert results[0].score == 0.95
        assert isinstance(results[0], RetrievedSource)

    def test_search_returns_empty_on_error(self) -> None:
        """Test search returns empty list on retriever error"""
        mock_retriever = Mock()
        mock_retriever.retrieve.side_effect = Exception("DB error")

        tool = BaselineRetrieverTool(retriever=mock_retriever)
        results = tool.search("test query")

        assert results == []


class TestTavilySearchTool:
    """Test suite for TavilySearchTool"""

    @patch("agentic_rag.tools.tavily_tool.TavilyClient")
    def test_tool_initialization(self, mock_client_class: Mock) -> None:
        """Test tool can be initialized with API key"""
        tool = TavilySearchTool(api_key="tvly-test-key")

        assert tool.client is not None
        mock_client_class.assert_called_once_with(api_key="tvly-test-key")

    @patch("agentic_rag.tools.tavily_tool.TavilyClient")
    def test_search_returns_formatted_results(self, mock_client_class: Mock) -> None:
        """Test search returns properly formatted RetrievedSource objects"""
        mock_client = Mock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test Article",
                    "content": "Test content about timber",
                    "url": "https://timber-online.net/test",
                    "score": 0.9,
                    "published_date": "2026-02-27",
                }
            ]
        }
        mock_client_class.return_value = mock_client

        tool = TavilySearchTool(api_key="tvly-test-key")
        results = tool.search("timber prices 2026", max_results=3)

        assert len(results) >= 1
        assert results[0].url == "https://timber-online.net/test"
        assert results[0].source_type == "tavily"
        assert isinstance(results[0], RetrievedSource)

    @patch("agentic_rag.tools.tavily_tool.TavilyClient")
    def test_search_error_handling(self, mock_client_class: Mock) -> None:
        """Test graceful error handling on API failure"""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        tool = TavilySearchTool(api_key="tvly-test-key")
        results = tool.search("test query")

        assert results == []

    def test_no_api_key_returns_empty(self) -> None:
        """Test tool without API key returns empty list"""
        with patch("agentic_rag.tools.tavily_tool.agentic_settings") as mock_settings:
            mock_settings.tavily_api_key = None
            mock_settings.min_allowed_evidence_date = "2026-01-01"
            tool = TavilySearchTool(api_key=None)
            results = tool.search("test query")

        assert results == []


class TestToolRegistry:
    """Test suite for ToolRegistry"""

    def test_registry_initialization(self) -> None:
        """Test registry can be initialized with mock tools"""
        mock_baseline = Mock(spec=BaselineRetrieverTool)
        mock_tavily = Mock(spec=TavilySearchTool)

        registry = ToolRegistry(baseline=mock_baseline, tavily=mock_tavily)

        assert registry.baseline == mock_baseline
        assert registry.tavily == mock_tavily

    def test_call_unknown_tool_returns_failure(self) -> None:
        """Test calling unknown tool returns failure ToolResult"""
        mock_baseline = Mock(spec=BaselineRetrieverTool)
        mock_tavily = Mock(spec=TavilySearchTool)
        registry = ToolRegistry(baseline=mock_baseline, tavily=mock_tavily)

        result = registry.call("unknown_tool", query="test")

        assert result.success is False
        assert "Unknown tool" in result.error

    def test_tool_schemas_returns_four_tools(self) -> None:
        """Test tool_schemas returns all four tool definitions"""
        mock_baseline = Mock(spec=BaselineRetrieverTool)
        mock_tavily = Mock(spec=TavilySearchTool)
        registry = ToolRegistry(baseline=mock_baseline, tavily=mock_tavily)

        schemas = registry.tool_schemas

        assert isinstance(schemas, list)
        assert len(schemas) == 4
        tool_names = [s["function"]["name"] for s in schemas]
        assert "search_baseline" in tool_names
        assert "search_tavily_specialist" in tool_names
        assert "search_tavily_web" in tool_names
        assert "search_mediastack" in tool_names

    def test_call_baseline_tool(self) -> None:
        """Test calling baseline tool through registry returns ToolResult"""
        mock_baseline = Mock(spec=BaselineRetrieverTool)
        mock_baseline.search.return_value = [
            RetrievedSource(chunk_id="c1", content="test content", score=0.9)
        ]
        mock_tavily = Mock(spec=TavilySearchTool)
        registry = ToolRegistry(baseline=mock_baseline, tavily=mock_tavily)

        result = registry.call("search_baseline", query="timber prices")

        assert result.success is True
        assert result.count == 1
        assert result.tool == "search_baseline"
        assert isinstance(result.data, list)
