"""
Tool Registry — one place to call any retrieval tool by name.

The Researcher passes a tool name and arguments here and gets back a ToolResult.
Any messiness (HTTP errors, empty responses, type issues) is handled inside so
the Researcher always gets a clean, consistent result.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from agentic_rag.tools.baseline_tool import BaselineRetrieverTool
from agentic_rag.tools.mediastack_tool import MediaStackAPITool
from agentic_rag.tools.tavily_tool import TavilySearchTool

logger = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    tool: str
    success: bool
    data: Any
    count: int = 0
    summary: str = ""
    cost_usd: float = 0.0
    tokens: int = 0
    error: str = ""


class ToolRegistry:
    """
    Holds all retrieval tools and dispatches calls to the right one.
    Never raises — errors are returned inside ToolResult.
    """

    def __init__(
        self,
        baseline: BaselineRetrieverTool | None = None,
        tavily: TavilySearchTool | None = None,
        mediastack: MediaStackAPITool | None = None,
    ) -> None:
        self.baseline = baseline or BaselineRetrieverTool()
        self.tavily = tavily or TavilySearchTool()
        self.mediastack = mediastack or MediaStackAPITool()

    def call(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Run a tool by name. Returns an error ToolResult if the name is unknown."""
        dispatch: dict[str, Callable[..., ToolResult]] = {
            "search_baseline": self._search_baseline,
            "search_tavily_specialist": self._search_tavily_specialist,
            "search_tavily_web": self._search_tavily_web,
            "search_mediastack": self._search_mediastack,
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            logger.error(f"ToolRegistry: unknown tool '{tool_name}'")
            return ToolResult(
                tool=tool_name, success=False, data=[], error=f"Unknown tool: {tool_name}"
            )
        try:
            return handler(**kwargs)
        except Exception as exc:
            logger.error(f"ToolRegistry: {tool_name} raised: {exc}")
            return ToolResult(tool=tool_name, success=False, data=[], error=str(exc))

    # Private wrappers — one per tool

    def _search_baseline(self, query: str, top_k: int = 5) -> ToolResult:
        top_k = min(int(top_k), 10)
        results = self.baseline.search(query, use_hybrid=True, top_k=top_k)
        dicts = [r.model_dump() for r in results]
        return ToolResult(
            tool="search_baseline",
            success=True,
            data=dicts,
            count=len(dicts),
            summary=f"Baseline: {len(dicts)} chunks retrieved for '{query[:60]}'",
        )

    def _search_tavily_specialist(self, query: str) -> ToolResult:
        results = self.tavily.search(query, search_angles=[], specialist_only=True)
        dicts = [r.model_dump() for r in results]
        return ToolResult(
            tool="search_tavily_specialist",
            success=True,
            data=dicts,
            count=len(dicts),
            summary=f"Tavily specialist: {len(dicts)} articles for '{query[:60]}'",
        )

    def _search_tavily_web(self, query: str, search_angles: list[str] | None = None) -> ToolResult:
        angles = search_angles or []
        results = self.tavily.search(query, search_angles=angles, specialist_only=False)
        dicts = [r.model_dump() for r in results]
        return ToolResult(
            tool="search_tavily_web",
            success=True,
            data=dicts,
            count=len(dicts),
            summary=f"Tavily web: {len(dicts)} articles for '{query[:60]}'",
        )

    def _search_mediastack(self, query: str) -> ToolResult:
        results = self.mediastack.search(query)
        dicts = [r.model_dump() for r in results]
        return ToolResult(
            tool="search_mediastack",
            success=True,
            data=dicts,
            count=len(dicts),
            summary=f"MediaStack: {len(dicts)} articles for '{query[:60]}'",
        )

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool schemas — passed to ChatOpenAI so the LLM knows which tools it can call."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_baseline",
                    "description": (
                        "Search the internal vector database for articles similar to the query."
                        " Good for background context and any data that was already indexed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"},
                            "top_k": {
                                "type": "integer",
                                "description": "Number of chunks to retrieve (1-10)",
                                "default": 5,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_tavily_specialist",
                    "description": (
                        "Search dedicated timber industry sites (timber-online.net,"
                        " holzkurier.com, euwid-holz.de, etc.)."
                        " Best starting point for current German timber market news."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query — include year for currency",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_tavily_web",
                    "description": (
                        "Broad web search via Tavily. Use this after the specialist search"
                        " when you need wider coverage. Accepts optional extra search angles."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Primary search query"},
                            "search_angles": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional additional search phrases (1-2 max)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_mediastack",
                    "description": (
                        "Fetch recent German timber news via the MediaStack API."
                        " Good for picking up the latest German-language coverage."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query to guide keyword selection",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]
