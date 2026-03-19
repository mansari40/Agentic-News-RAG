"""
Baseline Retriever — fetches semantically similar chunks from the vector database.
Results are passed through as-is; the Verifier decides what's actually useful.
"""

import structlog
from agentic_rag.configuration import agentic_settings
from agentic_rag.models import RetrievedSource
from rag_baseline.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)


class BaselineRetrieverTool:
    """Thin wrapper around HybridRetriever that returns results in the standard format."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def search(
        self,
        query: str,
        use_hybrid: bool = True,
        top_k: int = 5,
    ) -> list[RetrievedSource]:
        try:
            results = self.retriever.retrieve(query, use_hybrid=use_hybrid)

            seen_chunk_ids: set[str] = set()
            formatted: list[RetrievedSource] = []

            for result in results[: max(top_k * 2, 10)]:
                chunk_id = result.chunk_id or ""
                content = (result.content or "").strip()
                title = (result.title or "").strip() if result.title else None

                if not content and not title:
                    continue

                if chunk_id and chunk_id in seen_chunk_ids:
                    continue
                if chunk_id:
                    seen_chunk_ids.add(chunk_id)

                source = RetrievedSource(
                    chunk_id=chunk_id,
                    article_id=result.article_id or "",
                    content=content[: agentic_settings.max_content_chars_per_source * 2],
                    score=float(result.similarity_score or 0.5),
                    source_type="baseline",
                    source_name=result.source or "baseline",
                    title=title,
                    url=result.url,
                    published_at=str(result.published_at) if result.published_at else None,
                    keywords=list(result.keywords)
                    if hasattr(result, "keywords") and result.keywords
                    else [],
                )
                formatted.append(source)

                if len(formatted) >= top_k:
                    break

            logger.info(f"Baseline: {len(formatted)} chunks retrieved")
            return formatted

        except Exception as exc:
            logger.error(f"Baseline retrieval error: {exc}")
            return []
