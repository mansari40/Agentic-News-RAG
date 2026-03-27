import logging

from rag_baseline.ingestion.mediastack_fetcher import MediaStackClient
from rag_baseline.ingestion.text_chunker import TextChunker
from storage.postgres_repository import PostgresRepository

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self) -> None:
        self.client = MediaStackClient()
        self.chunker = TextChunker()
        self.repository = PostgresRepository()

    def ingest(self, days_back: int = 30) -> None:
        logger.info("Starting article ingestion")

        try:
            articles = self.client.fetch_latest_articles(days_back=days_back)
            logger.info(f"Fetched {len(articles)} articles from MediaStack")

            if not articles:
                logger.warning("No articles fetched. Check your API key and MediaStack account.")
                return

            for idx, article in enumerate(articles, 1):
                logger.info(f"Processing article {idx}/{len(articles)}: {article.title[:50]}...")
                self.repository.insert_article(article)

                chunks = self.chunker.chunk_article(article)
                logger.info(f"  Created {len(chunks)} chunks")
                self.repository.insert_chunks(chunks)

            logger.info(f"Successfully ingested {len(articles)} articles")

        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise
