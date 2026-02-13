from rag_baseline.ingestion.mediastack_client import MediaStackClient
from rag_baseline.ingestion.text_chunker import TextChunker
from rag_baseline.storage.postgres_repository import PostgresRepository


class IngestionService:
    def __init__(self) -> None:
        self.client = MediaStackClient()
        self.chunker = TextChunker()
        self.repository = PostgresRepository()

    def ingest(self) -> None:
        articles = self.client.fetch_latest_articles()

        for article in articles:
            self.repository.insert_article(article)
            chunks = self.chunker.chunk_article(article)
            self.repository.insert_chunks(chunks)
