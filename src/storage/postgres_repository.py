import psycopg2
from rag_baseline.configuration import settings
from rag_baseline.custom_exceptions import StorageLayerError
from rag_baseline.domain_models import NewsArticle, TextChunk


class PostgresRepository:
    def __init__(self) -> None:
        self.connection = psycopg2.connect(settings.postgres_url)
        self.connection.autocommit = True

    def insert_article(self, article: NewsArticle) -> None:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO articles (
                        article_id,
                        title,
                        content,
                        source,
                        published_at,
                        url
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (article_id) DO NOTHING;
                    """,
                    (
                        article.article_id,
                        article.title,
                        article.content,
                        article.source,
                        article.published_at,
                        article.url,
                    ),
                )
        except Exception as error:
            raise StorageLayerError(f"Failed to insert article {article.article_id}") from error

    def insert_chunks(self, chunks: list[TextChunk]) -> None:
        try:
            with self.connection.cursor() as cursor:
                for chunk in chunks:
                    cursor.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id,
                            article_id,
                            content,
                            chunk_index
                        )
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING;
                        """,
                        (
                            chunk.chunk_id,
                            chunk.article_id,
                            chunk.content,
                            chunk.chunk_index,
                        ),
                    )
        except Exception as error:
            raise StorageLayerError("Failed to insert text chunks") from error

    def close(self) -> None:
        if self.connection:
            self.connection.close()
