from openai import OpenAI
from rag_baseline.configuration import settings
from rag_baseline.custom_exceptions import EmbeddingServiceError
from rag_baseline.embedding.embedding_cache import EmbeddingCache


class EmbeddingService:
    def __init__(self) -> None:
        self.client: OpenAI = OpenAI(api_key=settings.openai_api_key)
        self.cache: EmbeddingCache = EmbeddingCache()

    def embed(self, text: str) -> list[float]:
        cached: list[float] | None = self.cache.get(text)
        if cached is not None:
            return cached

        try:
            response = self.client.embeddings.create(
                model=settings.embedding_model_name,
                input=text,
            )

            embedding: list[float] = list(response.data[0].embedding)

            self.cache.set(text, embedding)

            return embedding

        except Exception as error:
            raise EmbeddingServiceError(str(error)) from error
