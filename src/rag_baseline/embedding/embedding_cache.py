class EmbeddingCache:
    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def get(self, text: str) -> list[float] | None:
        return self._cache.get(text)

    def set(self, text: str, embedding: list[float]) -> None:
        self._cache[text] = embedding
