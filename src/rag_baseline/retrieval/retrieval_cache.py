from rag_baseline.domain_models import RetrievalResult


class RetrievalCache:
    def __init__(self) -> None:
        self.cache: dict[str, list[RetrievalResult]] = {}

    def get(self, query: str) -> list[RetrievalResult] | None:
        return self.cache.get(query)

    def set(self, query: str, results: list[RetrievalResult]) -> None:
        self.cache[query] = results
