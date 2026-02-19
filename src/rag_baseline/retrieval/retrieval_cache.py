import time

from rag_baseline.domain_models import RetrievalResult


class RetrievalCache:
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600) -> None:
        self.cache: dict[str, tuple[list[RetrievalResult], float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _normalize_query(self, query: str) -> str:
        """Normalize query for better cache hits"""
        return query.lower().strip()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, query: str) -> list[RetrievalResult] | None:
        """Get cached results if available and not expired"""
        normalized = self._normalize_query(query)

        if normalized not in self.cache:
            return None

        results, timestamp = self.cache[normalized]

        if self._is_expired(timestamp):
            del self.cache[normalized]
            return None

        return results

    def set(self, query: str, results: list[RetrievalResult]) -> None:
        """Store results in cache with timestamp"""
        normalized = self._normalize_query(query)

        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[normalized] = (results, time.time())

    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
