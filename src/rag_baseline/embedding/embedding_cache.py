import hashlib
import time


class EmbeddingCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 86400) -> None:
        self.cache: dict[str, tuple[list[float], float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text hash"""
        return hashlib.md5(text.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - timestamp) > self.ttl_seconds

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding if available and not expired"""
        cache_key = self._get_cache_key(text)

        if cache_key not in self.cache:
            return None

        embedding, timestamp = self.cache[cache_key]

        if self._is_expired(timestamp):
            del self.cache[cache_key]
            return None

        return embedding

    def set(self, text: str, embedding: list[float]) -> None:
        """Store embedding in cache with timestamp"""
        cache_key = self._get_cache_key(text)

        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        self.cache[cache_key] = (embedding, time.time())

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
