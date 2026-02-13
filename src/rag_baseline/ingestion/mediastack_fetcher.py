from datetime import datetime

import requests
from rag_baseline.configuration import settings
from rag_baseline.custom_exceptions import ExternalAPIError
from rag_baseline.domain_models import NewsArticle


class MediaStackClient:
    BASE_URL = "http://api.mediastack.com/v1/news"

    def fetch_latest_articles(self, limit: int = 20) -> list[NewsArticle]:
        params = {
            "access_key": settings.mediastack_api_key,
            "languages": "en",
            "limit": limit,
        }

        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise ExternalAPIError(f"MediaStack error: {response.text}")

        data = response.json().get("data", [])

        articles = []
        for item in data:
            if not item.get("content"):
                continue

            articles.append(
                NewsArticle(
                    article_id=item["url"],
                    title=item["title"],
                    content=item["content"],
                    source=item["source"],
                    published_at=datetime.fromisoformat(
                        item["published_at"].replace("Z", "+00:00")
                    ),
                    url=item["url"],
                )
            )

        return articles
