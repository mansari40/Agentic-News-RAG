import logging
from datetime import datetime

import requests
from rag_baseline.configuration import settings
from rag_baseline.custom_exceptions import ExternalAPIError
from rag_baseline.domain_models import NewsArticle

logger = logging.getLogger(__name__)


class MediaStackClient:
    BASE_URL = "http://api.mediastack.com/v1/news"

    def fetch_latest_articles(self, limit: int = 100) -> list[NewsArticle]:
        # Paid tier supports max 3 keywords
        keywords = "timber,lumber,wood"

        params = {
            "access_key": settings.mediastack_api_key,
            "keywords": keywords,
            "languages": "en",
            "limit": limit,
        }

        logger.info(f"Fetching timber market articles (keywords: {keywords}, limit={limit})...")
        response = requests.get(self.BASE_URL, params=params)

        if response.status_code != 200:
            raise ExternalAPIError(f"MediaStack error: {response.text}")

        data = response.json().get("data", [])
        logger.info(f"MediaStack returned {len(data)} articles")

        articles = []
        skipped = 0

        for item in data:
            text_content = item.get("content") or item.get("description")

            if not text_content:
                skipped += 1
                continue

            try:
                published_at = datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping article with invalid timestamp: {e}")
                skipped += 1
                continue

            articles.append(
                NewsArticle(
                    article_id=item["url"],
                    title=item.get("title", "No title"),
                    content=text_content,
                    source=item.get("source", "unknown"),
                    published_at=published_at,
                    url=item["url"],
                )
            )

        logger.info(f"Successfully parsed {len(articles)} timber articles (skipped {skipped})")
        return articles
