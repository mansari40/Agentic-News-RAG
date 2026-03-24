"""
Small helper functions shared across the pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime as _parse_rfc2822
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def strip_non_ascii(text: str) -> str:
    """Strip non-ASCII characters for Tavily"""
    return text.encode("ascii", errors="ignore").decode("ascii").strip()


_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def parse_date(raw: str | None) -> datetime | None:
    """Try to parse a date string into an aware datetime. Returns None if it can't."""
    if not raw:
        return None
    s = raw.strip()

    try:
        return _parse_rfc2822(s)
    except Exception:
        pass
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s[:25])
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except (ValueError, TypeError):
        pass
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s[:25], fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
    logger.debug(f"Could not parse date: {raw!r}")
    return None


def parse_min_date(date_str: str) -> datetime:
    """Parse the config cutoff date string into an aware datetime for comparisons."""
    dt = datetime.fromisoformat(date_str)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def extract_token_cost(
    response: Any,
    input_cost_per_token: float,
    output_cost_per_token: float,
) -> tuple[int, int, float]:
    """Pull token counts from a LangChain response and calculate the cost."""
    usage = getattr(response, "usage_metadata", None) or {}
    tokens_in = int(usage.get("input_tokens", 0))
    tokens_out = int(usage.get("output_tokens", 0))
    cost = tokens_in * input_cost_per_token + tokens_out * output_cost_per_token
    return tokens_in, tokens_out, cost


_TEMPORAL_BYPASS_KEYWORDS = frozenset(
    {
        "latest",
        "recent",
        "today",
        "current",
        "now",
        "just",
        "breaking",
        "this week",
        "this month",
        "neue",
        "aktuell",
        "heute",
        "jetzt",
    }
)

_SOCIAL_EXACT = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
        "goodbye",
        "how are you",
        "what's up",
        "good morning",
        "good afternoon",
        "good evening",
        "nice to meet",
        "see you",
    }
)

_SOCIAL_PREFIXES = ("hi ", "hello ", "hey ", "thanks ", "bye ")


def is_temporal_query(query: str) -> bool:
    """Returns True if the query is asking for current or recent information"""
    q = query.lower()
    return any(kw in q for kw in _TEMPORAL_BYPASS_KEYWORDS)


def is_pure_social(query: str) -> bool:
    """
    Returns True for greetings and small-talk that carry no actual question.
    Uses pattern matching only — no LLM call needed.
    """
    q = query.lower().strip()
    if q in _SOCIAL_EXACT:
        return True
    words = q.split()
    if len(words) <= 4 and any(q.startswith(p) for p in _SOCIAL_PREFIXES):
        from agentic_rag.domain import is_timber_related

        return not is_timber_related(q)
    return False


def with_year(text: str) -> str:
    """Add the current year to a search query if it's not already in there."""
    year = str(datetime.now().year)
    return text if year in text else f"{text} {year}"
