import logging

from openai import OpenAI, OpenAIError
from rag_baseline.configuration import settings
from rag_baseline.custom_exceptions import GenerationError
from rag_baseline.domain_models import RetrievalResult
from rag_baseline.utils.retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class AnswerGenerator:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model_name

    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        return "\n\n".join(chunk.content for chunk in chunks)

    @retry_with_backoff(  # type: ignore[misc]
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(OpenAIError, ConnectionError, TimeoutError),
    )
    def _call_openai(self, prompt: str) -> tuple[str, int, int]:
        """Call OpenAI API with retry logic. Returns (answer, prompt_tokens, completion_tokens)."""
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        result: str = content.strip() if content else ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        return result, prompt_tokens, completion_tokens

    # gpt-4o-mini pricing (per 1M tokens)
    _COST_INPUT_PER_TOKEN = 0.15 / 1_000_000
    _COST_OUTPUT_PER_TOKEN = 0.60 / 1_000_000

    def generate(self, question: str, chunks: list[RetrievalResult]) -> tuple[str, int, float]:
        """
        Generate answer with error handling and fallback.

        Args:
            question: User's question
            chunks: Retrieved context chunks

        Returns:
            (answer, total_tokens, cost_usd)
        """
        if not chunks:
            return "No relevant information found in the timber market news database.", 0, 0.0

        context = self._build_context(chunks)

        prompt = f"""You are a timber market analyst assistant specializing \
in the German wood industry.
Answer questions about timber markets, wood prices, forestry, lumber trade, \
and construction materials.
ALWAYS respond in English, even if the source documents are in German.
If source documents are in German, translate the relevant information to English in your answer.
Use ONLY the provided news context. If the answer is not in the context, \
say you don't have that information.

Context from recent timber market news:
{context}

Question:
{question}

Answer:"""

        try:
            answer, prompt_tokens, completion_tokens = self._call_openai(prompt)
            total_tokens = prompt_tokens + completion_tokens
            cost_usd = (
                prompt_tokens * self._COST_INPUT_PER_TOKEN
                + completion_tokens * self._COST_OUTPUT_PER_TOKEN
            )
            return answer, total_tokens, round(cost_usd, 8)
        except GenerationError as e:
            logger.error(f"Answer generation failed: {e}")
            return (
                "I'm experiencing technical difficulties generating an answer. "
                "Please try again in a moment.",
                0,
                0.0,
            )
        except Exception as e:
            logger.error(f"Unexpected error in answer generation: {e}")
            return "An unexpected error occurred.", 0, 0.0
