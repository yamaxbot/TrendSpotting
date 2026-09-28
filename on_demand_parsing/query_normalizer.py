import logging
import time
from openai import OpenAIError

from vsellm_chat import CHAT_TIMEOUT_SECONDS, chat_completion


MAX_NORMALIZATION_ATTEMPTS = 2
logger = logging.getLogger(__name__)


PROMPT = """Convert the request into a concise English OpenAlex search query.
Keep only the scientific/technical subject and essential domain constraints. Remove
search intent and novelty words. Translate accurately; never add concepts. Output
only the query, without quotes or commentary."""


def normalize_query(query):
    deadline = time.monotonic() + CHAT_TIMEOUT_SECONDS
    for attempt in range(1, MAX_NORMALIZATION_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            result = chat_completion(
                PROMPT, query, max_tokens=256,
                temperature=0.0, timeout=remaining,
            )
            if not result.strip():
                raise ValueError("Empty search query")
            normalized = result.strip().splitlines()[0].strip(" `*_\"'")
            if not normalized:
                raise ValueError("Empty normalized query")
            return normalized
        except (OpenAIError, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                "Query normalization attempt %s/%s failed: %s",
                attempt, MAX_NORMALIZATION_ATTEMPTS, type(exc).__name__,
            )
    raise RuntimeError("Не удалось подготовить поисковый запрос")
