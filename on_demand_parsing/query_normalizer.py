import os
import logging
import requests
from dotenv import load_dotenv


load_dotenv()


API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")
MAX_NORMALIZATION_ATTEMPTS = 2
logger = logging.getLogger(__name__)


PROMPT = """Convert the request into a concise English OpenAlex search query.
Keep only the scientific/technical subject and essential domain constraints. Remove
search intent and novelty words. Translate accurately; never add concepts. Output
only the query, without quotes or commentary."""


def normalize_query(query):
    if not API_KEY:
        raise RuntimeError("KIE_API_KEY не задан")
    for attempt in range(1, MAX_NORMALIZATION_ATTEMPTS + 1):
        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "temperature": 0.0,
                    "max_tokens": 40,
                    "messages": [
                        {"role": "system", "content": PROMPT},
                        {"role": "user", "content": query},
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            if not isinstance(result, str) or not result.strip():
                raise ValueError("Empty search query")
            normalized = result.strip().splitlines()[0].strip(" `*_\"'")
            if not normalized:
                raise ValueError("Empty normalized query")
            return normalized
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning(
                "Query normalization attempt %s/%s failed: %s",
                attempt, MAX_NORMALIZATION_ATTEMPTS, type(exc).__name__,
            )
    raise RuntimeError("Не удалось подготовить поисковый запрос")
