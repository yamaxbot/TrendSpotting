import os
import requests
from dotenv import load_dotenv


load_dotenv()


API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")


PROMPT = """Convert the request into a concise English OpenAlex search query.
Keep only the scientific/technical subject and essential domain constraints. Remove
search intent and novelty words. Translate accurately; never add concepts. Output
only the query, without quotes or commentary."""


def normalize_query(query):
    if not API_KEY:
        raise RuntimeError("KIE_API_KEY не задан")
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "temperature": 0.0,
            "max_tokens": 40,
            "messages": [
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    try:
        result = data["choices"][0]["message"]["content"]
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Empty search query")
        normalized = result.strip().splitlines()[0].strip(" `*_\"'")
        if not normalized:
            raise ValueError("Empty normalized query")
        return normalized
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Не удалось подготовить поисковый запрос") from exc
