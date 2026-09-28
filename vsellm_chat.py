import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI


API_BASE_URL = "https://api.vsellm.ru/v1"
MODEL_ID = "qwen/qwen3.7-flash"
CHAT_TIMEOUT_SECONDS = 120

load_dotenv(Path(__file__).resolve().parent / ".env")


@lru_cache(maxsize=1)
def get_client():
    api_key = os.getenv("VSELLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VSELLM_API_KEY не задан")
    return OpenAI(
        api_key=api_key,
        base_url=API_BASE_URL,
        http_client=DefaultHttpxClient(trust_env=False),
        max_retries=0,
    )


def chat_completion(system_prompt, user_content, *, max_tokens=None, temperature=0.2, timeout=30):
    request = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    response = get_client().with_options(timeout=timeout).chat.completions.create(**request)
    if not response.choices:
        raise ValueError("VseLLM вернул ответ без choices")
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("VseLLM вернул пустой ответ модели")
    return content.strip()
