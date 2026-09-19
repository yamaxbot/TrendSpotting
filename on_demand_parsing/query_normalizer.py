import os
import requests
from dotenv import load_dotenv


load_dotenv()


API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")


PROMPT = """
Convert the user's request into a concise English search query for OpenAlex.

Rules:
- The output MUST be in English.
- Translate Russian and other languages into English.
- Keep only the scientific or technological subject and important constraints.
- Remove intent words such as:
  find, show, new, promising, emerging, weak signals, topics about.
- Do not invent or add technologies.
- Preserve important domain constraints.
- Return ONLY the final English query.

Examples:

"ии"
-> "artificial intelligence"

"кибербезопасность"
-> "cybersecurity"

"новые технологии в робототехнике"
-> "robotics"

"Найди перспективные технологии защиты промышленных систем от кибератак"
-> "industrial control systems cybersecurity"
"""


def normalize_query(query):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
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

    data = response.json()

    return (
        data["choices"][0]["message"]["content"]
        .strip()
    )