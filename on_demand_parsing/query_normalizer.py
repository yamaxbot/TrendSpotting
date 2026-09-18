import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.kie.ai/gpt-5-2/v1/chat/completions"
API_KEY = os.getenv("KIE_API_KEY")

PROMPT = """
Convert the request into a concise English OpenAlex search query.
Keep only the scientific/technological subject and important constraints.
Remove user intent such as find, show, topics about, new, promising,
emerging, weak signals.
Do not invent or add technologies.
Return ONLY the query.
"""


def normalize_query(query):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": query}
            ]
        },
        timeout=30
    )

    return response.json()["choices"][0]["message"]["content"].strip()