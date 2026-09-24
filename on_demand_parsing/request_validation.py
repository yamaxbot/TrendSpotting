import os
from pathlib import Path

import requests
from dotenv import load_dotenv


# Загружаем .env из корня проекта TrendSpotting
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH)

api_key = os.getenv("KIE_API_KEY")

if not api_key:
    raise ValueError(
        f"KIE_API_KEY не найден. Проверь файл: {ENV_PATH}"
    )

response = requests.get(
    "https://api.kie.ai/api/v1/chat/credit",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    timeout=30,
)

print("HTTP status:", response.status_code)
print(response.json())