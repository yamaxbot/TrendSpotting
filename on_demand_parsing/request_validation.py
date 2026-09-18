import requests
import os

API_KEY = os.getenv("KIE_API_KEY")

response = requests.get(
    "https://api.kie.ai/api/v1/chat/credit",
    headers={
        "Authorization": f"Bearer {API_KEY}"
    }
)

print(response.json())