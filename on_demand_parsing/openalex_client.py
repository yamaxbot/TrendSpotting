import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")
URL = "https://api.openalex.org/works"


def search_works(query, limit=1000):
    works = []
    cursor = "*"

    while len(works) < limit:
        response = requests.get(
            URL,
            params={
                "search": query,
                "per-page": min(100, limit - len(works)),
                "cursor": cursor,
                "api_key": API_KEY
            },
            timeout=30
        )

        data = response.json()

        works.extend(data["results"])
        cursor = data["meta"]["next_cursor"]

        if not cursor or not data["results"]:
            break

    return works



def search_works_by_topic(topic_id, limit=1000):
    works = []
    cursor = "*"

    # https://openalex.org/T123... → T123...
    topic_id = topic_id.split("/")[-1]

    while len(works) < limit:
        response = requests.get(
            URL,
            params={
                "filter": f"topics.id:{topic_id}",
                "per-page": min(100, limit - len(works)),
                "cursor": cursor,
                "api_key": API_KEY
            },
            timeout=30
        )

        data = response.json()
        works.extend(data["results"])

        cursor = data["meta"]["next_cursor"]

        if not cursor or not data["results"]:
            break

    return works