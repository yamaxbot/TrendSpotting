import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")
URL = "https://api.openalex.org/works"


# Обычный текстовый поиск
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


# Получение публикаций по OpenAlex Topic
def search_works_by_topic(topic_id, limit=1000, year=None):
    works = []
    cursor = "*"

    topic_id = topic_id.split("/")[-1]

    filters = [f"topics.id:{topic_id}"]

    if year:
        filters.append(f"publication_year:{year}")

    while len(works) < limit:
        response = requests.get(
            URL,
            params={
                "filter": ",".join(filters),
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


# Реальное количество публикаций Topic за конкретный год
def get_topic_year_count(topic_id, year):
    topic_id = topic_id.split("/")[-1]

    response = requests.get(
        URL,
        params={
            "filter": (
                f"topics.id:{topic_id},"
                f"publication_year:{year}"
            ),
            "per-page": 1,
            "api_key": API_KEY
        },
        timeout=30
    )

    data = response.json()

    return data["meta"]["count"]