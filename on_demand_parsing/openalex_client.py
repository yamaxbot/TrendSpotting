import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("OPENALEX_API_KEY")
URL = "https://api.openalex.org/works"
QUALITY_FILTERS = ["has_abstract:true", "referenced_works_count:>0"]


# Обычный текстовый поиск
def search_works(query, limit=1000, years=None, require_references=True):
    works = []
    selected_years = list(years) if years else [None]
    base_limit, remainder = divmod(limit, len(selected_years))

    with requests.Session() as session:
        for index, year in enumerate(selected_years):
            year_limit = base_limit + (1 if index < remainder else 0)
            year_works = []
            cursor = "*"

            while len(year_works) < year_limit:
                filters = list(QUALITY_FILTERS if require_references else ["has_abstract:true"])
                if year is not None:
                    filters.append(f"publication_year:{year}")

                params = {
                    "search": query,
                    "filter": ",".join(filters),
                    "per-page": min(100, year_limit - len(year_works)),
                    "cursor": cursor,
                    "api_key": API_KEY,
                }

                response = session.get(URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                year_works.extend(data["results"])
                cursor = data["meta"]["next_cursor"]

                if not cursor or not data["results"]:
                    break

            works.extend(year_works)

    return works


# Получение публикаций по OpenAlex Topic
def search_works_by_topic(topic_id, limit=1000, year=None):
    works, _ = search_works_by_topic_with_count(topic_id, limit=limit, year=year)
    return works


def search_works_by_topic_with_count(topic_id, limit=1000, year=None):
    """Return a page-limited topic sample and the full OpenAlex match count."""
    works = []
    cursor = "*"
    total_count = 0

    topic_id = topic_id.split("/")[-1]

    filters = [f"topics.id:{topic_id}", *QUALITY_FILTERS]

    if year:
        filters.append(f"publication_year:{year}")

    with requests.Session() as session:
        while len(works) < limit:
            response = session.get(
                URL,
                params={
                    "filter": ",".join(filters),
                    "per-page": min(100, limit - len(works)),
                    "cursor": cursor,
                    "api_key": API_KEY,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            total_count = int(data["meta"]["count"])
            page = data["results"]
            works.extend(page)
            cursor = data["meta"].get("next_cursor")
            if not cursor or not page:
                break

    return works, total_count


# Реальное количество публикаций Topic за конкретный год
def get_topic_year_count(topic_id, year):
    topic_id = topic_id.split("/")[-1]

    response = requests.get(
        URL,
        params={
            "filter": (
                f"topics.id:{topic_id},"
                f"publication_year:{year},"
                + ",".join(QUALITY_FILTERS)
            ),
            "per-page": 1,
            "api_key": API_KEY
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    return data["meta"]["count"]
