from query_normalizer import normalize_query
from openalex_client import search_works
from topic_discovery import discover_topics
from topic_scoring import score_topics, select_topics


def run_parser(user_query):
    # 1. Нормализация запроса
    query = normalize_query(user_query)
    print("OpenAlex query:", query)

    # 2. Seed Search
    works = search_works(query, limit=1000)
    print("Seed works:", len(works))

    # 3. Topic Discovery
    topics = discover_topics(works)
    print("Unique topics:", len(topics))

    # 4. Topic Scoring
    scored_topics = score_topics(query, topics)

    # 5. Topic Selection
    selected_topics = select_topics(
        scored_topics,
        len(works)
    )

    print("Selected topics:", len(selected_topics))

    return selected_topics


if __name__ == "__main__":
    user_query = input("Введите направление: ")

    topics = run_parser(user_query)

    print("\nSELECTED TOPICS:\n")

    for topic in topics:
        print(
            f'{topic["similarity"]:.3f} | '
            f'{topic["frequency"]:3} | '
            f'{topic["coverage"]:.1%} | '
            f'{topic["name"]}'
        )