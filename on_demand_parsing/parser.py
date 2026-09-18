from query_normalizer import normalize_query
from openalex_client import search_works, search_works_by_topic
from topic_discovery import discover_topics
from topic_scoring import score_topics, select_topics


def run_parser(user_query):
    query = normalize_query(user_query)
    print("OpenAlex query:", query)

    works = search_works(query, limit=1000)
    print("Seed works:", len(works))

    topics = discover_topics(works)
    print("Unique topics:", len(topics))

    scored_topics = score_topics(query, topics)

    selected_topics = select_topics(
        scored_topics,
        len(works)
    )

    print("Selected topics:", len(selected_topics))

    # Topic Expansion
    print("\nTOPIC EXPANSION:\n")

    for topic in selected_topics:
        expanded_works = search_works_by_topic(
            topic["id"],
            limit=100
        )

        print(
            f'{topic["name"]}: '
            f'{len(expanded_works)} works'
        )

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