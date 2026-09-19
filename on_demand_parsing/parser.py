from query_normalizer import normalize_query
from openalex_client import search_works
from topic_discovery import discover_topics
from topic_scoring import score_topics, select_topics
from main_collection import collect_main_corpus
from normalization import normalize_works
from relevance_filter import score_relevance, filter_relevant
from storage import save_results


def run_parser(user_query):
    # 1. Query Normalization
    query = normalize_query(user_query)

    print("OpenAlex query:", query)

    # 2. Seed Search
    seed_works = search_works(
        query,
        limit=1000
    )

    print("Seed works:", len(seed_works))

    # 3. Topic Discovery
    topics = discover_topics(seed_works)

    print("Unique topics:", len(topics))

    # 4. Topic Scoring
    scored_topics = score_topics(
        query,
        topics
    )

    # 5. Topic Selection
    selected_topics = select_topics(
        scored_topics,
        len(seed_works)
    )

    print("Selected topics:", len(selected_topics))

    print("\nSELECTED TOPICS:\n")

    for topic in selected_topics:
        print(
            f'{topic["similarity"]:.3f} | '
            f'{topic["frequency"]:3} | '
            f'{topic["coverage"]:.1%} | '
            f'{topic["name"]}'
        )

    # 6. Main Collection + Exploration + Deduplication
    main_works, yearly_stats = collect_main_corpus(
        selected_topics,
        query
    )

    print("\nMAIN COLLECTION:")
    print("Unique works:", len(main_works))

    # 7. Normalization
    normalized_works = normalize_works(
        main_works
    )

    print("\nNORMALIZATION:")
    print("Normalized works:", len(normalized_works))

    # 8. Relevance Scoring
    scored_works = score_relevance(
        query,
        normalized_works
    )

    # 9. Relevance Filtering
    clean_works = filter_relevant(
        scored_works,
        remove_ratio=0.15
    )

    print("\nRELEVANCE FILTERING:")
    print("Before:", len(scored_works))
    print("After:", len(clean_works))
    print(
        "Removed:",
        len(scored_works) - len(clean_works)
    )

    if clean_works:
        print(
            "Lowest kept relevance:",
            f'{clean_works[-1]["relevance"]:.3f}'
        )

    # 10. Yearly Statistics
    print("\nYEARLY STATS:")

    for topic_name, years in yearly_stats.items():
        print(f"\n{topic_name}")

        for year, count in years.items():
            print(f"  {year}: {count}")

    # 11. Save Results
    save_results(
        raw_works=main_works,
        clean_works=clean_works,
        yearly_stats=yearly_stats
    )

    return clean_works, yearly_stats


if __name__ == "__main__":
    user_query = input("Введите направление: ")

    clean_works, yearly_stats = run_parser(
        user_query
    )