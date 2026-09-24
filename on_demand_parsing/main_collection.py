from on_demand_parsing.openalex_client import (
    search_works,
    search_works_by_topic,
    get_topic_year_count
)


# Для разработки пока используем небольшие лимиты
MAIN_LIMIT = 5000
EXPLORATION_LIMIT = 1000


def deduplicate_works(works):
    unique = {}

    for work in works:
        work_id = work["id"]
        unique[work_id] = work

    return list(unique.values())


def collect_main_corpus(selected_topics, query, years):
    works = []
    yearly_stats = {}

    # 1. Main Collection по Selected Topics
    if selected_topics:
        limit_per_topic_year = max(
            1,
            MAIN_LIMIT // (len(selected_topics) * len(years)),
        )

        for topic in selected_topics:
            topic_id = topic["id"]
            topic_name = topic["name"]

            print(f"\nCollecting: {topic_name}")

            topic_works = []
            for year in years:
                topic_works.extend(
                    search_works_by_topic(
                        topic_id,
                        limit=limit_per_topic_year,
                        year=year,
                    )
                )

            works.extend(topic_works)

            print(f"Collected: {len(topic_works)}")

            # Реальное количество публикаций
            # этого Topic по годам
            yearly_stats[topic_name] = {}

            for year in years:
                count = get_topic_year_count(
                    topic_id,
                    year
                )

                yearly_stats[topic_name][year] = count

    # 2. Exploration
    print("\nEXPLORATION:")

    exploration_works = search_works(
        query,
        limit=EXPLORATION_LIMIT,
        years=years,
    )

    print("Exploration works:", len(exploration_works))

    works.extend(exploration_works)

    # 3. Deduplication
    before = len(works)

    works = deduplicate_works(works)

    print("\nDEDUPLICATION:")
    print("Before:", before)
    print("After:", len(works))
    print("Removed:", before - len(works))

    return works, yearly_stats
