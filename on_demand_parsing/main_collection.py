"""Collect a bounded OpenAlex corpus for on-demand ranking."""

from concurrent.futures import ThreadPoolExecutor

from on_demand_parsing.openalex_client import (
    search_works,
    search_works_by_topic_with_count,
)


MAIN_LIMIT = 5000
EXPLORATION_LIMIT = 1000
OPENALEX_WORKERS = 3


def deduplicate_works(works):
    unique = {}
    for work in works:
        unique[work["id"]] = work
    return list(unique.values())


def collect_main_corpus(selected_topics, query, years, seed_works=None):
    works = []
    yearly_stats = {}

    if selected_topics:
        limit_per_topic_year = max(
            1, MAIN_LIMIT // (len(selected_topics) * len(years))
        )
        # Each task owns its HTTP session. Keep concurrency modest for OpenAlex.
        with ThreadPoolExecutor(max_workers=OPENALEX_WORKERS) as executor:
            futures = {
                (topic_index, year): executor.submit(
                    search_works_by_topic_with_count,
                    topic["id"],
                    limit=limit_per_topic_year,
                    year=year,
                )
                for topic_index, topic in enumerate(selected_topics)
                for year in years
            }
            # Consume in the previous order to preserve deduplication order.
            for topic_index, topic in enumerate(selected_topics):
                topic_name = topic["name"]
                print(f"\nCollecting: {topic_name}")
                topic_works = []
                yearly_stats[topic_name] = {}
                for year in years:
                    year_works, count = futures[(topic_index, year)].result()
                    topic_works.extend(year_works)
                    yearly_stats[topic_name][year] = count
                works.extend(topic_works)
                print(f"Collected: {len(topic_works)}")

    print("\nEXPLORATION:")
    exploration_works = (
        seed_works if seed_works is not None
        else search_works(query, limit=EXPLORATION_LIMIT, years=years)
    )
    print("Exploration works:", len(exploration_works))
    works.extend(exploration_works)

    before = len(works)
    works = deduplicate_works(works)
    print("\nDEDUPLICATION:")
    print("Before:", before)
    print("After:", len(works))
    print("Removed:", before - len(works))
    return works, yearly_stats
