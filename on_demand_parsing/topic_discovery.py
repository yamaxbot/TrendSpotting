from collections import Counter


def discover_topics(works):
    topics = Counter()

    for work in works:
        for topic in work.get("topics", []):
            topics[topic["display_name"]] += 1

    return topics