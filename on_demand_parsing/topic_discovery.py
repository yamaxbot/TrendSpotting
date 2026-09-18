def discover_topics(works):
    topics = {}

    for work in works:
        for topic in work.get("topics", []):
            name = topic["display_name"]

            if name not in topics:
                topics[name] = {
                    "id": topic["id"],
                    "frequency": 0
                }

            topics[name]["frequency"] += 1

    return topics