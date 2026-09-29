from __future__ import annotations

import json
import logging

from .common import Client, cached_get, database, short_id

LOG = logging.getLogger(__name__)


def growth(counts: dict[int, int], year: int, epsilon: float) -> float:
    before = sum(counts.get(y, 0) for y in range(year - 2, year + 1))
    after = sum(counts.get(y, 0) for y in range(year + 3, year + 6))
    return (after + epsilon) / (before + epsilon)


def topic_publications(connection, client: Client, topic: str, config: dict) -> dict[int, int]:
    lower, upper = config["start_year"] - 2, config["training_end_year"] + 5
    response = cached_get(connection, client, f"topic_counts:{topic}:{lower}:{upper}", "works",
                          filter=f"primary_topic.id:{topic},publication_year:{lower}-{upper}",
                          group_by="publication_year", per_page=100)
    return {int(row["key"]): row["count"] for row in response["group_by"]}


def new_authors(connection, client: Client, topic: str, end_year: int) -> dict[int, int]:
    state_key = f"{topic}:{end_year}"
    with connection:
        connection.execute("INSERT OR IGNORE INTO topic_cursor VALUES (?, '*', 0)", (state_key,))
    while True:
        cursor, complete = connection.execute(
            "SELECT cursor,complete FROM topic_cursor WHERE topic=?", (state_key,)
        ).fetchone()
        if complete:
            break
        response = client.get("works", filter=f"primary_topic.id:{topic},publication_year:<{end_year + 1}",
                              select="id,publication_year,authorships", per_page=100, cursor=cursor)
        next_cursor = response["meta"].get("next_cursor")
        with connection:
            for work in response["results"]:
                year = work.get("publication_year")
                if year is None:
                    continue
                for authorship in work.get("authorships", []):
                    author = short_id((authorship.get("author") or {}).get("id"))
                    if author:
                        connection.execute("""
                            INSERT INTO topic_authors VALUES (?,?,?)
                            ON CONFLICT(topic,author) DO UPDATE
                            SET first_year=min(first_year,excluded.first_year)
                        """, (state_key, author, year))
            connection.execute("UPDATE topic_cursor SET cursor=?,complete=? WHERE topic=?",
                               (next_cursor, int(not next_cursor or not response["results"]), state_key))
    return dict(connection.execute(
        "SELECT first_year,count(*) FROM topic_authors WHERE topic=? GROUP BY first_year", (state_key,)
    ))


def label(config: dict, client: Client) -> None:
    connection = database(config)
    topics = connection.execute("""
        SELECT DISTINCT json_extract(payload,'$.primary_topic_id') FROM documents
        WHERE json_extract(payload,'$.pub_year') <= ?
        ORDER BY 1
    """, (config["training_end_year"],)).fetchall()
    for (topic,) in topics:
        pending = connection.execute("""
            SELECT doc_id,payload FROM documents
            WHERE json_extract(payload,'$.primary_topic_id')=?
              AND json_extract(payload,'$.target_status')='pending'
        """, (topic,)).fetchall()
        if not pending:
            continue
        counts = topic_publications(connection, client, topic, config)
        records = [(doc_id, json.loads(payload)) for doc_id, payload in pending]
        positives = []
        with connection:
            for doc_id, record in records:
                ratio = growth(counts, record["pub_year"], config["epsilon"])
                record["future_growth_ratio"] = ratio
                record["target_population"] = "all_primary_topic_works"
                if ratio < 4:
                    record["target_emergence"] = 0
                    record["target_status"] = "publication_growth_below_threshold"
                    record["split"] = "train"
                    connection.execute("UPDATE documents SET payload=? WHERE doc_id=?",
                                       (json.dumps(record, ensure_ascii=False), doc_id))
                else:
                    positives.append((doc_id, record))
        if positives:
            authors = new_authors(connection, client, topic, config["training_end_year"] + 5)
            with connection:
                for doc_id, record in positives:
                    year = record["pub_year"]
                    before = sum(authors.get(y, 0) for y in range(year - 2, year + 1))
                    after = sum(authors.get(y, 0) for y in range(year + 3, year + 6))
                    record["new_authors_baseline"] = before
                    record["new_authors_future"] = after
                    if before == 0:
                        record["target_status"] = "insufficient_author_baseline"
                    else:
                        ratio = after / before
                        record["new_authors_growth_ratio"] = ratio
                        record["target_emergence"] = int(ratio >= 2.5)
                        record["target_status"] = "complete"
                        record["split"] = "train"
                    connection.execute("UPDATE documents SET payload=? WHERE doc_id=?",
                                       (json.dumps(record, ensure_ascii=False), doc_id))
        LOG.info("Topic %s labeled; author scan required for %s records", topic, len(positives))
    connection.close()


def historical_citations(client: Client, connection, doc_id: str, pub_year: int, cutoff: int) -> dict:
    response = cached_get(connection, client, f"citations:{doc_id}:{cutoff}", "works",
                          filter=f"cites:{doc_id},publication_year:{pub_year}-{cutoff}",
                          group_by="publication_year", per_page=100)
    counts = {int(row["key"]): row["count"] for row in response["group_by"]}
    return {"citations_historical": sum(counts.values()),
            "citation_counts_historical": counts, "feature_cutoff_year": cutoff}
