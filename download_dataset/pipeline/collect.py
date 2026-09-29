from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import logging
import math

from .common import Client, Paused, database, short_id
from .export import checkpoint_batches

LOG = logging.getLogger(__name__)
FIELDS = ",".join([
    "id", "doi", "publication_year", "publication_date", "title",
    "abstract_inverted_index", "type", "primary_location", "primary_topic",
    "topics", "authorships", "cited_by_count", "referenced_works", "counts_by_year",
])
QUALITY = "has_abstract:true,referenced_works_count:>0"


def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("Empty eligible population")
    exact = {key: total * weight / denominator for key, weight in weights.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    ranked = sorted(exact, key=lambda key: (-(exact[key] - result[key]), key))
    for key in ranked[: total - sum(result.values())]:
        result[key] += 1
    return result


def plan(config: dict, client: Client) -> None:
    connection = database(config)
    identity = {key: config[key] for key in (
        "start_year", "end_year", "total", "seed", "engineering_min_share_of_physical"
    )}
    fingerprint = json.dumps(identity, sort_keys=True)
    saved = connection.execute("SELECT value FROM metadata WHERE key='sampling_config'").fetchone()
    if saved:
        if saved[0] != fingerprint:
            raise ValueError("Sampling config changed. Use a new data_dir for a different corpus.")
        connection.close()
        return
    years = allocate(config["total"], {
        str(year): 1 for year in range(config["start_year"], config["end_year"] + 1)
    })
    strata = []
    for year, year_quota in years.items():
        base = f"publication_year:{year},{QUALITY}"
        groups = client.get("works", filter=base, group_by="primary_topic.domain.id", per_page=100)
        populations = {short_id(row["key"]): row["count"] for row in groups["group_by"]
                       if short_id(row["key"]) in {"1", "2", "3", "4"}}
        if len(populations) != 4:
            raise ValueError(f"Expected four domains in {year}: {populations}")
        quotas = allocate(year_quota, populations)
        for domain, quota in sorted(quotas.items()):
            domain_filter = f"{base},primary_topic.domain.id:{domain}"
            if domain != "3":
                strata.append((f"{year}-{domain}", domain_filter, quota, populations[domain]))
                continue
            engineering_filter = f"{domain_filter},primary_topic.field.id:22"
            engineering_count = client.get("works", filter=engineering_filter, per_page=1, select="id")["meta"]["count"]
            share = max(config["engineering_min_share_of_physical"], engineering_count / populations[domain])
            engineering_quota = min(quota, math.ceil(quota * share))
            strata.extend([
                (f"{year}-3-engineering", engineering_filter, engineering_quota, engineering_count),
                (f"{year}-3-other", f"{domain_filter},primary_topic.field.id:!22",
                 quota - engineering_quota, populations[domain] - engineering_count),
            ])
    for name, _, quota, population in strata:
        if quota > population:
            raise ValueError(f"Insufficient population for {name}: {population} < {quota}")
    with connection:
        connection.execute("INSERT INTO metadata VALUES ('sampling_config',?)", (fingerprint,))
        for index, (name, filters, quota, population) in enumerate(strata):
            connection.execute("INSERT INTO strata(name,filter,quota,population,seed) VALUES (?,?,?,?,?)",
                               (name, filters, quota, population, config["seed"] + index * 10000))
    connection.close()


def abstract(index: dict | None) -> str | None:
    if not index:
        return None
    tokens = {}
    for word, positions in index.items():
        for position in positions:
            if not isinstance(position, int) or position < 0 or position in tokens:
                return None
            tokens[position] = word
    if not tokens or len(tokens) != max(tokens) + 1:
        return None
    return " ".join(tokens[position] for position in range(len(tokens)))


def normalize(work: dict, config: dict, stratum: str) -> dict | None:
    text = abstract(work.get("abstract_inverted_index"))
    year = work.get("publication_year")
    topic = work.get("primary_topic") or {}
    references = [short_id(value) for value in work.get("referenced_works", []) if value]
    if (not text or not work.get("id") or not isinstance(year, int)
            or not config["start_year"] <= year <= config["end_year"]
            or not references or not topic.get("id")):
        return None
    source = (work.get("primary_location") or {}).get("source") or {}
    source_type, work_type = source.get("type"), work.get("type")
    tier = (work_type if work_type in {"preprint", "patent"} else
            "conference" if source_type == "conference" else
            "journal" if source_type == "journal" else "other")
    authorships = work.get("authorships") or []
    authors = list(dict.fromkeys(short_id(item["author"]["id"]) for item in authorships
                                if (item.get("author") or {}).get("id")))
    institutions = [institution for item in authorships for institution in item.get("institutions", [])]
    commercial = sum(any(i.get("type") == "company" for i in a.get("institutions", [])) for a in authorships)
    observed = sum(bool(a.get("institutions")) for a in authorships)
    fetched = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "doc_id": short_id(work["id"]), "pub_year": year, "pub_date": work.get("publication_date"),
        "title": work.get("title"), "abstract_text": text, "source_tier": tier,
        "primary_topic": topic.get("display_name"), "primary_topic_id": short_id(topic["id"]),
        "domain_id": (topic.get("domain") or {}).get("id"),
        "field_id": (topic.get("field") or {}).get("id"),
        "authors_ids": authors,
        "affiliations": sorted({i["display_name"] for i in institutions if i.get("display_name")}),
        "citations_at_cutoff": work.get("cited_by_count"),
        "referenced_works_ids": references,
        "counts_by_year": json.dumps(work.get("counts_by_year"), ensure_ascii=False),
        "authorships_json": json.dumps(authorships, ensure_ascii=False),
        "topics_json": json.dumps(work.get("topics"), ensure_ascii=False),
        "source_id": source.get("id"), "doi": work.get("doi"), "fetched_at": fetched,
        "stratum": stratum, "is_engineering": short_id((topic.get("field") or {}).get("id")) == "22",
        "commercial_maturity_index": commercial / len(authorships) if observed == len(authorships) and observed else None,
        "affiliation_coverage": observed / len(authorships) if authorships else None,
        "has_ref_data": 1,
        "target_emergence": None,
        "target_status": "pending" if year <= config["training_end_year"] else "future_window_incomplete",
        "split": "train_candidates" if year <= config["training_end_year"] else "inference",
    }


def collect_stratum(config: dict, client: Client, name: str) -> None:
    connection = database(config)
    try:
        while True:
            filters, quota, population, seed, page = connection.execute(
                "SELECT filter,quota,population,seed,page FROM strata WHERE name=?", (name,)
            ).fetchone()
            count = connection.execute("SELECT count(*) FROM documents WHERE stratum=?", (name,)).fetchone()[0]
            if count >= quota:
                return
            sample_size = min(10000, population)
            result = client.get("works", filter=filters, sample=sample_size, seed=seed,
                                page=page, per_page=100, select=FIELDS)
            rows = result.get("results", [])
            with connection:
                for work in rows:
                    record = normalize(work, config, name)
                    if record is not None and count < quota:
                        inserted = connection.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)",
                            (record["doc_id"], name, json.dumps(record, ensure_ascii=False))).rowcount
                        count += inserted
                if not rows or page * 100 >= sample_size:
                    seed, page = seed + 1, 1
                else:
                    page += 1
                connection.execute("UPDATE strata SET seed=?,page=? WHERE name=?", (seed, page, name))
            if page % 20 == 0 or count == quota:
                LOG.info("%s: %s / %s", name, count, quota)
    finally:
        connection.close()


def collect(config: dict, client: Client) -> None:
    plan(config, client)
    connection = database(config)
    names = [row[0] for row in connection.execute("SELECT name FROM strata ORDER BY name")]
    connection.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["workers"]) as executor:
        futures = [executor.submit(collect_stratum, config, client, name) for name in names]
        try:
            pending = set(futures)
            while pending:
                completed, pending = concurrent.futures.wait(
                    pending, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in completed:
                    future.result()
                checkpoint_batches(config)
        except BaseException:
            for future in futures:
                future.cancel()
            with client.lock:
                client.max_requests = client.calls
            raise
