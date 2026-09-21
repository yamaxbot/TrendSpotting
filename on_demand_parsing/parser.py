import datetime as dt
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from on_demand_parsing.query_normalizer import normalize_query
from on_demand_parsing.openalex_client import search_works
from on_demand_parsing.topic_discovery import discover_topics
from on_demand_parsing.topic_scoring import score_topics, select_topics
from on_demand_parsing.main_collection import collect_main_corpus
from on_demand_parsing.normalization import normalize_works
from on_demand_parsing.relevance_filter import score_relevance, filter_relevant
from on_demand_parsing.storage import save_results


# Та же схема, что у датасета на 1 млн статей

STRINGS = [
    "doc_id",
    "pub_date",
    "title",
    "abstract_text",
    "source_tier",
    "primary_topic",
    "primary_topic_id",
    "domain_id",
    "field_id",
    "counts_by_year",
    "authorships_json",
    "topics_json",
    "source_id",
    "doi",
    "fetched_at",
    "stratum",
    "target_status",
    "target_population",
    "split",
]

INTEGERS = [
    "pub_year",
    "citations_at_cutoff",
    "has_ref_data",
    "target_emergence",
    "new_authors_baseline",
    "new_authors_future",
]

FLOATS = [
    "commercial_maturity_index",
    "affiliation_coverage",
    "future_growth_ratio",
    "new_authors_growth_ratio",
]

LISTS = [
    "authors_ids",
    "affiliations",
    "referenced_works_ids",
]


SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRINGS]
    + [(name, pa.int64()) for name in INTEGERS]
    + [(name, pa.float64()) for name in FLOATS]
    + [(name, pa.list_(pa.string())) for name in LISTS]
    + [("is_engineering", pa.bool_())]
)


def short_id(value):
    if not value:
        return None

    return str(value).rsplit("/", 1)[-1]


def make_parquet_record(work, query):
    authorships = work.get("authorships") or []

    # -------------------------
    # Authors
    # -------------------------

    authors_ids = []

    for authorship in authorships:
        author = authorship.get("author") or {}
        author_id = short_id(author.get("id"))

        if author_id and author_id not in authors_ids:
            authors_ids.append(author_id)

    # -------------------------
    # Institutions
    # -------------------------

    institutions = [
        institution
        for authorship in authorships
        for institution in authorship.get("institutions", [])
    ]

    affiliations = sorted({
        institution["display_name"]
        for institution in institutions
        if institution.get("display_name")
    })

    # -------------------------
    # Commercial maturity
    # -------------------------

    commercial = sum(
        any(
            institution.get("type") == "company"
            for institution in authorship.get("institutions", [])
        )
        for authorship in authorships
    )

    observed = sum(
        bool(authorship.get("institutions"))
        for authorship in authorships
    )

    if (
        authorships
        and observed == len(authorships)
        and observed
    ):
        commercial_maturity_index = (
            commercial / len(authorships)
        )
    else:
        commercial_maturity_index = None

    if authorships:
        affiliation_coverage = (
            observed / len(authorships)
        )
    else:
        affiliation_coverage = None

    # -------------------------
    # Source tier
    # -------------------------

    source_type = work.get("source_type")
    work_type = work.get("type")

    if work_type in {"preprint", "patent"}:
        source_tier = work_type

    elif source_type == "conference":
        source_tier = "conference"

    elif source_type == "journal":
        source_tier = "journal"

    else:
        source_tier = "other"

    # -------------------------
    # References
    # -------------------------

    references = [
        short_id(value)
        for value in work.get("referenced_works_ids", [])
        if value
    ]

    # -------------------------
    # Field
    # -------------------------

    field_id = short_id(work.get("field_id"))

    # -------------------------
    # Final record
    # -------------------------

    return {
        "doc_id": short_id(work.get("id")),

        "pub_year": work.get("publication_year"),
        "pub_date": work.get("publication_date"),

        "title": work.get("title"),
        "abstract_text": work.get("abstract"),

        "source_tier": source_tier,

        "primary_topic": work.get("primary_topic"),
        "primary_topic_id": short_id(
            work.get("primary_topic_id")
        ),

        "domain_id": short_id(
            work.get("domain_id")
        ),

        "field_id": field_id,

        "authors_ids": authors_ids,
        "affiliations": affiliations,

        "citations_at_cutoff": work.get(
            "cited_by_count", 0
        ),

        "referenced_works_ids": references,

        # В 1M dataset эти поля — JSON строки
        "counts_by_year": json.dumps(
            work.get("counts_by_year", []),
            ensure_ascii=False
        ),

        "authorships_json": json.dumps(
            authorships,
            ensure_ascii=False
        ),

        "topics_json": json.dumps(
            work.get("topics_raw", []),
            ensure_ascii=False
        ),

        "source_id": work.get("source_id"),

        "doi": work.get("doi"),

        "fetched_at": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),

        # Для on-demand выборки вместо
        # исторического stratum сохраняем запрос
        "stratum": query,

        "is_engineering": field_id == "22",

        "commercial_maturity_index":
            commercial_maturity_index,

        "affiliation_coverage":
            affiliation_coverage,

        "has_ref_data": int(bool(references)),

        # Эти значения относятся к historical
        # labeling и здесь пока неизвестны
        "target_emergence": None,
        "target_status": "inference",
        "target_population": None,
        "split": "inference",

        "new_authors_baseline": None,
        "new_authors_future": None,
        "future_growth_ratio": None,
        "new_authors_growth_ratio": None,
    }


def save_parquet(
    works,
    query,
    output_path="data/openalex_corpus.parquet"
):
    records = [
        make_parquet_record(work, query)
        for work in works
    ]

    table = pa.Table.from_pylist(
        records,
        schema=SCHEMA
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pq.write_table(
        table,
        output_path,
        compression="zstd"
    )

    print("\nPARQUET:")
    print("Saved:", output_path)
    print("Works:", len(records))

    return table


def run_parser(user_query):
    # 1. Query normalization

    query = normalize_query(user_query)

    print("OpenAlex query:", query)

    # 2. Seed Search

    seed_works = search_works(
        query,
        limit=1000
    )

    print(
        "Seed works:",
        len(seed_works)
    )

    # 3. Topic Discovery

    topics = discover_topics(
        seed_works
    )

    print(
        "Unique topics:",
        len(topics)
    )

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

    print(
        "Selected topics:",
        len(selected_topics)
    )

    print("\nSELECTED TOPICS:\n")

    for topic in selected_topics:
        print(
            f'{topic["similarity"]:.3f} | '
            f'{topic["frequency"]:3} | '
            f'{topic["coverage"]:.1%} | '
            f'{topic["name"]}'
        )

    # 6. Main Collection

    main_works, yearly_stats = (
        collect_main_corpus(
            selected_topics,
            query
        )
    )

    print("\nMAIN COLLECTION:")
    print(
        "Unique works:",
        len(main_works)
    )

    # 7. Normalization

    normalized_works = normalize_works(
        main_works
    )

    print("\nNORMALIZATION:")
    print(
        "Normalized works:",
        len(normalized_works)
    )

    # 8. Relevance filtering

    scored_works = score_relevance(
        query,
        normalized_works
    )

    clean_works = filter_relevant(
        scored_works,
        remove_ratio=0.15
    )

    print("\nRELEVANCE FILTERING:")

    print(
        "Before:",
        len(scored_works)
    )

    print(
        "After:",
        len(clean_works)
    )

    print(
        "Removed:",
        len(scored_works)
        - len(clean_works)
    )

    if clean_works:
        print(
            "Lowest kept relevance:",
            f'{clean_works[-1]["relevance"]:.3f}'
        )

    # 9. Yearly statistics

    print("\nYEARLY STATS:")

    for topic_name, years in yearly_stats.items():

        print(f"\n{topic_name}")

        for year, count in years.items():

            print(
                f"  {year}: {count}"
            )

    # 10. JSON output

    save_results(
        raw_works=main_works,
        clean_works=clean_works,
        yearly_stats=yearly_stats
    )

    # 11. Parquet

    save_parquet(
        clean_works,
        query,
        "data/openalex_corpus.parquet"
    )

    return clean_works, yearly_stats


if __name__ == "__main__":

    user_query = input(
        "Введите направление: "
    )

    clean_works, yearly_stats = (
        run_parser(user_query)
    )