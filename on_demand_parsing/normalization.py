def restore_abstract(inverted_index):
    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(
        word for _, word in words
    )


def normalize_work(work):
    # -------------------------
    # Primary topic
    # -------------------------

    primary_topic = work.get("primary_topic") or {}

    domain = primary_topic.get("domain") or {}
    field = primary_topic.get("field") or {}

    # -------------------------
    # Topics
    # -------------------------

    topics_raw = work.get("topics") or []

    topics = [
        topic["display_name"]
        for topic in topics_raw
        if topic.get("display_name")
    ]

    # -------------------------
    # Keywords
    # -------------------------

    keywords_raw = work.get("keywords") or []

    keywords = [
        keyword["display_name"]
        for keyword in keywords_raw
        if keyword.get("display_name")
    ]

    # -------------------------
    # Authors
    # -------------------------

    authorships = work.get("authorships") or []

    authors = []
    authors_ids = []
    institutions = set()

    for authorship in authorships:

        author = authorship.get("author") or {}

        if author.get("display_name"):
            authors.append(
                author["display_name"]
            )

        if author.get("id"):
            authors_ids.append(
                author["id"]
            )

        for institution in authorship.get(
            "institutions", []
        ):
            name = institution.get(
                "display_name"
            )

            if name:
                institutions.add(name)

    # -------------------------
    # Source
    # -------------------------

    primary_location = (
        work.get("primary_location") or {}
    )

    source = (
        primary_location.get("source") or {}
    )

    # -------------------------
    # References
    # -------------------------

    referenced_works = (
        work.get("referenced_works") or []
    )

    # -------------------------
    # Result
    # -------------------------

    return {
        "id": work.get("id"),

        "doi": work.get("doi"),

        "title": work.get("title"),

        "abstract": restore_abstract(
            work.get("abstract_inverted_index")
        ),

        "publication_year":
            work.get("publication_year"),

        "publication_date":
            work.get("publication_date"),

        # Topic
        "primary_topic":
            primary_topic.get("display_name"),

        "primary_topic_id":
            primary_topic.get("id"),

        "domain_id":
            domain.get("id"),

        "field_id":
            field.get("id"),

        # Все topics
        "topics": topics,
        "topics_raw": topics_raw,

        # Keywords
        "keywords": keywords,

        # Authors
        "authors": authors,
        "authors_ids": authors_ids,
        "authorships": authorships,

        # Institutions
        "institutions": list(institutions),

        # Citations
        "cited_by_count":
            work.get("cited_by_count", 0),

        "counts_by_year":
            work.get("counts_by_year") or [],

        # References
        "referenced_works_ids":
            referenced_works,

        # Source
        "source":
            source.get("display_name"),

        "source_id":
            source.get("id"),

        "source_type":
            source.get("type"),

        # Work type
        "type":
            work.get("type"),

        "language":
            work.get("language"),
    }


def normalize_works(works):
    return [
        normalize_work(work)
        for work in works
    ]