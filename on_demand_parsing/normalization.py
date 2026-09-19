def restore_abstract(inverted_index):
    if not inverted_index:
        return None

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append(
                (position, word)
            )

    words.sort()

    return " ".join(
        word for _, word in words
    )


def normalize_work(work):
    # Topics
    topics = [
        topic["display_name"]
        for topic in work.get("topics", [])
        if topic.get("display_name")
    ]

    # Keywords
    keywords = [
        keyword["display_name"]
        for keyword in work.get("keywords", [])
        if keyword.get("display_name")
    ]

    # Authors
    authors = []

    # Institutions
    institutions = set()

    for authorship in work.get("authorships", []):
        author = authorship.get("author") or {}

        if author.get("display_name"):
            authors.append(
                author["display_name"]
            )

        for institution in authorship.get(
            "institutions",
            []
        ):
            name = institution.get(
                "display_name"
            )

            if name:
                institutions.add(name)

    # Source
    primary_location = (
        work.get("primary_location") or {}
    )

    source = (
        primary_location.get("source") or {}
    )

    return {
        "id": work.get("id"),

        "doi": work.get("doi"),

        "title": work.get("title"),

        "abstract": restore_abstract(
            work.get(
                "abstract_inverted_index"
            )
        ),

        "publication_year": work.get(
            "publication_year"
        ),

        "publication_date": work.get(
            "publication_date"
        ),

        "topics": topics,

        "keywords": keywords,

        "authors": authors,

        "institutions": list(
            institutions
        ),

        "cited_by_count": work.get(
            "cited_by_count",
            0
        ),

        "source": source.get(
            "display_name"
        ),

        "language": work.get(
            "language"
        ),

        "type": work.get(
            "type"
        )
    }


def normalize_works(works):
    return [
        normalize_work(work)
        for work in works
    ]