from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def score_relevance(query, works):
    if not works:
        return []

    texts = []

    for work in works:
        title = work.get("title") or ""
        abstract = work.get("abstract") or ""

        text = f"{title}. {abstract}".strip()
        texts.append(text)

    # Embedding запроса
    query_embedding = model.encode(
        query
    )

    # Embeddings статей
    work_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    # Сходство запрос ↔ статья
    similarities = cos_sim(
        query_embedding,
        work_embeddings
    )[0]

    result = []

    for work, similarity in zip(
        works,
        similarities
    ):
        work["relevance"] = similarity.item()
        result.append(work)

    # От наиболее релевантных
    # к наименее релевантным
    return sorted(
        result,
        key=lambda x: x["relevance"],
        reverse=True
    )


def filter_relevant(
    works,
    remove_ratio=0.15
):
    if not works:
        return []

    keep_count = int(
        len(works) * (1 - remove_ratio)
    )

    return works[:keep_count]