from sentence_transformers.util import cos_sim
from on_demand_parsing.embedding_model import get_model


def score_relevance(query, works, budget=None):
    if not works:
        return []

    texts = []

    for work in works:
        title = work.get("title") or ""
        abstract = work.get("abstract") or ""

        text = f"{title}. {abstract}".strip()
        texts.append(text)

    model = get_model()
    query_embedding = model.encode(
        query
    )

    chunk_size = 256 if budget else len(works)
    result = []
    for offset in range(0, len(works), chunk_size):
        if budget and budget.expired():
            budget.stop()
        work_embeddings = model.encode(
            texts[offset:offset + chunk_size], batch_size=32,
            show_progress_bar=budget is None,
        )
        similarities = cos_sim(query_embedding, work_embeddings)[0]
        for work, similarity in zip(works[offset:offset + chunk_size], similarities):
            work["relevance"] = similarity.item()
            result.append(work)
        if budget and budget.expired():
            budget.stop()


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

    keep_count = max(1, int(
        len(works) * (1 - remove_ratio)
    ))

    return works[:keep_count]
