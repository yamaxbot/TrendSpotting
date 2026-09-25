from sentence_transformers.util import cos_sim
from on_demand_parsing.embedding_model import get_model



def score_topics(query, topics):
    if not topics:
        return []
    names = list(topics.keys())

    model = get_model()
    query_embedding = model.encode(query)
    topic_embeddings = model.encode(names)

    similarities = cos_sim(
        query_embedding,
        topic_embeddings
    )[0]

    result = []

    for name, similarity in zip(names, similarities):
        result.append({
            "id": topics[name]["id"],
            "name": name,
            "frequency": topics[name]["frequency"],
            "similarity": similarity.item()
        })

    return sorted(
        result,
        key=lambda x: x["similarity"],
        reverse=True
    )


def select_topics(topics, seed_size):
    selected = []

    for topic in topics:
        similarity = topic["similarity"]
        frequency = topic["frequency"]
        coverage = frequency / seed_size

        core = similarity >= 0.50 and coverage >= 0.01
        niche = similarity >= 0.65 and frequency >= 2
        supported = similarity >= 0.30 and coverage >= 0.05

        if core or niche or supported:
            topic["coverage"] = coverage
            selected.append(topic)

    return selected
