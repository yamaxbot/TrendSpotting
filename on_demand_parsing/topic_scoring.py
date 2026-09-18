from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def score_topics(query, topics):
    names = list(topics.keys())

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