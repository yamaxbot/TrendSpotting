"""Semantic diversification for the final ranked list of publications."""

import numpy as np
import pandas as pd
from on_demand_parsing.embedding_model import MODEL_NAME as DEFAULT_MODEL_NAME
from on_demand_parsing.embedding_model import get_model as _get_model


DEFAULT_SIMILARITY_THRESHOLD = 0.78


def _article_text(row: pd.Series) -> str:
    abstract = row.get("abstract_text")
    if isinstance(abstract, str) and abstract.strip():
        return abstract.strip()

    title = row.get("title")
    return title.strip() if isinstance(title, str) else ""


def select_diverse_top(
    ranked: pd.DataFrame,
    article_texts: pd.DataFrame,
    limit: int = 15,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[list, list[float]]:
    """Select the highest-ranked articles subject to a semantic similarity cap.

    Candidates are considered in descending ``model_confidence`` order. An
    article is accepted only when its cosine similarity to every already
    selected article is at most ``similarity_threshold``. Consequently the
    function may return fewer than ``limit`` rows instead of filling the result
    with semantic duplicates.
    """
    if limit <= 0 or ranked.empty:
        return [], []
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")

    text_columns = [
        column
        for column in ("doc_id", "title", "abstract_text")
        if column in article_texts.columns
    ]
    if "doc_id" not in text_columns:
        raise ValueError("article_texts must contain doc_id")

    metadata = article_texts[text_columns].drop_duplicates("doc_id", keep="first")
    candidates = (
        ranked.sort_values("model_confidence", ascending=False, kind="stable")
        .reset_index(names="_source_index")
        .merge(metadata, on="doc_id", how="left")
    )
    candidates["_semantic_text"] = candidates.apply(_article_text, axis=1)

    # Articles entering this stage normally have an abstract. Keep the fallback
    # deterministic for incomplete records rather than failing the whole query.
    model = _get_model()
    selected_vectors: list[np.ndarray] = []
    selected_indices: list = []
    maximum_similarities: list[float] = []
    normalized_texts: set[str] = set()

    # Preserve rank order, but stop embedding once enough distinct articles
    # have been selected. A full corpus may contain thousands of candidates.
    chunk_size = 64
    for offset in range(0, len(candidates), chunk_size):
        chunk = candidates.iloc[offset:offset + chunk_size]
        embeddings = np.asarray(model.encode(
            chunk["_semantic_text"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunk):
            raise ValueError("embedding model returned an unexpected shape")

        for (_, row), vector in zip(chunk.iterrows(), embeddings):
            normalized_text = " ".join(row["_semantic_text"].lower().split())
            if normalized_text and normalized_text in normalized_texts:
                continue

            maximum_similarity = 0.0
            if selected_vectors:
                similarities = np.asarray(selected_vectors) @ vector
                maximum_similarity = float(np.max(similarities))
                if maximum_similarity > similarity_threshold:
                    continue

            selected_vectors.append(vector)
            selected_indices.append(row["_source_index"])
            maximum_similarities.append(round(maximum_similarity, 6))
            if normalized_text:
                normalized_texts.add(normalized_text)
            if len(selected_indices) == limit:
                return selected_indices, maximum_similarities

    return selected_indices, maximum_similarities
