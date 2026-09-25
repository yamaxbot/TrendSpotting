"""Query-specific article relevance check before final diversification."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from website.query_llm import _query_llm


GATE_VERSION = "intent-v1"
BATCH_SIZE = 20
MAX_CANDIDATES = 120
MAX_ABSTRACT_CHARACTERS = 900

RELEVANCE_PROMPT = """Decide which scientific works are directly relevant to the user's requested technology direction. Use the ORIGINAL user request to resolve the intended subject and direction of the relationship. For example, protecting AI systems is NOT the same as using AI for medical screening, fraud detection or other protection tasks. A keyword mention, broad OpenAlex topic, or generic application of AI is insufficient: the work's main research question, method or result must concern the requested direction. Be conservative if the title and abstract do not establish this. Return only JSON {"relevant_ids":["id",...]}, with IDs from the supplied works; omit all irrelevant or unclear works. Do not add explanations."""


def _cache_key(query: str, row: dict) -> str:
    payload = [GATE_VERSION, query.casefold().strip(), row["id"], row["title"], row["abstract"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != GATE_VERSION:
        return {}
    decisions = data.get("decisions")
    if not isinstance(decisions, dict):
        return {}
    return {key: value for key, value in decisions.items() if isinstance(value, bool)}


def _save_cache(path: Path, decisions: dict[str, bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"version": GATE_VERSION, "decisions": decisions}),
        encoding="utf-8",
    )
    temporary.replace(path)


def _classify_batch(query: str, works: list[dict]) -> set[str]:
    content = json.dumps(
        {"request": query, "works": works}, ensure_ascii=False,
        separators=(",", ":"),
    )
    # This endpoint may spend part of the completion budget on reasoning;
    # small budgets can therefore return no JSON even for a short ID list.
    answer = _query_llm(RELEVANCE_PROMPT, content, max_tokens=1200, temperature=0.0)
    try:
        data = json.loads(answer)
        ids = data["relevant_ids"]
        available_ids = {work["id"] for work in works}
        if not isinstance(ids, list) or any(
            not isinstance(value, str) or value not in available_ids for value in ids
        ):
            raise ValueError("invalid relevance IDs")
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Некорректный ответ проверки релевантности") from exc
    return set(ids)


def select_relevant_diverse_top(
    ranked: pd.DataFrame,
    article_texts: pd.DataFrame,
    query: str,
    cache_path: Path,
    limit: int = 15,
) -> tuple[list, list[float]]:
    """Check ranked works in batches and refill until a diverse top is found.

    The relevance decision is deliberately separate from the emergence score.
    If no sufficiently relevant articles exist, return fewer than ``limit``.
    """
    from website.diversity import select_diverse_top

    if ranked.empty or limit <= 0:
        return [], []

    candidates = (
        ranked.sort_values("model_confidence", ascending=False, kind="stable")
        .head(MAX_CANDIDATES)
        .reset_index(names="_source_index")
    )
    metadata = article_texts[
        [column for column in ("doc_id", "title", "abstract_text") if column in article_texts]
    ].drop_duplicates("doc_id", keep="first")
    candidates = candidates[["_source_index", "doc_id", "model_confidence"]].merge(
        metadata, on="doc_id", how="left", sort=False,
    )
    decisions = _load_cache(cache_path)
    accepted_indices = []
    selected = ([], [])

    for offset in range(0, len(candidates), BATCH_SIZE):
        batch = candidates.iloc[offset:offset + BATCH_SIZE]
        works = []
        keys = {}
        for row in batch.to_dict("records"):
            title = row.get("title") if isinstance(row.get("title"), str) else ""
            abstract = row.get("abstract_text") if isinstance(row.get("abstract_text"), str) else ""
            work = {
                "id": str(row["doc_id"]),
                "title": title.strip(),
                "abstract": abstract.strip()[:MAX_ABSTRACT_CHARACTERS],
            }
            key = _cache_key(query, work)
            keys[row["_source_index"]] = (key, work["id"])
            if key not in decisions:
                works.append(work)
        if works:
            relevant_ids = _classify_batch(query, works)
            for work in works:
                decisions[_cache_key(query, work)] = work["id"] in relevant_ids
            _save_cache(cache_path, decisions)

        accepted_indices.extend(
            source_index for source_index, (key, _) in keys.items() if decisions[key]
        )
        if len(accepted_indices) >= limit:
            selected = select_diverse_top(
                ranked.loc[accepted_indices], article_texts, limit=limit,
            )
            if len(selected[0]) >= limit:
                break

    if accepted_indices and not selected[0]:
        selected = select_diverse_top(
            ranked.loc[accepted_indices], article_texts, limit=limit,
        )
    return selected
