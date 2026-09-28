"""Query-specific article relevance check before final diversification."""

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAIError

from vsellm_chat import CHAT_TIMEOUT_SECONDS
from website.query_llm import _query_llm


GATE_VERSION = "qwen37-intent-v1"
BATCH_SIZE = 20
MAX_CANDIDATES = 240
MAX_ABSTRACT_CHARACTERS = 900
MAX_RELEVANCE_ATTEMPTS = 3
logger = logging.getLogger(__name__)
ION_DIRECTION = re.compile(
    r"\b(sodium|lithium|potassium|zinc|magnesium|calcium|alumin(?:um|ium))"
    r"[\s\-‐‑–]*ion\b", re.I,
)
ION_ALIASES = {
    "sodium": re.compile(r"(?<!\w)(?:Na[-‐‑– ]?ion|Na\+|SIBs?)(?!\w)"),
    "lithium": re.compile(r"(?<!\w)(?:Li[-‐‑– ]?ion|Li\+|LIBs?)(?!\w)"),
    "potassium": re.compile(r"(?<!\w)(?:K[-‐‑– ]?ion|KIBs?)(?!\w)"),
}
QUANTUM_REQUEST = re.compile(r"quantum|квант", re.I)
SENSING_REQUEST = re.compile(r"sens(?:or|ing)|metrolog|датчик|сенсор|измерен", re.I)
QUANTUM_SENSING_METHOD = re.compile(
    r"\b(?:quantum|qubit|entangl\w*|squeez\w*|rydberg|"
    r"spin[ -]?defect\w*|spin resonance|"
    r"(?:nitrogen|boron)[ -]?vacanc\w*|"
    r"(?:colou?r|NV)[ -]?cent(?:er|re)\w*|"
    r"atomic (?:sensor\w*|magnetomet\w*|clock\w*)|"
    r"atom interferomet\w*|SERF)\b",
    re.I,
)

RELEVANCE_PROMPT = """Decide which scientific works are directly relevant to the user's requested technology direction. Use the ORIGINAL user request to resolve the intended subject and direction of the relationship. For example, protecting AI systems is NOT using AI for medical screening. For quantum-sensor requests, reject conventional optical or thermal sensors without a quantum sensing mechanism. A keyword mention, broad OpenAlex topic, or generic application is insufficient: the main research question, method or result must concern the requested direction. Be conservative if the title and abstract do not establish this. Return only JSON {"relevant_ids":["id",...]}, with IDs from the supplied works; omit all irrelevant or unclear works. Do not add explanations."""


def _matches_ion_direction(work: dict, normalized_query: str | None) -> bool:
    """Require a named battery chemistry to be central, not a passing mention."""
    match = ION_DIRECTION.search(normalized_query or "")
    if not match:
        return True
    metal = match.group(1)
    if metal.lower() == "aluminium":
        metal = "aluminum"
    term = re.compile(rf"\b{re.escape(metal)}\b", re.I)
    title = work["title"]
    abstract = work["abstract"]
    alias = ION_ALIASES.get(metal.lower())
    title_hits = len(term.findall(title)) + (len(alias.findall(title)) if alias else 0)
    abstract_hits = len(term.findall(abstract)) + (len(alias.findall(abstract)) if alias else 0)
    return title_hits > 0 or abstract_hits >= 2


def _matches_quantum_sensing(work: dict, query: str, normalized_query: str | None) -> bool:
    request = f"{query} {normalized_query or ''}"
    if not (QUANTUM_REQUEST.search(request) and SENSING_REQUEST.search(request)):
        return True
    return bool(QUANTUM_SENSING_METHOD.search(f"{work['title']} {work['abstract']}"))


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
    deadline = time.monotonic() + CHAT_TIMEOUT_SECONDS
    for attempt in range(1, MAX_RELEVANCE_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            answer = _query_llm(
                RELEVANCE_PROMPT, content, max_tokens=1200,
                temperature=0.0, timeout=remaining,
            )
            response_content = answer.strip()
            if response_content.startswith("```"):
                response_content = re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", response_content, flags=re.I,
                )
            data = json.loads(response_content)
            ids = data["relevant_ids"]
            available_ids = {work["id"] for work in works}
            if not isinstance(ids, list) or any(
                not isinstance(value, str) or value not in available_ids for value in ids
            ):
                raise ValueError("invalid relevance IDs")
            return set(ids)
        except (RuntimeError, ValueError, TypeError, KeyError, OpenAIError) as exc:
            logger.warning(
                "Relevance check attempt %s/%s failed: %s",
                attempt, MAX_RELEVANCE_ATTEMPTS, type(exc).__name__,
            )
    raise RuntimeError("Некорректный ответ проверки релевантности")


def select_relevant_diverse_top(
    ranked: pd.DataFrame,
    article_texts: pd.DataFrame,
    query: str,
    cache_path: Path,
    limit: int = 15,
    budget=None,
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
        [column for column in ("doc_id", "title", "abstract_text", "stratum") if column in article_texts]
    ].drop_duplicates("doc_id", keep="first")
    candidates = candidates[["_source_index", "doc_id", "model_confidence"]].merge(
        metadata, on="doc_id", how="left", sort=False,
    )
    decisions = _load_cache(cache_path)
    accepted_indices = []
    selected = ([], [])

    def prepare_batch(offset):
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
            if not _matches_ion_direction(work, row.get("stratum")):
                continue
            if not _matches_quantum_sensing(work, query, row.get("stratum")):
                continue
            key = _cache_key(query, work)
            keys[row["_source_index"]] = (key, work["id"])
            if key not in decisions:
                works.append(work)
        return keys, works

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="trend-relevance")
    try:
        keys, works = prepare_batch(0)
        future = executor.submit(_classify_batch, query, works) if works else None
        for offset in range(0, len(candidates), BATCH_SIZE):
            if budget and budget.expired() and offset:
                budget.stop()
                break
            next_offset = offset + BATCH_SIZE
            next_batch = None
            if next_offset < len(candidates) and not (budget and budget.expired()):
                next_keys, next_works = prepare_batch(next_offset)
                next_future = (
                    executor.submit(_classify_batch, query, next_works)
                    if offset > 0 and next_works else None
                )
                next_batch = (next_keys, next_works, next_future)

            if future is not None:
                relevant_ids = future.result()
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
            if budget and budget.expired():
                budget.stop()
                break
            if next_batch is not None:
                keys, works, future = next_batch
                if future is None and works:
                    future = executor.submit(_classify_batch, query, works)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if accepted_indices and not selected[0]:
        selected = select_diverse_top(
            ranked.loc[accepted_indices], article_texts, limit=limit,
        )
    return selected
