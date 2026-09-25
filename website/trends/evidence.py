"""Find closely related OpenAlex works for a result article.

Counts describe the explicitly searched OpenAlex subset, not an estimate of
every publication about a technology in the world.
"""

import datetime as dt
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import numpy as np
import requests

from on_demand_parsing.normalization import restore_abstract
from on_demand_parsing.openalex_client import API_KEY, URL
from website.article_type import is_review_article
from website.diversity import _get_model
from website.query_llm import summarize_related_signal
from .statistics import YEARS, publication_counts


logger = logging.getLogger(__name__)
EVIDENCE_VERSION = "openalex-related-v5"
MAX_CANDIDATES_PER_YEAR = 1000
MAX_SOURCES = 5
MIN_COSINE_SIMILARITY = 0.72
CACHE_MAX_AGE = dt.timedelta(days=1)
GENERIC_TERMS = {
    "a", "an", "and", "application", "applications", "based", "comprehensive",
    "current", "development", "developments", "emerging", "emphasis", "evolution",
    "for", "from", "future", "in", "of", "on", "recent", "review", "smart",
    "study", "the", "towards", "trends", "using", "with", "advances",
    "advancements", "technology", "technologies", "overview", "new",
    "systematic", "analysis", "approach", "approaches", "opportunities",
    "challenges", "role", "does", "how", "what",
}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trend-evidence")
_lock = Lock()
_jobs = {}


def _terms(text):
    return [
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z0-9]+", text or "")
        if (len(word) > 2 or word.upper() in {"AI", "VR", "ML"})
        and word.lower() not in GENERIC_TERMS
    ]


def _search_phrase(title):
    terms = list(dict.fromkeys(_terms(title)))
    focus = [term for term in terms if term in _focus_terms(title)]
    if focus:
        context = [term for term in terms if term not in focus]
        selected = focus[:2] + context[:1]
    else:
        selected = terms[:3]
    return " ".join(selected)


def _focus_terms(title):
    """Keep a title's explicit application qualifier (for example, biomedical)."""
    match = re.search(r"\b(?:with emphasis on|for|towards?|in)\s+(.+)$", title or "", re.I)
    if not match:
        return set()
    return set(_terms(match.group(1)))


def _candidate_text(work):
    title = work.get("title") or ""
    abstract = restore_abstract(work.get("abstract_inverted_index")) or ""
    return f"{title}. {abstract}".strip(), title, abstract


def _candidate_url(work):
    doi = work.get("doi")
    if isinstance(doi, str) and doi.startswith(("https://", "http://")):
        return doi
    identifier = work.get("id") or ""
    return identifier if identifier.startswith("https://openalex.org/") else ""


def _fetch_year(phrase, topic_id, year, session):
    filters = [f"publication_year:{year}", "has_abstract:true"]
    if topic_id:
        filters.append(f"topics.id:{topic_id.rsplit('/', 1)[-1]}")

    works = []
    cursor = "*"
    total = None
    while len(works) < MAX_CANDIDATES_PER_YEAR:
        params = {
            "search": phrase,
            "filter": ",".join(filters),
            "per-page": min(100, MAX_CANDIDATES_PER_YEAR - len(works)),
            "cursor": cursor,
        }
        if API_KEY:
            params["api_key"] = API_KEY
        response = session.get(URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        page = data.get("results") or []
        total = int(data.get("meta", {}).get("count", 0))
        works.extend(page)
        cursor = data.get("meta", {}).get("next_cursor")
        if not page or not cursor:
            break
    return works, total or 0, len(works) >= (total or 0)


def build_evidence(anchor, session=None, encoder=None):
    """Count thematic works and select source links with the stricter matcher."""
    phrase = _search_phrase(anchor.get("title"))
    if not phrase:
        raise ValueError("The article title has no searchable terms")
    anchor_text = f"{anchor['title']}. {anchor.get('abstract_text') or ''}"
    anchor_terms = set(_terms(anchor.get("title")))
    focus_terms = _focus_terms(anchor.get("title"))
    required_overlap = min(3, len(anchor_terms))
    topic_id = anchor.get("primary_topic_id")
    model = encoder or _get_model()
    session = session or requests.Session()
    statistics_query = None
    thematic_years = None
    try:
        statistics_query, thematic_years = publication_counts(
            anchor.get("title"), anchor.get("abstract_text"), session=session,
        )
    except (requests.RequestException, ValueError, TypeError, KeyError):
        logger.exception("Unable to count thematic works for %s", anchor["doc_id"])
    anchor_vector = np.asarray(
        model.encode([anchor_text], convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )[0]

    counts = []
    selected_candidates = []
    for year in YEARS:
        works, search_count, exhaustive = _fetch_year(phrase, topic_id, year, session)
        candidate_rows = []
        for work in works:
            text, title, abstract = _candidate_text(work)
            if not text or not title:
                continue
            if not _candidate_url(work):
                continue
            overlap = len(anchor_terms.intersection(_terms(text)))
            if overlap < required_overlap:
                continue
            if focus_terms and not focus_terms.intersection(_terms(text)):
                continue
            candidate_rows.append((work, text, overlap, abstract))

        matches = []
        if candidate_rows:
            vectors = np.asarray(
                model.encode(
                    [row[1] for row in candidate_rows],
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            similarities = vectors @ anchor_vector
            for (work, _, overlap, abstract), similarity in zip(candidate_rows, similarities):
                if float(similarity) < MIN_COSINE_SIMILARITY:
                    continue
                matches.append({
                    "doc_id": str(work.get("id", "")).rsplit("/", 1)[-1],
                    "title": work["title"],
                    "url": _candidate_url(work),
                    "doi": work.get("doi"),
                    "year": year,
                    "date": work.get("publication_date") or str(year),
                    "source_type": work.get("type") or "work",
                    "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
                    "cited_by_count": int(work.get("cited_by_count") or 0),
                    "similarity": round(float(similarity), 3),
                    "term_overlap": overlap,
                    "abstract_text": abstract,
                })

        counts.append({
            "year": year,
            "count": len(matches),
            "search_count": search_count,
            "scanned": len(works),
            "exhaustive": exhaustive,
        })
        selected_candidates.extend(matches)

    selected_candidates.sort(
        key=lambda item: (item["similarity"], item["cited_by_count"]),
        reverse=True,
    )
    sources = []
    seen = set()
    for item in selected_candidates:
        identifier = item["doi"] or item["doc_id"]
        if identifier in seen or item["doc_id"] == anchor["doc_id"]:
            continue
        seen.add(identifier)
        sources.append(item)
        if len(sources) == MAX_SOURCES:
            break

    signal_text = None
    related_articles = [
        item for item in sources
        if item["abstract_text"].strip()
        and not is_review_article(item["title"], item["source_type"])
    ][:3]
    if len(related_articles) >= 2 and isinstance(anchor.get("abstract_text"), str):
        try:
            signal_text = summarize_related_signal(anchor, related_articles)
        except (requests.RequestException, RuntimeError, ValueError, TypeError):
            logger.exception("Unable to summarize related works for %s", anchor["doc_id"])

    public_sources = [
        {key: value for key, value in item.items() if key != "abstract_text"}
        for item in sources
    ]

    return {
        "version": EVIDENCE_VERSION,
        "anchor_id": anchor["doc_id"],
        "anchor_title": anchor["title"],
        "search_phrase": phrase,
        "statistics_query": statistics_query,
        "topic_id": topic_id,
        "years": thematic_years,
        "strict_years": counts,
        "sources": public_sources,
        "signal_text": signal_text,
        "complete": thematic_years is not None,
        "strict_complete": all(item["exhaustive"] for item in counts),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def cache_path(query, doc_id):
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)
    return Path(__file__).resolve().parent.parent / "data" / "evidence" / f"{digest}_{safe_id}.json"


def _read_cache(query, anchor):
    try:
        return json.loads(cache_path(query, anchor["doc_id"]).read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None


def _cache_matches(data, anchor, version):
    if not isinstance(data, dict):
        return False
    try:
        created = dt.datetime.fromisoformat(data["generated_at"])
        return (
            data.get("version") == version
            and data.get("anchor_title") == anchor["title"]
            and data.get("topic_id") == anchor.get("primary_topic_id")
            and dt.datetime.now(dt.timezone.utc) - created <= CACHE_MAX_AGE
        )
    except (ValueError, KeyError, TypeError):
        return False


def load_cached(query, anchor):
    data = _read_cache(query, anchor)
    return data if _cache_matches(data, anchor, EVIDENCE_VERSION) else None


def _upgrade_cached_statistics(query, anchor):
    """Reuse v4's strict links and signal instead of repeating their search."""
    old = _read_cache(query, anchor)
    if not _cache_matches(old, anchor, "openalex-related-v4"):
        return None
    try:
        statistics_query, thematic_years = publication_counts(
            anchor.get("title"), anchor.get("abstract_text"),
        )
    except (requests.RequestException, ValueError, TypeError, KeyError):
        logger.exception("Unable to count thematic works for %s", anchor["doc_id"])
        statistics_query, thematic_years = None, None
    return {
        **old,
        "version": EVIDENCE_VERSION,
        "statistics_query": statistics_query,
        "years": thematic_years,
        "strict_years": old.get("years"),
        "strict_complete": old.get("complete"),
        "complete": thematic_years is not None,
    }


def _run(query, anchor):
    key = (query, anchor["doc_id"])
    try:
        data = _upgrade_cached_statistics(query, anchor) or build_evidence(anchor)
        path = cache_path(query, anchor["doc_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        state = "complete"
    except Exception:
        logger.exception("Unable to build evidence for %s", anchor["doc_id"])
        state = "failed"
    with _lock:
        _jobs[key] = state


def start(query, anchor):
    key = (query, anchor["doc_id"])
    if load_cached(query, anchor):
        return "complete"
    with _lock:
        if _jobs.get(key) in {"queued", "running"}:
            return _jobs[key]
        _jobs[key] = "queued"
        _executor.submit(_run, query, anchor)
        return "queued"


def status(query, anchor):
    if load_cached(query, anchor):
        return "complete"
    with _lock:
        return _jobs.get((query, anchor["doc_id"]), "missing")
