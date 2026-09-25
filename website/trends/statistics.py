"""Broader OpenAlex publication counts, independent of strict source matching."""

import re
from collections import Counter

import requests

from on_demand_parsing.openalex_client import API_KEY, URL


YEARS = (2026, 2025, 2024, 2023)
WORDS = re.compile(r"[A-Za-z][A-Za-z0-9]*")
STOPWORDS = {
    "a", "about", "an", "and", "as", "based", "by", "conceptual", "enhancing",
    "for", "from", "future", "in", "of", "on", "or", "out", "research",
    "review", "study", "the", "through", "toward", "towards", "using",
    "via", "vs", "with", "challenges",
}
COMMON_TERMS = {
    "ai": 0.4, "agent": 0.5, "agents": 0.5, "artificial": 0.4,
    "attack": 0.4, "attacks": 0.4, "cancer": 0.5, "detected": 0.4,
    "detection": 0.4, "financial": 0.6, "intelligence": 0.4,
    "learning": 0.5, "machine": 0.5,
    "model": 0.5, "models": 0.5, "security": 0.6,
    "system": 0.5, "systems": 0.5,
}


def _stem(word):
    word = word.lower()
    if word.endswith("ing") and len(word) > 6:
        word = word[:-3]
        if len(word) > 2 and word[-1] == word[-2]:
            word = word[:-1]
    elif word.endswith("s") and len(word) > 4:
        word = word[:-1]
    return word


def _weight(word):
    return COMMON_TERMS.get(word.lower(), min(2.3, 1 + max(0, len(word) - 5) * 0.15))


def thematic_query(title, abstract):
    """Choose a repeated, specific title phrase; never trust the topic label alone."""
    title_matches = list(WORDS.finditer(title or ""))
    title_words = [match.group() for match in title_matches]
    abstract_counts = Counter(_stem(word) for word in WORDS.findall(abstract or ""))
    candidates = []
    for index, (first, second) in enumerate(zip(title_words, title_words[1:])):
        separator = title[title_matches[index].end():title_matches[index + 1].start()]
        if re.search(r"[.:;!?]", separator):
            continue
        if first.lower() in STOPWORDS or second.lower() in STOPWORDS:
            continue
        support = min(abstract_counts[_stem(first)], abstract_counts[_stem(second)])
        if not support:
            continue
        score = min(support, 5) * (_weight(first) + _weight(second))
        score += index / max(1, len(title_words))
        candidates.append((score, index, first, second))

    if not candidates:
        raise ValueError("The article has no title phrase supported by its abstract")
    _, index, first, second = max(candidates)
    phrase = f'"{first} {second}"'

    # Keep a short application qualifier when it precedes the selected phrase.
    for word in title_words[:index]:
        if (
            word.lower() not in {"ai", "ml"}
            and len(word) <= 5
            and sum(char.isupper() for char in word) >= 2
            and abstract_counts[_stem(word)]
        ):
            return f"{phrase} AND {word}"
    return phrase


def publication_counts(title, abstract, session=None):
    """Use one OpenAlex aggregation, not a capped sample of matching works."""
    query = thematic_query(title, abstract)
    session = session or requests.Session()
    params = {
        "filter": (
            f"publication_year:{min(YEARS)}-{max(YEARS)},"
            f"title_and_abstract.search:{query}"
        ),
        "group_by": "publication_year",
    }
    if API_KEY:
        params["api_key"] = API_KEY
    response = session.get(URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    groups = data.get("group_by")
    if not isinstance(groups, list):
        raise ValueError("OpenAlex did not return yearly groups")
    by_year = {int(group["key"]): int(group["count"]) for group in groups}
    years = [
        {"year": year, "count": max(0, by_year.get(year, 0)), "exhaustive": True}
        for year in YEARS
    ]
    return query, years
