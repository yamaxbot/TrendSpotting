
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
    "via", "vs", "with", "challenges", "opportunities",
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


def _chemistry(title):
    match = re.search(r"\b(sodium|lithium|potassium|zinc|magnesium|calcium)\b", title or "", re.I)
    return match.group(1).lower() if match else None


def _scope_to_chemistry(query, chemistry):
    if chemistry and not re.search(rf"\b{re.escape(chemistry)}\b", query, re.I):
        return f"{query} AND {chemistry}"
    return query


def thematic_query(title, abstract):
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

    chemistry = _chemistry(title)
    if not candidates:
        if not chemistry:
            raise ValueError("The article has no title phrase supported by its abstract")
        abstract_words = set(WORDS.findall(abstract or ""))
        for first, second in zip(title_words, title_words[1:]):
            if (
                first.lower() not in STOPWORDS
                and second.lower() not in STOPWORDS
                and first[0].upper() + second[0].upper() in abstract_words
            ):
                return _scope_to_chemistry(f'"{first} {second}"', chemistry)
        supported = [
            word for word in title_words
            if word.lower() not in STOPWORDS
            and word.lower() != chemistry
            and abstract_counts[_stem(word)]
        ]
        return _scope_to_chemistry(
            max(supported, key=_weight) if supported else f'"{chemistry} ion"',
            chemistry,
        )
    _, index, first, second = max(candidates)
    phrase = f'"{first} {second}"'

    for word in title_words[:index]:
        if (
            word.lower() not in {"ai", "ml"}
            and len(word) <= 5
            and sum(char.isupper() for char in word) >= 2
            and abstract_counts[_stem(word)]
        ):
            return _scope_to_chemistry(f"{phrase} AND {word}", chemistry)

    suffix = title[title_matches[index + 1].end():]
    application = re.search(
        r"\b(?:in|for|on|with|towards?)\s+([A-Za-z][A-Za-z0-9]*)\b",
        suffix,
        re.I,
    )
    if application:
        term = application.group(1)
        if term.lower() not in STOPWORDS and abstract_counts[_stem(term)]:
            return _scope_to_chemistry(f"{phrase} AND {term}", chemistry)

    if _weight(first) + _weight(second) < 1.5:
        raise ValueError("The title does not identify a specific search topic")
    return _scope_to_chemistry(phrase, chemistry)


def publication_counts(title, abstract, session=None):
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
