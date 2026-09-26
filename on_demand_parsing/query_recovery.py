"""Recover a small seed set when exact OpenAlex search finds no works.

The recovered works still enter the existing topic-discovery and expansion
pipeline. Broader retrieval is never treated as proof of relevance by itself.
"""

import math
import re

from on_demand_parsing.normalization import restore_abstract


STOPWORDS = {
    "a", "an", "and", "as", "by", "for", "from", "in", "of", "on",
    "the", "to", "using", "with", "application", "applications",
    "approach", "approaches", "based", "technology", "technologies",
}
RECOVERY_LIMIT = 300
MIN_RECOVERED_SEEDS = 10
MAX_RECOVERED_SEEDS = 100


def _terms(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", text or "")
    normalized = []
    for word in words:
        word = word.lower()
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return list(dict.fromkeys(normalized))


def query_variants(query):
    """Drop trailing constraints gradually; leave the original query intact."""
    words = query.split()
    lengths = [len(words) - 1, math.ceil(len(words) * .75),
               math.ceil(len(words) * .5), 2, 1]
    return [
        " ".join(words[:length])
        for length in dict.fromkeys(lengths)
        if 0 < length < len(words)
    ]


def matches_original_query(work, query):
    query_terms = set(_terms(query))
    if not query_terms:
        return False
    abstract = restore_abstract(work.get("abstract_inverted_index")) or ""
    text_terms = set(_terms(f"{work.get('title') or ''} {abstract}"))
    required = max(1, math.ceil(.6 * len(query_terms)))
    return len(query_terms & text_terms) >= required


def recover_seed_works(query, years, search_fn, budget=None):
    """Return original-query-matching seeds for the normal topic expansion."""
    recovered = {}
    attempts = [(query, False)]
    for variant in query_variants(query):
        attempts.extend(((variant, True), (variant, False)))

    for variant, require_references in attempts:
        if budget and budget.expired():
            budget.stop()
            break
        works = search_fn(
            variant,
            limit=RECOVERY_LIMIT,
            years=years,
            require_references=require_references,
        )
        for work in works:
            identifier = work.get("id")
            if identifier and identifier not in recovered and matches_original_query(work, query):
                recovered[identifier] = work
                if len(recovered) >= MAX_RECOVERED_SEEDS:
                    return list(recovered.values())
        if len(recovered) >= MIN_RECOVERED_SEEDS:
            break
    return list(recovered.values())
