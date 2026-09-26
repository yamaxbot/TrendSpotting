"""Identify synthesis papers that should not anchor a technology trend."""

import re


REVIEW_TITLE = re.compile(
    r"\b(?:review|survey|overview|meta[- ]analysis|bibliometric analysis|"
    r"systematic mapping|mapping study)\b|\b(?:обзор|метаанализ|"
    r"библиометрическ\w* анализ)\b",
    re.IGNORECASE,
)
REVIEW_ABSTRACT = re.compile(
    r"\b(?:this (?:paper|article|study) reviews\b|"
    r"here,?\s+we (?:critically )?review\b)",
    re.IGNORECASE,
)


def is_review_article(title, work_type=None, abstract=None):
    """Exclude works that explicitly identify themselves as reviews."""
    if isinstance(work_type, str) and work_type.lower() == "review":
        return True
    return (
        (isinstance(title, str) and bool(REVIEW_TITLE.search(title)))
        or (isinstance(abstract, str) and bool(REVIEW_ABSTRACT.search(abstract)))
    )
