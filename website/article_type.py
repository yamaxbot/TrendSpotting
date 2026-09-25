"""Identify synthesis papers that should not anchor a technology trend."""

import re


REVIEW_TITLE = re.compile(
    r"\b(?:review|survey|overview|meta[- ]analysis|bibliometric analysis|"
    r"systematic mapping|mapping study)\b|\b(?:обзор|метаанализ|"
    r"библиометрическ\w* анализ)\b",
    re.IGNORECASE,
)


def is_review_article(title, work_type=None):
    """Use OpenAlex's type first, with a title fallback for mislabeled reviews."""
    if isinstance(work_type, str) and work_type.lower() == "review":
        return True
    return isinstance(title, str) and bool(REVIEW_TITLE.search(title))
