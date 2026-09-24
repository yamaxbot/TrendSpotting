from pathlib import Path
import logging
from urllib.parse import urlsplit

import requests
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from . import jobs

logger = logging.getLogger(__name__)
TITLE_FORMAT_VERSION = "exact-v1"


def _valid_query(query):
    return len(query) <= 200 and not any(ord(char) < 32 for char in query)


def _safe_url(value):
    try:
        parts = urlsplit(str(value))
        return str(value) if parts.scheme in {"https", "http"} and parts.netloc else None
    except ValueError:
        return None


def _dataset_is_ready(dataset_path: Path, corpus_path: Path) -> bool:
    if not dataset_path.is_file() or not corpus_path.is_file():
        return False

    import pyarrow.parquet as pq

    required_columns = {
        "doc_id",
        "model_confidence",
        "diversity_max_similarity",
        "shap_values",
        "llm_title",
        "llm_title_version",
        "llm_description",
        "llm_weak_signal",
    }
    try:
        table = pq.read_table(dataset_path)
        corpus = pq.read_table(corpus_path, columns=["doc_id", "pub_year"])
        if not required_columns.issubset(table.column_names):
            return False
        title_versions = table["llm_title_version"].to_pylist()
        return (
            table.num_rows <= 15
            and all(value == TITLE_FORMAT_VERSION for value in title_versions)
            and dataset_path.stat().st_mtime_ns >= corpus_path.stat().st_mtime_ns
            and set(table["doc_id"].to_pylist()).issubset(set(corpus["doc_id"].to_pylist()))
        )
    except (OSError, ValueError):
        return False


def _load_trends(
    dataset_path: Path,
    corpus_path: Path,
    generate_llm_texts: bool = False,
) -> list[dict]:
    """Объединить предсказания модели с исходными данными публикаций."""
    import pandas as pd

    predictions = pd.read_parquet(dataset_path)
    corpus = pd.read_parquet(corpus_path)
    corpus_columns = [
        column
        for column in (
            "doc_id",
            "title",
            "abstract_text",
            "primary_topic",
            "doi",
            "article_url",
            "source_id",
            "pub_year",
        )
        if column in corpus.columns
    ]
    corpus = corpus[corpus_columns].drop_duplicates("doc_id", keep="first")
    rows = predictions.merge(corpus, on="doc_id", how="left")

    if generate_llm_texts:
        try:
            from query_llm import (
                explain_weak_signal,
                summarize_abstract,
                translate_article_title,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "query_llm":
                raise
            from website.query_llm import (
                explain_weak_signal,
                summarize_abstract,
                translate_article_title,
            )

        translated_titles = []
        descriptions = []
        signal_explanations = []
        for row_index, abstract_text in enumerate(
            rows.get(
                "abstract_text",
                pd.Series([None] * len(rows)),
            )
        ):
            original_title = rows.iloc[row_index].get("title")
            try:
                translated_titles.append(
                    translate_article_title(original_title, abstract_text)
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                RuntimeError, ValueError, AttributeError,
                requests.RequestException,
            ):
                translated_titles.append("None")

            if not isinstance(abstract_text, str) or not abstract_text.strip():
                descriptions.append("None")
                signal_explanations.append("None")
                continue

            try:
                descriptions.append(summarize_abstract(abstract_text))
            except (
                KeyError,
                IndexError,
                TypeError,
                RuntimeError, ValueError, AttributeError,
                requests.RequestException,
            ):
                descriptions.append("None")

            try:
                signal_explanations.append(
                    explain_weak_signal(
                        abstract_text,
                        rows.iloc[row_index].get("shap_values"),
                    )
                )
            except (
                KeyError,
                IndexError,
                TypeError,
                RuntimeError, ValueError, AttributeError,
                requests.RequestException,
            ):
                signal_explanations.append("None")

        rows["llm_title"] = translated_titles
        rows["llm_description"] = descriptions
        rows["llm_weak_signal"] = signal_explanations

        titles_by_id = dict(zip(rows["doc_id"], translated_titles))
        descriptions_by_id = dict(zip(rows["doc_id"], descriptions))
        signals_by_id = dict(zip(rows["doc_id"], signal_explanations))
        predictions["llm_title"] = predictions["doc_id"].map(titles_by_id)
        predictions["llm_title_version"] = TITLE_FORMAT_VERSION
        predictions["llm_description"] = predictions["doc_id"].map(
            descriptions_by_id
        )
        predictions["llm_weak_signal"] = predictions["doc_id"].map(signals_by_id)

        temporary_path = dataset_path.with_name(f".{dataset_path.name}.tmp")
        predictions.to_parquet(temporary_path, index=False)
        temporary_path.replace(dataset_path)

    def display_value(value):
        if value is None or pd.isna(value) or value == "":
            return "None"
        return value

    trends = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        doc_id = display_value(row.get("doc_id"))
        original_title = display_value(row.get("title"))
        translated_title = display_value(row.get("llm_title"))
        title = (
            translated_title
            if translated_title != "None"
            else original_title
        )
        doi = display_value(row.get("doi"))
        source_id = display_value(row.get("source_id"))
        publication_year = display_value(row.get("pub_year"))
        source_meta = doi if doi != "None" else source_id
        if publication_year != "None":
            source_meta = f"{source_meta}, {publication_year}"

        article_url = display_value(row.get("article_url"))
        if article_url == "None" and doi != "None":
            article_url = (
                doi
                if str(doi).startswith(("http://", "https://"))
                else f"https://doi.org/{doi}"
            )
        if article_url == "None" and doc_id != "None":
            article_url = f"https://openalex.org/{doc_id}"
        article_url = _safe_url(article_url) or _safe_url(doi) or f"https://openalex.org/{doc_id}"

        confidence = display_value(row.get("model_confidence"))
        if confidence != "None":
            confidence = round(float(confidence), 2)

        trends.append(
            {
                "rank": rank,
                "slug": f"result-{rank}",
                "doc_id": doc_id,
                "name": title,
                "area": display_value(row.get("primary_topic")),
                "description": display_value(row.get("llm_description")),
                "confidence": confidence,
                "target_emergence": display_value(row.get("target_emergence")),
                "direction": "None",
                "signal": display_value(row.get("llm_weak_signal")),
                "color": "blue",
                "years": "None",
                "period": "None",
                "predictors": ["None"],
                "sources": [(original_title, source_meta, article_url)],
            }
        )

    return trends


def _corpus_metrics(corpus_path: Path) -> tuple[int, str]:
    import pandas as pd

    corpus = pd.read_parquet(corpus_path, columns=["pub_year"])
    years = pd.to_numeric(corpus["pub_year"], errors="coerce").dropna()
    if years.empty:
        return len(corpus), "None"
    return len(corpus), f"{int(years.min())}—{int(years.max())}"


@require_GET
def dashboard(request):
    query = request.GET.get("q", "").strip()
    trends = []
    publication_count = 0
    observation_period = "None"
    error = None
    pending = False

    if not _valid_query(query):
        return render(request, "trends/dashboard.html", {
            "query": query[:200], "error": "Запрос должен содержать не более 200 символов без управляющих знаков.",
        }, status=400)

    if query:
        try:
            from website.main import (
                get_built_dataset_path,
                get_parsed_corpus_path,
            )

            corpus_path = get_parsed_corpus_path(query)
            dataset_path = get_built_dataset_path(query)
            if _dataset_is_ready(dataset_path, corpus_path):
                trends = _load_trends(dataset_path, corpus_path)
                publication_count, observation_period = _corpus_metrics(corpus_path)
            else:
                state = jobs.start(query)
                pending = state in {"running", "queued"}
                if not pending:
                    error = "Очередь поиска заполнена. Попробуйте немного позже."
        except Exception:
            logger.exception("Unable to display search results")
            error = "Не удалось загрузить результаты. Попробуйте повторить поиск."

    return render(
        request,
        "trends/dashboard.html",
        {
            "query": query,
            "trends": trends,
            "publication_count": publication_count,
            "observation_period": observation_period,
            "error": error,
            "pending": pending,
        },
    )


@require_GET
def trend_detail(request, slug):
    query = request.GET.get("q", "").strip()
    if not query or not _valid_query(query):
        raise Http404("Запрос не указан")

    try:
        from website.main import get_built_dataset_path, get_parsed_corpus_path

        dataset_path = get_built_dataset_path(query)
        trends = _load_trends(dataset_path, get_parsed_corpus_path(query))
    except (OSError, ValueError, KeyError):
        raise Http404("Результаты запроса не найдены")

    trend = next((item for item in trends if item["slug"] == slug), None)
    if trend is None:
        raise Http404("Направление не найдено")
    return render(request, "trends/detail.html", {"trend": trend, "query": query})


@require_GET
def search_status(request):
    query = request.GET.get("q", "").strip()
    if not query or not _valid_query(query):
        return JsonResponse({"status": "failed"}, status=400)
    state = jobs.status(query)
    if state not in {"running", "queued", "failed"}:
        from website.main import get_built_dataset_path, get_parsed_corpus_path
        state = "complete" if _dataset_is_ready(
            get_built_dataset_path(query), get_parsed_corpus_path(query)
        ) else "missing"
    response = JsonResponse({"status": state})
    response["Cache-Control"] = "no-store"
    return response
