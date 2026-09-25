from pathlib import Path
import logging
from urllib.parse import urlsplit

import requests
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from . import charts, evidence, jobs, organizations
from website.query_llm import ARTICLE_ANALYSIS_VERSION, TITLE_FORMAT_VERSION, clean_weak_signal

logger = logging.getLogger(__name__)


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
        "model_threshold",
        "model_version",
        "ranking_version",
        "diversity_max_similarity",
        "shap_values",
        "llm_title",
        "llm_title_version",
        "llm_description",
        "llm_weak_signal",
        "llm_analysis_version",
        "llm_problem",
        "llm_advantage",
        "llm_case_result",
    }
    try:
        from website.main import MODEL_VERSION, RANKING_VERSION

        table = pq.read_table(dataset_path)
        corpus = pq.read_table(corpus_path, columns=["doc_id", "pub_year", "title"])
        if not required_columns.issubset(table.column_names):
            return False
        title_versions = table["llm_title_version"].to_pylist()
        model_versions = table["model_version"].to_pylist()
        analysis_versions = table["llm_analysis_version"].to_pylist()
        from website.query_llm import is_generic_weak_signal, is_valid_title, is_valid_weak_signal
        titles_by_id = dict(zip(corpus["doc_id"].to_pylist(), corpus["title"].to_pylist()))
        titles_ready = all(
            is_valid_title(title, titles_by_id.get(doc_id))
            for doc_id, title in zip(table["doc_id"].to_pylist(), table["llm_title"].to_pylist())
        )
        signals_ready = all(
            is_valid_weak_signal(value) and not is_generic_weak_signal(value)
            for value in table["llm_weak_signal"].to_pylist()
        )
        return (
            table.num_rows <= 15
            and all(value == TITLE_FORMAT_VERSION for value in title_versions)
            and all(value == MODEL_VERSION for value in model_versions)
            and all(value == ARTICLE_ANALYSIS_VERSION for value in analysis_versions)
            and all(value == RANKING_VERSION for value in table["ranking_version"].to_pylist())
            and titles_ready
            and signals_ready
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
            "primary_topic_id",
            "doi",
            "article_url",
            "source_id",
            "pub_year",
            "authorships_json",
        )
        if column in corpus.columns
    ]
    corpus = corpus[corpus_columns].drop_duplicates("doc_id", keep="first")
    rows = predictions.merge(corpus, on="doc_id", how="left")

    if generate_llm_texts:
        from website.query_llm import (
            analyze_abstract,
            fallback_weak_signal,
            is_generic_weak_signal,
            is_valid_description,
            is_valid_title,
            is_valid_weak_signal,
            translate_article_title,
        )

        for column in (
            "llm_title", "llm_title_version", "llm_description", "llm_weak_signal",
            "llm_analysis_version", "llm_problem", "llm_advantage", "llm_case_result",
        ):
            if column not in predictions.columns:
                predictions[column] = None

        handled_errors = (
            KeyError, IndexError, TypeError, RuntimeError, ValueError,
            AttributeError, requests.RequestException,
        )

        def checkpoint():
            temporary_path = dataset_path.with_name(f".{dataset_path.name}.tmp")
            predictions.to_parquet(temporary_path, index=False)
            temporary_path.replace(dataset_path)

        for row_index, row in rows.iterrows():
            original_title = row.get("title")
            abstract_text = row.get("abstract_text")
            title = row.get("llm_title")
            description = row.get("llm_description")
            signal = row.get("llm_weak_signal")
            if not is_valid_title(title, original_title):
                try:
                    title = translate_article_title(original_title, abstract_text)
                except handled_errors:
                    title = original_title if isinstance(original_title, str) else "None"

            needs_analysis = (
                not is_valid_description(description, abstract_text)
                or not is_valid_weak_signal(signal)
                or is_generic_weak_signal(signal)
                or any(
                    not isinstance(row.get(field), str)
                    for field in ("llm_problem", "llm_advantage", "llm_case_result")
                )
            )
            if needs_analysis:
                try:
                    analysis = analyze_abstract(abstract_text, row.get("shap_values"))
                except handled_errors:
                    analysis = dict.fromkeys(
                        ("summary", "problem", "advantage", "case_result", "weak_signal"),
                        "None",
                    )
                description = analysis["summary"]
                signal = analysis.get("weak_signal")
            else:
                analysis = {
                    "problem": row.get("llm_problem"),
                    "advantage": row.get("llm_advantage"),
                    "case_result": row.get("llm_case_result"),
                }

            if not is_valid_weak_signal(signal):
                signal = fallback_weak_signal(
                    row.get("shap_values"), summary=description, title=title,
                )
            signal = clean_weak_signal(signal)

            rows.at[row_index, "llm_title"] = title
            rows.at[row_index, "llm_description"] = description
            rows.at[row_index, "llm_weak_signal"] = signal
            prediction_mask = predictions["doc_id"] == row["doc_id"]
            for field, value in (
                ("llm_problem", analysis["problem"]),
                ("llm_advantage", analysis["advantage"]),
                ("llm_case_result", analysis["case_result"]),
            ):
                rows.at[row_index, field] = value
                predictions.loc[prediction_mask, field] = value
            rows.at[row_index, "llm_analysis_version"] = ARTICLE_ANALYSIS_VERSION
            predictions.loc[prediction_mask, "llm_title"] = title
            predictions.loc[prediction_mask, "llm_title_version"] = TITLE_FORMAT_VERSION
            predictions.loc[prediction_mask, "llm_description"] = description
            predictions.loc[prediction_mask, "llm_weak_signal"] = signal
            predictions.loc[prediction_mask, "llm_analysis_version"] = ARTICLE_ANALYSIS_VERSION
            checkpoint()

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
                "original_title": original_title,
                "abstract_text": row.get("abstract_text"),
                "primary_topic_id": row.get("primary_topic_id"),
                "name": title,
                "area": display_value(row.get("primary_topic")),
                "description": display_value(row.get("llm_description")),
                "problem": display_value(row.get("llm_problem")),
                "advantage": display_value(row.get("llm_advantage")),
                "companies": organizations.company_names(row.get("authorships_json")),
                "case_result": display_value(row.get("llm_case_result")),
                "case_url": article_url,
                "confidence": confidence,
                "publication_year": publication_year,
                "target_emergence": display_value(row.get("target_emergence")),
                "direction": "None",
                "signal": display_value(clean_weak_signal(row.get("llm_weak_signal"))),
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
def how_it_works(request):
    return render(request, "trends/how_it_works.html")


@require_GET
def about_project(request):
    return render(request, "trends/about.html")


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

    anchor = {
        "doc_id": trend["doc_id"],
        "title": trend["original_title"],
        "abstract_text": trend["abstract_text"],
        "pub_year": str(trend["publication_year"]),
        "primary_topic_id": trend["primary_topic_id"],
    }
    related = evidence.load_cached(query, anchor)
    related_state = "complete" if related else evidence.start(query, anchor)
    timeline = (
        charts.publication_timeline(related["years"])
        if related and related.get("years") else None
    )
    if related:
        if related.get("signal_text"):
            trend["signal"] = related["signal_text"]
        for item in related["sources"]:
            meta = f"{item['date']} · {item['source_type']}"
            if item.get("venue"):
                meta += f" · {item['venue']}"
            trend["sources"].append((item["title"], meta, _safe_url(item["url"])))
    return render(request, "trends/detail.html", {
        "trend": trend,
        "query": query,
        "related": related,
        "timeline": timeline,
        "related_state": related_state,
    })


@require_GET
def evidence_status(request, slug):
    query = request.GET.get("q", "").strip()
    if not query or not _valid_query(query):
        return JsonResponse({"status": "failed"}, status=400)
    try:
        from website.main import get_built_dataset_path, get_parsed_corpus_path
        trends = _load_trends(
            get_built_dataset_path(query), get_parsed_corpus_path(query)
        )
    except (OSError, ValueError, KeyError):
        raise Http404("Результаты запроса не найдены")
    trend = next((item for item in trends if item["slug"] == slug), None)
    if trend is None:
        raise Http404("Направление не найдено")
    anchor = {
        "doc_id": trend["doc_id"],
        "title": trend["original_title"],
        "primary_topic_id": trend["primary_topic_id"],
    }
    response = JsonResponse({"status": evidence.status(query, anchor)})
    response["Cache-Control"] = "no-store"
    return response


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
