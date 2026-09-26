import re
import sys
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBSITE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "catboost_splits"
    / "random"
    / "catboost_model.cbm"
)
MODEL_METRICS_PATH = MODEL_PATH.with_name("metrics.json")
MODEL_VERSION = "catboost-random-split-2026-09-v1"
RANKING_VERSION = "intent-relevance-v1"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _make_safe_filename(value: str) -> str:
    """Преобразовать пользовательский запрос в безопасную часть имени файла."""
    filename = re.sub(r'[<>:"/\\|?*]+', "_", value.strip())
    filename = re.sub(r"\s+", "_", filename).strip("._")
    return filename[:100] or "query"


def get_parsed_corpus_path(query: str) -> Path:
    """Вернуть путь к корпусу, который создаёт on-demand парсер."""
    return WEBSITE_DIR / "data" / f"parse_corpus_{_make_safe_filename(query)}.parquet"


def get_built_dataset_path(query: str) -> Path:
    """Вернуть путь к итоговому датасету запроса."""
    return (
        WEBSITE_DIR
        / "data"
        / "build_dataset"
        / f"dataset_{_make_safe_filename(query)}.parquet"
    )


def model_dataset_is_ready(dataset_path: str | Path, corpus_path: str | Path) -> bool:
    """Return whether model results can be reused while LLM text is repaired."""
    dataset_path = Path(dataset_path)
    corpus_path = Path(corpus_path)
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
    }
    try:
        dataset = pq.read_table(dataset_path)
        corpus = pq.read_table(corpus_path, columns=["doc_id"])
        return (
            required_columns.issubset(dataset.column_names)
            and dataset.num_rows <= 15
            and all(
                value == MODEL_VERSION
                for value in dataset["model_version"].to_pylist()
            )
            and all(
                value == RANKING_VERSION
                for value in dataset["ranking_version"].to_pylist()
            )
            and dataset_path.stat().st_mtime_ns >= corpus_path.stat().st_mtime_ns
            and set(dataset["doc_id"].to_pylist()).issubset(
                set(corpus["doc_id"].to_pylist())
            )
        )
    except (OSError, ValueError):
        return False


def _load_model_threshold() -> float:
    if not MODEL_METRICS_PATH.is_file():
        raise FileNotFoundError(f"Метрики модели не найдены: {MODEL_METRICS_PATH}")
    try:
        threshold = float(
            json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))["threshold"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("В metrics.json отсутствует корректный threshold") from exc
    if not 0 < threshold < 1:
        raise ValueError("Threshold модели должен находиться между 0 и 1")
    return threshold


def _prepare_model_input(dataset: pd.DataFrame, model):
    feature_names = model.feature_names_
    dataset = dataset.copy()

    if "pub_year" in dataset.columns and "doc_id" in dataset.columns:
        year_totals = dataset.groupby("pub_year")["doc_id"].transform("count")
        derived_shares = {
            "topic_historical_share": "topic_historical_volume",
            "topic_local_share": "topic_local_volume",
            "historical_author_share": "historical_author_count",
        }
        for derived_feature, source_feature in derived_shares.items():
            if source_feature in dataset.columns:
                dataset[derived_feature] = (
                    dataset[source_feature] / (year_totals + 1e-5)
                )

    if "commercial_maturity_index" in dataset.columns:
        dataset["commercial_maturity_index"] = (
            dataset["commercial_maturity_index"].fillna(-1.0)
        )

    if "split" in feature_names and "split" not in dataset.columns:
        dataset["split"] = "inference"

    missing_features = [
        feature for feature in feature_names if feature not in dataset.columns
    ]
    if missing_features:
        raise ValueError(
            "В датасете отсутствуют признаки модели: "
            + ", ".join(missing_features)
        )

    model_input = dataset[feature_names].copy()
    categorical_features = [
        feature_names[index]
        for index in model.get_cat_feature_indices()
    ]
    for feature in categorical_features:
        model_input[feature] = model_input[feature].fillna("missing").astype(str)

    return model_input, categorical_features


def add_model_targets(
    dataset_path: str | Path,
    corpus_path: str | Path | None = None,
    query: str | None = None,
    budget=None,
) -> Path:
    """Добавить предсказания CatBoost и сохранить 15 лучших результатов."""
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    dataset = pd.read_parquet(dataset_path)
    dataset = dataset.drop(columns=["target_emergence"], errors="ignore")
    if budget and budget.expired():
        budget.stop()

    if dataset.empty:
        dataset["search_partial"] = bool(budget and budget.truncated)
        dataset["target_emergence"] = pd.Series(dtype="int8")
        dataset["model_confidence"] = pd.Series(dtype="float64")
        dataset["model_threshold"] = pd.Series(dtype="float64")
        dataset["model_version"] = pd.Series(dtype="str")
        dataset["ranking_version"] = pd.Series(dtype="str")
        dataset["diversity_max_similarity"] = pd.Series(dtype="float64")
        dataset["shap_values"] = pd.Series(dtype="str")
        dataset.to_parquet(dataset_path, index=False)
        return dataset_path

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    feature_names = model.feature_names_
    model_input, categorical_features = _prepare_model_input(dataset, model)
    probabilities = model.predict_proba(model_input)
    positive_class_index = list(model.classes_).index(1)
    positive_probabilities = probabilities[:, positive_class_index]
    model_threshold = _load_model_threshold()

    dataset["target_emergence"] = (
        positive_probabilities >= model_threshold
    ).astype("int8")
    dataset["model_confidence"] = positive_probabilities * 100
    dataset["model_threshold"] = model_threshold
    dataset["model_version"] = MODEL_VERSION
    dataset["ranking_version"] = RANKING_VERSION
    if corpus_path is None:
        text_columns = [
            column
            for column in ("doc_id", "title", "abstract_text", "work_type")
            if column in dataset.columns
        ]
        article_texts = dataset[text_columns].copy()
    else:
        corpus_path = Path(corpus_path).resolve()
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Корпус не найден: {corpus_path}")
        import pyarrow.parquet as pq
        corpus_columns = ["doc_id", "title", "abstract_text"]
        if "work_type" in pq.read_schema(corpus_path).names:
            corpus_columns.append("work_type")
        article_texts = pd.read_parquet(
            corpus_path,
            columns=corpus_columns,
        )

    from website.diversity import select_diverse_top
    from website.article_type import is_review_article

    primary_articles = article_texts.loc[
        ~article_texts.apply(
            lambda row: is_review_article(
                row.get("title"), row.get("work_type"), row.get("abstract_text"),
            ),
            axis=1,
        ),
        "doc_id",
    ]
    eligible_dataset = dataset[dataset["doc_id"].isin(primary_articles)]

    if query and query.strip():
        from website.relevance_gate import select_relevant_diverse_top

        top_indices, maximum_similarities = select_relevant_diverse_top(
            eligible_dataset,
            article_texts,
            query.strip(),
            dataset_path.with_name(f"{dataset_path.stem}.relevance.json"),
            limit=15,
            **({"budget": budget} if budget else {}),
        )
    else:
        top_indices, maximum_similarities = select_diverse_top(
            eligible_dataset,
            article_texts,
            limit=15,
        )
    dataset = dataset.loc[top_indices].copy()
    dataset["search_partial"] = bool(budget and budget.truncated)
    dataset["model_confidence"] = dataset["model_confidence"].round(2)
    dataset["diversity_max_similarity"] = maximum_similarities
    if dataset.empty:
        dataset["shap_values"] = pd.Series(dtype="str")
        dataset.to_parquet(dataset_path, index=False)
        return dataset_path
    top_model_input = model_input.loc[top_indices]

    shap_values = model.get_feature_importance(
        Pool(top_model_input, cat_features=categorical_features),
        type="ShapValues",
    )
    shap_values = shap_values[:, :-1]

    shap_payloads = []
    for (_, feature_row), contribution_row in zip(
        top_model_input.iterrows(),
        shap_values,
    ):
        payload = {}
        for feature, contribution in zip(feature_names, contribution_row):
            value = feature_row[feature]
            if hasattr(value, "item"):
                value = value.item()
            if pd.isna(value):
                value = None
            payload[feature] = {
                "value": value,
                "shap": round(float(contribution), 6),
            }
        shap_payloads.append(json.dumps(payload, ensure_ascii=False))

    dataset["shap_values"] = shap_payloads
    dataset = dataset.reset_index(drop=True)

    temporary_path = dataset_path.with_name(f".{dataset_path.name}.tmp")
    dataset.to_parquet(temporary_path, index=False)
    temporary_path.replace(dataset_path)
    return dataset_path


def build_dataset_for_query(query: str, budget=None) -> Path:
    """Собрать статьи по запросу и построить итоговый набор признаков."""
    from on_demand_parsing.parser import run_parser
    from preprocessing.create_features import build_real_features

    query = query.strip()
    if not query:
        raise ValueError("Запрос не должен быть пустым")

    run_parser(query, **({"budget": budget} if budget else {}))
    source_path = get_parsed_corpus_path(query)

    output_path = get_built_dataset_path(query)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    if pq.read_metadata(source_path).num_rows == 0:
        pd.DataFrame(columns=[
            "doc_id",
            "search_partial",
            "target_emergence",
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
        ]).assign(search_partial=bool(budget and budget.truncated)).to_parquet(output_path, index=False)
        return output_path

    build_real_features(str(source_path), str(output_path))
    return add_model_targets(output_path, source_path, query=query,
                             **({"budget": budget} if budget else {}))


if __name__ == "__main__":
    user_query = input("Введите запрос: ")
    dataset_path = build_dataset_for_query(user_query)
    print(f"Датасет сохранён: {dataset_path}")
