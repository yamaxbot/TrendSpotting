import re
import sys
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBSITE_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "education" / "catboost_model.cbm"

# Позволяет запускать файл напрямую: python website/main.py
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


def add_model_targets(
    dataset_path: str | Path,
    corpus_path: str | Path | None = None,
) -> Path:
    """Добавить предсказания CatBoost и сохранить 15 лучших результатов."""
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

    dataset = pd.read_parquet(dataset_path)
    dataset = dataset.drop(columns=["target_emergence"], errors="ignore")

    if dataset.empty:
        dataset["target_emergence"] = pd.Series(dtype="int8")
        dataset["model_confidence"] = pd.Series(dtype="float64")
        dataset["diversity_max_similarity"] = pd.Series(dtype="float64")
        dataset["shap_values"] = pd.Series(dtype="str")
        dataset.to_parquet(dataset_path, index=False)
        return dataset_path

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    feature_names = model.feature_names_
    missing_features = [
        feature for feature in feature_names if feature not in dataset.columns
    ]
    if missing_features:
        raise ValueError(
            "В датасете отсутствуют признаки модели: "
            + ", ".join(missing_features)
        )

    model_input = dataset[feature_names].copy()
    model_input["source_tier"] = (
        model_input["source_tier"].fillna("other").astype(str)
    )
    predictions = model.predict(model_input).astype("int8").ravel()
    probabilities = model.predict_proba(model_input)
    positive_class_index = list(model.classes_).index(1)

    dataset["target_emergence"] = predictions
    dataset["model_confidence"] = (
        probabilities[:, positive_class_index] * 100
    )
    if corpus_path is None:
        text_columns = [
            column
            for column in ("doc_id", "title", "abstract_text")
            if column in dataset.columns
        ]
        article_texts = dataset[text_columns].copy()
    else:
        corpus_path = Path(corpus_path).resolve()
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Корпус не найден: {corpus_path}")
        article_texts = pd.read_parquet(
            corpus_path,
            columns=["doc_id", "title", "abstract_text"],
        )

    from website.diversity import select_diverse_top

    top_indices, maximum_similarities = select_diverse_top(
        dataset,
        article_texts,
        limit=15,
    )
    dataset = dataset.loc[top_indices].copy()
    dataset["model_confidence"] = dataset["model_confidence"].round(2)
    dataset["diversity_max_similarity"] = maximum_similarities
    top_model_input = model_input.loc[top_indices]

    shap_values = model.get_feature_importance(
        Pool(top_model_input, cat_features=["source_tier"]),
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


def build_dataset_for_query(query: str) -> Path:
    """Собрать статьи по запросу и построить итоговый набор признаков."""
    from on_demand_parsing.parser import run_parser
    from preprocessing.create_features import build_real_features

    query = query.strip()
    if not query:
        raise ValueError("Запрос не должен быть пустым")

    run_parser(query)
    source_path = get_parsed_corpus_path(query)

    output_path = get_built_dataset_path(query)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import pyarrow.parquet as pq
    if pq.read_metadata(source_path).num_rows == 0:
        pd.DataFrame(columns=[
            "doc_id",
            "target_emergence",
            "model_confidence",
            "diversity_max_similarity",
            "shap_values",
        ]).to_parquet(output_path, index=False)
        return output_path

    build_real_features(str(source_path), str(output_path))
    return add_model_targets(output_path, source_path)


if __name__ == "__main__":
    user_query = input("Введите запрос: ")
    dataset_path = build_dataset_for_query(user_query)
    print(f"Датасет сохранён: {dataset_path}")
