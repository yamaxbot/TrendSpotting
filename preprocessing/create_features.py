from collections import defaultdict
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


ESI_ALPHA = 0.5
ESI_BETA = 0.5
ESI_GAMMA = 1.0
ESI_EPSILON = 1e-8
ESI_QUANTILE = 0.90
PAST_WINDOW_YEARS = 3
FUTURE_START_OFFSET = 1
FUTURE_WINDOW_YEARS = 3
LAST_COMPLETE_YEAR = 2025


def extract_author_ids(value) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
    if not isinstance(value, (list, tuple, np.ndarray)):
        return []
    if all(isinstance(author, str) for author in value):
        return [author for author in value if author]
    return [
        str((authorship.get("author") or {}).get("id"))
        for authorship in value
        if isinstance(authorship, dict) and (authorship.get("author") or {}).get("id")
    ]


def calculate_esi_target(df: pd.DataFrame) -> pd.Series:
    """Calculate topic-year emergence labels from three-year windows."""
    print("[INFO] Расчет целевой переменной ESI по трехлетним временным окнам.")

    publication_counts = defaultdict(int)
    topic_year_authors = defaultdict(set)

    authors_column = "authors_ids" if "authors_ids" in df.columns else "authorships_json"
    for topic, year, authors in zip(
        df["primary_topic_id"], df["pub_year"], df[authors_column]
    ):
        if pd.isna(topic) or pd.isna(year):
            continue
        key = (str(topic), int(year))
        publication_counts[key] += 1
        topic_year_authors[key].update(extract_author_ids(authors))

    years = pd.to_numeric(df["pub_year"], errors="coerce").dropna()
    if years.empty:
        return pd.Series(pd.NA, index=df.index, dtype="Int64")

    first_year = int(years.min())
    last_year = min(int(years.max()), LAST_COMPLETE_YEAR)
    valid_years = range(
        first_year + PAST_WINDOW_YEARS,
        last_year - FUTURE_START_OFFSET - FUTURE_WINDOW_YEARS + 2,
    )
    topics = {str(topic) for topic in df["primary_topic_id"].dropna().unique()}
    esi_by_year = defaultdict(dict)

    for year in valid_years:
        for topic in topics:
            past_publications = np.mean(
                [publication_counts[(topic, current_year)]
                 for current_year in range(year - PAST_WINDOW_YEARS, year)]
            )
            future_publications = np.mean(
                [publication_counts[(topic, current_year)]
                 for current_year in range(
                     year + FUTURE_START_OFFSET,
                     year + FUTURE_START_OFFSET + FUTURE_WINDOW_YEARS,
                 )]
            )
            past_authors = np.mean(
                [len(topic_year_authors[(topic, current_year)])
                 for current_year in range(year - PAST_WINDOW_YEARS, year)]
            )
            future_authors = np.mean(
                [len(topic_year_authors[(topic, current_year)])
                 for current_year in range(
                     year + FUTURE_START_OFFSET,
                     year + FUTURE_START_OFFSET + FUTURE_WINDOW_YEARS,
                 )]
            )

            publication_growth = np.log(
                (future_publications + ESI_EPSILON)
                / (past_publications + ESI_EPSILON)
            )
            author_growth = np.log(
                (future_authors + ESI_EPSILON)
                / (past_authors + ESI_EPSILON)
            )
            author_delta = future_authors - past_authors
            author_scale = author_delta / np.sqrt(past_authors + ESI_GAMMA)
            esi = np.arcsinh(publication_growth)
            esi *= 1 + ESI_ALPHA * np.tanh(max(0.0, author_growth))
            esi *= 1 / (1 + np.exp(-ESI_BETA * author_scale))
            esi_by_year[year][topic] = float(esi)

    thresholds = {
        year: float(np.quantile(list(scores.values()), ESI_QUANTILE))
        for year, scores in esi_by_year.items()
        if scores
    }
    target_by_topic_year = {
        (topic, year): int(score >= thresholds[year])
        for year, scores in esi_by_year.items()
        for topic, score in scores.items()
    }

    target = pd.Series(pd.NA, index=df.index, dtype="Int64")
    for index, (topic, year) in enumerate(
        zip(df["primary_topic_id"], df["pub_year"])
    ):
        if pd.isna(topic) or pd.isna(year):
            continue
        label = target_by_topic_year.get((str(topic), int(year)))
        if label is not None:
            target.iloc[index] = label

    print(
        f"[INFO] ESI рассчитан для {len(target_by_topic_year):,} пар тема-год; "
        f"размечены годы {first_year + PAST_WINDOW_YEARS}-"
        f"{last_year - FUTURE_START_OFFSET - FUTURE_WINDOW_YEARS + 1}."
    )
    return target


def calc_author_growth_fast(df: pd.DataFrame) -> pd.Series:
    """Расчет author_growth_rate без преобразования таблицы через explode."""
    print("[INFO] Расчет показателя динамики авторского состава.")

    # Используются только колонки, необходимые для расчета показателя.
    authors_column = "authors_ids" if "authors_ids" in df.columns else "authorships_json"
    subset = df[["primary_topic_id", "pub_year", authors_column]].dropna(
        subset=["primary_topic_id", "pub_year"]
    )

    # Множества авторов агрегируются по теме и году без формирования промежуточной таблицы.
    topic_year_authors = defaultdict(set)

    for top_id, y, auths in zip(
        subset["primary_topic_id"], subset["pub_year"], subset[authors_column]
    ):
        # Добавляем айдишники авторов в сет соответствующего (topic, year)
        topic_year_authors[(top_id, int(y))].update(extract_author_ids(auths))

    # Формируется таблица количества уникальных авторов по теме и году.
    topic_year_counts = {
        key: len(auth_set) for key, auth_set in topic_year_authors.items()
    }

    # Показатель рассчитывается итеративно, чтобы избежать преобразования
    # всей таблицы в object-массив.
    result = np.zeros(len(df), dtype=np.float32)
    for idx, (top_id, y) in enumerate(zip(df["primary_topic_id"], df["pub_year"])):
        if pd.isna(top_id) or pd.isna(y):
            continue
        y = int(y)

        a1 = topic_year_counts.get((top_id, y - 1), 0)
        a2 = topic_year_counts.get((top_id, y - 2), 0)

        if a1 == 0 and a2 == 0:
            continue

        result[idx] = float(np.log((a1 + 1) / (a2 + 1)))

    return pd.Series(result, index=df.index, dtype=np.float32)


def calculate_sample_graph_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate PPMI and venue-diversity proxies from links inside this corpus."""
    print("[INFO] Расчет выборочных показателей PPMI и разнообразия источников.")

    document_domain = dict(zip(df["doc_id"], df["domain_id"]))
    document_source = dict(zip(df["doc_id"], df["source_id"]))
    topic_domain_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    total_edges = 0

    for topic, references in zip(df["primary_topic_id"], df["referenced_works_ids"]):
        if isinstance(references, str):
            try:
                references = json.loads(references)
            except (TypeError, json.JSONDecodeError):
                references = []
        if not isinstance(references, (list, tuple, np.ndarray)):
            references = []

        domains = [document_domain.get(str(reference)) for reference in references]
        domains = [domain for domain in domains if pd.notna(domain)]

        if pd.isna(topic):
            continue
        topic = str(topic)
        for domain in domains:
            domain = str(domain)
            topic_domain_counts[(topic, domain)] += 1
            topic_counts[topic] += 1
            domain_counts[domain] += 1
            total_edges += 1

    ppmi_values = np.zeros(len(df), dtype=np.float32)
    venue_values = np.zeros(len(df), dtype=np.float32)
    for index, (topic, references) in enumerate(
        zip(df["primary_topic_id"], df["referenced_works_ids"])
    ):
        if isinstance(references, str):
            try:
                references = json.loads(references)
            except (TypeError, json.JSONDecodeError):
                references = []
        if not isinstance(references, (list, tuple, np.ndarray)):
            references = []
        domains = [document_domain.get(str(reference)) for reference in references]
        domains = [domain for domain in domains if pd.notna(domain)]

        if pd.isna(topic) or not domains or not total_edges:
            continue

        topic = str(topic)
        scores = []
        for domain in domains:
            domain = str(domain)
            joint = topic_domain_counts[(topic, domain)] / total_edges
            topic_probability = topic_counts[topic] / total_edges
            domain_probability = domain_counts[domain] / total_edges
            if joint and topic_probability and domain_probability:
                scores.append(max(0.0, np.log(joint / (topic_probability * domain_probability))))
        if scores:
            ppmi_values[index] = np.float32(np.mean(scores))

        sources = [document_source.get(str(reference)) for reference in references]
        sources = [source for source in sources if pd.notna(source)]
        if sources:
            venue_values[index] = np.float32(len(set(sources)) / len(sources))

    return (
        pd.Series(ppmi_values, index=df.index),
        pd.Series(venue_values, index=df.index),
    )


def calculate_outlier_proxy(df: pd.DataFrame) -> pd.Series:
    """Flag the top 5% novelty values within each topic as a sample proxy."""
    topic_thresholds = df.groupby("primary_topic_id")["novelty_raw"].transform(
        lambda values: values.quantile(0.95)
    )
    return (df["novelty_raw"] >= topic_thresholds).astype(np.int8)


def build_real_features(input_parquet_path: str, output_parquet_path: str):
    print(f"[INFO] Загрузка исходного набора: {input_parquet_path}")
    required_columns = [
        "doc_id", "source_tier", "primary_topic_id", "pub_year", "authors_ids",
        "authorships_json",
        "abstract_text", "title", "citations_at_cutoff", "counts_by_year",
        "domain_id", "source_id", "referenced_works_ids", "commercial_maturity_index",
        "has_ref_data", "target_emergence",
    ]
    df = pd.read_parquet(input_parquet_path, columns=required_columns)

    df["target_emergence"] = calculate_esi_target(df)

    # 1. Расчет динамики авторского состава.
    df["author_growth_rate"] = calc_author_growth_fast(df)

    # 2. Расчет текстовой новизны и тематической плотности через TF-IDF-центроиды.
    print("[INFO] Расчет текстовой новизны и тематической плотности.")
    abstracts = (
        df["abstract_text"].fillna("").astype(str)
        if "abstract_text" in df.columns
        else df["title"].fillna("").astype(str)
    )

    # L2-нормализованный TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=2000, stop_words="english", dtype=np.float32
    )
    tfidf_matrix = vectorizer.fit_transform(abstracts)

    # Расчет центроида TF-IDF-представлений для каждой темы.
    df["topic_clean"] = df["primary_topic_id"].fillna("unknown")
    unique_topics = df["topic_clean"].unique()

    # Считаем средний вектор (центроид) для каждой темы
    from scipy.sparse import csr_matrix

    topic_to_idx = {t: i for i, t in enumerate(unique_topics)}
    topic_indices = df["topic_clean"].map(topic_to_idx).values

    # Формирование разреженной матрицы принадлежности работ темам.
    group_matrix = csr_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (topic_indices, np.arange(len(df))),
        ),
        shape=(len(unique_topics), len(df)),
    )

    # Агрегация и нормализация тематических центроидов.
    centroids = group_matrix.dot(tfidf_matrix)
    counts = np.asarray(group_matrix.sum(axis=1)).ravel()
    counts[counts == 0] = 1
    centroids = centroids.multiply(1.0 / counts[:, None])

    # Новизна определяется как единица минус косинусное сходство работы
    # с центроидом соответствующей темы. Расчет выполняется по темам,
    # чтобы не формировать крупную промежуточную sparse-матрицу.
    from sklearn.preprocessing import normalize

    centroids_norm = normalize(centroids, axis=1)
    tfidf_norm = normalize(tfidf_matrix, axis=1)
    cosine_sim = np.zeros(len(df), dtype=np.float32)

    for topic, topic_idx in topic_to_idx.items():
        mask = topic_indices == topic_idx
        if not np.any(mask):
            continue

        centroid = np.asarray(centroids_norm[topic_idx].toarray()).ravel()
        topic_vectors = tfidf_norm[mask].toarray()
        cosine_sim[mask] = (topic_vectors @ centroid).astype(np.float32)

    # Вычисление текстовой новизны.
    df["novelty_raw"] = (1.0 - cosine_sim).astype(np.float32)

    df["is_outlier_cluster"] = calculate_outlier_proxy(df)
    df["ppmi_domain_score"], df["distinct_venues_ratio"] = calculate_sample_graph_features(df)

    # Норма центроида используется как приближенная оценка тематической плотности.
    topic_densities = (
        centroids.multiply(centroids).sum(axis=1)
    )  # Норма центроида как прокси кучности
    topic_density_dict = dict(
        zip(unique_topics, np.asarray(topic_densities).ravel())
    )
    df["cluster_density"] = (
        df["topic_clean"].map(topic_density_dict).fillna(0.0).astype(np.float32)
    )

    df.drop(columns=["topic_clean"], inplace=True)
    print("[INFO] Векторные признаки рассчитаны.")

    # 3. Динамика цитирований
    print("[INFO] Расчет показателей цитирования.")
    df["citation_velocity"] = df.apply(
        lambda r: (
            float(r["citations_at_cutoff"] / max(1, 2026 - int(r["pub_year"])))
            if pd.notna(r["citations_at_cutoff"]) and pd.notna(r["pub_year"])
            else 0.0
        ),
        axis=1,
    )

    def parse_acc(val):
        if not val or pd.isna(val):
            return 0.0
        try:
            data = json.loads(val) if isinstance(val, str) else val
            c_dict = {
                item["year"]: item.get("cited_by_count", 0) for item in data
            }
            years = sorted(c_dict.keys(), reverse=True)
            if len(years) < 2:
                return 0.0
            last_12m = c_dict.get(years[0], 0)
            prev_24m = sum(c_dict.get(y, 0) for y in years[1:3])
            return float((last_12m + 1e-5) / (prev_24m + 1e-5))
        except:
            return 0.0

    df["citation_acceleration"] = df["counts_by_year"].apply(parse_acc)

    # 4. Формирование итоговой таблицы из 13 колонок.
    print("[INFO] Формирование итогового набора признаков.")

    # Рассчитанные признаки.
    computed_cols = {
        "novelty_raw": df["novelty_raw"],
        "cluster_density": df["cluster_density"],
        "author_growth_rate": df["author_growth_rate"],
        "citation_velocity": df["citation_velocity"],
        "citation_acceleration": df["citation_acceleration"],
    }

    for col_name, val_series in computed_cols.items():
        df[col_name] = val_series

    # Восстановление обязательных колонок, отсутствующих в исходном наборе.
    default_schema = {
        "doc_id": "unknown",
        "source_tier": "other",
        "commercial_maturity_index": 0.0,
        "has_ref_data": 1,
    }

    for col, def_val in default_schema.items():
        if col not in df.columns:
            df[col] = def_val

    # Неразмеченные значения сохраняются как пропуски и не интерпретируются
    # как отрицательный класс.
    if "target_emergence" not in df.columns:
        df["target_emergence"] = pd.NA
    else:
        df["target_emergence"] = pd.to_numeric(df["target_emergence"], errors="coerce")

    # Строгий список из всех 13 колонок
    final_13_cols = [
        "doc_id",
        "source_tier",
        "novelty_raw",
        "cluster_density",
        "is_outlier_cluster",
        "ppmi_domain_score",
        "commercial_maturity_index",
        "citation_velocity",
        "citation_acceleration",
        "distinct_venues_ratio",
        "author_growth_rate",
        "has_ref_data",
        "target_emergence",
    ]

    out_df = df[final_13_cols]
    out_df.to_parquet(output_parquet_path, index=False)
    print(
        f"[DONE] Итоговый набор содержит {out_df.shape[1]} колонок и {out_df.shape[0]} строк."
    )

if __name__ == "__main__":
    build_real_features(
        "data/openalex_corpus_1m.parquet", "data/train_data/processed_ml_dataset.parquet"
    )