from collections import defaultdict
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
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
    """Calculate author growth from the two strictly preceding years."""
    print("[INFO] Расчет показателя динамики авторского состава.")

    # Используются только колонки, необходимые для расчета показателя.
    authors_column = "authors_ids" if "authors_ids" in df.columns else "authorships_json"
    subset = df[["primary_topic_id", "pub_year", authors_column]].dropna(
        subset=["primary_topic_id", "pub_year"]
    )

    # Future rows cannot contribute because each lookup is explicitly y-1/y-2.
    topic_year_authors = defaultdict(set)

    for top_id, y, auths in zip(
        subset["primary_topic_id"], subset["pub_year"], subset[authors_column]
    ):
        # Добавляем айдишники авторов в сет соответствующего (topic, year)
        topic_year_authors[(top_id, int(y))].update(extract_author_ids(auths))

    topic_year_counts = {
        key: len(auth_set) for key, auth_set in topic_year_authors.items()
    }

    result = np.zeros(len(df), dtype=np.float32)
    for idx, (top_id, y) in enumerate(zip(df["primary_topic_id"], df["pub_year"])):
        if pd.isna(top_id) or pd.isna(y):
            continue
        y = int(y)

        # Only information available before the prediction year is allowed.
        a1 = topic_year_counts.get((top_id, y - 1), 0)
        a2 = topic_year_counts.get((top_id, y - 2), 0)

        if a1 == 0 and a2 == 0:
            continue

        result[idx] = float(np.log((a1 + 1) / (a2 + 1)))

    return pd.Series(result, index=df.index, dtype=np.float32)


def _parse_list(value) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []
    return list(value) if isinstance(value, (list, tuple, np.ndarray)) else []


def calculate_temporal_text_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build fast point-in-time text features with cumulative topic centroids."""
    print("[INFO] Расчет быстрых временных текстовых признаков без будущих документов.")
    years = pd.to_numeric(df["pub_year"], errors="coerce")
    abstracts = df["abstract_text"].fillna("").astype(str)
    novelty = np.ones(len(df), dtype=np.float32)
    density = np.zeros(len(df), dtype=np.float32)

    # Hashing has a fixed vocabulary and needs no fit, so future documents never
    # influence the representation of earlier documents.
    vectorizer = HashingVectorizer(
        n_features=2000,
        stop_words="english",
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )
    text_matrix = vectorizer.transform(abstracts)
    topic_values = df["primary_topic_id"].fillna("unknown").astype(str).to_numpy()
    year_values = years.to_numpy()
    topic_sums: dict[str, np.ndarray] = {}
    topic_counts: defaultdict[str, int] = defaultdict(int)

    for year in sorted(years.dropna().astype(int).unique()):
        current_indices = np.flatnonzero(year_values == year)
        current_topics, current_topic_indices = np.unique(
            topic_values[current_indices], return_inverse=True
        )
        for topic_index, topic in enumerate(current_topics):
            selected_indices = current_indices[current_topic_indices == topic_index]
            count = topic_counts[topic]
            rows = text_matrix[selected_indices]
            if count:
                centroid = topic_sums[topic] / count
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm:
                    similarities = np.asarray(
                        rows @ (centroid / centroid_norm)
                    ).ravel()
                    novelty[selected_indices] = 1.0 - similarities.astype(np.float32)
                density[selected_indices] = np.float32(np.dot(centroid, centroid))

            topic_sum = np.asarray(rows.sum(axis=0)).ravel().astype(np.float32)
            if topic not in topic_sums:
                topic_sums[topic] = np.zeros(text_matrix.shape[1], dtype=np.float32)
            topic_sums[topic] += topic_sum
            topic_counts[topic] += len(selected_indices)

    return (
        pd.Series(novelty, index=df.index, dtype=np.float32),
        pd.Series(density, index=df.index, dtype=np.float32),
    )


def calculate_temporal_graph_features(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Calculate graph features using only the graph known by each year."""
    print("[INFO] Расчет временных PPMI и разнообразия источников.")
    years = pd.to_numeric(df["pub_year"], errors="coerce")
    document_info = {
        str(doc_id): (year, domain, source)
        for doc_id, year, domain, source in zip(
            df["doc_id"], years, df["domain_id"], df["source_id"]
        )
        if pd.notna(year)
    }
    references_by_index = [_parse_list(value) for value in df["referenced_works_ids"]]
    ppmi = np.zeros(len(df), dtype=np.float32)
    venue = np.zeros(len(df), dtype=np.float32)
    topic_domain_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    total_edges = 0

    for year in sorted(years.dropna().astype(int).unique()):
        current_indices = np.flatnonzero(years.to_numpy() == year)
        for index in current_indices:
            topic = df.iloc[index]["primary_topic_id"]
            if pd.isna(topic):
                continue
            topic = str(topic)
            domains = []
            sources = []
            for reference in references_by_index[index]:
                info = document_info.get(str(reference))
                if info is None or info[0] >= year:
                    continue
                if pd.notna(info[1]):
                    domains.append(str(info[1]))
                if pd.notna(info[2]):
                    sources.append(str(info[2]))
            if sources:
                venue[index] = np.float32(len(set(sources)) / len(sources))
            if total_edges:
                scores = []
                for domain in domains:
                    joint = topic_domain_counts[(topic, domain)] / total_edges
                    topic_probability = topic_counts[topic] / total_edges
                    domain_probability = domain_counts[domain] / total_edges
                    if joint > 0 and topic_probability > 0 and domain_probability > 0:
                        scores.append(max(0.0, np.log(joint / (topic_probability * domain_probability))))
                if scores:
                    ppmi[index] = np.float32(np.mean(scores))

        for index in current_indices:
            topic = df.iloc[index]["primary_topic_id"]
            if pd.isna(topic):
                continue
            topic = str(topic)
            domains = []
            for reference in references_by_index[index]:
                info = document_info.get(str(reference))
                if info is not None and info[0] < year and pd.notna(info[1]):
                    domains.append(str(info[1]))
            for domain in domains:
                topic_domain_counts[(topic, domain)] += 1
                topic_counts[topic] += 1
                domain_counts[domain] += 1
                total_edges += 1

    return pd.Series(ppmi, index=df.index), pd.Series(venue, index=df.index)


def calculate_outlier_proxy(df: pd.DataFrame) -> pd.Series:
    """Flag novelty using only rows available by each row's publication year."""
    years = pd.to_numeric(df["pub_year"], errors="coerce")
    result = np.zeros(len(df), dtype=np.int8)
    for year in sorted(years.dropna().astype(int).unique()):
        available = years < year
        current = years == year
        thresholds = df.loc[available].groupby("primary_topic_id")["novelty_raw"].quantile(0.95)
        current_thresholds = df.loc[current, "primary_topic_id"].map(thresholds)
        result[np.flatnonzero(current)] = (
            df.loc[current, "novelty_raw"].to_numpy() >= current_thresholds.to_numpy()
        ).astype(np.int8)
    return pd.Series(result, index=df.index)


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

    # 2. Расчет текстовых и графовых признаков только по доступной истории.
    df["novelty_raw"], df["cluster_density"] = calculate_temporal_text_features(df)
    df["is_outlier_cluster"] = calculate_outlier_proxy(df)
    df["ppmi_domain_score"], df["distinct_venues_ratio"] = calculate_temporal_graph_features(df)
    print("[INFO] Векторные признаки рассчитаны.")

    # 3. Динамика цитирований
    print("[INFO] Расчет показателей цитирования.")
    def parse_acc(val, cutoff_year: int) -> float:
        """Use fixed calendar-year bins ending at the prediction year."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0.0

        try:
            if isinstance(val, str):
                data = json.loads(val)
            elif isinstance(val, (list, tuple, np.ndarray)):
                data = val
            elif pd.isna(val):
                return 0.0
            else:
                return 0.0

            if len(data) == 0:
                return 0.0

            c_dict = {
                item["year"]: item.get("cited_by_count", 0)
                for item in data
            }
            last_12m = c_dict.get(cutoff_year, 0)
            prev_24m = c_dict.get(cutoff_year - 1, 0) + c_dict.get(cutoff_year - 2, 0)
            return float((last_12m + 1e-5) / (prev_24m + 1e-5))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            return 0.0

    historical_citations = []
    historical_acceleration = []
    for pub_year, counts in zip(df["pub_year"], df["counts_by_year"]):
        if pd.isna(pub_year):
            historical_citations.append(0.0)
            historical_acceleration.append(0.0)
            continue
        try:
            parsed = json.loads(counts) if isinstance(counts, str) else counts
            parsed = [
                item for item in (parsed or [])
                if item.get("year", 0) <= int(pub_year)
            ]
            historical_citations.append(
                float(sum(item.get("cited_by_count", 0) for item in parsed))
            )
            historical_acceleration.append(parse_acc(parsed, int(pub_year)))
        except (TypeError, ValueError, json.JSONDecodeError):
            historical_citations.append(0.0)
            historical_acceleration.append(0.0)

    df["citation_velocity"] = pd.Series(
        historical_citations, index=df.index, dtype=np.float32
    )
    df["citation_acceleration"] = pd.Series(
        historical_acceleration, index=df.index, dtype=np.float32
    )

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