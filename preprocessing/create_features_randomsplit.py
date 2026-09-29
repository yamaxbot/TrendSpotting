from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split


PAST_WINDOW_YEARS = 3
FUTURE_WINDOW_YEARS = 3

ESI_QUANTILE = 0.90
MIN_FUTURE_VOLUME = 3
SMOOTHING = 5.0

TEXT_N_FEATURES = 2000

LOG_GROWTH_CLIP = 5.0

SPLIT_MODE = "random"

RANDOM_STATE = 42
TRAIN_SIZE = 0.80
VALID_SIZE = 0.10
TEST_SIZE = 0.10

TEMPORAL_TRAIN_END_YEAR = 2018
TEMPORAL_VALID_END_YEAR = 2020

RANDOM_HISTORY_ONLY_TRAIN = True

INPUT_PATH = "download_dataset/new_dataset/new_data/openalex_corpus_1m.parquet"
OUTPUT_PATH = "download_dataset/new_dataset/new_data/final_openalex_dataset.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_json_list(value: Any) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    return []


def extract_author_ids(value: Any) -> list[str]:
    value = parse_json_list(value)

    if not value:
        return []

    if all(isinstance(author, str) for author in value):
        return [author for author in value if author]

    result: list[str] = []

    for authorship in value:
        if not isinstance(authorship, dict):
            continue

        author = authorship.get("author") or {}
        author_id = author.get("id")

        if author_id:
            result.append(str(author_id))

    return result


def parse_counts_by_year(value: Any) -> list[dict]:
    value = parse_json_list(value)
    return [item for item in value if isinstance(item, dict)]


def safe_len_json_list(value: Any) -> int:
    return len(parse_json_list(value))


def calculate_esi_target(df: pd.DataFrame) -> pd.Series:
    logger.info("Calculating historical emergence target...")

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    valid = pd.notna(topics) & pd.notna(years)

    work = pd.DataFrame(
        {
            "topic": topics[valid],
            "year": years[valid].astype(np.int32),
        }
    )

    topic_year_counts = work.groupby(["topic", "year"]).size().to_dict()

    keys = []
    for topic, year in zip(topics, years):
        if pd.isna(topic) or pd.isna(year):
            keys.append(None)
        else:
            keys.append((topic, int(year)))

    growth_by_key: dict[tuple, float] = {}
    future_by_key: dict[tuple, int] = {}

    for key in set(k for k in keys if k is not None):
        topic, year = key

        past = sum(
            topic_year_counts.get((topic, y), 0)
            for y in range(year - PAST_WINDOW_YEARS, year)
        )
        future = sum(
            topic_year_counts.get((topic, y), 0)
            for y in range(year + 1, year + FUTURE_WINDOW_YEARS + 1)
        )

        past_rate = (past + SMOOTHING) / PAST_WINDOW_YEARS
        future_rate = (future + SMOOTHING) / FUTURE_WINDOW_YEARS

        growth_by_key[key] = float(np.log2(future_rate / past_rate))
        future_by_key[key] = future

    growth = np.full(len(df), np.nan, dtype=np.float32)

    for i, key in enumerate(keys):
        if key is not None:
            growth[i] = growth_by_key[key]

    target = np.zeros(len(df), dtype=np.int8)

    for year in sorted(
        int(y) for y in pd.Series(years).dropna().unique()
    ):
        mask = years == year
        values = growth[mask]
        valid_values = values[np.isfinite(values)]

        if len(valid_values) == 0:
            continue

        threshold = np.quantile(valid_values, ESI_QUANTILE)

        enough_future = np.array(
            [
                future_by_key.get(keys[i], 0) >= MIN_FUTURE_VOLUME
                for i in np.flatnonzero(mask)
            ],
            dtype=bool,
        )

        target[mask] = (
            np.isfinite(values)
            & (values >= threshold)
            & enough_future
        ).astype(np.int8)

    result = pd.Series(target, index=df.index, dtype=np.int8)

    logger.info(
        "Target: %d positives / %d papers (%.3f%%)",
        int(result.sum()),
        len(result),
        100.0 * float(result.mean()),
    )

    return result


def _history_mask(
    df: pd.DataFrame,
    current_year: int,
    allowed_history_indices: np.ndarray | None,
) -> np.ndarray:
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    mask = np.isfinite(years) & (years < current_year)

    if allowed_history_indices is not None:
        allowed = np.zeros(len(df), dtype=bool)
        allowed[allowed_history_indices] = True
        mask &= allowed

    return mask


def calculate_topic_dynamics(
    df: pd.DataFrame,
    allowed_history_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    logger.info("Calculating topic publication dynamics...")

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    counts: dict[tuple, int] = defaultdict(int)

    if allowed_history_indices is None:
        history_indices = np.arange(len(df))
    else:
        history_indices = np.asarray(allowed_history_indices)

    for idx in history_indices:
        topic = topics[idx]
        year = years[idx]

        if pd.isna(topic) or pd.isna(year):
            continue

        counts[(topic, int(year))] += 1

    n = len(df)
    volume = np.zeros(n, dtype=np.float32)
    growth = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)
    age = np.zeros(n, dtype=np.float32)

    first_year: dict[Any, int] = {}
    for (topic, year), _count in counts.items():
        if topic not in first_year or year < first_year[topic]:
            first_year[topic] = year

    for i, (topic, year) in enumerate(zip(topics, years)):
        if pd.isna(topic) or pd.isna(year):
            continue

        year = int(year)

        y1 = counts.get((topic, year - 1), 0)
        y2 = counts.get((topic, year - 2), 0)
        y3 = counts.get((topic, year - 3), 0)

        volume[i] = np.float32(
            sum(counts.get((topic, y), 0) for y in range(year - 3, year))
        )

        g1 = np.log2((y1 + 1.0) / (y2 + 1.0))
        g2 = np.log2((y2 + 1.0) / (y3 + 1.0))

        growth[i] = np.float32(
            np.clip(g1, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP)
        )
        acceleration[i] = np.float32(
            np.clip(g1 - g2, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP)
        )

        age[i] = np.float32(
            max(0, year - first_year.get(topic, year))
        )

    return pd.DataFrame(
        {
            "topic_historical_volume": volume,
            "topic_publication_growth": growth,
            "topic_growth_acceleration": acceleration,
            "topic_age_years": age,
        },
        index=df.index,
    )


def calculate_author_features(
    df: pd.DataFrame,
    allowed_history_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    logger.info("Calculating author dynamics...")

    authors_column = (
        "authors_ids"
        if "authors_ids" in df.columns
        else "authorships_json"
    )

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    topic_year_authors: defaultdict[tuple, set] = defaultdict(set)

    if allowed_history_indices is None:
        history_indices = np.arange(len(df))
    else:
        history_indices = np.asarray(allowed_history_indices)

    for idx in history_indices:
        topic = topics[idx]
        year = years[idx]

        if pd.isna(topic) or pd.isna(year):
            continue

        year = int(year)

        for author_id in extract_author_ids(df.iloc[idx][authors_column]):
            topic_year_authors[(topic, year)].add(author_id)

    author_counts = {
        key: len(value)
        for key, value in topic_year_authors.items()
    }

    n = len(df)
    growth = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)
    volume = np.zeros(n, dtype=np.float32)

    for i, (topic, year) in enumerate(zip(topics, years)):
        if pd.isna(topic) or pd.isna(year):
            continue

        year = int(year)

        a1 = author_counts.get((topic, year - 1), 0)
        a2 = author_counts.get((topic, year - 2), 0)
        a3 = author_counts.get((topic, year - 3), 0)

        g1 = np.log2((a1 + 1.0) / (a2 + 1.0))
        g2 = np.log2((a2 + 1.0) / (a3 + 1.0))

        growth[i] = np.float32(
            np.clip(g1, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP)
        )
        acceleration[i] = np.float32(
            np.clip(g1 - g2, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP)
        )
        volume[i] = np.float32(a1)

    return pd.DataFrame(
        {
            "author_growth_rate": growth,
            "author_growth_acceleration": acceleration,
            "historical_author_count": volume,
        },
        index=df.index,
    )


def calculate_temporal_text_features(
    df: pd.DataFrame,
    text_matrix: scipy.sparse.csr_matrix,
    allowed_history_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    logger.info("Calculating temporal text geometry...")

    n = len(df)

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    novelty = np.ones(n, dtype=np.float32)
    own_similarity = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    local_volume = np.zeros(n, dtype=np.float32)
    nearest_similarity = np.zeros(n, dtype=np.float32)
    cross_gap = np.zeros(n, dtype=np.float32)
    outlier = np.zeros(n, dtype=np.int8)

    allowed = None
    if allowed_history_indices is not None:
        allowed = np.zeros(n, dtype=bool)
        allowed[np.asarray(allowed_history_indices)] = True

    valid_years = np.sort(
        np.unique(years[np.isfinite(years)]).astype(np.int32)
    )

    topic_sum: dict[Any, np.ndarray] = {}
    topic_count: defaultdict[Any, int] = defaultdict(int)

    previous_novelty: list[float] = []

    for year in valid_years:
        current = np.flatnonzero(years == year)

        if len(current) == 0:
            continue

        topic_to_indices: defaultdict[Any, list[int]] = defaultdict(list)

        for idx in current:
            topic = topics[idx]
            if pd.notna(topic):
                topic_to_indices[topic].append(idx)

        if not topic_to_indices:
            continue

        historical_topics = []
        centroid_vectors = []

        for topic, count in topic_count.items():
            if count <= 0:
                continue

            raw = topic_sum[topic] / count
            norm = np.linalg.norm(raw)

            if norm <= 1e-12:
                continue

            historical_topics.append(topic)
            centroid_vectors.append(raw / norm)

        if centroid_vectors:
            centroid_matrix = np.asarray(
                centroid_vectors,
                dtype=np.float32,
            )
            centroid_pos = {
                topic: i
                for i, topic in enumerate(historical_topics)
            }
        else:
            centroid_matrix = None
            centroid_pos = {}

        for topic, indices in topic_to_indices.items():
            count = topic_count[topic]

            if count <= 0:
                continue

            raw = topic_sum[topic] / count
            norm = np.linalg.norm(raw)

            local_volume[indices] = np.float32(count)
            density[indices] = np.float32(norm)

            if norm <= 1e-12:
                continue

            centroid_unit = raw / norm
            rows = text_matrix[indices]

            sims = np.asarray(rows @ centroid_unit).reshape(-1)
            sims = np.clip(sims, -1.0, 1.0).astype(np.float32)

            own_similarity[indices] = sims
            novelty[indices] = np.clip(
                1.0 - sims,
                0.0,
                2.0,
            )

        if centroid_matrix is not None and len(centroid_matrix) >= 2:
            rows = text_matrix[current]

            similarities = np.asarray(
                rows @ centroid_matrix.T
            )

            own_positions = np.full(
                len(current),
                -1,
                dtype=np.int32,
            )

            for i, idx in enumerate(current):
                own_positions[i] = centroid_pos.get(
                    topics[idx],
                    -1,
                )

            valid_own = own_positions >= 0
            valid_rows = np.flatnonzero(valid_own)

            if len(valid_rows):
                similarities[
                    valid_rows,
                    own_positions[valid_rows],
                ] = -np.inf

            nearest = similarities.max(axis=1)
            nearest = np.where(
                np.isfinite(nearest),
                nearest,
                0.0,
            )

            nearest = np.clip(
                nearest,
                -1.0,
                1.0,
            ).astype(np.float32)

            nearest_similarity[current] = nearest
            cross_gap[current] = (
                own_similarity[current] - nearest
            )

        if previous_novelty:
            threshold = np.quantile(
                np.asarray(previous_novelty),
                0.95,
            )
            outlier[current] = (
                novelty[current] >= threshold
            ).astype(np.int8)

        for topic, indices in topic_to_indices.items():
            if allowed is not None:
                history_indices = [
                    idx for idx in indices if allowed[idx]
                ]
            else:
                history_indices = indices

            if not history_indices:
                continue

            rows = text_matrix[history_indices]
            summed = np.asarray(
                rows.sum(axis=0)
            ).ravel().astype(np.float32)

            if topic in topic_sum:
                topic_sum[topic] += summed
            else:
                topic_sum[topic] = summed.copy()

            topic_count[topic] += len(history_indices)

            previous_novelty.extend(
                novelty[history_indices].tolist()
            )

    return pd.DataFrame(
        {
            "novelty_raw": novelty,
            "topic_centroid_similarity": own_similarity,
            "cluster_density": density,
            "topic_local_volume": local_volume,
            "nearest_topic_similarity": nearest_similarity,
            "cross_topic_gap": cross_gap,
            "is_outlier_cluster": outlier,
        },
        index=df.index,
    )


def calculate_temporal_graph_features(
    df: pd.DataFrame,
    allowed_history_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    logger.info("Calculating temporal topic-domain structure...")

    n = len(df)

    topics = df["primary_topic_id"].to_numpy()
    domains = df["domain_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    ppmi = np.zeros(n, dtype=np.float32)
    pair_rarity = np.zeros(n, dtype=np.float32)
    domain_history = np.zeros(n, dtype=np.float32)
    pair_age = np.zeros(n, dtype=np.float32)

    pair_counts: defaultdict[tuple, int] = defaultdict(int)
    topic_counts: defaultdict[Any, int] = defaultdict(int)
    domain_counts: defaultdict[Any, int] = defaultdict(int)
    pair_first_year: dict[tuple, int] = {}

    total = 0

    allowed = None
    if allowed_history_indices is not None:
        allowed = np.zeros(n, dtype=bool)
        allowed[np.asarray(allowed_history_indices)] = True

    valid_years = sorted(
        int(y) for y in pd.Series(years).dropna().unique()
    )

    for year in valid_years:
        current = np.flatnonzero(years == year)

        for idx in current:
            topic = topics[idx]
            domain = domains[idx]

            if pd.isna(topic) or pd.isna(domain):
                continue

            pair = (topic, domain)
            pair_count = pair_counts.get(pair, 0)

            if total > 0 and pair_count > 0:
                p_xy = pair_count / total
                p_x = topic_counts[topic] / total
                p_y = domain_counts[domain] / total

                if p_x > 0 and p_y > 0:
                    value = np.log2(
                        (p_xy + 1e-12)
                        / (p_x * p_y + 1e-12)
                    )
                    ppmi[idx] = np.float32(max(0.0, value))

            pair_rarity[idx] = np.float32(
                1.0 / np.log1p(pair_count + 1.0)
            )

            domain_history[idx] = np.float32(
                domain_counts.get(domain, 0)
            )

            if pair in pair_first_year:
                pair_age[idx] = np.float32(
                    max(0, year - pair_first_year[pair])
                )

        for idx in current:
            if allowed is not None and not allowed[idx]:
                continue

            topic = topics[idx]
            domain = domains[idx]

            if pd.isna(topic) or pd.isna(domain):
                continue

            pair = (topic, domain)

            pair_counts[pair] += 1
            topic_counts[topic] += 1
            domain_counts[domain] += 1
            total += 1

            if pair not in pair_first_year:
                pair_first_year[pair] = year

    return pd.DataFrame(
        {
            "ppmi_domain_score": ppmi,
            "topic_domain_rarity": pair_rarity,
            "domain_history_count": domain_history,
            "topic_domain_age_years": pair_age,
        },
        index=df.index,
    )


def calculate_source_features(
    df: pd.DataFrame,
    allowed_history_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    logger.info("Calculating historical source structure...")

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()
    sources = df["source_id"].to_numpy()

    n = len(df)

    source_counts: defaultdict[tuple, int] = defaultdict(int)
    topic_year_sources: defaultdict[Any, set] = defaultdict(set)

    source_history_count = np.zeros(n, dtype=np.float32)
    source_novelty = np.zeros(n, dtype=np.float32)
    topic_source_diversity = np.zeros(n, dtype=np.float32)

    allowed = None
    if allowed_history_indices is not None:
        allowed = np.zeros(n, dtype=bool)
        allowed[np.asarray(allowed_history_indices)] = True

    valid_years = sorted(
        int(y) for y in pd.Series(years).dropna().unique()
    )

    for year in valid_years:
        current = np.flatnonzero(years == year)
        topic_to_sources: defaultdict[Any, set] = defaultdict(set)

        for idx in current:
            topic = topics[idx]
            source = sources[idx]

            if pd.isna(topic) or pd.isna(source):
                continue

            key = (topic, source)

            source_history_count[idx] = np.float32(
                source_counts.get(key, 0)
            )
            source_novelty[idx] = np.float32(
                1.0 / np.log1p(
                    source_counts.get(key, 0) + 1.0
                )
            )

            historical_sources = topic_year_sources.get(
                topic,
                set(),
            )
            topic_source_diversity[idx] = np.float32(
                len(historical_sources)
            )

            if allowed is None or allowed[idx]:
                topic_to_sources[topic].add(source)

        for topic, source_set in topic_to_sources.items():
            topic_year_sources[topic].update(source_set)

        for idx in current:
            if allowed is not None and not allowed[idx]:
                continue

            topic = topics[idx]
            source = sources[idx]

            if pd.isna(topic) or pd.isna(source):
                continue

            source_counts[(topic, source)] += 1

    return pd.DataFrame(
        {
            "source_history_count": source_history_count,
            "source_novelty": source_novelty,
            "topic_source_diversity": topic_source_diversity,
        },
        index=df.index,
    )


def calculate_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating bibliographic features...")

    n = len(df)

    reference_count = np.zeros(n, dtype=np.float32)
    has_references = np.zeros(n, dtype=np.int8)

    refs = df["referenced_works_ids"]

    for i, value in enumerate(refs):
        count = safe_len_json_list(value)
        reference_count[i] = np.float32(count)
        has_references[i] = np.int8(count > 0)

    return pd.DataFrame(
        {
            "reference_count": reference_count,
            "has_reference_list": has_references,
        },
        index=df.index,
    )


def calculate_historical_citation_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Calculating historical citation features...")

    n = len(df)

    velocity = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)

    for i, (year, counts) in enumerate(
        zip(df["pub_year"], df["counts_by_year"])
    ):
        if pd.isna(year):
            continue

        year = int(year)

        parsed = parse_counts_by_year(counts)

        count_by_year = {
            int(item["year"]): float(item.get("cited_by_count", 0))
            for item in parsed
            if "year" in item and int(item["year"]) <= year
        }

        velocity[i] = np.float32(sum(count_by_year.values()))

        previous = count_by_year.get(year - 1, 0.0)
        current = count_by_year.get(year, 0.0)

        acceleration[i] = np.float32(
            np.clip(
                np.log2((current + 1.0) / (previous + 1.0)),
                -LOG_GROWTH_CLIP,
                LOG_GROWTH_CLIP,
            )
        )

    return pd.DataFrame(
        {
            "citation_velocity": velocity,
            "citation_acceleration": acceleration,
        },
        index=df.index,
    )


def make_splits(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    all_indices = np.arange(n)

    if SPLIT_MODE == "temporal":
        years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

        train_idx = np.flatnonzero(
            np.isfinite(years)
            & (years <= TEMPORAL_TRAIN_END_YEAR)
        )

        valid_idx = np.flatnonzero(
            np.isfinite(years)
            & (years > TEMPORAL_TRAIN_END_YEAR)
            & (years <= TEMPORAL_VALID_END_YEAR)
        )

        test_idx = np.flatnonzero(
            np.isfinite(years)
            & (years > TEMPORAL_VALID_END_YEAR)
        )

        return train_idx, valid_idx, test_idx

    if SPLIT_MODE != "random":
        raise ValueError(
            f"Unknown SPLIT_MODE={SPLIT_MODE!r}. "
            "Use 'random' or 'temporal'."
        )

    target = df["target_emergence"].to_numpy()

    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=VALID_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    relative_test_size = TEST_SIZE / (VALID_SIZE + TEST_SIZE)

    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=target[temp_idx],
    )

    return (
        np.sort(train_idx),
        np.sort(valid_idx),
        np.sort(test_idx),
    )


def build_real_features(
    input_parquet_path: str,
    output_parquet_path: str,
) -> None:
    logger.info("Loading source dataset: %s", input_parquet_path)

    required_columns = [
        "doc_id",
        "source_tier",
        "primary_topic_id",
        "pub_year",
        "authors_ids",
        "authorships_json",
        "abstract_text",
        "title",
        "counts_by_year",
        "domain_id",
        "source_id",
        "referenced_works_ids",
        "commercial_maturity_index",
        "has_ref_data",
    ]

    df = pd.read_parquet(
        input_parquet_path,
        columns=required_columns,
    )

    logger.info("Loaded %d papers.", len(df))

    df["pub_year"] = pd.to_numeric(
        df["pub_year"],
        errors="coerce",
    )

    df["target_emergence"] = calculate_esi_target(df)

    train_idx, valid_idx, test_idx = make_splits(df)

    split = np.full(len(df), "unused", dtype=object)
    split[train_idx] = "train"
    split[valid_idx] = "valid"
    split[test_idx] = "test"

    df["split"] = split

    logger.info(
        "Split sizes: train=%d (%.2f%%), valid=%d (%.2f%%), "
        "test=%d (%.2f%%)",
        len(train_idx),
        100.0 * len(train_idx) / len(df),
        len(valid_idx),
        100.0 * len(valid_idx) / len(df),
        len(test_idx),
        100.0 * len(test_idx) / len(df),
    )

    for name, idx in (
        ("train", train_idx),
        ("valid", valid_idx),
        ("test", test_idx),
    ):
        if len(idx):
            rate = float(df.iloc[idx]["target_emergence"].mean())
            years = pd.to_numeric(
                df.iloc[idx]["pub_year"],
                errors="coerce",
            )
            logger.info(
                "%s: positives=%d (%.3f%%), years=%s-%s",
                name,
                int(df.iloc[idx]["target_emergence"].sum()),
                100.0 * rate,
                int(years.min()) if years.notna().any() else "NA",
                int(years.max()) if years.notna().any() else "NA",
            )

    if SPLIT_MODE == "random" and RANDOM_HISTORY_ONLY_TRAIN:
        allowed_history_indices = train_idx
    else:
        allowed_history_indices = None

    topic_dyn = calculate_topic_dynamics(
        df,
        allowed_history_indices=allowed_history_indices,
    )

    author_dyn = calculate_author_features(
        df,
        allowed_history_indices=allowed_history_indices,
    )

    text = (
        df["title"].fillna("").astype(str)
        + " "
        + df["abstract_text"].fillna("").astype(str)
    )

    vectorizer = HashingVectorizer(
        n_features=TEXT_N_FEATURES,
        stop_words="english",
        alternate_sign=False,
        norm="l2",
        dtype=np.float32,
    )

    logger.info("Vectorizing title + abstract...")
    text_matrix = vectorizer.transform(text)

    text_features = calculate_temporal_text_features(
        df,
        text_matrix,
        allowed_history_indices=allowed_history_indices,
    )

    del text_matrix

    graph_features = calculate_temporal_graph_features(
        df,
        allowed_history_indices=allowed_history_indices,
    )

    source_features = calculate_source_features(
        df,
        allowed_history_indices=allowed_history_indices,
    )

    reference_features = calculate_reference_features(df)

    citation_features = calculate_historical_citation_features(df)

    feature_frames = [
        topic_dyn,
        author_dyn,
        text_features,
        graph_features,
        source_features,
        reference_features,
        citation_features,
    ]

    feature_df = pd.concat(
        feature_frames,
        axis=1,
    )

    for column in feature_df.columns:
        if feature_df[column].dtype != np.int8:
            feature_df[column] = (
                pd.to_numeric(
                    feature_df[column],
                    errors="coerce",
                )
                .fillna(0)
                .astype(np.float32)
            )

    out_df = pd.concat(
        [
            df[
                [
                    "doc_id",
                    "pub_year",
                    "split",
                    "source_tier",
                    "commercial_maturity_index",
                    "has_ref_data",
                ]
            ].copy(),
            feature_df,
            df[["target_emergence"]].copy(),
        ],
        axis=1,
    )

    logger.info("Running sanity checks...")

    numeric_columns = [
        c
        for c in out_df.columns
        if c not in {"doc_id", "source_tier", "split"}
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            out_df[column],
            errors="coerce",
        )

        if not np.isfinite(values.to_numpy()).all():
            logger.warning(
                "%s contains non-finite values.",
                column,
            )

    inspect_columns = [
        "novelty_raw",
        "cluster_density",
        "nearest_topic_similarity",
        "cross_topic_gap",
        "ppmi_domain_score",
        "topic_domain_rarity",
        "topic_publication_growth",
        "author_growth_rate",
        "source_novelty",
        "reference_count",
    ]

    for column in inspect_columns:
        logger.info(
            "%s: nunique=%d, mean=%.5f, std=%.5f",
            column,
            out_df[column].nunique(),
            out_df[column].mean(),
            out_df[column].std(),
        )

    if out_df["has_ref_data"].nunique() == 1:
        logger.warning(
            "has_ref_data is constant and should probably be removed "
            "from the ML feature matrix."
        )

    logger.info(
        "Positive target rate: %.3f%%",
        100.0 * out_df["target_emergence"].mean(),
    )

    logger.info(
        "Final dataset: %d rows x %d columns.",
        out_df.shape[0],
        out_df.shape[1],
    )

    logger.info("Saving to %s", output_parquet_path)

    out_df.to_parquet(
        output_parquet_path,
        index=False,
    )

    logger.info("DONE.")


if __name__ == "__main__":
    build_real_features(
        INPUT_PATH,
        OUTPUT_PATH,
    )
