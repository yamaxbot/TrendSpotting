from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import HashingVectorizer


# ============================================================
# Configuration
# ============================================================

PAST_WINDOW_YEARS = 3
FUTURE_WINDOW_YEARS = 3

# Historical emergence label:
# compare Y-3..Y-1 with Y+1..Y+3.
ESI_QUANTILE = 0.90
MIN_FUTURE_VOLUME = 3
SMOOTHING = 5.0

# Text representation.
TEXT_N_FEATURES = 2000

# Robust clipping for a few heavy-tailed dynamic features.
LOG_GROWTH_CLIP = 5.0

INPUT_PATH = "data/openalex_corpus_1m.parquet"
OUTPUT_PATH = "data/train_data/processed_ml_dataset.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Parsing helpers
# ============================================================

def parse_json_list(value: Any) -> list:
    """Safely parse a JSON/list-like OpenAlex field."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return []

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    return []


def extract_author_ids(value: Any) -> list[str]:
    """Extract OpenAlex author IDs from authorship data."""
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
    """Parse OpenAlex counts_by_year into dictionaries."""
    value = parse_json_list(value)
    return [item for item in value if isinstance(item, dict)]


def safe_len_json_list(value: Any) -> int:
    """Number of items in a JSON/list-like OpenAlex field."""
    return len(parse_json_list(value))


# ============================================================
# 1. Historical target
# ============================================================

def calculate_esi_target(df: pd.DataFrame) -> pd.Series:
    """
    Historical emergence proxy.

    Every paper in the same (topic, year) receives the same label.

    A paper from year Y is positive when its topic has unusually high
    publication growth during Y+1..Y+3 relative to Y-3..Y-1,
    with a minimum future publication volume.

    Future information is used ONLY for the label.
    """

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

    # Compute one growth value per (topic, year), then map it back to papers.
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


# ============================================================
# 2. Topic publication dynamics
# ============================================================

def calculate_topic_dynamics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Historical topic-level publication dynamics.

    These are deliberately separate from text geometry:
    they describe how much the research community is publishing
    around a topic and whether that activity is accelerating.
    """

    logger.info("Calculating topic publication dynamics...")

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()

    work = pd.DataFrame({"topic": topics, "year": years})
    counts = (
        work.dropna()
        .groupby(["topic", "year"])
        .size()
        .to_dict()
    )

    n = len(df)
    volume = np.zeros(n, dtype=np.float32)
    growth = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)
    age = np.zeros(n, dtype=np.float32)

    first_year: dict[Any, int] = {}

    for (topic, year), count in counts.items():
        year = int(year)
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

        growth[i] = np.float32(np.clip(g1, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP))
        acceleration[i] = np.float32(
            np.clip(g1 - g2, -LOG_GROWTH_CLIP, LOG_GROWTH_CLIP)
        )

        age[i] = np.float32(max(0, year - first_year.get(topic, year)))

    return pd.DataFrame(
        {
            "topic_historical_volume": volume,
            "topic_publication_growth": growth,
            "topic_growth_acceleration": acceleration,
            "topic_age_years": age,
        },
        index=df.index,
    )


# ============================================================
# 3. Author dynamics
# ============================================================

def calculate_author_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical author-community dynamics.

    All values use only years before the paper's publication year.
    """

    logger.info("Calculating author dynamics...")

    authors_column = (
        "authors_ids"
        if "authors_ids" in df.columns
        else "authorships_json"
    )

    topic_year_authors: defaultdict[tuple, set] = defaultdict(set)

    for topic, year, authors in zip(
        df["primary_topic_id"],
        df["pub_year"],
        df[authors_column],
    ):
        if pd.isna(topic) or pd.isna(year):
            continue

        year = int(year)

        for author_id in extract_author_ids(authors):
            topic_year_authors[(topic, year)].add(author_id)

    author_counts = {
        key: len(value)
        for key, value in topic_year_authors.items()
    }

    n = len(df)
    growth = np.zeros(n, dtype=np.float32)
    acceleration = np.zeros(n, dtype=np.float32)
    volume = np.zeros(n, dtype=np.float32)

    for i, (topic, year) in enumerate(
        zip(df["primary_topic_id"], df["pub_year"])
    ):
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


# ============================================================
# 4. Temporal text geometry
# ============================================================

def calculate_temporal_text_features(
    df: pd.DataFrame,
    text_matrix: scipy.sparse.csr_matrix,
) -> pd.DataFrame:
    """
    Multi-view historical text geometry.

    For a paper from year Y, only papers from years < Y are used.

    Returned views:
      - novelty_raw:
          1 - cosine similarity to historical topic centroid.
      - topic_centroid_similarity:
          same similarity, exposed explicitly for interpretability.
      - cluster_density:
          norm of historical topic centroid. This is a concentration
          proxy, not simply "number of papers".
      - topic_local_volume:
          number of historical papers in the topic.
      - nearest_topic_similarity:
          similarity to the closest *other* historical topic centroid.
      - cross_topic_gap:
          difference between own-topic similarity and nearest-other-topic
          similarity.

    The last two features are intended to capture whether a paper sits
    unusually close to another scientific territory, rather than merely
    being far from its own center.
    """

    logger.info("Calculating temporal text geometry...")

    n = len(df)

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(
        df["pub_year"],
        errors="coerce",
    ).to_numpy()

    novelty = np.ones(n, dtype=np.float32)
    own_similarity = np.zeros(n, dtype=np.float32)
    density = np.zeros(n, dtype=np.float32)
    local_volume = np.zeros(n, dtype=np.float32)
    nearest_similarity = np.zeros(n, dtype=np.float32)
    cross_gap = np.zeros(n, dtype=np.float32)

    # Historical topic statistics
    topic_sum: dict[Any, np.ndarray] = {}
    topic_count: defaultdict[Any, int] = defaultdict(int)

    # Only valid years
    valid_years = np.sort(
        np.unique(years[~np.isnan(years)]).astype(np.int32)
    )


    for year in valid_years:
        current = np.flatnonzero(years == year)

        if len(current) == 0:
            continue

        # ------------------------------------------------------------
        # Group current papers by topic
        # ------------------------------------------------------------
        topic_to_indices: defaultdict[Any, list[int]] = defaultdict(list)

        for idx in current:
            topic = topics[idx]

            if pd.notna(topic):
                topic_to_indices[topic].append(idx)

        if not topic_to_indices:
            continue

        # ------------------------------------------------------------
        # Build historical centroid matrix ONCE for this year
        # ------------------------------------------------------------
        historical_topics = []
        centroid_vectors = []
        centroid_norms = []

        for topic, count in topic_count.items():
            if count <= 0:
                continue

            raw = topic_sum[topic] / count
            norm = np.linalg.norm(raw)

            if norm <= 1e-12:
                continue

            historical_topics.append(topic)
            centroid_vectors.append(raw / norm)
            centroid_norms.append(norm)

        if centroid_vectors:
            centroid_matrix = np.asarray(
                centroid_vectors,
                dtype=np.float32,
            )

            # O(1) topic -> centroid column lookup
            centroid_pos = {
                topic: i
                for i, topic in enumerate(historical_topics)
            }
        else:
            centroid_matrix = None
            centroid_pos = {}

        # ------------------------------------------------------------
        # Own-topic geometry
        # ------------------------------------------------------------
        for topic, indices in topic_to_indices.items():
            count = topic_count[topic]

            if count <= 0:
                # No historical support for this topic
                continue

            raw = topic_sum[topic] / count
            norm = np.linalg.norm(raw)

            local_volume[indices] = np.float32(count)
            density[indices] = np.float32(norm)

            if norm <= 1e-12:
                continue

            centroid_unit = raw / norm

            rows = text_matrix[indices]

            sims = np.asarray(
                rows @ centroid_unit
            ).reshape(-1)

            sims = np.clip(
                sims,
                -1.0,
                1.0,
            ).astype(np.float32)

            own_similarity[indices] = sims
            novelty[indices] = np.clip(
                1.0 - sims,
                0.0,
                2.0,
            )

        # ------------------------------------------------------------
        # Cross-topic geometry
        #
        # One matrix multiplication for the WHOLE year instead
        # of one multiplication per topic.
        # ------------------------------------------------------------
        if centroid_matrix is not None and len(centroid_matrix) >= 2:

            rows = text_matrix[current]

            # shape:
            #   n_current × n_historical_topics
            similarities = np.asarray(
                rows @ centroid_matrix.T
            )

            # Remove each paper's own topic from nearest-topic search.
            #
            # This is much faster than:
            # centroid_topics.index(topic)
            # for every topic.
            own_positions = np.full(
                len(current),
                -1,
                dtype=np.int32,
            )

            for i, idx in enumerate(current):
                topic = topics[idx]
                own_positions[i] = centroid_pos.get(topic, -1)

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

        # ------------------------------------------------------------
        # Update history AFTER all feature calculations
        # ------------------------------------------------------------
        for topic, indices in topic_to_indices.items():

            rows = text_matrix[indices]

            # One dense vector per topic, not per paper
            summed = np.asarray(rows.sum(axis=0)).ravel().astype(np.float32)

            if topic in topic_sum:
                topic_sum[topic] += summed
            else:
                topic_sum[topic] = summed.copy()

            topic_count[topic] += len(indices)


    # ================================================================
    # Outlier detection
    # ================================================================
    outlier = np.zeros(
        n,
        dtype=np.int8,
    )

    # Keep exactly the original semantics:
    # threshold for year Y = 95th percentile of ALL papers before Y.
    #
    # Instead of constructing `years < year` every time, use a
    # cumulative list of previous-year indices.
    previous_indices = []

    for year in valid_years:
        current = np.flatnonzero(years == year)

        if previous_indices:
            previous = np.concatenate(previous_indices)

            threshold = np.quantile(
                novelty[previous],
                0.95,
            )

            outlier[current] = (
                novelty[current] >= threshold
            ).astype(np.int8)

        previous_indices.append(current)


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


# ============================================================
# 5. Temporal topic-domain structure
# ============================================================

def calculate_temporal_graph_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Historical topic-domain association features.

    NOTE:
    If the source has only one domain per paper, this is NOT a full
    domain-domain interdisciplinary graph. We therefore expose several
    different statistics rather than pretending one PPMI value is a
    complete interdisciplinary representation.

    Features:
      - ppmi_domain_score:
          historical association strength topic <-> domain.
      - topic_domain_rarity:
          inverse historical frequency of this exact pair.
      - domain_history_count:
          historical activity of the domain.
      - topic_domain_first_seen_age:
          how long this topic-domain connection has existed.
    """

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


# ============================================================
# 6. Venue / source structure
# ============================================================

def calculate_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical publication-source structure.

    These features give the model a view different from text and
    citations: whether a topic is concentrated in a few venues or
    spreads across many sources.
    """

    logger.info("Calculating historical source structure...")

    topics = df["primary_topic_id"].to_numpy()
    years = pd.to_numeric(df["pub_year"], errors="coerce").to_numpy()
    sources = df["source_id"].to_numpy()

    n = len(df)

    source_counts: defaultdict[tuple, int] = defaultdict(int)
    topic_year_sources: defaultdict[tuple, set] = defaultdict(set)

    valid_years = sorted(
        int(y) for y in pd.Series(years).dropna().unique()
    )

    source_history_count = np.zeros(n, dtype=np.float32)
    source_novelty = np.zeros(n, dtype=np.float32)
    topic_source_diversity = np.zeros(n, dtype=np.float32)

    for year in valid_years:
        current = np.flatnonzero(years == year)

        # Historical source counts for each topic.
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
                1.0 / np.log1p(source_counts.get(key, 0) + 1.0)
            )

            historical_sources = topic_year_sources.get(
                topic,
                set(),
            )

            topic_source_diversity[idx] = np.float32(
                len(historical_sources)
            )

            topic_to_sources[topic].add(source)

        # Update historical source state.
        for topic, source_set in topic_to_sources.items():
            topic_year_sources[topic].update(source_set)

        for idx in current:
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


# ============================================================
# 7. Bibliographic support
# ============================================================

def calculate_reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bibliographic-support features available at publication time.

    These are not future citations. They describe the paper's own
    reference structure.
    """

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


# ============================================================
# 8. Historical citation features
# ============================================================

def calculate_historical_citation_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Historical citation information only.

    For a paper published in Y:
      velocity = cumulative cited_by counts through Y.
      acceleration = log2((citations in Y + 1)/(citations in Y-1 + 1)).

    No Y+1 or later citation data is used.
    """

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


# ============================================================
# 9. Final builder
# ============================================================

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

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    df["target_emergence"] = calculate_esi_target(df)

    # --------------------------------------------------------
    # 1. Dynamics
    # --------------------------------------------------------
    topic_dyn = calculate_topic_dynamics(df)
    author_dyn = calculate_author_features(df)

    # --------------------------------------------------------
    # 2. Text / geometry
    # --------------------------------------------------------
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
    )

    # Free the large sparse matrix before later feature calculations.
    del text_matrix

    # --------------------------------------------------------
    # 3. Graph / interdisciplinarity proxy
    # --------------------------------------------------------
    graph_features = calculate_temporal_graph_features(df)

    # --------------------------------------------------------
    # 4. Source / venue structure
    # --------------------------------------------------------
    source_features = calculate_source_features(df)

    # --------------------------------------------------------
    # 5. Bibliographic support
    # --------------------------------------------------------
    reference_features = calculate_reference_features(df)

    # --------------------------------------------------------
    # 6. Historical citations
    # --------------------------------------------------------
    citation_features = calculate_historical_citation_features(df)

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------
    feature_frames = [
        topic_dyn,
        author_dyn,
        text_features,
        graph_features,
        source_features,
        reference_features,
        citation_features,
    ]

    feature_df = pd.concat(feature_frames, axis=1)

    for column in feature_df.columns:
        if feature_df[column].dtype != np.int8:
            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce",
            ).fillna(0).astype(np.float32)

    out_df = pd.concat(
        [
            df[
                [
                    "doc_id",
                    "pub_year",
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

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------
    logger.info("Running sanity checks...")

    numeric_columns = [
        c for c in out_df.columns
        if c not in {"doc_id", "source_tier"}
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

    # Features that should NOT be constant.
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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
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
