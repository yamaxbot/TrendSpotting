import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.stats import ks_2samp, spearmanr
from sklearn.feature_selection import mutual_info_classif

DATASET_PATH = "download_dataset/new_dataset/new_data/final_openalex_dataset.parquet"


def run_full_eda(parquet_path: str):
    start_time = time.time()
    print("=" * 70)
    print("1. ЗАГРУЗКА И СТРУКТУРА ДАТАСЕТА (через Polars)")
    print("=" * 70)

    # Загружаем через Polars для высокой скорости на 1M строк
    pldf = pl.read_parquet(parquet_path)
    df = pldf.to_pandas()  # Для специфических sklearn/scipy метрик
    print(df.columns.tolist())
    num_rows, num_cols = pldf.shape
    print(f"Размер датасета: {num_rows:,} строк x {num_cols} колонок")
    print(
        f"Использование памяти в RAM: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB\n"
    )

    # -------------------------------------------------------------
    print("=" * 70)
    print("2. АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (target_emergence)")
    print("=" * 70)

    target_counts = pldf["target_emergence"].value_counts().to_dicts()
    target_dist = {item["target_emergence"]: item["count"] for item in target_counts}

    labeled_rows = pldf.filter(pl.col("target_emergence").is_not_null())
    if labeled_rows.height == 0:
        print(
            "ОШИБКА: target_emergence пуст во всех строках. "
            "EDA остановлена: сначала нужен отдельный этап разметки 0/1."
        )
        return

    labeled_df = df[df["target_emergence"].notna()].copy()
    labeled_count = len(labeled_df)

    pos_class = target_dist.get(1, 0)
    neg_class = target_dist.get(0, 0)
    pos_ratio = (pos_class / labeled_count) * 100

    print(f"Размечено строк: {labeled_count:,} из {num_rows:,}")
    print(f"Класс 0 (Не тренд): {neg_class:,} ({100 - pos_ratio:.2f}%)")
    print(f"Класс 1 (Восходящий тренд): {pos_class:,} ({pos_ratio:.2f}%)")
    print(f"Дисбаланс классов (Imbalance Ratio): 1:{neg_class / max(1, pos_class):.1f}")

    if pos_ratio < 5.0:
        print(
            "[WARNING] Выраженный дисбаланс классов. Следует рассмотреть scale_pos_weight и использовать PR-AUC."
        )
    print()

    # -------------------------------------------------------------
    print("=" * 70)
    print("3. ПРОПУСКИ И ТИПЫ ДАННЫХ")
    print("=" * 70)

    null_summary = (
        pldf.null_count()
        .melt(variable_name="column", value_name="null_count")
        .with_columns(
            (pl.col("null_count") / num_rows * 100).round(2).alias("null_pct")
        )
    )
    print(null_summary)
    print()

    # -------------------------------------------------------------
    print("=" * 70)
    print("4. ОПИСАТЕЛЬНАЯ СТАТИСТИКА ПРИЗНАКОВ (Descriptive Statistics)")
    print("=" * 70)

    numeric_cols = [
        col
        for col in df.columns
        if col not in ["doc_id", "source_tier", "target_emergence"]
    ]

    stats_df = []
    for col in numeric_cols:
        series = df[col].dropna()
        q25, q50, q75 = np.percentile(series, [25, 50, 75])
        iqr = q75 - q25
        # Выбросы по правилу IQR
        outliers = ((series < (q25 - 1.5 * iqr)) | (series > (q75 + 1.5 * iqr))).sum()

        stats_df.append(
            {
                "feature": col,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "median": q50,
                "max": series.max(),
                "skewness": series.skew(),
                "outliers_count": outliers,
                "outliers_pct": round(outliers / len(series) * 100, 2),
            }
        )

    stats_summary = pd.DataFrame(stats_df)
    print(stats_summary.to_string(index=False))
    print()

    # -------------------------------------------------------------
    print("=" * 70)
    print("5. АНАЛИЗ СЕПАРЕБЕЛЬНОСТИ (KS-test & Mutual Information)")
    print("=" * 70)
    print("Оценивается разделимость классов по отдельным признакам.")

    # Сэмплируем 100k для быстрого расчета Mutual Info на огромном объеме
    sample_df = labeled_df.sample(n=min(100000, labeled_count), random_state=42)
    sample_clean = sample_df[numeric_cols].fillna(-999)

    mi_scores = mutual_info_classif(
        sample_clean, sample_df["target_emergence"], random_state=42
    )

    ks_results = []
    for col in numeric_cols:
        cls0 = labeled_df[labeled_df["target_emergence"] == 0][col].dropna()
        cls1 = labeled_df[labeled_df["target_emergence"] == 1][col].dropna()

        # Тест Колмогорова-Смирнова на различие распределений
        ks_stat, p_val = ks_2samp(cls0, cls1)

        ks_results.append(
            {"feature": col, "ks_statistic": ks_stat, "p_value": p_val}
        )

    sep_df = pd.DataFrame(ks_results)
    sep_df["mutual_info"] = mi_scores
    sep_df = sep_df.sort_values(by="mutual_info", ascending=False)

    print(sep_df.to_string(index=False))
    print()

    # -------------------------------------------------------------
    print("=" * 70)
    print("6. ПРОВЕРКА МУЛЬТИКОЛЛИНЕАРНОСТИ (Спирмен > 0.8)")
    print("=" * 70)

    corr_matrix, _ = spearmanr(sample_clean)
    corr_df = pd.DataFrame(
        corr_matrix, index=numeric_cols, columns=numeric_cols
    )

    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            val = corr_df.iloc[i, j]
            if abs(val) > 0.7:
                high_corr_pairs.append(
                    (numeric_cols[i], numeric_cols[j], round(val, 3))
                )

    if high_corr_pairs:
        print("[WARNING] Обнаружены пары признаков с высокой корреляцией:")
        for f1, f2, c in high_corr_pairs:
            print(f"  {f1} <-> {f2} | Spearman rho = {c}")
    else:
        print("[INFO] Высокой корреляции между непрерывными признаками не обнаружено.")

    # -------------------------------------------------------------
    print("\n=" * 70)
    print("7. РАЗБИЕНИЕ ПО SOURCE TIER")
    print("=" * 70)

    tier_breakdown = (
        pldf.group_by("source_tier")
        .agg(
            [
                pl.len().alias("total_docs"),
                pl.col("target_emergence").sum().alias("emergent_docs"),
                pl.col("commercial_maturity_index")
                .null_count()
                .alias("missing_comm_index"),
            ]
        )
        .with_columns(
            (pl.col("emergent_docs") / pl.col("total_docs") * 100)
            .round(2)
            .alias("emergence_rate_pct")
        )
    )
    print(tier_breakdown)

    print(f"\n[DONE] Анализ завершен. Время выполнения: {time.time() - start_time:.2f} сек.")


if __name__ == "__main__":
    run_full_eda(DATASET_PATH)