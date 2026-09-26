import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
    CatBoostRanker,
    Pool,
    CatBoostError,
)
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


SEED = 42
DEFAULT_MODE = "ranking"

DEFAULT_MAX_PAIRS = 1000
DEFAULT_RANK_TOP = 100


def safe_metric(metric_fn, *args, default=np.nan):
    try:
        value = metric_fn(*args)
    except (ValueError, ZeroDivisionError):
        return default
    return float(value)


def split_temporal(df, X, y_target):
    """Строгий временной split: train 2020-21 | valid 2022 | test 2023.
    2017-2019 / 2024-2026 — только support, в обучение не входят.
    """
    train = (df["pub_year"] >= 2020) & (df["pub_year"] <= 2021)
    valid = df["pub_year"] == 2022
    test = df["pub_year"] == 2023
    if (df.loc[train, "pub_year"] < 2020).any():
        raise ValueError("Refusing to train on past_support years (<2020)")

    for name, mask in [
        ("train", train),
        ("valid", valid),
        ("test", test),
    ]:
        if not mask.any():
            raise ValueError(f"{name} split is empty")

        y_bin = df.loc[mask, "target_emergence"]

        print(
            f"{name:5s}: {mask.sum():,} rows | "
            f"{df.loc[mask, 'pub_year'].min()}-"
            f"{df.loc[mask, 'pub_year'].max()} | "
            f"positive={y_bin.sum():,} "
            f"({100 * y_bin.mean():.2f}%)"
        )

    return (
        X.loc[train],
        X.loc[valid],
        X.loc[test],
        y_target[train],
        y_target[valid],
        y_target[test],
        df.loc[valid, "target_emergence"].values,
        df.loc[test, "target_emergence"].values,
        df.loc[test].copy(),
    )


def threshold_table(y, scores, n_thresholds=1000):
    """
    Строит таблицу threshold -> precision/recall/F1/MCC.

    В отличие от наивного варианта здесь НЕ создаётся матрица
    [n_thresholds x n_samples], поэтому поиск порога не должен
    внезапно съесть гигабайты RAM.
    """
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    if y.size == 0 or np.unique(y).size < 2:
        return pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "mcc"]
        )

    precision, recall, thresholds = precision_recall_curve(y, scores)

    precision = precision[:-1]
    recall = recall[:-1]

    if thresholds.size == 0:
        return pd.DataFrame(
            columns=["threshold", "precision", "recall", "f1", "mcc"]
        )

    if thresholds.size > n_thresholds:
        idx = np.linspace(
            0,
            thresholds.size - 1,
            n_thresholds,
            dtype=np.int64,
        )
        thresholds = thresholds[idx]
        precision = precision[idx]
        recall = recall[idx]

    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )

    positives = float(y.sum())
    negatives = float(y.size - y.sum())

    tp = recall * positives
    fn = positives - tp

    fp = np.divide(
        tp * (1.0 - precision),
        precision,
        out=np.zeros_like(tp),
        where=precision > 0,
    )
    tn = negatives - fp

    mcc_denom = np.sqrt(
        np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.0)
    )

    mcc = np.divide(
        tp * tn - fp * fn,
        mcc_denom,
        out=np.zeros_like(mcc_denom),
        where=mcc_denom > 0,
    )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
        }
    )


def evaluate(name, y, scores, threshold, out_dir):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pred = (scores >= threshold).astype(np.int8)

    cm = confusion_matrix(y, pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "roc_auc": safe_metric(roc_auc_score, y, scores),
        "pr_auc": safe_metric(average_precision_score, y, scores),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y, pred, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y, pred)
        ) if np.unique(y).size == 2 and np.unique(pred).size == 2 else np.nan,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    print(f"\n{name}")
    print("-" * 50)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:12s}: {value:.6f}")
        else:
            print(f"{key:12s}: {value}")

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y,
            pred,
            digits=4,
            zero_division=0,
        )
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_title(
        f"{name} confusion matrix "
        f"(threshold={threshold:.4f})"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{name.lower()}_confusion_matrix.png",
        dpi=180,
    )
    plt.close(fig)

    return metrics


def prepare_features(df):
    """
    Формирует признаки и исключает потенциально проблемные/мусорные
    колонки из текущего эксперимента.
    """
    year_totals = df.groupby("pub_year")["doc_id"].transform("count")

    if "topic_historical_volume" in df.columns:
        df["topic_historical_share"] = (
            df["topic_historical_volume"] / (year_totals + 1e-5)
        )

    if "topic_local_volume" in df.columns:
        df["topic_local_share"] = (
            df["topic_local_volume"] / (year_totals + 1e-5)
        )

    if "historical_author_count" in df.columns:
        df["historical_author_share"] = (
            df["historical_author_count"] / (year_totals + 1e-5)
        )

    junk_features = {
        "has_ref_data",
        "has_reference_list",
        "is_outlier_cluster",
        "citation_velocity",
        "topic_centroid_similarity",
        "commercial_maturity_index",
        "novelty_raw",
        "cross_topic_gap",
        "source_novelty",
        "citation_acceleration",
        "source_tier",
        "topic_historical_volume",
        "topic_local_volume",
        "historical_author_count",
    }

    excluded = {
        "target_emergence",
        "pub_year",
        "doc_id",
    }.union(junk_features)

    features = []

    for column in df.columns:
        if column in excluded:
            continue

        if (
            pd.api.types.is_numeric_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_object_dtype(df[column])
            or isinstance(
                df[column].dtype,
                pd.CategoricalDtype,
            )
        ):
            features.append(column)

    X = df[features].copy()
    cat_features = []

    for column in features:
        if (
            pd.api.types.is_numeric_dtype(X[column])
            or pd.api.types.is_bool_dtype(X[column])
        ):
            X[column] = pd.to_numeric(
                X[column],
                errors="coerce",
            ).astype(np.float32)
        else:
            X[column] = (
                X[column]
                .fillna("missing")
                .astype(str)
            )
            cat_features.append(column)

    return X, features, cat_features


def build_ranking_group_ids(df):
    """Build ranking groups from the information actually present in the dataset.

    The current parquet contains no topic identifier. Therefore the only
    defensible ranking group available here is publication year. Topic-derived
    numeric features are model inputs, not stable group identifiers.
    """
    if "pub_year" not in df.columns:
        raise ValueError("Ranking mode requires 'pub_year'.")

    return df["pub_year"].astype(str).to_numpy()


def summarize_ranking_groups(df_split, group_ids, split_name):
    tmp = pd.DataFrame({
        "group": group_ids,
        "target": df_split["target_emergence"].to_numpy(),
    })
    stats = tmp.groupby("group", sort=False).agg(
        size=("target", "size"),
        positives=("target", "sum"),
    )

    n_groups = len(stats)
    no_pos = int((stats["positives"] == 0).sum())
    one_pos = int((stats["positives"] == 1).sum())
    useful = int((stats["positives"] >= 2).sum())

    print(f"\\nRanking groups — {split_name}")
    print(f"groups: {n_groups:,}")
    print(
        f"size: median={stats['size'].median():.0f} | "
        f"p90={stats['size'].quantile(.90):.0f} | "
        f"max={stats['size'].max():,}"
    )
    print(
        f"positives/group: median={stats['positives'].median():.0f} | "
        f"p90={stats['positives'].quantile(.90):.0f} | "
        f"max={stats['positives'].max():,}"
    )
    print(f"groups with 0 positives: {no_pos:,}")
    print(f"groups with 1 positive : {one_pos:,}")
    print(f"groups with >=2 positives: {useful:,}")

    if useful == 0:
        raise ValueError(
            f"{split_name}: no ranking group has >=2 positives."
        )

    return stats


def make_ranking_pools(
    df,
    Xtr, Xva, Xte,
    ytr, yva, yte,
    yva_bin, yte_bin,
    test_df,
    cat_features,
):
    """Rank papers within each publication year."""
    group_all = build_ranking_group_ids(df)

    group_id_train = group_all[df.index.isin(Xtr.index)]
    group_id_valid = group_all[df.index.isin(Xva.index)]
    group_id_test = group_all[df.index.isin(Xte.index)]

    idx_tr = np.argsort(group_id_train, kind="stable")
    idx_va = np.argsort(group_id_valid, kind="stable")
    idx_te = np.argsort(group_id_test, kind="stable")

    Xtr_sorted = Xtr.iloc[idx_tr]
    ytr_sorted = np.asarray(ytr)[idx_tr]
    group_tr_sorted = group_id_train[idx_tr]

    Xva_sorted = Xva.iloc[idx_va]
    yva_sorted = np.asarray(yva)[idx_va]
    group_va_sorted = group_id_valid[idx_va]

    Xte_sorted = Xte.iloc[idx_te]
    yte_sorted = np.asarray(yte)[idx_te]
    group_te_sorted = group_id_test[idx_te]

    yva_bin_sorted = np.asarray(yva_bin)[idx_va]
    yte_bin_sorted = np.asarray(yte_bin)[idx_te]
    test_df_sorted = test_df.iloc[idx_te].copy()

    summarize_ranking_groups(df.loc[Xtr.index], group_id_train, "train")
    summarize_ranking_groups(df.loc[Xva.index], group_id_valid, "valid")
    summarize_ranking_groups(df.loc[Xte.index], group_id_test, "test")

    print(
        "\\nRanking definition: pub_year",
        flush=True,
    )

    train_pool = Pool(
        Xtr_sorted, ytr_sorted,
        group_id=group_tr_sorted,
        cat_features=cat_features,
    )
    valid_pool = Pool(
        Xva_sorted, yva_sorted,
        group_id=group_va_sorted,
        cat_features=cat_features,
    )
    test_pool = Pool(
        Xte_sorted, yte_sorted,
        group_id=group_te_sorted,
        cat_features=cat_features,
    )

    return (
        train_pool, valid_pool, test_pool,
        Xtr_sorted, Xva_sorted, Xte_sorted,
        ytr_sorted, yva_sorted, yte_sorted,
        yva_bin_sorted, yte_bin_sorted,
        test_df_sorted, group_te_sorted, group_va_sorted,
    )


def ranking_topk_metrics(scores, y, group_ids, top):
    """Compute mean Precision@K and Recall@K over ranking groups."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(group_ids)

    rows = []
    for group in pd.unique(groups):
        idx = np.flatnonzero(groups == group)
        if idx.size == 0:
            continue

        order = idx[np.argsort(scores[idx])[::-1]]
        k = min(top, len(order))
        top_idx = order[:k]
        positives_top = int(y[top_idx].sum())
        positives_total = int(y[idx].sum())

        rows.append({
            "group": group,
            "k": k,
            "precision_at_k": positives_top / k if k else 0.0,
            "recall_at_k": (
                positives_top / positives_total
                if positives_total > 0 else np.nan
            ),
            "positives_in_top_k": positives_top,
            "group_positives": positives_total,
        })

    group_df = pd.DataFrame(rows)
    metrics = {
        "top": int(top),
        "groups": int(len(group_df)),
        "mean_precision_at_k": float(group_df["precision_at_k"].mean()) if len(group_df) else np.nan,
        "mean_recall_at_k": float(group_df["recall_at_k"].dropna().mean()) if len(group_df) and group_df["recall_at_k"].notna().any() else np.nan,
        "global_roc_auc": safe_metric(roc_auc_score, y, scores),
        "global_pr_auc": safe_metric(average_precision_score, y, scores),
    }
    return metrics, group_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="./data/train_data/processed_ml_dataset.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/catboost",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["binary", "regression", "ranking"],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--shap-sample",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_MAX_PAIRS,
        help=(
            "Maximum number of automatically generated pairs "
            "per ranking group. Used only in ranking mode."
        ),
    )
    parser.add_argument(
        "--rank-top",
        type=int,
        default=DEFAULT_RANK_TOP,
        help="K for ranking diagnostics such as Precision@K.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        default=1000,
        help="Maximum number of thresholds evaluated on validation.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")

    mode = args.mode

    out = Path(args.output_dir) / mode
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"1. LOAD (MODE: {mode.upper()})")
    print("=" * 70)

    df = pd.read_parquet(args.dataset)

    df["pub_year"] = pd.to_numeric(
        df["pub_year"],
        errors="coerce",
    )

    df["target_emergence"] = pd.to_numeric(
        df["target_emergence"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["pub_year", "target_emergence"]
    ).copy()

    df["pub_year"] = df["pub_year"].astype(np.int32)
    df["target_emergence"] = (
        df["target_emergence"]
        .astype(np.int8)
    )

    if mode == "binary":
        y_all = df["target_emergence"].to_numpy()

        loss_func = "Logloss"
        eval_metric = "PRAUC"
        model_class = CatBoostClassifier

    elif mode == "regression":
        if "topic_publication_growth" in df.columns:
            growth_signal = np.maximum(
                0,
                pd.to_numeric(
                    df["topic_publication_growth"],
                    errors="coerce",
                )
                .fillna(0)
                .to_numpy(),
            )
        else:
            growth_signal = np.zeros(
                len(df),
                dtype=np.float32,
            )

        y_all = (
            df["target_emergence"].to_numpy()
            * (1.0 + np.log1p(growth_signal))
        )

        loss_func = "RMSE"
        eval_metric = "RMSE"
        model_class = CatBoostRegressor

    else:
        y_all = df["target_emergence"].to_numpy()

        loss_func = (
            f"PairLogit:max_pairs={args.max_pairs}"
        )
        eval_metric = "PairLogit"
        model_class = CatBoostRanker
        pairlogit = f"PairLogit:max_pairs={args.max_pairs}"

    print("\n" + "=" * 70)
    print("2. FEATURES")
    print("=" * 70)

    X, features, cat_features = prepare_features(df)

    print(f"Dataset: {df.shape}")
    print(f"Features: {len(features)}")
    print(f"Categorical features: {len(cat_features)}")

    if cat_features:
        print("Categorical:")
        for column in cat_features:
            print(f"  - {column}")

    print("\n" + "=" * 70)
    print("3. TEMPORAL SPLIT")
    print("=" * 70)

    (
        Xtr,
        Xva,
        Xte,
        ytr,
        yva,
        yte,
        yva_bin,
        yte_bin,
        test_df,
    ) = split_temporal(
        df,
        X,
        y_all,
    )

    print("\n" + "=" * 70)
    print("4. BUILD POOLS")
    print("=" * 70)

    group_te_sorted = None
    group_va_sorted = None

    if mode == "ranking":
        (
            train_pool,
            valid_pool,
            test_pool,
            Xtr_used,
            Xva_used,
            Xte_used,
            ytr_used,
            yva_used,
            yte_used,
            yva_bin,
            yte_bin,
            test_df,
            group_te_sorted,
            group_va_sorted,
        ) = make_ranking_pools(
            df,
            Xtr,
            Xva,
            Xte,
            ytr,
            yva,
            yte,
            yva_bin,
            yte_bin,
            test_df,
            cat_features,
        )
    else:
        train_pool = Pool(
            Xtr,
            ytr,
            feature_names=features,
            cat_features=cat_features,
        )

        valid_pool = Pool(
            Xva,
            yva,
            feature_names=features,
            cat_features=cat_features,
        )

        test_pool = Pool(
            Xte,
            yte,
            feature_names=features,
            cat_features=cat_features,
        )

        Xtr_used = Xtr
        Xva_used = Xva
        Xte_used = Xte
        ytr_used = ytr
        yva_used = yva
        yte_used = yte

    print("\n" + "=" * 70)
    print(f"5. CATBOOST TRAINING ({mode.upper()})")
    print("=" * 70)

    model_params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": pairlogit,
        "eval_metric": pairlogit,
        "random_seed": SEED,
        "verbose": 10,
        "allow_writing_files": False,
        "thread_count": args.threads,
    }

    if mode == "ranking":
        model_params["custom_metric"] = [
            f"NDCG:top={args.rank_top}",
            "AUC:type=Ranking",
        ]
        model_params["od_type"] = "Iter"
        model_params["od_wait"] = 200

    if mode == "binary":
        model_params["auto_class_weights"] = "Balanced"

    elif mode == "ranking":
        model_params["max_ctr_complexity"] = 1

    print(f"loss_function = {loss_func}")
    print(f"eval_metric   = {eval_metric}")

    if mode == "ranking":
        print(f"max_pairs    = {args.max_pairs} per group")
        print("ranking group = pub_year")
        print(f"diagnostic top = {args.rank_top}")

    print("Инициализация модели...", flush=True)

    model = model_class(**model_params)

    print("Запуск fit()...", flush=True)

    try:
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
        )
    except CatBoostError as exc:
        print("\nCATBOOST ERROR:")
        print(exc)
        print(
            "\nЕсли ошибка снова связана с pair generation, "
            "уменьши --max-pairs, например до 10000."
        )
        raise

    model.save_model(
        out / "catboost_model.cbm"
    )

    print(
        f"Best iteration: "
        f"{model.get_best_iteration()}"
    )

    try:
        with open(out / "evals_result.json", "w", encoding="utf-8") as f:
            import json
            json.dump(model.get_evals_result(), f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Warning: could not save eval history: {exc}")

    print("\n" + "=" * 70)
    print("6. PREDICTIONS")
    print("=" * 70)

    if mode == "binary":
        p_valid = model.predict_proba(
            valid_pool
        )[:, 1]

        p_test = model.predict_proba(
            test_pool
        )[:, 1]

    else:
        p_valid = np.asarray(
            model.predict(valid_pool)
        ).reshape(-1)

        p_test = np.asarray(
            model.predict(test_pool)
        ).reshape(-1)

    if mode == "ranking":
        print("\n" + "=" * 70)
        print("7. RANKING DIAGNOSTICS")
        print("=" * 70)

        valid_rank, valid_group_metrics = ranking_topk_metrics(
            p_valid, yva_bin, group_va_sorted, args.rank_top
        )
        test_rank, test_group_metrics = ranking_topk_metrics(
            p_test, yte_bin, group_te_sorted, args.rank_top
        )

        print("\nValidation ranking:")
        for k, v in valid_rank.items():
            print(f"{k:24s}: {v}")
        print("\nTest ranking:")
        for k, v in test_rank.items():
            print(f"{k:24s}: {v}")

        valid_group_metrics.to_csv(
            out / "validation_group_ranking_metrics.csv", index=False
        )
        test_group_metrics.to_csv(
            out / "test_group_ranking_metrics.csv", index=False
        )

    print("\n" + "=" * 70)
    print("7. THRESHOLD SEARCH — VALIDATION ONLY")
    print("=" * 70)

    threshold_metrics = threshold_table(
        yva_bin,
        p_valid,
        n_thresholds=args.thresholds,
    )

    if threshold_metrics.empty:
        threshold = 0.5
        best = {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
        }
    else:
        best = threshold_metrics.loc[
            threshold_metrics["f1"].idxmax()
        ]

        threshold = float(
            best["threshold"]
        )

        threshold_metrics.to_csv(
            out / "validation_threshold_metrics.csv",
            index=False,
        )

        print(
            f"Best threshold={threshold:.6f} | "
            f"F1={best['f1']:.4f} | "
            f"Precision={best['precision']:.4f} | "
            f"Recall={best['recall']:.4f} | "
            f"MCC={best['mcc']:.4f}",
            flush=True,
        )

    valid_metrics = evaluate(
        "Validation",
        yva_bin,
        p_valid,
        threshold,
        out,
    )

    test_metrics = evaluate(
        "Test",
        yte_bin,
        p_test,
        threshold,
        out,
    )

    print("\n" + "=" * 70)
    print("8. FEATURE IMPORTANCE")
    print("=" * 70)

    try:
        importance = model.get_feature_importance(
            data=train_pool,
            type="FeatureImportance",
        )
    except Exception:
        if mode == "ranking":
            importance = model.get_feature_importance(
                data=train_pool,
                type="LossFunctionChange",
            )
        else:
            importance = model.get_feature_importance(
                type="PredictionValuesChange",
            )

    importance_df = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": importance,
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        importance_df.to_string(index=False),
        flush=True,
    )

    importance_df.to_csv(
        out / "feature_importance.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("9. SHAP")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    if len(Xte_used) > args.shap_sample:
        idx_shap = np.sort(
            rng.choice(
                len(Xte_used),
                args.shap_sample,
                replace=False,
            )
        )

        Xshap = Xte_used.iloc[idx_shap].copy()

        if mode == "ranking":
            g_shap = group_te_sorted[idx_shap]
        else:
            g_shap = None
    else:
        Xshap = Xte_used.copy()

        if mode == "ranking":
            g_shap = group_te_sorted
        else:
            g_shap = None

    if mode == "ranking":
        shap_pool = Pool(
            Xshap,
            group_id=g_shap,
            cat_features=cat_features,
        )
    else:
        shap_pool = Pool(
            Xshap,
            cat_features=cat_features,
        )

    shap_values = model.get_feature_importance(
        shap_pool,
        type="ShapValues",
    )

    shap_values = np.asarray(
        shap_values
    )

    shap_matrix = np.asarray(
        shap_values[:, :-1],
        dtype=np.float32,
    )

    mean_abs_shap = (
        np.abs(shap_matrix)
        .mean(axis=0)
    )

    shap_df = (
        pd.DataFrame(
            {
                "feature": features,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        shap_df.to_string(index=False),
        flush=True,
    )

    shap_df.to_csv(
        out / "shap_importance.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("10. SAVE TEST PREDICTIONS")
    print("=" * 70)

    predictions = (
        p_test >= threshold
    ).astype(np.int8)

    pred_df = pd.DataFrame(
        {
            "doc_id": (
                test_df["doc_id"].values
                if "doc_id" in test_df.columns
                else np.arange(len(test_df))
            ),
            "pub_year": test_df["pub_year"].values,
            "target_emergence_true": (
                test_df["target_emergence"].values
            ),
            "predicted_score": p_test,
            "predicted_label": predictions,
        }
    )

    if mode == "ranking":
        pred_df["ranking_group"] = group_te_sorted

    pred_df.to_parquet(
        out / "test_predictions.parquet",
        index=False,
    )

    summary = {
        "mode": mode,
        "loss_function": loss_func,
        "eval_metric": eval_metric,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "max_pairs": (
            args.max_pairs
            if mode == "ranking"
            else None
        ),
        "best_iteration": int(
            model.get_best_iteration()
        ),
        "threshold": float(threshold),
        "validation_pr_auc": float(
            valid_metrics["pr_auc"]
        ),
        "validation_f1": float(
            valid_metrics["f1"]
        ),
        "validation_mcc": float(
            valid_metrics["mcc"]
        ) if not np.isnan(valid_metrics["mcc"])
        else None,
        "test_pr_auc": float(
            test_metrics["pr_auc"]
        ),
        "test_f1": float(
            test_metrics["f1"]
        ),
        "test_mcc": float(
            test_metrics["mcc"]
        ) if not np.isnan(test_metrics["mcc"])
        else None,
    }

    if mode == "ranking":
        summary["ranking"] = {
            "group_definition": "pub_year",
            "max_pairs": int(args.max_pairs),
            "rank_top": int(args.rank_top),
            "validation": valid_rank,
            "test": test_rank,
        }

    pd.DataFrame([summary]).to_json(
        out / "experiment_summary.json",
        orient="records",
        indent=2,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Mode:          {mode.upper()}")
    print(f"Output:        {out.resolve()}")
    print(f"Best threshold:{threshold:.6f}")
    print(f"TEST PR-AUC:   {test_metrics['pr_auc']:.6f}")
    print(f"TEST F1:       {test_metrics['f1']:.6f}")
    print(f"TEST MCC:      {test_metrics['mcc']:.6f}")


if __name__ == "__main__":
    main()
