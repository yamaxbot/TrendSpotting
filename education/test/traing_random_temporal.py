import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


SEED = 42
ML_START_YEAR = 2020
ML_END_YEAR = 2023


def safe_metric(metric_fn, *args, default=np.nan):
    try:
        value = metric_fn(*args)
    except (ValueError, ZeroDivisionError):
        return default
    return float(value)


def threshold_table(y, proba, n_thresholds=1000):
    y = np.asarray(y, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)

    if y.size == 0 or np.unique(y).size < 2:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "mcc"])

    precision, recall, pr_thresholds = precision_recall_curve(y, proba)

    if len(pr_thresholds) > n_thresholds:
        idx = np.linspace(0, len(pr_thresholds) - 1, n_thresholds, dtype=int)
        precision = precision[idx]
        recall = recall[idx]
        thresholds = pr_thresholds[idx]
    else:
        thresholds = pr_thresholds
        precision, recall = precision[:-1], recall[:-1]

    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    preds = proba[:, None] >= thresholds[None, :]
    tp = (preds & (y[:, None] == 1)).sum(axis=0)
    tn = (~preds & (y[:, None] == 0)).sum(axis=0)
    fp = (preds & (y[:, None] == 0)).sum(axis=0)
    fn = (~preds & (y[:, None] == 1)).sum(axis=0)

    mcc_denom = np.sqrt(
        (tp + fp).astype(np.float64)
        * (tp + fn).astype(np.float64)
        * (tn + fp).astype(np.float64)
        * (tn + fn).astype(np.float64)
    )
    mcc = np.divide(
        (tp * tn) - (fp * fn),
        mcc_denom,
        out=np.zeros_like(mcc_denom, dtype=np.float64),
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


def evaluate(name, y, proba, threshold, out_dir):
    y = np.asarray(y, dtype=np.int8)
    proba = np.asarray(proba, dtype=np.float64)
    pred = (proba >= threshold).astype(np.int8)

    cm = confusion_matrix(y, pred, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "roc_auc": safe_metric(roc_auc_score, y, proba, default=np.nan),
        "pr_auc": safe_metric(average_precision_score, y, proba, default=np.nan),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred))
        if np.unique(y).size == 2 and np.unique(pred).size == 2
        else np.nan,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    print(f"\n{name}")
    print("-" * 55)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.6f}" if isinstance(v, float) else f"{k:12s}: {v}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(f"{name} confusion matrix (threshold={threshold:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    return metrics


def print_split_stats(name, df):
    y = df["target_emergence"]
    print(
        f"{name:8s}: {len(df):,} строк | "
        f"Года: {df['pub_year'].min()}-{df['pub_year'].max()} | "
        f"Позитивных: {int(y.sum()):,} ({100 * y.mean():.2f}%)"
    )


def split_random(df, X):
    ml = df["pub_year"].between(ML_START_YEAR, ML_END_YEAR)
    df_ml = df.loc[ml].copy()
    X_ml = X.loc[ml].copy()

    if df_ml.empty:
        raise ValueError("Нет ML-примеров за 2020-2023.")

    idx = np.arange(len(df_ml))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=0.30,
        random_state=SEED,
        stratify=df_ml["target_emergence"].to_numpy(),
    )

    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=df_ml.iloc[temp_idx]["target_emergence"].to_numpy(),
    )

    train_mask = np.zeros(len(df_ml), dtype=bool)
    valid_mask = np.zeros(len(df_ml), dtype=bool)
    test_mask = np.zeros(len(df_ml), dtype=bool)

    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    test_mask[test_idx] = True

    train_df = df_ml.iloc[train_idx].copy()
    valid_df = df_ml.iloc[valid_idx].copy()
    test_df = df_ml.iloc[test_idx].copy()

    print_split_stats("train", train_df)
    print_split_stats("valid", valid_df)
    print_split_stats("test", test_df)

    return (
        X_ml.iloc[train_idx],
        X_ml.iloc[valid_idx],
        X_ml.iloc[test_idx],
        train_df["target_emergence"],
        valid_df["target_emergence"],
        test_df["target_emergence"],
        test_df,
    )


def split_temporal(df, X):
    train_mask = df["pub_year"].between(2020, 2021)
    valid_mask = df["pub_year"].eq(2022)
    test_mask = df["pub_year"].eq(2023)

    for name, mask in [
        ("train", train_mask),
        ("valid", valid_mask),
        ("test", test_mask),
    ]:
        if not mask.any():
            raise ValueError(f"Temporal split '{name}' пуст.")

    print_split_stats("train", df.loc[train_mask])
    print_split_stats("valid", df.loc[valid_mask])
    print_split_stats("test", df.loc[test_mask])

    return (
        X.loc[train_mask],
        X.loc[valid_mask],
        X.loc[test_mask],
        df.loc[train_mask, "target_emergence"],
        df.loc[valid_mask, "target_emergence"],
        df.loc[test_mask, "target_emergence"],
        df.loc[test_mask].copy(),
    )


def train_one_experiment(
    name,
    Xtr,
    Xva,
    Xte,
    ytr,
    yva,
    yte,
    features,
    cat_features,
    out_dir,
    shap_sample,
):
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"{name.upper()} — CATBOOST TRAINING")
    print("=" * 70)

    train_pool = Pool(
        Xtr, ytr, feature_names=features, cat_features=cat_features
    )
    valid_pool = Pool(
        Xva, yva, feature_names=features, cat_features=cat_features
    )

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=15.0,
        rsm=0.8,
        subsample=0.8,
        random_seed=SEED,
        loss_function="Logloss",
        eval_metric="Logloss",
        od_type="Iter",
        od_wait=150,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(exp_dir / "catboost_model.cbm")

    p_valid = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    print("\n" + "=" * 70)
    print(f"{name.upper()} — THRESHOLD SEARCH (VALIDATION ONLY)")
    print("=" * 70)

    tt = threshold_table(yva, p_valid)

    if tt.empty:
        threshold = 0.5
    else:
        best = tt.loc[tt["f1"].idxmax()]
        threshold = float(best["threshold"])
        print(
            f"Best threshold={threshold:.6f} | "
            f"F1={best['f1']:.4f} | "
            f"Precision={best['precision']:.4f} | "
            f"Recall={best['recall']:.4f} | "
            f"MCC={best['mcc']:.4f}"
        )
        tt.to_csv(exp_dir / "validation_threshold_metrics.csv", index=False)

    valid_metrics = evaluate(
        "Validation", yva, p_valid, threshold, exp_dir
    )
    test_metrics = evaluate(
        "Test", yte, p_test, threshold, exp_dir
    )

    print("\n" + "=" * 70)
    print(f"{name.upper()} — FEATURE IMPORTANCE & SHAP")
    print("=" * 70)

    gain = model.get_feature_importance(type="PredictionValuesChange")
    gain_df = (
        pd.DataFrame({"feature": features, "importance": gain})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    gain_df.to_csv(exp_dir / "feature_importance_gain.csv", index=False)
    print(gain_df.to_string(index=False))

    rng = np.random.default_rng(SEED)
    if len(Xte) > shap_sample:
        shap_idx = rng.choice(len(Xte), shap_sample, replace=False)
        Xshap = Xte.iloc[shap_idx].copy()
    else:
        Xshap = Xte.copy()

    shap_values = model.get_feature_importance(
        Pool(Xshap, feature_names=features, cat_features=cat_features),
        type="ShapValues",
    )
    shap_matrix = np.asarray(shap_values[:, :-1], dtype=np.float32)
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    shap_df = (
        pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df.to_csv(exp_dir / "shap_importance.csv", index=False)

    metrics = {
        "experiment": name,
        "dataset": str(Path(args.dataset).resolve()),
        "ml_years": "2020-2023",
        "n_features": len(features),
        "threshold": threshold,
        "validation": valid_metrics,
        "test": test_metrics,
    }

    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metrics


def main():
    global args

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="download_dataset/new_dataset/new_data/final_openalex_dataset.parquet",
    )
    ap.add_argument("--output-dir", default="outputs/catboost_splits")
    ap.add_argument("--shap-sample", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. LOAD & PREPROCESS")
    print("=" * 70)

    df = pd.read_parquet(args.dataset)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["target_emergence"] = pd.to_numeric(
        df["target_emergence"], errors="coerce"
    )
    df = df.dropna(subset=["pub_year", "target_emergence"]).copy()
    df["pub_year"] = df["pub_year"].astype(np.int32)
    df["target_emergence"] = df["target_emergence"].astype(np.int8)

    year_totals = df.groupby("pub_year")["doc_id"].transform("count")

    if "topic_historical_volume" in df.columns:
        df["topic_historical_share"] = df["topic_historical_volume"] / (
            year_totals + 1e-5
        )

    if "topic_local_volume" in df.columns:
        df["topic_local_share"] = df["topic_local_volume"] / (
            year_totals + 1e-5
        )

    if "historical_author_count" in df.columns:
        df["historical_author_share"] = df["historical_author_count"] / (
            year_totals + 1e-5
        )

    if "commercial_maturity_index" in df.columns:
        df["commercial_maturity_index"] = df[
            "commercial_maturity_index"
        ].fillna(-1.0)

    drop_features = {
        "has_ref_data",
        "has_reference_list",
        "topic_centroid_similarity",
        "topic_domain_rarity",
        "topic_domain_age_years",
        "topic_historical_volume",
        "topic_local_volume",
        "historical_author_count",
    }

    excluded = {
        "target_emergence",
        "pub_year",
        "doc_id",
    }.union(drop_features)

    features = [c for c in df.columns if c not in excluded]

    X = df[features].copy()
    cat_features = []

    for c in features:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        else:
            X[c] = (
                X[c]
                .fillna("missing")
                .astype("string")
                .astype("category")
            )
            cat_features.append(c)

    print(f"Всего строк в файле: {len(df):,}")
    print(
        f"ML-примеры: {df['pub_year'].between(ML_START_YEAR, ML_END_YEAR).sum():,} "
        f"строк ({ML_START_YEAR}-{ML_END_YEAR})"
    )
    print(f"Фичей в модели ({len(features)}):")
    for i, feature in enumerate(features, 1):
        print(f"  {i:2d}. {feature}")

    if cat_features:
        print(f"Категориальные: {cat_features}")

    print("\n" + "=" * 70)
    print("2. RANDOM SPLIT (2020-2023, STRATIFIED)")
    print("=" * 70)

    random_split = split_random(df, X)

    random_metrics = train_one_experiment(
        "random",
        *random_split[:6],
        features,
        cat_features,
        out,
        args.shap_sample,
    )

    print("\n" + "=" * 70)
    print("3. TEMPORAL SPLIT (2020-2021 | 2022 | 2023)")
    print("=" * 70)

    temporal_split = split_temporal(df, X)

    temporal_metrics = train_one_experiment(
        "temporal",
        *temporal_split[:6],
        features,
        cat_features,
        out,
        args.shap_sample,
    )

    comparison = {
        "random": random_metrics,
        "temporal": temporal_metrics,
    }

    (out / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("DONE — TWO INDEPENDENT EXPERIMENTS")
    print("=" * 70)
    print(
        f"RANDOM   | PR-AUC={random_metrics['test']['pr_auc']:.6f} | "
        f"F1={random_metrics['test']['f1']:.6f}"
    )
    print(
        f"TEMPORAL | PR-AUC={temporal_metrics['test']['pr_auc']:.6f} | "
        f"F1={temporal_metrics['test']['f1']:.6f}"
    )


if __name__ == "__main__":
    main()
