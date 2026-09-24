import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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


def safe_metric(metric_fn, *args, default=np.nan):
    try:
        value = metric_fn(*args)
    except (ValueError, ZeroDivisionError):
        return default
    return float(value)


def split_temporal_strict(df, X):
    """Строгий temporal split:

    Train: 2020-2021
    Valid: 2022
    Test:  2023
    2017-2019 и 2024-2026 ЖЕСТКО ИГНОРИРУЮТСЯ при обучении.
    """
    train_mask = (df["pub_year"] >= 2020) & (df["pub_year"] <= 2021)
    valid_mask = df["pub_year"] == 2022
    test_mask = df["pub_year"] == 2023

    # Проверка на утечку старых лет
    if (df.loc[train_mask, "pub_year"] < 2020).any():
        raise ValueError(
            "ОШИБКА: Попытка затянуть 2017-2019 года в обучающую выборку!"
        )

    for name, mask in [
        ("train", train_mask),
        ("valid", valid_mask),
        ("test", test_mask),
    ]:
        if not mask.any():
            raise ValueError(f"Сплит '{name}' пуст! Проверь года в датасете.")
        y = df.loc[mask, "target_emergence"]
        print(
            f"{name:5s}: {mask.sum():,} строк | "
            f"Года: {df.loc[mask, 'pub_year'].min()}-{df.loc[mask, 'pub_year'].max()} | "
            f"Позитивных: {y.sum():,} ({100*y.mean():.2f}%)"
        )

    return (
        X.loc[train_mask],
        X.loc[valid_mask],
        X.loc[test_mask],
        df.loc[train_mask, "target_emergence"],
        df.loc[valid_mask, "target_emergence"],
        df.loc[test_mask, "target_emergence"],
        df.loc[test_mask].copy(),
    )


def threshold_table(y, proba, n_thresholds=1000):
    y = np.asarray(y, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)

    if y.size == 0 or np.unique(y).size < 2:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "mcc"])

    precision, recall, pr_thresholds = precision_recall_curve(y, proba)

    if len(pr_thresholds) > n_thresholds:
        idx = np.linspace(0, len(pr_thresholds) - 1, n_thresholds, dtype=int)
        precision, recall, thresholds = precision[idx], recall[idx], pr_thresholds[idx]
    else:
        thresholds = pr_thresholds
        precision, recall = precision[:-1], recall[:-1]

    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0
    )

    preds = proba >= thresholds[:, None]
    tp = (preds & (y == 1)).sum(axis=1)
    tn = (~preds & (y == 0)).sum(axis=1)
    fp = (preds & (y == 0)).sum(axis=1)
    fn = (~preds & (y == 1)).sum(axis=1)

    mcc_denom = np.sqrt(
        (tp + fp).astype(np.float64)
        * (tp + fn).astype(np.float64)
        * (tn + fp).astype(np.float64)
        * (tn + fn).astype(np.float64)
    )
    mcc = np.divide(
        (tp * tn) - (fp * fn),
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


def evaluate(name, y, proba, threshold, out_dir):
    y = np.asarray(y, dtype=np.int8)
    pred = (proba >= threshold).astype(np.int8)

    labels = (
        np.unique(np.concatenate([y, pred]))
        if y.size
        else np.array([0, 1], dtype=np.int8)
    )
    cm = confusion_matrix(
        y, pred, labels=np.array(sorted(np.unique(labels)), dtype=np.int8)
    )

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
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }

    print(f"\n{name}")
    print("-" * 50)
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
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name.lower()}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="download_dataset/new_dataset/new_data/final_openalex_dataset.parquet",
    )
    ap.add_argument("--output-dir", default="outputs/catboost")
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--shap-sample", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. LOAD & PREPROCESS")
    print("=" * 70)
    df = pd.read_parquet(args.dataset)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["target_emergence"] = pd.to_numeric(df["target_emergence"], errors="coerce")
    df = df.dropna(subset=["pub_year", "target_emergence"]).copy()
    df["pub_year"] = df["pub_year"].astype(np.int32)
    df["target_emergence"] = df["target_emergence"].astype(np.int8)

    # 1.1. Относительные доли
    year_totals = df.groupby("pub_year")["doc_id"].transform("count")
    if "topic_historical_volume" in df.columns:
        df["topic_historical_share"] = df["topic_historical_volume"] / (
            year_totals + 1e-5
        )
    if "topic_local_volume" in df.columns:
        df["topic_local_share"] = df["topic_local_volume"] / (year_totals + 1e-5)
    if "historical_author_count" in df.columns:
        df["historical_author_share"] = df["historical_author_count"] / (
            year_totals + 1e-5
        )

    # 1.2. Обработка пропусков
    if "commercial_maturity_index" in df.columns:
        df["commercial_maturity_index"] = df["commercial_maturity_index"].fillna(-1.0)

    # 1.3. Исключаем строго константы и 100% дубли
    drop_features = {
        "has_ref_data",  # Константа = 1
        "has_reference_list",  # Константа = 1
        "topic_centroid_similarity",  # Дубль novelty_raw (rho = -1.0)
        "topic_domain_rarity",  # Дубль topic_local_volume (rho = -1.0)
        "topic_domain_age_years",  # Дубль topic_age_years (rho = 1.0)
        # Абсолютные счетчики выкидываем, так как есть относительные *_share
        "topic_historical_volume",
        "topic_local_volume",
        "historical_author_count",
    }

    excluded = {"target_emergence", "pub_year", "doc_id"}.union(drop_features)
    features = [c for c in df.columns if c not in excluded]

    X = df[features].copy()
    cat_features = []
    for c in features:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        else:
            X[c] = X[c].fillna("missing").astype("string").astype("category")
            cat_features.append(c)

    print(f"Всего строк в файле: {len(df):,}")
    print(f"Фичей в модели ({len(features)}):")
    for i, x in enumerate(features, 1):
        print(f"  {i:2d}. {x}")
    if cat_features:
        print(f"Категориальные: {cat_features}")

    print("\n" + "=" * 70)
    print("2. STRICT TEMPORAL SPLIT (TRAIN: 2020-2021 | VALID: 2022 | TEST: 2023)")
    print("=" * 70)
    Xtr, Xva, Xte, ytr, yva, yte, test_df = split_temporal_strict(df, X)

    print("\n" + "=" * 70)
    print("3. CATBOOST TRAINING")
    print("=" * 70)

    train_pool = Pool(Xtr, ytr, feature_names=features, cat_features=cat_features)
    valid_pool = Pool(Xva, yva, feature_names=features, cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=15.0,
        rsm=0.8,  # colsample_bylevel: берем 80% фич на каждый сплит
        subsample=0.8,
        random_seed=SEED,
        # auto_class_weights УБРАН - сохраняем реальное распределение
        loss_function="Logloss",
        eval_metric="Logloss",
        od_type="Iter",
        od_wait=150,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    model.save_model(out / "catboost_model.cbm")

    p_valid = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    print("\n" + "=" * 70)
    print("4. THRESHOLD SEARCH (VALIDATION ONLY)")
    print("=" * 70)

    tt = threshold_table(np.asarray(yva), p_valid)
    if tt.empty:
        threshold = 0.5
    else:
        best = tt.loc[tt["f1"].idxmax()]
        threshold = float(best["threshold"])
        print(
            f"Best threshold={threshold:.6f} | F1={best['f1']:.4f} | "
            f"Precision={best['precision']:.4f} | Recall={best['recall']:.4f} | MCC={best['mcc']:.4f}"
        )
        tt.to_csv(out / "validation_threshold_metrics.csv", index=False)

    valid_metrics = evaluate("Validation", yva, p_valid, threshold, out)
    test_metrics = evaluate("Test", yte, p_test, threshold, out)

    print("\n" + "=" * 70)
    print("5. FEATURE IMPORTANCE & SHAP")
    print("=" * 70)

    gain = model.get_feature_importance(type="PredictionValuesChange")
    gain_df = (
        pd.DataFrame({"feature": features, "importance": gain})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    gain_df.to_csv(out / "feature_importance_gain.csv", index=False)
    print(gain_df.to_string(index=False))

    # SHAP
    rng = np.random.default_rng(SEED)
    Xshap = (
        Xte.iloc[rng.choice(len(Xte), args.shap_sample, replace=False)].copy()
        if len(Xte) > args.shap_sample
        else Xte.copy()
    )
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
    shap_df.to_csv(out / "shap_importance.csv", index=False)

    # Сохранение результатов
    metrics = {
        "dataset": str(Path(args.dataset).resolve()),
        "train_years": "2020-2021",
        "valid_year": 2022,
        "test_year": 2023,
        "n_features": len(features),
        "threshold": threshold,
        "validation": valid_metrics,
        "test": test_metrics,
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"TEST PR-AUC: {test_metrics['pr_auc']:.6f}")
    print(f"TEST F1:     {test_metrics['f1']:.6f}")


if __name__ == "__main__":
    main()