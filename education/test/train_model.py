import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def split_temporal(df, X):
    # Чистый стыковой сплит без перекрытий
    train = df["pub_year"] <= 2018
    valid = (df["pub_year"] >= 2019) & (df["pub_year"] <= 2020)
    test = (df["pub_year"] >= 2021) & (df["pub_year"] <= 2022)

    for name, mask in [("train", train), ("valid", valid), ("test", test)]:
        if not mask.any():
            raise ValueError(f"{name} split is empty")
        y = df.loc[mask, "target_emergence"]
        print(
            f"{name:5s}: {mask.sum():,} rows | "
            f"{df.loc[mask, 'pub_year'].min()}-{df.loc[mask, 'pub_year'].max()} | "
            f"positive={y.sum():,} ({100*y.mean():.2f}%)"
        )

    return (
        X.loc[train], X.loc[valid], X.loc[test],
        df.loc[train, "target_emergence"],
        df.loc[valid, "target_emergence"],
        df.loc[test, "target_emergence"],
        df.loc[test].copy(),
    )


def threshold_table(y, proba, n_thresholds=1000):
    """Быстрый векторный поиск порогов без OOM и зависаний."""
    y = np.asarray(y, dtype=np.float64)
    proba = np.asarray(proba, dtype=np.float64)

    if y.size == 0 or np.unique(y).size < 2:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "mcc"])

    # 1. Быстрый PR-curve из sklearn (считается мгновенно)
    precision, recall, pr_thresholds = precision_recall_curve(y, proba)

    # 2. Если порогов слишком много, квантуем сетку до 1000 точек
    if len(pr_thresholds) > n_thresholds:
        idx = np.linspace(0, len(pr_thresholds) - 1, n_thresholds, dtype=int)
        precision, recall, thresholds = precision[idx], recall[idx], pr_thresholds[idx]
    else:
        thresholds = pr_thresholds
        precision, recall = precision[:-1], recall[:-1]

    # 3. Векторный F1
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    # 4. Векторный MCC через бродкастинг (без вызовов matthews_corrcoef в цикле!)
    preds = proba >= thresholds[:, None]
    tp = (preds & (y == 1)).sum(axis=1)
    tn = (~preds & (y == 0)).sum(axis=1)
    fp = (preds & (y == 0)).sum(axis=1)
    fn = (~preds & (y == 1)).sum(axis=1)

    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide((tp * tn) - (fp * fn), mcc_denom, out=np.zeros_like(mcc_denom), where=mcc_denom > 0)

    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    })


def evaluate(name, y, proba, threshold, out_dir):
    y = np.asarray(y, dtype=np.int8)
    pred = (proba >= threshold).astype(np.int8)

    labels = np.unique(np.concatenate([y, pred])) if y.size else np.array([0, 1], dtype=np.int8)
    cm = confusion_matrix(y, pred, labels=np.array(sorted(np.unique(labels)), dtype=np.int8))

    metrics = {
        "threshold": float(threshold),
        "roc_auc": safe_metric(roc_auc_score, y, proba, default=np.nan),
        "pr_auc": safe_metric(average_precision_score, y, proba, default=np.nan),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if np.unique(y).size == 2 and np.unique(pred).size == 2 else np.nan,
        "tn": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "fp": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "fn": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        "tp": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
    }

    print(f"\n{name}")
    print("-" * 50)
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.6f}" if isinstance(v, float) else f"{k:12s}: {v}")
    print("\nConfusion matrix:")
    print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"],
                       columns=["Pred 0", "Pred 1"]))
    print("\nClassification report:")
    print(classification_report(y, pred, digits=4, zero_division=0))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title(f"{name} confusion matrix (threshold={threshold:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
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
    ap.add_argument("--dataset", default="./data/train_data/processed_ml_dataset.parquet")
    ap.add_argument("--output-dir", default="outputs/catboost")
    ap.add_argument("--train-end", type=int, default=2018)
    ap.add_argument("--valid-end", type=int, default=2020)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--shap-sample", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("1. LOAD")
    print("=" * 70)
    df = pd.read_parquet(args.dataset)
    df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce")
    df["target_emergence"] = pd.to_numeric(df["target_emergence"], errors="coerce")
    df = df.dropna(subset=["pub_year", "target_emergence"]).copy()
    df["pub_year"] = df["pub_year"].astype(np.int32)
    df["target_emergence"] = df["target_emergence"].astype(np.int8)

    # =========================================================================
    # 1.1. ДЕЛАЕМ ОТНОСИТЕЛЬНЫЕ ФИЧИ "НА ЛЕТУ" (Без перезаписи датасета на диск)
    # =========================================================================
    year_totals = df.groupby("pub_year")["doc_id"].transform("count")
    
    if "topic_historical_volume" in df.columns:
        df["topic_historical_share"] = df["topic_historical_volume"] / (year_totals + 1e-5)
    if "topic_local_volume" in df.columns:
        df["topic_local_share"] = df["topic_local_volume"] / (year_totals + 1e-5)
    if "historical_author_count" in df.columns:
        df["historical_author_share"] = df["historical_author_count"] / (year_totals + 1e-5)

    # =========================================================================
    # 1.2. ИСКЛЮЧАЕМ МУСОР + СТАРЫЕ АБСОЛЮТНЫЕ СЧЕТЧИКИ
    # =========================================================================
    junk_features = {
        # Старый мусор с близким к нулю SHAP
        "has_ref_data", "has_reference_list", "is_outlier_cluster", 
        "citation_velocity", "topic_centroid_similarity", 
        "commercial_maturity_index", "novelty_raw", "cross_topic_gap",
        "source_novelty", "citation_acceleration", "source_tier",
        
        # Выкидываем абсолютные гиганты, чтобы модель смотрела только на доли (shares)
        "topic_historical_volume",
        "topic_local_volume",
        "historical_author_count"
    }

    excluded = {"target_emergence", "pub_year", "doc_id"}.union(junk_features)

    features = [
        c for c in df.columns
        if c not in excluded
        and (
            pd.api.types.is_numeric_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_categorical_dtype(df[c])
        )
    ]
    if not features:
        raise ValueError("No usable feature columns found after excluding targets and metadata.")

    X = df[features].copy()
    cat_features = []
    for c in features:
        if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
        else:
            X[c] = X[c].fillna("missing").astype("string").astype("category")
            cat_features.append(c)

    print(f"Dataset: {df.shape}")
    print(f"Features: {len(features)}")
    print("\n".join(f"  {i+1:2d}. {x}" for i, x in enumerate(features)))
    if cat_features:
        print(f"Categorical features: {cat_features}")

    print("\n" + "=" * 70)
    print("2. TEMPORAL SPLIT")
    print("=" * 70)
    Xtr, Xva, Xte, ytr, yva, yte, test_df = split_temporal(df, X)

    print("\n" + "=" * 70)
    print("3. CATBOOST TRAINING (WITH HEAVY REGULARIZATION)")
    print("=" * 70)

    train_pool = Pool(Xtr, ytr, feature_names=features, cat_features=cat_features)
    valid_pool = Pool(Xva, yva, feature_names=features, cat_features=cat_features)

    # =========================================================================
    # 3.1. МОДИФИЦИРОВАННЫЙ CATBOOST (Сбиваем спесь с оверфиттинга)
    # =========================================================================
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=0.03,         # Уменьшили шаг, чтобы не заучивал моментально
        depth=5,                    # Уменьшили глубину деревьев (было скорее всего 6 или 8)
        loss_function="Logloss",
        eval_metric="PRAUC",
        l2_leaf_reg=10.0,           # Зажали L2 регуляризацию (было 5.0)
        random_strength=2.0,        # Заставили делать более случайные сплиты
        subsample=0.8,              # Берем 80% датасета на дерево
        random_seed=SEED,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=100,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1,
        cat_features=cat_features,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    model.save_model(out / "catboost_model.cbm")
    print(f"Best iteration: {model.get_best_iteration()}")

    p_valid = model.predict_proba(Xva)[:, 1]
    p_test = model.predict_proba(Xte)[:, 1]

    print("\n" + "=" * 70)
    print("4. THRESHOLD SEARCH — VALIDATION ONLY")
    print("=" * 70)

    tt = threshold_table(np.asarray(yva), p_valid)
    if tt.empty:
        threshold = 0.5
        best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0}
        print("Validation split does not contain both classes; using default threshold=0.5")
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

        tt.to_csv(out / "validation_threshold_metrics.csv", index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        for col in ["f1", "precision", "recall", "mcc"]:
            ax.plot(tt["threshold"], tt[col], label=col.upper())
        ax.axvline(threshold, linestyle="--", label=f"best={threshold:.4f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Metric")
        ax.set_title("Validation threshold search")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "validation_threshold_search.png", dpi=180)
        plt.close(fig)

    valid_metrics = evaluate("Validation", yva, p_valid, threshold, out)
    test_metrics = evaluate("Test", yte, p_test, threshold, out)

    print("\n" + "=" * 70)
    print("5. GAIN-LIKE FEATURE IMPORTANCE")
    print("=" * 70)

    # CatBoost does not expose XGBoost's literal 'gain' statistic.
    # PredictionValuesChange is CatBoost's standard tree feature
    # importance and is the closest useful analogue for this purpose.
    gain = model.get_feature_importance(type="PredictionValuesChange")
    gain_df = pd.DataFrame({"feature": features, "importance": gain}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)

    print(gain_df.to_string(index=False))
    gain_df.to_csv(out / "feature_importance_gain_like.csv", index=False)

    top = gain_df.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, len(top) * 0.32)))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("PredictionValuesChange (CatBoost Gain-like)")
    ax.set_title("CatBoost feature importance")
    fig.tight_layout()
    fig.savefig(out / "feature_importance_gain_like.png", dpi=180)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("6. SHAP")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    if len(Xte) > args.shap_sample:
        idx = rng.choice(len(Xte), args.shap_sample, replace=False)
        Xshap = Xte.iloc[idx].copy()
    else:
        Xshap = Xte.copy()

    shap_values = model.get_feature_importance(
        Pool(Xshap, feature_names=features, cat_features=cat_features),
        type="ShapValues",
    )
    shap_matrix = np.asarray(shap_values[:, :-1], dtype=np.float32)
    mean_abs = np.abs(shap_matrix).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(shap_df.to_string(index=False))
    shap_df.to_csv(out / "shap_mean_abs_importance.csv", index=False)

    # SHAP beeswarm-like summary, no dependency on the standalone shap package.
    top_features = shap_df.head(min(20, len(features)))["feature"].tolist()
    pos = {f: i for i, f in enumerate(features)}

    fig, ax = plt.subplots(figsize=(11, max(7, len(top_features) * 0.38)))
    for row, feature in enumerate(top_features):
        vals = shap_matrix[:, pos[feature]]
        if len(vals) > 10000:
            ii = rng.choice(len(vals), 10000, replace=False)
            vals = vals[ii]
        jitter = rng.uniform(-0.20, 0.20, len(vals))
        ax.scatter(vals, row + jitter, s=5, alpha=0.25)

    ax.axvline(0, linestyle="--")
    ax.set_yticks(np.arange(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (positive → higher predicted emergence)")
    ax.set_title("SHAP summary — top features")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "shap_summary.png", dpi=180)
    plt.close(fig)

    # Test predictions
    predictions = (p_test >= threshold).astype(np.int8)
    pred_df = pd.DataFrame({
        "doc_id": test_df["doc_id"].values if "doc_id" in test_df else np.arange(len(test_df)),
        "pub_year": test_df["pub_year"].values,
        "target_emergence": test_df["target_emergence"].values,
        "predicted_probability": p_test,
        "predicted_label": predictions,
    })
    pred_df.to_parquet(out / "test_predictions.parquet", index=False)

    metrics = {
        "dataset": str(Path(args.dataset).resolve()),
        "train_end": args.train_end,
        "valid_end": args.valid_end,
        "test_start": args.valid_end + 1,
        "n_features": len(features),
        "features": features,
        "best_iteration": int(model.get_best_iteration()),
        "threshold": threshold,
        "validation": valid_metrics,
        "test": test_metrics,
        "catboost": {
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "depth": args.depth,
            "auto_class_weights": "Balanced",
            "eval_metric": "PRAUC",
        },
    }
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Output: {out.resolve()}")
    print(f"Threshold: {threshold:.6f}")
    print(f"TEST PR-AUC: {test_metrics['pr_auc']:.6f}")
    print(f"TEST F1:     {test_metrics['f1']:.6f}")
    print(f"TEST MCC:    {test_metrics['mcc']:.6f}")

    
if __name__ == "__main__":
    main()

