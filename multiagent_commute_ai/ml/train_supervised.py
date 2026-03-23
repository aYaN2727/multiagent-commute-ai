"""
ml/train_supervised.py
======================
Fine-tunes the anomaly detector by replacing the unsupervised Isolation Forest
with a supervised GradientBoosting classifier trained on labeled IEEE-CIS data.

Why this is "fine-tuning":
  - We already have a trained IsolationForest baseline (ROC-AUC 0.9541, F1 0.69).
  - We now have 220k ground-truth labels (isFraud/is_anomaly) from IEEE-CIS.
  - We FINE-TUNE the detector by training a supervised GBM on those labels,
    keeping the same 7-feature space and SHAP explainability pipeline.
  - Expected improvement: ROC-AUC 0.97+, F1 0.85+, PR-AUC 0.90+

Model choice - GradientBoostingClassifier:
  - Full sklearn SHAP TreeExplainer support (no compatibility issues)
  - Calibrated probabilities out of the box (log-loss objective)
  - Excellent on tabular data with correlated features
  - Interpretable via SHAP

Usage:
    python -m ml.train_supervised              # standard run
    python -m ml.train_supervised --cv         # include 5-fold cross-validation
    python -m ml.train_supervised --quick      # faster (100 trees, depth 3)

Output artefacts:
    models/supervised_model.pkl               # GradientBoostingClassifier
    models/supervised_shap_explainer.pkl      # SHAP TreeExplainer
    models/supervised_metrics.json            # full evaluation report
    models/feature_columns.json               # (unchanged, reused by inference)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("ml.train_supervised")
settings = get_settings()

FEATURE_COLUMNS = [
    "distance_km",
    "delay_minutes",
    "route_avg_delay_min",
    "day_of_week",
    "hour_of_day",
    "claim_frequency_30d",
    "delay_ratio",
]

DATA_PATH            = Path("data/commute_records.csv")
MODEL_OUT            = Path("models/supervised_model.pkl")
SHAP_OUT             = Path("models/supervised_shap_explainer.pkl")
METRICS_OUT          = Path("models/supervised_metrics.json")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")


# ── 1. LOAD + PREPARE ─────────────────────────────────────────────────────────

def load_and_prepare(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load commute_records.csv, engineer delay_ratio feature, return X, y, df.
    Requires 'is_anomaly' column (IEEE-CIS adapted data).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}\n"
            "Run first:  python data/adapt_ieee_cis.py"
        )

    df = pd.read_csv(DATA_PATH)

    if "is_anomaly" not in df.columns:
        raise ValueError(
            "'is_anomaly' column missing from commute_records.csv.\n"
            "This script requires labeled data from adapt_ieee_cis.py."
        )

    # Feature engineering: delay_ratio
    df["delay_ratio"] = df["delay_minutes"] / (df["route_avg_delay_min"] + 1.0)

    # Fill NaNs with column median
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df["is_anomaly"].values.astype(int)

    if verbose:
        print(f"\n[Data] Rows: {len(df):,}  |  Anomaly rate: {y.mean():.3%}")
        print(f"[Data] Features: {FEATURE_COLUMNS}")
        pos, neg = y.sum(), (y == 0).sum()
        print(f"[Data] Class balance -> anomaly: {pos:,}  normal: {neg:,}  ratio: 1:{neg//max(pos,1)}")

    return X, y, df


# ── 2. TRAIN ──────────────────────────────────────────────────────────────────

def train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> GradientBoostingClassifier:
    """
    Train a GradientBoostingClassifier on labeled commute/fraud data.

    Hyperparameters:
      n_estimators=200  - enough trees for good performance, not too slow
      max_depth=4       - captures feature interactions without overfitting
      learning_rate=0.1 - standard shrinkage
      subsample=0.8     - stochastic boosting reduces variance
      min_samples_leaf=20 - prevents overfitting on noisy samples
    """
    print(f"\n[Train] GradientBoostingClassifier")
    print(f"        n_estimators={n_estimators}  max_depth={max_depth}  "
          f"lr={learning_rate}  train_rows={len(X_train):,}")

    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        verbose=1,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[Train] Done in {elapsed:.1f}s")
    return model


# ── 3. EVALUATE ───────────────────────────────────────────────────────────────

def evaluate(
    model: GradientBoostingClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    split_label: str = "Test",
) -> dict:
    """Full classification metrics on a held-out set."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc  = roc_auc_score(y_test, y_prob)
    pr   = average_precision_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    acc  = accuracy_score(y_test, y_pred)

    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  {split_label} Evaluation")
    print(f"  Samples: {len(y_test):,}  |  Anomalies: {y_test.sum():,}")
    print(f"  ROC-AUC            : {roc:.4f}")
    print(f"  PR-AUC             : {pr:.4f}")
    print(f"  F1 Score           : {f1:.4f}")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(sep)

    return {
        "split": split_label,
        "n_samples": len(y_test),
        "n_anomalies": int(y_test.sum()),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 4,
) -> dict:
    """5-fold stratified cross-validation — catches overfitting across all folds."""
    print("\n[CV] 5-fold StratifiedKFold cross-validation…")
    model_cv = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_scores = cross_val_score(model_cv, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_scores  = cross_val_score(model_cv, X, y, cv=cv, scoring="f1",      n_jobs=-1)

    print(f"[CV] ROC-AUC : {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
    print(f"[CV] F1      : {f1_scores.mean():.4f}  (+/- {f1_scores.std():.4f})")

    return {
        "cv_roc_auc_mean": round(float(roc_scores.mean()), 4),
        "cv_roc_auc_std":  round(float(roc_scores.std()),  4),
        "cv_f1_mean":      round(float(f1_scores.mean()),  4),
        "cv_f1_std":       round(float(f1_scores.std()),   4),
        "cv_folds": 5,
    }


# ── 4. SHAP EXPLAINER ─────────────────────────────────────────────────────────

def build_shap_explainer(
    model: GradientBoostingClassifier,
    X_sample: np.ndarray,
) -> shap.TreeExplainer:
    """Fit SHAP TreeExplainer and print feature importances."""
    print("\n[SHAP] Building TreeExplainer…")
    explainer = shap.TreeExplainer(model)

    sv = explainer.shap_values(X_sample[:500])
    # GBM binary classifier: shap_values returns shape (n, p, 2) or (n, p)
    if isinstance(sv, list):
        sv = sv[1]           # class=1 (anomaly)
    elif sv.ndim == 3:
        sv = sv[:, :, 1]     # last axis is class

    mean_abs = np.abs(sv).mean(axis=0)
    ranked   = sorted(zip(FEATURE_COLUMNS, mean_abs.tolist()), key=lambda x: x[1], reverse=True)

    print("[SHAP] Feature importance (mean |SHAP|):")
    for feat, imp in ranked:
        bar = "#" * int(imp / max(mean_abs) * 30)
        print(f"  {feat:30s}: {imp:.5f}  {bar}")

    return explainer


# ── 5. SAVE ───────────────────────────────────────────────────────────────────

def save_artefacts(
    model: GradientBoostingClassifier,
    explainer: shap.TreeExplainer,
    metrics: dict,
) -> None:
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, str(MODEL_OUT))
    print(f"\n[Save] Supervised model    -> {MODEL_OUT}")

    with open(str(SHAP_OUT), "wb") as f:
        pickle.dump(explainer, f)
    print(f"[Save] SHAP explainer      -> {SHAP_OUT}")

    with open(str(METRICS_OUT), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] Metrics report      -> {METRICS_OUT}")

    # Re-save feature columns (same as IF training, for inference compatibility)
    with open(str(FEATURE_COLUMNS_PATH), "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    print(f"[Save] Feature columns     -> {FEATURE_COLUMNS_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune anomaly detector: IF -> Supervised GBM"
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold cross-validation (slower, more thorough)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 100 trees, depth 3 (for rapid iteration)"
    )
    args = parser.parse_args()

    n_estimators = 100 if args.quick else 200
    max_depth    = 3   if args.quick else 4

    print("=" * 60)
    print("  Fine-Tuning: Isolation Forest -> Supervised GBM")
    print(f"  Mode: {'quick' if args.quick else 'full'}  |  "
          f"CV: {'yes' if args.cv else 'no'}")
    print("=" * 60)

    # 1. Load
    X, y, df = load_and_prepare()

    # 2. Train/test split (stratified, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"\n[Split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # 3. Train
    model = train_gbm(X_train, y_train, n_estimators=n_estimators, max_depth=max_depth)

    # 4. Evaluate on train (to check overfitting) and test
    train_metrics = evaluate(model, X_train, y_train, split_label="Train")
    test_metrics  = evaluate(model, X_test,  y_test,  split_label="Test (held-out)")

    overfit_gap = round(train_metrics["roc_auc"] - test_metrics["roc_auc"], 4)
    print(f"\n[Overfit check] ROC-AUC gap (train - test): {overfit_gap:+.4f}")
    if abs(overfit_gap) < 0.02:
        print("[Overfit check] PASSED - gap < 0.02  (no overfitting detected)")
    else:
        print("[Overfit check] WARNING - gap >= 0.02 (consider reducing max_depth)")

    # 5. Optional cross-validation
    cv_metrics = {}
    if args.cv:
        cv_metrics = cross_validate(X, y, n_estimators=n_estimators, max_depth=max_depth)

    # 6. SHAP explainer
    explainer = build_shap_explainer(model, X_train)

    # 7. Assemble metrics report
    metrics = {
        "model": "GradientBoostingClassifier",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": 0.1,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "total_rows": len(X),
        "anomaly_rate": round(float(y.mean()), 4),
        "train": train_metrics,
        "test": test_metrics,
        "overfit_roc_gap": overfit_gap,
        **cv_metrics,
        "features": FEATURE_COLUMNS,
    }

    # 8. Save
    save_artefacts(model, explainer, metrics)

    print("\n" + "=" * 60)
    print("  Fine-tuning complete!")
    print(f"  Test ROC-AUC : {test_metrics['roc_auc']:.4f}")
    print(f"  Test F1      : {test_metrics['f1']:.4f}")
    print(f"  Test PR-AUC  : {test_metrics['pr_auc']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 60)
    print("\nNext steps:")
    print("  python main.py   (inference.py auto-detects supervised model)")


if __name__ == "__main__":
    main()
