"""
ml/train_isolation_forest.py
Train Isolation Forest + SHAP explainer and save all artefacts.

Usage:
    python -m ml.train_isolation_forest
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("ml.train")
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

DATA_PATH = Path("data/commute_records.csv")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")
SHAP_SAMPLE_PATH = Path("models/shap_values_sample.csv")


# ── 1. DATA LOADING ───────────────────────────────────────────────────────────

def load_and_prepare_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load CSV, engineer features, handle nulls. Returns (df, X_array)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_PATH}. "
            "Run: python data/generate_commute_records.py"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"\n[Data] Shape: {df.shape}")
    print(df.describe().to_string())
    print(f"\n[Data] Missing values:\n{df.isnull().sum().to_string()}")

    # Feature engineering
    df["delay_ratio"] = df["delay_minutes"] / (df["route_avg_delay_min"] + 1.0)

    # Fill missing values with column median
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            missing = df[col].isnull().sum()
            if missing > 0:
                print(f"[Data] Filling {missing} missing values in '{col}' with median={median_val:.3f}")
            df[col] = df[col].fillna(median_val)

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    return df, X


# ── 2. TRAIN ISOLATION FOREST ─────────────────────────────────────────────────

def train_isolation_forest(X: np.ndarray, contamination: float | None = None) -> IsolationForest:
    """Fit Isolation Forest on all rows (unsupervised).

    Args:
        contamination: If None, uses settings value. Pass the real fraud rate
                       from labeled data for a more accurate threshold.
    """
    cont = contamination if contamination is not None else settings.CONTAMINATION
    print(f"\n[Train] Fitting IsolationForest on {X.shape[0]} samples, "
          f"{X.shape[1]} features…  contamination={cont:.4f}")
    model = IsolationForest(
        n_estimators=settings.N_ESTIMATORS,
        max_samples=min(512, X.shape[0]),
        contamination=cont,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)

    labels = model.predict(X)
    n_anomalous = int((labels == -1).sum())
    n_normal = int((labels == 1).sum())
    print(f"[Train] Results -> Normal: {n_normal}, Anomalous: {n_anomalous} "
          f"({n_anomalous / len(labels) * 100:.1f}%)")
    return model


# ── 3. TRAIN SHAP EXPLAINER ───────────────────────────────────────────────────

def train_shap_explainer(
    model: IsolationForest,
    X: np.ndarray,
    df: pd.DataFrame,
) -> shap.TreeExplainer:
    """Create SHAP TreeExplainer and validate on a sample."""
    print("\n[SHAP] Building TreeExplainer…")
    explainer = shap.TreeExplainer(model)

    sample_size = min(50, X.shape[0])
    X_sample = X[:sample_size]
    shap_values_raw = explainer.shap_values(X_sample)

    # shap_values_raw may be a list (one per tree output) or plain array
    if isinstance(shap_values_raw, list):
        sv = shap_values_raw[0]
    else:
        sv = shap_values_raw

    mean_abs = np.abs(sv).mean(axis=0)
    feature_importance = sorted(
        zip(FEATURE_COLUMNS, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    print("[SHAP] Top features by mean |SHAP|:")
    for feat, imp in feature_importance:
        print(f"  {feat:30s}: {imp:.6f}")

    # Save SHAP sample for inspection
    shap_df = pd.DataFrame(sv, columns=FEATURE_COLUMNS)
    SHAP_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_SAMPLE_PATH, index=False)
    print(f"[SHAP] Sample saved -> {SHAP_SAMPLE_PATH}")

    return explainer


# ── 4. SAVE MODELS ────────────────────────────────────────────────────────────

def save_models(model: IsolationForest, explainer: shap.TreeExplainer) -> None:
    """Persist all model artefacts to disk."""
    model_path = Path(settings.ISOLATION_FOREST_MODEL_PATH)
    shap_path = Path(settings.SHAP_EXPLAINER_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, str(model_path))
    print(f"[Save] IsolationForest -> {model_path}")

    with open(str(shap_path), "wb") as f:
        pickle.dump(explainer, f)
    print(f"[Save] SHAP explainer  -> {shap_path}")

    with open(str(FEATURE_COLUMNS_PATH), "w") as f:
        json.dump(FEATURE_COLUMNS, f)
    print(f"[Save] Feature columns -> {FEATURE_COLUMNS_PATH}")

    print("\nModels saved successfully. Run main.py to start the API.")


# ── 5. VALIDATION ─────────────────────────────────────────────────────────────

def validate_on_anomalies(model: IsolationForest, df: pd.DataFrame) -> None:
    """
    Validate recall/precision using ground-truth labels when available.

    - If df has 'is_anomaly' column (IEEE-CIS adapted data) → use it for
      full precision / recall / F1 reporting.
    - Fallback: last-25-rows heuristic for the original synthetic dataset.
    """
    has_labels = "is_anomaly" in df.columns

    if has_labels:
        print("\n[Validate] Using ground-truth 'is_anomaly' labels…")
        df_v = df.copy()
        df_v["delay_ratio"] = df_v["delay_minutes"] / (df_v["route_avg_delay_min"] + 1.0)
        X_v   = df_v[FEATURE_COLUMNS].values.astype(np.float64)
        preds = model.predict(X_v)          # 1=normal, -1=anomaly
        y_true = df_v["is_anomaly"].values  # 1=anomaly, 0=normal

        tp = int(((preds == -1) & (y_true == 1)).sum())
        fp = int(((preds == -1) & (y_true == 0)).sum())
        fn = int(((preds ==  1) & (y_true == 1)).sum())
        tn = int(((preds ==  1) & (y_true == 0)).sum())

        total_pos  = tp + fn
        precision  = tp / (tp + fp + 1e-9)
        recall     = tp / (total_pos + 1e-9)
        f1         = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy   = (tp + tn) / len(y_true)

        print(f"[Validate] Ground-truth anomalies : {total_pos:,}")
        print(f"[Validate] True Positives  (TP)   : {tp:,}")
        print(f"[Validate] False Positives (FP)   : {fp:,}")
        print(f"[Validate] False Negatives (FN)   : {fn:,}")
        print(f"[Validate] True Negatives  (TN)   : {tn:,}")
        print(f"[Validate] Precision               : {precision:.3f}")
        print(f"[Validate] Recall                  : {recall:.3f}")
        print(f"[Validate] F1 Score                : {f1:.3f}")
        print(f"[Validate] Accuracy                : {accuracy:.3f}")

        # Save metrics
        metrics_path = Path("models/validation_metrics.json")
        import json
        metrics = {
            "total_samples": len(y_true),
            "total_anomalies": int(total_pos),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "data_source": "ieee_cis_adapted",
        }
        with open(str(metrics_path), "w") as mf:
            json.dump(metrics, mf, indent=2)
        print(f"[Validate] Metrics saved -> {metrics_path}")

    else:
        # Legacy: last-25-rows are synthetic anomalies
        print("\n[Validate] Checking last 25 rows (synthetic anomalies)…")
        df_anom = df.tail(25).copy()
        df_anom["delay_ratio"] = df_anom["delay_minutes"] / (df_anom["route_avg_delay_min"] + 1.0)

        X_anom = df_anom[FEATURE_COLUMNS].values.astype(np.float64)
        preds  = model.predict(X_anom)
        n_flagged = int((preds == -1).sum())
        n_missed  = 25 - n_flagged

        print(f"[Validate] True anomalies flagged  : {n_flagged}/25")
        print(f"[Validate] True anomalies missed    : {n_missed}/25")
        print(f"[Validate] Recall on anomaly class  : {n_flagged / 25 * 100:.1f}%")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # If IEEE-CIS adapted data exists, use it (it's better).
    # Otherwise fall back to synthetic generation.
    if not DATA_PATH.exists():
        ieee_adapted = Path("data/commute_records_ieee.csv")
        if ieee_adapted.exists():
            print(f"[Setup] Using IEEE-CIS adapted data: {ieee_adapted}")
            import shutil
            shutil.copy(ieee_adapted, DATA_PATH)
        else:
            print("[Setup] commute_records.csv not found. Generating synthetic data…")
            import subprocess
            subprocess.run([sys.executable, "data/generate_commute_records.py"], check=True)

    df, X = load_and_prepare_data()

    # Use real fraud rate as contamination if ground-truth labels exist
    real_contamination = None
    if "is_anomaly" in df.columns:
        real_contamination = float(df["is_anomaly"].mean())
        # Clamp to sklearn's valid range [0.0001, 0.5]
        real_contamination = max(0.0001, min(0.5, real_contamination))
        print(f"[Train] Real anomaly rate from labels: {real_contamination:.4f}")

    model = train_isolation_forest(X, contamination=real_contamination)
    explainer = train_shap_explainer(model, X, df)
    save_models(model, explainer)
    validate_on_anomalies(model, df)
