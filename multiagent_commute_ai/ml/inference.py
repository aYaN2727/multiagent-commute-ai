"""
ml/inference.py
===============
Unified inference interface that auto-selects the best available model:

  Priority 1: models/supervised_model.pkl  (GradientBoostingClassifier)
              -> better calibrated probabilities, higher F1/ROC-AUC
  Priority 2: models/isolation_forest.pkl  (IsolationForest, unsupervised)
              -> fallback when supervised model not yet trained

Both expose the same predict() / explain() API so the rest of the codebase
does not need to know which model is active.

Usage:
    from ml.inference import get_inference
    inf = get_inference()
    result = inf.predict(commute_record_dict)
    expl   = inf.explain(commute_record_dict)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("ml.inference")
settings = get_settings()

FEATURE_COLUMNS_PATH    = Path("models/feature_columns.json")
SUPERVISED_MODEL_PATH   = Path("models/supervised_model.pkl")
SUPERVISED_SHAP_PATH    = Path("models/supervised_shap_explainer.pkl")

# IsolationForest score normalisation (calibrated on training distribution)
_IF_SCORE_MIN: float = -0.6
_IF_SCORE_MAX: float =  0.0


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_feature_columns() -> List[str]:
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns not found: {FEATURE_COLUMNS_PATH}. "
            "Run: python -m ml.train_isolation_forest  OR  python -m ml.train_supervised"
        )
    with open(str(FEATURE_COLUMNS_PATH), "r") as f:
        return json.load(f)


def _prepare_features(
    commute_record: Dict[str, Any],
    feature_columns: List[str],
) -> np.ndarray:
    """Extract and engineer features; return shape (1, n_features) float64 array."""
    delay_minutes = float(commute_record.get("delay_minutes", 0.0))
    route_avg     = float(commute_record.get("route_avg_delay_min", 10.0))

    raw = {
        "distance_km":         float(commute_record.get("distance_km", 10.0)),
        "delay_minutes":       delay_minutes,
        "route_avg_delay_min": route_avg,
        "day_of_week":         float(commute_record.get("day_of_week", 0)),
        "hour_of_day":         float(commute_record.get("hour_of_day", 9)),
        "claim_frequency_30d": float(commute_record.get("claim_frequency_30d", 0.0)),
        "delay_ratio":         delay_minutes / (route_avg + 1.0),
    }
    row = [raw.get(col, 0.0) for col in feature_columns]
    return np.array([row], dtype=np.float64)


def _top3_factors(
    shap_row: np.ndarray,
    X_row: np.ndarray,
    feature_columns: List[str],
) -> List[str]:
    """Return human-readable top-3 SHAP factor strings."""
    shap_dict = {col: float(shap_row[i]) for i, col in enumerate(feature_columns)}
    sorted_factors = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    top_3: List[str] = []
    for feat, val in sorted_factors[:3]:
        feat_val = float(X_row[feature_columns.index(feat)])
        direction = "raised anomaly score" if val > 0 else "lowered anomaly score"
        top_3.append(f"{feat} = {feat_val:.1f} ({direction} by {abs(val):.4f})")
    return top_3


# ── Supervised (GBM) Inference ────────────────────────────────────────────────

class SupervisedInference:
    """
    GradientBoostingClassifier inference with SHAP explanations.
    Trained via: python -m ml.train_supervised
    """

    MODEL_TYPE = "supervised_gbm"

    def __init__(self) -> None:
        if not SUPERVISED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Supervised model not found: {SUPERVISED_MODEL_PATH}. "
                "Run: python -m ml.train_supervised"
            )
        if not SUPERVISED_SHAP_PATH.exists():
            raise FileNotFoundError(
                f"SHAP explainer not found: {SUPERVISED_SHAP_PATH}. "
                "Run: python -m ml.train_supervised"
            )

        self._model: GradientBoostingClassifier = joblib.load(str(SUPERVISED_MODEL_PATH))
        with open(str(SUPERVISED_SHAP_PATH), "rb") as f:
            self._explainer: shap.TreeExplainer = pickle.load(f)
        self._feature_columns: List[str] = _load_feature_columns()

        logger.info(
            "SupervisedInference (GBM) loaded. "
            f"Features: {self._feature_columns}",
            extra={"agent_name": "inference"},
        )

    def prepare_features(self, commute_record: Dict[str, Any]) -> np.ndarray:
        return _prepare_features(commute_record, self._feature_columns)

    def predict(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run GBM prediction on a single commute record.

        Returns:
            {
                "anomaly_score": float,        # raw GBM anomaly probability
                "anomaly_probability": float,  # same (already 0-1 calibrated)
                "is_anomalous": bool,
                "features_used": dict,
                "model_type": "supervised_gbm",
            }
        """
        X = self.prepare_features(commute_record)
        proba: float = float(self._model.predict_proba(X)[0, 1])  # P(anomaly)
        is_anom: bool = proba >= 0.5

        features_used = {
            col: float(X[0, i]) for i, col in enumerate(self._feature_columns)
        }

        logger.debug(
            "GBM prediction",
            extra={
                "agent_name": "inference",
                "anomaly_prob": proba,
                "is_anomalous": is_anom,
            },
        )

        return {
            "anomaly_score":       proba,
            "anomaly_probability": proba,
            "is_anomalous":        is_anom,
            "features_used":       features_used,
            "model_type":          self.MODEL_TYPE,
        }

    def explain(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SHAP values for one record.

        Returns:
            {
                "shap_values": dict,         # feature -> shap_value
                "top_3_factors": list[str],  # human-readable strings
                "base_value": float,
            }
        """
        X = self.prepare_features(commute_record)
        sv_raw = self._explainer.shap_values(X)

        # GBM binary: sv_raw is list[array] or ndarray of shape (1, p, 2)
        if isinstance(sv_raw, list):
            sv_row = sv_raw[1][0]   # class=1 (anomaly)
        elif sv_raw.ndim == 3:
            sv_row = sv_raw[0, :, 1]
        else:
            sv_row = sv_raw[0]

        shap_dict = {
            col: float(sv_row[i]) for i, col in enumerate(self._feature_columns)
        }

        base_val = self._explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
        else:
            base_val = float(base_val)

        top_3 = _top3_factors(sv_row, X[0], self._feature_columns)

        return {
            "shap_values":   shap_dict,
            "top_3_factors": top_3,
            "base_value":    base_val,
        }


# ── Isolation Forest Inference (fallback) ─────────────────────────────────────

class IsolationForestInference:
    """
    Loads the trained IsolationForest, SHAP TreeExplainer, and feature column order.
    Used as fallback when supervised_model.pkl is not available.
    Provides predict() and explain() for a single commute record dict.
    """

    MODEL_TYPE = "isolation_forest"

    def __init__(self) -> None:
        model_path = Path(settings.ISOLATION_FOREST_MODEL_PATH)
        shap_path  = Path(settings.SHAP_EXPLAINER_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Run: python -m ml.train_isolation_forest"
            )
        if not shap_path.exists():
            raise FileNotFoundError(
                f"SHAP explainer not found: {shap_path}. "
                "Run: python -m ml.train_isolation_forest"
            )

        self._model: IsolationForest = joblib.load(str(model_path))
        with open(str(shap_path), "rb") as f:
            self._explainer: shap.TreeExplainer = pickle.load(f)
        self._feature_columns: List[str] = _load_feature_columns()

        logger.info(
            "IsolationForestInference loaded. "
            f"Features: {self._feature_columns}",
            extra={"agent_name": "inference"},
        )

    def prepare_features(self, commute_record: Dict[str, Any]) -> np.ndarray:
        return _prepare_features(commute_record, self._feature_columns)

    def predict(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run IF prediction on a single commute record.

        Returns:
            {
                "anomaly_score": float,
                "anomaly_probability": float,  # normalised 0-1 (higher = more suspicious)
                "is_anomalous": bool,
                "features_used": dict,
                "model_type": "isolation_forest",
            }
        """
        X = self.prepare_features(commute_record)

        raw_score: float = float(self._model.score_samples(X)[0])
        label: int = int(self._model.predict(X)[0])  # 1=normal, -1=anomalous

        # Normalise: flip so higher = more anomalous
        clipped = max(_IF_SCORE_MIN, min(_IF_SCORE_MAX, raw_score))
        anomaly_prob = (clipped - _IF_SCORE_MAX) / (_IF_SCORE_MIN - _IF_SCORE_MAX)
        anomaly_prob = float(np.clip(anomaly_prob, 0.0, 1.0))

        features_used = {
            col: float(X[0, i]) for i, col in enumerate(self._feature_columns)
        }

        logger.debug(
            "IF prediction",
            extra={
                "agent_name": "inference",
                "raw_score": raw_score,
                "anomaly_prob": anomaly_prob,
                "is_anomalous": label == -1,
            },
        )

        return {
            "anomaly_score":       raw_score,
            "anomaly_probability": anomaly_prob,
            "is_anomalous":        label == -1,
            "features_used":       features_used,
            "model_type":          self.MODEL_TYPE,
        }

    def explain(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SHAP values for one record.

        Returns:
            {
                "shap_values": dict,
                "top_3_factors": list[str],
                "base_value": float,
            }
        """
        X = self.prepare_features(commute_record)
        shap_raw = self._explainer.shap_values(X)

        if isinstance(shap_raw, list):
            sv_row = shap_raw[0][0]
        else:
            sv_row = shap_raw[0]

        shap_dict = {
            col: float(sv_row[i]) for i, col in enumerate(self._feature_columns)
        }

        base_val = self._explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        top_3 = _top3_factors(sv_row, X[0], self._feature_columns)

        return {
            "shap_values":   shap_dict,
            "top_3_factors": top_3,
            "base_value":    base_val,
        }


# ── Module-level singleton (auto-selects best model) ─────────────────────────

_inference_instance: Optional[IsolationForestInference | SupervisedInference] = None


def get_inference() -> IsolationForestInference | SupervisedInference:
    """
    Lazy-initialised singleton.
    Automatically uses SupervisedInference (GBM) if supervised_model.pkl exists,
    otherwise falls back to IsolationForestInference.
    """
    global _inference_instance
    if _inference_instance is None:
        if SUPERVISED_MODEL_PATH.exists() and SUPERVISED_SHAP_PATH.exists():
            logger.info(
                "Auto-selecting SupervisedInference (GBM) — supervised_model.pkl found.",
                extra={"agent_name": "inference"},
            )
            _inference_instance = SupervisedInference()
        else:
            logger.info(
                "Using IsolationForestInference (GBM model not found). "
                "Run python -m ml.train_supervised to upgrade.",
                extra={"agent_name": "inference"},
            )
            _inference_instance = IsolationForestInference()
    return _inference_instance
