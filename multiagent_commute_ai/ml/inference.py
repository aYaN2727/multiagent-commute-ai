"""
ml/inference.py
IsolationForestInference: load trained model + SHAP explainer, run predictions.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
import shap

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("ml.inference")
settings = get_settings()

FEATURE_COLUMNS_PATH = Path("models/feature_columns.json")

# Score normalisation constants — calibrated on training data distribution.
# Raw IF score_samples() returns values roughly in [-0.6, 0.0] for this domain.
_SCORE_MIN: float = -0.6
_SCORE_MAX: float = 0.0


class IsolationForestInference:
    """
    Loads the trained IsolationForest, SHAP TreeExplainer, and feature column order.
    Provides predict() and explain() for a single commute record dict.
    """

    def __init__(self) -> None:
        model_path = Path(settings.ISOLATION_FOREST_MODEL_PATH)
        shap_path = Path(settings.SHAP_EXPLAINER_PATH)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run: python -m ml.train_isolation_forest"
            )
        if not shap_path.exists():
            raise FileNotFoundError(
                f"SHAP explainer not found: {shap_path}. Run: python -m ml.train_isolation_forest"
            )
        if not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                f"Feature columns not found: {FEATURE_COLUMNS_PATH}. "
                "Run: python -m ml.train_isolation_forest"
            )

        self._model: IsolationForest = joblib.load(str(model_path))
        with open(str(shap_path), "rb") as f:
            self._explainer: shap.TreeExplainer = pickle.load(f)
        with open(str(FEATURE_COLUMNS_PATH), "r") as f:
            self._feature_columns: List[str] = json.load(f)

        logger.info(
            f"IsolationForestInference loaded. Features: {self._feature_columns}",
            extra={"agent_name": "inference"},
        )

    def prepare_features(self, commute_record: Dict[str, Any]) -> np.ndarray:
        """
        Extract and engineer features from a commute record dict.
        Returns shape (1, n_features) float64 numpy array in training column order.
        """
        delay_minutes = float(commute_record.get("delay_minutes", 0.0))
        route_avg = float(commute_record.get("route_avg_delay_min", 10.0))

        raw = {
            "distance_km": float(commute_record.get("distance_km", 10.0)),
            "delay_minutes": delay_minutes,
            "route_avg_delay_min": route_avg,
            "day_of_week": float(commute_record.get("day_of_week", 0)),
            "hour_of_day": float(commute_record.get("hour_of_day", 9)),
            "claim_frequency_30d": float(commute_record.get("claim_frequency_30d", 0.0)),
            "delay_ratio": delay_minutes / (route_avg + 1.0),
        }

        # Build array in exact training order
        row = [raw.get(col, 0.0) for col in self._feature_columns]
        return np.array([row], dtype=np.float64)

    def predict(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run IF prediction on a single commute record.

        Returns:
            {
                "anomaly_score": float,       # Raw score_samples() value
                "anomaly_probability": float,  # Normalised 0-1 (higher = more suspicious)
                "is_anomalous": bool,
                "features_used": dict,
            }
        """
        X = self.prepare_features(commute_record)

        raw_score: float = float(self._model.score_samples(X)[0])
        label: int = int(self._model.predict(X)[0])  # 1=normal, -1=anomalous

        # Normalise: flip so higher = more anomalous
        clipped = max(_SCORE_MIN, min(_SCORE_MAX, raw_score))
        anomaly_prob = (clipped - _SCORE_MAX) / (_SCORE_MIN - _SCORE_MAX)
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
            "anomaly_score": raw_score,
            "anomaly_probability": anomaly_prob,
            "is_anomalous": label == -1,
            "features_used": features_used,
        }

    def explain(self, commute_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute SHAP values for one record.

        Returns:
            {
                "shap_values": dict,          # feature_name -> shap_value
                "top_3_factors": list[str],   # human-readable strings
                "base_value": float,
            }
        """
        X = self.prepare_features(commute_record)
        shap_raw = self._explainer.shap_values(X)

        # Handle list vs array output
        if isinstance(shap_raw, list):
            sv_row = shap_raw[0][0]
        else:
            sv_row = shap_raw[0]

        shap_dict = {
            col: float(sv_row[i]) for i, col in enumerate(self._feature_columns)
        }

        # base_value may be a list; take first element
        base_val = self._explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])
        else:
            base_val = float(base_val)

        # Top-3 factors sorted by absolute SHAP value (descending)
        sorted_factors = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
        top_3: List[str] = []
        for feat, val in sorted_factors[:3]:
            feat_val = float(X[0, self._feature_columns.index(feat)])
            direction = "pushed score UP" if val < 0 else "pushed score DOWN"
            top_3.append(f"{feat} = {feat_val:.1f} {direction} by {abs(val):.4f}")

        return {
            "shap_values": shap_dict,
            "top_3_factors": top_3,
            "base_value": base_val,
        }


# ── Module-level singleton ────────────────────────────────────────────────────
_inference_instance: Optional[IsolationForestInference] = None


def get_inference() -> IsolationForestInference:
    """Lazy-initialised singleton IsolationForestInference."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = IsolationForestInference()
    return _inference_instance
