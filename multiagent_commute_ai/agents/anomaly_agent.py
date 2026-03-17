"""
agents/anomaly_agent.py
Isolation Forest inference on the employee's commute record.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from agents.state import AgentState
from ml.inference import get_inference
from utils.logger import get_logger

logger = get_logger("agent.anomaly")


async def anomaly_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: run Isolation Forest on the commute record.
    Returns: anomaly_score, anomaly_probability, is_anomalous, anomaly_features.
    """
    t0 = time.perf_counter()
    logger.info("Anomaly agent starting", extra={"agent_name": "anomaly_agent"})

    commute_record: Dict[str, Any] = state.get("commute_record", {})

    if not commute_record:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "No commute record provided; skipping anomaly detection",
            extra={"agent_name": "anomaly_agent", "elapsed_ms": round(elapsed, 2)},
        )
        errors = list(state.get("errors", [])) + ["No commute record provided for anomaly detection"]
        return {
            "anomaly_score": 0.0,
            "anomaly_probability": 0.0,
            "is_anomalous": False,
            "anomaly_features": {},
            "errors": errors,
        }

    try:
        inference = get_inference()
        result = await asyncio.to_thread(inference.predict, commute_record)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Anomaly detection complete — is_anomalous={result['is_anomalous']}, "
            f"prob={result['anomaly_probability']:.3f}",
            extra={"agent_name": "anomaly_agent", "elapsed_ms": round(elapsed, 2)},
        )

        return {
            "anomaly_score": result["anomaly_score"],
            "anomaly_probability": result["anomaly_probability"],
            "is_anomalous": result["is_anomalous"],
            "anomaly_features": result["features_used"],
        }

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Anomaly agent error: {exc}",
            extra={"agent_name": "anomaly_agent", "elapsed_ms": round(elapsed, 2)},
            exc_info=True,
        )
        errors = list(state.get("errors", [])) + [f"anomaly_agent error: {exc}"]
        return {
            "anomaly_score": 0.0,
            "anomaly_probability": 0.0,
            "is_anomalous": False,
            "anomaly_features": {},
            "errors": errors,
        }
