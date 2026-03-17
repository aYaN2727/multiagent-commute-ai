"""
agents/explain_agent.py
SHAP computation + LLM-generated employee-facing explanation.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from agents.state import AgentState
from ml.inference import get_inference
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agent.explain")

_EXPLANATION_SYSTEM = "You are a professional and empathetic HR communication assistant."

_EXPLANATION_PROMPT_TEMPLATE = """
You are an HR system assistant. An employee's delay claim has been automatically
flagged for review by our anomaly detection system.

The top factors that triggered the flag are:
{top_factors}

Write a SHORT, empathetic explanation (2-4 sentences) for the employee that:
1. Acknowledges their claim has been flagged for manual review
2. Mentions the specific data factors that triggered the flag (in plain English)
3. Explains they can submit supporting documents (transport authority notice, screenshot)
4. Does NOT accuse them of fraud or use harsh language

Respond with ONLY the explanation text, no JSON, no preamble.
""".strip()


async def explain_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: compute SHAP values and generate a plain-language explanation.
    Returns: shap_values, top_factors, explanation_text.
    """
    t0 = time.perf_counter()
    logger.info("Explain agent starting", extra={"agent_name": "explain_agent"})

    commute_record: Dict[str, Any] = state.get("commute_record", {})
    anomaly_features: Dict[str, float] = state.get("anomaly_features", {})

    if not commute_record or not anomaly_features:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "No commute data to explain",
            extra={"agent_name": "explain_agent", "elapsed_ms": round(elapsed, 2)},
        )
        return {
            "shap_values": {},
            "top_factors": [],
            "explanation_text": "No commute data to explain.",
        }

    try:
        inference = get_inference()
        shap_result = await asyncio.to_thread(inference.explain, commute_record)

        shap_values: Dict[str, float] = shap_result["shap_values"]
        top_3: List[str] = shap_result["top_3_factors"]

        # Build LLM prompt
        factors_text = "\n".join(f"  - {f}" for f in top_3)
        prompt = _EXPLANATION_PROMPT_TEMPLATE.format(top_factors=factors_text)

        llm = get_llm_client()
        explanation: str = await llm.complete_chat(_EXPLANATION_SYSTEM, prompt)
        explanation = explanation.strip()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Explain agent done",
            extra={"agent_name": "explain_agent", "elapsed_ms": round(elapsed, 2)},
        )

        return {
            "shap_values": shap_values,
            "top_factors": top_3,
            "explanation_text": explanation,
        }

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Explain agent error: {exc}",
            extra={"agent_name": "explain_agent", "elapsed_ms": round(elapsed, 2)},
            exc_info=True,
        )
        errors = list(state.get("errors", [])) + [f"explain_agent error: {exc}"]
        return {
            "shap_values": {},
            "top_factors": [],
            "explanation_text": (
                "Your claim has been flagged for manual review. "
                "Please submit supporting documentation to hr-support@acmecorp.in."
            ),
            "errors": errors,  # type: ignore[typeddict-item]
        }
