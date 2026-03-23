"""
agents/synth_agent.py
Aggregates all agent outputs into a single coherent employee-facing response.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from agents.state import AgentState
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agent.synth")

_SYNTH_SYSTEM = "You are a concise, professional HR assistant."


def _build_synth_prompt(state: "AgentState") -> str:
    """Build a focused, single-instruction prompt based on the actual state."""
    intent = state.get("intent", "policy_query")
    policy_answer = state.get("policy_answer", "")
    is_anomalous = state.get("is_anomalous", False)
    explanation_text = state.get("explanation_text", "")

    if intent == "out_of_scope":
        return (
            "An employee asked a question outside the scope of commute and travel policy. "
            "Write a single polite sentence redirecting them to hr-support@acmecorp.in."
        )

    if intent == "policy_query":
        # If the policy agent couldn't find a direct answer, check if there
        # are retrieved chunks that give partial context to help the employee.
        retrieved = state.get("retrieved_chunks", [])
        partial_context = ""
        if "not covered" in policy_answer.lower() and retrieved:
            # Include a snippet of the most relevant chunk so the LLM can
            # at least point the employee toward the right policy area.
            partial_context = (
                f"\n\nHOWEVER, here is the most relevant policy excerpt that may apply:\n"
                f"{retrieved[0][:300]}"
            )
        return (
            f"Rewrite the following HR policy answer as a clear, friendly reply in 2-3 sentences. "
            f"If the answer says 'not covered' but a related policy excerpt is provided, "
            f"mention the related policy briefly before suggesting HR contact. "
            f"Do not add information not present below. Do not repeat instructions.\n\n"
            f"POLICY ANSWER:\n{policy_answer}{partial_context}"
        )

    if intent == "delay_claim":
        if is_anomalous:
            return (
                f"An employee submitted a delay claim that has been flagged for review. "
                f"Write a short empathetic reply (2-3 sentences) telling them:\n"
                f"1. Their claim is under manual review.\n"
                f"2. The reason: {explanation_text}\n"
                f"3. They should submit supporting documents to hr-support@acmecorp.in."
            )
        else:
            return (
                f"An employee submitted a delay claim that looks normal (no anomaly detected). "
                f"Write a short friendly reply (2-3 sentences) confirming their claim is being processed "
                f"and telling them to submit any required documents via the TravelDesk portal."
            )

    # intent == "both"
    if is_anomalous:
        return (
            f"An employee has a policy question and a flagged delay claim. "
            f"Write a clear reply in 3-4 sentences covering:\n"
            f"1. The policy answer: {policy_answer}\n"
            f"2. Their claim is under manual review because: {explanation_text}\n"
            f"3. Next step: submit evidence to hr-support@acmecorp.in."
        )
    return (
        f"An employee has a policy question and a delay claim (no anomaly detected). "
        f"Write a clear reply in 3 sentences covering:\n"
        f"1. The policy answer: {policy_answer}\n"
        f"2. Their claim is being processed normally.\n"
        f"3. Next step: submit documents via TravelDesk portal."
    )


async def synth_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: synthesise all outputs into the final employee response.
    Returns: final_response, overall_confidence, needs_escalation, escalation_reason.
    """
    t0 = time.perf_counter()
    logger.info("Synth agent starting", extra={"agent_name": "synth_agent"})

    try:
        errors: List[str] = state.get("errors", [])
        prompt = _build_synth_prompt(state)

        llm = get_llm_client()
        raw: str = await llm.complete_chat(_SYNTH_SYSTEM, prompt)

        final_response = raw.strip()

        # Strip common LLM preambles that leak into the response
        _preambles = (
            "here's a short friendly reply:",
            "here's a friendly reply:",
            "here's a reply:",
            "here is a reply:",
            "here's my reply:",
            "sure!", "sure,", "of course!",
            "dear [employee's name],",
            "dear [employee name],",
        )
        lower = final_response.lower()
        for pre in _preambles:
            if lower.startswith(pre):
                final_response = final_response[len(pre):].strip().strip('"').strip()
                break
        # Also strip surrounding quotes the LLM sometimes adds
        if final_response.startswith('"') and final_response.endswith('"'):
            final_response = final_response[1:-1].strip()

        # If LLM returned nothing, fall back to the policy answer directly
        if not final_response:
            policy_ans = state.get("policy_answer", "")
            final_response = policy_ans if policy_ans else (
                "I was unable to generate a full response. "
                "Please contact hr-support@acmecorp.in for assistance."
            )

        # Derive escalation from state (no longer parsed from LLM JSON)
        needs_esc: bool = bool(state.get("needs_escalation", False))
        policy_conf: float = float(state.get("policy_confidence", 0.5))
        esc_reason: Optional[str] = state.get("escalation_reason")

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Synth agent done",
            extra={"agent_name": "synth_agent", "elapsed_ms": round(elapsed, 2)},
        )

        return {
            "final_response": final_response,
            "overall_confidence": policy_conf,
            "needs_escalation": needs_esc,
            "escalation_reason": esc_reason,
        }

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Synth agent error: {exc}",
            extra={"agent_name": "synth_agent", "elapsed_ms": round(elapsed, 2)},
            exc_info=True,
        )
        errors = list(state.get("errors", [])) + [f"synth_agent error: {exc}"]
        return {
            "final_response": (
                "We encountered an issue processing your request. "
                "Please contact hr-support@acmecorp.in for assistance."
            ),
            "overall_confidence": 0.0,
            "needs_escalation": True,
            "escalation_reason": f"Synthesis failed: {exc}",
            "errors": errors,  # type: ignore[typeddict-item]
        }
