"""
agents/intent_agent.py
Classifies the user query into one of 4 intents using an LLM.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

from agents.state import AgentState
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agent.intent")

_SYSTEM_PROMPT = """
You are an HR query classifier. Classify the employee's message into exactly one intent.

INTENTS:
- "policy_query": Asking about commute/travel rules, eligibility, procedures, complaints, driver behaviour, billing, emergency procedures, or definitions. Use this for questions AND for reporting issues with drivers or vehicles.
- "delay_claim": Reporting a specific commute/travel delay event and wanting reimbursement for it.
- "both": The message contains BOTH a delay reimbursement claim AND a policy question.
- "out_of_scope": Completely unrelated to commute, travel, or HR policies (e.g. salary, food, IT, leaves).

Respond with ONLY this JSON (no other text):
{"intent": "<intent>", "confidence": <0.0-1.0>}

Examples:
Q: "What is the maximum cab reimbursement per day?"
A: {"intent": "policy_query", "confidence": 0.97}

Q: "Who is eligible for commute transport?"
A: {"intent": "policy_query", "confidence": 0.96}

Q: "I want to register a complaint against a driver"
A: {"intent": "policy_query", "confidence": 0.93}

Q: "The driver was misbehaving with me"
A: {"intent": "policy_query", "confidence": 0.92}

Q: "How does the panic button work?"
A: {"intent": "policy_query", "confidence": 0.95}

Q: "My metro was 45 minutes late today, can I claim cab fare?"
A: {"intent": "delay_claim", "confidence": 0.95}

Q: "My bus was 2 hours late. What is the reimbursement rule?"
A: {"intent": "both", "confidence": 0.92}

Q: "What time does the cafeteria open?"
A: {"intent": "out_of_scope", "confidence": 0.95}

Q: "Can you help me with my salary slip?"
A: {"intent": "out_of_scope", "confidence": 0.97}
""".strip()

_VALID_INTENTS = {"policy_query", "delay_claim", "both", "out_of_scope"}


async def intent_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph async node: classify the user query.
    Returns updated fields: intent, intent_confidence.
    """
    t0 = time.perf_counter()
    logger.info("Intent agent starting", extra={"agent_name": "intent_agent"})

    try:
        llm = get_llm_client()
        raw: str = await llm.complete_chat(_SYSTEM_PROMPT, state["user_query"])

        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])

        # Try direct parse first; fallback to regex extraction if LLM added extra text
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            import re as _re
            m = _re.search(r'\{[^{}]*"intent"\s*:\s*"(\w+)"[^{}]*"confidence"\s*:\s*([\d.]+)[^{}]*\}', cleaned)
            if m:
                parsed = {"intent": m.group(1), "confidence": float(m.group(2))}
            else:
                raise
        intent: str = parsed.get("intent", "policy_query")
        confidence: float = float(parsed.get("confidence", 0.5))

        if intent not in _VALID_INTENTS:
            logger.warning(
                f"Unknown intent '{intent}', defaulting to policy_query",
                extra={"agent_name": "intent_agent"},
            )
            intent = "policy_query"
            confidence = 0.5

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Intent agent error: {exc}",
            extra={"agent_name": "intent_agent", "elapsed_ms": round(elapsed, 2)},
        )
        intent = "policy_query"
        confidence = 0.5
        errors = list(state.get("errors", [])) + [f"intent_agent error: {exc}"]
        return {
            "intent": intent,
            "intent_confidence": confidence,
            "errors": errors,
        }

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"Intent classified -> '{intent}' ({confidence:.2f})",
        extra={"agent_name": "intent_agent", "elapsed_ms": round(elapsed, 2)},
    )

    return {
        "intent": intent,
        "intent_confidence": confidence,
    }
