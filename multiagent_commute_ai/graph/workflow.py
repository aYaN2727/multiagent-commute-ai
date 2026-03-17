"""
graph/workflow.py
LangGraph StateGraph orchestrating the five agents.

Routing approach used: SEQUENTIAL fallback for the "both" intent.
---------------------------------------------------------------------
LangGraph >= 0.1.0 supports true parallel fan-out via Send() objects,
but this requires nodes that accept Send payloads rather than full state.
To keep all agents operating on the same shared AgentState TypedDict
(which is simpler, more maintainable, and avoids reducer conflicts),
we use sequential routing for "both":
    intent -> policy_node -> anomaly_node -> explain_node -> synth_node
For "policy_query" or "delay_claim" alone we skip the irrelevant branch.
This is the recommended approach when agents share mutable state.
---------------------------------------------------------------------
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agents.anomaly_agent import anomaly_agent
from agents.explain_agent import explain_agent
from agents.intent_agent import intent_agent
from agents.policy_agent import policy_agent
from agents.state import AgentState
from agents.synth_agent import synth_agent
from utils.logger import get_logger

logger = get_logger("graph.workflow")


# ── Routing function ──────────────────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    Conditional edge from intent_node.
    Returns the name of the next node to execute.
    """
    intent: str = state.get("intent", "policy_query")

    if intent == "policy_query":
        return "policy_node"
    if intent == "delay_claim":
        return "anomaly_node"
    if intent == "both":
        # Sequential: run policy first, then anomaly in the same pass
        # (true parallelism requires Send() which changes the node signatures)
        return "policy_node"
    # "out_of_scope" -> skip directly to synthesis
    return "synth_node"


def route_after_policy(state: AgentState) -> str:
    """
    After policy_node, decide whether to continue to anomaly (for 'both' intent)
    or jump straight to synth_node.
    """
    intent: str = state.get("intent", "policy_query")
    if intent == "both":
        return "anomaly_node"
    return "synth_node"


# ── Graph construction ────────────────────────────────────────────────────────

def build_workflow() -> Any:
    """
    Build and compile the LangGraph StateGraph.
    Returns the compiled graph app (supports .invoke() and .ainvoke()).
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("intent_node", intent_agent)
    workflow.add_node("policy_node", policy_agent)
    workflow.add_node("anomaly_node", anomaly_agent)
    workflow.add_node("explain_node", explain_agent)
    workflow.add_node("synth_node", synth_agent)

    # Entry point
    workflow.set_entry_point("intent_node")

    # Conditional routing from intent
    workflow.add_conditional_edges(
        "intent_node",
        route_after_intent,
        {
            "policy_node": "policy_node",
            "anomaly_node": "anomaly_node",
            "synth_node": "synth_node",
        },
    )

    # After policy: either go to anomaly (for 'both') or to synth
    workflow.add_conditional_edges(
        "policy_node",
        route_after_policy,
        {
            "anomaly_node": "anomaly_node",
            "synth_node": "synth_node",
        },
    )

    # Linear edges
    workflow.add_edge("anomaly_node", "explain_node")
    workflow.add_edge("explain_node", "synth_node")
    workflow.add_edge("synth_node", END)

    app = workflow.compile()
    logger.info("LangGraph workflow compiled successfully.", extra={"agent_name": "workflow"})
    return app


# ── Singleton compiled app ────────────────────────────────────────────────────
_compiled_app: Any = None


def get_compiled_app() -> Any:
    """Return the module-level singleton compiled graph."""
    global _compiled_app
    if _compiled_app is None:
        _compiled_app = build_workflow()
    return _compiled_app


# ── Helper to run the workflow ────────────────────────────────────────────────

async def run_workflow(
    query: str,
    employee_id: str,
    commute_record: Dict[str, Any] | None = None,
) -> AgentState:
    """
    Build the initial AgentState and run the compiled workflow.

    Args:
        query:          Employee's natural-language question.
        employee_id:    Employee identifier string.
        commute_record: Optional dict of commute data fields.

    Returns:
        Final AgentState after all agents have run.
    """
    if commute_record is None:
        commute_record = {}

    initial_state: AgentState = {
        # Inputs
        "user_query": query,
        "employee_id": employee_id,
        "commute_record": commute_record,
        # Intent agent outputs (defaults)
        "intent": "",
        "intent_confidence": 0.0,
        # Policy agent outputs (defaults)
        "retrieved_chunks": [],
        "source_sections": [],
        "policy_answer": "",
        "policy_confidence": 0.0,
        # Anomaly agent outputs (defaults)
        "anomaly_score": 0.0,
        "anomaly_probability": 0.0,
        "is_anomalous": False,
        "anomaly_features": {},
        # Explain agent outputs (defaults)
        "shap_values": {},
        "top_factors": [],
        "explanation_text": "",
        # Synth agent outputs (defaults)
        "final_response": "",
        "overall_confidence": 0.0,
        "needs_escalation": False,
        "escalation_reason": None,
        # Error tracking
        "errors": [],
    }

    app = get_compiled_app()
    final_state: AgentState = await app.ainvoke(initial_state)
    return final_state
