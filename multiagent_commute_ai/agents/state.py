"""
agents/state.py
Central TypedDict that carries all state across the LangGraph workflow.
"""
from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    # ── Input fields (set before graph invocation) ──────────────────────────
    user_query: str
    employee_id: str
    commute_record: Dict[str, Any]   # Can be empty dict {}

    # ── Set by Intent Agent ──────────────────────────────────────────────────
    intent: str              # "policy_query" | "delay_claim" | "both" | "out_of_scope"
    intent_confidence: float

    # ── Set by Policy Agent ──────────────────────────────────────────────────
    retrieved_chunks: List[str]
    source_sections: List[str]
    policy_answer: str
    policy_confidence: float

    # ── Set by Anomaly Agent ─────────────────────────────────────────────────
    anomaly_score: float            # Raw IF score (negative = more anomalous)
    anomaly_probability: float      # Normalised 0-1 (higher = more suspicious)
    is_anomalous: bool
    anomaly_features: Dict[str, float]  # Feature values used in scoring

    # ── Set by Explain Agent ─────────────────────────────────────────────────
    shap_values: Dict[str, float]
    top_factors: List[str]          # Human-readable factor strings
    explanation_text: str           # LLM-generated natural language explanation

    # ── Set by Synth Agent ──────────────────────────────────────────────────
    final_response: str
    overall_confidence: float
    needs_escalation: bool
    escalation_reason: Optional[str]

    # ── Error tracking ───────────────────────────────────────────────────────
    errors: List[str]               # Any agent errors logged here
