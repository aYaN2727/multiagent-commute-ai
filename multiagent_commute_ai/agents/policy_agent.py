"""
agents/policy_agent.py
RAG-based policy retrieval + grounded answer generation.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from agents.state import AgentState
from rag.retriever import get_retriever
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("agent.policy")

_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful HR policy assistant. Answer the employee's question using the policy excerpts below.

RULES:
1. Answer clearly in 2-4 sentences.
2. If the question is RELATED to a topic in the excerpts (even if not word-for-word), apply the relevant policy clause. For example: "dropped at wrong address" relates to route deviation / geofence policy; "driver was rude" relates to complaint procedures.
3. Only say "This is not covered in the current policy. Please contact hr-support@acmecorp.in." if the topic is COMPLETELY unrelated to ALL excerpts (e.g. salary, food, IT issues).
4. Never invent amounts, rules, or eligibility criteria not present in the excerpts.
5. End your answer with: SOURCE: <section name>

POLICY EXCERPTS:
{context}

EMPLOYEE QUESTION: {query}

Answer:""".strip()

# Regex to detect monetary amounts — only explicit INR/₹ prefixed figures
_MONEY_RE = re.compile(r"INR\s*[\d,]+|₹\s*[\d,]+")


def _contains_fabricated_amount(answer: str, chunks: List[Dict[str, Any]]) -> bool:
    """
    Basic heuristic: if the answer contains an INR figure that does NOT appear
    in any retrieved chunk, flag it for escalation.
    """
    answer_amounts = set(_MONEY_RE.findall(answer))
    if not answer_amounts:
        return False
    chunk_text_all = " ".join(c["text"] for c in chunks)
    for amount in answer_amounts:
        digits_only = re.sub(r"[^\d]", "", amount)
        if digits_only and digits_only not in chunk_text_all:
            logger.warning(
                f"Possibly fabricated amount '{amount}' not found in retrieved chunks",
                extra={"agent_name": "policy_agent"},
            )
            return True
    return False


def _build_search_query(state: AgentState) -> str:
    """
    Build an enriched RAG search query for follow-up messages.

    Key insight: the PREVIOUS ASSISTANT RESPONSE already contains domain
    keywords extracted from policy chunks (e.g. "500 meters", "geofence",
    "automatic alert"). Prepending it to the current follow-up dramatically
    improves semantic similarity to the correct policy chunks.

    Strategy (only applied when current message is short / a follow-up):
        enriched = prev_user_msg + prev_assistant_msg + current_msg
    """
    current = state["user_query"]
    history = state.get("conversation_history") or []

    # Only enrich short follow-ups — standalone questions search as-is
    if not history or len(current.split()) >= 15:
        return current

    # Collect last 2 turns (up to 4 messages) in chronological order
    recent = history[-4:]
    context_parts = [m["content"] for m in recent]
    return " ".join(context_parts) + " " + current


def _build_policy_prompt_with_history(
    context: str,
    query: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Build the policy answer prompt, optionally including recent conversation
    so the LLM can give a contextually coherent answer on follow-ups.
    """
    recent = history[-4:] if len(history) > 4 else history

    if not recent:
        return _SYSTEM_PROMPT_TEMPLATE.format(context=context, query=query)

    history_block = "\n".join(
        f"{'Employee' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )
    return (
        _SYSTEM_PROMPT_TEMPLATE.replace(
            "EMPLOYEE QUESTION: {query}",
            f"CONVERSATION SO FAR:\n{history_block}\n\nEMPLOYEE FOLLOW-UP: {{query}}",
        ).format(context=context, query=query)
    )


async def policy_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: retrieve policy chunks and generate a grounded answer.
    Returns: retrieved_chunks, source_sections, policy_answer, policy_confidence,
             needs_escalation.
    """
    t0 = time.perf_counter()
    logger.info("Policy agent starting", extra={"agent_name": "policy_agent"})

    default_return = {
        "retrieved_chunks": [],
        "source_sections": [],
        "policy_answer": "Unable to retrieve policy information at this time.",
        "policy_confidence": 0.0,
        "needs_escalation": True,
    }

    try:
        retriever = get_retriever()
        # Use enriched query for retrieval so follow-ups find the right chunks.
        # Use top_k=7 to cast a wider net — the LLM filters for relevance anyway.
        search_query = _build_search_query(state)
        chunks = retriever.retrieve(search_query, top_k=7)
        context = retriever.format_context(chunks)

        history = state.get("conversation_history") or []
        prompt = _build_policy_prompt_with_history(context, state["user_query"], history)

        llm = get_llm_client()
        raw: str = await llm.complete_chat("", prompt)

        answer = raw.strip()

        # Extract source sections from "SOURCE: ..." line if present
        source_sections: List[str] = []
        lines = answer.splitlines()
        body_lines = []
        for line in lines:
            if line.strip().upper().startswith("SOURCE:"):
                src = line.split(":", 1)[-1].strip()
                if src:
                    source_sections.append(src)
            else:
                body_lines.append(line)
        # Use estimated_section from the top retrieved chunk as fallback
        if not source_sections and chunks:
            source_sections = [chunks[0].get("estimated_section", "Policy")]
        answer = "\n".join(body_lines).strip() or answer

        not_covered = "not covered" in answer.lower()
        no_answer    = not answer or len(answer.strip()) < 20

        # Confidence tiers:
        #   0.8 — answer found and grounded in excerpts
        #   0.5 — LLM said "not covered" but still gave a contact/redirect
        #   0.0 — no answer generated at all (exception / empty)
        confidence: float = 0.0 if no_answer else (0.5 if not_covered else 0.8)

        # Only escalate when there is genuinely NO answer.
        # "Not covered" responses already include the HR email — that is
        # sufficient guidance; don't stack an extra escalation badge on top.
        needs_esc: bool = no_answer

        # Validate: if any INR amount in the answer is not in the chunks -> escalate
        if _contains_fabricated_amount(answer, chunks):
            needs_esc = True

        chunk_texts = [c["text"] for c in chunks]
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Policy agent done — confidence={confidence:.2f}",
            extra={"agent_name": "policy_agent", "elapsed_ms": round(elapsed, 2)},
        )

        return {
            "retrieved_chunks": chunk_texts,
            "source_sections": source_sections,
            "policy_answer": answer,
            "policy_confidence": confidence,
            "needs_escalation": needs_esc,
        }

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.error(
            f"Policy agent error: {exc}",
            extra={"agent_name": "policy_agent", "elapsed_ms": round(elapsed, 2)},
            exc_info=True,
        )
        errors = list(state.get("errors", [])) + [f"policy_agent error: {exc}"]
        result = dict(default_return)
        result["errors"] = errors  # type: ignore[assignment]
        return result
