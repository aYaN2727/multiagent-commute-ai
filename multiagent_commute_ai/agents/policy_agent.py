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
You are a precise HR policy assistant. Answer the employee's question using ONLY the policy excerpts below.

RULES:
1. Answer clearly in 2-4 sentences using only the excerpts.
2. If the answer is NOT in the excerpts, say: "This is not covered in the current policy. Please contact hr-support@acmecorp.in."
3. Never invent amounts, rules, or eligibility criteria not present in the excerpts.
4. End your answer with: SOURCE: <section name>

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
    Build an enriched RAG search query by combining the current message
    with keywords extracted from the last user turn in the conversation.
    This ensures follow-up questions (e.g. "but he took the wrong route")
    retrieve relevant chunks even when the message lacks policy keywords.
    """
    current = state["user_query"]
    history = state.get("conversation_history") or []

    # Find the last user message in history (i.e. the previous turn)
    prev_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"),
        None,
    )
    if prev_user and len(current.split()) < 10:
        # Short follow-up: prepend previous query for semantic richness
        return f"{prev_user} {current}"
    return current


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

        confidence: float = 0.8 if answer and "not covered" not in answer.lower() else 0.3
        needs_esc: bool = confidence < 0.4

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
