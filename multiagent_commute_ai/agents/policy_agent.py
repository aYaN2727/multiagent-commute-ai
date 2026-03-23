"""
agents/policy_agent.py
RAG-based policy retrieval + grounded answer generation.

Follow-up handling — permanent solution
----------------------------------------
Uses LLM Query Rewriting (Conversational RAG pattern):

  User turn 1: "What happens if the driver deviates from route?"
  Bot turn 1 : "Any detour beyond 500 meters triggers an automatic alert..."
  User turn 2: "but the driver dropped me at the wrong address"
                           │
               ┌───────────▼────────────┐
               │   LLM Query Rewriter   │  (fast, targeted prompt)
               └───────────┬────────────┘
                           │
               "What is the policy when a driver drops an employee
                at the wrong address or deviates from the approved
                geofenced route?"
                           │
               ┌───────────▼────────────┐
               │   FAISS Semantic Search │  always finds right chunks
               └────────────────────────┘

This is robust because:
  - Resolves pronouns and references ("he", "it", "the driver")
  - Converts casual language to policy terminology
  - Works for ANY follow-up regardless of wording or length
  - Falls back gracefully if the rewrite LLM call fails
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

# ── Prompts ───────────────────────────────────────────────────────────────────

_REWRITE_SYSTEM = (
    "You are a search query rewriter for an HR policy assistant. "
    "Your only job is to convert a follow-up message into a complete, "
    "standalone HR policy search query. "
    "Output ONLY the rewritten query — no explanation, no punctuation changes, "
    "no extra text."
)

_REWRITE_USER_TEMPLATE = """\
Conversation so far:
{history}

Follow-up message: {current}

Rewrite the follow-up as a complete, standalone HR policy question that \
captures the full context from the conversation. Use policy terminology \
(e.g. route deviation, geofence, reimbursement, eligibility)."""


_ANSWER_SYSTEM_TEMPLATE = """\
You are a helpful HR policy assistant. Answer the employee's question \
using the policy excerpts below.

RULES:
1. Answer clearly in 2-4 sentences.
2. If the question is RELATED to a topic in the excerpts (even if not \
word-for-word), apply the relevant policy clause.
   Examples: "dropped at wrong address" -> route deviation / geofence policy;
             "driver was rude" -> complaint procedures.
3. Only say "This is not covered in the current policy. \
Please contact hr-support@acmecorp.in." if the topic is COMPLETELY unrelated \
to ALL excerpts (e.g. salary, food, IT issues).
4. Never invent amounts, rules, or eligibility criteria not in the excerpts.
5. IMPORTANT: State what the policy actually says — do NOT just say \
"refer to Section X" or "see policy B3". Quote or paraphrase the actual rule.
6. End your answer with: SOURCE: <section name>

POLICY EXCERPTS:
CONTEXT_PLACEHOLDER

QUESTION_PLACEHOLDER

Answer:"""

# Regex to detect monetary amounts — only explicit INR/₹ prefixed figures
_MONEY_RE = re.compile(r"INR\s*[\d,]+|₹\s*[\d,]+")


# ── Helper: detect fabricated amounts ────────────────────────────────────────

def _contains_fabricated_amount(answer: str, chunks: List[Dict[str, Any]]) -> bool:
    """Return True if the answer contains an INR figure absent from all chunks."""
    answer_amounts = set(_MONEY_RE.findall(answer))
    if not answer_amounts:
        return False
    chunk_text_all = " ".join(c["text"] for c in chunks)
    for amount in answer_amounts:
        digits_only = re.sub(r"[^\d]", "", amount)
        if digits_only and digits_only not in chunk_text_all:
            logger.warning(
                f"Possibly fabricated amount '{amount}' not in retrieved chunks",
                extra={"agent_name": "policy_agent"},
            )
            return True
    return False


# ── LLM Query Rewriter ────────────────────────────────────────────────────────

async def _rewrite_query(
    current: str,
    history: List[Dict[str, str]],
    llm: Any,
) -> str:
    """
    Use the LLM to rewrite a follow-up message as a complete standalone query.

    Returns the rewritten query, or the original `current` if:
      - There is no history (already standalone)
      - The current message is long enough to be self-contained (>= 15 words)
      - The rewrite LLM call fails for any reason
    """
    if not history or len(current.split()) >= 15:
        return current

    # Build a compact history block — last 4 messages (2 turns) is enough
    recent = history[-4:]
    history_lines = "\n".join(
        f"{'Employee' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )

    user_msg = _REWRITE_USER_TEMPLATE.format(
        history=history_lines,
        current=current,
    )

    try:
        rewritten = await llm.complete_chat(_REWRITE_SYSTEM, user_msg)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) > 5:
            logger.info(
                f"Query rewritten: '{current}' -> '{rewritten}'",
                extra={"agent_name": "policy_agent"},
            )
            return rewritten
    except Exception as exc:
        logger.warning(
            f"Query rewrite failed ({exc}), using original query",
            extra={"agent_name": "policy_agent"},
        )

    # Fallback: concatenate history keywords with current message
    context_parts = [m["content"] for m in recent]
    return " ".join(context_parts) + " " + current


# ── Build answer prompt ───────────────────────────────────────────────────────

def _build_answer_prompt(
    context: str,
    original_query: str,
    history: List[Dict[str, str]],
) -> str:
    """Build the policy answer prompt with optional conversation context."""
    recent = history[-4:] if len(history) > 4 else history

    if recent:
        history_block = "\n".join(
            f"{'Employee' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )
        question_block = (
            f"CONVERSATION SO FAR:\n{history_block}\n\n"
            f"EMPLOYEE FOLLOW-UP: {original_query}"
        )
    else:
        question_block = f"EMPLOYEE QUESTION: {original_query}"

    # Use .replace() instead of .format() so curly braces in PDF chunk text
    # (e.g. "{route_id}") never crash with a KeyError.
    return (
        _ANSWER_SYSTEM_TEMPLATE
        .replace("CONTEXT_PLACEHOLDER", context)
        .replace("QUESTION_PLACEHOLDER", question_block)
    )


# ── Main agent node ───────────────────────────────────────────────────────────

async def policy_agent(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph node: rewrite query -> retrieve policy chunks -> generate answer.
    Returns: retrieved_chunks, source_sections, policy_answer, policy_confidence,
             needs_escalation.
    """
    t0 = time.perf_counter()
    logger.info("Policy agent starting", extra={"agent_name": "policy_agent"})

    default_return: Dict[str, Any] = {
        "retrieved_chunks": [],
        "source_sections": [],
        "policy_answer": "Unable to retrieve policy information at this time.",
        "policy_confidence": 0.0,
        "needs_escalation": True,
    }

    try:
        llm = get_llm_client()
        history = state.get("conversation_history") or []

        # ── Step 1: Rewrite the query for RAG retrieval ───────────────────
        search_query = await _rewrite_query(state["user_query"], history, llm)

        # ── Step 2: Retrieve relevant policy chunks ───────────────────────
        retriever = get_retriever()
        chunks = retriever.retrieve(search_query, top_k=7)
        context = retriever.format_context(chunks)

        # ── Step 3: Generate a grounded answer ────────────────────────────
        prompt = _build_answer_prompt(context, state["user_query"], history)
        raw: str = await llm.complete_chat("", prompt)
        answer = raw.strip()

        # Extract "SOURCE: ..." line
        source_sections: List[str] = []
        body_lines: List[str] = []
        for line in answer.splitlines():
            if line.strip().upper().startswith("SOURCE:"):
                src = line.split(":", 1)[-1].strip()
                if src:
                    source_sections.append(src)
            else:
                body_lines.append(line)
        if not source_sections and chunks:
            source_sections = [chunks[0].get("section", "Policy")]
        answer = "\n".join(body_lines).strip() or answer

        # ── Step 4: Confidence + escalation ──────────────────────────────
        not_covered = "not covered" in answer.lower()
        no_answer   = not answer or len(answer.strip()) < 20

        # 0.8 = grounded answer found
        # 0.5 = LLM said "not covered" but gave contact info
        # 0.0 = no answer at all
        confidence: float = 0.0 if no_answer else (0.5 if not_covered else 0.8)

        # Escalate ONLY when no answer was produced — not on "not covered"
        # ("not covered" replies already contain the HR contact email)
        needs_esc: bool = no_answer

        if _contains_fabricated_amount(answer, chunks):
            needs_esc = True

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"Policy agent done — confidence={confidence:.2f} "
            f"search='{search_query[:60]}'",
            extra={"agent_name": "policy_agent", "elapsed_ms": round(elapsed, 2)},
        )

        return {
            "retrieved_chunks": [c["text"] for c in chunks],
            "source_sections":  source_sections,
            "policy_answer":    answer,
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
