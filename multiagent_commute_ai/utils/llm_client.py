"""
utils/llm_client.py
Async LLM client wrapper supporting OpenAI, Ollama, and mock mode.

Provider selection via LLM_PROVIDER in .env:
  LLM_PROVIDER=openai   -> uses OPENAI_API_KEY + LLM_MODEL
  LLM_PROVIDER=ollama   -> uses OLLAMA_BASE_URL + OLLAMA_MODEL (no API key needed)
  LLM_MOCK_MODE=true    -> skips all LLM calls (for testing)
"""
import asyncio
import json
import time
from typing import Any, Dict, Optional, Type, TypeVar

import httpx
from openai import AsyncOpenAI, APIError, RateLimitError
from pydantic import BaseModel

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("llm_client")
settings = get_settings()

T = TypeVar("T", bound=BaseModel)

# ── Mock responses keyed by prompt keywords ──────────────────────────────────
MOCK_RESPONSES: Dict[str, str] = {
    "classify": json.dumps({"intent": "policy_query", "confidence": 0.85}),
    "policy assistant": json.dumps({
        "answer": (
            "Based on the policy, the maximum hotel accommodation limit per night "
            "in Tier 1 cities (Mumbai, Delhi NCR, Bengaluru, Chennai, Hyderabad, "
            "Kolkata, Pune) is INR 5,000 per night inclusive of all taxes. "
            "[Source: sample_travel_policy.txt, Section 4.4]"
        ),
        "source_sections": ["Section 4.4"],
        "confidence": 0.92,
        "needs_escalation": False,
    }),
    "synthesize": json.dumps({
        "final_response": (
            "Thank you for your query. Based on the current travel and commute policy, "
            "your question has been answered above. Please review the relevant sections "
            "and reach out to hr-support@acmecorp.in if you need further assistance."
        ),
        "overall_confidence": 0.88,
        "needs_escalation": False,
        "escalation_reason": None,
    }),
    "explanation": (
        "Your delay claim has been flagged for manual review by our automated system. "
        "The primary factors that triggered the review are the reported delay duration "
        "and the claim frequency over the past month. Please submit your transport "
        "authority delay certificate and cab receipt via the TravelDesk portal to "
        "expedite the review process."
    ),
}


def _pick_mock_response(system_prompt: str, user_message: str) -> str:
    """Select the most appropriate mock response based on prompt content."""
    sys_lower = system_prompt.lower()
    user_lower = user_message.lower()
    combined = sys_lower + " " + user_lower

    # Intent classification: check only the USER message for domain keywords
    if "classify" in sys_lower or ("intent" in sys_lower and "confidence" in sys_lower):
        if ("delay" in user_lower or "late" in user_lower) and (
            "policy" in user_lower or "rule" in user_lower or "what" in user_lower
        ):
            return json.dumps({"intent": "both", "confidence": 0.90})
        if "delay" in user_lower or "late" in user_lower or "claim" in user_lower or "hours" in user_lower:
            return json.dumps({"intent": "delay_claim", "confidence": 0.93})
        if "wifi" in user_lower or "cafeteria" in user_lower or "salary" in user_lower or "password" in user_lower:
            return json.dumps({"intent": "out_of_scope", "confidence": 0.91})
        return json.dumps({"intent": "policy_query", "confidence": 0.88})

    if "synthesize" in combined or "synthesise" in combined or "aggregate" in combined:
        return MOCK_RESPONSES["synthesize"]
    # Only return explanation mock when the system/user prompt is explicitly about
    # anomaly detection explanation (not when "flagged" appears in retrieved policy text)
    if "anomaly detection system" in combined or (
        "empathetic" in sys_lower and "flagged" in combined
    ):
        return MOCK_RESPONSES["explanation"]
    return MOCK_RESPONSES["policy assistant"]


class LLMClient:
    """
    Async LLM wrapper with:
      - OpenAI chat completions
      - Ollama (OpenAI-compatible local API)
      - JSON-structured outputs via Pydantic
      - Retry logic (3 attempts, exponential backoff)
      - Mock mode for testing without any LLM
    """

    def __init__(self) -> None:
        self._mock: bool = settings.LLM_MOCK_MODE
        self._provider: str = settings.LLM_PROVIDER.lower()
        self._temperature: float = settings.LLM_TEMPERATURE
        self._max_tokens: int = settings.LLM_MAX_TOKENS

        if not self._mock:
            if self._provider == "ollama":
                # Ollama exposes an OpenAI-compatible API; "ollama" is a dummy key
                self._client: AsyncOpenAI = AsyncOpenAI(
                    base_url=settings.OLLAMA_BASE_URL,
                    api_key="ollama",
                )
                self._model: str = settings.OLLAMA_MODEL
                logger.info(
                    f"LLM provider: Ollama | model: {self._model} | base: {settings.OLLAMA_BASE_URL}",
                    extra={"agent_name": "llm_client"},
                )
            else:
                self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
                self._model = settings.LLM_MODEL
                logger.info(
                    f"LLM provider: OpenAI | model: {self._model}",
                    extra={"agent_name": "llm_client"},
                )

    async def complete_chat(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """
        Sends a chat completion request and returns the raw text response.
        Falls back to a safe default string after 3 failed retries.
        """
        if self._mock:
            logger.debug("Mock LLM returning canned response", extra={"agent_name": "llm_client"})
            await asyncio.sleep(0.01)  # simulate tiny latency
            return _pick_mock_response(system_prompt, user_message)

        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            t0 = time.perf_counter()
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                elapsed = (time.perf_counter() - t0) * 1000
                usage = resp.usage
                logger.info(
                    "LLM call completed",
                    extra={
                        "agent_name": "llm_client",
                        "model": self._model,
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "latency_ms": round(elapsed, 2),
                        "attempt": attempt,
                    },
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except (RateLimitError, APIError, httpx.ConnectError, httpx.HTTPError) as exc:
                last_error = exc
                wait = 2 ** attempt  # 2, 4, 8 seconds
                logger.warning(
                    f"LLM API error (attempt {attempt}/3), retrying in {wait}s",
                    extra={"agent_name": "llm_client", "error": str(exc)},
                )
                await asyncio.sleep(wait)

        logger.error(
            "LLM failed after 3 attempts, returning safe default",
            extra={"agent_name": "llm_client", "error": str(last_error)},
        )
        return json.dumps({
            "error": "LLM unavailable",
            "message": "Unable to process request. Please try again later.",
        })

    async def complete_structured(
        self,
        system_prompt: str,
        user_message: str,
        output_schema: Type[T],
    ) -> T:
        """
        Calls complete_chat and parses the JSON response into a Pydantic model.
        Falls back to a zeroed-out default model on parse failure.
        """
        raw = await self.complete_chat(system_prompt, user_message)
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            data = json.loads(cleaned)
            return output_schema.model_validate(data)
        except Exception as exc:
            logger.error(
                "Failed to parse structured LLM response",
                extra={"agent_name": "llm_client", "error": str(exc), "raw": raw[:200]},
            )
            # Return a default-constructed model (all fields at their defaults)
            return output_schema.model_validate({})


# Module-level singleton
_llm_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Return the module-level singleton LLMClient."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMClient()
    return _llm_instance
