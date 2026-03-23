"""
main.py
FastAPI application entry point for the Policy-Aware Multi-Agent GenAI System.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings
from graph.workflow import get_compiled_app, run_workflow
from schemas.api_schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from utils.logger import get_logger

logger = get_logger("main")
settings = get_settings()

# ── Model-loaded flags ────────────────────────────────────────────────────────
_models_loaded: Dict[str, bool] = {
    "isolation_forest": False,
    "shap_explainer": False,
    "faiss_index": False,
    "langgraph_workflow": False,
    "llm_client": False,
}


# ── Lifespan context manager ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load all heavy singletons once at startup."""
    logger.info("Starting up — loading components…", extra={"agent_name": "startup"})

    # 1. LLM client
    try:
        from utils.llm_client import get_llm_client
        app.state.llm_client = get_llm_client()
        _models_loaded["llm_client"] = True
        logger.info("LLM client loaded.", extra={"agent_name": "startup"})
    except Exception as exc:
        logger.error(f"LLM client failed: {exc}", extra={"agent_name": "startup"})

    # 2. PolicyRetriever (loads FAISS index)
    try:
        from rag.retriever import get_retriever
        app.state.retriever = get_retriever()
        _models_loaded["faiss_index"] = True
        logger.info("FAISS index and PolicyRetriever loaded.", extra={"agent_name": "startup"})
    except Exception as exc:
        logger.error(f"PolicyRetriever failed: {exc}", extra={"agent_name": "startup"})

    # 3. IsolationForestInference
    try:
        from ml.inference import get_inference
        app.state.inference = get_inference()
        _models_loaded["isolation_forest"] = True
        _models_loaded["shap_explainer"] = True
        logger.info("IsolationForest + SHAP loaded.", extra={"agent_name": "startup"})
    except Exception as exc:
        logger.error(f"IsolationForestInference failed: {exc}", extra={"agent_name": "startup"})

    # 4. LangGraph compiled workflow
    try:
        app.state.workflow = get_compiled_app()
        _models_loaded["langgraph_workflow"] = True
        logger.info("LangGraph workflow compiled and cached.", extra={"agent_name": "startup"})
    except Exception as exc:
        logger.error(f"LangGraph workflow failed: {exc}", extra={"agent_name": "startup"})

    logger.info(
        f"Startup complete. Models loaded: {_models_loaded}",
        extra={"agent_name": "startup"},
    )

    yield

    # Shutdown
    logger.info("Shutting down API server.", extra={"agent_name": "startup"})


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Policy-Aware Multi-Agent GenAI API",
    description=(
        "Intelligent HR support system: RAG-based policy Q&A, "
        "Isolation Forest anomaly detection, and SHAP explanations — "
        "orchestrated by LangGraph."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS (open for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Request logging middleware ────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code}",
        extra={
            "agent_name": "middleware",
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
        },
    )
    return response


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/chat", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return FileResponse("static/chat.html")


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    return {
        "message": "Policy-Aware Multi-Agent GenAI API",
        "chat": "/chat",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    all_ok = all(_models_loaded.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        models_loaded=_models_loaded,
        version="1.0.0",
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main endpoint: accepts an employee query (+ optional commute record),
    runs the full LangGraph multi-agent pipeline, and returns structured output.
    """
    try:
        commute_dict: Dict[str, Any] = (
            request.commute_record.model_dump() if request.commute_record else {}
        )

        history = [m.model_dump() for m in request.conversation_history]

        final_state = await run_workflow(
            query=request.query,
            employee_id=request.employee_id,
            commute_record=commute_dict,
            conversation_history=history,
        )

        return QueryResponse(
            employee_id=request.employee_id,
            intent=final_state.get("intent", "unknown"),
            final_response=final_state.get("final_response", ""),
            policy_answer=final_state.get("policy_answer") or None,
            source_sections=final_state.get("source_sections", []),
            anomaly_flagged=final_state.get("is_anomalous", False),
            anomaly_score=final_state.get("anomaly_score") or None,
            anomaly_probability=final_state.get("anomaly_probability") or None,
            shap_explanation=final_state.get("top_factors", []),
            overall_confidence=final_state.get("overall_confidence", 0.0),
            needs_escalation=final_state.get("needs_escalation", False),
            escalation_reason=final_state.get("escalation_reason"),
            processing_errors=final_state.get("errors", []),
        )

    except Exception as exc:
        logger.error(
            f"Unhandled error in /query: {exc}",
            extra={"agent_name": "main"},
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                detail=str(exc),
            ).model_dump(),
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
