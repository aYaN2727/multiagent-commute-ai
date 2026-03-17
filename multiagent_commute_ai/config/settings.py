"""
config/settings.py
Central configuration using pydantic-settings with environment variable support.
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Configuration ────────────────────────────────────────────────────
    OPENAI_API_KEY: str = "sk-mock"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_MOCK_MODE: bool = False

    # ── Ollama Configuration (used when LLM_PROVIDER=ollama) ─────────────────
    LLM_PROVIDER: str = "openai"          # "openai" | "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "llama3.2"

    # ── Embedding Model ──────────────────────────────────────────────────────
    EMBED_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    EMBED_DIMENSION: int = 768

    # ── RAG Configuration ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 5
    POLICY_DOCS_DIR: str = "data/policies"
    FAISS_INDEX_PATH: str = "indexes/policy_index.faiss"

    # ── ML Configuration ─────────────────────────────────────────────────────
    ISOLATION_FOREST_MODEL_PATH: str = "models/isolation_forest.pkl"
    SHAP_EXPLAINER_PATH: str = "models/shap_explainer.pkl"
    CONTAMINATION: float = 0.05
    N_ESTIMATORS: int = 100
    ANOMALY_THRESHOLD: float = -0.1

    # ── API Configuration ────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
