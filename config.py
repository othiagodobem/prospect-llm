"""Configuração central e segura do PROSPECT-LLM."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "PROSPECT-LLM")
    APP_VERSION: str = os.getenv("APP_VERSION", "3.2.0")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/prospect_llm.db")
    LANGGRAPH_CHECKPOINT_DB: str = os.getenv(
        "LANGGRAPH_CHECKPOINT_DB", "data/checkpoints.db"
    )

    # A API DeepSeek é compatível com o SDK da OpenAI.
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-v4-flash")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    )
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "5000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # Recuperação local híbrida.
    RETRIEVAL_BACKEND: str = os.getenv("RETRIEVAL_BACKEND", "local_hybrid")
    RETRIEVAL_INDEX_DIR: str = os.getenv("RETRIEVAL_INDEX_DIR", "data/retrieval")
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    RETRIEVAL_MIN_RELEVANCE: float = float(
        os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.15")
    )
    RETRIEVAL_SEMANTIC_WEIGHT: float = float(
        os.getenv("RETRIEVAL_SEMANTIC_WEIGHT", "0.60")
    )
    RETRIEVAL_LEXICAL_WEIGHT: float = float(
        os.getenv("RETRIEVAL_LEXICAL_WEIGHT", "0.40")
    )

    # Limite duro do MVP. Mesmo que o .env traga valor maior, a aplicação
    # preserva o teto para evitar travamentos e ingestões acidentais.
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    HARD_MAX_UPLOAD_MB: int = int(os.getenv("HARD_MAX_UPLOAD_MB", "20"))
    ALLOW_RESTRICTED_CLOUD: bool = os.getenv(
        "ALLOW_RESTRICTED_CLOUD", "false"
    ).lower() == "true"

    # Protege demonstrações e avaliações contra documentos que já contenham
    # respostas esperadas, gabaritos ou instruções de preenchimento.
    ENABLE_KNOWLEDGE_BASE_GUARD: bool = os.getenv(
        "ENABLE_KNOWLEDGE_BASE_GUARD", "true"
    ).lower() == "true"

    REQUIRE_FACT_SOURCE: bool = os.getenv(
        "REQUIRE_FACT_SOURCE", "true"
    ).lower() == "true"
    REQUIRE_HIGH_PRIORITY_QUESTIONS: bool = os.getenv(
        "REQUIRE_HIGH_PRIORITY_QUESTIONS", "true"
    ).lower() == "true"

    LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    @property
    def is_cloud_llm(self) -> bool:
        return self.LLM_PROVIDER.lower() in {"deepseek", "openai", "anthropic"}

    @property
    def effective_max_upload_mb(self) -> int:
        requested = max(1, self.MAX_UPLOAD_MB)
        hard_limit = max(1, self.HARD_MAX_UPLOAD_MB)
        return min(requested, hard_limit)

    @property
    def data_dir(self) -> Path:
        return Path("data")

    @property
    def suspicious_document_patterns(self) -> tuple[str, ...]:
        raw = os.getenv(
            "SUSPICIOUS_DOCUMENT_PATTERNS",
            "guia_demonstracao,gabarito,respostas_esperadas,resultado_esperado,"
            "resultados_esperados,leia-me,readme_demo",
        )
        return tuple(item.strip().lower() for item in raw.split(",") if item.strip())

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        Path(self.RETRIEVAL_INDEX_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LANGGRAPH_CHECKPOINT_DB).parent.mkdir(parents=True, exist_ok=True)

    def validate_for_generation(self) -> list[str]:
        errors: list[str] = []
        if self.LLM_PROVIDER.lower() == "deepseek" and not self.DEEPSEEK_API_KEY:
            errors.append("DEEPSEEK_API_KEY não foi configurada no arquivo .env.")
        if self.LLM_PROVIDER.lower() != "deepseek":
            errors.append("Esta distribuição foi preparada para LLM_PROVIDER=deepseek.")
        if not 0 <= self.RETRIEVAL_SEMANTIC_WEIGHT <= 1:
            errors.append("RETRIEVAL_SEMANTIC_WEIGHT deve estar entre 0 e 1.")
        if not 0 <= self.RETRIEVAL_LEXICAL_WEIGHT <= 1:
            errors.append("RETRIEVAL_LEXICAL_WEIGHT deve estar entre 0 e 1.")
        if self.RETRIEVAL_SEMANTIC_WEIGHT + self.RETRIEVAL_LEXICAL_WEIGHT <= 0:
            errors.append("Os pesos de recuperação não podem ser ambos zero.")
        return errors


settings = Settings()
settings.ensure_directories()
