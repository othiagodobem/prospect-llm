from __future__ import annotations

import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import settings
from errors import DocumentProcessingError, DocumentSecurityError

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def is_suspicious_knowledge_filename(filename: str) -> bool:
    normalized = re.sub(r"[^a-z0-9à-ÿ]+", "_", Path(filename).name.lower())
    return any(pattern in normalized for pattern in settings.suspicious_document_patterns)


def validate_upload(filename: str, size_bytes: int, classificacao: str) -> str:
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise DocumentProcessingError("Use somente PDF, DOCX ou TXT.")
    limit = settings.effective_max_upload_mb
    if size_bytes > limit * 1024 * 1024:
        raise DocumentProcessingError(f"O arquivo excede o limite de {limit} MB.")
    if not classificacao:
        raise DocumentProcessingError("Selecione a classificação do documento.")
    if (
        settings.is_cloud_llm
        and not settings.ALLOW_RESTRICTED_CLOUD
        and classificacao in {"restrito", "sensivel_confidencial"}
    ):
        raise DocumentSecurityError(
            "Documentos restritos ou sensíveis não podem alimentar um LLM em nuvem nesta configuração. "
            "Use dados anonimizados/não sensíveis ou um provedor local aprovado."
        )
    if settings.ENABLE_KNOWLEDGE_BASE_GUARD and is_suspicious_knowledge_filename(safe_name):
        raise DocumentSecurityError(
            "O nome do arquivo indica guia, gabarito ou resultado esperado. Esse tipo de material deve "
            "ficar fora da base de conhecimento para não contaminar a avaliação do framework."
        )
    return safe_name


@contextmanager
def temporary_upload(uploaded_file, safe_name: str) -> Iterator[str]:
    suffix = Path(safe_name).suffix.lower()
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded_file.getbuffer())
            path = temp.name
        yield path
    finally:
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
