"""Contrato estável dos mecanismos de recuperação."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List

from schemas.artifacts import TrechoRecuperado


class Retriever(ABC):
    @abstractmethod
    def ingest_document(
        self,
        file_path: str,
        classificacao: str,
        original_filename: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Indexa um documento e retorna sua referência persistente."""

    @abstractmethod
    def search(self, query: str, top_k: int = 8) -> List[TrechoRecuperado]:
        """Recupera trechos rastreáveis, sem síntese sem fonte."""

    def list_documents(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def set_document_active(self, document_id: str, active: bool) -> None:
        raise NotImplementedError

    def remove_document(self, document_id: str) -> None:
        raise NotImplementedError
