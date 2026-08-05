"""Compatibilidade legada. O backend padrão agora é local_hybrid."""
from retrieval.factory import get_retriever

__all__ = ["get_retriever"]
