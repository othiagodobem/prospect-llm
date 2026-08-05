from __future__ import annotations

from functools import lru_cache

from config import settings
from retrieval.base import Retriever
from retrieval.local_hybrid import LocalHybridRetriever


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    if settings.RETRIEVAL_BACKEND == "local_hybrid":
        return LocalHybridRetriever()
    raise NotImplementedError(
        f"Backend '{settings.RETRIEVAL_BACKEND}' não implementado nesta distribuição."
    )
