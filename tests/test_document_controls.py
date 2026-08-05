import datetime as dt
from pathlib import Path

import pytest

from errors import DocumentSecurityError
from retrieval.local_hybrid import LocalHybridRetriever


def test_demo_guide_is_blocked(tmp_path: Path):
    source = tmp_path / "09_GUIA_Demonstracao_PROSPECT_LLM.txt"
    source.write_text(
        "Guia de demonstração PROSPECT-LLM. Hipóteses esperadas. "
        "Resultado esperado da qualificação. Perguntas esperadas.",
        encoding="utf-8",
    )
    retriever = LocalHybridRetriever(index_dir=str(tmp_path / "index"))
    with pytest.raises(DocumentSecurityError):
        retriever.ingest_document(str(source), "interno_nao_sensivel", source.name)


def test_inactive_and_expired_sources_are_not_retrieved(tmp_path: Path):
    source = tmp_path / "portfolio.txt"
    source.write_text(
        "Serviços de integração de sistemas, APIs e governança de dados para empresas B2B.",
        encoding="utf-8",
    )
    retriever = LocalHybridRetriever(index_dir=str(tmp_path / "index"))
    reference = retriever.ingest_document(
        str(source),
        "publico",
        source.name,
        metadata={
            "active": True,
            "use_in_generation": True,
            "valido_ate": (dt.date.today() + dt.timedelta(days=1)).isoformat(),
        },
    )
    assert retriever.search("integração por APIs", top_k=3)
    retriever.set_document_active(reference, False)
    assert retriever.search("integração por APIs", top_k=3) == []
