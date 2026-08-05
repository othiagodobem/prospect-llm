from pathlib import Path

from retrieval.local_hybrid import LocalHybridRetriever


def test_ingest_and_search_txt(tmp_path: Path):
    source = tmp_path / "portfolio.txt"
    source.write_text(
        "A empresa oferece integração de sistemas, automação de processos e análise de dados. "
        "Os projetos incluem APIs, modernização de legados e observabilidade.",
        encoding="utf-8",
    )
    retriever = LocalHybridRetriever(index_dir=str(tmp_path / "index"))
    ref = retriever.ingest_document(str(source), "publico", "portfolio.txt")
    assert ref
    results = retriever.search("integração de sistemas e APIs", top_k=3)
    assert results
    assert results[0].fonte == "portfolio.txt"
    assert results[0].relevancia > 0
