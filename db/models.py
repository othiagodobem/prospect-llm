"""
Modelo de dados alinhado 1:1 aos artefatos A1-A8 do artigo.

Roda em SQLite por padrão (Opção 1) e em Postgres sem mudar uma linha de
código (Opção 2) -- só troque DATABASE_URL no .env.

Alguns artefatos (A2-A6) são guardados como JSON dentro de PipelineRun por
simplicidade na fase MVP, mas com a MESMA estrutura dos schemas Pydantic
em schemas/artifacts.py. Ao migrar para Postgres em produção, considere
normalizá-los em tabelas próprias se for fazer consultas analíticas
pesadas (ex.: "quantas hipóteses da dimensão técnica tiveram confiança
alta no último trimestre").
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    """A1. Inventário de fontes (Etapa 1 - governança, Quadro 5)."""
    __tablename__ = "source_document"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(500))
    classificacao: Mapped[str] = mapped_column(String(50))  # ver ClassificacaoDado
    responsavel: Mapped[str] = mapped_column(String(200), nullable=True)
    # Referência no backend de recuperação ativo: file_id da OpenAI na
    # Opção 1; caminho/ID de chunk no seu índice próprio na Opção 2.
    retrieval_ref: Mapped[str] = mapped_column(String(300), nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Prospect(Base):
    __tablename__ = "prospect"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    empresa: Mapped[str] = mapped_column(String(300))
    setor: Mapped[str] = mapped_column(String(200), nullable=True)
    porte: Mapped[str] = mapped_column(String(100), nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    runs: Mapped[list["PipelineRun"]] = relationship(back_populates="prospect")


class PipelineRun(Base):
    """
    Uma execução do fluxo (Etapas 2-7) para um prospect. `id` é usado
    como thread_id do checkpointer do LangGraph -- é o que liga o estado
    do grafo a esta linha do banco.
    """
    __tablename__ = "pipeline_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    prospect_id: Mapped[str] = mapped_column(ForeignKey("prospect.id"))
    status: Mapped[str] = mapped_column(String(30), default="GERADO")
    # GERADO | PROCESSANDO | EM_REVISAO | AJUSTES_SOLICITADOS | APROVADO | REJEITADO | FALHA

    perfil_contextual: Mapped[dict] = mapped_column(JSON, nullable=True)      # A2
    pacote_contexto: Mapped[list] = mapped_column(JSON, nullable=True)        # A3
    matriz_hipoteses: Mapped[dict] = mapped_column(JSON, nullable=True)       # A4
    roteiro_diagnostico: Mapped[dict] = mapped_column(JSON, nullable=True)    # A5
    qualificacao: Mapped[dict] = mapped_column(JSON, nullable=True)           # A6

    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    atualizado_em: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    prospect: Mapped["Prospect"] = relationship(back_populates="runs")
    validacoes: Mapped[list["ValidationDecision"]] = relationship(back_populates="run")
    model_runs: Mapped[list["ModelRun"]] = relationship(back_populates="run")


class ValidationDecision(Base):
    """A7. Registro de validação -- uma linha por critério do Quadro 9."""
    __tablename__ = "validation_decision"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"))
    criterio: Mapped[str] = mapped_column(String(50))
    pergunta_controle: Mapped[str] = mapped_column(Text)
    decisao: Mapped[str] = mapped_column(String(20))
    comentario: Mapped[str] = mapped_column(Text, nullable=True)
    revisor: Mapped[str] = mapped_column(String(200))
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    run: Mapped["PipelineRun"] = relationship(back_populates="validacoes")


class ApprovedScript(Base):
    """A8. Roteiro aprovado."""
    __tablename__ = "approved_script"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"))
    conteudo: Mapped[dict] = mapped_column(JSON)
    aprovado_por: Mapped[str] = mapped_column(String(200))
    aprovado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ModelRun(Base):
    """
    C9. Registro e rastreabilidade -- uma linha por chamada ao LLM.

    Se LANGFUSE_ENABLED=true, o mesmo evento também é enviado ao Langfuse
    (custo, latência, versão de prompt, anotação humana); esta tabela
    garante rastreabilidade básica mesmo sem depender de um serviço
    externo, o que importa para reprodutibilidade de pesquisa.
    """
    __tablename__ = "model_run"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"))
    etapa: Mapped[str] = mapped_column(String(50))
    prompt_version: Mapped[str] = mapped_column(String(20))
    modelo: Mapped[str] = mapped_column(String(100))
    duracao_ms: Mapped[int] = mapped_column(nullable=True)
    custo_estimado_usd: Mapped[float] = mapped_column(Float, nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    run: Mapped["PipelineRun"] = relationship(back_populates="model_runs")


class AuditEvent(Base):
    """Eventos adicionais de rastreabilidade sem alterar tabelas existentes."""
    __tablename__ = "audit_event"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSON, nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class DocumentGovernance(Base):
    """Metadados de versão e validade do A1 sem alterar a tabela legada."""
    __tablename__ = "document_governance"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source_document_id: Mapped[str] = mapped_column(ForeignKey("source_document.id"), unique=True)
    version: Mapped[str] = mapped_column(String(80), nullable=True)
    data_documento: Mapped[dt.datetime] = mapped_column(DateTime, nullable=True)
    valido_ate: Mapped[dt.datetime] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="ATIVO")
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class DocumentUsagePolicy(Base):
    """Política de uso da fonte na geração e no processo de avaliação."""
    __tablename__ = "document_usage_policy"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source_document_id: Mapped[str] = mapped_column(
        ForeignKey("source_document.id"), unique=True
    )
    source_type: Mapped[str] = mapped_column(String(80), nullable=True)
    use_in_generation: Mapped[bool] = mapped_column(default=True)
    authorization_statement: Mapped[str] = mapped_column(Text, nullable=True)
    authorization_confirmed_at: Mapped[dt.datetime] = mapped_column(
        DateTime, nullable=True
    )
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class RunLineage(Base):
    """Relaciona uma execução revisada à execução que lhe deu origem."""
    __tablename__ = "run_lineage"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    child_run_id: Mapped[str] = mapped_column(
        ForeignKey("pipeline_run.id"), unique=True
    )
    parent_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"))
    revision_number: Mapped[int] = mapped_column(default=1)
    reason: Mapped[str] = mapped_column(Text, nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class RunSourceSnapshot(Base):
    """Snapshot das fontes consideradas em cada execução."""
    __tablename__ = "run_source_snapshot"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_run.id"))
    source_document_id: Mapped[str] = mapped_column(String(36), nullable=True)
    filename: Mapped[str] = mapped_column(String(500))
    retrieval_ref: Mapped[str] = mapped_column(String(300), nullable=True)
    version: Mapped[str] = mapped_column(String(80), nullable=True)
    classificacao: Mapped[str] = mapped_column(String(50), nullable=True)
    relevancia: Mapped[float] = mapped_column(Float, nullable=True)
    criado_em: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
