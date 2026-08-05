"""Interface Streamlit do PROSPECT-LLM, orientada a workflow e revisão humana."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st
from langgraph.types import Command
from sqlalchemy import desc

from config import settings
from db.models import (
    ApprovedScript,
    AuditEvent,
    DocumentGovernance,
    DocumentUsagePolicy,
    ModelRun,
    PipelineRun,
    Prospect,
    RunLineage,
    RunSourceSnapshot,
    SourceDocument,
    ValidationDecision,
)
from db.session import SessionLocal, init_db
from errors import (
    ConfigurationError,
    DocumentProcessingError,
    DocumentSecurityError,
    LLMAuthenticationError,
    LLMBillingError,
    LLMRateLimitError,
    LLMServiceError,
    LLMStructuredOutputError,
)
from llm.provider import get_llm_provider
from retrieval.factory import get_retriever
from schemas.artifacts import CriterioValidacao, PerfilContextual
from ui.components import (
    questions_dataframe,
    render_approved_script,
    render_evidence,
    render_hypotheses,
    render_qualification,
    render_questions,
)
from ui.styles import apply_styles, render_header, render_stepper
from utils.files import is_suspicious_knowledge_filename, temporary_upload, validate_upload
from workflow.graph import build_graph


CLASSIFICATION_DETAILS = {
    "publico": {
        "label": "Público",
        "description": (
            "Informação já publicada e acessível externamente. Exemplos: site institucional, "
            "relatório anual, notícia corporativa, portfólio público e caso de sucesso publicado."
        ),
    },
    "interno_nao_sensivel": {
        "label": "Interno não sensível",
        "description": (
            "Material interno sem dados pessoais, segredos comerciais ou informações confidenciais. "
            "Exemplos: catálogo interno de serviços, apresentação comercial, FAQ, roteiro de diagnóstico "
            "e proposta-modelo totalmente anonimizada."
        ),
    },
    "restrito": {
        "label": "Restrito",
        "description": (
            "Material com acesso limitado e finalidade definida. Exemplos: atas de reunião, registros "
            "autorizados de CRM, proposta com condições comerciais e documentos de processo interno. "
            "O envio para LLM em nuvem é bloqueado por padrão."
        ),
    },
    "sensivel_confidencial": {
        "label": "Sensível ou confidencial",
        "description": (
            "Material com dados pessoais, contratos, valores, credenciais, estratégia ou informação "
            "protegida. Não anexe ao MVP em nuvem; use ambiente aprovado, anonimização e autorização formal."
        ),
    },
}

SOURCE_TYPE_DETAILS = {
    "portfolio": "Portfólio ou catálogo",
    "case": "Caso autorizado",
    "faq_tecnica": "FAQ ou material técnico",
    "proposta_modelo": "Proposta-modelo anonimizada",
    "apresentacao": "Apresentação institucional",
    "registro_autorizado": "Registro autorizado de reunião ou processo",
    "outro": "Outro documento de apoio",
}

CONTEXT_TYPE_DETAILS = {
    "fato": (
        "Informação verificável. Ex.: “A empresa anunciou a abertura de duas novas unidades em 2026”."
    ),
    "inferencia": (
        "Interpretação ainda não confirmada. Ex.: “A expansão pode aumentar a necessidade de integração "
        "entre unidades”."
    ),
    "lacuna": (
        "Informação que precisa ser obtida. Ex.: “Não sabemos quais sistemas são usados atualmente”."
    ),
}

VALIDATION_COMMENT_EXAMPLES = {
    CriterioValidacao.FUNDAMENTACAO: (
        "Ex.: Ajustar a hipótese 2 para indicar que se trata de inferência e vincular a fonte "
        "Catálogo 2026, p. 8."
    ),
    CriterioValidacao.ADERENCIA_PORTFOLIO: (
        "Ex.: A solução citada não consta no portfólio atual; substituir por Integração de Sistemas."
    ),
    CriterioValidacao.QUALIDADE_DIAGNOSTICA: (
        "Ex.: Reformular a pergunta 3 para evitar indução e investigar primeiro o processo atual."
    ),
    CriterioValidacao.PROTECAO_DADOS: (
        "Ex.: Remover o nome do colaborador e manter apenas o cargo e a área responsável."
    ),
    CriterioValidacao.LINGUAGEM_COMERCIAL: (
        "Ex.: Retirar a promessa de redução de custos, pois ainda não há evidência ou diagnóstico confirmado."
    ),
    CriterioValidacao.PROXIMO_PASSO: (
        "Ex.: Indicar como próximo passo uma reunião de diagnóstico com TI e Operações."
    ),
}

CRITERIA_TEXT = {
    CriterioValidacao.FUNDAMENTACAO: (
        "As afirmações e hipóteses estão vinculadas a fontes ou sinalizadas como inferência?"
    ),
    CriterioValidacao.ADERENCIA_PORTFOLIO: (
        "As soluções mencionadas existem e atendem ao escopo descrito?"
    ),
    CriterioValidacao.QUALIDADE_DIAGNOSTICA: (
        "As perguntas são abertas, não indutivas e úteis para confirmar o problema?"
    ),
    CriterioValidacao.PROTECAO_DADOS: (
        "O roteiro expõe informação pessoal, restrita ou desnecessária?"
    ),
    CriterioValidacao.LINGUAGEM_COMERCIAL: (
        "O texto evita promessas, exageros e conclusões não confirmadas?"
    ),
    CriterioValidacao.PROXIMO_PASSO: (
        "O roteiro indica objetivo da conversa e condição para continuidade?"
    ),
}

PAGE_STEP = {
    "Base de conhecimento": 1,
    "Coleta de contexto": 2,
    "Análise": 3,
    "Validação": 4,
}

STATUS_PRESENTATION = {
    "PROCESSANDO": ("Processando", "info"),
    "GERADO": ("Gerado", "info"),
    "EM_REVISAO": ("Em revisão", "warn"),
    "AJUSTES_SOLICITADOS": ("Ajustes solicitados", "warn"),
    "APROVADO": ("Aprovado", "ok"),
    "REJEITADO": ("Rejeitado", "danger"),
    "FALHA": ("Falha", "danger"),
}

CRITERION_LABELS = {
    CriterioValidacao.FUNDAMENTACAO: "Fundamentação",
    CriterioValidacao.ADERENCIA_PORTFOLIO: "Aderência ao portfólio",
    CriterioValidacao.QUALIDADE_DIAGNOSTICA: "Qualidade diagnóstica",
    CriterioValidacao.PROTECAO_DADOS: "Proteção de dados",
    CriterioValidacao.LINGUAGEM_COMERCIAL: "Linguagem comercial",
    CriterioValidacao.PROXIMO_PASSO: "Próximo passo",
}

DECISION_LABELS = {
    "nao_avaliado": "Não avaliado",
    "aprovar": "Aprovar",
    "ajustar": "Ajustar",
    "anonimizar": "Anonimizar",
    "rejeitar": "Rejeitar",
}

VALIDATION_OPTIONS = {
    CriterioValidacao.FUNDAMENTACAO: ["nao_avaliado", "aprovar", "ajustar", "rejeitar"],
    CriterioValidacao.ADERENCIA_PORTFOLIO: ["nao_avaliado", "aprovar", "ajustar", "rejeitar"],
    CriterioValidacao.QUALIDADE_DIAGNOSTICA: ["nao_avaliado", "aprovar", "ajustar", "rejeitar"],
    CriterioValidacao.PROTECAO_DADOS: ["nao_avaliado", "aprovar", "anonimizar", "rejeitar"],
    CriterioValidacao.LINGUAGEM_COMERCIAL: ["nao_avaliado", "aprovar", "ajustar"],
    CriterioValidacao.PROXIMO_PASSO: ["nao_avaliado", "aprovar", "ajustar"],
}



st.set_page_config(
    page_title="PROSPECT-LLM",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_styles()
init_db()

def normalize_context_rows(value: object) -> pd.DataFrame:
    """Normaliza a tabela editável para tipos compatíveis com o Streamlit.

    Sessões iniciadas em versões anteriores podem manter datas como texto
    (por exemplo, ``15/06/2026`` ou ``None``). A DateColumn exige uma coluna
    datetime; por isso a conversão é feita antes de cada renderização.
    Valores inválidos ficam vazios, sem interromper a interface.
    """

    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif value is None:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame(value)

    defaults = {
        "Tipo": "fato",
        "Conteúdo": "",
        "Fonte": "",
        "Data": pd.NaT,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame = frame[["Tipo", "Conteúdo", "Fonte", "Data"]].copy()
    frame["Tipo"] = frame["Tipo"].fillna("fato").astype(str)
    frame["Conteúdo"] = frame["Conteúdo"].fillna("").astype(str)
    frame["Fonte"] = frame["Fonte"].fillna("").astype(str)
    frame["Data"] = pd.to_datetime(frame["Data"], errors="coerce", dayfirst=True)
    return frame


def context_item_date(value: object):
    """Converte a data do editor para ``datetime.date`` ou ``None``."""

    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "page" not in st.session_state:
    st.session_state.page = "Base de conhecimento"
if "context_rows" not in st.session_state:
    st.session_state.context_rows = pd.DataFrame(
        [
            {"Tipo": "fato", "Conteúdo": "", "Fonte": "", "Data": pd.NaT},
            {"Tipo": "lacuna", "Conteúdo": "", "Fonte": "", "Data": pd.NaT},
        ]
    )
st.session_state.context_rows = normalize_context_rows(st.session_state.context_rows)

for _key, _default in {
    "context_company": "",
    "context_sector": "",
    "context_size": "",
    "context_location": "",
    "context_offer": "",
    "context_contact": "",
    "context_role": "",
    "context_decision_role": "",
    "context_history": "",
    "context_restrictions": "",
}.items():
    st.session_state.setdefault(_key, _default)

graph = st.session_state.graph


def load_run(run_id: str | None) -> PipelineRun | None:
    if not run_id:
        return None
    with SessionLocal() as db:
        return db.get(PipelineRun, run_id)


def mark_run_failed(run_id: str | None, message: str) -> None:
    if not run_id:
        return
    with SessionLocal() as db:
        run = db.get(PipelineRun, run_id)
        if run:
            run.status = "FALHA"
        db.add(
            AuditEvent(
                run_id=run_id,
                event_type="pipeline_error",
                payload={"message": message},
            )
        )
        db.commit()


def friendly_error(exc: Exception) -> None:
    messages = {
        LLMBillingError: (
            "Saldo insuficiente na API DeepSeek. Acesse o painel da DeepSeek e adicione crédito."
        ),
        LLMRateLimitError: (
            "Limite temporário da DeepSeek atingido. Aguarde alguns instantes e tente novamente."
        ),
        LLMAuthenticationError: (
            "A chave da DeepSeek é inválida ou foi revogada. Atualize DEEPSEEK_API_KEY no .env."
        ),
        LLMStructuredOutputError: (
            "A resposta do modelo não passou na validação estrutural. Tente novamente."
        ),
        LLMServiceError: str(exc),
        ConfigurationError: str(exc),
        DocumentSecurityError: str(exc),
        DocumentProcessingError: str(exc),
    }
    message = next(
        (text for error_type, text in messages.items() if isinstance(exc, error_type)),
        None,
    )
    st.error(message or "Ocorreu um erro inesperado. A execução foi interrompida com segurança.")
    if settings.DEBUG:
        st.exception(exc)



def load_run_summary(run_id: str | None) -> dict | None:
    """Carrega o resumo priorizando o A2 salvo na própria execução.

    Isso evita que registros antigos ou colunas legadas exibam o interlocutor
    no lugar do setor. O perfil contextual é a fonte de verdade do run.
    """
    if not run_id:
        return None
    with SessionLocal() as db:
        row = (
            db.query(PipelineRun, Prospect)
            .join(Prospect, PipelineRun.prospect_id == Prospect.id)
            .filter(PipelineRun.id == run_id)
            .first()
        )
        if not row:
            return None
        pipeline_run, prospect = row
        profile = pipeline_run.perfil_contextual or {}
        lineage = (
            db.query(RunLineage)
            .filter(RunLineage.child_run_id == run_id)
            .first()
        )
        sector = profile.get("setor") or prospect.setor or "—"
        size = profile.get("porte") or prospect.porte or "—"
        return {
            "run_id": pipeline_run.id,
            "empresa": profile.get("empresa") or prospect.empresa,
            "setor": sector,
            "porte": size,
            "interlocutor": profile.get("interlocutor") or "—",
            "cargo_interlocutor": profile.get("cargo_interlocutor") or "—",
            "papel_decisorio": profile.get("papel_decisorio") or "—",
            "status": pipeline_run.status,
            "criado_em": pipeline_run.criado_em,
            "atualizado_em": pipeline_run.atualizado_em,
            "parent_run_id": lineage.parent_run_id if lineage else None,
            "revision_number": lineage.revision_number if lineage else 0,
        }

def load_run_alerts(run_id: str) -> list[str]:
    with SessionLocal() as db:
        event = (
            db.query(AuditEvent)
            .filter(
                AuditEvent.run_id == run_id,
                AuditEvent.event_type == "pipeline_generated",
            )
            .order_by(desc(AuditEvent.criado_em))
            .first()
        )
    if not event or not event.payload:
        return []
    return [str(item) for item in event.payload.get("alerts", []) if str(item).strip()]


def load_traceability(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with SessionLocal() as db:
        model_runs = (
            db.query(ModelRun)
            .filter(ModelRun.run_id == run_id)
            .order_by(ModelRun.criado_em)
            .all()
        )
        audit_events = (
            db.query(AuditEvent)
            .filter(AuditEvent.run_id == run_id)
            .order_by(AuditEvent.criado_em)
            .all()
        )

    model_frame = pd.DataFrame(
        [
            {
                "Etapa": item.etapa.replace("_", " ").title(),
                "Prompt": item.prompt_version,
                "Modelo": item.modelo,
                "Duração (ms)": item.duracao_ms,
                "Custo estimado (US$)": item.custo_estimado_usd,
                "Data": item.criado_em.strftime("%d/%m/%Y %H:%M:%S"),
            }
            for item in model_runs
        ]
    )
    audit_frame = pd.DataFrame(
        [
            {
                "Evento": item.event_type.replace("_", " ").title(),
                "Data": item.criado_em.strftime("%d/%m/%Y %H:%M:%S"),
                "Resumo": str(item.payload or {}),
            }
            for item in audit_events
        ]
    )
    return model_frame, audit_frame


def load_validation_history(run_id: str) -> pd.DataFrame:
    with SessionLocal() as db:
        decisions = (
            db.query(ValidationDecision)
            .filter(ValidationDecision.run_id == run_id)
            .order_by(ValidationDecision.criado_em)
            .all()
        )
    return pd.DataFrame(
        [
            {
                "Critério": CRITERION_LABELS.get(
                    CriterioValidacao(item.criterio), item.criterio
                ),
                "Decisão": DECISION_LABELS.get(item.decisao, item.decisao),
                "Comentário": item.comentario or "—",
                "Revisor": item.revisor,
                "Data": item.criado_em.strftime("%d/%m/%Y %H:%M:%S"),
            }
            for item in decisions
        ]
    )


def has_indexed_sources() -> bool:
    try:
        documents = get_retriever().list_documents()
    except Exception:
        return False
    today = dt.date.today()
    for document in documents:
        if not document.get("active", True) or not document.get("use_in_generation", True):
            continue
        valid_until = document.get("valido_ate")
        if valid_until:
            try:
                if dt.date.fromisoformat(str(valid_until)[:10]) < today:
                    continue
            except ValueError:
                continue
        if settings.is_cloud_llm and not settings.ALLOW_RESTRICTED_CLOUD:
            if document.get("classificacao") not in {"publico", "interno_nao_sensivel"}:
                continue
        return True
    return False

def _document_policy_defaults(document: SourceDocument, governance: DocumentGovernance | None) -> dict:
    suspicious = is_suspicious_knowledge_filename(document.filename)
    return {
        "status": governance.status if governance else ("QUARENTENA" if suspicious else "ATIVO"),
        "use_in_generation": not suspicious,
        "source_type": None,
    }


def quarantine_suspicious_sources() -> int:
    """Desativa automaticamente guias e gabaritos já presentes em bases antigas."""
    if not settings.ENABLE_KNOWLEDGE_BASE_GUARD:
        return 0
    changed = 0
    retriever = get_retriever()
    with SessionLocal() as db:
        rows = (
            db.query(SourceDocument, DocumentGovernance, DocumentUsagePolicy)
            .outerjoin(
                DocumentGovernance,
                DocumentGovernance.source_document_id == SourceDocument.id,
            )
            .outerjoin(
                DocumentUsagePolicy,
                DocumentUsagePolicy.source_document_id == SourceDocument.id,
            )
            .all()
        )
        for document, governance, policy in rows:
            if not is_suspicious_knowledge_filename(document.filename):
                continue
            if governance is None:
                governance = DocumentGovernance(
                    source_document_id=document.id,
                    status="QUARENTENA",
                )
                db.add(governance)
            else:
                governance.status = "QUARENTENA"
            if policy is None:
                policy = DocumentUsagePolicy(
                    source_document_id=document.id,
                    use_in_generation=False,
                    notes="Quarentena automática: possível guia, gabarito ou resposta esperada.",
                )
                db.add(policy)
            else:
                policy.use_in_generation = False
                policy.notes = (
                    (policy.notes + "\n") if policy.notes else ""
                ) + "Quarentena automática: possível material de avaliação."
            try:
                if document.retrieval_ref:
                    retriever.set_document_active(document.retrieval_ref, False)
            except Exception:
                pass
            changed += 1
        if changed:
            db.add(
                AuditEvent(
                    run_id=None,
                    event_type="knowledge_base_quarantine",
                    payload={"documents_quarantined": changed},
                )
            )
            db.commit()
    return changed


def run_suspicious_sources(run: PipelineRun | None) -> list[str]:
    if not run:
        return []
    sources = {
        str(item.get("fonte", ""))
        for item in (run.pacote_contexto or [])
        if is_suspicious_knowledge_filename(str(item.get("fonte", "")))
    }
    return sorted(item for item in sources if item)


def load_profile_into_form(run_id: str, reason: str | None = None) -> None:
    """Preenche o formulário de contexto para uma revisão rastreável."""
    run = load_run(run_id)
    if not run or not run.perfil_contextual:
        raise RuntimeError("O perfil contextual da execução não foi encontrado.")
    profile = run.perfil_contextual
    valid_sizes = {"", "Micro", "Pequeno", "Médio", "Grande"}
    valid_decision_roles = {
        "", "Decisor", "Influenciador", "Responsável técnico", "Usuário",
        "Comprador", "Ainda não confirmado",
    }
    profile_size = profile.get("porte", "") or ""
    decision_role = profile.get("papel_decisorio", "") or ""
    mapping = {
        "context_company": profile.get("empresa", ""),
        "context_sector": profile.get("setor", ""),
        "context_size": profile_size if profile_size in valid_sizes else "",
        "context_location": profile.get("localizacao", ""),
        "context_offer": profile.get("oferta_principal", ""),
        "context_contact": profile.get("interlocutor", ""),
        "context_role": profile.get("cargo_interlocutor", ""),
        "context_decision_role": decision_role if decision_role in valid_decision_roles else "Ainda não confirmado",
        "context_history": profile.get("historico_contato", ""),
        "context_restrictions": profile.get("restricoes_conhecidas", ""),
    }
    for key, value in mapping.items():
        st.session_state[key] = value or ""
    st.session_state.pop("context_editor_v3", None)
    st.session_state.context_rows = normalize_context_rows(
        [
            {
                "Tipo": item.get("tipo", "fato"),
                "Conteúdo": item.get("conteudo", ""),
                "Fonte": item.get("fonte", ""),
                "Data": item.get("data"),
            }
            for item in profile.get("itens", [])
        ]
    )
    st.session_state.parent_run_id = run_id
    st.session_state.revision_reason = reason or "Revisão solicitada após validação humana."
    st.session_state.page = "Coleta de contexto"


def _revision_number(db, parent_run_id: str) -> int:
    parent_lineage = (
        db.query(RunLineage)
        .filter(RunLineage.child_run_id == parent_run_id)
        .first()
    )
    return (parent_lineage.revision_number if parent_lineage else 0) + 1


def snapshot_sources_for_run(db, run_id: str, excerpts: list[dict]) -> None:
    seen: set[str] = set()
    for excerpt in excerpts:
        document_id = excerpt.get("document_id")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        source = (
            db.query(SourceDocument)
            .filter(SourceDocument.retrieval_ref == document_id)
            .first()
        )
        governance = None
        if source:
            governance = (
                db.query(DocumentGovernance)
                .filter(DocumentGovernance.source_document_id == source.id)
                .first()
            )
        db.add(
            RunSourceSnapshot(
                run_id=run_id,
                source_document_id=source.id if source else None,
                filename=excerpt.get("fonte", "Fonte não identificada"),
                retrieval_ref=document_id,
                version=governance.version if governance else excerpt.get("versao_documento"),
                classificacao=excerpt.get("classificacao"),
                relevancia=float(excerpt.get("relevancia", 0) or 0),
            )
        )


def render_status(status: str) -> None:
    label, style = STATUS_PRESENTATION.get(
        status, (status.replace("_", " ").title(), "info")
    )
    st.markdown(
        f'<span class="status-badge status-{style}">{label}</span>',
        unsafe_allow_html=True,
    )


def render_execution_summary(summary: dict | None) -> None:
    if not summary:
        return
    with st.container(border=True):
        top_left, top_right = st.columns([3, 1])
        with top_left:
            revision = (
                f" · Revisão R{summary['revision_number']}"
                if summary.get("revision_number")
                else ""
            )
            st.markdown(f"### {summary['empresa']}{revision}")
            st.caption(
                f"Setor: {summary['setor']} · Porte: {summary['porte']} · "
                f"Criada em {summary['criado_em']:%d/%m/%Y %H:%M}"
            )
            if summary.get("interlocutor") != "—" or summary.get("cargo_interlocutor") != "—":
                st.caption(
                    f"Interlocutor: {summary.get('interlocutor', '—')} · "
                    f"Cargo: {summary.get('cargo_interlocutor', '—')} · "
                    f"Papel decisório: {summary.get('papel_decisorio', '—')}"
                )
        with top_right:
            st.caption("Status da execução")
            render_status(summary["status"])
        st.caption(f"ID da execução: `{summary['run_id']}`")
        if summary.get("parent_run_id"):
            st.caption(f"Derivada da execução: `{summary['parent_run_id']}`")

def render_traceability(run_id: str) -> None:
    model_frame, audit_frame = load_traceability(run_id)
    with SessionLocal() as db:
        snapshots = (
            db.query(RunSourceSnapshot)
            .filter(RunSourceSnapshot.run_id == run_id)
            .order_by(desc(RunSourceSnapshot.relevancia))
            .all()
        )
        lineage = (
            db.query(RunLineage)
            .filter(RunLineage.child_run_id == run_id)
            .first()
        )

    st.subheader("Rastreabilidade")
    st.caption(
        "Registro técnico das chamadas ao modelo, prompts, fontes, configuração e eventos da execução."
    )
    if lineage:
        st.info(
            f"Revisão R{lineage.revision_number} derivada de `{lineage.parent_run_id}`. "
            f"Motivo: {lineage.reason or 'não informado'}."
        )
    if model_frame.empty:
        st.info("Ainda não há chamadas ao modelo registradas para esta execução.")
    else:
        st.dataframe(model_frame, width="stretch", hide_index=True)

    st.markdown("#### Snapshot das fontes")
    if not snapshots:
        st.caption("Nenhum snapshot de fonte foi registrado.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Arquivo": item.filename,
                        "Versão": item.version or "—",
                        "Classificação": item.classificacao or "—",
                        "Relevância máxima": item.relevancia,
                        "Referência": item.retrieval_ref or "—",
                    }
                    for item in snapshots
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Histórico de eventos", expanded=False):
        if audit_frame.empty:
            st.caption("Nenhum evento adicional registrado.")
        else:
            st.dataframe(audit_frame, width="stretch", hide_index=True)

def render_document_examples() -> None:
    with st.expander("Quais documentos posso anexar?", expanded=True):
        st.markdown(
            """
            - **Portfólio e catálogo:** serviços, capacidades, tecnologias, diferenciais e limites de atuação.
            - **Casos autorizados:** problema atendido, solução aplicada e resultados aprovados para divulgação.
            - **Propostas-modelo:** estrutura de escopo, entregáveis e premissas, sem nomes, valores ou dados de clientes.
            - **FAQs e materiais técnicos:** dúvidas frequentes, requisitos, integrações, limitações e implantação.
            - **Apresentações institucionais:** posicionamento, segmentos atendidos, método de trabalho e oferta.
            - **Registros autorizados:** atas ou roteiros de reunião anonimizados, quando houver autorização.
            """
        )
        st.warning(
            "Não anexe senhas, credenciais, dados pessoais desnecessários, contratos, valores confidenciais "
            "ou documentos de clientes sem autorização formal."
        )


def render_context_examples() -> None:
    with st.expander("Ver exemplos de fatos, inferências e lacunas", expanded=True):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Tipo": "fato",
                        "Conteúdo": "A empresa anunciou a abertura de duas novas unidades em 2026.",
                        "Fonte": "Notícia no site institucional",
                        "Data": "15/06/2026",
                    },
                    {
                        "Tipo": "inferencia",
                        "Conteúdo": "A expansão pode aumentar a necessidade de integração entre unidades.",
                        "Fonte": "Inferência baseada na notícia de expansão",
                        "Data": "15/06/2026",
                    },
                    {
                        "Tipo": "lacuna",
                        "Conteúdo": "Não sabemos quais sistemas são usados atualmente pelas unidades.",
                        "Fonte": "Informação a confirmar na reunião",
                        "Data": "—",
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        for item_type, description in CONTEXT_TYPE_DETAILS.items():
            st.markdown(f"- **{item_type.title()}:** {description}")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Navegação")
        pages = ["Base de conhecimento", "Coleta de contexto", "Análise", "Validação"]
        selected_page = st.radio(
            "Etapas",
            pages,
            index=pages.index(st.session_state.page),
            label_visibility="collapsed",
        )
        st.session_state.page = selected_page

        st.divider()
        st.markdown("### Ambiente")
        st.caption(f"Versão: `{settings.APP_VERSION}`")
        st.caption(f"Modelo: `{settings.LLM_MODEL}`")
        st.caption(f"Recuperação: `{settings.RETRIEVAL_BACKEND}`")
        key_state = "configurada, não testada" if settings.DEEPSEEK_API_KEY else "não configurada"
        st.caption(f"Chave DeepSeek: **{key_state}**")
        st.caption(f"Ambiente: `{settings.APP_ENV}`")
        if settings.is_cloud_llm:
            st.caption("Trechos recuperados podem ser enviados ao provedor em nuvem.")
        if st.button(
            "Testar conexão",
            width="stretch",
            disabled=not bool(settings.DEEPSEEK_API_KEY),
            help="Valida a chave e consulta a lista de modelos disponíveis sem gerar uma análise.",
        ):
            try:
                with st.spinner("Verificando a API..."):
                    result = get_llm_provider().health_check()
                if result.get("ok"):
                    st.success(f"Conexão validada. Modelo configurado: {result['model']}.")
                else:
                    st.warning(
                        "A conexão funcionou, mas o modelo configurado não apareceu na lista retornada."
                    )
            except Exception as exc:
                friendly_error(exc)

        st.divider()
        st.markdown("### Execuções recentes")
        with SessionLocal() as db:
            recent_rows = (
                db.query(PipelineRun, Prospect)
                .join(Prospect, PipelineRun.prospect_id == Prospect.id)
                .order_by(desc(PipelineRun.criado_em))
                .limit(20)
                .all()
            )
            options = {
                (
                    f"{(pipeline_run.perfil_contextual or {}).get('empresa') or prospect.empresa} · "
                    f"{pipeline_run.criado_em:%d/%m/%Y %H:%M} · "
                    f"{STATUS_PRESENTATION.get(pipeline_run.status, (pipeline_run.status, 'info'))[0]}"
                ): pipeline_run.id
                for pipeline_run, prospect in recent_rows
            }

        if options:
            labels = list(options)
            current_id = st.session_state.get("run_id")
            current_index = next(
                (index for index, label in enumerate(labels) if options[label] == current_id),
                0,
            )
            selected_label = st.selectbox(
                "Selecionar execução",
                labels,
                index=current_index,
                label_visibility="collapsed",
                help="Escolha uma análise anterior para consultar seus artefatos, validação e histórico.",
            )
            selected_id = options[selected_label]
            if selected_id == current_id:
                st.caption("Execução atualmente aberta.")
            if st.button("Abrir execução", width="stretch"):
                st.session_state.run_id = selected_id
                st.session_state.page = "Análise"
                st.rerun()
        else:
            st.caption("Nenhuma execução criada até o momento.")

        if st.button(
            "Nova análise",
            width="stretch",
            help="Inicia um novo perfil sem apagar as execuções anteriores.",
        ):
            st.session_state.pop("run_id", None)
            st.session_state.pop("parent_run_id", None)
            st.session_state.pop("revision_reason", None)
            for key in (
                "context_company",
                "context_sector",
                "context_size",
                "context_location",
                "context_offer",
                "context_contact",
                "context_role",
                "context_decision_role",
                "context_history",
                "context_restrictions",
            ):
                st.session_state.pop(key, None)
            st.session_state.context_rows = normalize_context_rows(None)
            st.session_state.page = "Coleta de contexto"
            st.rerun()

def render_source_inventory() -> None:
    with st.container(border=True):
        st.subheader("Inventário de fontes")
        st.caption(
            "Documentos preparados para recuperação. Fontes vencidas, desativadas ou em quarentena "
            "não participam da geração."
        )
        with SessionLocal() as db:
            documents = (
                db.query(SourceDocument, DocumentGovernance, DocumentUsagePolicy)
                .outerjoin(
                    DocumentGovernance,
                    DocumentGovernance.source_document_id == SourceDocument.id,
                )
                .outerjoin(
                    DocumentUsagePolicy,
                    DocumentUsagePolicy.source_document_id == SourceDocument.id,
                )
                .order_by(desc(SourceDocument.criado_em))
                .all()
            )

        if not documents:
            st.info(
                "Nenhum documento indexado. Comece por um portfólio, FAQ, apresentação institucional "
                "ou caso autorizado."
            )
            return

        today = dt.date.today()
        rows = []
        suspicious_active = []
        for document, governance, policy in documents:
            defaults = _document_policy_defaults(document, governance)
            status = governance.status if governance else defaults["status"]
            use_in_generation = (
                policy.use_in_generation if policy else defaults["use_in_generation"]
            )
            valid_until = governance.valido_ate.date() if governance and governance.valido_ate else None
            expired = bool(valid_until and valid_until < today)
            effective = status == "ATIVO" and use_in_generation and not expired
            if is_suspicious_knowledge_filename(document.filename) and effective:
                suspicious_active.append(document.filename)
            rows.append(
                {
                    "Arquivo": document.filename,
                    "Versão": governance.version if governance else "—",
                    "Tipo": SOURCE_TYPE_DETAILS.get(policy.source_type, policy.source_type or "—") if policy else "—",
                    "Classificação": CLASSIFICATION_DETAILS.get(
                        document.classificacao,
                        {"label": document.classificacao.replace("_", " ").title()},
                    )["label"],
                    "Status": "Vencido" if expired else status.replace("_", " ").title(),
                    "Na geração": "Sim" if effective else "Não",
                    "Responsável": document.responsavel or "—",
                    "Data": (
                        governance.data_documento.strftime("%d/%m/%Y")
                        if governance and governance.data_documento
                        else "—"
                    ),
                    "Validade": valid_until.strftime("%d/%m/%Y") if valid_until else "—",
                    "Incluído em": document.criado_em.strftime("%d/%m/%Y %H:%M"),
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if suspicious_active:
            st.error(
                "Há materiais com possível gabarito ou resultado esperado ainda ativos: "
                + ", ".join(suspicious_active)
                + ". Desative-os antes de gerar nova análise."
            )

        st.markdown("#### Gerenciar fonte")
        options = {
            f"{document.filename} · {document.id[:8]}": (document, governance, policy)
            for document, governance, policy in documents
        }
        selected_label = st.selectbox(
            "Fonte",
            list(options),
            key="inventory_selected_document",
            help="Selecione um documento para ativar, desativar ou excluir.",
        )
        document, governance, policy = options[selected_label]
        status = governance.status if governance else "ATIVO"
        active = status == "ATIVO" and (policy.use_in_generation if policy else True)

        action_left, action_middle, action_right = st.columns(3)
        with action_left:
            if st.button(
                "Desativar" if active else "Ativar",
                width="stretch",
                key=f"toggle_source_{document.id}",
            ):
                target_active = not active
                if target_active and is_suspicious_knowledge_filename(document.filename):
                    st.error(
                        "Esta fonte permanece bloqueada por possível contaminação de avaliação. "
                        "Mantenha o guia fora da base de geração."
                    )
                    return
                try:
                    get_retriever().set_document_active(document.retrieval_ref, target_active)
                    with SessionLocal() as db:
                        gov = (
                            db.query(DocumentGovernance)
                            .filter(DocumentGovernance.source_document_id == document.id)
                            .first()
                        )
                        if not gov:
                            gov = DocumentGovernance(source_document_id=document.id)
                            db.add(gov)
                        gov.status = "ATIVO" if target_active else "INATIVO"
                        pol = (
                            db.query(DocumentUsagePolicy)
                            .filter(DocumentUsagePolicy.source_document_id == document.id)
                            .first()
                        )
                        if not pol:
                            pol = DocumentUsagePolicy(source_document_id=document.id)
                            db.add(pol)
                        pol.use_in_generation = target_active
                        db.add(
                            AuditEvent(
                                run_id=None,
                                event_type="source_status_changed",
                                payload={
                                    "source_document_id": document.id,
                                    "filename": document.filename,
                                    "active": target_active,
                                },
                            )
                        )
                        db.commit()
                    st.rerun()
                except Exception as exc:
                    friendly_error(exc)
        with action_middle:
            if st.button(
                "Reindexar por novo upload",
                width="stretch",
                key=f"reindex_source_{document.id}",
                help="Reabra o formulário de upload e envie a versão atualizada. O hash evita duplicação.",
            ):
                st.info("Use o formulário à esquerda para enviar a versão atualizada com os metadados corretos.")
        with action_right:
            confirm_delete = st.checkbox(
                "Confirmar exclusão",
                key=f"confirm_delete_{document.id}",
                help="A exclusão remove a fonte da base ativa, mas execuções históricas preservam seus snapshots.",
            )
            if st.button(
                "Excluir fonte",
                width="stretch",
                disabled=not confirm_delete,
                key=f"delete_source_{document.id}",
            ):
                try:
                    get_retriever().remove_document(document.retrieval_ref)
                    with SessionLocal() as db:
                        db.query(DocumentUsagePolicy).filter(
                            DocumentUsagePolicy.source_document_id == document.id
                        ).delete()
                        db.query(DocumentGovernance).filter(
                            DocumentGovernance.source_document_id == document.id
                        ).delete()
                        db.query(SourceDocument).filter(SourceDocument.id == document.id).delete()
                        db.add(
                            AuditEvent(
                                run_id=None,
                                event_type="source_deleted",
                                payload={
                                    "source_document_id": document.id,
                                    "filename": document.filename,
                                },
                            )
                        )
                        db.commit()
                    st.rerun()
                except Exception as exc:
                    friendly_error(exc)

        st.caption(
            "Documentos restritos, sensíveis, vencidos, desativados ou marcados fora da geração "
            "são ignorados pelo recuperador."
        )

def render_base_page() -> None:
    st.header("1. Base de conhecimento")
    st.caption("Governança, classificação e inventário de fontes autorizadas.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Adicionar documento")
            render_document_examples()

            uploaded = st.file_uploader(
                "Documento da base de conhecimento *",
                type=["pdf", "docx", "txt"],
                help=(
                    f"Envie um arquivo por vez, com até {settings.effective_max_upload_mb} MB. "
                    "Use PDF, DOCX ou TXT. Prefira documentos atualizados, legíveis e autorizados."
                ),
            )
            st.caption(
                "Exemplo: `Catalogo_de_Servicos_2026.pdf`, `FAQ_Integracoes.docx` "
                "ou `Proposta_Modelo_Anonimizada.docx`. Não envie guias ou gabaritos da demonstração."
            )

            classification = st.selectbox(
                "Classificação do documento *",
                [""] + list(CLASSIFICATION_DETAILS),
                format_func=lambda value: (
                    "Selecione uma classificação"
                    if not value
                    else CLASSIFICATION_DETAILS[value]["label"]
                ),
                help=(
                    "A classificação determina os controles de acesso e se o documento pode ser processado "
                    "por um modelo em nuvem."
                ),
            )
            if classification:
                st.caption(CLASSIFICATION_DETAILS[classification]["description"])

            source_type = st.selectbox(
                "Tipo de fonte *",
                [""] + list(SOURCE_TYPE_DETAILS),
                format_func=lambda value: "Selecione o tipo" if not value else SOURCE_TYPE_DETAILS[value],
                help="O tipo ajuda a interpretar o papel do documento no processo comercial.",
            )

            use_in_generation = st.checkbox(
                "Permitir uso desta fonte na geração",
                value=True,
                help=(
                    "Desmarque para manter o arquivo apenas no inventário. Guias, instruções e materiais de "
                    "avaliação nunca devem participar da geração."
                ),
            )

            responsible = st.text_input(
                "Responsável pela autorização *",
                placeholder="Ex.: Mariana Souza — Gerente Comercial",
                help=(
                    "Informe a pessoa ou função que confirmou que o documento pode ser usado no sistema. "
                    "Não precisa ser o autor do arquivo."
                ),
            )
            st.caption(
                "Exemplo: gestor do portfólio, responsável comercial, jurídico ou proprietário da informação."
            )

            version = st.text_input(
                "Versão do documento",
                placeholder="Ex.: 2026.1, v3 ou julho/2026",
                help="Use a versão exibida no arquivo. Se não houver, informe o mês e o ano da revisão.",
            )

            date_column, validity_column = st.columns(2)
            with date_column:
                document_date = st.date_input(
                    "Data de publicação ou última revisão",
                    value=None,
                    help="Data de publicação, emissão ou última revisão do documento.",
                )
            with validity_column:
                valid_until = st.date_input(
                    "Válido até",
                    value=None,
                    help="Preencha quando o conteúdo tiver prazo de validade ou revisão programada.",
                )

            notes = st.text_area(
                "Observações de governança",
                placeholder="Ex.: case público autorizado; métricas podem ser citadas apenas como referência histórica.",
                height=80,
            )

            authorization_text = (
                "Confirmo que tenho autorização para processar este documento na finalidade declarada, "
                "que a classificação está correta e que não incluí dados desnecessários."
            )
            authorized = st.checkbox(authorization_text)

            if st.button("Indexar documento", type="primary", width="stretch"):
                if not uploaded:
                    st.warning("Selecione um documento para indexar.")
                    return
                if not classification:
                    st.warning("Selecione a classificação do documento.")
                    return
                if not source_type:
                    st.warning("Selecione o tipo de fonte.")
                    return
                if not authorized:
                    st.warning("Confirme a autorização de uso do documento.")
                    return
                if not responsible.strip():
                    st.warning("Informe o responsável pela autorização.")
                    return
                if document_date and valid_until and valid_until < document_date:
                    st.warning("A validade não pode ser anterior à data do documento.")
                    return

                try:
                    safe_name = validate_upload(uploaded.name, uploaded.size, classification)
                    metadata = {
                        "active": True,
                        "use_in_generation": use_in_generation,
                        "version": version.strip() or None,
                        "data_documento": document_date.isoformat() if document_date else None,
                        "valido_ate": valid_until.isoformat() if valid_until else None,
                        "source_type": source_type,
                    }
                    with st.status("Processando documento...", expanded=True) as progress:
                        progress.write("Validando tamanho, formato, finalidade e classificação")
                        with temporary_upload(uploaded, safe_name) as temporary_path:
                            progress.write("Extraindo e segmentando o conteúdo localmente")
                            reference = get_retriever().ingest_document(
                                temporary_path,
                                classification,
                                original_filename=safe_name,
                                metadata=metadata,
                            )

                        with SessionLocal() as db:
                            source = (
                                db.query(SourceDocument)
                                .filter_by(retrieval_ref=reference)
                                .first()
                            )
                            if not source:
                                source = SourceDocument(
                                    filename=safe_name,
                                    classificacao=classification,
                                    responsavel=responsible.strip(),
                                    retrieval_ref=reference,
                                )
                                db.add(source)
                                db.flush()
                            else:
                                source.filename = safe_name
                                source.classificacao = classification
                                source.responsavel = responsible.strip()

                            governance = (
                                db.query(DocumentGovernance)
                                .filter(DocumentGovernance.source_document_id == source.id)
                                .first()
                            )
                            if not governance:
                                governance = DocumentGovernance(source_document_id=source.id)
                                db.add(governance)
                            governance.version = version.strip() or None
                            governance.data_documento = (
                                pd.Timestamp(document_date).to_pydatetime() if document_date else None
                            )
                            governance.valido_ate = (
                                pd.Timestamp(valid_until).to_pydatetime() if valid_until else None
                            )
                            governance.status = "ATIVO"

                            policy = (
                                db.query(DocumentUsagePolicy)
                                .filter(DocumentUsagePolicy.source_document_id == source.id)
                                .first()
                            )
                            if not policy:
                                policy = DocumentUsagePolicy(source_document_id=source.id)
                                db.add(policy)
                            policy.source_type = source_type
                            policy.use_in_generation = use_in_generation
                            policy.authorization_statement = authorization_text
                            policy.authorization_confirmed_at = dt.datetime.utcnow()
                            policy.notes = notes.strip() or None

                            db.add(
                                AuditEvent(
                                    run_id=None,
                                    event_type="source_indexed",
                                    payload={
                                        "source_document_id": source.id,
                                        "filename": safe_name,
                                        "classification": classification,
                                        "source_type": source_type,
                                        "use_in_generation": use_in_generation,
                                        "version": version.strip() or None,
                                    },
                                )
                            )
                            db.commit()
                        progress.update(label="Documento indexado", state="complete")

                    st.success(f"{safe_name} foi incluído no inventário.")
                    st.rerun()
                except Exception as exc:
                    friendly_error(exc)

    with right:
        render_source_inventory()

def context_editor_config() -> dict:
    return {
        "Tipo": st.column_config.SelectboxColumn(
            "Tipo",
            options=["fato", "inferencia", "lacuna"],
            required=True,
            help=(
                "Fato = informação comprovada; inferência = interpretação a confirmar; "
                "lacuna = informação ainda desconhecida."
            ),
        ),
        "Conteúdo": st.column_config.TextColumn(
            "Conteúdo",
            required=True,
            width="large",
            help=(
                "Escreva uma única informação por linha. Evite misturar fato e interpretação na mesma frase."
            ),
        ),
        "Fonte": st.column_config.TextColumn(
            "Fonte",
            help=(
                "Ex.: URL, nome do documento e página, reunião com data, e-mail corporativo "
                "ou registro autorizado de CRM."
            ),
        ),
        "Data": st.column_config.DateColumn(
            "Data",
            format="DD/MM/YYYY",
            help="Data da fonte, do fato ou do contato. Pode ficar vazia quando ainda não houver data.",
        ),
    }


def create_pipeline_run(
    profile: dict,
    parent_run_id: str | None = None,
    revision_reason: str | None = None,
) -> str:
    with SessionLocal() as db:
        prospect = Prospect(
            empresa=profile.get("empresa", ""),
            setor=profile.get("setor") or None,
            porte=profile.get("porte") or None,
        )
        db.add(prospect)
        db.flush()

        pipeline_run = PipelineRun(
            prospect_id=prospect.id,
            status="PROCESSANDO",
            perfil_contextual=profile,
        )
        db.add(pipeline_run)
        db.flush()

        if parent_run_id:
            db.add(
                RunLineage(
                    child_run_id=pipeline_run.id,
                    parent_run_id=parent_run_id,
                    revision_number=_revision_number(db, parent_run_id),
                    reason=revision_reason or "Revisão derivada de execução anterior.",
                )
            )

        db.add(
            AuditEvent(
                run_id=pipeline_run.id,
                event_type="run_created",
                payload={
                    "parent_run_id": parent_run_id,
                    "revision_reason": revision_reason,
                    "app_version": settings.APP_VERSION,
                    "profile_summary": {
                        "empresa": profile.get("empresa"),
                        "setor": profile.get("setor"),
                        "porte": profile.get("porte"),
                        "interlocutor": profile.get("interlocutor"),
                        "papel_decisorio": profile.get("papel_decisorio"),
                    },
                },
            )
        )
        db.commit()
        return pipeline_run.id

def persist_pipeline_result(run_id: str, profile: dict, result: dict) -> None:
    with SessionLocal() as db:
        pipeline_run = db.get(PipelineRun, run_id)
        if not pipeline_run:
            raise RuntimeError("A execução não foi encontrada para persistência.")

        pipeline_run.perfil_contextual = profile
        pipeline_run.pacote_contexto = result.get("pacote_contexto") or []
        pipeline_run.matriz_hipoteses = result.get("matriz_hipoteses") or {}
        pipeline_run.roteiro_diagnostico = result.get("roteiro_diagnostico") or {}
        pipeline_run.qualificacao = result.get("qualificacao") or {}
        pipeline_run.status = "EM_REVISAO"

        # Corrige registros antigos ou inconsistentes usando o A2 como fonte de verdade.
        prospect = db.get(Prospect, pipeline_run.prospect_id)
        if prospect:
            prospect.empresa = profile.get("empresa") or prospect.empresa
            prospect.setor = profile.get("setor") or None
            prospect.porte = profile.get("porte") or None

        for record in result.get("model_runs", []):
            allowed = {
                "etapa": record.get("etapa"),
                "prompt_version": record.get("prompt_version"),
                "modelo": record.get("modelo"),
                "duracao_ms": record.get("duracao_ms"),
                "custo_estimado_usd": record.get("custo_estimado_usd"),
            }
            db.add(ModelRun(run_id=run_id, **allowed))

        for audit in result.get("model_audits", []):
            db.add(
                AuditEvent(
                    run_id=run_id,
                    event_type="model_execution",
                    payload=audit,
                )
            )

        db.query(RunSourceSnapshot).filter(RunSourceSnapshot.run_id == run_id).delete()
        snapshot_sources_for_run(db, run_id, pipeline_run.pacote_contexto or [])

        suspicious = [
            item.get("fonte", "")
            for item in (pipeline_run.pacote_contexto or [])
            if is_suspicious_knowledge_filename(str(item.get("fonte", "")))
        ]
        alerts = list(result.get("alertas_validacao", []))
        if suspicious:
            alerts.insert(
                0,
                "Execução potencialmente contaminada por material de demonstração: "
                + ", ".join(sorted(set(suspicious)))
                + ". Não utilize o resultado; remova a fonte e gere nova execução.",
            )

        db.add(
            AuditEvent(
                run_id=run_id,
                event_type="pipeline_generated",
                payload={
                    "alerts": alerts,
                    "source_count": len({item.get('document_id') for item in pipeline_run.pacote_contexto or []}),
                    "app_version": settings.APP_VERSION,
                    "retrieval_config": {
                        "backend": settings.RETRIEVAL_BACKEND,
                        "top_k": settings.RETRIEVAL_TOP_K,
                        "min_relevance": settings.RETRIEVAL_MIN_RELEVANCE,
                        "semantic_weight": settings.RETRIEVAL_SEMANTIC_WEIGHT,
                        "lexical_weight": settings.RETRIEVAL_LEXICAL_WEIGHT,
                    },
                },
            )
        )
        db.commit()

def render_context_page() -> None:
    st.header("2. Coleta de contexto")
    st.caption(
        "Registre fatos, inferências e lacunas separadamente, com fonte e data quando disponíveis."
    )

    parent_run_id = st.session_state.get("parent_run_id")
    if parent_run_id:
        st.info(
            f"Você está preparando uma revisão da execução `{parent_run_id}`. "
            "Os dados foram carregados para correção e a nova execução ficará vinculada ao histórico."
        )

    with st.form("context_form_v3"):
        with st.expander("Como preencher o perfil do potencial cliente?", expanded=True):
            st.markdown(
                """
                Use apenas informações profissionais necessárias para preparar a conversa.

                - **Empresa:** organização, unidade de negócio ou órgão analisado.
                - **Setor:** atividade principal que contextualiza a operação.
                - **Porte e localização:** dimensão e abrangência relevantes.
                - **Oferta principal:** o que a organização entrega ao próprio mercado.
                - **Interlocutor, cargo e papel decisório:** contato profissional e sua participação na decisão.
                - **Histórico e restrições:** contatos anteriores, prazos, políticas e limites já conhecidos.
                """
            )

        left, right = st.columns(2, gap="large")
        with left:
            company = st.text_input(
                "Empresa *",
                key="context_company",
                placeholder="Ex.: Aurora Energia Integrada S.A.",
            )
            sector = st.text_input(
                "Setor",
                key="context_sector",
                placeholder="Ex.: Energia, saúde, logística ou varejo",
            )
            size = st.selectbox(
                "Porte",
                ["", "Micro", "Pequeno", "Médio", "Grande"],
                key="context_size",
                help="Use o critério adotado por sua organização; deixe em branco quando não houver evidência.",
            )
            location = st.text_input(
                "Localização",
                key="context_location",
                placeholder="Ex.: Recife/PE, Brasil; operação em nove unidades no Nordeste",
            )
            offer = st.text_input(
                "Oferta principal do potencial cliente",
                key="context_offer",
                placeholder="Ex.: geração e distribuição de energia para clientes corporativos",
                help="Descreve o negócio do cliente, não a solução que sua empresa deseja vender.",
            )

        with right:
            contact = st.text_input(
                "Interlocutor",
                key="context_contact",
                placeholder="Ex.: Adriano Lima",
            )
            role = st.text_input(
                "Cargo ou função do interlocutor",
                key="context_role",
                placeholder="Ex.: Gestor de Tecnologia da Informação",
            )
            decision_role = st.selectbox(
                "Papel no processo decisório",
                ["", "Decisor", "Influenciador", "Responsável técnico", "Usuário", "Comprador", "Ainda não confirmado"],
                key="context_decision_role",
                help="Separe o cargo formal da participação real na decisão."
            )
            history = st.text_area(
                "Histórico de contatos",
                key="context_history",
                placeholder="Ex.: reunião inicial em 08/07/2026; diretoria solicitou roadmap; interesse em piloto de 90 dias.",
                height=90,
            )
            restrictions = st.text_area(
                "Restrições conhecidas",
                key="context_restrictions",
                placeholder="Ex.: exige SSO e auditoria; orçamento ainda não aprovado; operação não pode parar.",
                height=90,
            )
            st.info(
                "Princípio de minimização: registre somente dados profissionais necessários. "
                "Prefira cargo, área, responsabilidade e contexto em vez de dados pessoais."
            )

        st.markdown("#### Fatos, inferências e lacunas")
        st.caption(
            "Adicione uma informação por linha. Fatos devem ser verificáveis; inferências precisam ser "
            "confirmadas; lacunas devem orientar perguntas."
        )
        render_context_examples()

        st.session_state.context_rows = normalize_context_rows(st.session_state.context_rows)
        rows = st.data_editor(
            st.session_state.context_rows,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config=context_editor_config(),
            key="context_editor_v3",
        )
        rows = normalize_context_rows(rows)

        if parent_run_id:
            revision_reason = st.text_input(
                "Motivo da revisão *",
                value=st.session_state.get("revision_reason", "Revisão solicitada após validação humana."),
                key="revision_reason_input",
            )
        else:
            revision_reason = None

        submitted = st.form_submit_button(
            "Gerar análise consultiva",
            type="primary",
            width="stretch",
        )

    st.session_state.context_rows = rows
    if not submitted:
        return

    new_run_id: str | None = None
    try:
        configuration_errors = settings.validate_for_generation()
        if configuration_errors:
            raise ConfigurationError(" ".join(configuration_errors))
        if not has_indexed_sources():
            st.warning(
                "Não há fontes ativas e autorizadas para geração. Ative ou indexe ao menos um documento válido."
            )
            return
        if not company.strip():
            st.warning("Informe a empresa.")
            return
        if parent_run_id and not str(revision_reason or "").strip():
            st.warning("Informe o motivo da revisão.")
            return

        valid_rows = rows.to_dict("records")
        items = [
            {
                "tipo": str(row.get("Tipo") or "fato"),
                "conteudo": str(row.get("Conteúdo") or "").strip(),
                "fonte": str(row.get("Fonte") or "").strip() or None,
                "data": context_item_date(row.get("Data")),
            }
            for row in valid_rows
            if str(row.get("Conteúdo") or "").strip()
        ]

        profile_model = PerfilContextual(
            empresa=company.strip(),
            setor=sector.strip() or None,
            porte=size or None,
            localizacao=location.strip() or None,
            oferta_principal=offer.strip() or None,
            interlocutor=contact.strip() or None,
            cargo_interlocutor=role.strip() or None,
            papel_decisorio=decision_role or None,
            historico_contato=history.strip() or None,
            restricoes_conhecidas=restrictions.strip() or None,
            itens=items,
        )

        facts = [item for item in profile_model.itens if item.tipo.value == "fato"]
        if not facts:
            st.warning("Inclua ao menos um fato verificável para iniciar a análise.")
            return
        if settings.REQUIRE_FACT_SOURCE:
            facts_without_source = [item.conteudo for item in facts if not item.fonte]
            if facts_without_source:
                st.warning(
                    "Informe a fonte dos fatos antes de gerar a análise. Itens sem fonte: "
                    + "; ".join(facts_without_source[:3])
                )
                return

        profile = profile_model.model_dump(mode="json")
        new_run_id = create_pipeline_run(
            profile,
            parent_run_id=parent_run_id,
            revision_reason=str(revision_reason or "").strip() or None,
        )
        st.session_state.run_id = new_run_id
        graph_config = {"configurable": {"thread_id": new_run_id}}

        with st.status("Executando o pipeline...", expanded=True) as progress:
            progress.write("3/7 Recuperando evidências autorizadas e ativas")
            result = graph.invoke(
                {"empresa": company.strip(), "perfil_contextual": profile},
                config=graph_config,
            )
            progress.write("4/7 Formulando hipóteses verificáveis")
            progress.write("5/7 Organizando perguntas diagnósticas")
            progress.write("6/7 Qualificando a oportunidade")
            progress.update(label="Análise pronta para revisão", state="complete")

        persist_pipeline_result(new_run_id, profile, result)
        st.session_state.pop("parent_run_id", None)
        st.session_state.pop("revision_reason", None)
        st.session_state.page = "Análise"
        st.rerun()
    except Exception as exc:
        mark_run_failed(new_run_id, str(exc))
        friendly_error(exc)


def render_analysis_page() -> None:
    st.header("3. Análise assistida")
    st.caption("Examine as fontes, hipóteses, perguntas e critérios antes de seguir para validação.")

    current_run = load_run(st.session_state.get("run_id"))
    if not current_run:
        st.info(
            "Nenhuma execução está aberta. Selecione uma execução na barra lateral e clique em "
            "‘Abrir execução’, ou inicie uma nova análise na etapa de contexto."
        )
        return

    summary = load_run_summary(current_run.id)
    render_execution_summary(summary)

    contaminated_sources = run_suspicious_sources(current_run)
    if contaminated_sources:
        st.error(
            "Esta execução utilizou material que parece conter guia, gabarito ou resultados esperados: "
            + ", ".join(contaminated_sources)
            + ". O resultado deve ser considerado contaminado. Desative ou exclua a fonte e gere uma nova execução."
        )

    with st.expander("O que revisar nesta etapa?", expanded=False):
        st.markdown(
            """
            - **Evidências:** confirme se o trecho é atual, relevante e pertence ao documento indicado.
            - **Hipóteses:** verifique se aparecem como possibilidades, e não como problemas comprovados.
            - **Perguntas:** avalie se são abertas, não indutivas e adequadas ao interlocutor.
            - **Qualificação:** confirme se a recomendação está apoiada em evidências e registra as lacunas.
            - **Rastreabilidade:** confira modelo, versão do prompt, duração, fontes e eventos registrados.
            """
        )

    alerts = load_run_alerts(current_run.id)
    if alerts:
        with st.expander(f"Alertas para validação ({len(alerts)})", expanded=True):
            for alert in alerts:
                st.warning(alert)

    evidence_tab, hypotheses_tab, questions_tab, qualification_tab, trace_tab = st.tabs(
        ["Evidências", "Hipóteses", "Perguntas", "Qualificação", "Rastreabilidade"]
    )
    with evidence_tab:
        render_evidence(current_run.pacote_contexto or [])
    with hypotheses_tab:
        render_hypotheses(current_run.matriz_hipoteses or {})
    with questions_tab:
        render_questions(current_run.roteiro_diagnostico or {})
    with qualification_tab:
        render_qualification(current_run.qualificacao or {})
    with trace_tab:
        render_traceability(current_run.id)

    if contaminated_sources:
        if st.button(
            "Corrigir base e criar nova execução",
            type="primary",
            width="stretch",
        ):
            load_profile_into_form(
                current_run.id,
                reason="Nova execução após remoção de fonte contaminante.",
            )
            st.rerun()
        return

    button_label = (
        "Abrir resultado da validação"
        if current_run.status in {"APROVADO", "REJEITADO", "AJUSTES_SOLICITADOS"}
        else "Ir para validação humana"
    )
    if st.button(
        button_label,
        type="primary",
        width="stretch",
        help="Abre a edição das perguntas, o checklist formal e o histórico de decisão.",
    ):
        st.session_state.page = "Validação"
        st.rerun()

def validation_editor_config() -> dict:
    return {
        "Tipo": st.column_config.SelectboxColumn(
            "Tipo",
            options=[
                "contexto",
                "operacao",
                "impacto",
                "maturidade",
                "prioridade",
                "decisao",
                "aderencia",
            ],
            help=(
                "Contexto = processo atual; operação = execução; impacto = consequências; "
                "maturidade = tentativas e capacidade; prioridade = prazo; decisão = envolvidos; "
                "aderência = requisitos para uma solução."
            ),
        ),
        "Prioridade": st.column_config.NumberColumn(
            "Prioridade",
            min_value=1,
            max_value=5,
            help="1 = indispensável ou perguntar primeiro; 5 = complementar ou opcional.",
        ),
        "Pergunta": st.column_config.TextColumn(
            "Pergunta",
            width="large",
            help="Ex.: Como esse processo é executado atualmente e quais áreas participam dele?",
        ),
        "Finalidade": st.column_config.TextColumn(
            "Finalidade",
            width="medium",
            help="Ex.: mapear etapas manuais, sistemas envolvidos e pontos de transferência de informação.",
        ),
    }


def persist_validation_result(
    run_id: str,
    decisions: dict[str, dict],
    reviewer: str,
    final_status: str,
    original_questions: list[dict],
    edited_questions: list[dict],
    approved_script: dict | None,
) -> None:
    with SessionLocal() as db:
        db.query(ValidationDecision).filter(
            ValidationDecision.run_id == run_id
        ).delete(synchronize_session=False)

        validation_time = dt.datetime.utcnow()
        for criterion, data in decisions.items():
            db.add(
                ValidationDecision(
                    run_id=run_id,
                    criterio=criterion,
                    pergunta_controle=CRITERIA_TEXT[CriterioValidacao(criterion)],
                    decisao=data["decisao"],
                    comentario=data["comentario"],
                    revisor=reviewer,
                    criado_em=validation_time,
                )
            )

        pipeline_run = db.get(PipelineRun, run_id)
        if not pipeline_run:
            raise RuntimeError("A execução não foi encontrada para registrar a validação.")
        pipeline_run.status = final_status.upper()

        changed_questions = []
        max_length = max(len(original_questions), len(edited_questions))
        for index in range(max_length):
            original = original_questions[index] if index < len(original_questions) else None
            edited = edited_questions[index] if index < len(edited_questions) else None
            if original != edited:
                changed_questions.append(
                    {"position": index + 1, "original": original, "edited": edited}
                )

        db.add(
            AuditEvent(
                run_id=run_id,
                event_type="human_validation",
                payload={
                    "status": final_status,
                    "reviewer": reviewer,
                    "validated_at": validation_time.isoformat(),
                    "decisions": decisions,
                    "original_questions": original_questions,
                    "edited_questions": edited_questions,
                    "changed_questions": changed_questions,
                    "app_version": settings.APP_VERSION,
                },
            )
        )

        if approved_script:
            existing = db.query(ApprovedScript).filter_by(run_id=run_id).first()
            if existing:
                existing.conteudo = approved_script
                existing.aprovado_por = reviewer
                existing.aprovado_em = validation_time
            else:
                db.add(
                    ApprovedScript(
                        run_id=run_id,
                        conteudo=approved_script,
                        aprovado_por=reviewer,
                        aprovado_em=validation_time,
                    )
                )
        db.commit()

def render_validation_page() -> None:
    st.header("4. Validação humana")
    st.caption("Aprovação, ajuste ou rejeição com critérios explícitos e registro do responsável.")

    run_id = st.session_state.get("run_id")
    current_run = load_run(run_id)
    if not current_run:
        st.info("Nenhuma execução está aberta. Selecione uma execução antes de validar.")
        return

    render_execution_summary(load_run_summary(run_id))

    contaminated_sources = run_suspicious_sources(current_run)
    if contaminated_sources:
        st.error(
            "A validação foi bloqueada porque a execução utilizou fonte potencialmente contaminante: "
            + ", ".join(contaminated_sources)
            + ". Corrija a base e gere uma nova execução."
        )
        if st.button("Carregar contexto para nova execução", type="primary", width="stretch"):
            load_profile_into_form(
                run_id,
                reason="Nova execução após remoção de fonte contaminante.",
            )
            st.rerun()
        return

    with st.expander("Como validar o roteiro?", expanded=False):
        st.markdown(
            """
            - **Aprovar:** o conteúdo atende ao critério e pode seguir sem mudança.
            - **Ajustar:** existe uma correção necessária; descreva exatamente o que deve mudar.
            - **Anonimizar:** há dados pessoais ou restritos que devem ser removidos ou generalizados.
            - **Rejeitar:** o problema compromete o roteiro e exige nova análise com outros insumos.

            As opções obedecem ao protocolo do artigo. Linguagem comercial e próximo passo permitem
            somente aprovação ou ajuste; proteção de dados permite aprovação, anonimização ou rejeição.
            """
        )

    if current_run.status in {"APROVADO", "REJEITADO", "AJUSTES_SOLICITADOS"}:
        if current_run.status == "APROVADO":
            st.success("Esta execução foi aprovada e possui roteiro final registrado.")
            with SessionLocal() as db:
                approved = db.query(ApprovedScript).filter_by(run_id=run_id).first()
            if approved:
                render_approved_script(approved.conteudo)
        elif current_run.status == "REJEITADO":
            st.error("Esta execução foi rejeitada e não está autorizada para uso comercial.")
        else:
            st.warning(
                "Foram solicitados ajustes. Crie uma revisão para corrigir o contexto ou as fontes "
                "sem perder a ligação com esta execução."
            )

        history = load_validation_history(run_id)
        st.subheader("Registro de validação")
        if history.empty:
            st.caption("Nenhum detalhe de validação foi encontrado.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)

        action_left, action_right = st.columns(2)
        with action_left:
            if st.button(
                "Criar revisão desta execução",
                type="primary" if current_run.status != "APROVADO" else "secondary",
                width="stretch",
            ):
                reason = (
                    "Correções solicitadas pela validação humana."
                    if current_run.status == "AJUSTES_SOLICITADOS"
                    else "Nova revisão derivada de execução encerrada."
                )
                load_profile_into_form(run_id, reason=reason)
                st.rerun()
        with action_right:
            if st.button("Voltar à análise", width="stretch"):
                st.session_state.page = "Análise"
                st.rerun()

        with st.expander("Rastreabilidade técnica", expanded=False):
            render_traceability(run_id)
        return

    graph_config = {"configurable": {"thread_id": run_id}}
    state = graph.get_state(graph_config)
    if not state.next:
        st.warning("O workflow não está pausado para validação. Verifique o status da execução.")
        return

    review_tab, questions_tab, trace_tab = st.tabs(
        ["Hipóteses e qualificação", "Perguntas editáveis", "Rastreabilidade"]
    )
    with review_tab:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            render_hypotheses(current_run.matriz_hipoteses or {})
        with right:
            render_qualification(current_run.qualificacao or {})

    original_questions = (current_run.roteiro_diagnostico or {}).get("perguntas", [])
    with questions_tab:
        st.subheader("Perguntas editáveis")
        st.caption(
            "Revise tipo, prioridade, redação e finalidade. Prioridade 1 é indispensável; 5 é opcional."
        )
        question_count = len(original_questions)
        high_priority = sum(
            1 for item in original_questions if int(item.get("prioridade", 3)) <= 2
        )
        metric_one, metric_two = st.columns(2)
        metric_one.metric("Perguntas geradas", question_count)
        metric_two.metric("Prioridade alta (1–2)", high_priority)
        if high_priority < 2:
            st.warning(
                "O roteiro possui menos de duas perguntas prioritárias. Reordene as perguntas essenciais "
                "antes de aprovar."
            )
        with st.expander("Comparar com a versão original", expanded=False):
            render_questions({"perguntas": original_questions})
        edited_questions_frame = st.data_editor(
            questions_dataframe(current_run.roteiro_diagnostico or {}),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            key=f"questions_editor_{run_id}",
            column_config=validation_editor_config(),
        )
        st.caption("As diferenças entre a versão original e a editada serão registradas.")

    with trace_tab:
        render_traceability(run_id)

    decisions: dict[str, dict] = {}
    with st.form(f"validation_form_{run_id}"):
        st.subheader("Checklist de revisão")
        st.caption(
            "Avalie todos os critérios. Justificativas são obrigatórias para ajuste, anonimização ou rejeição."
        )

        criterion_items = list(CRITERIA_TEXT.items())
        left_column, right_column = st.columns(2, gap="large")
        for index, (criterion, question) in enumerate(criterion_items):
            target = left_column if index % 2 == 0 else right_column
            with target:
                with st.container(border=True):
                    st.markdown(f"**{CRITERION_LABELS[criterion]}**")
                    st.caption(question)
                    options = VALIDATION_OPTIONS[criterion]
                    decision = st.radio(
                        "Decisão",
                        options,
                        index=0,
                        key=f"decision_{run_id}_{criterion.value}",
                        horizontal=True,
                        format_func=lambda value: DECISION_LABELS[value],
                    )
                    comment = st.text_area(
                        "Justificativa ou instrução de ajuste",
                        key=f"comment_{run_id}_{criterion.value}",
                        placeholder=VALIDATION_COMMENT_EXAMPLES[criterion],
                        height=86,
                    )
                    decisions[criterion.value] = {
                        "decisao": decision,
                        "comentario": comment.strip(),
                    }

        reviewer = st.text_input(
            "Responsável pela revisão *",
            placeholder="Ex.: Thiago Laurentino — Marketing e Pré-vendas",
        )
        responsibility_confirmed = st.checkbox(
            "Confirmo que revisei evidências, hipóteses, perguntas, qualificação e fontes desta execução."
        )
        register = st.form_submit_button(
            "Registrar decisão",
            type="primary",
            width="stretch",
        )

    if not register:
        return
    if not reviewer.strip():
        st.warning("Informe o responsável pela revisão.")
        return
    if not responsibility_confirmed:
        st.warning("Confirme que todos os artefatos foram revisados.")
        return
    if any(item["decisao"] == "nao_avaliado" for item in decisions.values()):
        st.warning("Avalie todos os critérios antes de registrar a decisão.")
        return

    missing_comments = [
        CRITERION_LABELS[CriterioValidacao(criterion)]
        for criterion, item in decisions.items()
        if item["decisao"] in {"ajustar", "anonimizar", "rejeitar"}
        and not item["comentario"]
    ]
    if missing_comments:
        st.warning("Inclua justificativa nos critérios: " + ", ".join(missing_comments) + ".")
        return

    if any(item["decisao"] == "rejeitar" for item in decisions.values()):
        final_status = "rejeitado"
    elif any(item["decisao"] in {"ajustar", "anonimizar"} for item in decisions.values()):
        final_status = "ajustes_solicitados"
    else:
        final_status = "aprovado"

    edited_payload = [
        {
            "tipo": row["Tipo"],
            "pergunta": str(row["Pergunta"]).strip(),
            "prioridade": int(row.get("Prioridade") or 3),
            "finalidade": str(row["Finalidade"]).strip(),
        }
        for row in edited_questions_frame.fillna("").to_dict("records")
        if str(row.get("Pergunta", "")).strip()
    ]
    if not edited_payload:
        st.warning("Mantenha ao menos uma pergunta diagnóstica.")
        return
    if final_status == "aprovado" and settings.REQUIRE_HIGH_PRIORITY_QUESTIONS:
        if sum(1 for item in edited_payload if int(item["prioridade"]) <= 2) < 2:
            st.warning(
                "Para aprovar, o roteiro deve conter ao menos duas perguntas com prioridade 1 ou 2."
            )
            return

    reviewer_name = reviewer.strip()
    try:
        graph.invoke(
            Command(
                resume={
                    "status": final_status,
                    "revisor": reviewer_name,
                    "perguntas_editadas": edited_payload,
                }
            ),
            config=graph_config,
        )
        final_state = graph.get_state(graph_config)
        approved_script = final_state.values.get("roteiro_aprovado")
        persist_validation_result(
            run_id=run_id,
            decisions=decisions,
            reviewer=reviewer_name,
            final_status=final_status,
            original_questions=original_questions,
            edited_questions=edited_payload,
            approved_script=approved_script,
        )
        st.rerun()
    except Exception as exc:
        friendly_error(exc)

def render_app() -> None:
    if not st.session_state.get("knowledge_guard_done"):
        quarantined = quarantine_suspicious_sources()
        st.session_state.knowledge_guard_done = True
        if quarantined:
            st.session_state.knowledge_guard_message = quarantined

    # A barra lateral precisa ser renderizada antes do stepper para que a
    # página escolhida no mesmo ciclo seja refletida no indicador horizontal.
    render_sidebar()

    current_run = load_run(st.session_state.get("run_id"))
    completed: set[int] = set()
    if has_indexed_sources():
        completed.add(1)
    if current_run and current_run.perfil_contextual:
        completed.add(2)
    if current_run and (
        current_run.pacote_contexto
        or current_run.matriz_hipoteses
        or current_run.roteiro_diagnostico
        or current_run.qualificacao
    ):
        completed.add(3)
    if current_run and current_run.status in {
        "AJUSTES_SOLICITADOS",
        "APROVADO",
        "REJEITADO",
    }:
        completed.add(4)

    render_header(
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        settings.RETRIEVAL_BACKEND,
        app_version=settings.APP_VERSION,
        environment=settings.APP_ENV,
    )
    render_stepper(PAGE_STEP[st.session_state.page], completed)

    if st.session_state.pop("knowledge_guard_message", None):
        st.warning(
            "Materiais com possível gabarito ou resultados esperados foram colocados em quarentena "
            "automaticamente e não participarão de novas análises."
        )

    page_renderers = {
        "Base de conhecimento": render_base_page,
        "Coleta de contexto": render_context_page,
        "Análise": render_analysis_page,
        "Validação": render_validation_page,
    }
    page_renderers[st.session_state.page]()



# Ponto de entrada da aplicação Streamlit.
render_app()
