"""Workflow determinístico das Etapas 3 a 7 do PROSPECT-LLM."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from config import settings
from llm.provider import get_llm_provider
from retrieval.factory import get_retriever
from schemas.artifacts import (
    GeracaoHipoteses,
    GeracaoPerguntas,
    GeracaoQualificacao,
    RoteiroAprovado,
)
from workflow.prompts import PROMPT_VERSION, prompt_hipoteses, prompt_perguntas, prompt_qualificacao
from workflow.state import PipelineState


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _model_run(stage: str, metadata: dict) -> dict:
    return {
        "etapa": stage,
        "prompt_version": PROMPT_VERSION,
        "modelo": metadata.get("modelo", settings.LLM_MODEL),
        "duracao_ms": metadata.get("duracao_ms"),
        "custo_estimado_usd": metadata.get("custo_estimado_usd"),
    }




def _resumo_factual(profile: dict) -> str:
    fatos = [
        str(item.get("conteudo", "")).strip()
        for item in profile.get("itens", [])
        if item.get("tipo") == "fato" and str(item.get("conteudo", "")).strip()
    ]
    lacunas = [
        str(item.get("conteudo", "")).strip()
        for item in profile.get("itens", [])
        if item.get("tipo") == "lacuna" and str(item.get("conteudo", "")).strip()
    ]
    lines = [f"Empresa: {profile.get('empresa', '—')}"]
    for label, key in [
        ("Setor", "setor"),
        ("Porte", "porte"),
        ("Localização", "localizacao"),
        ("Oferta principal", "oferta_principal"),
        ("Interlocutor", "interlocutor"),
        ("Cargo", "cargo_interlocutor"),
        ("Papel decisório", "papel_decisorio"),
    ]:
        if profile.get(key):
            lines.append(f"{label}: {profile[key]}")
    if fatos:
        lines.append("Fatos confirmados:")
        lines.extend(f"- {item}" for item in fatos)
    if lacunas:
        lines.append("Lacunas registradas:")
        lines.extend(f"- {item}" for item in lacunas)
    return "\n".join(lines)


def recuperar_evidencias(state: PipelineState) -> PipelineState:
    retriever = get_retriever()
    profile = state.get("perfil_contextual", {})
    facts = " ".join(
        item.get("conteudo", "") for item in profile.get("itens", [])
        if item.get("tipo") == "fato"
    )
    query = " ".join(
        filter(
            None,
            [
                state.get("empresa", ""),
                profile.get("setor", ""),
                profile.get("oferta_principal", ""),
                profile.get("cargo_interlocutor", ""),
                profile.get("papel_decisorio", ""),
                profile.get("historico_contato", ""),
                profile.get("restricoes_conhecidas", ""),
                facts,
            ],
        )
    )
    excerpts = retriever.search(query, top_k=settings.RETRIEVAL_TOP_K)
    return {"pacote_contexto": [item.model_dump(mode="json") for item in excerpts]}


def gerar_hipoteses(state: PipelineState) -> PipelineState:
    system, user = prompt_hipoteses(
        perfil=_json(state.get("perfil_contextual", {})),
        contexto=_json(state.get("pacote_contexto", [])),
    )
    result, metadata = get_llm_provider().generate_structured(
        system, user, GeracaoHipoteses
    )
    return {
        "matriz_hipoteses": result.matriz_hipoteses.model_dump(mode="json"),
        "alertas_validacao": result.alertas_validacao,
        "model_runs": [_model_run("gerar_hipoteses", metadata)],
        "model_audits": [{"etapa": "gerar_hipoteses", "prompt_version": PROMPT_VERSION, **metadata}],
    }


def gerar_perguntas(state: PipelineState) -> PipelineState:
    system, user = prompt_perguntas(
        perfil=_json(state.get("perfil_contextual", {})),
        hipoteses=_json(state.get("matriz_hipoteses", {})),
    )
    result, metadata = get_llm_provider().generate_structured(
        system, user, GeracaoPerguntas
    )
    return {
        "roteiro_diagnostico": result.roteiro_diagnostico.model_dump(mode="json"),
        "alertas_validacao": result.alertas_validacao,
        "model_runs": [_model_run("gerar_perguntas", metadata)],
        "model_audits": [{"etapa": "gerar_perguntas", "prompt_version": PROMPT_VERSION, **metadata}],
    }


def qualificar_oportunidade(state: PipelineState) -> PipelineState:
    system, user = prompt_qualificacao(
        perfil=_json(state.get("perfil_contextual", {})),
        hipoteses=_json(state.get("matriz_hipoteses", {})),
        perguntas=_json(state.get("roteiro_diagnostico", {})),
    )
    result, metadata = get_llm_provider().generate_structured(
        system, user, GeracaoQualificacao
    )
    return {
        "qualificacao": result.qualificacao.model_dump(mode="json"),
        "alertas_validacao": result.alertas_validacao,
        "model_runs": [_model_run("qualificar_oportunidade", metadata)],
        "model_audits": [{"etapa": "qualificar_oportunidade", "prompt_version": PROMPT_VERSION, **metadata}],
    }


def validar_humano(state: PipelineState) -> PipelineState:
    decision = interrupt(
        {
            "matriz_hipoteses": state.get("matriz_hipoteses"),
            "roteiro_diagnostico": state.get("roteiro_diagnostico"),
            "qualificacao": state.get("qualificacao"),
            "alertas_validacao": state.get("alertas_validacao", []),
        }
    )
    return {"decisao_humana": decision}


def finalizar(state: PipelineState) -> PipelineState:
    decision = state.get("decisao_humana", {}) or {}
    if decision.get("status") != "aprovado":
        return {"roteiro_aprovado": {}}

    edited_questions = decision.get("perguntas_editadas")
    questions = edited_questions or state.get("roteiro_diagnostico", {}).get("perguntas", [])
    script = RoteiroAprovado(
        prospect=state["empresa"],
        resumo_factual=_resumo_factual(state.get("perfil_contextual", {})),
        perguntas_finais=questions,
        recomendacao_final=state.get("qualificacao", {}).get("recomendacao"),
        aprovado_por=decision.get("revisor", ""),
        data_aprovacao=dt.date.today(),
    )
    return {"roteiro_aprovado": script.model_dump(mode="json")}


def _get_checkpointer() -> SqliteSaver:
    connection = sqlite3.connect(
        settings.LANGGRAPH_CHECKPOINT_DB, check_same_thread=False
    )
    return SqliteSaver(connection)


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("recuperar_evidencias", recuperar_evidencias)
    graph.add_node("gerar_hipoteses", gerar_hipoteses)
    graph.add_node("gerar_perguntas", gerar_perguntas)
    graph.add_node("qualificar_oportunidade", qualificar_oportunidade)
    graph.add_node("validar_humano", validar_humano)
    graph.add_node("finalizar", finalizar)
    graph.set_entry_point("recuperar_evidencias")
    graph.add_edge("recuperar_evidencias", "gerar_hipoteses")
    graph.add_edge("gerar_hipoteses", "gerar_perguntas")
    graph.add_edge("gerar_perguntas", "qualificar_oportunidade")
    graph.add_edge("qualificar_oportunidade", "validar_humano")
    graph.add_edge("validar_humano", "finalizar")
    graph.add_edge("finalizar", END)
    return graph.compile(checkpointer=_get_checkpointer())
