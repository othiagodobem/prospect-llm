from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    run_id: str
    empresa: str
    perfil_contextual: dict[str, Any]
    pacote_contexto: list[dict[str, Any]]
    matriz_hipoteses: dict[str, Any]
    roteiro_diagnostico: dict[str, Any]
    qualificacao: dict[str, Any]
    alertas_validacao: Annotated[list[str], operator.add]
    model_runs: Annotated[list[dict[str, Any]], operator.add]
    model_audits: Annotated[list[dict[str, Any]], operator.add]
    decisao_humana: dict[str, Any]
    roteiro_aprovado: dict[str, Any]
