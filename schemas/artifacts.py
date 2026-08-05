"""Schemas Pydantic dos artefatos A1-A8 do PROSPECT-LLM."""
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Confianca(str, Enum):
    BAIXA = "baixa"
    MEDIA = "media"
    ALTA = "alta"


class DimensaoHipotese(str, Enum):
    TECNICA = "tecnica"
    OPERACIONAL = "operacional"
    ESTRATEGICA = "estrategica"
    FINANCEIRA = "financeira"
    EXPERIENCIA = "experiencia"


class TipoPergunta(str, Enum):
    CONTEXTO = "contexto"
    OPERACAO = "operacao"
    IMPACTO = "impacto"
    MATURIDADE = "maturidade"
    PRIORIDADE = "prioridade"
    DECISAO = "decisao"
    ADERENCIA = "aderencia"


class Recomendacao(str, Enum):
    AVANCAR = "avancar"
    APROFUNDAR = "aprofundar"
    NUTRIR = "nutrir"
    ENCERRAR = "encerrar"


class CriterioValidacao(str, Enum):
    FUNDAMENTACAO = "fundamentacao"
    ADERENCIA_PORTFOLIO = "aderencia_portfolio"
    QUALIDADE_DIAGNOSTICA = "qualidade_diagnostica"
    PROTECAO_DADOS = "protecao_dados"
    LINGUAGEM_COMERCIAL = "linguagem_comercial"
    PROXIMO_PASSO = "proximo_passo"


class DecisaoValidacao(str, Enum):
    APROVAR = "aprovar"
    AJUSTAR = "ajustar"
    REJEITAR = "rejeitar"
    ANONIMIZAR = "anonimizar"


class ClassificacaoDado(str, Enum):
    PUBLICO = "publico"
    INTERNO_NAO_SENSIVEL = "interno_nao_sensivel"
    RESTRITO = "restrito"
    SENSIVEL_CONFIDENCIAL = "sensivel_confidencial"


class TipoItemContextual(str, Enum):
    FATO = "fato"
    INFERENCIA = "inferencia"
    LACUNA = "lacuna"


class ItemContextual(BaseModel):
    conteudo: str = Field(min_length=3)
    tipo: TipoItemContextual
    fonte: Optional[str] = None
    data: Optional[dt.date] = None
    source_document_id: Optional[str] = None
    source_url: Optional[str] = None


class PerfilContextual(BaseModel):
    empresa: str = Field(min_length=2)
    setor: Optional[str] = None
    porte: Optional[str] = None
    localizacao: Optional[str] = None
    oferta_principal: Optional[str] = None
    interlocutor: Optional[str] = None
    cargo_interlocutor: Optional[str] = None
    papel_decisorio: Optional[str] = None
    historico_contato: Optional[str] = None
    restricoes_conhecidas: Optional[str] = None
    itens: List[ItemContextual] = Field(default_factory=list)


class TrechoRecuperado(BaseModel):
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    trecho: str
    fonte: str
    classificacao: Optional[str] = None
    data_documento: Optional[dt.date] = None
    versao_documento: Optional[str] = None
    relevancia: float = Field(ge=0.0, le=1.0)


class EvidenciaHipotese(BaseModel):
    descricao: str
    fonte: str
    chunk_id: Optional[str] = None


class Hipotese(BaseModel):
    dimensao: DimensaoHipotese
    hipotese: str
    evidencias: List[EvidenciaHipotese] = Field(default_factory=list)
    confianca: Confianca
    fundamentacao_confianca: str = Field(
        min_length=5,
        description=(
            "Explique por que a confiança é alta, média ou baixa, considerando quantidade, "
            "independência e natureza das evidências."
        ),
    )
    pergunta_confirmacao: str


class MatrizHipoteses(BaseModel):
    hipoteses: List[Hipotese] = Field(min_length=1, max_length=10)


class PerguntaDiagnostica(BaseModel):
    tipo: TipoPergunta
    pergunta: str
    prioridade: int = Field(ge=1, le=5)
    finalidade: str

    @field_validator("pergunta")
    @classmethod
    def ensure_question_mark(cls, value: str) -> str:
        value = value.strip()
        return value if value.endswith("?") else value + "?"


class RoteiroDiagnostico(BaseModel):
    perguntas: List[PerguntaDiagnostica] = Field(min_length=3, max_length=20)

    @model_validator(mode="after")
    def ensure_actionable_priorities(self):
        """Evita roteiros sem perguntas essenciais por inconsistência do modelo.

        Se o modelo não produzir ao menos duas perguntas de prioridade 1-2, a
        normalização promove perguntas de contexto/operação/impacto/decisão,
        preservando a ordem relativa e sem alterar seu texto.
        """

        if sum(1 for item in self.perguntas if item.prioridade <= 2) >= 2:
            self.perguntas.sort(key=lambda item: (item.prioridade, item.tipo.value))
            return self

        type_rank = {
            TipoPergunta.CONTEXTO: 0,
            TipoPergunta.OPERACAO: 1,
            TipoPergunta.IMPACTO: 2,
            TipoPergunta.DECISAO: 3,
            TipoPergunta.MATURIDADE: 4,
            TipoPergunta.ADERENCIA: 5,
            TipoPergunta.PRIORIDADE: 6,
        }
        ordered = sorted(
            self.perguntas,
            key=lambda item: (type_rank.get(item.tipo, 99), item.prioridade),
        )
        if ordered:
            ordered[0].prioridade = 1
        if len(ordered) > 1:
            ordered[1].prioridade = 2
        self.perguntas.sort(key=lambda item: (item.prioridade, item.tipo.value))
        return self


class CriteriosQualificacao(BaseModel):
    aderencia: str
    urgencia: str
    impacto: str
    autoridade: str
    maturidade: str
    viabilidade: str
    evidencia: str


class Qualificacao(BaseModel):
    criterios: CriteriosQualificacao
    recomendacao: Recomendacao
    justificativa: str
    lacunas: List[str] = Field(default_factory=list)


class GeracaoHipoteses(BaseModel):
    matriz_hipoteses: MatrizHipoteses
    alertas_validacao: List[str] = Field(default_factory=list)


class GeracaoPerguntas(BaseModel):
    roteiro_diagnostico: RoteiroDiagnostico
    alertas_validacao: List[str] = Field(default_factory=list)


class GeracaoQualificacao(BaseModel):
    qualificacao: Qualificacao
    alertas_validacao: List[str] = Field(default_factory=list)


class ItemValidacao(BaseModel):
    criterio: CriterioValidacao
    pergunta_controle: str
    decisao: DecisaoValidacao
    comentario: Optional[str] = None


class RegistroValidacao(BaseModel):
    itens: List[ItemValidacao]
    revisor: str
    status: str


class RoteiroAprovado(BaseModel):
    prospect: str
    resumo_factual: str
    perguntas_finais: List[PerguntaDiagnostica]
    recomendacao_final: Optional[Recomendacao] = None
    aprovado_por: str
    data_aprovacao: dt.date
