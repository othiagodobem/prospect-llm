import datetime as dt

from workflow.graph import finalizar


def test_finalizar_rejects_non_approved():
    state = {"decisao_humana": {"status": "rejeitado"}}
    assert finalizar(state)["roteiro_aprovado"] == {}


def test_finalizar_approved():
    state = {
        "empresa": "Empresa X",
        "perfil_contextual": {"empresa": "Empresa X"},
        "roteiro_diagnostico": {
            "perguntas": [
                {
                    "tipo": "contexto",
                    "pergunta": "Como funciona hoje?",
                    "prioridade": 1,
                    "finalidade": "Entender",
                }
            ]
        },
        "qualificacao": {"recomendacao": "aprofundar"},
        "decisao_humana": {"status": "aprovado", "revisor": "Revisor"},
    }
    result = finalizar(state)["roteiro_aprovado"]
    assert result["prospect"] == "Empresa X"
    assert result["aprovado_por"] == "Revisor"
