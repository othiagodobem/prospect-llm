from schemas.artifacts import RoteiroDiagnostico


def test_at_least_two_questions_are_high_priority():
    roteiro = RoteiroDiagnostico.model_validate(
        {
            "perguntas": [
                {
                    "tipo": "contexto",
                    "pergunta": "Quais sistemas são usados?",
                    "prioridade": 4,
                    "finalidade": "Mapear sistemas",
                },
                {
                    "tipo": "operacao",
                    "pergunta": "Como o processo funciona?",
                    "prioridade": 5,
                    "finalidade": "Mapear operação",
                },
                {
                    "tipo": "decisao",
                    "pergunta": "Quem participa da decisão?",
                    "prioridade": 5,
                    "finalidade": "Mapear decisão",
                },
            ]
        }
    )
    assert sum(item.prioridade <= 2 for item in roteiro.perguntas) >= 2
