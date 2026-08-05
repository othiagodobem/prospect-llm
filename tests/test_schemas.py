from schemas.artifacts import PerguntaDiagnostica


def test_question_mark_is_added():
    item = PerguntaDiagnostica(
        tipo="contexto", pergunta="Como o processo funciona hoje", prioridade=1,
        finalidade="Compreender o processo atual",
    )
    assert item.pergunta.endswith("?")
